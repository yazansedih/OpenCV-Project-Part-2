import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
import os
from fractions import Fraction
from tkinter import Tk, Frame, Label, StringVar, OptionMenu, Entry
from tkinter import ttk
import scipy.signal


original_image = gray_temp = root = panel = error_label = error_label2 = None
type = -1
numExportImage = 0

def select_image():
    global original_image, gray_temp, panel, type, error_label

    file_path = filedialog.askopenfilename(
        initialdir="/",
        title="Select Image File",
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
    )

    if file_path:
        try:
            original_image = cv2.imread(file_path)
            if original_image is not None:
                gray_temp = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                type = 0
                show_image()
                error_label.config(text='')
            else:
                error_label.config(text="Error reading the image. Please select a valid image file.")
        except Exception as e:
            error_label.config(text=f"Error: {str(e)}")
    else:
        error_label.config(text="No file selected. Please select an image.")

    # print(str(file_path))

def get_original():
    global original_image, type, error_label
    if original_image is not None:
        type = 0
        show_image()
    else:
        error_label.config(text="No file selected. Please select an image.")



def convert_gray():
    global original_image, gray_temp, type, error_label
    if original_image is not None:
        gray_temp = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        type = 1
        show_image()
    else:
        error_label.config(text="No file selected. Please select an image.")

def apply_zero_crossing_filter():
    global gray_temp, zero_crossing_image, type, error_label
    if gray_temp is not None:
        sigma = 1.0
        kernel_size = 5
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray_temp, (kernel_size, kernel_size), sigma)

        # Apply Laplacian filter
        kernel = np.array([[0, -1, 0],
                           [-1, 4, -1],
                           [0, -1, 0]])

        laplacian = scipy.signal.convolve2d(blurred, kernel, mode='same', boundary='symm')
        laplacian = np.clip(laplacian, 0, 255).astype(np.uint8)

        # Find zero-crossings
        zero_crossing_image = np.zeros_like(laplacian, dtype=np.uint8)
        zero_crossing_image[laplacian > 0] = 255

        type = 5
        show_image()
    else:
        error_label.config(text="No file selected. Please select an image.")

def apply_threshold():
    global gray_temp, segmented_image, type, error_label
    if gray_temp is not None:
        _, segmented_image = cv2.threshold(gray_temp, 120, 255, cv2.THRESH_BINARY)
        type = 2
        show_image()
    else:
        error_label.config(text="No file selected. Please select an image.")

def apply_adaptive_threshold():
    global gray_temp, segmented_image, type, error_label
    if gray_temp is not None:
        segmented_image = cv2.adaptiveThreshold(gray_temp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY, 39, 5)
        type = 2
        show_image()
    else:
        error_label.config(text="No file selected. Please select an image.")

def apply_point_detection():
    global gray_temp, segmented_image, type, error_label
    if gray_temp is not None:
        kernel = np.array([[-1, -1, -1],
                           [-1, 8, -1],
                           [-1, -1, -1]])

        segmented_image = scipy.signal.convolve2d(gray_temp, kernel, mode='same', boundary='symm')
        segmented_image = np.clip(segmented_image, 0, 255).astype(np.uint8)

        type = 2
        show_image()
    else:
        error_label.config(text="No file selected. Please select an image.")

def apply_horizontal_line_detection():
    global gray_temp, segmented_image, type, error_label
    if gray_temp is not None:
        kernel = np.array([[-1, -1, -1],
                           [2, 2, 2],
                           [-1, -1, -1]])

        segmented_image = scipy.signal.convolve2d(gray_temp, kernel, mode='same', boundary='symm')
        segmented_image = np.clip(segmented_image, 0, 255).astype(np.uint8)

        type = 2
        show_image()
    else:
        error_label.config(text="No file selected. Please select an image.")

def apply_vertical_line_detection():
    global gray_temp, segmented_image, type, error_label
    if gray_temp is not None:
        kernel = np.array([[-1, 2, -1],
                           [-1, 2, -1],
                           [-1, 2, -1]])

        segmented_image = scipy.signal.convolve2d(gray_temp, kernel, mode='same', boundary='symm')
        segmented_image = np.clip(segmented_image, 0, 255).astype(np.uint8)

        type = 2
        show_image()
    else:
        error_label.config(text="No file selected. Please select an image.")

def apply_p45_line_detection():
    global gray_temp, segmented_image, type, error_label
    if gray_temp is not None:
        kernel = np.array([[-1, -1, 2],
                           [-1, 2, -1],
                           [2, -1, -1]])

        segmented_image = scipy.signal.convolve2d(gray_temp, kernel, mode='same', boundary='symm')
        segmented_image = np.clip(segmented_image, 0, 255).astype(np.uint8)

        type = 2
        show_image()
    else:
        error_label.config(text="No file selected. Please select an image.")

def apply_n45_line_detection():
    global gray_temp, segmented_image, type, error_label
    if gray_temp is not None:
        kernel = np.array([[2, -1, -1],
                           [-1, 2, -1],
                           [-1, -1, 2]])

        segmented_image = scipy.signal.convolve2d(gray_temp, kernel, mode='same', boundary='symm')
        segmented_image = np.clip(segmented_image, 0, 255).astype(np.uint8)

        type = 2
        show_image()
    else:
        error_label.config(text="No file selected. Please select an image.")

def apply_horizontal_edge_detection():
    global gray_temp, segmented_image, type, error_label
    if gray_temp is not None:
        kernel = np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]])

        segmented_image = scipy.signal.convolve2d(gray_temp, kernel, mode='same', boundary='symm')
        segmented_image = np.clip(segmented_image, 0, 255).astype(np.uint8)

        type = 2
        show_image()
    else:
        error_label.config(text="No file selected. Please select an image.")

def apply_vertical_edge_detection():
    global gray_temp, segmented_image, type, error_label
    if gray_temp is not None:
        kernel = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])

        segmented_image = scipy.signal.convolve2d(gray_temp, kernel, mode='same', boundary='symm')
        segmented_image = np.clip(segmented_image, 0, 255).astype(np.uint8)

        type = 2
        show_image()
    else:
        error_label.config(text="No file selected. Please select an image.")

def apply_p45_edge_detection():
    global gray_temp, segmented_image, type, error_label
    if gray_temp is not None:
        kernel = np.array([[-2, -1, 0],
                           [-1, 0, 1],
                           [0, 1, 2]])

        segmented_image = scipy.signal.convolve2d(gray_temp, kernel, mode='same', boundary='symm')
        segmented_image = np.clip(segmented_image, 0, 255).astype(np.uint8)

        type = 2
        show_image()
    else:
        error_label.config(text="No file selected. Please select an image.")

def apply_n45_edge_detection():
    global gray_temp, segmented_image, type, error_label
    if gray_temp is not None:
        kernel = np.array([[0, 1, 2],
                           [-1, 0, 1],
                           [-2, -1, 0]])

        segmented_image = scipy.signal.convolve2d(gray_temp, kernel, mode='same', boundary='symm')
        segmented_image = np.clip(segmented_image, 0, 255).astype(np.uint8)

        type = 2
        show_image()
    else:
        error_label.config(text="No file selected. Please select an image.")

def apply_laplacian_filter():
    global gray_temp, segmented_image, type, error_label
    if gray_temp is not None:
        kernel = np.array([[0, -1, 0],
                           [-1, 4, -1],
                           [0, -1, 0]])

        segmented_image = scipy.signal.convolve2d(gray_temp, kernel, mode='same', boundary='symm')
        segmented_image = np.clip(segmented_image, 0, 255).astype(np.uint8)

        type = 2
        show_image()
    else:
        error_label.config(text="No file selected. Please select an image.")

def laplacian_of_gaussian():
    global gray_temp, segmented_image, type, error_label
    if gray_temp is not None:
        kernel = np.array([[0, 0, -1, 0, 0],
                           [0, -1, -2, -1, 0],
                           [-1, -2, 16, -2, -1],
                           [0, -1, -2, -1, 0],
                           [0, 0, -1, 0,  0]])

        segmented_image = scipy.signal.convolve2d(gray_temp, kernel, mode='same', boundary='symm')
        segmented_image = np.clip(segmented_image, 0, 255).astype(np.uint8)

        type = 2
        show_image()
    else:
        error_label.config(text="No file selected. Please select an image.")

def apply_filter():
    global gray_temp, filtered_image, type, error_label, error_label2
    if gray_temp is not None:
        if entry_filter_size.get() is not None:
            if entry_filter_coefficients.get() is not None:
                # Get filter size and coefficients
                size = int(entry_filter_size.get())
                coefficients_str = entry_filter_coefficients.get()
                coefficients = [Fraction(x) for x in coefficients_str.split(',')]

                kernel = np.array(coefficients, dtype=np.float32).reshape((size, size))
                # print(str(kernel) + "zzz")
                filtered_image = cv2.filter2D(gray_temp, -1, kernel)

                type = 3
                show_image()
            else:
                error_label2.config(text="Please enter coefficients.")
        else:
            error_label2.config(text="Please enter size of filter.")
            # 1/15, 1/15, 1/15, 1/15, 1/15, 1/15, 1/15, 1/15, 1/15
    else:
        error_label.config(text="No file selected. Please select an image.")

def apply_image_processing(method):
    if method == 'Thresholding':
        apply_threshold()
    elif method == 'Adaptive Thresholding':
        apply_adaptive_threshold()
    elif method == 'Point Detection':
        apply_point_detection()
    elif method == 'Horizontal Line Detection':
        apply_horizontal_line_detection()
    elif method == 'Vertical Line Detection':
        apply_vertical_line_detection()
    elif method == '+45 Line Detection':
        apply_p45_line_detection()
    elif method == '-45 Line Detection':
        apply_n45_line_detection()
    elif method == 'Horizontal Edge Detection':
        apply_horizontal_edge_detection()
    elif method == 'Vertical Edge Detection':
        apply_vertical_edge_detection()
    elif method == '+45 Edge Detection':
        apply_p45_edge_detection()
    elif method == '-45 Edge Detection':
        apply_n45_edge_detection()
    elif method == 'Laplacian Mask':
        apply_laplacian_filter()
    elif method == 'Laplacian of Gaussian(LoG)':
        laplacian_of_gaussian()
    elif method == 'Zero Crossing Filter':
        apply_zero_crossing_filter()

def show_image():
    global original_image, gray_temp, segmented_image, filtered_image, sobel_filter, zero_crossing_image, type, panel
    if type == 0:
        img = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    elif type == 1:
        img = cv2.cvtColor(gray_temp, cv2.COLOR_GRAY2BGR)
    elif type == 2:
        img = segmented_image
    elif type == 3:
        img = filtered_image
    elif type == 4:
        img = sobel_filter
    elif type == 5:
        img = zero_crossing_image
    else:
        img = np.zeros((500, 500, 3), np.uint8)

    if not img.size == 0:
        img = cv2.resize(img, (500, 500))

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = PhotoImage(data=cv2.imencode('.ppm', img_rgb)[1].tobytes())

        # Update the image displayed in the Tkinter label
        panel.config(image=img)
        panel.image = img

def save_image():
    global original_image, gray_temp, segmented_image, filtered_image, sobel_filter, type, numExportImage, error_label,zero_crossing_image

    try:
        file_path = filedialog.asksaveasfilename(
            initialdir="/path/to/default/folder",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        # print(str(file_path))

        if file_path:
            if type == 0:
                cv2.imwrite(file_path, original_image)
            elif type == 1:
                cv2.imwrite(file_path, gray_temp)
            elif type == 2:
                cv2.imwrite(file_path, segmented_image)
            elif type == 3:
                cv2.imwrite(file_path, filtered_image)
            elif type == 4:
                cv2.imwrite(file_path, sobel_filter)
            elif type == 5:
                cv2.imwrite(file_path, zero_crossing_image)
            else:
                error_label.config(text="No file selected. Please select an image.")

            # numExportImage += 1
    except Exception as e:
        error_label.config("Error saving image:", str(e))

def close_window():
    if root is not None:
        cv2.destroyAllWindows()
        root.destroy()

def main():
    global variable, label, panel, root, entry_filter_size, entry_filter_coefficients, error_label
    root = Tk()
    root.title('Image Processing')
    # Frames
    left_frame = Frame(root, padx=30, pady=10)
    left_frame.pack(side='left', fill='both', expand=True)

    right_frame = Frame(root, padx=20, pady=20)
    right_frame.pack(side='right', fill='both', expand=True)

    panel = Label(right_frame)
    panel.pack(fill='both', expand=True)

    # Error Label
    error_label = Label(left_frame, text='', height=2, font=('Arial', 10), fg='red')
    error_label.pack(pady=5)

    # Buttons
    style = ttk.Style()
    style.configure("TButton", padding=(10, 5), font=('Arial', 10))

    ttk.Button(left_frame, text='Select Image...', command=select_image, style="TButton").pack(pady=5, fill='both')
    ttk.Button(left_frame, text='Original', command=get_original, style="TButton").pack(pady=5, fill='both')
    ttk.Button(left_frame, text='Gray Scale', command=convert_gray, style="TButton").pack(pady=5, fill='both')

    # Option Menu
    Label(left_frame, text='Segment Image:', height=2, font=('Arial', 10)).pack(pady=0)
    variable = StringVar(root)
    variable.set('Thresholding')  # default value
    option_menu = OptionMenu(left_frame, variable, 'Thresholding', 'Adaptive Thresholding', 'Point Detection',
                             'Horizontal Line Detection', 'Vertical Line Detection', '+45 Line Detection',
                             '-45 Line Detection', 'Horizontal Edge Detection', 'Vertical Edge Detection',
                             '+45 Edge Detection', '-45 Edge Detection', 'Laplacian Mask', 'Laplacian of Gaussian(LoG)',
                             'Zero Crossing Filter',
                             command=lambda method=variable: apply_image_processing(method))
    option_menu.pack(pady=5, fill='both')

    # Error Label
    error_label2 = Label(left_frame, text='', height=2, font=('Arial', 10), fg='red')
    error_label2.pack(pady=5)

    # User-Defined Filter
    Label(left_frame, text='Apply User-Defined Filter:', height=2, font=('Arial', 10)).pack(pady=5)
    Label(left_frame, text='Filter Size:', height=2, font=('Arial', 10)).pack(pady=0)
    entry_filter_size = Entry(left_frame, font=('Arial', 10))
    entry_filter_size.pack(pady=0, fill='both')
    Label(left_frame, text='Filter Coefficients (comma-separated):', height=2, font=('Arial', 10)).pack(pady=0)
    entry_filter_coefficients = Entry(left_frame, font=('Arial', 10))
    entry_filter_coefficients.pack(pady=0, fill='both')
    ttk.Button(left_frame, text='Apply Filter', command=apply_filter, style="TButton").pack(pady=10, fill='both')

    ttk.Button(left_frame, text='Save Image', command=save_image, style="TButton").pack(pady=10, fill='both')
    ttk.Button(left_frame, text='Exit', command=close_window, style="TButton").pack(pady=10, fill='both')

    label = Label(left_frame, height=2, font=('Arial', 10))
    label.pack(pady=5)

    root.mainloop()

if __name__ == '__main__':
    main()

