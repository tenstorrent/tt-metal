#!/usr/bin/env python3
"""
Verify Mandelbrot Math - Debug and Test Coordinate Mapping
"""

import numpy as np
import matplotlib.pyplot as plt


def test_mandelbrot_point(cx, cy, max_iter=100):
    """Test a single Mandelbrot point"""
    zx, zy = 0.0, 0.0
    for i in range(max_iter):
        zx2, zy2 = zx * zx, zy * zy
        if zx2 + zy2 > 4.0:
            return i
        zx, zy = zx2 - zy2 + cx, 2 * zx * zy + cy
    return max_iter


def analyze_kernel_coordinates():
    """Analyze the coordinate mapping from the debug output"""
    print("🔍 Mandelbrot Math Analysis")
    print("=" * 50)

    # Parameters from kernels
    IMAGE_WIDTH = 512
    IMAGE_HEIGHT = 512
    x_min, x_max = -2.5, 1.5
    y_min, y_max = -2.0, 2.0

    print(f"Image size: {IMAGE_WIDTH}×{IMAGE_HEIGHT}")
    print(f"Complex plane: x[{x_min}, {x_max}] y[{y_min}, {y_max}]")

    # Calculate deltas (same as kernel)
    dx = (x_max - x_min) / IMAGE_WIDTH
    dy = (y_max - y_min) / IMAGE_HEIGHT

    print(f"Coordinate deltas: dx={dx:.7f} dy={dy:.7f}")
    print()

    # Test coordinate mapping for debug points
    debug_points = [
        (0, 448),  # From device 7 (top part)
        (0, 384),  # From device 6
        (0, 256),  # From device 4 (middle)
        (0, 128),  # From device 2
        (0, 64),  # From device 1
        (0, 0),  # From device 0 (bottom part)
    ]

    print("📊 Coordinate Mapping Analysis:")
    print("Pixel(x,y) → Complex(cx,cy) → Expected cy → Kernel cy → Iterations")
    print("-" * 80)

    for x, y in debug_points:
        # Current kernel mapping (WRONG)
        cx_kernel = x_min + x * dx
        cy_kernel = y_min + y * dy

        # Correct mapping (flip Y)
        cx_correct = x_min + x * dx
        cy_correct = y_max - y * dy  # Flip Y coordinate

        # Test both
        iter_kernel = test_mandelbrot_point(cx_kernel, cy_kernel)
        iter_correct = test_mandelbrot_point(cx_correct, cy_correct)

        print(
            f"Pixel({x},{y:3d}) → c({cx_kernel:7.3f},{cy_kernel:6.3f}) → "
            f"should be ({cx_correct:7.3f},{cy_correct:6.3f}) → "
            f"iter_wrong={iter_kernel:2d} iter_correct={iter_correct:2d}"
        )

    print()

    # Test some known Mandelbrot points
    print("🎯 Known Mandelbrot Set Points:")
    known_points = [
        (0.0, 0.0, "Origin - should be in set"),
        (-0.5, 0.0, "Real axis point - in set"),
        (-1.0, 0.0, "Real axis point - in set"),
        (-2.0, 0.0, "Outside set - should escape quickly"),
        (0.0, 1.0, "Imaginary axis - should escape"),
        (-0.7269, 0.1889, "Known point in set"),
    ]

    for cx, cy, desc in known_points:
        iterations = test_mandelbrot_point(cx, cy)
        status = "IN SET" if iterations == 100 else f"ESCAPES in {iterations}"
        print(f"c({cx:7.3f},{cy:6.3f}) → {status:12s} ({desc})")

    print()


def analyze_device_partitioning():
    """Analyze how work should be distributed across 8 devices"""
    print("🔧 Device Partitioning Analysis:")
    print("=" * 50)

    IMAGE_WIDTH = 512
    IMAGE_HEIGHT = 512
    NUM_DEVICES = 8

    pixels_per_device = (IMAGE_WIDTH * IMAGE_HEIGHT) // NUM_DEVICES

    print(f"Total pixels: {IMAGE_WIDTH * IMAGE_HEIGHT}")
    print(f"Pixels per device: {pixels_per_device}")
    print()

    print("Device ID → Pixel Range → Y Range → Complex Y Range")
    print("-" * 60)

    x_min, x_max = -2.5, 1.5
    y_min, y_max = -2.0, 2.0
    dy = (y_max - y_min) / IMAGE_HEIGHT

    for device_id in range(NUM_DEVICES):
        start_pixel = device_id * pixels_per_device
        end_pixel = (device_id + 1) * pixels_per_device if device_id < 7 else IMAGE_WIDTH * IMAGE_HEIGHT

        # Convert to y ranges
        start_y = start_pixel // IMAGE_WIDTH
        end_y = (end_pixel - 1) // IMAGE_WIDTH

        # Convert to complex coordinates (CORRECTED)
        cy_start = y_max - start_y * dy  # Flip Y
        cy_end = y_max - end_y * dy  # Flip Y

        print(
            f"Device {device_id} → pixels {start_pixel:5d}-{end_pixel:5d} → "
            f"y {start_y:3d}-{end_y:3d} → cy {cy_end:6.3f} to {cy_start:6.3f}"
        )


def create_test_image():
    """Create a small test Mandelbrot image to verify math"""
    print("🎨 Creating Test Mandelbrot Image:")
    print("=" * 50)

    width, height = 64, 64
    x_min, x_max = -2.0, 1.0
    y_min, y_max = -1.5, 1.5
    max_iter = 50

    # Create coordinate arrays
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_max, y_min, height)  # Flip Y for correct orientation
    X, Y = np.meshgrid(x, y)

    # Compute Mandelbrot set
    mandelbrot = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            cx, cy = X[i, j], Y[i, j]
            mandelbrot[i, j] = test_mandelbrot_point(cx, cy, max_iter)

    # Save as simple text visualization
    print("Mandelbrot Test (64×64, '*'=in set, '.'=escapes quickly, numbers=iterations):")
    print()

    for i in range(0, height, 4):  # Sample every 4th row
        row = ""
        for j in range(0, width, 2):  # Sample every 2nd column
            val = int(mandelbrot[i, j])
            if val >= max_iter:
                row += "*"  # In the set
            elif val <= 2:
                row += "."  # Escapes very quickly
            else:
                row += str(min(val, 9))  # Show iteration count
        print(f"Row {i:2d}: {row}")

    print()
    print("Legend: * = in Mandelbrot set, . = escapes in ≤2 iterations, numbers = iteration count")

    return mandelbrot


def main():
    print("🧮 Mandelbrot Math Verification Tool")
    print("=" * 60)
    print()

    analyze_kernel_coordinates()
    print()
    analyze_device_partitioning()
    print()
    create_test_image()

    print()
    print("🔧 Fixes Needed:")
    print("1. Fix Y-coordinate mapping: cy = y_max - y * dy")
    print("2. Fix device coordinate partitioning")
    print("3. Verify iteration logic in kernels")
    print("4. Test with known Mandelbrot points")


if __name__ == "__main__":
    main()
