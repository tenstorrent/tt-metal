#!/usr/bin/env python3
# Create a sample PPM file to demonstrate visualization

import numpy as np


def create_sample_mandelbrot_ppm(filename="sample_mandelbrot.ppm", width=256, height=256):
    """Create a sample PPM file with Mandelbrot-like pattern"""

    print(f"Creating sample PPM file: {filename}")
    print(f"Resolution: {width} × {height}")

    # Create coordinate arrays
    x = np.linspace(-2.5, 1.5, width)
    y = np.linspace(-2.0, 2.0, height)
    X, Y = np.meshgrid(x, y)

    # Complex plane
    C = X + 1j * Y

    # Simple Mandelbrot computation
    Z = np.zeros_like(C)
    iterations = np.zeros(C.shape, dtype=int)
    max_iter = 50

    for i in range(max_iter):
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask] ** 2 + C[mask]
        iterations[mask] = i

    # Create PPM file
    with open(filename, "w") as f:
        # PPM header
        f.write("P3\n")
        f.write(f"{width} {height}\n")
        f.write("255\n")

        # Write pixel data
        for y in range(height):
            for x in range(width):
                # Normalize and create color
                ratio = iterations[y, x] / max_iter

                if ratio < 0.25:
                    r = int(ratio * 4 * 255)
                    g = 0
                    b = 0
                elif ratio < 0.5:
                    r = 255
                    g = int((ratio - 0.25) * 4 * 255)
                    b = 0
                elif ratio < 0.75:
                    r = 255
                    g = 255
                    b = int((ratio - 0.5) * 4 * 255)
                else:
                    r = g = b = 255

                f.write(f"{r} {g} {b} ")
            f.write("\n")

    print(f"✅ Created {filename}")


if __name__ == "__main__":
    # Create sample PPM files
    create_sample_mandelbrot_ppm("mandelbrot_mesh_sample.ppm", 256, 256)
    create_sample_mandelbrot_ppm("mandelbrot_mesh_simple_sample.ppm", 256, 256)
    print("\n📁 Sample PPM files created for demonstration")
