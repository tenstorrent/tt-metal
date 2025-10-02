#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent
# SPDX-License-Identifier: Apache-2.0

"""
Demo script that shows CPU Mandelbrot computation and image generation
This can run without TT hardware to demonstrate the concept
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time


def mandelbrot_cpu(width, height, max_iter=100, x_min=-2.5, x_max=1.5, y_min=-2.0, y_max=2.0):
    """CPU implementation of Mandelbrot set computation"""
    print(f"Computing {width}x{height} Mandelbrot set with {max_iter} iterations...")
    start_time = time.time()

    # Create coordinate arrays
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    X, Y = np.meshgrid(x, y)

    # Complex plane
    C = X + 1j * Y

    # Initialize Z and iteration count
    Z = np.zeros_like(C)
    iterations = np.zeros(C.shape, dtype=int)

    # Mandelbrot iteration
    for i in range(max_iter):
        if i % 20 == 0:
            print(f"  Iteration {i}/{max_iter}")

        # Find points that haven't escaped
        mask = np.abs(Z) <= 2

        # Update Z for non-escaped points
        Z[mask] = Z[mask] ** 2 + C[mask]

        # Update iteration count
        iterations[mask] = i

    computation_time = time.time() - start_time
    print(f"  Computation completed in {computation_time:.2f} seconds")

    return iterations


def save_mandelbrot_image(data, width, height, filename, max_iterations=100):
    """Save Mandelbrot data as a colorful image"""
    print(f"Saving image to {filename}...")

    # Normalize iteration data
    normalized = data.astype(np.float32) / max_iterations

    # Create colormap
    colors = np.zeros((height, width, 3))

    # Color gradient: black -> red -> yellow -> white
    for y in range(height):
        for x in range(width):
            ratio = normalized[y, x]

            if ratio < 0.25:
                # Black to red
                colors[y, x] = [ratio * 4, 0, 0]
            elif ratio < 0.5:
                # Red to yellow
                colors[y, x] = [1, (ratio - 0.25) * 4, 0]
            elif ratio < 0.75:
                # Yellow to white
                colors[y, x] = [1, 1, (ratio - 0.5) * 4]
            else:
                # White
                colors[y, x] = [1, 1, 1]

    # Convert to 8-bit and save
    colors_8bit = (colors * 255).astype(np.uint8)
    image = Image.fromarray(colors_8bit)
    image.save(filename)
    print(f"✓ Image saved to {filename}")


def create_matplotlib_visualization(data, max_iterations=100):
    """Create a matplotlib visualization of the Mandelbrot set"""
    plt.figure(figsize=(12, 10))

    # Create custom colormap
    plt.imshow(
        data, extent=[-2.5, 1.5, -2.0, 2.0], cmap="hot", origin="lower", interpolation="bilinear", vmax=max_iterations
    )

    plt.colorbar(label="Iterations to escape")
    plt.title(
        "Mandelbrot Set - CPU Reference Implementation\n(Demonstrating TT Mesh Device Concept)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Real axis")
    plt.ylabel("Imaginary axis")

    # Add some annotations
    plt.text(
        -2.3, 1.7, "Classic Mandelbrot Set View", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
    )
    plt.text(
        -2.3,
        1.5,
        f"Resolution: {data.shape[1]}×{data.shape[0]}",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    plt.text(
        -2.3,
        1.3,
        f"Max iterations: {max_iterations}",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig("mandelbrot_demo_plot.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✓ Matplotlib visualization saved to mandelbrot_demo_plot.png")


def main():
    """Main demo function"""
    print("=" * 60)
    print("Mandelbrot Set Demo - CPU Reference Implementation")
    print("(Demonstrating concept for TT Mesh Device)")
    print("=" * 60)

    # Parameters
    width, height = 800, 600
    max_iterations = 150

    print(f"Configuration:")
    print(f"  Resolution: {width} × {height}")
    print(f"  Max iterations: {max_iterations}")
    print(f"  View window: [-2.5, 1.5] × [-2.0, 2.0]")
    print()

    # Compute Mandelbrot set
    mandelbrot_data = mandelbrot_cpu(width, height, max_iterations)

    # Save as image
    save_mandelbrot_image(mandelbrot_data, width, height, "mandelbrot_demo.png", max_iterations)

    # Create matplotlib visualization
    try:
        create_matplotlib_visualization(mandelbrot_data, max_iterations)
    except ImportError:
        print("⚠ Matplotlib not available, skipping plot visualization")

    # Print statistics
    print("\nStatistics:")
    print(f"  Total pixels: {width * height:,}")
    print(f"  Pixels in set (max iterations): {np.sum(mandelbrot_data == max_iterations-1):,}")
    print(f"  Pixels escaped: {np.sum(mandelbrot_data < max_iterations-1):,}")
    print(f"  Average iterations: {np.mean(mandelbrot_data):.2f}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("This demonstrates the computational pattern that would")
    print("be distributed across the TT mesh device in the full implementation.")
    print("=" * 60)


if __name__ == "__main__":
    main()
