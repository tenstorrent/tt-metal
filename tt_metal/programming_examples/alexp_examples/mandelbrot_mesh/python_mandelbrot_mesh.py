#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from ttnn.distributed import visualize_system_mesh, visualize_mesh_device, visualize_tensor


def mandelbrot_cpu(width, height, max_iter=100, x_min=-2.5, x_max=1.5, y_min=-2.0, y_max=2.0):
    """CPU reference implementation of Mandelbrot set"""
    # Create coordinate arrays
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    X, Y = np.meshgrid(x, y)

    # Complex plane
    C = X + 1j * Y

    # Initialize Z and iteration count
    Z = np.zeros_like(C)
    iterations = np.zeros(C.shape, dtype=int)

    for i in range(max_iter):
        # Find points that haven't escaped
        mask = np.abs(Z) <= 2

        # Update Z for non-escaped points
        Z[mask] = Z[mask] ** 2 + C[mask]

        # Update iteration count
        iterations[mask] = i

    return iterations


def save_mandelbrot_image(data, width, height, filename, max_iterations=100):
    """Save Mandelbrot data as a colorful image with enhanced coloring"""
    # Normalize iteration data with better scaling
    normalized = np.clip(data.astype(np.float32) / max_iterations, 0, 1)

    # Create colormap with better contrast
    colors = np.zeros((height, width, 3))

    # Enhanced color scheme with smoother gradients
    for y in range(height):
        for x in range(width):
            ratio = normalized[y, x]

            if ratio == 1.0:  # Points in the set (reached max iterations)
                colors[y, x] = [0, 0, 0]  # Pure black
            elif ratio < 0.16:
                # Deep blue to blue
                t = ratio / 0.16
                colors[y, x] = [0, 0, 0.5 + 0.5 * t]
            elif ratio < 0.33:
                # Blue to cyan
                t = (ratio - 0.16) / 0.17
                colors[y, x] = [0, t, 1]
            elif ratio < 0.5:
                # Cyan to green
                t = (ratio - 0.33) / 0.17
                colors[y, x] = [0, 1, 1 - t]
            elif ratio < 0.66:
                # Green to yellow
                t = (ratio - 0.5) / 0.16
                colors[y, x] = [t, 1, 0]
            elif ratio < 0.83:
                # Yellow to red
                t = (ratio - 0.66) / 0.17
                colors[y, x] = [1, 1 - t, 0]
            else:
                # Red to white
                t = (ratio - 0.83) / 0.17
                colors[y, x] = [1, t, t]

    # Convert to 8-bit and save
    colors_8bit = (colors * 255).astype(np.uint8)
    image = Image.fromarray(colors_8bit)
    image.save(filename)
    print(f"Mandelbrot set saved to {filename}")


def mandelbrot_ttnn_mesh(width=512, height=512, max_iterations=100):
    """Compute Mandelbrot set using TTNN on mesh device"""

    # Open mesh device
    device = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))

    # Display mesh topology
    print("System mesh topology:")
    visualize_system_mesh()
    visualize_mesh_device(device)

    # Mandelbrot parameters - use the classic view with better precision
    x_min, x_max = -2.5, 1.5
    y_min, y_max = -2.0, 2.0

    # Create coordinate tensors on host
    x_coords = torch.linspace(x_min, x_max, width, dtype=torch.float32)
    y_coords = torch.linspace(y_min, y_max, height, dtype=torch.float32)

    # Create meshgrid
    X, Y = torch.meshgrid(x_coords, y_coords, indexing="xy")

    # Flatten for processing
    x_flat = X.flatten()
    y_flat = Y.flatten()

    # Pad to tile boundary (32x32 = 1024 elements per tile)
    total_elements = len(x_flat)
    elements_per_tile = 32 * 32
    num_tiles = (total_elements + elements_per_tile - 1) // elements_per_tile
    padded_size = num_tiles * elements_per_tile

    # Pad coordinates
    x_padded = torch.zeros(padded_size, dtype=torch.float32)
    y_padded = torch.zeros(padded_size, dtype=torch.float32)
    x_padded[:total_elements] = x_flat
    y_padded[:total_elements] = y_flat

    # Reshape for tiling
    x_tiled = x_padded.reshape(-1, 32, 32)
    y_tiled = y_padded.reshape(-1, 32, 32)

    # Convert to TTNN tensors
    x_host = ttnn.from_torch(x_tiled, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    y_host = ttnn.from_torch(y_tiled, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)

    # Create mesh mapper for distribution
    mapper = ttnn.create_mesh_mapper(
        device,
        ttnn.MeshMapperConfig(
            [
                ttnn.PlacementShard(0),
                ttnn.PlacementShard(1),
            ],  # Shard along both dimensions for 2D mesh, replicated along the other dimension
            ttnn.MeshShape(2, 4),
        ),
    )

    # Distribute tensors across mesh
    x_mesh = ttnn.distribute_tensor(x_host, mapper, device)
    y_mesh = ttnn.distribute_tensor(y_host, mapper, device)

    print("Tensors distributed across mesh:")
    visualize_tensor(x_mesh)

    # Initialize Z tensor (complex numbers as real and imaginary parts)
    zx_mesh = ttnn.zeros_like(x_mesh, dtype=ttnn.float32)
    zy_mesh = ttnn.zeros_like(y_mesh, dtype=ttnn.float32)

    # Initialize iteration counter
    iterations_mesh = ttnn.zeros_like(x_mesh, dtype=ttnn.float32)

    # Mandelbrot iteration loop
    print(f"Computing Mandelbrot set with {max_iterations} iterations...")

    for i in range(max_iterations):
        if i % 10 == 0:
            print(f"Iteration {i}/{max_iterations}")

        # Calculate |Z|^2 = zx^2 + zy^2
        zx_squared = ttnn.multiply(zx_mesh, zx_mesh)
        zy_squared = ttnn.multiply(zy_mesh, zy_mesh)
        z_magnitude_squared = ttnn.add(zx_squared, zy_squared)

        # Create mask for points that haven't escaped (|Z|^2 <= 4)
        four_tensor = ttnn.full_like(z_magnitude_squared, 4.0)
        escape_mask = ttnn.gt(z_magnitude_squared, four_tensor)  # Points that have escaped
        continue_mask = ttnn.logical_not(escape_mask)  # Points that continue iterating

        # Only update iteration count for points that are still iterating
        # Set escaped points to their final iteration count
        current_iteration = ttnn.full_like(iterations_mesh, float(i))
        iterations_mesh = ttnn.where(
            ttnn.logical_and(escape_mask, ttnn.eq(iterations_mesh, ttnn.full_like(iterations_mesh, 0.0))),
            current_iteration,
            iterations_mesh,
        )

        # Update Z: Z = Z^2 + C for continuing points only
        # New zx = zx^2 - zy^2 + cx
        # New zy = 2*zx*zy + cy

        zx_zy_product = ttnn.multiply(zx_mesh, zy_mesh)
        two_zx_zy = ttnn.multiply(zx_zy_product, ttnn.full_like(zx_zy_product, 2.0))

        new_zx = ttnn.subtract(zx_squared, zy_squared)
        new_zx = ttnn.add(new_zx, x_mesh)

        new_zy = ttnn.add(two_zx_zy, y_mesh)

        # Apply mask to update only non-escaped points
        zx_mesh = ttnn.where(continue_mask, new_zx, zx_mesh)
        zy_mesh = ttnn.where(continue_mask, new_zy, zy_mesh)

    # Gather results back to host with mesh composer
    print("Gathering results from mesh...")

    # Create mesh composer to concatenate shards from 2x4 mesh
    mesh_composer = ttnn.create_mesh_composer(
        device,
        ttnn.MeshComposerConfig([0, 1], ttnn.MeshShape(2, 4)),  # Concatenate along dimensions 0 and 1  # 2x4 mesh shape
    )

    # Convert distributed tensor back to host tensor
    iterations_host = ttnn.to_torch(iterations_mesh, mesh_composer=mesh_composer)

    # Reshape and extract original data
    iterations_flat = iterations_host.flatten()[:total_elements]
    iterations_2d = iterations_flat.reshape(height, width).numpy()

    # Close mesh device
    ttnn.close_device(device)

    return iterations_2d


def main():
    """Main function to run Mandelbrot computation"""
    print("=" * 60)
    print("Mandelbrot Set Computation on Tenstorrent Mesh Device")
    print("=" * 60)

    width, height = 2048, 2048
    max_iterations = 100

    print(f"Image dimensions: {width} x {height}")
    print(f"Max iterations: {max_iterations}")
    print()

    # Compute using TTNN mesh device
    print("Computing Mandelbrot set on Tenstorrent mesh device...")
    try:
        mandelbrot_ttnn = mandelbrot_ttnn_mesh(width, height, max_iterations)
        save_mandelbrot_image(mandelbrot_ttnn, width, height, "mandelbrot_ttnn_mesh.png", max_iterations)
        print("✓ TTNN mesh computation completed successfully!")
    except Exception as e:
        print(f"✗ TTNN mesh computation failed: {e}")
        print("Computing CPU reference instead...")

        # Fallback to CPU computation
        mandelbrot_cpu_result = mandelbrot_cpu(width, height, max_iterations)
        save_mandelbrot_image(mandelbrot_cpu_result, width, height, "mandelbrot_cpu_reference.png", max_iterations)
        print("✓ CPU reference computation completed!")

    print("\nMandelbrot set computation finished!")
    print("Check the generated PNG images to see the results.")


if __name__ == "__main__":
    main()
