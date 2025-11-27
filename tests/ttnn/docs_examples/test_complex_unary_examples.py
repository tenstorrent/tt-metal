# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch
from loguru import logger


def test_real(device):
    # Create a complex tensor from real and imaginary parts
    real_part = ttnn.from_torch(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    imag_part = ttnn.from_torch(
        torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    complex_tensor = ttnn.complex_tensor(real_part, imag_part)

    # Get the real part
    output = ttnn.real(complex_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    logger.info(f"Real part: {output}")


def test_imag(device):
    # Create a complex tensor from real and imaginary parts
    real_part = ttnn.from_torch(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    imag_part = ttnn.from_torch(
        torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    complex_tensor = ttnn.complex_tensor(real_part, imag_part)

    # Get the imaginary part
    output = ttnn.imag(complex_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    logger.info(f"Imaginary part: {output}")


def test_angle(device):
    # Create a complex tensor from real and imaginary parts
    real_part = ttnn.from_torch(
        torch.tensor([[1.0, 0.0], [1.0, -1.0]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    imag_part = ttnn.from_torch(
        torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    complex_tensor = ttnn.complex_tensor(real_part, imag_part)

    # Get the angle (phase) of the complex tensor
    output = ttnn.angle(complex_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    logger.info(f"Angle: {output}")


def test_is_imag(device):
    # Create a complex tensor with zero real part (purely imaginary)
    real_part = ttnn.from_torch(
        torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    imag_part = ttnn.from_torch(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    complex_tensor = ttnn.complex_tensor(real_part, imag_part)

    # Check if tensor values are purely imaginary
    output = ttnn.is_imag(complex_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    logger.info(f"Is imaginary: {output}")


def test_is_real(device):
    # Create a complex tensor with zero imaginary part (purely real)
    real_part = ttnn.from_torch(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    imag_part = ttnn.from_torch(
        torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    complex_tensor = ttnn.complex_tensor(real_part, imag_part)

    # Check if tensor values are purely real
    output = ttnn.is_real(complex_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    logger.info(f"Is real: {output}")


def test_conj(device):
    # Create a complex tensor from real and imaginary parts
    real_part = ttnn.from_torch(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    imag_part = ttnn.from_torch(
        torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    complex_tensor = ttnn.complex_tensor(real_part, imag_part)

    # Get the complex conjugate (negates imaginary part)
    output = ttnn.conj(complex_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    logger.info(f"Complex conjugate: {output}")


def test_polar(device):
    # Create a complex tensor representing polar coordinates (r, theta)
    # r is stored in real part, theta in imaginary part
    r = ttnn.from_torch(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    theta = ttnn.from_torch(
        torch.tensor([[0.0, 1.57], [3.14, 4.71]], dtype=torch.bfloat16),  # 0, π/2, π, 3π/2
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    polar_tensor = ttnn.complex_tensor(r, theta)

    # Convert polar to Cartesian (x + iy)
    output = ttnn.polar(polar_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    logger.info(f"Polar to Cartesian: {output}")


def test_alt_complex_rotate90(device):
    # Create a tensor with alternating real/imaginary values in TILE layout
    tensor = ttnn.from_torch(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Rotate complex values by 90 degrees
    output = ttnn.alt_complex_rotate90(tensor)
    logger.info(f"Complex rotated by 90 degrees: {output}")
