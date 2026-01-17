# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch
from loguru import logger


def test_real(device):
    # Create a complex tensor
    real_part = torch.rand([1, 1, 32, 32], dtype=torch.bfloat16)
    imag_part = torch.rand([1, 1, 32, 32], dtype=torch.bfloat16)
    tensor = ttnn.complex_tensor(
        ttnn.Tensor(real_part, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
        ttnn.Tensor(imag_part, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
    )

    # Get the real part
    output = ttnn.real(tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    logger.info(f"Real part: {output}")


def test_imag(device):
    # Create a complex tensor
    real_part = torch.rand([1, 1, 32, 32], dtype=torch.bfloat16)
    imag_part = torch.rand([1, 1, 32, 32], dtype=torch.bfloat16)
    tensor = ttnn.complex_tensor(
        ttnn.Tensor(real_part, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
        ttnn.Tensor(imag_part, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
    )

    # Get the imaginary part
    output = ttnn.imag(tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    logger.info(f"Imaginary part: {output}")


def test_angle(device):
    # Create a complex tensor
    real_part = torch.rand([1, 1, 32, 32], dtype=torch.bfloat16)
    imag_part = torch.rand([1, 1, 32, 32], dtype=torch.bfloat16)
    tensor = ttnn.complex_tensor(
        ttnn.Tensor(real_part, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
        ttnn.Tensor(imag_part, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
    )

    # Get the angle (phase) of the complex tensor
    output = ttnn.angle(tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    logger.info(f"Angle: {output}")


def test_is_imag(device):
    # Create a complex tensor with only imaginary part
    real_part = torch.zeros([1, 1, 32, 32], dtype=torch.bfloat16)
    imag_part = torch.rand([1, 1, 32, 32], dtype=torch.bfloat16)
    tensor = ttnn.complex_tensor(
        ttnn.Tensor(real_part, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
        ttnn.Tensor(imag_part, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
    )

    # Check if tensor values are purely imaginary
    output = ttnn.is_imag(tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    logger.info(f"Is imaginary: {output}")


def test_is_real(device):
    # Create a complex tensor with only real part
    real_part = torch.rand([1, 1, 32, 32], dtype=torch.bfloat16)
    imag_part = torch.zeros([1, 1, 32, 32], dtype=torch.bfloat16)
    tensor = ttnn.complex_tensor(
        ttnn.Tensor(real_part, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
        ttnn.Tensor(imag_part, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
    )

    # Check if tensor values are purely real
    output = ttnn.is_real(tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    logger.info(f"Is real: {output}")


def test_conj(device):
    # Create a complex tensor
    real_part = torch.rand([1, 1, 32, 32], dtype=torch.bfloat16)
    imag_part = torch.rand([1, 1, 32, 32], dtype=torch.bfloat16)
    tensor = ttnn.complex_tensor(
        ttnn.Tensor(real_part, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
        ttnn.Tensor(imag_part, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
    )

    # Get the complex conjugate
    output = ttnn.conj(tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    logger.info(f"Complex conjugate: {output}")


def test_polar(device):
    # Create a complex tensor where real=magnitude, imag=angle
    magnitude = torch.rand([1, 1, 32, 32], dtype=torch.bfloat16)
    angle = torch.rand([1, 1, 32, 32], dtype=torch.bfloat16) * 2 * 3.14159  # angle in radians
    tensor = ttnn.complex_tensor(
        ttnn.Tensor(magnitude, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
        ttnn.Tensor(angle, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
    )

    # Convert from polar to Cartesian form
    output = ttnn.polar(tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    logger.info(f"Polar to Cartesian: {output}")


def test_alt_complex_rotate90(device):
    # Create a tensor with alternating complex layout (even last dimension required)
    tensor = ttnn.from_torch(
        torch.rand([1, 1, 32, 64], dtype=torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Rotate complex values by 90 degrees in alternating format
    output = ttnn.alt_complex_rotate90(tensor)
    logger.info(f"Complex rotated by 90 degrees: {output}")
