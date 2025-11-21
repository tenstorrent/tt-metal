# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch
from loguru import logger


@pytest.mark.skip("Non-working example from the documentation. GH issue: #32364")
def test_real(device):
    # Create a complex tensor
    tensor = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device=device)

    # Get the real part
    output = ttnn.real(tensor)
    logger.info(f"Real part: {output}")


@pytest.mark.skip("Non-working example from the documentation. GH issue: #32364")
def test_imag(device):
    # Create a complex tensor
    tensor = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device=device)

    # Get the imaginary part
    output = ttnn.imag(tensor)
    logger.info(f"Imaginary part: {output}")


@pytest.mark.skip("Non-working example from the documentation. GH issue: #32364")
def test_angle(device):
    # Create a complex tensor
    tensor = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device=device)

    # Get the angle (phase) of the complex tensor
    output = ttnn.angle(tensor)
    logger.info(f"Angle: {output}")


@pytest.mark.skip("Non-working example from the documentation. GH issue: #32364")
def test_is_imag(device):
    # Create a complex tensor
    tensor = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device=device)

    # Check if tensor values are purely imaginary
    output = ttnn.is_imag(tensor)
    logger.info(f"Is imaginary: {output}")


@pytest.mark.skip("Non-working example from the documentation. GH issue: #32364")
def test_is_real(device):
    # Create a complex tensor
    tensor = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device=device)

    # Check if tensor values are purely real
    output = ttnn.is_real(tensor)
    logger.info(f"Is real: {output}")


@pytest.mark.skip("Non-working example from the documentation. GH issue: #32364")
def test_conj(device):
    # Create a complex tensor
    tensor = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device=device)

    # Get the complex conjugate
    output = ttnn.conj(tensor)
    logger.info(f"Complex conjugate: {output}")


@pytest.mark.skip("Non-working example from the documentation. GH issue: #32364")
def test_polar(device):
    # Create a complex tensor
    tensor = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device=device)

    # Convert to polar form
    output = ttnn.polar(tensor)
    logger.info(f"Polar form: {output}")


@pytest.mark.skip("Non-working example from the documentation. GH issue: #32364")
def test_alt_complex_rotate90(device):
    # Create a tensor with alternating complex layout
    tensor = ttnn.rand([2, 2], dtype=torch.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Rotate complex values by 90 degrees
    output = ttnn.alt_complex_rotate90(tensor)
    logger.info(f"Complex rotated by 90 degrees: {output}")
