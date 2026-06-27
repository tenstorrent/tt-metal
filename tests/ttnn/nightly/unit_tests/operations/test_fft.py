# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("size", [32, 64, 128])
@pytest.mark.parametrize("dim", [-1, -2])
def test_fft_accuracy(device, dtype, size, dim):
    # Setup input shape
    # We use a 4D shape standard in ttnn
    input_shape = (1, 1, 64, 128) if dim == -1 else (1, 1, 128, 64)
    # Ensure the dimension size matches 'size'
    shape_list = list(input_shape)
    shape_list[dim] = size
    input_shape = tuple(shape_list)

    # Generate random real and imaginary parts
    torch.manual_seed(42)
    x_real = torch.randn(input_shape, dtype=torch.float32)
    x_imag = torch.randn(input_shape, dtype=torch.float32)
    x_torch = torch.complex(x_real, x_imag)

    # Convert torch complex tensor to ttnn ComplexTensor on device
    t_real = ttnn.Tensor(x_real, dtype).to(ttnn.TILE_LAYOUT).to(device)
    t_imag = ttnn.Tensor(x_imag, dtype).to(ttnn.TILE_LAYOUT).to(device)
    t_complex = ttnn.complex_tensor(t_real, t_imag)

    # Run ttnn FFT
    y_complex = ttnn.fft.fft(t_complex, dim=dim)

    # Extract real and imag parts, bring back to host
    y_real_dev = y_complex.real.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    y_imag_dev = y_complex.imag.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    y_dev = torch.complex(y_real_dev, y_imag_dev)

    # Run PyTorch reference
    y_torch = torch.fft.fft(x_torch, dim=dim)

    # Verify accuracy
    passing, output = comp_pcc(y_torch, y_dev, 0.99)
    assert passing, f"FFT failed: {output}"


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("size", [32, 64, 128])
@pytest.mark.parametrize("dim", [-1, -2])
def test_ifft_accuracy(device, dtype, size, dim):
    # Setup input shape
    input_shape = (1, 1, 64, 128) if dim == -1 else (1, 1, 128, 64)
    shape_list = list(input_shape)
    shape_list[dim] = size
    input_shape = tuple(shape_list)

    torch.manual_seed(42)
    x_real = torch.randn(input_shape, dtype=torch.float32)
    x_imag = torch.randn(input_shape, dtype=torch.float32)
    x_torch = torch.complex(x_real, x_imag)

    t_real = ttnn.Tensor(x_real, dtype).to(ttnn.TILE_LAYOUT).to(device)
    t_imag = ttnn.Tensor(x_imag, dtype).to(ttnn.TILE_LAYOUT).to(device)
    t_complex = ttnn.complex_tensor(t_real, t_imag)

    # Run ttnn IFFT
    y_complex = ttnn.fft.ifft(t_complex, dim=dim)

    # Extract real and imag parts, bring back to host
    y_real_dev = y_complex.real.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    y_imag_dev = y_complex.imag.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    y_dev = torch.complex(y_real_dev, y_imag_dev)

    # Run PyTorch reference
    y_torch = torch.fft.ifft(x_torch, dim=dim)

    # Verify accuracy
    passing, output = comp_pcc(y_torch, y_dev, 0.99)
    assert passing, f"IFFT failed: {output}"


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_fft_real_input(device, dtype):
    # Test FFT with real input tensor
    input_shape = (1, 1, 32, 64)
    x_real = torch.randn(input_shape, dtype=torch.float32)
    t_real = ttnn.Tensor(x_real, dtype).to(ttnn.TILE_LAYOUT).to(device)

    # Run ttnn FFT
    y_complex = ttnn.fft.fft(t_real, dim=-1)

    y_real_dev = y_complex.real.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    y_imag_dev = y_complex.imag.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    y_dev = torch.complex(y_real_dev, y_imag_dev)

    x_torch = torch.complex(x_real, torch.zeros_like(x_real))
    y_torch = torch.fft.fft(x_torch, dim=-1)

    passing, output = comp_pcc(y_torch, y_dev, 0.99)
    assert passing, f"Real FFT failed: {output}"
