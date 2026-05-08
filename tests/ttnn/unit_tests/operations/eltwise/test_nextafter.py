# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("shape", [(1, 1, 32, 32), (1, 1, 64, 128), (1, 3, 320, 384)])
def test_nextafter(device, shape):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand(shape, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand(shape, dtype=torch.bfloat16)

    torch_output_tensor = torch.nextafter(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.nextafter(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.999)


@pytest.mark.parametrize("ttnn_dtype", [ttnn.bfloat16, ttnn.float32])
def test_nextafter_inf(device, ttnn_dtype):
    torch_dtype = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}[ttnn_dtype]
    inf = float("inf")
    a = torch.tensor([[[[inf, -inf, inf, -inf, inf, -inf]]]], dtype=torch_dtype)
    b = torch.tensor([[[[inf, -inf, 1.0, -1.0, -inf, inf]]]], dtype=torch_dtype)

    expected = torch.nextafter(a, b)
    a_tt = ttnn.from_torch(a, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    b_tt = ttnn.from_torch(b, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    out_tt = ttnn.nextafter(a_tt, b_tt)
    actual = ttnn.to_torch(out_tt)
    assert_with_pcc(expected, actual, 0.999)


def test_nextafter_nan_propagation(device):
    nan = float("nan")
    a = torch.tensor([[[[nan, 1.0, nan]]]], dtype=torch.float32)
    b = torch.tensor([[[[1.0, nan, nan]]]], dtype=torch.float32)
    a_tt = ttnn.from_torch(a, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    b_tt = ttnn.from_torch(b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    out_tt = ttnn.nextafter(a_tt, b_tt)
    actual = ttnn.to_torch(out_tt)
    assert torch.isnan(actual).all(), f"expected all NaN, got {actual}"
