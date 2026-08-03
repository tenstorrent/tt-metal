# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_equal


# Regression test for #41631: integer dtypes must not round-trip through bf16.
@pytest.mark.parametrize("repeats", [2, 4, 32])
@pytest.mark.parametrize("dim", [0, 1, 2, 3, -1])
@pytest.mark.parametrize(
    "dtype, torch_dtype, lo, hi",
    [
        (ttnn.uint32, torch.int32, 256, 1_000_000),
        (ttnn.int32, torch.int32, 256, 1_000_000),
        (ttnn.uint16, torch.int16, 256, 30_000),
        (ttnn.uint8, torch.uint8, 0, 256),
    ],
)
def test_repeat_interleave_preserves_integer_values(device, repeats, dim, dtype, torch_dtype, lo, hi):
    torch.manual_seed(0)
    torch_input_tensor = torch.randint(lo, hi, (1, 1, 32, 32), dtype=torch_dtype)
    torch_result = torch.repeat_interleave(torch_input_tensor, repeats, dim=dim)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device)
    output = ttnn.to_torch(ttnn.repeat_interleave(input_tensor, repeats, dim=dim)).to(torch_dtype)
    assert_equal(torch_result, output)


# Regression test for #41631: fp32 must not round-trip through bf16.
@pytest.mark.parametrize("repeats", [2, 4, 32])
@pytest.mark.parametrize("dim", [0, 1, 2, 3, -1])
def test_repeat_interleave_preserves_fp32_precision(device, repeats, dim):
    torch.manual_seed(0)
    torch_input_tensor = torch.rand(1, 1, 32, 32, dtype=torch.float32) * 1000.0
    torch_result = torch.repeat_interleave(torch_input_tensor, repeats, dim=dim)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32, device=device)
    output = ttnn.to_torch(ttnn.repeat_interleave(input_tensor, repeats, dim=dim))
    assert_equal(torch_result, output)
