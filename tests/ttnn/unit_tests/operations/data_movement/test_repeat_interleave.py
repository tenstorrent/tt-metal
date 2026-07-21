# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_equal


@pytest.mark.parametrize("repeats", [1, 2, 3, 58])
@pytest.mark.parametrize("dim", [0, 1, 2, 3])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.uint16])
def test_repeat_interleave(device, repeats, dim, dtype):
    if dtype == ttnn.uint16:
        torch_dtype = torch.int16
        torch_input_tensor = torch.randint(0, 100, (1, 1, 32, 32), dtype=torch_dtype)
    else:
        torch_dtype = torch.bfloat16
        torch_input_tensor = torch.rand(1, 1, 32, 32, dtype=torch_dtype)

    torch_result = torch.repeat_interleave(torch_input_tensor, repeats, dim=dim)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device)
    output = ttnn.repeat_interleave(input_tensor, repeats, dim=dim)
    output = ttnn.to_torch(output)
    assert_equal(torch_result, output)


@pytest.mark.skip(reason="ttnn.repeat_interleave only supports `repeats` as int")
def test_repeat_interleave_with_repeat_tensor(device):
    torch_input_tensor = torch.rand(1, 2, 32, 32, dtype=torch.bfloat16)
    torch_repeats = torch.tensor([1, 2])
    torch_result = torch.repeat_interleave(torch_input_tensor, torch_repeats, dim=1)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    repeats = ttnn.from_torch(torch_repeats)
    output = ttnn.repeat_interleave(input_tensor, repeats, dim=1)
    output = ttnn.to_torch(output)

    assert_equal(torch_result, output)
