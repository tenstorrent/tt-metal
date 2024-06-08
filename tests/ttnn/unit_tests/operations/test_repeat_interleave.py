# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("repeats", [1, 2, 3])
@pytest.mark.parametrize("dim", [0, 1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_repeat_interleave(device, repeats, dim, dtype):
    torch_input_tensor = torch.rand(1, 1, 32, 32, dtype=dtype)
    torch_result = torch.repeat_interleave(torch_input_tensor, repeats, dim=dim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.repeat_interleave(input_tensor, repeats, dim=dim)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_result, output, 0.9999)


@pytest.mark.skip(reason="ttnn.repeat_interleave only supports `repeats` as int")
def test_repeat_interleave_with_repeat_tensor(device):
    torch_input_tensor = torch.rand(1, 2, 32, 32, dtype=torch.bfloat16)
    torch_repeats = torch.tensor([1, 2])
    torch_result = torch.repeat_interleave(torch_input_tensor, torch_repeats, dim=1)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    repeats = ttnn.from_torch(torch_repeats)
    output = ttnn.repeat_interleave(input_tensor, repeats, dim=1)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_result, output, 0.9999)
