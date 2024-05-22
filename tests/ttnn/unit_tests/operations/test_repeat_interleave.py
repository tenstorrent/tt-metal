# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.skip(reason="ttnn.repeat_interleave only supports repeat over dim 0 or 1")
def test_repeat_interleave(device):
    torch_input_tensor = torch.tensor([[1, 2], [3, 4]])
    torch_result = torch.repeat_interleave(torch_input_tensor, 2, dim=0)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.repeat_interleave(input_tensor, 2, dim=0)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_result, output, 0.9999)


@pytest.mark.skip(reason="ttnn.repeat_interleave only supports repeat over dim 0 or 1")
def test_repeat_interleave_with_repeat_tensor(device):
    torch_input_tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16)
    torch_repeats = torch.tensor([1, 2])
    torch_result = torch.repeat_interleave(torch_input_tensor, torch_repeats, dim=1)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    repeats = ttnn.from_torch(torch_repeats)
    output = ttnn.repeat_interleave(input_tensor, repeats, dim=1)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_result, output, 0.9999)
