# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("h", [20])
@pytest.mark.parametrize("w", [4])
def test_repeat_interleave(device, h, w):
    tensor_tensor_2d = torch.tensor([[1, 2], [3, 4]])
    torch_result = torch.repeat_interleave(tensor_tensor_2d, 2, dim=0)

    input_tensor = ttnn.from_torch(tensor_tensor_2d, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.repeat_interleave(input_tensor, 2, dim=0)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_result, output, 0.9999)
