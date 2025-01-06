# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_5d_concat_tile(device, layout):
    torch_input_tensors = [torch.rand(1, 32, 56, 56, 1, dtype=torch.bfloat16) for _ in range(3)]
    torch_result = torch.cat(torch_input_tensors, dim=-1)

    ttnn_input_tensors = [ttnn.from_torch(x, layout=layout, device=device) for x in torch_input_tensors]
    ttnn_result = ttnn.concat(ttnn_input_tensors, dim=-1)  # throws here
    ttnn_result = ttnn.to_torch(ttnn_result)
    assert_with_pcc(torch_result, ttnn_result, 0.9999)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_5d_concat_rm(device, layout):
    torch_input_tensors = [torch.rand(1, 32, 56, 56, 1, dtype=torch.bfloat16) for _ in range(3)]
    torch_result = torch.cat(torch_input_tensors, dim=-1)

    ttnn_input_tensors = [ttnn.from_torch(x, layout=ttnn.ROW_MAJOR_LAYOUT, device=device) for x in torch_input_tensors]
    ttnn_result = ttnn.concat(ttnn_input_tensors, dim=-1)
    ttnn_result = ttnn.to_layout(ttnn_result, layout)  # throws here
    ttnn_result = ttnn.to_torch(ttnn_result)
    assert_with_pcc(torch_result, ttnn_result, 0.9999)
