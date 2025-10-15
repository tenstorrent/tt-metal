# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from models.common.utility_functions import is_grayskull, is_blackhole
from tests.ttnn.utils_for_testing import assert_with_pcc
from .test_utils import round_up
import math


@pytest.mark.parametrize(
    "input_shape, target_shape",
    [
        ((1, 1, 64, 96), (1, 1, 96, 64)),
    ],
)
def test_reshape(input_shape, target_shape, device):
    # Power of 2 reshape
    N = input_shape[0]
    C = input_shape[1]
    H = input_shape[2]
    W = input_shape[3]
    torch_tensor = torch.randn((N, C, H, W), dtype=torch.float32)

    ttnn_tensor = ttnn.Tensor(torch_tensor, ttnn.float32).to(ttnn.TILE_LAYOUT).to(device)

    splits = ttnn.jit_split(ttnn_tensor, 2, 3)
    old_splits = ttnn.split(ttnn_tensor, 2, 3)
    assert len(splits) == 2
    assert len(old_splits) == 2

    # failing because is using a composite operation and all the old infra code was not being used
    # they actually do different things, old code behaves as pytorch, new code that uses old infra splits in n pieces instead of chunks
    assert splits[0] == old_splits[0]
    assert splits[1] == old_splits[1]


# @pytest.mark.parametrize(
#     "input_shape, target_shape",
#     [
#         ((1, 1, 64, 96), (1, 1, 96, 64)),
#     ],
# )
# def test_reshape(input_shape, target_shape, device):
#     # Power of 2 reshape
#     N = input_shape[0]
#     C = input_shape[1]
#     H = input_shape[2]
#     W = input_shape[3]
#     torch_tensor = torch.randn((N, C, H, W), dtype=torch.float32)

#     ttnn_tensor = ttnn.Tensor(torch_tensor, ttnn.float32).to(ttnn.TILE_LAYOUT).to(device)
#     reshaped_tensor = ttnn.jit_reshape(ttnn_tensor, target_shape[0], target_shape[1], target_shape[2], target_shape[3])
#     assert reshaped_tensor.producer_node() is not None
#     assert reshaped_tensor.producer_node() != 0
#     assert list(reshaped_tensor.padded_shape) == list(target_shape)

#     splits = ttnn.jit_split(reshaped_tensor, 2, 3)
#     assert len(splits) == 2
#     new_target_shape = (target_shape[0], target_shape[1], target_shape[2], target_shape[3] // 2)
#     assert splits[0].padded_shape == new_target_shape
#     assert splits[1].padded_shape == new_target_shape
#     assert splits[0].producer_node() is not None
#     assert splits[0].producer_node() != 0
#     assert splits[1].producer_node() is not None
#     assert splits[1].producer_node() != 0

#     out_tensor = ttnn.to_torch(splits[0])
#     assert out_tensor.shape == new_target_shape

#     torch_reshaped = torch.reshape(torch_tensor,  [N, C, H, W])
#     torch_converted = ttnn.to_torch(reshaped_tensor)
#     assert torch_reshaped == torch_converted

#     torch_splits = torch.reshape(torch_reshaped,  new_target_shape)

#     assert torch_splits[0] == splits[0].to_torch()
#     assert torch_splits[1] == splits[1].to_torch()
