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
    "input_shape",
    [
        ((1, 1, 64, 96)),
    ],
)
def test_split(input_shape, device):
    # Power of 2 reshape
    N = input_shape[0]
    C = input_shape[1]
    H = input_shape[2]
    W = input_shape[3]
    torch_tensor = torch.randn((N, C, H, W), dtype=torch.float32)

    ttnn_tensor = ttnn.Tensor(torch_tensor, ttnn.float32).to(ttnn.TILE_LAYOUT).to(device)

    splits = ttnn.jit_split(ttnn_tensor, 2, 3)
    old_splits = ttnn.split(ttnn_tensor, 2, 3)

    assert len(old_splits) == 48
    assert len(splits) == 48

    assert splits[0] == old_splits[0]
    assert splits[1] == old_splits[1]

    assert ttnn.to_torch(splits[0]) == ttnn.to_torch(old_splits[0])
    assert ttnn.to_torch(splits[1]) == ttnn.to_torch(old_splits[1])


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
    reshaped_tensor = ttnn.jit_reshape(ttnn_tensor, target_shape[0], target_shape[1], target_shape[2], target_shape[3])
    assert reshaped_tensor.producer_node() is not None
    assert reshaped_tensor.producer_node() != 0
    assert list(reshaped_tensor.padded_shape) == list(target_shape)

    ttnn.to_torch(reshaped_tensor) == torch.reshape(torch_tensor, target_shape)
