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
        ((1, 1, 128, 128), (1, 128, 2, 64)),
    ],
)
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("dtype", [ttnn.float32])
def test_reshape(input_shape, target_shape, layout, dtype, device):
    # Power of 2 reshape
    N = input_shape[0]
    C = input_shape[1]
    H = input_shape[2]
    W = input_shape[3]
    x = torch.rand(N * C * H * W).reshape(N, C, H, W).bfloat16().float()
    xtt = ttnn.Tensor(x, ttnn.bfloat16).to(device)

    lazy_reshaped = ttnn._ttnn.operations.data_movement.experimental_reshape(
        xtt, target_shape[0], target_shape[1], target_shape[2], target_shape[3]
    )
    reshaped = lazy_reshaped.cpu().to_torch()
    assert reshaped.shape == target_shape
    assert lazy_reshaped.producer_node() is not None
    assert lazy_reshaped.producer_node() != 0
    assert lazy_reshaped.shape == reshaped.shape
