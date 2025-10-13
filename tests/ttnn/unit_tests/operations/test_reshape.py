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
        ((1, 1, 16, 32), (1, 4, 4, 32)),
    ],
)
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("dtype", [ttnn.float32])
def test_reshape(input_shape, target_shape, layout, dtype, device):
    # Power of 2 reshape
    N = 1
    C = 1
    H = 128
    W = 128
    x = torch.rand(N * C * H * W).reshape(N, C, H, W).bfloat16().float()
    xtt = ttnn.Tensor(x, ttnn.bfloat16).to(device)

    lazy_reshaped = ttnn._ttnn.operations.data_movement.experimental_reshape(xtt, 1, 128, 2, 64)
    reshaped = lazy_reshaped.cpu().to_torch()
    assert reshaped.shape == (1, 128, 2, 64)
    assert lazy_reshaped.producer_node() is not None
    assert lazy_reshaped.producer_node() != 0
    assert lazy_reshaped.shape == reshaped.shape
