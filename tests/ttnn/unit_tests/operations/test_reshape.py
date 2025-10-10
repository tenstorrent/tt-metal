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
        ((2, 3), (0, 0, 1, 1)),
    ],
)
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("dtype", [ttnn.float32])
def test_reshape(input_shape, target_shape, layout, dtype, device):
    torch.manual_seed(2005)
    torch_input = torch.rand(input_shape)
    ttnn_input = ttnn.from_torch(torch_input, dtype=dtype, layout=layout, device=device)
    ttnn_output = ttnn._ttnn.operations.data_movement.experimental_reshape(ttnn_input, 0, 0, 1, 1)
    assert ttnn_output.producer_node() is not None
    assert ttnn_output.producer_node() != 0
    print(ttnn_output.producer_node())
    assert ttnn_output.shape == target_shape
    output = ttnn.to_torch(ttnn_output)
