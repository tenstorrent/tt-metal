# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import torch
import pytest
import math
from models.utility_functions import skip_for_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc
import ttnn


@pytest.mark.parametrize(
    "input_shape",
    (([1, 2048, 7, 7], ([1, 64, 1, 32]))),
    ids=["resnet50_unpadded", "tile_divisible"],
)
@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16,),
)
def test_run_average_pool2d(
    input_shape,
    dtype,
    device,
):
    torch.manual_seed(0)

    torch_input_tensor = torch.randn(input_shape)
    torch_output_tensor = torch.nn.functional.adaptive_avg_pool2d(torch_input_tensor, (1, 1))

    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))  # ttnn operates on channels-last tensors
    input_tensor = ttnn.from_torch(input_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.global_avg_pool2d(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

    assert_with_pcc(torch_output_tensor, output_tensor)
