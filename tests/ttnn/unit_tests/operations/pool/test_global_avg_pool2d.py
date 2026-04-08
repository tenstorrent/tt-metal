# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import torch
import pytest
import math
from models.common.utility_functions import skip_for_wormhole_b0
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


@pytest.mark.parametrize(
    "input_shape",
    (
        [1, 144, 7, 7],  # EfficientNet case: 144 channels (not tile-aligned)
        [12, 144, 56, 56],  # Larger batch with non-aligned channels
        [1, 48, 14, 14],  # 48 channels (padded to 64)
        [1, 80, 14, 14],  # 80 channels (padded to 96)
        [1, 112, 14, 14],  # 112 channels (padded to 128)
    ),
    ids=["efficientnet_144", "efficientnet_144_batch12", "mobilenet_48", "mobilenet_80", "mobilenet_112"],
)
@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16,),
)
def test_global_avg_pool2d_non_tile_aligned(
    input_shape,
    dtype,
    device,
):
    """
    Regression test for non-tile-aligned channel dimensions.

    This test ensures that global_avg_pool2d correctly handles tensors where
    the channel dimension is not a multiple of 32 (tile size). Previously,
    the operation would return the padded channel count in the output shape
    instead of the logical (unpadded) channel count.

    Example: Input with 144 channels would incorrectly output 160 channels
    (144 padded to next multiple of 32) instead of preserving 144 channels.
    """
    torch.manual_seed(0)

    torch_input_tensor = torch.randn(input_shape)
    torch_output_tensor = torch.nn.functional.adaptive_avg_pool2d(torch_input_tensor, (1, 1))

    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))  # ttnn operates on channels-last tensors
    input_tensor = ttnn.from_torch(input_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.global_avg_pool2d(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

    assert_with_pcc(torch_output_tensor, output_tensor)
