# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import math
from loguru import logger
from typing import Union, Tuple

import torch
import torch.nn as nn
import ttnn
from models.utility_functions import skip_for_blackhole
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc_without_tensor_printout
from models.experimental.yolo_common.yolo_utils import determine_num_cores, get_core_grid_from_num_cores

TILE_WIDTH = 32


def determine_num_cores_for_upsample(nhw: int, width: int, max_cores=64) -> int:
    gcd_nhw_width = math.gcd(nhw, width)
    cores = nhw // gcd_nhw_width
    if cores <= max_cores:
        return cores
    for divisor in range(max_cores, 0, -1):
        if nhw % divisor == 0 and (nhw // divisor) % width == 0:
            return divisor
    return cores


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "input_shapes",
    [
        [1, 8, 28, 28],
    ],
)
@pytest.mark.parametrize("memory_layout", [ttnn.ROW_MAJOR_LAYOUT])
def test_upsample_nearest_sharded(device, input_shapes, memory_layout):
    input = torch.rand(input_shapes, dtype=torch.bfloat16)
    tt_input = input.permute(0, 2, 3, 1)
    nhw = tt_input.shape[0] * tt_input.shape[1] * tt_input.shape[2]
    num_cores = determine_num_cores_for_upsample(nhw, tt_input.shape[2])
    core_grid = get_core_grid_from_num_cores(num_cores)
    input_tensor = ttnn.from_torch(tt_input, device=device, layout=memory_layout, memory_config=ttnn.L1_MEMORY_CONFIG)

    shardspec = ttnn.create_sharded_memory_config_(
        input_tensor.shape, core_grid, ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )

    if not input_tensor.is_sharded():
        input_tensor = ttnn.interleaved_to_sharded(input_tensor, shardspec)

    input_tensor = ttnn.upsample(input_tensor, scale_factor=(8, 8), mode="nearest")
    return input_tensor
