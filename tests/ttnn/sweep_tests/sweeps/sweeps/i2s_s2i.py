# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc
from models.utility_functions import torch_random
from tests.ttnn.sweep_tests.shard_utility import sharded_run


parameters = {
    "dtype": [ttnn.bfloat16],
    "height": [16, 32],
    "width": [16, 64, 512],
    "layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
    "input_shard_orientation": [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
    "input_num_cores_x": [1, 2, 8],
    "input_num_cores_y": [1, 4, 8],
    "input_shard_strategy": [ttnn.ShardStrategy.HEIGHT, ttnn.ShardStrategy.WIDTH],
}


def skip(**_) -> Tuple[bool, Optional[str]]:
    return False, None


def is_expected_to_fail(**_) -> Tuple[bool, Optional[str]]:
    return False, None


def run(
    dtype,
    height,
    width,
    layout,
    input_shard_orientation,
    input_num_cores_x,
    input_num_cores_y,
    input_shard_strategy,
    *,
    device,
) -> Tuple[bool, Optional[str]]:
    torch_input_tensor, output = sharded_run(
        dtype,
        height,
        width,
        layout,
        input_shard_orientation,
        input_num_cores_x,
        input_num_cores_y,
        input_shard_strategy,
        True,
        True,
        False,
        device,
    )
    return check_with_pcc(torch_input_tensor, output, 0.999)
