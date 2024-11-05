# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch
import random
import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

TIMEOUT = 10
# seed for random
random.seed(0)

parameters = {
    "nightly": {
        "stack_specs": [
            {"tensors_shapes": [[0, 1], [0, 1], [0, 1], [0, 1]], "dim": 2},
            {"tensors_shapes": [[0, 2], [0, 2]], "dim": 2},
            {"tensors_shapes": [[0], [0], [0], [0]], "dim": 1},
            {"tensors_shapes": [[1, 1, 16, 16], [1, 1, 16, 16]], "dim": -1},
            {"tensors_shapes": [[1, 23, 40, 64], [1, 23, 40, 64]], "dim": 4},
            {"tensors_shapes": [[1, 5, 16, 16], [1, 5, 16, 16]], "dim": -1},
            {
                "tensors_shapes": [
                    [100, 1, 256],
                    [100, 1, 256],
                    [100, 1, 256],
                    [100, 1, 256],
                    [100, 1, 256],
                    [100, 1, 256],
                ]
            },
            {
                "tensors_shapes": [[100], [100], [100], [100], [100], [100], [100], [100], [100], [100], [100], [100]],
                "dim": -1,
            },
            {"tensors_shapes": [[12, 16], [12, 16]], "dim": -1},
            {"tensors_shapes": [[12], [12], [12], [12]], "dim": 1},
            {"tensors_shapes": [[13600], [13600], [13600], [13600]], "dim": 1},
            {"tensors_shapes": [[1444], [1444], [1444], [1444], [1444], [1444], [1444], [1444]], "dim": -1},
            {"tensors_shapes": [[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], "dim": -1},
            {"tensors_shapes": [[1], [1], [1], [1], [1], [1], [1], [1]], "dim": -1},
            {"tensors_shapes": [[221], [221], [221], [221]], "dim": 1},
            {"tensors_shapes": [[25], [25], [25], [25], [25], [25], [25], [25], [25], [25], [25], [25]], "dim": -1},
            {"tensors_shapes": [[300], [300], [300], [300]], "dim": 1},
            {"tensors_shapes": [[3234, 1], [3234, 1], [3234, 1], [3234, 1]], "dim": 2},
            {"tensors_shapes": [[3234, 2], [3234, 2]], "dim": 2},
            {"tensors_shapes": [[3400], [3400], [3400], [3400]], "dim": 1},
            {
                "tensors_shapes": [[361], [361], [361], [361], [361], [361], [361], [361], [361], [361], [361], [361]],
                "dim": -1,
            },
            {
                "tensors_shapes": [[400], [400], [400], [400], [400], [400], [400], [400], [400], [400], [400], [400]],
                "dim": -1,
            },
            {"tensors_shapes": [[4], [4], [4], [4], [4], [4], [4], [4], [4], [4], [4], [4]], "dim": -1},
            {"tensors_shapes": [[63], [63], [63], [63]], "dim": 1},
            {"tensors_shapes": [[850], [850], [850], [850]], "dim": 1},
            {"tensors_shapes": [[8732, 1], [8732, 1], [8732, 1], [8732, 1]], "dim": 2},
            {"tensors_shapes": [[8732, 2], [8732, 2]], "dim": 2},
            {"tensors_shapes": [[9], [9], [9], [9], [9], [9], [9], [9], [9], [9], [9], [9]], "dim": -1},
            {"tensors_shapes": [[9], [9], [9], [9], [9], [9], [9], [9]], "dim": -1},
        ],
        "dtype": [ttnn.bfloat16],
        "layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
    }
}


# Invalidate vector is called during the generation phase where each vector will be passed in.
# If invalidated, the vector will still be stored but will be skipped.
# Returns False, None if the vector is valid, and True, str with a reason for invalidation if it is invalid.
def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if test_vector["layout"] == ttnn.ROW_MAJOR_LAYOUT:
        if test_vector["dtype"] == ttnn.bfloat8_b:
            return True, "bfloat8_b not supported with ROW_MAJOR_LAYOUT"

    return False, None


def run(
    split_specs,
    dtype,
    layout,
    *,
    device,
):
    raise Exception("Stack is not supported, TODO: Pybind wrapper for concat")
