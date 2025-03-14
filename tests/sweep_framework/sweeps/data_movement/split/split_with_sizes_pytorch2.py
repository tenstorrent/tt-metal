# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch
import random
import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

TIMEOUT = 10
random.seed(0)

parameters = {
    "nightly": {
        "split_with_sizes_specs": [
            {"shape": [1, 163206, 4], "split_sizes": [122400, 30600, 7650, 1989, 567], "dim": 1},
            {"shape": [1, 163206, 91], "split_sizes": [122400, 30600, 7650, 1989, 567], "dim": 1},
            {"shape": [1, 3, 16, 16, 85], "split_sizes": [2, 2, 81], "dim": 4},
            {"shape": [1, 3, 32, 32, 85], "split_sizes": [2, 2, 81], "dim": 4},
            {"shape": [1, 3, 64, 64, 85], "split_sizes": [2, 2, 81], "dim": 4},
            {"shape": [163206, 4], "split_sizes": [122400, 30600, 7650, 1989, 567]},
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
    if test_vector["dtype"] == ttnn.bfloat8_b:
        if len(test_vector["slice_specs"]["dims"]) < 2:
            return True, "bfloat8_b not supported with dims  < 2"

    return False, None


def run(
    split_with_sizes_specs,
    dtype,
    layout,
    *,
    device,
):
    raise Exception("Split with sizes is not supported, TODO: bind to slice")
