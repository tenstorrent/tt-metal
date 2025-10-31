# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import pytest
from tests.ttnn.unit_tests.operations.conv.data_movement.test_fold_op import run_fold_sharded_test


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "act_shape,stride_h,stride_w,padding, core_grid",
    [
        ((8, 18, 12, 8), 3, 2, (3, 2), None),
        ((2, 30, 30, 32), 5, 5, (0, 0), None),
        ((1, 24, 32, 16), 3, 4, (3, 0), None),
        ((1, 20, 20, 8), 2, 2, (1, 3, 0, 2), None),
        ((2, 16, 16, 16), 4, 4, (2, 2, 1, 3), None),
        ((1, 6, 6, 3), 1, 1, (3, 3), (1, 4)),
        ((2, 6, 6, 5), 2, 2, (1, 1), (2, 4)),
        ((4, 8, 8, 7), 2, 2, (4, 4), (4, 4)),
        ((8, 6, 6, 11), 2, 2, (1, 1), (4, 4)),
        ((16, 6, 6, 13), 2, 2, (5, 5), (8, 2)),
        ((4, 8, 8, 17), 1, 2, (2, 2), (6, 2)),
        ((8, 4, 8, 19), 2, 1, (3, 1), (2, 5)),
        ((8, 10, 6, 23), 1, 2, (0, 2), (2, 5)),
        ((10, 4, 8, 29), 2, 1, (1, 1), (2, 5)),
        ((4, 16, 16, 15), 3, 3, (1, 1), (4, 6)),
        ((16, 16, 16, 63), 3, 3, (1, 1), (8, 6)),
        ((16, 224, 224, 8), 2, 2, (4, 4), (8, 8)),
    ],
)
def test_fold_sharded(device, act_shape, stride_h, stride_w, padding, core_grid):
    run_fold_sharded_test(device, act_shape, stride_h, stride_w, padding, core_grid)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "act_shape,stride_h,stride_w,padding, core_grid",
    [
        ((1, 32, 32, 32), 2, 2, (0, 0), (2, 2)),
        ((2, 32, 32, 32), 4, 4, (0, 0), (4, 4)),
        ((1, 256, 256, 64), 8, 8, (0, 0), (8, 8)),
    ],
)
def test_fold_sharded_tile_layout(device, act_shape, stride_h, stride_w, padding, core_grid):
    run_fold_sharded_test(device, act_shape, stride_h, stride_w, padding, core_grid, ttnn.TILE_LAYOUT)
