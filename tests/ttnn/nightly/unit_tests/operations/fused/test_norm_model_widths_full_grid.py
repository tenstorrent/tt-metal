# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Layer norm and RMS norm at model widths with 4 tile rows on every core.

The large-tensor path loops over the tile rows of a core, so this runs the same check as
tests/ttnn/unit_tests/operations/fused/test_norm_model_widths.py over the whole grid (16640 x 8192 on
Blackhole). It is too big for the simulator, so it runs here, on silicon.
"""

import pytest

from tests.ttnn.unit_tests.operations.fused.test_norm_model_widths import run_norm_model_width

TILE_ROWS_PER_CORE = 4


@pytest.mark.parametrize("op", ["layer_norm", "rms_norm"])
@pytest.mark.parametrize("w", [4096, 5120, 8192])
def test_norm_model_width_hifi2_fp32_acc_full_grid(device, op, w):
    grid = device.compute_with_storage_grid_size()
    run_norm_model_width(device, op, w, h=32 * TILE_ROWS_PER_CORE * grid.x * grid.y)
