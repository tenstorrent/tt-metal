# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of the custom ops' work planners (custom_ops/grid.py) on the Wormhole (8x8) and Blackhole
(13x10) compute grids: every split is derived from the grid, fits it, and covers the work exactly.

    pytest models/demos/gemma4/tests/unit/test_custom_op_planners_host.py --noconftest -p no:cacheprovider
"""

import pytest

from models.demos.gemma4.custom_ops import grid as g

GRIDS = [(8, 8), (13, 10)]


@pytest.mark.parametrize("grid", GRIDS, ids=["wh_8x8", "bh_13x10"])
@pytest.mark.parametrize("batch", [1, 8, 21, 22, 32])
def test_qkv_heads_plan(grid, batch):
    n_cores, per_core = g.plan_qkv_heads_units(grid, batch)
    assert n_cores <= g.grid_cores(grid)
    assert n_cores * per_core == 3 * batch
    assert per_core in (1, 3)
    if 3 * batch <= g.grid_cores(grid):
        assert per_core == 1
    assert len(g.row_major_cores(grid, n_cores)) == n_cores


def test_qkv_heads_plan_falls_back_to_one_user_per_core():
    # a 4-core grid: 3 users need 9 (user, tensor) cores -> 3 cores with the three tensors in sequence
    assert g.plan_qkv_heads_units((2, 2), 3) == (3, 3)
    assert g.plan_qkv_heads_units((2, 2), 1) == (3, 1)


@pytest.mark.parametrize("grid", GRIDS, ids=["wh_8x8", "bh_13x10"])
@pytest.mark.parametrize("num_heads, cols", [(8, 8), (8, 16), (4, 8), (2, 8), (1, 16), (16, 4)])
def test_concat_heads_plan(grid, num_heads, cols):
    n_cores, hg, groups, per = g.plan_concat_heads_units(grid, num_heads, cols)
    assert n_cores <= g.grid_cores(grid) and len(per) == n_cores
    assert sum(per) == cols * groups
    assert groups * hg >= num_heads and (groups - 1) * hg < num_heads
    assert max(per) - min(per) <= 1


@pytest.mark.parametrize("grid", GRIDS, ids=["wh_8x8", "bh_13x10"])
@pytest.mark.parametrize("cols_t, nin", [(88, 1), (120, 1), (168, 1), (88, 2), (120, 2), (168, 2), (32, 1)])
def test_norm_cores_plan(grid, cols_t, nin, monkeypatch):
    monkeypatch.delenv("GEMMA4_FUSED_NORM_CORES", raising=False)
    n = g.plan_norm_cores(grid, cols_t, nin)
    assert 1 <= n <= min(64, g.grid_cores(grid))
    assert cols_t % n == 0 and cols_t // n <= 8 // nin


def test_compute_grid_cap(monkeypatch):
    class Dev:
        class G:
            x, y = 13, 10

        def compute_with_storage_grid_size(self):
            return self.G()

    monkeypatch.delenv("GEMMA4_TEST_GRID", raising=False)
    assert g.compute_grid(Dev()) == (13, 10)
    monkeypatch.setenv("GEMMA4_TEST_GRID", "8x8")
    assert g.compute_grid(Dev()) == (8, 8)
    monkeypatch.setenv("GEMMA4_TEST_GRID", "16x16")  # never above the device grid
    assert g.compute_grid(Dev()) == (13, 10)
