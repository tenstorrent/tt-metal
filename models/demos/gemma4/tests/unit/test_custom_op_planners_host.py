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
    # The packed-verify decode SDPA planner: sdpa_decode_program_factory's CB total per core, and the first
    # (k_chunk, cores per head) rung that fits 15/16 of the 1.5 MB L1. The two programs that overflowed on the
    # 31B at TP4 (a 2x2 Galaxy block): the global layer's packed verify (8 local q heads x 6 rows = PNHt 2,
    # head_dim 512, 1 local KV head, the 8x4 grid) read 1,884,544 B against a model of 925 tiles = 1,894,400 B
    # at (64, 16); the dFlash drafter (16 local heads x 17 rows = PNHt 9, head_dim 128, 2 local KV heads, the
    # 12x10 grid) 2,362,752 B against 1232 tiles = 2,523,136 B -- the model over-reads both, never under.
    assert g.sdpa_decode_cores_per_head(32, 1, 1, 16) == 16 and g.sdpa_decode_cores_per_head(120, 1, 2, 16) == 16
    assert g.sdpa_decode_cores_per_head(32, 5, 1, 16) == 6 and g.sdpa_decode_cores_per_head(120, 16, 4, 16) == 1
    assert g.sdpa_decode_cb_tiles(2, 16, 2, 16) == 925 and g.sdpa_decode_cb_tiles(9, 4, 2, 16) == 1232
    assert g.sdpa_decode_cb_tiles(1, 16, 2, 16) == 528 and g.sdpa_decode_cb_tiles(1, 8, 2, 16) == 288
    for name in (g.PACKED_SDPA_K_CHUNK_ENV, g.PACKED_SDPA_Q_CHUNK_ENV, g.PACKED_SDPA_K_CHUNK_LEGACY_ENV):
        monkeypatch.delenv(name, raising=False)
    plan = g.plan_packed_sdpa  # (h_local, nkv_local, rows, head_dim, batch, grid cores)
    # 12B (16 q heads, 8 kv, global 1 kv at head_dim 512) and 31B (32 q, 16 kv, global 4 kv at head_dim 512):
    # the (1,5) record (6 rows, batch 1); sliding layers on the 13x10 grid, global layers on the 8x4 grid
    cases = {
        "12B TP8 sliding": (2, 1, 6, 256, 1, 130),
        "12B TP8 global": (2, 1, 6, 512, 1, 32),
        "12B TP4 sliding": (4, 2, 6, 256, 1, 130),
        "12B TP4 global": (4, 1, 6, 512, 1, 32),
        "31B TP8 sliding": (4, 2, 6, 256, 1, 130),
        "31B TP8 global": (4, 1, 6, 512, 1, 32),
        "31B TP4 sliding": (8, 4, 6, 256, 1, 130),
        "31B TP4 global": (8, 1, 6, 512, 1, 32),
    }
    chosen = {name: plan(*args) for name, args in cases.items()}
    assert all(p.fits and p.q_chunk == 32 for p in chosen.values())
    assert {name: (p.k_chunk, p.max_cores_per_head_batch, p.tiles) for name, p in chosen.items()} == {
        "12B TP8 sliding": (64, 16, 288),
        "12B TP8 global": (64, 16, 528),
        "12B TP4 sliding": (64, 16, 288),
        "12B TP4 global": (64, 16, 528),
        "31B TP8 sliding": (64, 16, 288),
        "31B TP8 global": (64, 16, 528),
        "31B TP4 sliding": (64, 16, 509),
        "31B TP4 global": (64, 8, 637),  # 925 tiles at the defaults, 857 at a 32 chunk: the cores halve
    }
    assert chosen["31B TP4 global"].cb_bytes == 637 * 2048 <= 15 * g.SDPA_DECODE_L1_BYTES // 16
    # the 31B dFlash drafter at TP4 (PNHt 9 on head_dim 128): 1232 / 1198 / 800 / 766 tiles down the ladder, (64, 4)
    # fits at 584; at TP8 (8 local heads x 17 rows: PNHt 5, 1 KV head) the defaults fit at 700 and stay
    assert (plan(16, 2, 17, 128, 1, 120).k_chunk, plan(16, 2, 17, 128, 1, 120).max_cores_per_head_batch) == (64, 4)
    assert plan(16, 2, 17, 128, 1, 120).tiles == 584 and plan(8, 1, 17, 128, 1, 120).tiles == 700
    assert plan(8, 1, 17, 128, 1, 120).max_cores_per_head_batch == 16
    # the batched records: (5,5) at batch 5 shares the 32-core global grid six ways per row, so the defaults fit
    assert (plan(8, 1, 6, 512, 5, 32).max_cores_per_head_batch, plan(8, 1, 6, 512, 5, 32).cores_per_head) == (16, 6)
    assert plan(8, 1, 6, 512, 5, 32).tiles == 565
    # the env pins a chunk and the ladder walks the cores alone; the earlier GEMMA4_PV_K_CHUNK name still reads;
    # a pin outside the multiples of 32 is refused by name
    pinned = plan(8, 1, 6, 512, 1, 32, env={g.PACKED_SDPA_K_CHUNK_ENV: "32", g.PACKED_SDPA_Q_CHUNK_ENV: "64"})
    assert (pinned.k_chunk, pinned.q_chunk, pinned.max_cores_per_head_batch, pinned.tiles) == (32, 64, 8, 569)
    assert plan(8, 1, 6, 512, 1, 32, env={g.PACKED_SDPA_K_CHUNK_LEGACY_ENV: "32"}).k_chunk == 32
    assert plan(8, 1, 6, 512, 1, 32, env={g.PACKED_SDPA_K_CHUNK_ENV: "128"}).k_chunk == 128
    with pytest.raises(ValueError, match="K_CHUNK must be a positive multiple of 32"):  # allow-pytest.raises (pure)
        plan(8, 1, 6, 512, 1, 32, env={g.PACKED_SDPA_K_CHUNK_ENV: "48"})
    # nothing on the ladder fits: the smallest rung comes back flagged, never an exception
    huge = plan(64, 1, 32, 512, 1, 32)
    assert (
        not huge.fits and (huge.k_chunk, huge.max_cores_per_head_batch) == (32, 1) and "over the L1" in huge.describe()
    )
