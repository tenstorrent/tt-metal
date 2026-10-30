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
    # the served batches: one (user, tensor) unit per core wherever 3 B cores exist -- the finest split the
    # kernels have (a user's heads share one 32-row tile: the norm and the RoPE are tile-granular, so a per-head
    # split would not cut the compute per core); 8x8 folds the three tensors onto one core from 22 users
    served = {1: (3, 1), 16: (48, 1), 20: (60, 1), 24: (72, 1) if grid == (13, 10) else (24, 3)}
    for b, want in served.items():
        assert g.plan_qkv_heads_units(grid, b) == want


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
    # every output tile (head, column) is a unit of its own wherever the grid holds them (the finest split: a
    # unit assembles one output tile from the B input tiles of its column); only the 31B TP4 global layer (8
    # heads x 16 columns = 128 tiles) folds to head pairs on 8x8
    if num_heads * cols <= g.grid_cores(grid):
        assert (hg, n_cores) == (1, num_heads * cols)
    else:
        assert hg > 1 and n_cores == g.grid_cores(grid)
    served = {(4, 8): 32, (4, 16): 64, (8, 8): 64, (8, 16): 128 if grid == (13, 10) else 64}  # (heads, cols)
    assert {k: g.plan_concat_heads_units(grid, *k)[0] for k in served} == served


@pytest.mark.parametrize("grid", GRIDS, ids=["wh_8x8", "bh_13x10"])
@pytest.mark.parametrize("cols_t, nin", [(88, 1), (120, 1), (168, 1), (88, 2), (120, 2), (168, 2), (32, 1)])
def test_norm_cores_plan(grid, cols_t, nin, monkeypatch):
    monkeypatch.delenv("GEMMA4_FUSED_NORM_CORES", raising=False)
    n = g.plan_norm_cores(grid, cols_t, nin)
    assert 1 <= n <= min(64, g.grid_cores(grid))
    assert cols_t % n == 0 and cols_t // n <= 8 // nin
    # the fewest cores whose tiles fit one dest window (the all-to-all exchange grows with N^2: 11 cores beat 22
    # and 44 on a P150 at 2816 wide), the same on both grids for one row tile
    assert n == min(k for k in range(1, 65) if cols_t % k == 0 and cols_t // k <= 8 // nin)
    assert {(88, 1): 11, (88, 2): 22, (120, 1): 15, (168, 1): 21}.get((cols_t, nin), n) == n
    # a preferred count that fits is taken; the grid caps the groups of a prefill call, never the N of a row
    assert g.plan_norm_split(g.grid_cores(grid), cols_t, nin, 1, preferred=n * 2 if cols_t % (n * 2) == 0 else n)[
        0
    ] in (n, n * 2)
    assert g.plan_norm_split(g.grid_cores(grid), 88, 1, 4) == (11, 8, 4, 1)
    assert g.plan_norm_split(g.grid_cores(grid), 88, 1, 64)[:2] == (11, 8)


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
    # The packed decode SDPA planner: sdpa_decode_program_factory's CB total per core, and the first (k_chunk,
    # cores per head) rung whose bytes fit 15/16 of the L1 above the tensix unreserved base the CBs start at
    # (program.cpp checks the CB region's END ADDRESS: Blackhole base 112,000 B, Wormhole 106,080 B). The model
    # is byte-exact on the three programs that overflowed on the 31B at TP4 (a 2x2 Galaxy block): the global
    # layer's packed verify (8 local q heads x 6 rows = PNHt 2, head_dim 512, 1 local KV head in bfp8 tiles of
    # 1088 B, the 8x4 grid, a 1024 B page-table row) 1,884,544 B at the op defaults; the dFlash drafter (16 local
    # heads x 16 rows = PNHt 8, head_dim 128, 2 local KV heads in bf16, the 12x10 grid) 2,362,752 B at the
    # defaults and 1,576,320 B at (64, 8 cores per head): 715 tiles = 1,464,320 B that fit the 1,572,864 B L1
    # alone and pass it by 3,456 B above the base.
    assert g.sdpa_decode_l1("Arch.BLACKHOLE") == (1_572_864, 112_000) and g.sdpa_decode_l1(None) == (1_572_864, 112_000)
    assert g.sdpa_decode_l1("Arch.WORMHOLE_B0") == (1_499_136, 106_080)
    assert g.sdpa_decode_cb_budget_bytes() == 1_369_560 and g.sdpa_decode_cb_budget_bytes("wormhole_b0") == 1_305_990
    assert (g.sdpa_tile_bytes("DataType.BFLOAT8_B"), g.sdpa_tile_bytes("DataType.FLOAT32")) == (1088, 4096)
    assert g.sdpa_tile_bytes(None) == 2048 and g.sdpa_tile_bytes("DataType.BFLOAT16") == 2048
    assert g.sdpa_decode_page_table_bytes(256) == 1024 and g.sdpa_decode_page_table_bytes(100) == 448
    assert g.sdpa_decode_cores_per_head(32, 1, 1, 16) == 16 and g.sdpa_decode_cores_per_head(120, 1, 2, 16) == 16
    assert g.sdpa_decode_cores_per_head(32, 5, 1, 16) == 6 and g.sdpa_decode_cores_per_head(120, 16, 4, 16) == 1
    assert g.sdpa_decode_cb_tiles(2, 16, 2, 16) == 925 and g.sdpa_decode_cb_tiles(8, 4, 2, 16) == 1099
    assert g.sdpa_decode_cb_tiles(1, 16, 2, 16) == 528 and g.sdpa_decode_cb_tiles(1, 8, 2, 16) == 288
    base = g.SDPA_DECODE_L1_UNRESERVED_BYTES
    assert base + g.sdpa_decode_cb_bytes(2, 16, 2, 16, kv_tile_bytes=1088, extra_bytes=1024) == 1_884_544
    assert (
        base + g.sdpa_decode_cb_bytes(8, 4, 2, 16) == 2_362_752
        and base + g.sdpa_decode_cb_bytes(8, 4, 2, 8) == 1_576_320
    )
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
    assert all(p.fits and p.q_chunk == 32 and p.l1_end == base + p.cb_bytes for p in chosen.values())
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
    assert chosen["31B TP4 global"].cb_bytes == 637 * 2048 <= g.sdpa_decode_cb_budget_bytes()
    # the same global layer with its bfp8 cache and page-table row: the defaults reproduce the thrown 1,884,544 B
    # and the rung the target serves with, (64, 8), ends at 1,294,720 B
    served = plan(8, 1, 6, 512, 1, 32, kv_tile_bytes=1088, extra_bytes=1024)
    assert (served.k_chunk, served.max_cores_per_head_batch, served.tiles, served.l1_end) == (64, 8, 637, 1_294_720)
    assert (
        "K/V tile 1088 B, untiled 1024 B" in served.describe()
        and "8 cores/head, 7 reduction blocks" in served.describe()
    )
    # the 31B dFlash drafter at TP4 (PNHt 8 on head_dim 128, bf16 caches): 1099 / 1067 / 715 / 683 / 523 tiles down
    # the ladder, the first two rungs reproducing the thrown programs (2,362,752 and 1,576,320 B end addresses) and
    # (32, 8) still 1,510,784 B; (64, 4) fits at 523 tiles, ending at 1,183,104 B. A 17-row block (PNHt 9) lands on
    # the same rung at 584 tiles. At TP8 (8 local heads: PNHt 4, 1 KV head) the defaults fit at 567 and stay.
    drafter = plan(16, 2, 16, 128, 1, 120)
    assert (drafter.k_chunk, drafter.max_cores_per_head_batch, drafter.tiles, drafter.l1_end) == (64, 4, 523, 1_183_104)
    assert (
        "H_local 16 x rows 16 = PNHt 8, nkv_local 2, head_dim 128 (DHt 4), batch 1, grid 120 cores"
        in drafter.describe()
    )
    rungs = [
        (k, c, g.sdpa_decode_cb_tiles(8, 4, k // 32, g.sdpa_decode_cores_per_head(120, 1, 2, c)))
        for k, c in g.PACKED_SDPA_LADDER[:5]
    ]
    assert rungs == [(64, 16, 1099), (32, 16, 1067), (64, 8, 715), (32, 8, 683), (64, 4, 523)]
    assert [base + t * 2048 for _k, _c, t in rungs] == [2_362_752, 2_297_216, 1_576_320, 1_510_784, 1_183_104]
    assert (plan(16, 2, 17, 128, 1, 120).max_cores_per_head_batch, plan(16, 2, 17, 128, 1, 120).tiles) == (4, 584)
    assert (plan(8, 1, 16, 128, 1, 120).max_cores_per_head_batch, plan(8, 1, 16, 128, 1, 120).tiles) == (16, 567)
    # the same drafter on a Wormhole 8x8 grid reads that arch's L1 and base
    wh = plan(16, 2, 16, 128, 1, 64, arch="Arch.WORMHOLE_B0")
    assert (wh.max_cores_per_head_batch, wh.l1_bytes, wh.l1_base, wh.budget_bytes, wh.l1_end) == (
        4,
        1_499_136,
        106_080,
        1_305_990,
        1_177_184,
    )
    # the batched records: (5,5) at batch 5 shares the 32-core global grid six ways per row, so the defaults fit
    assert (plan(8, 1, 6, 512, 5, 32).max_cores_per_head_batch, plan(8, 1, 6, 512, 5, 32).cores_per_head) == (16, 6)
    assert plan(8, 1, 6, 512, 5, 32).tiles == 565 and plan(8, 1, 4, 512, 8, 32).tiles == 312
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
