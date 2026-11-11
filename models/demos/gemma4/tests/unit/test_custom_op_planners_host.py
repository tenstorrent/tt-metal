# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of the custom ops' work planners (custom_ops/grid.py) on the Wormhole (8x8) and Blackhole
(13x10) compute grids: every split is derived from the grid, fits it, and covers the work exactly.

    pytest models/demos/gemma4/tests/unit/test_custom_op_planners_host.py --noconftest -p no:cacheprovider
"""

import math

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


@pytest.mark.parametrize("grid", GRIDS, ids=["wh_8x8", "bh_13x10"])
@pytest.mark.parametrize("batch, heads", [(1, 1), (1, 2), (5, 8), (8, 2), (16, 1), (16, 2), (32, 2), (32, 8)])
def test_paged_update_rows_plan(grid, batch, heads):
    n_cores, per = g.plan_paged_update_rows(grid, batch, heads)
    assert n_cores == min(batch * heads, g.grid_cores(grid)) and len(per) == n_cores
    assert sum(per) == batch * heads and max(per) - min(per) <= 1 and min(per) >= 1
    assert len(g.row_major_cores(grid, n_cores)) == n_cores
    # the serving records on the 26B at TP4 (1 global / 2 sliding KV heads): one (row, head) unit per core on
    # both grids; only the 256-unit form folds (4 per core on 8x8, 1-2 on 13x10)
    served = {(1, 1): 1, (1, 2): 2, (8, 2): 16, (16, 1): 16, (16, 2): 32, (32, 2): 64}
    assert {k: g.plan_paged_update_rows(grid, *k)[0] for k in served} == served
    assert max(g.plan_paged_update_rows(grid, 32, 8)[1]) == (4 if grid == (8, 8) else 2)


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


# ── sdpa_decode_splitfree: the fixed chunking, the static (row, group, unit) -> core mapping, the scratch ──────
from models.demos.gemma4.custom_ops.sdpa_decode_splitfree import planner as sp  # noqa: E402

CTX_256K = 262144


def _plan(grid, batch, groups=1, hpg=4, head_dim=512, ctx=CTX_256K, window=None, env=None, **kw):
    return sp.plan_splitfree(
        grid, batch, groups, hpg, head_dim, ctx, window=window, kv_dtype="BFLOAT8_B", env=env or {}, **kw
    )


@pytest.mark.parametrize("grid", GRIDS, ids=["wh_8x8", "bh_13x10"])
@pytest.mark.parametrize("batch", [1, 8, 32])
def test_splitfree_mapping_covers_every_pair_once_and_balances(grid, batch):
    """Every (row, group, unit) is assigned exactly once by a function of the geometry alone (never of cur_pos);
    a lone row at 256K and 32 rows at 256K land within one unit per core of each other."""
    plan = _plan(grid, batch)
    assert plan.fits and plan.chunk == 256 and plan.unit == 4096 and plan.nsub == 16 and not plan.grid_unit
    assert plan.max_units == 64 and plan.n_slots == 64 and plan.modulus == min(plan.n_cores, 64)
    # the same (b, g, u) always goes to the same core, whatever the positions are
    cores = {(b, 0, u): plan.core_of(b, 0, u) for b in range(batch) for u in range(plan.max_units)}
    assert all(0 <= c < plan.n_cores for c in cores.values())
    full = [CTX_256K - 1] * batch
    seen = []
    for c in range(plan.n_cores):
        units = plan.units_of_core(c, full)
        assert all(cores[(b, g, u)] == c for b, g, u in units)
        assert units == sorted(units, key=lambda t: (t[0] * plan.groups + t[1], t[2]))  # ascending bg, then u
        seen += units
    assert sorted(seen) == [(b, 0, u) for b in range(batch) for u in range(plan.max_units)]
    counts = [len(plan.units_of_core(c, full)) for c in range(plan.n_cores)]
    # the stride windows tile the grid within two units per core (32 x 64 = 2048 pairs: 15..17 on 130 cores)
    assert max(counts) - min(counts) <= 2 and sum(counts) == batch * plan.max_units
    # the mapping is the same under a different cur_pos: only the live subset changes
    short = [plan.unit - 1] * batch  # one live unit per row
    assert all(plan.core_of(b, 0, 0) == cores[(b, 0, 0)] for b in range(batch))
    assert sum(len(plan.units_of_core(c, short)) for c in range(plan.n_cores)) == batch
    # padding rows have no units
    pad = [-1] * batch
    assert all(plan.units_of_core(c, pad) == [] for c in range(plan.n_cores))


@pytest.mark.parametrize("grid", GRIDS, ids=["wh_8x8", "bh_13x10"])
def test_splitfree_lone_row_spreads_over_the_grid(grid):
    plan = _plan(grid, 1)
    busy = [c for c in range(plan.n_cores) if plan.units_of_core(c, [CTX_256K - 1])]
    assert len(busy) == 64  # 64 units of 4096 tokens, one per core, on both grids
    assert max(len(plan.units_of_core(c, [CTX_256K - 1])) for c in range(plan.n_cores)) == 1
    # the same lone row inside a padded batch: its units sit on consecutive cores (core = (u' + bg) mod n), so
    # the busy core count is the same for B = 1, 8 and 32 on every grid (the earlier p = u'*BG + bg mapping
    # collapsed a B=32 lone row onto n / gcd(32, n) cores: 15 of 120, 65 of 130, 2 of 64)
    for hg in (grid, (12, 10), (11, 10)):
        for batch in (8, 32):
            bp = _plan(hg, batch)
            pos = [CTX_256K - 1] + [-1] * (batch - 1)
            bb = [c for c in range(bp.n_cores) if bp.units_of_core(c, pos)]
            assert len(bb) == min(64, bp.n_cores) and max(len(bp.units_of_core(c, pos)) for c in bb) == 1
            assert bp.stride >= batch and math.gcd(bp.stride, bp.n_cores) == 1
            assert bp.stride * bp.stride_inv % bp.n_cores == 1
            assert bp.core_of(3, 0, 5) == (5 * bp.stride + 3) % bp.n_cores
    assert sp.unit_stride(32, 130) == (33, pow(33, -1, 130)) and sp.unit_stride(32, 120) == (37, pow(37, -1, 120))
    assert sp.unit_stride(32, 64) == (33, pow(33, -1, 64)) and sp.unit_stride(1, 130) == (1, 1)
    assert sp.unit_stride(8, 120) == (11, pow(11, -1, 120)) and sp.unit_stride(4, 1) == (1, 0)
    # GEMMA4_SDPA_SPLITFREE_UNIT=grid derives the unit from the grid: one unit per core at the full context (128
    # of 130 cores with 2048 tokens; the harvested 12x10 / 11x10 / 10x10 and a one-core grid, the cap); measured
    # slower than 4096 on a 12x10 P150 (the serial combine), so it is a mode, not the default
    gp = _plan(grid, 1, env={sp.UNIT_ENV: "grid"})
    gb = [c for c in range(gp.n_cores) if gp.units_of_core(c, [CTX_256K - 1])]
    assert gp.grid_unit and (gp.unit, len(gb)) == {(8, 8): (4096, 64), (13, 10): (2048, 128)}[grid]
    assert len(gb) >= 0.98 * gp.n_cores and gp.scratch_tiles_bound == gp.batch * gp.groups * gp.n_cores * 18
    assert sp.derive_unit(256, CTX_256K, 130) == 2048 and sp.derive_unit(256, CTX_256K, 64) == 4096
    assert sp.derive_unit(256, CTX_256K, 120) == 2304 and sp.derive_unit(256, CTX_256K, 110) == 2560
    assert sp.derive_unit(256, CTX_256K, 100) == 2816 and sp.derive_unit(256, CTX_256K, 1) == 4096
    assert sp.derive_unit(256, 4096, 130) == 256 and sp.derive_unit(128, CTX_256K, 64) == 4096
    assert sp.derive_unit(256, 1 << 20, 130) == 4096
    for hg, unit, units in (((12, 10), 2304, 114), ((11, 10), 2560, 103), ((10, 10), 2816, 94)):
        hp = _plan(hg, 1, env={sp.UNIT_ENV: "grid"})
        hb = [c for c in range(hp.n_cores) if hp.units_of_core(c, [CTX_256K - 1])]
        assert (hp.unit, hp.max_units, len(hb)) == (unit, units, units) and len(hb) >= 0.9 * hp.n_cores
    pinned = _plan(grid, 1, env={sp.UNIT_ENV: "2048"})
    assert pinned.unit == 2048 and pinned.max_units == 128 and not pinned.grid_unit
    assert len([c for c in range(pinned.n_cores) if pinned.units_of_core(c, [CTX_256K - 1])]) == min(
        128, pinned.n_cores
    )
    # at 1M tokens every core of the grid carries units (the cap keeps the unit at 4096: 256 units)
    big = _plan(grid, 1, ctx=1 << 20)
    assert big.max_units == 256 and big.modulus == min(big.n_cores, 256)
    counts = [len(big.units_of_core(c, [(1 << 20) - 1])) for c in range(big.n_cores)]
    assert min(counts) >= 1 and max(counts) - min(counts) <= 1


def test_splitfree_chunks_and_masks_follow_cur_pos_like_the_stock():
    plan = _plan((13, 10), 1)  # the arithmetic below at the default 4096 unit
    derived = _plan((13, 10), 1, env={sp.UNIT_ENV: "grid"})  # the grid-derived unit halves it: chunks [8u, 8u+7]
    assert derived.unit == 2048 and derived.live_units(4095) == range(0, 2)
    assert derived.unit_chunks(1, 4095) == range(8, 16) and derived.unit_chunks(0, 4095) == range(0, 8)
    assert derived.live_units(CTX_256K - 1) == range(0, 128) and derived.combine_order(0, 0, 4095) == [(0, 0), (1, 1)]
    # unit u holds chunks [16u, 16u+15]; the chunk holding cur_pos is the tail, later chunks are skipped
    assert plan.live_units(0) == range(0, 1) and plan.unit_chunks(0, 0) == range(0, 1)
    assert plan.live_units(4095) == range(0, 1) and plan.unit_chunks(0, 4095) == range(0, 16)
    assert plan.live_units(4096) == range(0, 2) and plan.unit_chunks(1, 4096) == range(16, 17)
    assert plan.live_units(CTX_256K - 1) == range(0, 64) and plan.unit_chunks(63, CTX_256K - 1) == range(1008, 1024)
    assert plan.row_range(70000) == (0, 273, 0)  # (first chunk, tail chunk, window start)
    assert plan.live_units(-1) == range(0) and plan.unit_chunks(0, None) == range(0)
    # the combine walks the live units ascending, each from its own slot
    assert plan.combine_order(0, 0, 8191) == [(0, 0), (1, 1)]
    assert plan.combine_order(3, 0, 4096) == [(0, 3 * 64), (1, 3 * 64 + 1)]
    assert plan.combine_core(0, 0) == 0 and plan.combine_core(31, 0) == 31


@pytest.mark.parametrize("grid", GRIDS, ids=["wh_8x8", "bh_13x10"])
def test_splitfree_sliding_layer(grid):
    """A 1024-token window on 256-token chunks: one chunk per unit, five slots, the live range follows the
    position, the window start chunk carries the mask, consecutive live units never share a slot."""
    plan = _plan(grid, 32, groups=2, hpg=2, head_dim=256, window=1024)
    assert plan.unit == plan.chunk == 256 and plan.nsub == 1 and plan.n_slots == 5
    assert plan.max_units == 1024 and plan.modulus == plan.n_cores
    assert plan.row_range(5000) == (15, 19, 3977)  # chunks 15..19 live, window start 3977 inside chunk 15
    assert plan.live_units(5000) == range(15, 20) and plan.live_units(1023) == range(0, 4)
    assert plan.live_units(1024) == range(0, 5) and plan.row_range(1024) == (0, 4, 1)
    slots = [s for _u, s in plan.combine_order(0, 1, 5000)]
    assert len(set(slots)) == 5 and all(5 <= s < 10 for s in slots)
    # every (b, g, u) of the live range lands on exactly one core, balanced over the grid
    pos = [5000 + 37 * b for b in range(32)]
    units = [t for c in range(plan.n_cores) for t in plan.units_of_core(c, pos)]
    want = [(b, g, u) for b in range(32) for g in range(2) for u in plan.live_units(pos[b])]
    assert sorted(units) == sorted(want)
    counts = [len(plan.units_of_core(c, pos)) for c in range(plan.n_cores)]
    # 320 live units over the grid: a sparse subset of the pair space, so cores carry 0..3 (8x8: 5 each exactly)
    assert sum(counts) == 320 and max(counts) <= 2 * sp.ceil_div(320, plan.n_cores)
    assert all(plan.core_of(b, g, u) == plan.core_of(b, g, u + plan.modulus) for b, g, u in want[:40])


def test_splitfree_chunk_follows_the_l1_budget_not_the_batch():
    """The chunk is decided by head dim, cache format and arch (never batch or grid): bfp8 at head_dim 512 takes
    256 on both archs; bf16 still fits 256 on Blackhole (1,362,048 of 1,369,560 B with the fp32 scores and scaler CBs) and
    drops to 128 on Wormhole; the env caps it; the unit stays 4096 (``=grid`` derives it, never the batch)."""
    bfp8 = {(g, b): _plan(g, b) for g in GRIDS for b in (1, 32)}
    assert {p.chunk for p in bfp8.values()} == {256} and {p.unit for p in bfp8.values()} == {4096}
    assert all(p.fits for p in bfp8.values())
    bf16 = sp.plan_splitfree((13, 10), 32, 1, 4, 512, CTX_256K, kv_dtype="BFLOAT16", env={})
    assert bf16.chunk == 256 and bf16.unit == 4096 and bf16.fits and bf16.partial_cb_bytes == 1_362_048
    bf16_qk = sp.plan_splitfree((13, 10), 32, 1, 4, 512, CTX_256K, kv_dtype="BFLOAT16", env={sp.QK_FP32_ENV: "0"})
    assert bf16_qk.qk_tile_bytes == 2048 and bf16_qk.partial_cb_bytes == 1_343_616 and bf16_qk.chunk == 256
    wh = sp.plan_splitfree((8, 8), 32, 1, 4, 512, CTX_256K, kv_dtype="BFLOAT16", arch="Arch.WORMHOLE_B0", env={})
    assert wh.chunk == 128 and wh.fits and wh.budget_bytes == 1_305_990
    capped = _plan((13, 10), 32, env={sp.CHUNK_ENV: "64", sp.UNIT_ENV: "2048"})
    assert capped.chunk == 64 and capped.unit == 2048 and capped.max_units == 128
    with pytest.raises(ValueError, match="multiple of 32"):  # allow-pytest.raises (pure)
        _plan((13, 10), 1, env={sp.CHUNK_ENV: "48"})
    with pytest.raises(ValueError, match="one core per"):  # allow-pytest.raises (pure)
        sp.plan_splitfree((4, 4), 32, 1, 4, 512, CTX_256K, env={})
    # the bytes behind the choice: bfp8 chunk 256 at head_dim 512 on the Blackhole budget (fp32 scores: 16 KB
    # over the bf16 scores CB the stock keeps, which cannot hold a 256K row's scores to 0.999)
    p = bfp8[((13, 10), 32)]
    assert (
        p.qk_tile_bytes == 4096 and p.partial_cb_bytes == sp.partial_cb_bytes(16, 8, 1088, 0, 128, 0, 4096) == 870_528
    )
    assert sp.partial_cb_bytes(16, 8, 1088, 0, 128, 0) == 852_096
    assert p.combine_cb_bytes == sp.combine_cb_bytes(16, 128) == 340_096
    assert "chunk 256 x 16 = unit 4096, 64 units, 64 slots" in p.describe()


def test_splitfree_scratch_sizes_of_the_galaxy_builds():
    """Per engine (one chip), 32 padded rows at 256K: 64 slots x 18 fp32 tiles per row for the head_dim-512
    global layers = 144 MiB whatever the local head count or the grid; the sliding layers (head_dim 256, 2 local
    KV groups) need 12.5 MiB, so one model scratch takes the global figure. Under ``=grid`` the scratch takes
    one slot per core (288 MiB on 13x10) so a call on a narrower page table (a smaller unit, up to n_cores
    units) still fits."""
    p31 = _plan((13, 10), 32, groups=1, hpg=4)  # 31B TP8 global: 4 local q heads, one replicated KV head
    p12 = _plan((13, 10), 32, groups=1, hpg=2)  # 12B TP8 global: 2 local q heads
    assert p31.scratch_tiles == p12.scratch_tiles == 32 * 64 * 18 == 36864
    assert p31.scratch_bytes == 36864 * 4096 == 150_994_944 and p31.scratch_tiles_bound == 36864
    assert _plan((8, 8), 32, groups=1, hpg=4).scratch_tiles == 36864
    sliding = _plan((13, 10), 32, groups=2, hpg=2, head_dim=256, window=1024)
    assert sliding.scratch_tiles == 32 * 2 * 5 * 10 == 3200 and sliding.scratch_bytes == 13_107_200
    specs = [(1, 4, 512, None), (2, 2, 256, 1024)]
    assert sp.scratch_tiles_for((13, 10), 32, specs, CTX_256K, kv_dtype="BFLOAT8_B", env={}) == 36864
    assert sp.scratch_tiles_for((8, 8), 32, specs, CTX_256K, kv_dtype="BFLOAT16", env={}) == 36864
    assert sp.scratch_tiles_for((13, 10), 1, specs, CTX_256K, env={}) == 64 * 18
    gm = {sp.UNIT_ENV: "grid"}
    g31 = _plan((13, 10), 32, groups=1, hpg=4, env=gm)
    assert (g31.unit, g31.scratch_tiles, g31.scratch_tiles_bound) == (2048, 32 * 128 * 18, 32 * 130 * 18)
    assert sp.scratch_tiles_for((13, 10), 32, specs, CTX_256K, kv_dtype="BFLOAT8_B", env=gm) == 74880
    narrow = _plan((13, 10), 32, ctx=33280, env=gm)
    assert (narrow.unit, narrow.max_units, narrow.scratch_tiles) == (256, 130, 74880)
    assert all(_plan((13, 10), 32, ctx=c, env=gm).scratch_tiles <= 74880 for c in range(4096, CTX_256K + 1, 4096))
    # a 128K build halves it; the 31B at TP4 (8 local q heads over one KV group) needs the same
    assert sp.scratch_tiles_for((13, 10), 32, [(1, 8, 512, None)], CTX_256K // 2, env={}) == 32 * 32 * 18
    # the model's own sizing (attach_sdpa_splitfree_scratch, also run by the vLLM generator once the plugin's
    # caches exist): the covered layers and their per-chip specs of the 12B (16 q / 8 kv heads, head_dim 256 /
    # 512 global, window 1024, 48 layers in the 5:1 pattern) at TP4 and TP8, mode "1" (global) and "2" (all)
    from types import SimpleNamespace

    from models.demos.gemma4.tt.model import splitfree_layer_specs

    hf = SimpleNamespace(
        layer_types=[("full_attention" if i % 6 == 5 else "sliding_attention") for i in range(48)],
        hidden_size=3840,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=256,
        global_head_dim=512,
        sliding_window=1024,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        global_rope_theta=1000000.0,
        partial_rotary_factor=0.25,
    )
    covered, specs = splitfree_layer_specs(hf, 48, 4, "1")
    assert covered == {i for i in range(48) if i % 6 == 5} and specs == [(2, 2, 512, None)]
    assert sp.scratch_tiles_for((13, 10), 32, specs, CTX_256K, env={}) == 32 * 2 * 64 * 18  # 288 MiB at TP4
    covered, specs = splitfree_layer_specs(hf, 48, 4, "2")
    assert covered == set(range(48)) and specs == [(2, 2, 256, 1024), (2, 2, 512, None)]
    assert sp.scratch_tiles_for((13, 10), 32, specs, CTX_256K, env={}) == 32 * 2 * 64 * 18  # the global figure
    assert splitfree_layer_specs(hf, 48, 8, "1")[1] == [(1, 2, 512, None)]
    assert splitfree_layer_specs(hf, 12, 16, "2")[1] == [(1, 1, 256, 1024), (1, 1, 512, None)]  # kv replicated
    assert splitfree_layer_specs(hf, 48, 4, "0") == (set(), [])


def test_splitfree_golden_gathers_through_the_page_table():
    """The torch reference of the device test (``SdpaDecodeSplitfree.golden``): per row and KV group, the
    softmax over the positions the page table maps, the window and the bounded ring applied as the kernels
    apply them (checked against an explicit gather)."""
    import torch

    from models.demos.gemma4.custom_ops.sdpa_decode_splitfree import SdpaDecodeSplitfree

    torch.manual_seed(0)
    blocks, nkv, block, D, Hq, B = 16, 2, 32, 64, 4, 3
    k = torch.randn(blocks, nkv, block, D)
    v = torch.randn(blocks, nkv, block, D)
    q = torch.randn(1, B, Hq, D)
    tables = torch.stack([torch.randperm(blocks) for _ in range(B)]).to(torch.int32)
    cur_pos = torch.tensor([200, -1, 37])
    for window, ring in ((None, None), (100, None), (100, 256)):
        got = SdpaDecodeSplitfree.golden(
            q, k, v, tables, cur_pos, scale=0.5, sliding_window=window, cache_position_modulo=ring
        )
        assert got.shape == (1, B, Hq, D) and torch.all(got[0, 1] == 0)
        for b in (0, 2):
            pos = int(cur_pos[b])
            lo = max(0, pos + 1 - window) if window else 0
            for h in range(Hq):
                g = h // (Hq // nkv)
                rows = []
                for t in range(lo, pos + 1):
                    tv = t % ring if ring else t
                    rows.append((int(tables[b, tv // block]), tv % block))
                kk = torch.stack([k[blk, g, off] for blk, off in rows])
                vv = torch.stack([v[blk, g, off] for blk, off in rows])
                want = torch.softmax(q[0, b, h] @ kk.T * 0.5, dim=-1) @ vv
                assert torch.allclose(got[0, b, h], want, atol=1e-5), (window, ring, b, h)


# ── sdpa_decode_lse: the stock flash-decode factory's core assignment and tree, re-expressed in python ─────────────
from models.demos.gemma4.custom_ops.sdpa_decode_lse import planner as lp  # noqa: E402


def test_sdpa_decode_lse_tree_merges_every_member_once_and_gives_the_lse():
    """For every group size up to 64 cores and every chunk count around it: each member's (parent, send round) is a
    child slot of its parent (so the parent waits exactly once per sender, at that round), parents precede their
    children (no wait cycle), the rounds stay below ceil(log2 n), the K chunks of ``workload_for_core`` cover the
    row once, and the kernels' merge along the tree of random per-member partials (raw-score max m, l = sum of
    exp((s - m) * scale), unnormalised O) gives softmax(scale * s) @ V and ``scale * m + ln(l)`` = logsumexp."""
    import torch

    torch.manual_seed(0)
    scale, chunk, D = 0.7, 2, 3
    for n in range(1, lp.MAX_CORES_PER_HEAD + 1):
        params = [lp.tree_reduction_params(c, n) for c in range(n)]
        rounds = lp.tree_rounds(n)
        assert params[0].is_root and params[0].parent == lp.NONE
        for c in range(1, n):
            p = params[c]
            assert p.parent < c and p.send_at_round < rounds, (n, c)
            assert params[p.parent].children[p.send_at_round] == c, (n, c)
        for c in range(n):
            assert all(ch == lp.NONE or (r < rounds and ch > c) for r, ch in enumerate(params[c].children))
        for k in sorted({1, max(1, n - 1), n, n + 1, 2 * n + 3}):
            s = torch.randn(k * chunk, dtype=torch.float64) * 3
            v = torch.randn(k * chunk, D, dtype=torch.float64)
            spans = [lp.workload_for_core(k * chunk - 1, c, n, chunk)[2:] for c in range(n)]
            assert sorted(i for a, b in spans for i in range(a, b)) == list(range(k)), (n, k)

            def merged(c):  # the member's partial after its rounds (writer receives, compute correction_block)
                a, b = spans[c]
                keys = slice(a * chunk, b * chunk)
                m = s[keys].max()
                p = torch.exp((s[keys] - m) * scale)
                l, o = p.sum(), p @ v[keys]
                for child in params[c].children:
                    if child != lp.NONE and child < k:  # the kernels' "child has data" test
                        mc, lc, oc = merged(child)
                        mn = torch.maximum(m, mc)
                        ea, eb = torch.exp((m - mn) * scale), torch.exp((mc - mn) * scale)
                        m, l, o = mn, l * ea + lc * eb, o * ea + oc * eb
                return m, l, o

            m, l, o = merged(0)
            assert torch.allclose(o / l, torch.softmax(s * scale, 0) @ v), (n, k)
            assert torch.allclose(scale * m + torch.log(l), torch.logsumexp(s * scale, 0)), (n, k)


@pytest.mark.parametrize("grid", GRIDS + [(12, 10), (8, 4)], ids=["wh_8x8", "bh_13x10", "bh_12x10", "glx_8x4"])
def test_sdpa_decode_lse_plan_is_the_factory_split(grid):
    """The core plan is the stock factory's (``grid.sdpa_decode_cores_per_head`` mirrors it), groups are contiguous
    row-major runs with the root first, a group past 64 cores is refused, every per-core argument list has the
    length the copied kernels parse, the reduction staging holds one block per round, and the served shapes
    (head_dim 512 bfp8, k chunk 64, caps 16 / 32 / 64) fit the per-core L1."""
    n_grid = g.grid_cores(grid)
    for B in (1, 4, 8, 32):
        for cap in (1, 3, 16, 32, 64, 128):
            if B > n_grid:
                continue
            want = g.sdpa_decode_cores_per_head(n_grid, B, 1, cap)
            if lp.tree_rounds(want) > lp.MAX_TREE_REDUCTION_ROUNDS:
                with pytest.raises(ValueError, match="rounds"):  # allow-pytest.raises (pure)
                    lp.plan_cores(grid, B, cap)
                continue
            plan = lp.plan_cores(grid, B, cap)
            assert plan.cores_per_head == want and plan.active_cores == want * B <= n_grid
            for b in range(B):
                grp = plan.group(b)
                assert [w.member for w in grp] == list(range(want)) and {w.batch for w in grp} == {b}
                assert grp[0].do_reduce and grp[0].do_output and grp[0].tree.is_root
            assert [w.core for w in plan.work] == [(i % grid[0], i // grid[0]) for i in range(plan.active_cores)]
            if B <= 4 and cap in (16, 32, 64):
                geo = lp.Geometry(
                    batch=B,
                    num_q_heads=4,
                    head_dim=512,
                    max_blocks=4096,
                    block_size=64,
                    num_cache_blocks=4096 * B,
                    kv_dtype="bfloat8_b",
                    k_chunk_size=64,
                    scale=1.0,
                    cur_pos_stick=64,
                    page_table_stick=4096 * 4,
                )
                full = lp.plan(geo, grid, cap)
                assert len(full.reader_ct) == lp.READER_CT_SCALARS and len(full.writer_ct) == lp.WRITER_CT_SCALARS
                assert len(full.compute_ct) == lp.COMPUTE_CT_ARGS
                stage = [s for s in full.cbs if s.index == lp.CB_INTERMED]
                assert (stage[0].pages if stage else 0) == (geo.pnht * geo.dht + 2 * geo.pnht) * plan.rounds
                assert full.cb_total_bytes <= g.sdpa_decode_cb_budget_bytes(), full.describe()
                addrs = dict(q=1, k=2, v=3, cur_pos=4, page_table=5, out=6, stats=7)
                rt = lp.runtime_args(geo, full.cores, lambda xy: (xy[0] + 1, xy[1] + 2), addrs)
                assert set(rt) == {w.core for w in plan.work}
                for w in plan.work:
                    reader, writer, compute = rt[w.core]
                    assert len(reader) == lp.READER_RT_FIXED + 2 * B and reader[0] == 1 and reader[10] == w.batch
                    assert len(writer) == lp.WRITER_RT_FIXED + 2 * want + 4 * B and writer[:2] == [6, 7]
                    assert writer[lp.WRITER_RT_FIXED : lp.WRITER_RT_FIXED + want] == [
                        m.core[0] + 1 for m in plan.group(w.batch)
                    ]
                    assert len(compute) == lp.COMPUTE_RT_ARGS and compute[0] == int(w.member == 0)


def test_sdpa_decode_lse_merge_of_two_halves_is_the_full_row():
    """The torch side of the pair split the device test runs: the golden over keys [0, mid) and over [mid, pos]
    through the page table shifted by mid / block blocks (cur_pos - mid), merged by the LSE formula, equals the
    golden over [0, pos]; ``lse_from_stats`` reads m at column 0 and l at column 32; a skipped half (-inf) with a
    garbage output carries no weight."""
    import torch

    from models.demos.gemma4.custom_ops.sdpa_decode_lse import SdpaDecodeLse

    torch.manual_seed(0)
    blocks, block, D, Hq, B, scale = 24, 32, 64, 4, 3, 0.5
    k = torch.randn(blocks, 1, block, D)
    v = torch.randn(blocks, 1, block, D)
    q = torch.randn(1, B, Hq, D)
    width = blocks // B
    table = torch.randperm(blocks).reshape(B, width).to(torch.int32)
    cur_pos = torch.tensor([width * block - 1, 40, 100])
    mid = 64
    shifted = torch.zeros_like(table)
    shifted[:, : width - mid // block] = table[:, mid // block :]
    pos_a = torch.where(cur_pos >= mid, torch.full_like(cur_pos, mid - 1), cur_pos)
    pos_b = torch.where(cur_pos >= mid, cur_pos - mid, torch.full_like(cur_pos, -1))
    full, lse_full = SdpaDecodeLse.golden(q, k, v, table, cur_pos, scale=scale)
    out_a, lse_a = SdpaDecodeLse.golden(q, k, v, table, pos_a, scale=scale)
    out_b, lse_b = SdpaDecodeLse.golden(q, k, v, shifted, pos_b, scale=scale)
    out_b[0, 1] = float("nan")  # the device leaves a skipped row's output untouched
    merged, lse_m = SdpaDecodeLse.merge_torch(out_a, lse_a, out_b, lse_b)
    assert torch.allclose(merged, full, atol=1e-5) and torch.allclose(lse_m, lse_full, atol=1e-5)
    # the stats layout: [1, B, 32, 64], m at column 0, l at column 32
    m = torch.randn(1, B, 32) * 4
    l = torch.rand(1, B, 32) * 100 + 1
    stats = torch.zeros(1, B, 32, 64)
    stats[..., 0], stats[..., 32] = m, l
    got = SdpaDecodeLse.lse_from_stats(stats, Hq, scale)
    assert torch.allclose(got, scale * m[..., :Hq] + torch.log(l[..., :Hq]))


def _lse_geometry(**kw):
    base = dict(
        batch=4,
        num_q_heads=8,
        head_dim=512,
        max_blocks=2048,
        block_size=64,
        num_cache_blocks=8192,
        kv_dtype="bfloat8_b",
        k_chunk_size=64,
        scale=1.0,
        cur_pos_stick=64,
        page_table_stick=8192,
    )
    base.update(kw)
    return lp.Geometry(**base)


def test_sdpa_decode_lse_pair_split_refusals_and_plan():
    """pair_split runs on a 2-D mesh with an even number of columns (the pair is columns (c, c ^ 1), the half is the
    column's parity) and is refused on a device that is not a mesh, a single-device mesh, a non-2-D shape and an odd
    column count. The pair plan is the stock plan plus the SDPA_LSE_PAIR_SPLIT define on every kernel and one
    compile-time arg (the half); without pair_split no define or argument is added."""
    assert "mesh device" in lp.pair_split_reason(None)
    assert "single device" in lp.pair_split_reason((1, 1))
    assert "2-D" in lp.pair_split_reason((8,))
    for shape in [(1, 3), (2, 3), (4, 1), (8, 1)]:
        assert "odd" in lp.pair_split_reason(shape), shape
    for shape in [(1, 2), (1, 8), (2, 4), (8, 4), (4, 8)]:
        assert lp.pair_split_reason(shape) is None, shape
    assert [lp.pair_half_of(c) for c in range(8)] == [0, 1] * 4
    assert [lp.pair_partner((3, c)) for c in range(4)] == [(3, 1), (3, 0), (3, 3), (3, 2)]

    geo = _lse_geometry()
    stock = lp.plan(geo, (13, 10), 16)
    pair = lp.plan(geo, (13, 10), 16, pair_split=True)
    assert not stock.pair_split and stock.dataflow_defines == () and stock.pair_ct(0) == stock.pair_ct(1) == ()
    assert lp.PAIR_SPLIT_DEFINE not in stock.defines
    assert pair.defines == stock.defines + (lp.PAIR_SPLIT_DEFINE,) and pair.dataflow_defines == (lp.PAIR_SPLIT_DEFINE,)
    assert [pair.pair_ct(h) for h in (0, 1)] == [(0,), (1,)]
    assert (pair.cores, pair.cbs, pair.reader_ct, pair.writer_ct, pair.compute_ct) == (
        stock.cores,
        stock.cbs,
        stock.reader_ct,
        stock.writer_ct,
        stock.compute_ct,
    )
    assert len(pair.compute_ct) == lp.COMPUTE_CT_ARGS  # the half lands at compute arg COMPUTE_CT_ARGS
    with pytest.raises(ValueError, match="pair half"):  # allow-pytest.raises (pure)
        pair.pair_ct(2)


def test_sdpa_decode_lse_pair_halves_split_the_row():
    """The pair split's chunk ranges (``pair_workload_for_core``, the mirror of kernels/pair_split.hpp): half 0 takes
    chunks [0, ceil(n/2)), half 1 [ceil(n/2), n), together every chunk once; on a device the members with data are a
    prefix and ``member < half_chunks`` is exactly "has data" (the kernels' active-child test); member 0 ends its
    half, so the row's last chunk (the causal mask's) sits on the root of the half holding it -- half 1, or half 0
    of a one-chunk row, whose half 1 is the empty one; no core takes more than ceil(n / (2 * cores)) chunks (the
    balance of one 2 * cores group); ``pair_key_range`` is the half's chunks clipped to cur_pos."""
    for k in (32, 64, 128):
        for cores in (1, 2, 3, 5, 16, 64):
            positions = {0, 1, k - 2, k - 1, k, k + 36, 2 * k - 1, 2 * k, 3 * k + 5, 4095, 4096, 65536, 131071}
            positions |= {2 * cores * k - 1, 2 * cores * k, 2 * cores * k + k, cores * k + 3}
            for cur_pos in sorted(positions):
                n = -(-(cur_pos + 1) // k)
                first = (n + 1) // 2
                seen = []
                for half in (0, 1):
                    work = [lp.pair_workload_for_core(cur_pos, c, cores, k, half) for c in range(cores)]
                    base, cnt = (0, first) if half == 0 else (first, n - first)
                    assert lp.pair_half_chunks(n, half) == (base, cnt)
                    assert all(w[0] == n * k // lp.TILE and w[1] == n and w[4] == cnt for w in work), (
                        k,
                        cores,
                        cur_pos,
                    )
                    chunks = sorted(i for w in work for i in range(w[2], w[3]))
                    assert chunks == list(range(base, base + cnt)), (k, cores, cur_pos, half)
                    assert [w[2] < w[3] for w in work] == [c < cnt for c in range(cores)], (k, cores, cur_pos, half)
                    assert max(w[3] - w[2] for w in work) <= -(-n // (2 * cores)), (k, cores, cur_pos, half)
                    if cnt:
                        assert work[0][3] == base + cnt
                        if base + cnt == n:  # this half holds the row's last chunk: on its root
                            assert [c for c, w in enumerate(work) if w[2] <= n - 1 < w[3]] == [0]
                    lo, hi = lp.pair_key_range(cur_pos, k, half)
                    want = (base * k, min((base + cnt) * k, cur_pos + 1)) if cnt else (cur_pos + 1, cur_pos + 1)
                    assert (lo, hi) == want, (k, cur_pos, half)
                    seen += chunks
                assert sorted(seen) == list(range(n))
                assert (n == 1) == (lp.pair_half_chunks(n, 1)[1] == 0)
    assert lp.pair_key_range(-1, 64, 0) == lp.pair_key_range(-1, 64, 1) == (0, 0)


def test_sdpa_decode_lse_pair_split_tree_and_merge_give_the_full_row():
    """Both chips of a pair in torch: per half, each member's chunks (the causal mask clips the last), the group tree
    with the pair active-child test (``child < half_chunks``), the root's (m, l, O / l) -- or the empty-half sentinel
    O = 0, m = bf16(-30000), l = 1 -- then the partner merge ``w_self = sigmoid(lse_self - lse_partner)``,
    ``O = w_self * O_self + w_partner * O_partner``: softmax(scale * s) @ V, and logaddexp of the halves' lse is the
    row's logsumexp."""
    import torch

    torch.manual_seed(1)
    scale, k, D = 0.5, 32, 3
    m_empty = float(torch.tensor(lp.PAIR_EMPTY_M).to(torch.bfloat16))
    for cores in (1, 2, 3, 4, 5, 8):
        params = [lp.tree_reduction_params(c, cores) for c in range(cores)]
        for cur_pos in sorted({0, 5, 31, 32, 40, 95, 96, 200, 2 * cores * k + 7, 5 * cores * k + 70}):
            s = torch.randn(cur_pos + 1, dtype=torch.float64) * 3
            v = torch.randn(cur_pos + 1, D, dtype=torch.float64)
            halves = []
            for half in (0, 1):
                work = [lp.pair_workload_for_core(cur_pos, c, cores, k, half) for c in range(cores)]
                cnt = work[0][4]

                def merged(c):  # the member's partial after its rounds (writer receives, compute correction_block)
                    keys = slice(work[c][2] * k, min(work[c][3] * k, cur_pos + 1))
                    m = s[keys].max()
                    p = torch.exp((s[keys] - m) * scale)
                    l, o = p.sum(), p @ v[keys]
                    for child in params[c].children:
                        if child != lp.NONE and child < cnt:  # the pair "child has data" test
                            mc, lc, oc = merged(child)
                            mn = torch.maximum(m, mc)
                            ea, eb = torch.exp((m - mn) * scale), torch.exp((mc - mn) * scale)
                            m, l, o = mn, l * ea + lc * eb, o * ea + oc * eb
                    return m, l, o

                if work[0][2] == work[0][3]:  # an empty half: no member has data, the root writes the sentinel
                    assert cnt == 0 and half == 1 and cur_pos < k
                    m, l, o = torch.tensor(m_empty, dtype=torch.float64), torch.tensor(lp.PAIR_EMPTY_L), torch.zeros(D)
                else:
                    m, l, o = merged(0)
                    o = o / l
                halves.append((scale * m + torch.log(l), o.double()))
            (lse0, o0), (lse1, o1) = halves
            w0, w1 = torch.sigmoid(lse0 - lse1), torch.sigmoid(lse1 - lse0)
            assert torch.allclose(w0 * o0 + w1 * o1, torch.softmax(s * scale, 0) @ v), (cores, cur_pos)
            assert torch.allclose(torch.logaddexp(lse0, lse1), torch.logsumexp(s * scale, 0)), (cores, cur_pos)


def test_sdpa_decode_lse_merge_with_an_empty_half_is_the_other_half():
    """``merge_torch`` with the empty-half sentinel (O = 0 and stats m = -30000, l = 1, read in bf16 through
    ``lse_from_stats`` like the device's) returns the other half unchanged, in either order, at the model's scale 1
    and at 1/sqrt(512); the sentinel's lse is finite (no NaN through a 0 weight). The planner's packed bf16
    patterns are torch's bf16 roundings and are the ones kernels/pair_split.hpp writes."""
    import re
    from pathlib import Path

    import torch

    from models.demos.gemma4.custom_ops.sdpa_decode_lse import SdpaDecodeLse

    torch.manual_seed(2)
    B, Hq, D = 3, 4, 64
    for scale in (1.0, 512**-0.5):
        out_a = torch.randn(1, B, Hq, D)
        lse_a = torch.randn(1, B, Hq) * 10 + 20
        stats = torch.zeros(1, B, 32, lp.STATS_COLS, dtype=torch.bfloat16)
        stats[..., 0], stats[..., lp.TILE] = lp.PAIR_EMPTY_M, lp.PAIR_EMPTY_L
        lse_e = SdpaDecodeLse.lse_from_stats(stats, Hq, scale)
        assert torch.isfinite(lse_e).all() and (lse_e < -1000).all()
        empty = torch.zeros(1, B, Hq, D)
        for out, lse in (
            SdpaDecodeLse.merge_torch(out_a, lse_a, empty, lse_e),
            SdpaDecodeLse.merge_torch(empty, lse_e, out_a, lse_a),
        ):
            assert torch.equal(out, out_a) and torch.equal(lse, lse_a), scale
        w_self = torch.sigmoid(lse_a - lse_e)[..., None]
        assert torch.equal(w_self * out_a + torch.sigmoid(lse_e - lse_a)[..., None] * empty, out_a)

    def bf16_bits(x):
        return int(torch.tensor([x]).to(torch.bfloat16).view(torch.int16).item()) & 0xFFFF

    assert float(torch.tensor(lp.PAIR_EMPTY_M).to(torch.bfloat16)) == -29952.0
    assert lp.PAIR_EMPTY_M_PACKED == bf16_bits(lp.PAIR_EMPTY_M) * 0x10001
    assert lp.PAIR_EMPTY_L_PACKED == bf16_bits(lp.PAIR_EMPTY_L) * 0x10001
    hpp = (Path(lp.__file__).parent / "kernels" / "pair_split.hpp").read_text()
    assert re.search(rf"PAIR_EMPTY_M_PACKED = 0x{lp.PAIR_EMPTY_M_PACKED:08X};", hpp)
    assert re.search(rf"PAIR_EMPTY_L_PACKED = 0x{lp.PAIR_EMPTY_L_PACKED:08X};", hpp)
