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
    assert plan.fits and plan.chunk == 256 and plan.unit == 4096 and plan.nsub == 16
    assert plan.max_units == 64 and plan.n_slots == 64 and plan.modulus == min(plan.n_cores, 64)
    # the same (b, g, u) always goes to the same core, whatever the positions are
    cores = {(b, 0, u): plan.core_of(b, 0, u) for b in range(batch) for u in range(plan.max_units)}
    assert all(0 <= c < plan.n_cores for c in cores.values())
    full = [CTX_256K - 1] * batch
    seen = []
    for c in range(plan.n_cores):
        units = plan.units_of_core(c, full)
        assert all(cores[(b, g, u)] == c for b, g, u in units)
        assert units == sorted(units, key=lambda t: (t[2] % plan.modulus) * batch + t[0] * plan.groups + t[1])
        seen += units
    assert sorted(seen) == [(b, 0, u) for b in range(batch) for u in range(plan.max_units)]
    counts = [len(plan.units_of_core(c, full)) for c in range(plan.n_cores)]
    assert max(counts) - min(counts) <= 1 and sum(counts) == batch * plan.max_units
    # the mapping is the same under a different cur_pos: only the live subset changes
    short = [4095] * batch
    assert all(plan.core_of(b, 0, 0) == cores[(b, 0, 0)] for b in range(batch))
    assert sum(len(plan.units_of_core(c, short)) for c in range(plan.n_cores)) == batch
    # padding rows have no units
    pad = [-1] * batch
    assert all(plan.units_of_core(c, pad) == [] for c in range(plan.n_cores))


@pytest.mark.parametrize("grid", GRIDS, ids=["wh_8x8", "bh_13x10"])
def test_splitfree_lone_row_spreads_over_the_grid(grid):
    plan = _plan(grid, 1)
    busy = [c for c in range(plan.n_cores) if plan.units_of_core(c, [CTX_256K - 1])]
    assert len(busy) == min(plan.n_cores, 64)  # 64 units of 4096 tokens
    # at 1M tokens every core of the grid carries units
    big = _plan(grid, 1, ctx=1 << 20)
    assert big.max_units == 256 and big.modulus == min(big.n_cores, 256)
    counts = [len(big.units_of_core(c, [(1 << 20) - 1])) for c in range(big.n_cores)]
    assert min(counts) >= 1 and max(counts) - min(counts) <= 1


def test_splitfree_chunks_and_masks_follow_cur_pos_like_the_stock():
    plan = _plan((13, 10), 1)
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
    drops to 128 on Wormhole; the env caps it; the unit stays 4096."""
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
    global layers = 144 MiB whatever the local head count; the sliding layers (head_dim 256, 2 local KV groups)
    need 12.5 MiB, so one model scratch takes the global figure."""
    p31 = _plan((13, 10), 32, groups=1, hpg=4)  # 31B TP8 global: 4 local q heads, one replicated KV head
    p12 = _plan((13, 10), 32, groups=1, hpg=2)  # 12B TP8 global: 2 local q heads
    assert p31.scratch_tiles == p12.scratch_tiles == 32 * 64 * 18 == 36864
    assert p31.scratch_bytes == 36864 * 4096 == 150_994_944
    sliding = _plan((13, 10), 32, groups=2, hpg=2, head_dim=256, window=1024)
    assert sliding.scratch_tiles == 32 * 2 * 5 * 10 == 3200 and sliding.scratch_bytes == 13_107_200
    specs = [(1, 4, 512, None), (2, 2, 256, 1024)]
    assert sp.scratch_tiles_for((13, 10), 32, specs, CTX_256K, kv_dtype="BFLOAT8_B", env={}) == 36864
    assert sp.scratch_tiles_for((8, 8), 32, specs, CTX_256K, kv_dtype="BFLOAT16", env={}) == 36864
    assert sp.scratch_tiles_for((13, 10), 1, specs, CTX_256K, env={}) == 64 * 18
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
