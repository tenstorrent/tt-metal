# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of the matmul knob lab (no device): the per-chip shape table of the dense decode matmuls, the
device-free knob enumeration (``KnobSpace``) on a Blackhole 13x10 / 12x10 and a Wormhole 8x8 grid, and the resolver
table (``DecodeMMConfig`` rows: today's configs, the proposed rows, every grid-spec form) evaluated on both grids.
Collected by the whole-directory Gemma CI entries (a few seconds, one process, no device); the device sweep
harness itself (``test_matmul_knob_lab.py``) stays on their ignore lists.
"""

import math

import pytest

import ttnn
from models.demos.gemma4.tests.unit.test_matmul_knob_lab import (
    DECODE_MM_TABLE,
    ENV_PREFIX,
    L1_BUDGET,
    DecodeMMConfig,
    KnobSpace,
    _dest_tiles,
    _l1_1d,
    candidate_grids,
    current_config_rows,
    grid_spec_of,
    op_shapes,
    resolve_1d_knobs,
    resolve_decode_mm_config,
    resolve_grid,
)
from models.demos.gemma4.tt.decode_matmul import decode_in0_block_w
from models.demos.gemma4.tt.dram_sharded import (
    TILE_SIZE,
    _decode_core_grid,
    _estimate_decode_l1_bytes_for_in0,
    _padded_n_tiles,
)


@pytest.mark.parametrize(
    "model, op, tp, expected",
    [
        ("31B", "qkv", 4, [("sliding", 5376, 4096), ("global", 5376, 5120)]),
        ("31B", "qkv", 8, [("sliding", 5376, 2048), ("global", 5376, 3072)]),
        ("31B", "out_proj", 4, [("sliding", 2048, 5376), ("global", 4096, 5376)]),
        ("31B", "gate_up", 8, [("mlp", 5376, 5376)]),
        ("31B", "down", 8, [("mlp", 2688, 5376)]),
        ("12B", "qkv", 4, [("sliding", 3840, 2048), ("global", 3840, 3072)]),
        ("12B", "out_proj", 8, [("sliding", 512, 3840), ("global", 1024, 3840)]),
        ("12B", "gate_up", 4, [("mlp", 3840, 7680)]),
        ("12B", "down", 4, [("mlp", 3840, 3840)]),
        ("31B", "lm_head", 4, [("lm_head", 5376, 65536)]),
        ("12B", "lm_head", 8, [("lm_head", 3840, 32768)]),
    ],
)
def test_op_shapes_host(model, op, tp, expected):
    """The per-chip shard shapes the lab sweeps are the served ones (G6's inventory table, attention/weights.py and
    shared_mlp.py at TP 4 / 8). Host only."""
    got = [(tag, k, n) for tag, k, n, _dtype, _per_step in op_shapes(model, op, tp)]
    assert got == expected
    dtypes = {dtype for _t, _k, _n, dtype, _p in op_shapes(model, op, tp)}
    assert dtypes == {ttnn.bfloat16 if op == "lm_head" else ttnn.bfloat8_b}


@pytest.mark.parametrize("gx, gy", [(13, 10), (12, 10), (8, 8)], ids=["bh13x10", "bh12x10", "wh8x8"])
@pytest.mark.parametrize(
    "k, n, wdtype, act_cores",
    [(5376, 4096, ttnn.bfloat8_b, None), (5376, 10752, ttnn.bfloat8_b, None), (5376, 4096, ttnn.bfloat8_b, 8)],
    ids=["wqkv31b_tp4", "gateup31b_tp4", "wqkv31b_tp4_actws8"],
)
def test_knob_space_host(k, n, wdtype, act_cores, gx, gy):
    """The device-free knob enumeration on a Blackhole and a Wormhole grid: every entry divides K, fits the L1
    estimate and the per-core stream bound, the resolver's block and the DRAM-sharded grid's blocks are in it,
    the grids are the device grid and its divisor sub-grids, and the cross product stays bounded."""
    ds_cores = _decode_core_grid(k, n)[2]
    space = KnobSpace(k, n, wdtype, (gx, gy), ds_cores=ds_cores, act_cores=act_cores)
    kt = k // TILE_SIZE
    ds_blocks = space.ds_blocks()
    assert ds_blocks and all(kt % bw == 0 for bw in ds_blocks)
    assert all(_estimate_decode_l1_bytes_for_in0(_padded_n_tiles(n), bw, wdtype) <= L1_BUDGET for bw in ds_blocks)
    pcn = space.one_d_pcn(gx * gy)
    blocks = space.one_d_blocks(pcn)
    assert all(kt % bw == 0 and _l1_1d(bw, pcn, wdtype) <= L1_BUDGET for bw in blocks)
    if act_cores:
        # a sharded activation: only blocks that divide the 21-tile shard (the resolver's 8 does not)
        assert blocks == [bw for bw in blocks if (kt // act_cores) % bw == 0] and 8 not in blocks and 21 in blocks
    else:
        assert decode_in0_block_w(k) in blocks
    grids = space.one_d_grids(blocks[0], True)
    assert (gx, gy, pcn, 1) in grids
    assert all(math.ceil(space.nt / p) <= vx * vy and osw <= 4 and p % osw == 0 for vx, vy, p, osw in grids)
    assert all(gx % vx == 0 and gy % vy == 0 for vx, vy, _p, _o in grids)  # divisor sub-grids of the device grid
    assert candidate_grids(13, 10) == [(13, 10), (13, 5)]
    assert {(8, 4), (4, 8), (4, 4)} <= set(candidate_grids(8, 8))
    cands = space.enumerate()
    assert 20 <= len(cands) <= 2000
    assert all(c["cores"] >= 1 and space.core_bound_ok(c["cores"]) or c["family"] == "ds" for c in cands)
    # the per-core stream bound prunes a one-core 1D config of a 22 MB weight
    assert not space.core_bound_ok(1)


# Sample rows exercising every grid-spec form on both grids (not a proposal; the sweeps fill DECODE_MM_TABLE).
SAMPLE_TABLE_ROWS = {
    ("31B", "qkv", 4, 32): DecodeMMConfig("1d", "y/2", 24, None, 1, "hifi2", True),
    ("12B", "gate_up", 8, 32): DecodeMMConfig(
        "1d", {"blackhole": (13, 10), "wormhole_b0": (8, 8)}, 12, 2, 2, "hifi2", True
    ),
    ("31B", "down", 4, 1): DecodeMMConfig("1d", "13x10", 8, None, 4, "lofi", False),
}


@pytest.mark.parametrize("gx, gy, arch", [(13, 10, "blackhole"), (8, 8, "wormhole_b0")], ids=["bh13x10", "wh8x8"])
def test_resolver_table_both_grids(gx, gy, arch, monkeypatch):
    """Every row of the resolver table (today's configs, the proposed rows, the sample grid-spec forms) resolves on
    a Blackhole 13x10 and a Wormhole 8x8 grid: 1D rows to a grid that fits, a per-core N that covers N, a subblock
    that divides it within Dest, a K block dividing K and CBs within L1; an explicit grid larger than the device
    clamps to the largest fitting grid (never None); a Blackhole-only family resolves to None on Wormhole and the
    row says so; the env override wins and round-trips the RESULT best config string."""
    rows = dict(current_config_rows())
    rows.update(DECODE_MM_TABLE)
    rows.update(SAMPLE_TABLE_ROWS)
    for (model, op, tp, batch), cfg in rows.items():
        monkeypatch.delenv(ENV_PREFIX + op.upper(), raising=False)
        got = resolve_decode_mm_config(rows, model, op, tp, batch, arch)
        if cfg.arch != "any" and cfg.arch != arch:
            assert got is None and cfg.family == "ds" and cfg.arch == "blackhole"
            continue
        assert got == cfg
        if cfg.family == "ds":
            continue  # its grid is the kernel's own gcd grid (_decode_core_grid), not a device-grid fraction
        for _tag, k, n, wdtype, _per in op_shapes(model, op, tp):
            nt = n // TILE_SIZE
            vx, vy, pcn, osw = resolve_1d_knobs(cfg, gx, gy, nt, arch)
            assert 1 <= vx <= gx and 1 <= vy <= gy, (model, op, tp, cfg)
            assert pcn * vx * vy >= nt and pcn % osw == 0 and osw <= _dest_tiles(cfg.fp32_acc)
            assert (k // TILE_SIZE) % cfg.in0_block_w == 0
            assert _l1_1d(cfg.in0_block_w, pcn, wdtype) <= L1_BUDGET, (model, op, tp, cfg, gx, gy, pcn)
    assert resolve_grid("13x10", gx, gy, arch) == (min(13, gx), min(10, gy))
    assert resolve_grid({"blackhole": (13, 10), "wormhole_b0": (8, 8)}, gx, gy, arch) == (gx, gy)
    assert resolve_grid("x/2+y/5", gx, gy, arch) == (gx // 2, max(1, gy // 5))
    assert grid_spec_of(gx, gy // 2, gx, gy) == "y/2" and grid_spec_of(gx, gy, gx, gy) == "full"
    monkeypatch.setenv(ENV_PREFIX + "QKV", "1d:grid=y/2,bw=24,pc=2,osw=1,fid=hifi2,acc=1,wpb=0,out=dram")
    cfg = resolve_decode_mm_config(rows, "31B", "qkv", 4, 32, arch)
    assert cfg.in0_block_w == 24 and cfg.per_core_N == 2 and resolve_grid(cfg.grid, gx, gy, arch) == (gx, gy // 2)
