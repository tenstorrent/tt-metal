# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""The gate/up matmul runs over REAL tile-rows (`m_tiles_real`), not the padded `m_tiles_eff`.

    scripts/run_safe_pytest.sh --run-all <this file>
    scripts/run_safe_pytest.sh --profile --run-all <this file>   # also reports the plateau table

`m_tiles_eff` rounds a tail M-block UP to a power of two, so M 160 (5 tile-rows) is carried in an
8-row block. The round-up is mandatory for the PAGE count: a CB push that does not divide the buffer
never wraps — the wrap test is `== fifo_limit`, so an overshooting push walks into the NEXT CB — and
the reduce-scatter's slice plan must be uniform and agree on every core with no communication.

It is NOT mandatory for the arithmetic. The gate/up output block is m-MAJOR, so the real rows are a
contiguous PREFIX and the matmul simply does `m_real` sub-blocks, leaving the pad rows stale. Rows
[count, m_eff*32) are undefined tile padding by contract — the op used to fill them with
silu(pad @ Wg) * (pad @ Wu) @ Wd, undefined in exactly the same way.

WHY ONLY GATE/UP, and why this is not a free choice. `matmul_block` derives its in0 CB accounting
from the block shape (`in0_block_num_tiles = in0_subblock_num_tiles * shape.in0_num_subblocks`,
matmul_block_helpers.inl:196), so shrinking the shape shrinks the pops too. gate/up survives that
because it runs with num_k_blocks == 1: every call is the last K-block, `WaitAndRetainOnLastBlock`
never pops, and the compute kernel pops x itself at `m_eff * KR_PAD`. So the shrink only shrinks a
`wait_front`, and waiting for a prefix of what the reader pushed is always satisfied. `down` reads h
under `WaitAndPopPerKBlock`, where the shrink DID shrink a `cb_pop_front` — cb_h_local drifted by
(m_eff - m_rows) * HN_PAD tiles per K-block and the op hung. `down` therefore stays on m_eff.

WHAT THIS FILE PINS. Every reachable `m_t` shape, because the shrink is a per-tail-block decision and
the interesting values are exactly the ones no other test in this directory uses: m_t 3, 5, 6 and 7
inside one block, and the same remainders as the SECOND block of a multi-block dispatch. Bit-identity
of the DEFINED REGION against the pre-change kernel is `test_moe_fused_swiglu_bitwise_gate.py`'s job
(its padded-tail cases were added for this change); what is checked here is that every shape is
numerically right and stable, including the ones where m_real and m_eff differ most.
"""

import statistics

import pytest
import torch

import ttnn

from ttnn.operations.moe_fused_swiglu import moe_fused_swiglu
from ttnn.operations.moe_fused_swiglu.moe_fused_swiglu_program_descriptor import (
    nd_shard_n_tiles,
    weight_memory_configs,
)

from .test_moe_fused_swiglu_determinism import defined_region, written_rows

TILE = 32
M_BLOCK = 8  # mirrors geo.M_BLOCK; asserted against it below
EMB, HIDDEN, CAPACITY = 7168, 2048, 5120
NUM_GLOBAL_EXPERTS, NUM_LOCAL_EXPERTS, LOCAL_EXPERT_ID, GLOBAL_EXPERT_ID = 256, 8, 3, 137
GRID = (11, 8)
PCC_GATE = 0.975  # the bfp4 format floor, same gate the golden suite uses


def m_eff_of(m_t, b):
    """`m_tiles_eff` in Python — the PAGE row count of block b."""
    rem = max(0, m_t - b * M_BLOCK)
    if rem >= M_BLOCK:
        return M_BLOCK
    p = 1
    while p < rem:
        p <<= 1
    return min(p, M_BLOCK)


def blocks_of(m_t):
    """[(m_real, m_eff)] per M-block, i.e. exactly what the kernel walks."""
    n = (m_t + M_BLOCK - 1) // M_BLOCK
    return [(min(m_t - b * M_BLOCK, M_BLOCK), m_eff_of(m_t, b)) for b in range(n)]


#: Every m_t in [1, 8] so each single-block remainder is covered, then the same remainders as the
#: TAIL of a multi-block dispatch (the shrink is per block, and only the last block can shrink), then
#: a ragged count and count == capacity. 150 is the case where `count` is neither tile-aligned NOR a
#: power of two of tile-rows: written extent 160, m_real 5, m_eff 8.
COUNTS = [32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 480, 512, 150, 5120]

#: Consecutive M for the plateau table — before this change 160/192/224 all cost the same as 256.
PLATEAU = [128, 160, 192, 224, 256]


def _pcc(a, b):
    a, b = a.flatten().to(torch.float32), b.flatten().to(torch.float32)
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def _reference(x, wg, wu, wd, count):
    xs = x[0, 0, :count].to(torch.float32)
    h = torch.nn.functional.silu(xs @ wg.to(torch.float32)) * (xs @ wu.to(torch.float32))
    return h @ wd.to(torch.float32)


def _weight_configs(device, wplace):
    """`nd_shard` is the op's DESIGNED placement and the one every graded number is quoted at.

    Placement is the CALLER's choice, not a knob: the op reads whatever it is handed and takes the
    coalesced path only when `nd_shard_n_tiles` can prove a contiguous run. An interleaved weight is
    silently CORRECT and merely takes the uncoalesced one-request-per-tile stream, which is exactly
    why the arm has to be verified rather than assumed — see `_assert_placement`.
    """
    if wplace == "nd_shard":
        return weight_memory_configs(device, EMB, HIDDEN, core_grid=GRID)
    if wplace == "interleaved":
        return ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
    raise ValueError(f"unknown wplace {wplace!r}")


def _assert_placement(tt_w, wplace):
    """Assert the READER's own predicate, in BOTH directions — a placement that failed to apply
    would otherwise be reported as a legitimate number for the wrong path."""
    widths = [nd_shard_n_tiles(w) for w in tt_w]
    if wplace == "nd_shard":
        assert all(w > 0 for w in widths), f"asked for nd_shard but the reader sees interleaved: {widths}"
    else:
        assert all(w == 0 for w in widths), f"asked for interleaved but the reader sees shards: {widths}"
    return widths


def _build(device, count, wplace="interleaved"):
    torch.manual_seed(42)
    x = torch.randn((1, 1, CAPACITY, EMB), dtype=torch.float32)
    # HOSTILE pad. The whole premise is that the rows the matmul stops computing are undefined, so a
    # leak from them into a real row has to be visible rather than plausible.
    if count < CAPACITY:
        x[:, :, count:, :] = 100.0
    xb = x.to(torch.bfloat16)
    wg, wu = (torch.randn((EMB, HIDDEN), dtype=torch.bfloat16) for _ in range(2))
    wd = torch.randn((HIDDEN, EMB), dtype=torch.bfloat16)
    d = lambda t, dt, l: ttnn.from_torch(t, dtype=dt, layout=l, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    gate_up_mc, down_mc = _weight_configs(device, wplace)
    tt_w = [
        ttnn.from_torch(w, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
        for w, mc in ((wg, gate_up_mc), (wu, gate_up_mc), (wd, down_mc))
    ]
    _assert_placement(tt_w, wplace)
    counts = torch.zeros(NUM_GLOBAL_EXPERTS, dtype=torch.int32)
    counts[GLOBAL_EXPERT_ID] = count
    idx = torch.tensor([(11 + 37 * i) % NUM_GLOBAL_EXPERTS for i in range(NUM_LOCAL_EXPERTS)], dtype=torch.int32)
    idx[LOCAL_EXPERT_ID] = GLOBAL_EXPERT_ID
    return (
        (xb, wg, wu, wd),
        d(xb, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
        tt_w,
        d(counts, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
        d(idx, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
    )


def _run(args, want_rows=None):
    _, tt_x, tt_w, tt_counts, tt_idx = args
    out = moe_fused_swiglu(tt_x, tt_w[0], tt_w[1], tt_w[2], tt_counts, tt_idx, LOCAL_EXPERT_ID, core_grid=GRID)
    assert list(out.shape) == [1, 1, CAPACITY, EMB]
    got = None if want_rows is None else ttnn.to_torch(defined_region(out, want_rows)).float().clone()
    ttnn.deallocate(out)
    return got


def test_m_block_constant_matches_the_op():
    """This file's shape arithmetic mirrors the kernel's; a divergence would silently mis-label cases."""
    from ttnn.operations.moe_fused_swiglu import moe_fused_swiglu_geometry as geo

    assert M_BLOCK == geo.M_BLOCK, f"M_BLOCK drifted: this file says {M_BLOCK}, the op says {geo.M_BLOCK}"
    assert geo.OUT_SUBBLOCK_H_GU == 1, (
        f"OUT_SUBBLOCK_H_GU is {geo.OUT_SUBBLOCK_H_GU}, not 1. The gate/up shrink then needs "
        f"`round_up_capped` to keep `m_rows / height` exact — it is already applied in the kernel, "
        f"but this file's PLATEAU expectations assume height 1."
    )


@pytest.mark.parametrize("count", COUNTS)
def test_every_tail_shape_is_correct(device, count):
    """Correct and STABLE at every reachable m_t, especially where m_real < m_eff."""
    m_t = -(-count // TILE)
    blocks = blocks_of(m_t)
    rows = written_rows(count)
    args = _build(device, count)

    got = _run(args, rows)
    again = _run(args, rows)

    # Reproducibility FIRST, over the whole defined extent rather than [:count]: a shrink that
    # desynchronised a CB shows up as instability before it shows up as a wrong number, and the pad
    # rows *inside* the last written tile are part of what the op must produce deterministically.
    assert torch.equal(got, again), (
        f"count={count} (m_t={m_t}, blocks={blocks}): not reproducible over the defined "
        f"[0, {rows}) — max|delta| {(got - again).abs().max().item()}"
    )
    assert torch.isfinite(got).all(), f"count={count}: non-finite value inside the defined region"

    ref = _reference(*args[0], count)
    pcc = _pcc(got[0, 0, :count], ref)
    assert pcc >= PCC_GATE, f"count={count} (m_t={m_t}, blocks={blocks}): pcc {pcc:.6f} < {PCC_GATE}"
    padded = [f"{r}/{e}" for r, e in blocks if r != e] or ["none"]
    print(f"[m_real] count={count:>5} m_t={m_t:>3} blocks={blocks} padded={','.join(padded)} pcc={pcc:.6f}")


#: Repetitions per point of the plateau table, plus one warmup that carries the shape's JIT build.
PLATEAU_REPS = 5

#: Both weight placements. `nd_shard` FIRST because it is the op's designed placement and the one
#: every graded number is quoted at; `interleaved` is kept because its fixed weight cost is LARGER,
#: which dilutes this change — the same row saving is a smaller share of a bigger total.
PLATEAU_WPLACES = ["nd_shard", "interleaved"]


@pytest.mark.parametrize("wplace", PLATEAU_WPLACES)
def test_no_m_eff_plateau(device, wplace):
    """REPORT the M-cost curve where the round-up used to flatten it.

    Before this change every M in (128, 256] computed 8 tile-rows, so 160, 192 and 224 all cost what
    256 cost. Measured, never asserted: correctness is this file's only pass/fail, and a
    DEVICE KERNEL DURATION is not visible from Python at all. Run under `--profile` and read the
    per-dispatch ns out of the CSV — the dispatch order is wplace-MAJOR and M-minor with one warmup
    per point, so the run is parseable without a manifest.

    The host wall-clock printed below is a LIVENESS signal, not a measurement: it is ~8.5 ms per
    dispatch here, pure host dispatch overhead, near two orders of magnitude above the kernel. It
    must not be quoted as if it said anything about the kernel.
    """
    import time

    out = {}
    for count in PLATEAU:
        args = _build(device, count, wplace)
        _run(args)  # warmup: first dispatch of a shape carries the JIT build
        ts = []
        for _ in range(PLATEAU_REPS):
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            _run(args)
            ttnn.synchronize_device(device)
            ts.append(time.perf_counter() - t0)
        ttnn.ReadDeviceProfiler(device)  # a no-op when the profiler is off
        out[count] = statistics.median(ts) * 1e6

    base = out[PLATEAU[0]]
    print(f"\n[m_real] {wplace}: host wall-clock per dispatch (dispatch-bound; the CSV is the number)")
    print(f"{'M':>6} {'m_t':>4} {'m_real/m_eff':>13} {'host us':>9} {'vs M=128':>9}")
    for count in PLATEAU:
        m_t = -(-count // TILE)
        r, e = blocks_of(m_t)[-1]
        print(f"{count:>6} {m_t:>4} {f'{r}/{e}':>13} {out[count]:>9.1f} {out[count] / base:>9.3f}")
