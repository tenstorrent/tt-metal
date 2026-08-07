# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The perf harness for the composed read. It measures and it never gates.

No timing assertion appears in this file. A wall-clock threshold on a shared box is a
flake generator, and the numbers here exist to be read into `bringup_log.md` as numbered
perf iterations, not to fail CI. `_report` writes through loguru, so `pytest -s` is what
makes them visible at all.

The op's settled cost model lives in `ROOFLINE.md`. What runs here are the two numbers
the schedule is still chosen on:

  * one read replayed from a captured trace, swept over `S` — the only form whose total
    is device time rather than host dispatch;
  * the split read form against the direct one over a whole 12-layer block.

Both run at the production per-chip shape, 640 rows and `d/4` per device, on the two
meshes that shape occurs on: `(2, 4)` LoudBox and `(8, 4)` Galaxy. Per-device tensors and
the all-reduce's ring size are identical across the pair, so what differs is the fabric
beneath them — 8 concurrent TP rows against 2, on links half as wide.
"""

import time

import pytest
import torch
import ttnn
from loguru import logger

from models.experimental.kimi_k3_attn_res.torch_functional.attn_res import EPS
from models.experimental.kimi_k3_attn_res.tt.attn_res import TtAttnRes

HIDDEN_SIZE = 7168
PRODUCTION_TOKENS = 5120
GALAXY_SP = 8
READ_SITES = 24  # read sites per 12-layer block
PROJ_STD = 0.02

ITERATIONS = 20
WARMUP = 3  # the first read compiles kernels and fills the program cache

# Cross-mesh perf comparisons hold tokens *per chip* fixed, the way the sparse-MLA
# harnesses do: production is `PRODUCTION_TOKENS / sp` on the `(8, 4)` Galaxy, so a mesh
# with a shorter `sp` axis needs a proportionally shorter chunk to profile the same
# per-chip shape. Every token count here is that number scaled by the mesh's `sp_factor`.
PRODUCTION_ROWS = PRODUCTION_TOKENS // GALAXY_SP
ROWS_PER_CHIP = [PRODUCTION_ROWS]

# Enough for the composed read's ~25 programs many times over; the trace region is
# carved out of DRAM at device open, so it is not free.
TRACE_REGION_SIZE = 33554432

# `ttnn.all_reduce` needs an initialized fabric context — without this the op dies on
# `control_plane.cpp:2186` rather than returning wrong numbers. Both benchmarks capture
# traces, so the trace region belongs here too.
FABRIC = {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": TRACE_REGION_SIZE}

# The production shape occurs on exactly these two meshes: TP factor 4 at 640 rows a
# chip. The Galaxy row collects everywhere and runs where there are 32 chips — the
# `mesh_device` fixture skips a request larger than the box.
PLACEMENTS = [
    ((2, 4), FABRIC),  # LoudBox
    ((8, 4), FABRIC),  # Galaxy
]
PLACEMENT_IDS = ["mesh-2x4", "mesh-8x4"]

on_placements = pytest.mark.parametrize(
    "mesh_device, device_params",
    PLACEMENTS,
    indirect=["mesh_device", "device_params"],
    ids=PLACEMENT_IDS,
)


def _make_case(num_tokens, hidden_size, num_sealed, seed=0):
    generator = torch.Generator().manual_seed(seed)
    randn = lambda *shape: torch.randn(*shape, generator=generator)
    return (
        randn(num_tokens, hidden_size),
        randn(num_tokens, num_sealed, hidden_size),
        (1.0 + 0.1 * randn(hidden_size)) * (PROJ_STD * randn(hidden_size)),
    )


def _place(op, prefix_sum, block_residual, query):
    to_tt = lambda t: ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=op.mesh_device, mesh_mapper=op.stream_mapper
    )
    return (
        to_tt(prefix_sum.unsqueeze(0).unsqueeze(0)),
        to_tt(block_residual.permute(1, 0, 2).unsqueeze(0)),
        op.to_query(query),
    )


def _bench(mesh_device, body, iterations=ITERATIONS, warmup=WARMUP):
    """Time `body` twice over: host-side enqueue only, then to device completion.

    Returns microseconds per call for each. The gap between them is the device
    time the host never waited for; no gap means the host was the bottleneck.
    """
    for _ in range(warmup):
        body()
    ttnn.synchronize_device(mesh_device)

    start = time.perf_counter()
    for _ in range(iterations):
        body()
    enqueued = time.perf_counter()
    ttnn.synchronize_device(mesh_device)
    finished = time.perf_counter()

    return (
        (enqueued - start) / iterations * 1e6,
        (finished - start) / iterations * 1e6,
    )


def _report(label, enqueue_us, total_us):
    """One row per configuration, in the form the ledger records.

    The verdict is deliberately weak. `enqueue == total` does **not** mean the
    device was idle: dispatch is pipelined, so it also happens when host and
    device run alongside each other at the same rate. Under a trace, where enqueue
    is ~10 us, the total is device time on its own and the verdict says nothing.
    """
    trailing = total_us - enqueue_us
    verdict = "enqueue-limited" if trailing <= 0.1 * total_us else "device-limited"
    logger.info(
        f"{label:<44} enqueue {enqueue_us:>9.1f} us   total {total_us:>9.1f} us   "
        f"after-enqueue {trailing:>9.1f} us   {verdict}"
    )
    return dict(label=label, enqueue_us=enqueue_us, total_us=total_us, verdict=verdict)


@on_placements
@pytest.mark.parametrize("num_sealed", [1, 4, 8])
def test_perf_read_traced(mesh_device, request, num_sealed):
    """One read replayed from a captured trace, against `S`.

    A trace replaces the per-program host dispatch with one `execute_trace`, so this is
    the only configuration here whose total time is device time and nothing else. A
    forward touches `12(S+1)` planes of `[rows, d/tp]` and captures the same ~22 programs
    at every `S`, so the slope against `S` is DRAM and the intercept is dispatch.

    The output tensor stays live until after `release_trace`. A trace replays into the
    buffer addresses it captured, so freeing the output would let the allocator hand that
    memory to something else and the next replay would overwrite it.
    """
    placement = request.node.callspec.id

    op = TtAttnRes(mesh_device, hidden_size=HIDDEN_SIZE, eps=EPS)
    prefix_sum, block_residual, query = _make_case(PRODUCTION_ROWS * op.sp_factor, HIDDEN_SIZE, num_sealed)
    tt_prefix, tt_block, tt_query = _place(op, prefix_sum, block_residual, query)

    # Compile outside the trace: capture records dispatch commands, and a program
    # that has to be built during capture is not what gets replayed.
    ttnn.deallocate(op.forward(tt_prefix, tt_block, tt_query))
    ttnn.synchronize_device(mesh_device)

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = op.forward(tt_prefix, tt_block, tt_query)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

    def body():
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)

    enqueue_us, total_us = _bench(mesh_device, body)
    _report(f"{placement} S={num_sealed} traced", enqueue_us, total_us)

    ttnn.release_trace(mesh_device, trace_id)
    for tensor in (out, tt_prefix, tt_block, tt_query):
        ttnn.deallocate(tensor)


@on_placements
@pytest.mark.parametrize("num_sealed", [1, 4, 8])
@pytest.mark.parametrize("fused_mix", [True], ids=["fused-mix"])
@pytest.mark.parametrize("rows_per_chip", ROWS_PER_CHIP, ids=lambda rows: f"rows-{rows}")
def test_perf_block_split_vs_direct(mesh_device, num_sealed, fused_mix, rows_per_chip):
    """The two read forms over a whole 12-layer block, traced.

    The split form's premise is that a sealed snapshot is write-once, so the sealed half
    of a read is loop-invariant across a block's 24 read sites. What it pays for that is
    collectives: 26 per block against the direct form's 24, one for the sealed RMS, one
    for the batched sealed dots, one per site inside `merge`.

    Swept over `S` because the split form's saving *is* the sealed half, which is the
    part that grows with it: at `S = 1` there is almost nothing to amortize. The
    schedule's blocks run `S = 0..7`, so a single peak-shape number would flatter the
    form the model actually runs.

    The fused mixture is priced inside this comparison rather than on its own, because
    its share of a block differs between the forms — 24 of the split form's 26 passes
    against 1 of each of the direct form's 24 — so an isolated number answers for the
    wrong schedule.

    Both forms are captured whole, so the comparison is device time for the entire block
    and the per-site number is that divided by `READ_SITES`.
    """
    op = TtAttnRes(mesh_device, hidden_size=HIDDEN_SIZE, eps=EPS, fused_mix=fused_mix)
    prefix_sum, block_residual, query = _make_case(rows_per_chip * op.sp_factor, HIDDEN_SIZE, num_sealed)
    tt_prefix, tt_block, tt_query = _place(op, prefix_sum, block_residual, query)
    queries = [tt_query] * READ_SITES

    def direct():
        return [op.forward(tt_prefix, tt_block, q) for q in queries]

    def split():
        partials, shifts, masses = op.inter_block(tt_block, queries)
        merged = [op.merge(p, s, m, tt_prefix, tt_query) for p, s, m in zip(partials, shifts, masses)]
        for group in (partials, shifts, masses):
            for tensor in group:
                ttnn.deallocate(tensor)
        return merged

    for label, form in (("direct", direct), ("split", split)):
        for tensor in form():
            ttnn.deallocate(tensor)
        ttnn.synchronize_device(mesh_device)

        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        outputs = form()
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

        enqueue_us, total_us = _bench(
            mesh_device,
            lambda: ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False),
            iterations=5,
            warmup=2,
        )
        mix_label = "fused mix" if fused_mix else "composed mix"
        _report(
            f"{rows_per_chip} rows/chip, S={num_sealed}, {label}, {mix_label}, traced",
            enqueue_us,
            total_us / READ_SITES,
        )

        ttnn.release_trace(mesh_device, trace_id)
        for tensor in outputs:
            ttnn.deallocate(tensor)

    for tensor in (tt_prefix, tt_block, tt_query):
        ttnn.deallocate(tensor)
