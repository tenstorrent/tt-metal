# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Phase 9: the perf harness. It measures and it never gates.

No timing assertion appears in this file. A wall-clock threshold on a shared
LoudBox is a flake generator, and the numbers here exist to be read into
`bringup_log.md` as numbered perf iterations, not to fail CI.

The one question this harness answers first, because `ROOFLINE.md` §6 says the
two regimes are 100x apart: does host launch overhead or DRAM bandwidth bind the
composed op? The test is enqueue-versus-sync. `body()` only enqueues — no
readback, no synchronize — so if the host finishes enqueueing R reads long
before the device finishes them, the device is the bottleneck; if the two times
are the same, the device was idle waiting on Python.

Three placements make that a controlled comparison rather than a bare number:

  * `(1, 1)`, `T = 5120` — the anchor. One device holds the whole 7.9 GB of
    per-read traffic against ~2 ms of launches, so it *must* read device-bound.
    If it does not, the method is wrong and the other two rows mean nothing.
  * `(8, 1)`, `T = 5120` — 8 devices, sequence-parallel only. 640 tokens and the
    full `d` per device.
  * `(2, 4)`, `T = 5120` — 8 devices, 2560 tokens and `d/4` per device.

The last two carry *identical* per-device DRAM traffic (991 MB, `ROOFLINE.md`
§3) and differ only in the collective: 15 ttnn calls and no fabric traffic
against 22 calls and one all-reduce. Their difference is the price of TP.
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
NUM_SEALED = 8
READ_SITES = 24  # read sites per 12-layer block
PROJ_STD = 0.02

ITERATIONS = 20
WARMUP = 3  # the first read compiles kernels and fills the program cache

# Enough for the composed read's ~25 programs many times over; the trace region is
# carved out of DRAM at device open, so it is not free.
TRACE_REGION_SIZE = 33554432

FABRIC = {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": TRACE_REGION_SIZE}
LOCAL = {"trace_region_size": TRACE_REGION_SIZE}

# `(4, 2)` is the third valid LoudBox mesh and adds nothing here: it is `(2, 4)`
# with the two axes swapped, same per-device bytes, TP factor 2 instead of 4.
PLACEMENTS = [
    ((1, 1), LOCAL),
    ((8, 1), FABRIC),
    ((2, 4), FABRIC),
]
PLACEMENT_IDS = ["mesh-1x1", "mesh-8x1", "mesh-2x4"]

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
    device run alongside each other at the same rate. Only the traced rows,
    where enqueue is ~10 us, report device time on its own.
    """
    trailing = total_us - enqueue_us
    verdict = "enqueue-limited" if trailing <= 0.1 * total_us else "device-limited"
    logger.info(
        f"{label:<44} enqueue {enqueue_us:>9.1f} us   total {total_us:>9.1f} us   "
        f"after-enqueue {trailing:>9.1f} us   {verdict}"
    )
    return dict(label=label, enqueue_us=enqueue_us, total_us=total_us, verdict=verdict)


@on_placements
def test_perf_read_launch_term(mesh_device, request):
    """P1: enqueue versus completion for one read at production shape.

    The op's `num_links` default is 1; `ROOFLINE.md` §4 models 2 as half the
    fabric time, and the sweep below is what decides whether that is visible at
    all against the per-device DRAM floor.
    """
    placement = request.node.callspec.id
    prefix_sum, block_residual, query = _make_case(PRODUCTION_TOKENS, HIDDEN_SIZE, NUM_SEALED)

    for num_links in (1, 2) if tuple(mesh_device.shape)[1] > 1 else (1,):
        op = TtAttnRes(mesh_device, hidden_size=HIDDEN_SIZE, eps=EPS, num_links=num_links)
        tt_prefix, tt_block, tt_query = _place(op, prefix_sum, block_residual, query)

        def body():
            ttnn.deallocate(op.forward(tt_prefix, tt_block, tt_query))

        enqueue_us, total_us = _bench(mesh_device, body)
        _report(f"{placement} S={NUM_SEALED} links={num_links}", enqueue_us, total_us)

        for tensor in (tt_prefix, tt_block, tt_query):
            ttnn.deallocate(tensor)


@on_placements
@pytest.mark.parametrize("num_sealed", [1, 4, 8])
def test_perf_read_traced(mesh_device, request, num_sealed):
    """P2: the same read replayed from a captured trace.

    A trace replaces the per-program host dispatch with one `execute_trace`, so
    this is the only configuration here whose total time is device time and
    nothing else — and therefore the only one that can say whether the host
    enqueue P1 measured was *in the way* or merely running alongside.

    The output tensor stays live until after `release_trace`. A trace replays
    into the buffer addresses it captured, so freeing the output would let the
    allocator hand that memory to something else and the next replay would
    overwrite it.
    """
    placement = request.node.callspec.id
    prefix_sum, block_residual, query = _make_case(PRODUCTION_TOKENS, HIDDEN_SIZE, num_sealed)

    op = TtAttnRes(mesh_device, hidden_size=HIDDEN_SIZE, eps=EPS)
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


@pytest.mark.parametrize(
    "mesh_device, device_params", [((2, 4), FABRIC)], indirect=["mesh_device", "device_params"], ids=["mesh-2x4"]
)
@pytest.mark.parametrize("num_links", [1, 2], ids=["links-1", "links-2"])
@pytest.mark.parametrize(
    "planes, width",
    [(18, 1), (1, 18), (18, 32), (2, 1)],
    ids=["planes-18x1", "folded-1x18", "full-tiles-18x32", "minimal-2x1"],
)
def test_perf_collective_by_payload(mesh_device, planes, width, num_links):
    """P4: does the statistics all-reduce cost bytes, or does it cost a program?

    The read stacks `2(S+1)` statistics planes on dim 1 with a 1-wide last dim,
    which tile-pads 1 -> 32. `DISTRIBUTION.md` §4 dismissed that 32x on the
    grounds that it is 0.6% of the op's bytes; the profiler puts the collective
    at ~19% of device time, so bytes were the wrong denominator.

    `planes-18x1` is what the op sends today. `folded-1x18` is the same
    statistics with the candidate axis folded into the last dim — 18x fewer
    bytes, 18 of 32 columns useful, two permutes to build. If device time tracks
    the payload the fold is worth writing; if it is flat, the collective is
    paying per-program latency and shrinking it changes nothing.

    Traced, so the number is device time and not Python.
    """
    tokens_per_rank = PRODUCTION_TOKENS // 2
    stats = ttnn.from_torch(
        torch.ones(1, planes, tokens_per_rank, width),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
    )
    reduce = lambda: ttnn.all_reduce(stats, cluster_axis=1, num_links=num_links, topology=ttnn.Topology.Linear)

    ttnn.deallocate(reduce())
    ttnn.synchronize_device(mesh_device)

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = reduce()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

    enqueue_us, total_us = _bench(
        mesh_device, lambda: ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    )
    padded_kib = planes * tokens_per_rank * max(width, ttnn.TILE_SIZE) * 4 / 1024
    _report(
        f"all_reduce [1,{planes},{tokens_per_rank},{width}] {padded_kib:.0f} KiB links={num_links}",
        enqueue_us,
        total_us,
    )

    ttnn.release_trace(mesh_device, trace_id)
    for tensor in (out, stats):
        ttnn.deallocate(tensor)


@on_placements
@pytest.mark.parametrize("num_sealed", [1, 4, 8])
def test_perf_read_by_candidate_count(mesh_device, request, num_sealed):
    """P3: how the read scales with `S`, the axis the schedule actually varies.

    A forward touches `12(S+1)` planes of `[T/sp, d/tp]` and enqueues the same
    ~22 programs at every `S`, so DRAM time scales with `S+1` and the launch term
    does not. The slope against `S` separates them without a profiler.
    """
    placement = request.node.callspec.id
    prefix_sum, block_residual, query = _make_case(PRODUCTION_TOKENS, HIDDEN_SIZE, num_sealed)

    op = TtAttnRes(mesh_device, hidden_size=HIDDEN_SIZE, eps=EPS)
    tt_prefix, tt_block, tt_query = _place(op, prefix_sum, block_residual, query)

    def body():
        ttnn.deallocate(op.forward(tt_prefix, tt_block, tt_query))

    enqueue_us, total_us = _bench(mesh_device, body)
    _report(f"{placement} S={num_sealed}", enqueue_us, total_us)

    for tensor in (tt_prefix, tt_block, tt_query):
        ttnn.deallocate(tensor)


@pytest.mark.parametrize(
    "mesh_device, device_params", [((2, 4), FABRIC)], indirect=["mesh_device", "device_params"], ids=["mesh-2x4"]
)
def test_perf_block_split_vs_direct(mesh_device):
    """P5: the two read forms over a whole 12-layer block, traced.

    Phase 7 measured the split form 1.50x faster on one device by wall clock with
    no profiler. On a TP mesh it issues 49 collectives per block against the
    direct form's 24 — one for the sealed RMS, one per site for the sealed dots,
    one per site inside `merge` — so the question is whether amortizing the
    sealed half survives paying twice the collectives.

    Both forms are captured whole, so the comparison is device time for the
    entire block and the per-site number is that divided by 24.
    """
    prefix_sum, block_residual, query = _make_case(PRODUCTION_TOKENS, HIDDEN_SIZE, NUM_SEALED)
    op = TtAttnRes(mesh_device, hidden_size=HIDDEN_SIZE, eps=EPS)
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
        _report(f"block of {READ_SITES} sites, {label}, traced", enqueue_us, total_us / READ_SITES)

        ttnn.release_trace(mesh_device, trace_id)
        for tensor in outputs:
            ttnn.deallocate(tensor)

    for tensor in (tt_prefix, tt_block, tt_query):
        ttnn.deallocate(tensor)


@pytest.mark.parametrize(
    "mesh_device, device_params", [((2, 4), FABRIC)], indirect=["mesh_device", "device_params"], ids=["mesh-2x4"]
)
@pytest.mark.parametrize("num_sealed", [1, 8])
@pytest.mark.parametrize("num_links", [1, 2], ids=["links-1", "links-2"])
@pytest.mark.parametrize("fold_stats", [False, True], ids=["unfolded", "folded"])
def test_perf_read_by_stats_layout(mesh_device, num_sealed, num_links, fold_stats):
    """P6: the fold P4 priced, in the op rather than in isolation, traced.

    P4 measured a standalone `all_reduce` 7.4x faster on the folded layout and
    `num_links = 2` worth 1.48x only at the padded payload — which predicts that
    the two are alternatives and that folding makes the second link pointless.
    This sweeps both together against a whole read, where the fold also pays for
    two `ttnn.permute` calls that P4 never charged it for.

    `S = 1` is here because the schedule's mean is 5.39, not 8: the folded payload
    is constant in `S` while the padded one is not, so the fold should widen its
    lead as `S` falls even though the absolute saving shrinks.
    """
    prefix_sum, block_residual, query = _make_case(PRODUCTION_TOKENS, HIDDEN_SIZE, num_sealed)

    op = TtAttnRes(mesh_device, hidden_size=HIDDEN_SIZE, eps=EPS, num_links=num_links, fold_stats=fold_stats)
    tt_prefix, tt_block, tt_query = _place(op, prefix_sum, block_residual, query)

    ttnn.deallocate(op.forward(tt_prefix, tt_block, tt_query))
    ttnn.synchronize_device(mesh_device)

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = op.forward(tt_prefix, tt_block, tt_query)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

    enqueue_us, total_us = _bench(
        mesh_device, lambda: ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    )
    layout = "folded" if fold_stats else "unfolded"
    _report(f"S={num_sealed} links={num_links} {layout} traced", enqueue_us, total_us)

    ttnn.release_trace(mesh_device, trace_id)
    for tensor in (out, tt_prefix, tt_block, tt_query):
        ttnn.deallocate(tensor)
