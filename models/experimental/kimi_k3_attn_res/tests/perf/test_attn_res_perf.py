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
@pytest.mark.parametrize("num_sealed", [1, 4, 8])
@pytest.mark.parametrize("fused_mix", [False, True], ids=["composed-mix", "fused-mix"])
def test_perf_block_split_vs_direct(mesh_device, num_sealed, fused_mix):
    """P5: the two read forms over a whole 12-layer block, traced.

    Phase 7 measured the split form 1.50x faster on one device by wall clock with
    no profiler. On a TP mesh the question is whether amortizing the sealed half
    survives the extra collectives — 26 per block against the direct form's 24,
    one for the sealed RMS, one for the batched sealed dots, one per site inside
    `merge`.

    Swept over `S` because the split form's saving is the sealed half, which is
    the part that grows with it: at `S = 1` there is almost nothing to amortize.
    The schedule's blocks run `S = 0..7`, so a single peak-shape number would
    flatter the form the model actually runs.

    Both forms are captured whole, so the comparison is device time for the
    entire block and the per-site number is that divided by 24.

    P10 added the `fused_mix` axis, which is where the whole-op price of the
    fused mixture is read rather than extrapolated from P9's isolated 79 MiB
    row. It belongs on this test and not on a new one: the mixture's share of a
    block differs between the two read forms — 24 of the split form's 26 passes
    against 1 of the direct form's per-site passes — so a single-form
    measurement would answer for the wrong schedule.
    """
    prefix_sum, block_residual, query = _make_case(PRODUCTION_TOKENS, HIDDEN_SIZE, num_sealed)
    op = TtAttnRes(mesh_device, hidden_size=HIDDEN_SIZE, eps=EPS, fused_mix=fused_mix)
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
            f"block of {READ_SITES} sites, S={num_sealed}, {label}, {mix_label}, traced",
            enqueue_us,
            total_us / READ_SITES,
        )

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


HIFI = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
)


def _reduce_variants(v, q, weights, weights_fp32):
    """The ways to get one scalar per (token, candidate) out of `v`.

    Each entry reads `[1, C, N, d/tp]` and returns `[1, C, N, *]`. `mul-sum` is
    what the op does today: an elementwise pass that writes a second full-size
    tensor to DRAM, then a reduce that reads it back. Three passes over the
    largest tensor in the op to produce 0.6% of its bytes.

    The `hifi` rows are the only *admissible* one-pass forms. At default fidelity
    both single-op replacements lose an order of magnitude of accuracy — the
    RMSNorm statistics kernel goes to 4.8e-2 against today's 2.4e-3, and the
    matvec to 1.3e-2 against 3.2e-3 — because LoFi truncates the bf16 mantissa
    going into the multiply. HiFi4 with fp32 dest accumulation fixes both
    (7.4e-4 and 3.2e-3), so it is part of the candidate, not a tuning knob, and
    the timing below has to be read off those rows.
    """
    q_col = lambda: ttnn.permute(q, [0, 1, 3, 2])
    return {
        "floor: sum(v)": lambda: ttnn.sum(v, dim=3, keepdim=True),
        "sumsq: mul+sum (today)": lambda: _consume(ttnn.mul(v, v), lambda s: ttnn.sum(s, dim=3, keepdim=True)),
        "sumsq: rms_norm_pre_ag": lambda: ttnn.rms_norm_pre_all_gather(v, dtype=ttnn.bfloat16),
        "sumsq: rms_norm_pre_ag hifi": lambda: ttnn.rms_norm_pre_all_gather(
            v, dtype=ttnn.float32, compute_kernel_config=HIFI
        ),
        "dots: mul+sum (today)": lambda: _consume(ttnn.mul(v, q), lambda p: ttnn.sum(p, dim=3, keepdim=True)),
        "dots: matmul": lambda: ttnn.matmul(v, q_col()),
        "dots: matmul hifi": lambda: ttnn.matmul(v, q_col(), compute_kernel_config=HIFI),
        # `_mix` has the same three-pass shape but reduces over candidates, so its
        # output is full `d` wide. Its floor is one read of `v` plus a `d`-wide write,
        # and there is no fused elementwise-then-reduce-over-a-batch-dim op to reach
        # it — the gap prices a `[1, N, C, d/tp]` layout, where the mix is a batched
        # matmul, against the cost of transposing to get there.
        "mix: mul+sum (today)": lambda: _consume(ttnn.mul(v, q), lambda w: ttnn.sum(w, dim=1, keepdim=True)),
        "mix floor: sum(v, dim=1)": lambda: ttnn.sum(v, dim=1, keepdim=True),
        # The two rows above multiply by `q`, a `[1, 1, 1, d/tp]` broadcast, which
        # reads the same bytes but is not the broadcast the op performs. These four
        # use the real `[1, C, N, 1]` weight, so the pair is comparable to `_mix`
        # itself and not only to each other.
        #
        # `fast_reduce_nc` is not the fused op the comment above says does not exist
        # — it is reduce-only, so the elementwise pass stays. It is a *different
        # dim-1 reduce kernel*, which is a question the P7 table never asked. Its
        # numerics land one bf16 ulp from `ttnn.sum` (`scratchpad/probe_fast_reduce_nc.py`).
        "mix shaped: mul+sum": lambda: _consume(ttnn.mul(v, weights), lambda w: ttnn.sum(w, dim=1, keepdim=True)),
        "mix shaped: mul+fast_reduce_nc": lambda: _consume(
            ttnn.mul(v, weights), lambda w: ttnn.experimental.fast_reduce_nc(w, dims=[1])
        ),
        "mix floor: fast_reduce_nc(v, dim=1)": lambda: ttnn.experimental.fast_reduce_nc(v, dims=[1]),
        # Not a fidelity variant: `fast_reduce_nc` already defaults to HiFi4
        # (`fast_reduce_nc.cpp:31`), so what HIFI adds over the bare call above is
        # `fp32_dest_acc_en` and `packer_l1_acc`. Named for what it varies.
        "mix floor: fast_reduce_nc fp32acc": lambda: ttnn.experimental.fast_reduce_nc(
            v, dims=[1], compute_kernel_config=HIFI
        ),
        # P10. The fused op the comment above said did not exist, because it did
        # not: one pass over `v`, the weight MAC'd into the accumulator, no
        # intermediate and no transpose. Read against `mix shaped: mul+sum` for
        # what the fusion is worth and against the floor rows for what is left.
        # The fp32 row is the one the op actually calls — its score chain runs in
        # fp32 — and it prices a 4 KiB weight tile against a 2 KiB one on traffic
        # that is 3% of the read.
        "mix fused: weighted_reduce_nc": lambda: ttnn.experimental.fast_weighted_reduce_nc(v, weights, dim=1),
        "mix fused: weighted_reduce_nc fp32 w": lambda: ttnn.experimental.fast_weighted_reduce_nc(
            v, weights_fp32, dim=1
        ),
    }


def _consume(intermediate, then):
    """Run `then` on a full-size intermediate and free it, as the op does."""
    result = then(intermediate)
    ttnn.deallocate(intermediate)
    return result


@pytest.mark.parametrize(
    "mesh_device, device_params", [((2, 4), FABRIC)], indirect=["mesh_device", "device_params"], ids=["mesh-2x4"]
)
@pytest.mark.parametrize(
    "variant",
    [
        "floor: sum(v)",
        "sumsq: mul+sum (today)",
        "sumsq: rms_norm_pre_ag",
        "sumsq: rms_norm_pre_ag hifi",
        "dots: mul+sum (today)",
        "dots: matmul",
        "dots: matmul hifi",
        "mix: mul+sum (today)",
        "mix floor: sum(v, dim=1)",
        "mix shaped: mul+sum",
        "mix shaped: mul+fast_reduce_nc",
        "mix floor: fast_reduce_nc(v, dim=1)",
        "mix floor: fast_reduce_nc fp32acc",
        "mix fused: weighted_reduce_nc",
        "mix fused: weighted_reduce_nc fp32 w",
    ],
    ids=[
        "floor",
        "sumsq-today",
        "sumsq-pre-ag",
        "sumsq-pre-ag-hifi",
        "dots-today",
        "dots-matmul",
        "dots-matmul-hifi",
        "mix-today",
        "mix-floor",
        "mix-shaped-sum",
        "mix-shaped-frnc",
        "mix-floor-frnc",
        "mix-floor-frnc-fp32acc",
        "mix-fused",
        "mix-fused-fp32w",
    ],
)
def test_perf_d_reduction_by_form(mesh_device, variant):
    """P7: the `d`-wide reductions, which are 76% of the read's device time.

    P4 through P6 chased the collective and won 4.5%. The profiler says the
    real weight is elsewhere: seven ops that touch the full `[1, C, N, d/tp]`
    tensor, of which `mul(v, v)` + `sum` alone is 1 041 µs per read. Both of the
    op's `d`-reductions are written as elementwise-then-reduce, so each one
    writes a second copy of the biggest tensor in the op to DRAM and reads it
    back — three passes where the arithmetic needs one.

    `floor: sum(v)` is the control: one pass over `v`, a tile-column out, no
    intermediate. Nothing in this table can beat it, and the gap between it and
    `mul+sum` is what the extra two passes cost. `rms_norm_pre_ag` is the
    distributed-RMSNorm statistics op doing the square inside the reduce kernel;
    `matmul` is the dot as the matvec it actually is.

    Traced, so this is device time. Precision is not free at either candidate —
    the matvec accumulates ~3x looser than `mul` + `sum` at default fidelity
    (`tests/probe_stats_primitive.py`) — so a win here buys a PCC re-gate, not
    a drop-in.

    P9 added the four `mix shaped` / `fast_reduce_nc` rows and they say two
    things. The reduce is refuted as a lever: `fast_reduce_nc` lands within
    0.08% of `ttnn.sum` over two runs, inside a 0.35% band, because both already
    run at the memory floor — which the four floor rows now pin to ~229 µs across
    two axes, two kernels, and fp32 dest accumulation on and off. Not two
    fidelities: `ttnn.sum` defaults to HiFi4 + fp32 dest acc on Blackhole
    (`reduce_op.cpp:109`) and `fast_reduce_nc` defaults to HiFi4, so every row
    here is HiFi4. And P7's own `mix` row above sits
    15% high — it reuses `q`, so it measures a `[1, 1, 1, d/tp]` broadcast where
    the op performs a `[1, C, N, 1]` one. Both rows are kept: the first for
    continuity with the recorded P7 number, the second because it is the op.
    """
    tokens_per_rank = PRODUCTION_TOKENS // 2
    width_per_rank = HIDDEN_SIZE // 4
    torch.manual_seed(0)
    place = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    v = place(torch.randn(1, NUM_SEALED + 1, tokens_per_rank, width_per_rank))
    q = place(torch.randn(1, 1, 1, width_per_rank))
    weights = place(torch.randn(1, NUM_SEALED + 1, tokens_per_rank, 1))
    weights_fp32 = ttnn.from_torch(
        torch.randn(1, NUM_SEALED + 1, tokens_per_rank, 1),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
    )

    body = _reduce_variants(v, q, weights, weights_fp32)[variant]
    ttnn.deallocate(body())
    ttnn.synchronize_device(mesh_device)

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = body()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

    enqueue_us, total_us = _bench(
        mesh_device, lambda: ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    )
    read_mib = (NUM_SEALED + 1) * tokens_per_rank * width_per_rank * 2 / 1024**2
    _report(f"{variant} [{read_mib:.0f} MiB in] traced", enqueue_us, total_us)

    ttnn.release_trace(mesh_device, trace_id)
    for tensor in (out, v, q, weights, weights_fp32):
        ttnn.deallocate(tensor)


@pytest.mark.parametrize(
    "mesh_device, device_params", [((2, 4), FABRIC)], indirect=["mesh_device", "device_params"], ids=["mesh-2x4"]
)
@pytest.mark.parametrize("num_sealed", [1, 4, 8])
@pytest.mark.parametrize("one_pass_stats", [False, True], ids=["three-pass", "one-pass"])
def test_perf_read_by_reduction_form(mesh_device, num_sealed, one_pass_stats):
    """P7 in the op, not in isolation — the lesson P6 paid for.

    The variants above say the one-pass forms are worth 892 µs of a 3 136 µs read
    at `S = 8`. They were measured on a bare `v`, so they were not charged for
    what the op has to do to use them: slice column 0 out of the statistics
    kernel's 32-wide output, and transpose `q` into a column for the matvec. P6's
    fold looked like a 301 µs win standalone and netted 148 in place.

    Both `S` are here for the same reason as in the layout sweep: the schedule's
    mean is 5.39. Unlike the fold, this saving is proportional to `C = S + 1`,
    so the ratio should hold at both and the absolute saving should not.
    """
    prefix_sum, block_residual, query = _make_case(PRODUCTION_TOKENS, HIDDEN_SIZE, num_sealed)

    op = TtAttnRes(mesh_device, hidden_size=HIDDEN_SIZE, eps=EPS, one_pass_stats=one_pass_stats)
    assert op.one_pass_squares == one_pass_stats, "the width gate should not fire at d/tp = 1792"
    tt_prefix, tt_block, tt_query = _place(op, prefix_sum, block_residual, query)

    ttnn.deallocate(op.forward(tt_prefix, tt_block, tt_query))
    ttnn.synchronize_device(mesh_device)

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = op.forward(tt_prefix, tt_block, tt_query)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

    enqueue_us, total_us = _bench(
        mesh_device, lambda: ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    )
    form = "one-pass" if one_pass_stats else "three-pass"
    _report(f"S={num_sealed} {form} traced", enqueue_us, total_us)

    ttnn.release_trace(mesh_device, trace_id)
    for tensor in (out, tt_prefix, tt_block, tt_query):
        ttnn.deallocate(tensor)


NUM_SITES = 24


def _inter_block_variants(sealed, queries, sites):
    """`inter_block` reads the sealed tensor once per read site, twice over.

    The split form's whole premise is that a sealed snapshot is write-once, so
    work over it is loop-invariant across the 12 layers of a block. The
    reciprocal-RMS pass is hoisted on exactly that argument — but the dots and the
    mixture are not: both run inside the `for q in queries` loop, so the sealed
    tensor is read **48 times per block** to serve 24 read sites.

    Both loops are one contraction each, and they differ in where the contracted
    axis sits. The dots contract over `d`, already the last axis, so stacking the
    24 queries into a `[1, 1, d/tp, R]` matrix turns 24 matvecs into one matmul
    with no layout change at all — and it fixes the matvec's other problem, since
    a 24-wide output wastes 8 of 32 columns instead of 31. The mixture contracts
    over the *candidate* axis, which has to become a tile axis for a matmul to
    reach it, and `S = 8` tile-pads to 32: a 4x tax on every byte of the sealed
    tensor. That tax is fatal at one read site and cheap across 24.
    """
    stacked = ttnn.concat([ttnn.permute(q, [0, 1, 3, 2]) for q in queries], dim=3)  # [1, 1, d/tp, R]
    weights = ttnn.ones(
        [1, sealed.shape[1], sealed.shape[2], 1], dtype=sealed.dtype, layout=ttnn.TILE_LAYOUT, device=sealed.device()
    )
    # Built here, not inside `mix_batched`: `ttnn.ones` writes from host, and a
    # write inside a trace capture is fatal. Stand-in for the mixture weights,
    # which a real caller would already hold on device.
    selector = ttnn.ones(
        [1, sealed.shape[2], sites, sealed.shape[1]],
        dtype=sealed.dtype,
        layout=ttnn.TILE_LAYOUT,
        device=sealed.device(),
    )

    def dots_per_site():
        return [ttnn.matmul(sealed, ttnn.permute(q, [0, 1, 3, 2]), compute_kernel_config=HIFI) for q in queries]

    def dots_batched():
        return [ttnn.matmul(sealed, stacked, compute_kernel_config=HIFI)]

    def mix_per_site():
        return [ttnn.sum(ttnn.mul(sealed, weights), dim=1, keepdim=True) for _ in range(sites)]

    def mix_batched():
        transposed = ttnn.permute(sealed, [0, 2, 1, 3])  # [1, N, S, d/tp] — S tile-pads 8 -> 32
        stacked_out = ttnn.matmul(selector, transposed, compute_kernel_config=HIFI)  # [1, N, R, d/tp]
        ttnn.deallocate(transposed)
        # Both conversions are charged. The matmul alone beats the loop; the
        # question P6 settled is whether it still does once the caller is handed
        # partials in the layout it consumes, one per site on dim 1.
        out = ttnn.permute(stacked_out, [0, 2, 1, 3])
        ttnn.deallocate(stacked_out)
        return [out]

    return {
        "floor: one pass over sealed": lambda: [ttnn.sum(sealed, dim=1, keepdim=True)],
        "dots: x24 (today)": dots_per_site,
        "dots: batched matmul": dots_batched,
        "mix: x24 (today)": mix_per_site,
        "mix: batched matmul": mix_batched,
    }, (stacked, weights, selector)


@pytest.mark.parametrize(
    "mesh_device, device_params", [((2, 4), FABRIC)], indirect=["mesh_device", "device_params"], ids=["mesh-2x4"]
)
@pytest.mark.parametrize(
    "variant",
    [
        "floor: one pass over sealed",
        "dots: x24 (today)",
        "dots: batched matmul",
        "mix: x24 (today)",
        "mix: batched matmul",
    ],
    ids=["floor", "dots-x24", "dots-batched", "mix-x24", "mix-batched"],
)
def test_perf_inter_block_batching(mesh_device, variant):
    """P8: the 48 reads of the sealed tensor per block, and whether 2 will do.

    Priced per *block* of 12 layers (24 read sites), not per read, because that is
    the scope `inter_block` is hoisted to. The floor is one pass over the sealed
    tensor — 70 MiB at `S = 8` — so every row here reads in multiples of it.
    """
    tokens_per_rank = PRODUCTION_TOKENS // 2
    width_per_rank = HIDDEN_SIZE // 4
    torch.manual_seed(0)
    place = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    sealed = place(torch.randn(1, NUM_SEALED, tokens_per_rank, width_per_rank))
    queries = [place(torch.randn(1, 1, 1, width_per_rank)) for _ in range(NUM_SITES)]

    variants, held = _inter_block_variants(sealed, queries, NUM_SITES)
    body = variants[variant]
    for tensor in body():
        ttnn.deallocate(tensor)
    ttnn.synchronize_device(mesh_device)

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    outputs = body()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

    enqueue_us, total_us = _bench(
        mesh_device, lambda: ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    )
    sealed_mib = NUM_SEALED * tokens_per_rank * width_per_rank * 2 / 1024**2
    _report(f"{variant} [{sealed_mib:.0f} MiB sealed] traced", enqueue_us, total_us)

    ttnn.release_trace(mesh_device, trace_id)
    for tensor in (*outputs, *held, sealed, *queries):
        ttnn.deallocate(tensor)
