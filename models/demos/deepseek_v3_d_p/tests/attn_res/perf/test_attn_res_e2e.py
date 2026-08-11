# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end wall clock for the whole read schedule under ordinary dispatch.

The perf harness next to this file replays single reads and whole blocks from a trace, so
what it reports is device time. It never runs the schedule and never contains a
microsecond of host time, so it cannot say what a model pays to *issue* 186 reads through
Python.

This file runs the schedule, host-dispatched, program by program, the way a model costs
it before anyone captures a trace. `attn_res_stack_split` walks all 93 layers with the
real seal cadence, `S` ramping 0 -> 8 across 186 reads, at the 640 rows per chip prefill
shards to. What it reports is the cumulative cost of AttnRes across a whole forward.

Enqueue is reported separately from completion because their gap bounds how much of the
walk the device was still working on after the host stopped issuing. The two converging
is consistent with a host bottleneck but does not prove one on its own — pipelined
dispatch looks the same when host and device happen to run at matched rates.

No timing assertion appears here. A wall-clock threshold on a shared box is a flake
generator; the numbers go out through loguru, so `pytest -s` is what makes them visible.
The one assertion in the file is `test_walk_matches_reference`, which gates the walk the
timings are taken over — CI runs that test alone, by node id.
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.attn_res.attn_res import EPS, attn_res_stack
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res import TtAttnRes
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res_stream import BLOCK_SIZE, attn_res_stack_split

HIDDEN_SIZE = 7168
PRODUCTION_TOKENS = 5120
GALAXY_SP = 8
PROJ_STD = 0.02

# The schedule the model runs. Layer 0 has no sealed snapshot so its pre-attention read is
# skipped, which is why 93 layers hold 187 queries and take 186 reads: 92 pre, 93 post,
# and one model-level read after the stack.
LAYERS = 93
READS = 2 * LAYERS

# Seals fire on the first layer of every block, layer 0 included, so a stack that does not
# divide evenly still seals its trailing partial block. Layer 0's seal precedes every
# executed read, so `S` ramps 1 -> MAX_SEALED and no read runs at `S == 0`.
MAX_SEALED = -(-LAYERS // BLOCK_SIZE)

# The same gate the single-read suite holds the op to, and the walk clears it: 186 rounds of
# bf16 accumulation against an fp32 reference cost about as much as one read does, because
# every read renormalizes the stream against the sealed set rather than compounding it.
PCC_GATE = 0.9999

# Cross-mesh comparisons hold tokens *per chip* fixed: production is
# `PRODUCTION_TOKENS / sp` on the `(8, 4)` Galaxy, so a mesh with a shorter `sp` axis needs
# a proportionally shorter chunk to walk the same per-chip shape.
PER_CHIP_TOKENS = PRODUCTION_TOKENS // GALAXY_SP

# The first iterations compile kernels and fill the program cache. Ten is well past the
# point where an untraced walk stops improving, and the minimum of the three that follow
# is the least noisy estimator available on a box with other tenants.
WARMUP_ITERATIONS = 10
MEASURED_ITERATIONS = 3

# `ttnn.all_reduce` needs an initialized fabric context; without it the op dies in the
# control plane rather than returning wrong numbers. The region has to hold the whole
# schedule's dispatch commands at once, an order more programs than a single block, and it
# is carved out of DRAM at device open whether a test captures anything or not.
TRACE_REGION_SIZE = 134217728
FABRIC = {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": TRACE_REGION_SIZE}

# The production shape occurs on exactly these two meshes: TP factor 4 at 640 rows a chip.
# The Galaxy row collects everywhere and runs where there are 32 chips.
PLACEMENTS = [
    pytest.param((2, 4), FABRIC, id="mesh-2x4"),
    pytest.param((8, 4), FABRIC, id="mesh-8x4"),
]

on_placements = pytest.mark.parametrize(
    "mesh_device, device_params", PLACEMENTS, indirect=["mesh_device", "device_params"]
)

pytestmark = pytest.mark.skipif(not is_blackhole(), reason="Kimi K3 AttnRes is brought up on Blackhole only")

# Scales each module's contribution so 93 rounds of accumulation stay in bf16's range.
MODULE_SCALE = 0.02


def _module_stub(h):
    """Stands in for an attention or MLP block.

    `accumulate` takes ownership of what a module returns and the layer driver frees `h`
    afterwards, so a stub cannot hand back `h` itself without a double free. A scalar
    multiply is the cheapest genuinely-new tensor, which keeps the residual contribution
    itself nearly free and leaves the walk's cost as AttnRes's own.
    """
    return ttnn.multiply(h, MODULE_SCALE)


def _make_stack(op, seed=0):
    """Everything the walk consumes, on host: the embeddings and all 187 folded queries."""
    generator = torch.Generator().manual_seed(seed)
    randn = lambda *shape: torch.randn(*shape, generator=generator)

    hidden_states = randn(PER_CHIP_TOKENS * op.sp_factor, HIDDEN_SIZE)
    fold = lambda: (1.0 + 0.1 * randn(HIDDEN_SIZE)) * (PROJ_STD * randn(HIDDEN_SIZE))
    return hidden_states, [fold() for _ in range(LAYERS)], [fold() for _ in range(LAYERS)], fold()


def _place_stack(op, hidden_states, q_pre, q_post, q_out):
    """The same inputs on device, placed once.

    `attn_res_stack_split` takes ownership of the stream it is handed and frees it, so each
    timed iteration needs its own copy. The caller clones the embeddings on device — a host
    transfer inside the timed region would swamp what is being measured.
    """
    embeddings = ttnn.from_torch(
        hidden_states.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=op.mesh_device,
        mesh_mapper=op.stream_mapper,
    )
    return embeddings, [op.to_query(q) for q in q_pre], [op.to_query(q) for q in q_post], op.to_query(q_out)


def _min_of(mesh_device, body):
    """Minimum enqueue and completion time in milliseconds, over `MEASURED_ITERATIONS`.

    A body that returns None is a trace replay, whose output tensor is owned by the capture
    and must outlive it.
    """
    for _ in range(WARMUP_ITERATIONS):
        warm = body()
        if warm is not None:
            ttnn.deallocate(warm)
    ttnn.synchronize_device(mesh_device)

    best_enqueue = best_total = float("inf")
    for _ in range(MEASURED_ITERATIONS):
        start = time.perf_counter()
        out = body()
        enqueued = time.perf_counter()
        ttnn.synchronize_device(mesh_device)
        finished = time.perf_counter()
        if out is not None:
            ttnn.deallocate(out)
        best_enqueue = min(best_enqueue, enqueued - start)
        best_total = min(best_total, finished - start)

    return best_enqueue * 1e3, best_total * 1e3


def _pcc(got, want):
    stacked = torch.stack((got.double().reshape(-1), want.double().reshape(-1)))
    return torch.corrcoef(stacked)[0, 1].item()


def _walker(op, embeddings, queries, stub):
    """The whole 93-layer walk as a nullary callable.

    `attn_res_stack_split` takes ownership of the stream it is handed and frees it, so every
    iteration clones the embeddings on device rather than re-placing them from host.
    """
    q_pre, q_post, q_out = queries
    return lambda: attn_res_stack_split(
        op,
        ttnn.clone(embeddings),
        q_pre,
        q_post,
        q_out,
        [stub] * LAYERS,
        [stub] * LAYERS,
        block_size=BLOCK_SIZE,
    )


@on_placements
def test_walk_matches_reference(mesh_device, device_params):
    """The device walk against the torch reference walk, which is what makes the timing below
    a number for the right computation.

    A driver that batches the wrong sites, or seals on the wrong layer, is fast and wrong.
    The single-read suite cannot see either: it issues one `inter_block` and indexes its
    sites by hand, so it never exercises the seal cadence or the site bookkeeping across a
    stack. This is the only thing in the repo that does, which is why CI runs it even though
    the timing tests beside it are bringup-only.
    """
    op = TtAttnRes(mesh_device, hidden_size=HIDDEN_SIZE, eps=EPS)
    hidden_states, q_pre, q_post, q_out = _make_stack(op)
    embeddings, tt_pre, tt_post, tt_out = _place_stack(op, hidden_states, q_pre, q_post, q_out)

    device_walk = _walker(op, embeddings, (tt_pre, tt_post, tt_out), _module_stub)()
    got = ttnn.to_torch(device_walk, mesh_composer=op.stream_composer).reshape(-1, HIDDEN_SIZE)
    ttnn.deallocate(device_walk)

    torch_stub = lambda h: h * MODULE_SCALE
    want = attn_res_stack(
        hidden_states,
        q_pre,
        q_post,
        q_out,
        [torch_stub] * LAYERS,
        [torch_stub] * LAYERS,
        block_size=BLOCK_SIZE,
        eps=EPS,
    )

    pcc = _pcc(got, want)
    logger.info(f"device vs reference over {LAYERS} layers, {READS} reads: PCC {pcc:.7f}")
    assert pcc >= PCC_GATE, f"device walk disagrees with the reference: PCC {pcc:.7f} < {PCC_GATE}"

    for tensor in (embeddings, tt_out, *tt_pre, *tt_post):
        ttnn.deallocate(tensor)


@on_placements
def test_e2e_untraced_vs_traced(mesh_device, device_params, request):
    """The whole schedule, host-dispatched and replayed from a trace.

    The traced column is the same walk with host dispatch removed, so the pair brackets the
    model's two operating points: a forward that captures a trace, and one that does not.
    Their difference is what the read's dispatch count costs, which is the quantity every
    fusion in this module is aimed at.
    """
    placement = request.node.callspec.id
    op = TtAttnRes(mesh_device, hidden_size=HIDDEN_SIZE, eps=EPS)
    embeddings, q_pre, q_post, q_out = _place_stack(op, *_make_stack(op))
    walk = _walker(op, embeddings, (q_pre, q_post, q_out), _module_stub)

    untraced_enqueue, untraced_total = _min_of(mesh_device, walk)

    # Capture only after the untraced arm has run: capture records dispatch commands, and a
    # program built during capture is not the one that gets replayed.
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    captured = walk()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

    traced_enqueue, traced_total = _min_of(
        mesh_device, lambda: ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    )

    dispatch_ms = untraced_total - traced_total
    logger.info(
        f"{placement}: {LAYERS} layers, {READS} reads, S 1->{MAX_SEALED}, "
        f"{PER_CHIP_TOKENS} rows/chip, d/{op.tp_factor}={op.shard_width}"
    )
    logger.info(f"  untraced   enqueue {untraced_enqueue:>9.2f} ms   total {untraced_total:>9.2f} ms")
    logger.info(f"  traced     enqueue {traced_enqueue:>9.2f} ms   total {traced_total:>9.2f} ms")
    logger.info(
        f"  dispatch   {dispatch_ms:>9.2f} ms   {100 * dispatch_ms / untraced_total:>5.1f}% of untraced   "
        f"trace buys {untraced_total / traced_total:>5.2f}x"
    )

    ttnn.release_trace(mesh_device, trace_id)
    for tensor in (captured, embeddings, q_out, *q_pre, *q_post):
        ttnn.deallocate(tensor)
