# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end wall clock for the whole read schedule under ordinary dispatch.

Every other number this module records is device time: `ROOFLINE.md`'s schedule totals
are per-site Tracy measurements fitted over `S` and summed, and the perf harness next to
this file replays single reads and whole blocks from a trace. None of them run the
schedule, and none of them contain a microsecond of host time — so none of them can say
what a model pays to *issue* 186 reads through Python.

This file runs the schedule, host-dispatched, program by program, the way a model costs
it before anyone captures a trace. `attn_res_stack` walks all 93 layers with the real
seal cadence, `S` ramping 0 -> 8 across 186 reads, at the 640 rows per chip prefill
shards to. What it reports is the cumulative cost of AttnRes across a whole forward.

Enqueue is reported separately from completion because their gap bounds how much of the
walk the device was still working on after the host stopped issuing. The two converging
is consistent with a host bottleneck but does not prove one on its own — pipelined
dispatch looks the same when host and device happen to run at matched rates. Settling
that needs the device-time figure, which is what `ROOFLINE.md` holds.

No timing assertion appears here. A wall-clock threshold on a shared box is a flake
generator; `_report` writes through loguru, so `pytest -s` is what makes these visible.
"""

import time
from collections import defaultdict
from contextlib import contextmanager

import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import is_blackhole
from models.experimental.kimi_k3_attn_res.torch_functional.attn_res import EPS
from models.experimental.kimi_k3_attn_res.tt.attn_res import TtAttnRes
from models.experimental.kimi_k3_attn_res.tt.attn_res_stream import BLOCK_SIZE, attn_res_stack, attn_res_stack_split

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

# The same gate the correctness suite holds the op to. Here it separates the two stack
# drivers, whose only legitimate difference is rounding order.
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

# Square dimension of the ballast matmul standing in for a real block's device work; 0
# runs the residual schedule bare. One matmul is `2*dim^3` flops in a *single* program, so
# device time sweeps across three orders while host dispatch moves by one launch — which
# is the whole point. A chain of small ops would raise both together and settle nothing.
BALLAST_DIMS = [0, 2048, 4096, 8192]


def _make_ballast(mesh_device, dim):
    """Operands for the per-block ballast matmul, replicated so no collective is involved.

    Every chip runs the same local matmul. Sharding it would drag the fabric into a
    measurement about host dispatch.
    """
    if dim == 0:
        return None
    operand = lambda: ttnn.from_torch(
        torch.zeros(1, 1, dim, dim),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    return operand(), operand()


def _make_module_stub(ballast):
    """Stands in for an attention or MLP block.

    `accumulate` takes ownership of what a module returns and the layer driver frees `h`
    afterwards, so a stub cannot hand back `h` itself without a double free. A scalar
    multiply is the cheapest genuinely-new tensor, which keeps the residual contribution
    itself nearly free.

    The ballast's result is discarded: its only job is to occupy the device for a
    controlled interval, so that whether AttnRes's dispatch hides behind a real block's
    compute becomes a measurement rather than an argument.
    """
    if ballast is None:
        return lambda h: ttnn.multiply(h, MODULE_SCALE)

    lhs, rhs = ballast

    def stub(h):
        ttnn.deallocate(ttnn.matmul(lhs, rhs))
        return ttnn.multiply(h, MODULE_SCALE)

    return stub


def _place_stack(op, seed=0):
    """Everything the walk consumes, placed once: the embeddings and all 187 queries.

    `attn_res_stack` takes ownership of the stream it is handed and frees it, so each
    timed iteration needs its own copy. The caller clones this on device — a host transfer
    inside the timed region would swamp what is being measured.
    """
    generator = torch.Generator().manual_seed(seed)
    randn = lambda *shape: torch.randn(*shape, generator=generator)

    hidden_states = randn(1, 1, PER_CHIP_TOKENS * op.sp_factor, HIDDEN_SIZE)
    embeddings = ttnn.from_torch(
        hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=op.mesh_device, mesh_mapper=op.stream_mapper
    )

    fold = lambda: op.to_query((1.0 + 0.1 * randn(HIDDEN_SIZE)) * (PROJ_STD * randn(HIDDEN_SIZE)))
    q_pre = [fold() for _ in range(LAYERS)]
    q_post = [fold() for _ in range(LAYERS)]
    return embeddings, q_pre, q_post, fold()


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


FORMS = {"direct": attn_res_stack, "split": attn_res_stack_split}

# Every `ttnn` entry point the op and its drivers reach for. A name missing from here is
# silently uncharged, so it lands in the unattributed remainder rather than being lost.
INSTRUMENTED_OPS = (
    "add",
    "all_reduce",
    "clone",
    "concat",
    "deallocate",
    "div",
    "exp",
    "experimental.attn_res_merge",
    "experimental.attn_res_scores",
    "experimental.attn_res_stats",
    "experimental.fast_weighted_reduce_nc",
    "matmul",
    "max",
    "maximum",
    "mul",
    "permute",
    "rms_norm_pre_all_gather",
    "rsqrt",
    "slice",
    "sub",
    "sum",
    "typecast",
)


@contextmanager
def _host_time_by_op(names):
    """Wall time and call count per `ttnn` entry point, for the duration of the block.

    What a wrapper here sees is the *whole* host cost of a call — pybind marshalling, shape
    inference, output allocation, and writing the dispatch commands — because a ttnn call
    returns once the program is enqueued. Dispatch cannot be separated out from this side of
    the boundary; that split needs instrumentation inside the C++ command queue.

    The timing pair costs ~100 ns against a launch that costs a hundred microseconds, so the
    instrument does not meaningfully move what it measures.
    """
    totals, counts = defaultdict(float), defaultdict(int)
    patched = []

    def timed(name, original):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = original(*args, **kwargs)
            totals[name] += time.perf_counter() - start
            counts[name] += 1
            return result

        return wrapper

    for name in names:
        holder, _, attribute = name.rpartition(".")
        target = getattr(ttnn, holder) if holder else ttnn
        original = getattr(target, attribute)
        patched.append((target, attribute, original))
        setattr(target, attribute, timed(name, original))

    try:
        yield totals, counts
    finally:
        for target, attribute, original in patched:
            setattr(target, attribute, original)


def _walker(drive, op, embeddings, queries, stub):
    """The whole 93-layer walk as a nullary callable.

    `attn_res_stack` takes ownership of the stream it is handed and frees it, so every
    iteration clones the embeddings on device rather than re-placing them from host.
    """
    q_pre, q_post, q_out = queries
    return lambda: drive(
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
@pytest.mark.parametrize("ballast_dim", BALLAST_DIMS, ids=lambda dim: f"ballast-{dim}")
def test_e2e_stack_cost(mesh_device, device_params, request, ballast_dim):
    """The full 93-layer walk under ordinary dispatch, minimum of three after ten warm.

    Swept over how much device work each block carries, because that decides whether
    AttnRes's dispatch is wall clock or is hidden. `enqueue` is when the host stopped
    issuing and `total` is when the device stopped working, so their gap reads directly:
    converged means the host is the critical path and the ballast hid behind it, while a
    gap that grows with the ballast means the device became the bottleneck and the
    dispatch disappeared into it.
    """
    placement = request.node.callspec.id
    op = TtAttnRes(mesh_device, hidden_size=HIDDEN_SIZE, eps=EPS)
    embeddings, q_pre, q_post, q_out = _place_stack(op)

    ballast = _make_ballast(mesh_device, ballast_dim)
    stub = _make_module_stub(ballast)

    walk = lambda: attn_res_stack(
        op,
        ttnn.clone(embeddings),
        q_pre,
        q_post,
        q_out,
        [stub] * LAYERS,
        [stub] * LAYERS,
        block_size=BLOCK_SIZE,
    )

    enqueue_ms, total_ms = _min_of(mesh_device, walk)

    logger.info(
        f"{placement}: {LAYERS} layers, {READS} reads, S 1->{MAX_SEALED}, "
        f"{PER_CHIP_TOKENS} rows/chip, d/{op.tp_factor}={op.shard_width}, ballast {ballast_dim}"
    )
    logger.info(
        f"  ballast {ballast_dim:<5} enqueue {enqueue_ms:>9.2f} ms   total {total_ms:>9.2f} ms   "
        f"waited {total_ms - enqueue_ms:>9.2f} ms   "
        f"{'device-bound' if total_ms - enqueue_ms > 0.1 * total_ms else 'host-bound'}"
    )

    for tensor in (embeddings, q_out, *q_pre, *q_post, *(ballast or ())):
        ttnn.deallocate(tensor)


@on_placements
def test_split_matches_direct(mesh_device, device_params):
    """The two stack drivers agree, which is what makes their timings comparable.

    A driver that batches the wrong sites is fast and wrong, and nothing else in the repo
    walks the schedule through `merge`, so this gate is the only thing standing between the
    split timing below and a number for the wrong computation.
    """
    op = TtAttnRes(mesh_device, hidden_size=HIDDEN_SIZE, eps=EPS)
    embeddings, q_pre, q_post, q_out = _place_stack(op)
    stub = _make_module_stub(None)

    outputs = {}
    for name, drive in FORMS.items():
        out = _walker(drive, op, embeddings, (q_pre, q_post, q_out), stub)()
        outputs[name] = ttnn.to_torch(out, mesh_composer=op.stream_composer)
        ttnn.deallocate(out)

    pcc = _pcc(outputs["split"], outputs["direct"])
    logger.info(f"split vs direct over {LAYERS} layers, {READS} reads: PCC {pcc:.7f}")
    assert pcc >= PCC_GATE, f"split driver disagrees with direct: PCC {pcc:.7f} < {PCC_GATE}"

    for tensor in (embeddings, q_out, *q_pre, *q_post):
        ttnn.deallocate(tensor)


@on_placements
@pytest.mark.parametrize("form", list(FORMS), ids=lambda name: f"form-{name}")
def test_e2e_split_vs_direct(mesh_device, device_params, request, form):
    """Both read forms over the whole schedule, host-dispatched and replayed from a trace.

    The forms trade host work against device work in opposite directions, so which one wins
    depends on which side is the critical path. The split form hoists the sealed half of 24
    reads into one `inter_block`, cutting device time; whether it also cuts the ~22 programs
    a read dispatches is what the untraced column answers. The traced column is the same
    walk with host dispatch removed, so the pair brackets the model's two operating points:
    a forward that captures a trace, and one that does not.
    """
    placement = request.node.callspec.id
    op = TtAttnRes(mesh_device, hidden_size=HIDDEN_SIZE, eps=EPS)
    embeddings, q_pre, q_post, q_out = _place_stack(op)
    walk = _walker(FORMS[form], op, embeddings, (q_pre, q_post, q_out), _make_module_stub(None))

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


@on_placements
@pytest.mark.parametrize("form", list(FORMS), ids=lambda name: f"form-{name}")
def test_host_time_by_op(mesh_device, device_params, request, form):
    """Where the untraced walk's host time goes, charged to the `ttnn` call that spent it.

    The untraced arm is ~87% host, so what to remove is a question about *which* calls, and
    a whole-walk number cannot answer it. Each row is the full host cost of that entry
    point, dispatch included — see `_host_time_by_op` on why the two do not separate here.

    Reported as a mean rather than the minimum the timing tests use: a per-op minimum would
    take each row from a different walk and the column would no longer sum to anything.
    """
    placement = request.node.callspec.id
    op = TtAttnRes(mesh_device, hidden_size=HIDDEN_SIZE, eps=EPS)
    embeddings, q_pre, q_post, q_out = _place_stack(op)
    walk = _walker(FORMS[form], op, embeddings, (q_pre, q_post, q_out), _make_module_stub(None))

    for _ in range(WARMUP_ITERATIONS):
        ttnn.deallocate(walk())
    ttnn.synchronize_device(mesh_device)

    with _host_time_by_op(INSTRUMENTED_OPS) as (totals, counts):
        start = time.perf_counter()
        for _ in range(MEASURED_ITERATIONS):
            ttnn.deallocate(walk())
        enqueued = time.perf_counter()
    ttnn.synchronize_device(mesh_device)

    walk_ms = (enqueued - start) * 1e3 / MEASURED_ITERATIONS
    attributed_ms = sum(totals.values()) * 1e3 / MEASURED_ITERATIONS

    logger.info(f"{placement}: host time per walk, {LAYERS} layers, {READS} reads, {PER_CHIP_TOKENS} rows/chip")
    logger.info(f"  {'op':<38} {'calls':>7} {'ms':>9} {'us/call':>9} {'%':>6}")
    for name, seconds in sorted(totals.items(), key=lambda item: -item[1]):
        per_walk_ms = seconds * 1e3 / MEASURED_ITERATIONS
        calls = counts[name] / MEASURED_ITERATIONS
        logger.info(
            f"  {name:<38} {calls:>7.0f} {per_walk_ms:>9.2f} "
            f"{per_walk_ms * 1e3 / calls:>9.1f} {100 * per_walk_ms / walk_ms:>5.1f}%"
        )
    logger.info(
        f"  {'attributed':<38} {sum(counts.values()) / MEASURED_ITERATIONS:>7.0f} {attributed_ms:>9.2f} "
        f"{'':>9} {100 * attributed_ms / walk_ms:>5.1f}%"
    )
    logger.info(f"  {'python glue (unattributed)':<38} {'':>7} {walk_ms - attributed_ms:>9.2f}")
    logger.info(f"  {'walk':<38} {'':>7} {walk_ms:>9.2f}")

    for tensor in (embeddings, q_out, *q_pre, *q_post):
        ttnn.deallocate(tensor)
