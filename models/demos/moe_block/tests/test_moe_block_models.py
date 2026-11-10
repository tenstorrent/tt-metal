# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The whole MoE block (MoE head -> sparse dispatch -> flat experts + combine -> MoE tail) for every model it runs:
accuracy, determinism and per-op device time, each its own test function, all parametrized by the same test cases.

* A model config (BlockModel, MODELS) is what a model brings: its block dims (through TtK3MoEBlockFfn), seeded random
  weights, the host reference on them, and - Kimi-K3 - its real layers and golden activations on Weka.
* A test config (BlockCase, CASES) is one run of a model: the mesh (8x4 torus, 4x4 SP ring / TP line, 4x4 torus), the
  chunk, random or Weka weights, and the expected numbers - the accuracy floor and each op's traced device time.

test_moe_block_accuracy: the block's output vs the model's host reference (random weights) or the golden trace's MoE
    output (Weka), PCC at least the case's floor.
test_moe_block_determinism: `calls` eager calls, then `calls` replays of a captured trace, every output bit-identical to
    the first - compared on the device, one scalar a chip read back (op_checks.DeterminismMarker). Runs with the
    combine's snap (q 2^-14): without it the combine's fp32 sums depend on the rows' arrival order.
test_moe_block_perf: traced replays on the real-time program profiler, each op's device time (its slowest chip,
    the median over op_checks.PERF_RUNS (5) replays) no slower than its baseline + margin; baselines in
    moe_block_perf_baselines.json, per machine and case (MOE_PERF_RECORD=1 records them; a case without them on the
    machine skips, naming the command that records them).

4x4 cases run on half a Galaxy: TT_MESH_GRAPH_DESC_PATH (the sub-torus descriptor) and TT_VISIBLE_DEVICES set as for
test_k3_moe_block's 4x4 runs. Kimi-K3 Weka cases need $KIMI_K3_CKPT and the golden trace ($KIMI_K3_GOLDEN_TRACE or the
default) - they skip without them, and fail in CI."""

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from tests.ttnn.utils_for_testing import comp_pcc

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tests/ttnn/unit_tests/operations/experimental/moe_block"))
from op_checks import (  # noqa: E402
    BLOCK_OPS,
    DeterminismMarker,
    PERF_RUNS,
    RECORD,
    MeshBarrier,
    PerfBaselines,
    machine,
    median_us,
    op_device_times,
    traced_op_device_times,
)
from block_cases import (  # noqa: E402
    MESHES,
    MESH_SP_AXIS,
    MODELS,
)


@dataclass(frozen=True)
class BlockCase:
    """One run of a model and its expected numbers."""

    model: str
    mesh: str = "8x4"
    chunk: int = 5120
    weights: str = "random"  # "random" | "weka" (Kimi-K3: the layer's real weights, the golden trace's activations)
    layer: int = 1  # (weka: the model layer)
    pcc: Optional[float] = None  # (None: the model's floor; weka: K3_GOLDEN_PCC)
    perf_margin: float = 0.05  # (the baselines: moe_block_perf_baselines.json)
    calls: int = 10
    # a padded chunk (actual_start, actual_end): main's chunked-prefill metadata - the chunk's real tokens, in main's
    # rotated layout over the SP rows (chunk_layout.rotated_chunk_positions); None: every row real
    pad: Optional[tuple] = None
    packing: str = ""  # (the id's label of the production packing the chunk stands for)
    # a packed forward through the model adapter: each chunk's (slot, actual_start, actual_end), the forward's metadata
    # words read on device (common/prefill/prefill_chunk.py; each SP row holds every chunk's rows back to back)
    chunks: Optional[tuple] = None

    @property
    def sp_axis(self) -> int:
        return MESH_SP_AXIS.get(self.mesh, 0)

    @property
    def id(self) -> str:
        w = "" if self.weights == "random" else f"-weka-L{self.layer}"
        p = f"-{self.packing}" if self.packing else ""
        pad = f"-real{self.pad[1] - self.pad[0]}@{self.pad[0]}" if self.pad else ""
        if self.chunks:
            pad += "-packed-" + "-".join(f"{e - s0}@{s0}" for _, s0, e in self.chunks)
        return f"{self.model}-{self.mesh}-{self.chunk}{w}{p}{pad}"


# MiniMax-M3 / M3.1 in production: the 16-stage pipeline (4 Galaxies) / 4-stage intra-Galaxy pipeline, every stage a
# [2, 4] tray (SP 2 x TP 4, Fabric2D, no wrap). The packings its runner runs (models/demos/minimax_m3:
# scripts/prefill_matrix, tt/runners/manifests):
# - one 5120-token chunk a forward (every binding and the perf matrix: PREFILL_CHUNK_SIZE 5120), full;
# - 5120 with the last chunk partial: the perf matrix's new-token sizes 640 / 1600 / 3072 (p25 / p50 / p75) and
#   6900's (p90) second chunk 1780, after a chunk-aligned cached prefix (61440, its p25);
# - a multi-turn resume mid-chunk (any 32-token boundary: PREFILL_PRODUCER_MULTI_TURN_PROB) - the chunk starts mid-slab,
#   its tokens rotated over the SP rows: full and partial;
# - four 1024-token chunks packed in one 4096-token forward (manifests/*_packed_*: PREFILL_FORWARD_TOKENS_BUDGET 4096),
#   all real (each SP row holds every chunk's 512 rows back to back - for the MoE one 4096-token chunk). A packed forward
#   with a partial chunk is not expressible: the block takes one padded range a call;
# - 2048-token chunks (the perf matrix's smallest CHUNK).
M3_PREFIX = 61440
M3_RESUME = M3_PREFIX + 1600  # (a resume after the p50 request: mid-slab, 32-aligned)
M3_PRODUCTION = [
    BlockCase("minimax_m3", "2x4", 5120, packing="c5120"),
    *(
        BlockCase("minimax_m3", "2x4", 5120, packing="c5120", pad=(M3_PREFIX, M3_PREFIX + n))
        for n in (640, 1600, 3072, 1780)
    ),
    BlockCase("minimax_m3", "2x4", 5120, packing="c5120-resume", pad=(M3_RESUME, M3_RESUME + 5120)),
    BlockCase("minimax_m3", "2x4", 5120, packing="c5120-resume", pad=(M3_RESUME, M3_RESUME + 1780)),
    BlockCase("minimax_m3", "2x4", 4096, packing="4xc1024"),
    BlockCase("minimax_m3", "2x4", 2048, packing="c2048"),
]

CASES = [
    BlockCase("glm53", "8x4", 5120),
    BlockCase("glm53", "8x4", 10240),
    BlockCase("glm53", "4x4-y", 5120),
    BlockCase("glm53", "4x4-y", 10240),  # (local baselines: the SP-ring 10240 point, K3 past the old head L1 limit)
    BlockCase("kimi_k3", "8x4", 10240),
    BlockCase("kimi_k3", "4x4-y", 5120),
    BlockCase("kimi_k3", "4x4-y", 7680),
    BlockCase("kimi_k3", "4x4-y", 10240),
    BlockCase("glm53", "4x4-xy", 7680),
    BlockCase("kimi_k3", "8x4", 5120),
    BlockCase("kimi_k3", "4x4-y", 2560),
    BlockCase("kimi_k3", "8x4", 5120, weights="weka", layer=1),
    BlockCase("dsv41_flash", "8x4", 5120),
    *M3_PRODUCTION,
    # padding / packed forwards through the model adapter (TtK3MoEBlockFfn: main's metadata, or the forward's words)
    BlockCase("glm53", "8x4", 5120, pad=(1000, 4100)),
    BlockCase("kimi_k3", "8x4", 5120, pad=(5120, 5120 + 2000)),
    BlockCase("glm53", "8x4", 5120, chunks=((0, 0, 2560), (1, 300, 2000))),
    # the 4x4 tray-half stage (#61481): SP on the axis-1 ring, TP on the axis-0 line - the block on the transposed mesh
    BlockCase("glm53", "4x4-sp1", 2560),
    BlockCase("glm53", "4x4-sp1", 5120),
    BlockCase("kimi_k3", "4x4-sp1", 2560),
    BlockCase("kimi_k3", "4x4-sp1", 5120),
    BlockCase("glm53", "4x4-sp1", 5120, pad=(1000, 3100)),
    BlockCase("kimi_k3", "4x4-sp1", 2560, pad=(3 * 2560, 3 * 2560 + 1500)),
    BlockCase("glm53", "4x4-sp1", 5120, chunks=((0, 0, 2560), (1, 700, 1900))),
    BlockCase("kimi_k3", "4x4-sp1", 5120, chunks=((0, 0, 1800), (1, 2560, 5120))),
    # batching as the 4x4 stage runs it (#63276 / #63937: 2,560-token chunks, PREFILL_FORWARD_TOKENS_BUDGET 5120 /
    # 10240): two full chunks a forward, and four with partial ones (p25 / p50 new-token sizes, a resume mid-slab)
    BlockCase("glm53", "4x4-sp1", 5120, chunks=((0, 0, 2560), (1, 2560, 5120))),
    BlockCase("kimi_k3", "4x4-sp1", 5120, chunks=((0, 0, 2560), (1, 2560, 5120))),
    BlockCase("glm53", "4x4-sp1", 10240, chunks=((0, 0, 2560), (1, 2560, 4160), (2, 0, 640), (3, 6400, 7680))),
    BlockCase("kimi_k3", "4x4-sp1", 10240, chunks=((0, 0, 2560), (1, 2560, 4160), (2, 0, 640), (3, 6400, 7680))),
    BlockCase("glm53", "4x4-sp1", 10240),
    BlockCase("kimi_k3", "4x4-sp1", 10240),
    BlockCase("glm53", "4x4-sp1", 2560, pad=(2560, 2560 + 1600)),
]


def _params(cases):
    out = []
    for c in cases:
        shape, make_params, topology = MESHES[c.mesh]
        out.append(
            pytest.param(
                c,
                shape,
                make_params(**MODELS[c.model].mesh_params()),
                marks=pytest.mark.requires_mesh_topology(mesh_shape=shape, topology=topology),
                id=c.id,
            )
        )
    return out


PARAMS = pytest.mark.parametrize("case, mesh_device, device_params", _params(CASES), indirect=["mesh_device", "device_params"])


# ---------------------------------------------------------------------------------------------------------- the runs
COMBINE_SNAP = 2.0**-14  # (the determinism test's combine snap q)
K3_GOLDEN_PCC = float(os.environ.get("K3_GOLDEN_PCC", "0.97"))
K3_GOLDEN_START = int(os.environ.get("K3_GOLDEN_START", str(4 * 5120)))  # (the chunk's first token in the trace)


def _require(ok: bool, what: str):
    if ok:
        return
    if os.environ.get("GITHUB_ACTIONS"):
        pytest.fail(f"CI runs this case: {what}")
    pytest.skip(what)


@dataclass
class _Run:
    ffn: object
    x_tt: object
    want: Optional[torch.Tensor]  # [S, H] fp32: the reference or the golden output
    pcc: float
    pad: Optional[tuple] = None  # (a padded chunk: (actual_start, actual_end))
    pad_words: Optional[tuple] = None  # (... and the block's actual_start / actual_end device words: traceable)
    meta: Optional[object] = None  # (a packed forward: its metadata words, one uint32 device tensor)
    F: int = 1

    def __call__(self, eager: bool = False):
        """One call through the model adapter (TtK3MoEBlockFfn) on the case's input, its padding as the model passes it:
        main's one-chunk contract - eager, actual_isl / actual_start (host-written words); else the metadata tuple of
        device scalars (traceable) - or a packed forward's metadata words."""
        if self.meta is not None:
            return self.ffn(self.x_tt, forward_metadata=self.meta, forward_chunks=self.F)
        if self.pad is None:
            return self.ffn(self.x_tt)
        start, end = self.pad
        if eager:
            return self.ffn(self.x_tt, actual_isl=end - start, actual_start=start)
        return self.ffn(self.x_tt, actual_isl=end - start, metadata=(None, *self.pad_words))


def _real_rows(case: BlockCase, R: int) -> torch.Tensor:
    """[S] bool: the case's real rows in the host x's order (SP row r's rows r S/R ..: a prefix of each, as main's
    rotated layout puts them; a packed forward: each chunk's block of B = S / (R F) rows of each SP row)."""
    from models.demos.common.prefill.chunk_layout import rotated_chunk_real_counts

    local = case.chunk // R
    real = torch.ones(case.chunk, dtype=torch.bool)
    if case.pad is not None:
        start, end = case.pad
        for r, n in enumerate(rotated_chunk_real_counts(start, end - start, R, local)):
            real[r * local + n : (r + 1) * local] = False
    if case.chunks:
        F = len(case.chunks)
        B = local // F
        real[:] = False
        for k, (_, start, end) in enumerate(case.chunks):
            for r, n in enumerate(rotated_chunk_real_counts(start, end - start, R, B)):
                real[r * local + k * B : r * local + k * B + n] = True
    return real


def _sp(case: BlockCase, mesh_device) -> int:
    return tuple(mesh_device.shape)[case.sp_axis]


def _mesh_dims(case: BlockCase) -> tuple:
    """ShardTensor2dMesh / ConcatMesh2dToTensor dims: tokens (dim 2) over the SP axis, hidden (dim 3) over TP"""
    return (2, 3) if case.sp_axis == 0 else (3, 2)


# the combine's landing for a landing study run (rings / shared / flat_cores / shared_flat_cores); unset: the planner's.
# A run with it set measures, never gates or records the perf baselines.
TEST_LANDING = os.environ.get("MOE_TEST_LANDING") or None


def _setup(
    case: BlockCase,
    mesh_device,
    combine_snap: Optional[float] = None,
    landing: Optional[str] = TEST_LANDING,
    sd: Optional[dict] = None,
) -> _Run:
    """The block for the case (weights loaded on its first call) and its input on the device. combine_snap: the
    combine's snap q (None: production's, off); landing: the combine's landing (None: the planner's); sd: the model's
    random state dict, when the caller already made it (random-weight cases)."""
    from models.demos.deepseek_v3_d_p.tt.kimi_k3.moe_block_ffn import MoEBlockFfnOptions, TtK3MoEBlockFfn
    from models.demos.moe_block.tt.moe_block import forward_words

    opts = MoEBlockFfnOptions(combine_snap=combine_snap, sp_axis=case.sp_axis, combine_landing=landing)

    m = MODELS[case.model]
    R, C = tuple(mesh_device.shape)
    H = m.cfg.EMB_SIZE
    if case.weights == "weka":
        assert m.weka, f"{m.name}: no Weka weights / goldens in these tests"
        assert case.sp_axis == 0, "Weka cases: SP on mesh axis 0 (the TtMoe cache's placement)"
        from models.demos.deepseek_v3_d_p.tests.kimi_k3.golden import TRACE_100K, resolve_checkpoint, resolve_trace

        trace = resolve_trace(TRACE_100K)
        L = case.layer
        _require(trace is not None, f"no K3 golden trace at {TRACE_100K} ($KIMI_K3_GOLDEN_TRACE)")
        _require(
            trace.has("moe_io", f"moe_input_layer_{L}") and trace.has("moe_io", f"moe_output_layer_{L}"),
            f"{trace.path.name} has no moe_io for layer {L}",
        )
        _require(resolve_checkpoint() is not None, "no K3 checkpoint ($KIMI_K3_CKPT)")
        s0, s1 = K3_GOLDEN_START, K3_GOLDEN_START + case.chunk
        x = trace.rows("moe_io", f"moe_input_layer_{L}", s0, s1).to(torch.bfloat16)
        want = trace.rows("moe_io", f"moe_output_layer_{L}", s0, s1).float()
        ffn = TtK3MoEBlockFfn(mesh_device, m.cfg, layer_idx=L, options=opts)
        pcc = case.pcc if case.pcc is not None else K3_GOLDEN_PCC
    else:
        sd = sd if sd is not None else m.weights(R * C)
        x = (torch.randn(case.chunk, H, generator=torch.Generator().manual_seed(7)) * m.x_scale).to(torch.bfloat16)
        want = None  # (computed only by the accuracy test: the host reference costs minutes)
        ffn = TtK3MoEBlockFfn(mesh_device, m.cfg, layer_idx=m.layer, state_dict=sd, options=opts)
        ffn._test_sd = sd
        pcc = case.pcc if case.pcc is not None else m.pcc
    x_tt = ttnn.from_torch(
        x.reshape(1, 1, case.chunk, H),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(R, C), dims=_mesh_dims(case)),
    )
    ffn._test_x = x
    run = _Run(ffn, x_tt, want, pcc)
    if case.pad is not None:
        assert case.weights == "random", "padded cases: random weights only"
        block = ffn._block(case.chunk)
        run.pad = case.pad
        run.pad_words = block.chunk_scalars(*case.pad)  # (written now, before any trace: replays read them)
    if case.chunks:
        assert case.weights == "random" and case.pad is None, "packed cases: random weights, no one-chunk span"
        run.F = len(case.chunks)
        block = ffn._block(case.chunk, run.F)
        run.meta = block._meta_tensor(forward_words(list(case.chunks), run.F))  # (before any trace)
    return run


def _to_host(y_tt, mesh_device, chunk, H, case: Optional[BlockCase] = None):
    R, C = tuple(mesh_device.shape)
    dims = _mesh_dims(case) if case is not None else (2, 3)
    comp = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=(R, C), dims=dims)
    return ttnn.to_torch(y_tt, mesh_composer=comp).float().reshape(chunk, H)


# ---------------------------------------------------------------------------------------------------------- the tests
@PARAMS
@pytest.mark.skipif(not is_blackhole(), reason="Blackhole only")
@pytest.mark.timeout(1800)
def test_moe_block_accuracy(case, mesh_device, device_params):
    m = MODELS[case.model]
    run = _setup(case, mesh_device)
    H = m.cfg.EMB_SIZE
    y = _to_host(run(eager=True), mesh_device, case.chunk, H, case)
    # (a padded chunk: its real rows against the reference; the padding rows carry the shared expert's output only and
    # are discarded by the model - finite is all they need be)
    real = _real_rows(case, _sp(case, mesh_device))
    assert torch.isfinite(y).all(), f"{case.id}: non-finite outputs"
    x = run.ffn._test_x if run.want is None else None
    want = run.want[real] if run.want is not None else m.reference(x[real], run.ffn._test_sd)
    ok, msg = comp_pcc(want.reshape(-1, H), y[real], run.pcc)
    logger.info(f"ACCURACY {case.id}: {msg} over {int(real.sum())} real rows (floor {run.pcc})")
    assert ok, f"{case.id}: the block vs the reference: {msg} (need {run.pcc})"


@PARAMS
@pytest.mark.skipif(not is_blackhole(), reason="Blackhole only")
@pytest.mark.timeout(1800)
def test_moe_block_determinism(case, mesh_device, device_params):
    # (with the combine's snap: every routed term rounded to a multiple of q, so its fp32 sums do not depend on the order
    # the rows land in - without it they do, and runs differ in the last bits)
    run = _setup(case, mesh_device, combine_snap=COMBINE_SNAP)
    ttnn.deallocate(run())  # (weights, programs)
    det = DeterminismMarker(mesh_device)
    for _ in range(case.calls):
        y = run()
        det.add({"y": y})
        ttnn.deallocate(y)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    y_tr = run()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    for _ in range(case.calls):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        det.add({"y": y_tr})
    ttnn.release_trace(mesh_device, tid)
    det.assert_identical(f"{case.id}: {case.calls} eager calls + {case.calls} traced replays")


BASELINES = PerfBaselines(Path(__file__).with_name("moe_block_perf_baselines.json"))


def _spread(t):
    """{op: [ns]} -> 'op median (min-max) us | ...'"""
    return " | ".join(
        f"{op} {median_us(v):.0f} ({min(v) / 1e3:.0f}-{max(v) / 1e3:.0f}) us x{len(v)}"
        for op, v in t.items()
        if v and op != "other"
    ) + (f" | other {len(t['other'])} programs {sum(t['other']) / 1e3:.0f} us" if t.get("other") else "")


def _traced_times(run: _Run, mesh_device, out=None) -> dict:
    """{op: [ns]} of PERF_RUNS replays of one captured call, each op's slowest chip. A barrier ahead of the block in
    the trace (op_checks.MeshBarrier): every chip starts the head together - with it the head is 202 (199-204) us, with
    or without a long device lead ahead. MOE_PERF_BARRIER=0 measures without it. out: a list the captured call's output
    is appended to (read after the replays)."""
    barrier = MeshBarrier(mesh_device) if os.environ.get("MOE_PERF_BARRIER", "1") == "1" else None

    def sync():
        if barrier is not None:
            barrier()

    sync()
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    sync()
    y = run()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)  # (a warm replay)
    t = traced_op_device_times(mesh_device, tid, PERF_RUNS)
    ttnn.release_trace(mesh_device, tid)
    if barrier is not None:
        barrier.close()
    if out is not None:
        out.append(y)
    return t


@PARAMS
@pytest.mark.skipif(not is_blackhole(), reason="Blackhole only")
@pytest.mark.timeout(1800)
def test_moe_block_perf(case, mesh_device, device_params):
    if TEST_LANDING is None and not RECORD:
        # (a case without its baselines on this machine is not gated - skipped before it takes the mesh, as tt-metal's
        # perf tests treat a missing baseline; it is recorded on the machine with MOE_PERF_RECORD=1)
        missing = [op for op in BLOCK_OPS if BASELINES.expected(f"{case.id} {op}") is None]
        if missing:
            pytest.skip(
                f"no recorded {machine()} baseline for {case.id} ({', '.join(missing)}) in {BASELINES.path.name}; "
                f"record with MOE_PERF_RECORD=1 pytest '{Path(__file__).name}::test_moe_block_perf[{case.id}]' "
                "(4x4-sp1: TT_MESH_GRAPH_DESC_PATH / TT_VISIBLE_DEVICES as block_cases.MESHES says)"
            )
    run = _setup(case, mesh_device)
    ttnn.deallocate(run())  # (weights, programs)

    def eager():
        for _ in range(PERF_RUNS):
            ttnn.deallocate(run())

    # (eager, logged only: a chip's program that starts late on host dispatch shows the wait as device time)
    te = op_device_times(mesh_device, eager)
    logger.info(f"PERF {case.id} eager: " + _spread(te))
    t = _traced_times(run, mesh_device)
    measured = {op: median_us(t[op]) for op in BLOCK_OPS}
    logger.info(f"PERF {case.id} traced: " + _spread(t))
    if TEST_LANDING is not None:
        logger.info(f"PERF {case.id} landing {TEST_LANDING}: measured only")
        return
    for op in BLOCK_OPS:
        if t[op]:  # (an op the block did not run: none)
            BASELINES.check(f"{case.id} {op}", measured[op], case.perf_margin)


# ------------------------------------------------------------------------------------------------- eager (untraced) time
EAGER_LOOPS = (1, 8, 32)  # (forwards back to back before one sync)
EAGER_RUNS = 5
EAGER_FLUSH_BYTES = 256 << 20  # (past the host's last-level cache: a cold forward, as one between other layers' ops)


@PARAMS
@pytest.mark.skipif(not is_blackhole(), reason="Blackhole only")
@pytest.mark.timeout(1800)
def test_moe_block_eager_time(case, mesh_device, device_params):
    """An untraced forward through the model adapter, as a model's untraced layer runs it: host time to issue (enqueue)
    and wall time to the sync after, a forward - n forwards back to back (the host's caches warm), and one forward after
    the host's caches are flushed (cold: what a layer between other layers' ops sees). Measured, not gated (host time is
    the machine's); compare two builds on one Galaxy."""
    import time

    run = _setup(case, mesh_device)
    for _ in range(2):
        ttnn.deallocate(run(eager=True))  # (weights, programs: every op's cache hit from here)
    ttnn.synchronize_device(mesh_device)
    junk = torch.empty(EAGER_FLUSH_BYTES // 4, dtype=torch.float32)

    def med(v):
        return sorted(v)[len(v) // 2]

    out = []
    for n in EAGER_LOOPS:
        enq, wall = [], []
        for _ in range(EAGER_RUNS):
            ttnn.synchronize_device(mesh_device)
            spin = time.perf_counter() + 0.05  # (the core's clock up: schedutil)
            while time.perf_counter() < spin:
                pass
            t0 = time.perf_counter()
            for _ in range(n):
                ttnn.deallocate(run(eager=True))
            t1 = time.perf_counter()
            ttnn.synchronize_device(mesh_device)
            t2 = time.perf_counter()
            enq.append((t1 - t0) / n * 1e6)
            wall.append((t2 - t0) / n * 1e6)
        out.append(f"n={n} enqueue {med(enq):.0f} wall {med(wall):.0f}")
    cold = []
    for _ in range(EAGER_RUNS):
        ttnn.synchronize_device(mesh_device)
        junk.add_(1.0)  # (every line of 256 MB written: the host's caches hold none of the forward's state)
        t0 = time.perf_counter()
        y = run(eager=True)
        cold.append((time.perf_counter() - t0) * 1e6)
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(y)
    out.append(f"cold enqueue {med(cold):.0f} (min {min(cold):.0f})")
    logger.info(f"EAGER {case.id}: " + " | ".join(out) + " us a forward")


# ------------------------------------------------------------------------------------------------- combine landing fit
# Which CombineLanding modes plan (fit L1) for each model, stage and chunk, and what the planner picks by default
# (rings or flat_cores, combine_choice_key): plans only - no weights, no forward. One mesh a test, every chunk and
# landing inside it.
LANDINGS = (None, "rings", "shared", "flat_cores", "shared_flat_cores")
LANDING_CHUNKS = {
    "glm53": (2560, 5120, 7680, 10240),
    "kimi_k3": (2560, 5120, 7680, 10240),
    "dsv41_flash": (2560, 5120, 7680, 10240),
    "minimax_m3": (2048, 2560, 4096, 5120, 7680, 10240),
}
LANDING_MESHES = {
    "glm53": ("8x4", "4x4-y", "4x4-sp1"),
    "kimi_k3": ("8x4", "4x4-y", "4x4-sp1"),
    "dsv41_flash": ("8x4", "4x4-y", "4x4-sp1"),
    "minimax_m3": ("8x4", "4x4-y", "4x4-sp1", "2x4", "4x2"),
}


def _landing_params(stages=None):
    out = []
    for model, meshes in (stages or LANDING_MESHES).items():
        for mesh in meshes:
            shape, make_params, topology = MESHES[mesh]
            out.append(
                pytest.param(
                    model,
                    mesh,
                    shape,
                    make_params(**MODELS[model].mesh_params()),
                    marks=pytest.mark.requires_mesh_topology(mesh_shape=shape, topology=topology),
                    id=f"{model}-{mesh}",
                )
            )
    return out


@pytest.mark.parametrize(
    "model, mesh, mesh_device, device_params", _landing_params(), indirect=["mesh_device", "device_params"]
)
@pytest.mark.skipif(not is_blackhole(), reason="Blackhole only")
@pytest.mark.timeout(1800)
def test_combine_landing_fit(model, mesh, mesh_device, device_params):
    """Each landing's plan (fits or why not; accumulators, landing depth, injection / local rings, accumulators on the
    ports, the flat cells' pool) for every chunk: LANDING lines, one per (chunk, landing). Fails only if production's
    planner choice (landing None) does not fit a chunk the model runs at on this stage."""
    from models.demos.deepseek_v3_d_p.tt.kimi_k3 import moe_block_ffn as mbf

    mb = ttnn.experimental.moe_block
    m = MODELS[model]
    sp_axis = MESH_SP_AXIS.get(mesh, 0)
    no_fit = []
    for chunk in LANDING_CHUNKS[model]:
        if chunk % (tuple(mesh_device.shape)[sp_axis] * 32):
            continue
        for landing in LANDINGS:
            opts = mbf.MoEBlockFfnOptions(sp_axis=sp_axis, combine_landing=landing)
            ffn = mbf.TtK3MoEBlockFfn(mesh_device, m.cfg, layer_idx=m.layer, options=opts)
            tag = f"LANDING {model} {mesh} {chunk} {landing or 'auto'}:"
            try:
                block = ffn._block(chunk)
                kw = block.expert_kw
                d = mb.routed_experts_combine_l1_plan(
                    block.resources,
                    max_tokens_per_expert=kw["max_tokens_per_expert"],
                    layout=kw["layout"],
                    combine=block.combine,
                    dispatch_group=block.plan.l1.dispatch_group,
                    head_port_cb=block.plan.l1.head_port_cb,
                )
                used = ("shared_" if d["shared"] else "") + ("flat_cores" if d["flat_pool"] else "rings")
                used = {"shared_rings": "shared"}.get(used, used)
                logger.info(
                    f"{tag} FITS used {used} A {d['A']} D {d['D']} QP {d['QP']} QL {d['QL']} acc_on_ports "
                    f"{int(d['acc_on_ports'])} PSL {d['PSL']} pool_slots {d['pool_slots']} arena "
                    f"{d['arena_bytes'] >> 10} KB static {d['static_bytes'] >> 10} KB row blocks x splits "
                    f"{block.row_blocks}x{block.splits}"
                )
            except Exception as e:  # (a plan that does not fit raises: the reason is the first line)
                why = str(e).strip().splitlines()
                why = next((w for w in why if "does not fit" in w or "TT_FATAL" in w or "fit" in w), why[0] if why else "")
                logger.info(f"{tag} NO-FIT {why[:300]}")
                if landing is None:
                    no_fit.append(chunk)
            finally:
                _release_blocks(mesh_device)
    assert not no_fit, f"{model} {mesh}: the planner's choice does not fit chunks {no_fit}"


def _release_blocks(mesh_device):
    """Drop the model adapter's cached blocks on this mesh and release their resources (arena, words)."""
    from models.demos.deepseek_v3_d_p.tt.kimi_k3 import moe_block_ffn as mbf

    mbf.release_blocks(mesh_device)


# ------------------------------------------------------------------------------------------------- combine landing perf
LANDING_MODES = ("rings", "shared", "flat_cores", "shared_flat_cores")
# Rounds a chunk's landings are measured in, each round in a rotated order: a drift over the sweep (clock, power)
# falls on every landing alike, and the planner's choice next to the landing it picks is the run's noise.
LANDING_ROUNDS = 3
# The planner's refusal of a landing: MoEBlock's plan (MoEBlock._plan_splits, at the block's build) raises the first
# plan's TT_FATAL as a RuntimeError - routed_experts_combine's "does not fit (N cells, K KB): why" / "FlatCores landing -
# no free slot on the flat cells", or, at the landing's accumulators, "moe_block: the MoE head does not fit".
_REFUSAL = re.compile(r"does not fit|no free slot on the flat cells")


def _plan_refusal(e: BaseException) -> Optional[str]:
    """The planner's refusal line in e (a RuntimeError from the block's plans), else None."""
    if not isinstance(e, RuntimeError):
        return None
    return next((w for w in str(e).strip().splitlines() if _REFUSAL.search(w)), None)


@pytest.mark.parametrize(
    "model, mesh, mesh_device, device_params", _landing_params(), indirect=["mesh_device", "device_params"]
)
@pytest.mark.skipif(not is_blackhole(), reason="Blackhole only")
@pytest.mark.timeout(7200)
def test_combine_landing_sweep(model, mesh, mesh_device, device_params):
    """Each landing's traced device time for every chunk the model runs at on this stage (random weights, a full chunk):
    a LANDINGPERF line per (round, chunk, landing) - each op's median over PERF_RUNS replays (MeshBarrier ahead), and
    the output bit-compared with the planner's choice's (the combine's snap makes the sum independent of the landing
    order: a landing whose output differs is wrong, not merely slower). LANDING_ROUNDS rounds a chunk, each in a
    rotated landing order. A landing the planner refuses (_plan_refusal) logs NO-FIT with the reason and the sweep goes
    on. Measures only; fails if the planner's choice does not plan or run at a chunk, if any landing that planned fails
    to run or be profiled, or if a landing that ran gives a different output."""
    import gc

    m = MODELS[model]
    sp_axis = MESH_SP_AXIS.get(mesh, 0)
    R, C = tuple(mesh_device.shape)
    sd = m.weights(R * C)
    wrong = []
    landings = (None, *LANDING_MODES)
    for chunk in LANDING_CHUNKS[model]:
        if chunk % (tuple(mesh_device.shape)[sp_axis] * 32):
            continue
        case = BlockCase(model, mesh, chunk)
        ref = None  # (the planner's choice's output: round 0 runs it first)
        order = [(r, landings[(r + i) % len(landings)]) for r in range(LANDING_ROUNDS) for i in range(len(landings))]
        for rnd, landing in order:
            tag = f"LANDINGPERF {model} {mesh} {chunk} {landing or 'auto'}: round {rnd}"
            run = None
            try:
                run = _setup(case, mesh_device, landing=landing, sd=sd)
                try:
                    block = run.ffn._block(chunk)  # (the block's plans: where a landing that does not fit is refused)
                except RuntimeError as e:
                    why = _plan_refusal(e)
                    if landing is None or why is None:  # (the planner's own choice must plan; other errors are bugs)
                        raise
                    logger.info(f"{tag} NO-FIT {why[:300]}")
                    continue
                # (from here every failure - weights, programs, execution, the profiler - fails the test)
                ttnn.deallocate(run())  # (weights, programs)
                out = []
                t = _traced_times(run, mesh_device, out)
                y = _to_host(out[0], mesh_device, chunk, m.cfg.EMB_SIZE, case)
                ttnn.deallocate(out[0])
                if ref is None:
                    assert landing is None, f"{model} {mesh} {chunk}: {landing} ran before the planner's choice"
                    ref, same = y, "ref"
                else:
                    d = (y - ref).abs().max().item()
                    same = "bit-identical" if d == 0 else f"DIFFERS max |d| {d:.3g}"
                    if d != 0:
                        wrong.append(f"{chunk} {landing}")
                logger.info(
                    f"{tag} " + " | ".join(f"{op} {median_us(t[op]):.1f}" for op in BLOCK_OPS if t[op])
                    + f" us | accumulators {block.accumulators} | output {same}"
                )
            finally:
                del run
                gc.collect()
                _release_blocks(mesh_device)
    assert not wrong, f"{model} {mesh}: landings whose output differs from the planner's choice: {wrong}"


# ------------------------------------------------------------------------------------------------------- chunk switch
# One layer's adapter called at two chunks whose plans pack its weights differently (a model whose calls change chunk):
# each block must get weights packed for its own plans (TtK3MoEBlockFfn._packing), so each output is a fresh adapter's at
# that chunk, bit for bit (the combine's snap: the sums do not depend on the rows' landing order).
SWITCH_STAGES = {"glm53": ("8x4", "4x4-sp1"), "kimi_k3": ("8x4", "4x4-y", "4x4-sp1")}


@pytest.mark.parametrize(
    "model, mesh, mesh_device, device_params", _landing_params(SWITCH_STAGES), indirect=["mesh_device", "device_params"]
)
@pytest.mark.skipif(not is_blackhole(), reason="Blackhole only")
@pytest.mark.timeout(3600)
def test_moe_block_chunk_switch(model, mesh, mesh_device, device_params):
    """The chunks the model runs at on this stage (LANDING_CHUNKS) are planned - plans only, a SWITCH line each with
    its head layout - and the first two whose head layouts (GX, GY, MT, KB) differ, else whose routed experts' layouts
    differ, run on one adapter (A, B, A) and each on a fresh adapter: every output must equal the fresh adapter's at its
    chunk. Skips when no two chunks pack the layer's weights differently on this stage."""
    import gc

    from models.demos.deepseek_v3_d_p.tt.kimi_k3 import moe_block_ffn as mbf

    m = MODELS[model]
    sp_axis = MESH_SP_AXIS.get(mesh, 0)
    R, C = tuple(mesh_device.shape)
    H = m.cfg.EMB_SIZE
    sd = m.weights(R * C)
    chunks = [c for c in LANDING_CHUNKS[model] if c % (tuple(mesh_device.shape)[sp_axis] * 32) == 0]
    keys = {}
    try:
        for chunk in chunks:
            opts = mbf.MoEBlockFfnOptions(combine_snap=COMBINE_SNAP, sp_axis=sp_axis)
            probe = mbf.TtK3MoEBlockFfn(mesh_device, m.cfg, layer_idx=m.layer, state_dict=sd, options=opts)
            keys[chunk] = probe._packing(probe._block(chunk))
            logger.info(f"SWITCH {model} {mesh} {chunk}: head (GX, GY, MT, KB, dtype) {keys[chunk][0]}")
            _release_blocks(mesh_device)
        pairs = [(a, b) for i, a in enumerate(chunks) for b in chunks[i + 1 :]]
        pick = next((p for p in pairs if keys[p[0]][0] != keys[p[1]][0]), None)
        pick = pick or next((p for p in pairs if keys[p[0]][1] != keys[p[1]][1]), None)
        if pick is None:
            pytest.skip(f"{model} {mesh}: chunks {chunks} all pack the layer's weights alike")
        a, b = pick
        logger.info(f"SWITCH {model} {mesh}: chunks {a} / {b}, head {keys[a][0]} / {keys[b][0]}")
        runs = {c: _setup(BlockCase(model, mesh, c), mesh_device, COMBINE_SNAP, landing=None, sd=sd) for c in pick}
        shared, got = runs[a].ffn, []
        for c in (a, b, a):
            y = shared(runs[c].x_tt)
            got.append((c, _to_host(y, mesh_device, c, H, BlockCase(model, mesh, c))))
            ttnn.deallocate(y)
        want = {}
        for c in pick:
            fresh = _setup(BlockCase(model, mesh, c), mesh_device, COMBINE_SNAP, landing=None, sd=sd)
            y = fresh.ffn(fresh.x_tt)
            want[c] = _to_host(y, mesh_device, c, H, BlockCase(model, mesh, c))
            ttnn.deallocate(y)
            del fresh
            gc.collect()
        bad = []
        for i, (c, y) in enumerate(got):
            d = (y - want[c]).abs().max().item()
            logger.info(f"SWITCH {model} {mesh}: call {i} chunk {c} vs a fresh adapter: max |d| {d:.3g}")
            if not torch.isfinite(y).all() or d != 0:
                bad.append(f"call {i} chunk {c}: max |d| {d:.3g}")
        assert not bad, f"{model} {mesh}: one adapter at chunks {a} / {b} differs from fresh adapters: {bad}"
    finally:
        gc.collect()
        _release_blocks(mesh_device)
