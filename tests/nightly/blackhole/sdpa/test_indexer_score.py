# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Test for the DeepSeek-V3.2 DSA ``indexer_score`` op
(design: models/demos/deepseek_v32/INDEXER_OP.md).

    score[b, s, t] = sum_h relu(q[b, h, s, :] . k[b, t, :]) * w[b, h, s]

Causality from scalar ``chunk_start``: key ``t`` visible to query ``s`` iff
``t <= chunk_start + s``; future columns are -inf.

Main case: Galaxy chunked prefill, 5K queries vs 55K keys, SP=8 (640
queries/device).  SP enters only via ``chunk_start``, so this is single-chip
with ``sp_rank`` selecting the ring position.
"""

import os
import time
from unittest import mock

import pytest
import torch
from loguru import logger

import ttnn

GLX_HEADS, GLX_DIM = 64, 128  # full indexer head count
GLX_SQ = 640  # queries per device (5120 chunk / SP=8)
# Production tests below run two per-device head counts: 64 (whole indexer on one device) and
# 16 (the 64 heads split across TP=4 -> 16 heads/device, partial scores all-reduce-summed across
# TP). The op needs no change for the shard: it sums only the heads it is given and the causal
# -inf mask is head-independent, so -inf survives the cross-TP sum.
GLX_HEAD_CASES = [16, 64]
GLX_T = 56320  # all-gathered keys: 50K history + 5K chunk = 55K, tile-aligned
GLX_HISTORY = GLX_T - 8 * GLX_SQ  # 51200 keys visible to every query

# Small single-chip shape (sp_rank 0 of 2) used by the fast knob / validation tests.
MINI = dict(heads=64, dim=128, sq=64, t=256)


def indexer_score_ref(q, k, w, chunk_start):
    """Per-head fp32 accumulation (a full [Hi,Sq,T] tensor is ~9 GB here)."""
    b, hi, sq, _ = q.shape
    t = k.shape[2]
    q, k, w = q.float(), k.float(), w.float()
    score = torch.zeros(b, sq, t)
    for h in range(hi):
        score += torch.relu(q[:, h] @ k[:, 0].transpose(-2, -1)) * w[:, h]
    future = torch.arange(t).unsqueeze(0) > chunk_start + torch.arange(sq).unsqueeze(1)
    return score.masked_fill(future, float("-inf")).unsqueeze(1)


def make_inputs(heads, dim, sq, t, seed=42):
    """q [1,Hi,Sq,D], k [1,1,T,D], weights [1,Hi,Sq,1], all bf16.

    Weights are random (so some gates are negative): real scores can be negative,
    so -inf-padded columns must not be confused with low-but-valid scores by topk.
    """
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, heads, sq, dim, generator=g, dtype=torch.bfloat16)
    k = torch.randn(1, 1, t, dim, generator=g, dtype=torch.bfloat16)
    w = torch.randn(1, heads, sq, 1, generator=g, dtype=torch.bfloat16)
    return q, k, w


def to_device(t, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t, device=device, layout=layout, dtype=dtype)


def run_indexer(q, k, w, chunk_start, device, program_config=None, k_dtype=ttnn.bfloat16):
    """Run the device op and return the row-major bf16 score as a torch tensor.

    k may be bfp8_b (matmul srcA only); q/weights stay bf16.
    """
    kwargs = {} if program_config is None else {"program_config": program_config}
    out = ttnn.experimental.deepseek.indexer_score(
        to_device(q, device),
        to_device(k, device, dtype=k_dtype),
        to_device(w, device),
        chunk_start_idx=chunk_start,
        **kwargs,
    )
    return ttnn.to_torch(out)


def assert_indexer_match(out, ref, sq, t, check_neg=False):
    """Check the -inf map is exact and the visible scores match the reference by PCC."""
    assert out.shape == (1, 1, sq, t)
    # -inf maps must agree exactly (<= bf16 lowest counts as masked on device)
    masked = ref == float("-inf")
    assert torch.equal(out <= torch.finfo(torch.bfloat16).min, masked)
    # visible values by PCC (0.999 floor for the bf16 device op)
    a, b = out[~masked].flatten().float(), ref[~masked].flatten().float()
    pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    assert pcc >= 0.999, f"PCC {pcc} < 0.999"
    if check_neg:
        # negative gates so a zero-filled column can't masquerade as a valid score
        assert (ref[~masked] < 0).any()


@pytest.mark.parametrize(
    "heads, dim, sq, t, chunk_start",
    [
        (64, 128, 64, 256, 128),  # mini, sp_rank 0 of 2
        (64, 128, 64, 256, 192),  # mini, sp_rank 1 of 2 (fully causal corner)
        (32, 64, 64, 256, 128),  # generality: fewer heads, Dt=2
        (16, 256, 64, 256, 128),  # generality: wide head dim, Dt=8
        (GLX_HEADS, GLX_DIM, GLX_SQ, GLX_T, GLX_HISTORY + 0 * GLX_SQ),  # GLX sp_rank 0
        (GLX_HEADS, GLX_DIM, GLX_SQ, GLX_T, GLX_HISTORY + 7 * GLX_SQ),  # GLX sp_rank 7
    ],
    ids=["mini_rank0", "mini_rank1", "heads32_dim64", "heads16_dim256", "glx_rank0", "glx_rank7"],
)
def test_indexer_score_glx_chunked(device, heads, dim, sq, t, chunk_start):
    q, k, w = make_inputs(heads, dim, sq, t)
    # GLX perf config: all heads resident (default streams 1 head — safe, slow)
    cfg = None if sq <= 64 else ttnn.IndexerScoreProgramConfig(head_group_size=0)
    out = run_indexer(q, k, w, chunk_start, device, program_config=cfg)
    ref = indexer_score_ref(q, k, w, chunk_start)
    assert_indexer_match(out, ref, sq, t, check_neg=True)


def production_config(heads):
    """Production (GLX chunked-prefill) knobs. The factory no longer auto-tunes -- the caller picks
    QC/KC/HB and an oversized config is rejected, so this helper owns the per-head-count choice.

    - head_group_size=0 keeps all of this device's heads resident: head streaming re-reads q per
      output tile and is ~24x slower (heads8 sp7 0.48ms -> 11.5ms), so it is never used here.
    - QC=2 (q_chunk_size=64) reuses each resident K chunk across 2 q-rows, cutting the reader's
      redundant K reads ~2x -- the K-bandwidth knee and the win for the 8-head TP=8 shard
      (sp7 bfp8 0.73 -> 0.48 ms). At 64 heads, QC=2 with all heads resident overflows L1 (cb_q
      alone ~1 MB), so QC stays 1 (32); the 64-head case is compute-bound and gains nothing from
      QC=2 anyway.
    - k_chunk=256 (KC=8) over 512: KC=8 load-balances the work units slightly better on heads8
      (0.48 vs 0.49 ms sp7) and stays small enough to leave L1 room for QC=2; KC>=32 is too large.
    """
    # Sweep override (compute-ceiling investigation): INDEXER_SWEEP_QC / _KC are in TILES.
    sweep_qc = int(os.environ.get("INDEXER_SWEEP_QC", "0"))
    sweep_kc = int(os.environ.get("INDEXER_SWEEP_KC", "0"))
    if sweep_qc or sweep_kc:
        return ttnn.IndexerScoreProgramConfig(
            q_chunk_size=(sweep_qc if sweep_qc else (2 if heads <= 16 else 1)) * 32,
            k_chunk_size=(sweep_kc if sweep_kc else 8) * 32,
            head_group_size=0,
        )
    return ttnn.IndexerScoreProgramConfig(
        q_chunk_size=64 if heads <= 16 else 32,
        k_chunk_size=256,
        head_group_size=0,
    )


@pytest.mark.parametrize("heads", GLX_HEAD_CASES, ids=["heads16", "heads64"])
@pytest.mark.parametrize("sp_rank", [0, 7], ids=["rank0", "rank7"])
def test_indexer_score_production(device, sp_rank, heads):
    """GLX chunked prefill with the production knobs (64-head whole indexer, 16-head TP=4 shard)."""
    chunk_start = GLX_HISTORY + sp_rank * GLX_SQ
    q, k, w = make_inputs(heads, GLX_DIM, GLX_SQ, GLX_T)
    out = run_indexer(q, k, w, chunk_start, device, program_config=production_config(heads))
    ref = indexer_score_ref(q, k, w, chunk_start)
    assert_indexer_match(out, ref, GLX_SQ, GLX_T)


@pytest.mark.parametrize("heads", GLX_HEAD_CASES, ids=["heads16", "heads64"])
@pytest.mark.parametrize("sp_rank", [0, 7], ids=["rank0", "rank7"])
def test_indexer_score_bfp8_k(device, sp_rank, heads):
    """k as bfp8_b (matmul srcA): halves k BW and selects LoFi in the factory.

    PCC stays >= 0.999 (the bfp8 quantization of well-conditioned k is well below the
    bf16 sum's noise); negative gates keep -inf padding distinguishable from low scores.
    """
    chunk_start = GLX_HISTORY + sp_rank * GLX_SQ
    q, k, w = make_inputs(heads, GLX_DIM, GLX_SQ, GLX_T)
    out = run_indexer(q, k, w, chunk_start, device, program_config=production_config(heads), k_dtype=ttnn.bfloat8_b)
    ref = indexer_score_ref(q, k, w, chunk_start)
    assert_indexer_match(out, ref, GLX_SQ, GLX_T, check_neg=True)


@pytest.mark.parametrize("heads", GLX_HEAD_CASES, ids=["heads16", "heads64"])
@pytest.mark.parametrize("k_dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["k_bf16", "k_bfp8"])
@pytest.mark.parametrize("sp_rank", [0, 7], ids=["rank0", "rank7"])
def test_indexer_score_production_perf(device, sp_rank, k_dtype, heads):
    """Absolute wall-clock latency per op for the production GLX shape (boundary SP ranks).

    Runs both the 64-head whole indexer and the 16-head TP=4 shard. Inputs are placed on
    device once (host transfer excluded) and the op is run program-cache-warm with a device
    sync around a fixed iteration count. This is host-dispatched single-op latency: it
    includes enqueue overhead, not pure device-kernel time (use the tracy device profiler
    for that). The logged ms is the signal; the assert is only a coarse hang / gross-
    regression guard (absolute wall-clock thresholds are board-dependent).
    """
    warmup_iters, measured_iters = 3, 20
    chunk_start = GLX_HISTORY + sp_rank * GLX_SQ
    cfg = production_config(heads)
    q, k, w = make_inputs(heads, GLX_DIM, GLX_SQ, GLX_T)
    q_dev = to_device(q, device)
    k_dev = to_device(k, device, dtype=k_dtype)  # bfp8_b k selects LoFi in the factory
    w_dev = to_device(w, device)

    def run_once():
        return ttnn.experimental.deepseek.indexer_score(
            q_dev, k_dev, w_dev, chunk_start_idx=chunk_start, program_config=cfg
        )

    for _ in range(warmup_iters):  # compile + program-cache warm
        run_once().deallocate()
    ttnn.synchronize_device(device)

    start = time.perf_counter()
    for _ in range(measured_iters):
        run_once().deallocate()
    ttnn.synchronize_device(device)
    ms_per_op = (time.perf_counter() - start) / measured_iters * 1e3

    k_tag = "bf16" if k_dtype == ttnn.bfloat16 else "bfp8"
    logger.info(
        f"indexer_score production rank{sp_rank} heads={heads} k={k_tag}: "
        f"{ms_per_op:.3f} ms/op (mean of {measured_iters} iters)"
    )
    assert (
        ms_per_op < 50.0
    ), f"rank{sp_rank} heads={heads} k={k_tag}: {ms_per_op:.3f} ms/op exceeds 50 ms guard (regression or hang)"


@pytest.mark.parametrize(
    "q_chunk, k_chunk, head_group, chunk_start",
    [
        (32, 32, 1, 128),  # default: one head resident, streamed per tile
        (64, 32, 16, 128),  # QC=2: multi-row groups, full-future tiles masked in compute (HB=16 of
        # 64 resident: QC=2 with all 64 heads would overflow L1, and the factory no longer clamps)
        (32, 128, 0, 128),  # KC=4: chunked k, partial edge chunks
        (32, 32, 32, 128),  # head streaming, 2 groups
        (64, 128, 16, 128),  # everything at once
        (64, 128, 16, 160),  # diagonal mid-group + partial edge, chunk_start not a k_chunk multiple
    ],
    ids=["hb1", "qc2", "kc4", "hb32", "all_at_once", "diag_mid_group"],
)
def test_indexer_score_knobs(device, q_chunk, k_chunk, head_group, chunk_start):
    heads, dim, sq, t = MINI["heads"], MINI["dim"], MINI["sq"], MINI["t"]
    q, k, w = make_inputs(heads, dim, sq, t)
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=q_chunk, k_chunk_size=k_chunk, head_group_size=head_group)
    out = run_indexer(q, k, w, chunk_start, device, program_config=cfg)
    ref = indexer_score_ref(q, k, w, chunk_start)
    assert_indexer_match(out, ref, sq, t, check_neg=True)


@pytest.mark.parametrize(
    "heads, dim, sq, t, chunk_start, head_group",
    [
        (64, 128, 128, 128, 0, 0),  # pure prefill, no history: square fully-causal triangle from tile 0
        (64, 128, 32, 32, 0, 0),  # minimal: single k-tile (Tt=1), in-tile diagonal only
        (64, 128, 64, 256, 128, 8),  # non-trivial head-group divisor (8 streamed groups of 8)
        (64, 128, 32, 256, 0, 1),  # first q-tile-row of a chunk: only 1 valid k-tile, long -inf tail
    ],
    ids=["prefill_square", "single_ktile", "head_group8", "row0_long_tail"],
)
def test_indexer_score_corner_shapes(device, heads, dim, sq, t, chunk_start, head_group):
    q, k, w = make_inputs(heads, dim, sq, t)
    cfg = ttnn.IndexerScoreProgramConfig(head_group_size=head_group)
    out = run_indexer(q, k, w, chunk_start, device, program_config=cfg)
    ref = indexer_score_ref(q, k, w, chunk_start)
    assert_indexer_match(out, ref, sq, t, check_neg=True)


def test_indexer_score_multicore_qc2(device):
    """QC=2 over a shape large enough that a q-row-group splits across many cores.

    Exercises the writer's group-level -inf tail fill (every row tail starts at the
    group's valid width) when the tail owner and the cores producing the group's
    future tiles are different cores.

    16 heads (not 64): QC=2 with all heads resident must fit L1 (cb_q ~ HB*QC*Dt), and
    64 heads * QC=2 overflows. The factory no longer clamps QC down, so the caller picks a
    head count where QC=2 actually takes effect rather than silently falling back to QC=1.
    """
    heads, dim, sq, t, chunk_start = 16, 128, 128, 2048, 512
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=64, k_chunk_size=32, head_group_size=0)
    q, k, w = make_inputs(heads, dim, sq, t)
    out = run_indexer(q, k, w, chunk_start, device, program_config=cfg)
    ref = indexer_score_ref(q, k, w, chunk_start)
    assert_indexer_match(out, ref, sq, t, check_neg=True)


@pytest.mark.parametrize(
    "is_causal, chunk_start, cfg",
    [
        (True, 100, None),  # chunk_start not tile-aligned
        (True, 256, None),  # chunk window [256, 320) exceeds T=256
        (False, 128, None),  # non-causal not implemented
        (True, 128, ttnn.IndexerScoreProgramConfig(q_chunk_size=96)),  # QC=3 does not divide Sqt=2
        (True, 128, ttnn.IndexerScoreProgramConfig(k_chunk_size=512)),  # KC=16 > Tt=8
        (True, 128, ttnn.IndexerScoreProgramConfig(head_group_size=3)),  # 3 does not divide Hi=64
        # QC=2 with all 64 heads resident overflows L1 (cb_q ~1 MB); the factory rejects it up front
        # instead of clamping QC down (the caller owns the knobs).
        (True, 128, ttnn.IndexerScoreProgramConfig(q_chunk_size=64, head_group_size=0)),
    ],
    ids=[
        "chunk_unaligned",
        "window_overflow",
        "non_causal",
        "qc_indivisible",
        "kc_oversize",
        "hb_indivisible",
        "l1_overflow",
    ],
)
def test_indexer_score_invalid_config(device, is_causal, chunk_start, cfg):
    """Host-side validation should reject bad inputs before any kernel launches."""
    q, k, w = make_inputs(**MINI)
    kwargs = {} if cfg is None else {"program_config": cfg}
    with pytest.raises(RuntimeError):
        ttnn.experimental.deepseek.indexer_score(
            to_device(q, device),
            to_device(k, device),
            to_device(w, device),
            is_causal=is_causal,
            chunk_start_idx=chunk_start,
            **kwargs,
        )


@pytest.mark.parametrize(
    "bad",
    ["dtype", "layout"],
)
def test_indexer_score_rejects_bad_inputs(device, bad):
    """q/k/weights must be bf16 TILE tensors; anything else corrupts the bf16 kernels."""
    heads, dim, sq, t = MINI["heads"], MINI["dim"], MINI["sq"], MINI["t"]
    q = torch.randn(1, heads, sq, dim)  # fp32
    k = torch.randn(1, 1, t, dim)
    w = torch.randn(1, heads, sq, 1)
    if bad == "dtype":
        q_dev = to_device(q, device, dtype=ttnn.float32)
        k_dev, w_dev = to_device(k, device), to_device(w, device)
    else:  # row-major layout
        q_dev = to_device(q, device, layout=ttnn.ROW_MAJOR_LAYOUT)
        k_dev, w_dev = to_device(k, device), to_device(w, device)
    with pytest.raises(RuntimeError):
        ttnn.experimental.deepseek.indexer_score(q_dev, k_dev, w_dev, chunk_start_idx=128)


# ---------------------------------------------------------------------------
# sp_rank 7 math-utilization perf check (tracy device profiler; no accuracy check)
#
# math_util = matmul FLOPs the kernel performs / (cores x device cycles x matmul peak).
# Device kernel duration comes from tracy; theoretical FLOPs are computed from the shape.
# Run locally with a profiler-enabled build (build_metal.sh --enable-profiler):
#   pytest tests/nightly/blackhole/sdpa/test_indexer_score.py::test_indexer_score_sp7_math_util
# ---------------------------------------------------------------------------
SP7_CHUNK_START = GLX_HISTORY + 7 * GLX_SQ  # fullest causal case (99.5% valid)

# Blackhole matmul peak (tests/nightly/sdpa_perf_utils.py): 4096 mm FLOP/cycle/core at LoFi,
# halved per extra math-fidelity phase. The `fidelity` param below must match the factory's
# choice for the measured k dtype (currently HiFi2).
_BH_CLOCK_GHZ = 1.35
_MM_FLOPS_PER_CYCLE_PER_CORE = {"LoFi": 4096, "HiFi2": 2048, "HiFi3": 1365, "HiFi4": 1024}


def sp7_valid_tiles():
    """Causal-valid output tiles V at sp_rank 7 = sum_s min(Tt, chunk_t + s + 1) over q-tile-rows."""
    chunk_t = SP7_CHUNK_START // 32
    tt_tiles = GLX_T // 32
    sqt = GLX_SQ // 32
    return sum(min(tt_tiles, chunk_t + s + 1) for s in range(sqt))


def indexer_mm_flops(valid_tiles, heads):
    """Matmul FLOPs the kernel performs: each valid 32x32 output tile is, per head, a 32x32xD
    matmul = (32*32) elements x (2*D) FLOPs; summed over heads and valid tiles."""
    return valid_tiles * heads * (32 * 32) * (2 * GLX_DIM)


# (heads, k_id, fidelity) cases for the sp7 profiler tests. fidelity must match the factory
# choice for the k dtype (HiFi2 today). The id is shared by perf_impl (the profiled inner test)
# and math_util (which spawns perf_impl by id), so they must stay in lockstep.
_SP7_PERF_CASES = [
    (8, "k_bfp8", "HiFi2"),
    (16, "k_bf16", "HiFi2"),
    (16, "k_bfp8", "HiFi2"),
    (64, "k_bf16", "HiFi2"),
    (64, "k_bfp8", "HiFi2"),
]
_SP7_PERF_IDS = [f"heads{h}_{k}" for h, k, _ in _SP7_PERF_CASES]


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="perf test - run locally with tracy")
@pytest.mark.parametrize("heads, k_id", [(h, k) for h, k, _ in _SP7_PERF_CASES], ids=_SP7_PERF_IDS)
def test_indexer_score_sp7_perf_impl(device, heads, k_id):
    """Inner test profiled by tracy: a few indexer_score ops at GLX sp_rank 7. No accuracy check."""
    k_dtype = ttnn.bfloat16 if k_id == "k_bf16" else ttnn.bfloat8_b
    q, k, w = make_inputs(heads, GLX_DIM, GLX_SQ, GLX_T)
    q_dev = to_device(q, device)
    k_dev = to_device(k, device, dtype=k_dtype)
    w_dev = to_device(w, device)
    cfg = production_config(heads)
    for _ in range(5):  # tracy logs each op's device duration; the outer test takes the min
        ttnn.experimental.deepseek.indexer_score(
            q_dev, k_dev, w_dev, chunk_start_idx=SP7_CHUNK_START, program_config=cfg
        ).deallocate()
    ttnn.synchronize_device(device)


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="perf test - run locally with tracy")
@pytest.mark.parametrize("heads, k_id, fidelity", _SP7_PERF_CASES, ids=_SP7_PERF_IDS)
def test_indexer_score_sp7_math_util(heads, k_id, fidelity):
    """sp_rank 7 matmul math utilization from a tracy device-kernel-duration measurement.

    Spawns the inner perf_impl test under the device profiler, reads the minimum
    DEVICE KERNEL DURATION, and reports math_util = mm_flops / (cores * cycles * peak).
    Runs the 64-head whole indexer and the 16-head TP=4 shard. No accuracy check.
    """
    from tracy.process_model_log import run_device_profiler
    from tests.nightly.sdpa_perf_utils import post_process_ops_log

    subdir = "ttnn_indexer_score_sp7"
    perf_id = f"heads{heads}_{k_id}"
    command = (
        "pytest tests/nightly/blackhole/sdpa/test_indexer_score.py::" f"test_indexer_score_sp7_perf_impl[{perf_id}]"
    )
    with mock.patch.dict(os.environ, {"CI": "false"}):
        run_device_profiler(command, subdir, device_analysis_types=["device_kernel_duration"])
    r = post_process_ops_log(
        subdir,
        float_columns=["CORE COUNT", "DEVICE KERNEL DURATION [ns]"],
        columns=["ATTRIBUTES"],
        sum_vals=False,
        has_signposts=False,
    )
    assert len(r["DEVICE KERNEL DURATION [ns]"]) > 0, "profiler returned no indexer_score ops"

    core_count = int(r["CORE COUNT"][0])
    duration_ns = float(r["DEVICE KERNEL DURATION [ns]"].min())

    valid_tiles = sp7_valid_tiles()
    mm_flops = indexer_mm_flops(valid_tiles, heads)
    peak = _MM_FLOPS_PER_CYCLE_PER_CORE[fidelity]
    cycles = duration_ns * _BH_CLOCK_GHZ
    theoretical_flops = core_count * cycles * peak
    math_util = (mm_flops / theoretical_flops) * 100 if theoretical_flops > 0 else 0.0

    logger.info(
        f"indexer_score sp7 heads={heads} {k_id} ({fidelity}): device={duration_ns / 1e6:.3f} ms, "
        f"cores={core_count}, V={valid_tiles} tiles, mm={mm_flops / 1e9:.1f} GFLOP, "
        f"peak={peak} FLOP/cyc/core @ {_BH_CLOCK_GHZ} GHz -> math_util={math_util:.1f}%"
    )
