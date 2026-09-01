# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""N300 matmul program-config + sharding sweep for the Talker prefill layer (seq 64 / 128).

Both Talker prefill buckets currently run every projection on ``program_config=None``
(auto-routing) with interleaved activations — ``attention.py`` and ``mlp.py`` only build
explicit configs for decode and for ``seq_len <= short_seq_limit`` (32). The profile shows
the cost of that: at seq=128 the five matmuls are 378 us of an 840 us window at 36-57 %
of DRAM peak and 6-12 % of FLOP peak, i.e. neither bandwidth- nor math-bound — just
badly blocked.

This sweep searches program-config family x sharding strategy x ``in0_block_w`` x subblock
x ``fp32_dest_acc_en`` for each shape and reports what beats the production baseline.

Two harness rules this file must never break (they invalidate every result):
  * ``packer_l1_acc=True`` always — False reports matmuls ~3.5x slower than reality.
  * the core grid comes from ``device.compute_with_storage_grid_size()``, never a literal.

And one sanity gate: the measured baseline must match the in-model Tracy number for the
same op. If it does not, the harness is wrong, not the production config.

    # full sweep (~20 min)
    pytest -s models/demos/qwen3_tts/tests/test_qwen3_tts_matmul_sweep_n300.py

    # one shape
    pytest -s models/demos/qwen3_tts/tests/test_qwen3_tts_matmul_sweep_n300.py -k s128_down

    QWEN3_TTS_SWEEP_REPS=64      # matmul enqueues per timed round
    QWEN3_TTS_SWEEP_ROUNDS=3     # timed rounds; the median is reported
    QWEN3_TTS_SWEEP_OUT=path.md  # append a results table here
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import pytest
import torch

import ttnn

TILE = 32
# Reps captured inside one Metal trace, and how many times that trace is replayed per
# timed round. Timing MUST go through a trace: measured eagerly, a 29 us matmul reads as
# 88 us because host dispatch, not the device, is the bottleneck at these sizes.
REPS = int(os.environ.get("QWEN3_TTS_SWEEP_REPS", "16"))
REPLAYS = int(os.environ.get("QWEN3_TTS_SWEEP_REPLAYS", "20"))
ROUNDS = int(os.environ.get("QWEN3_TTS_SWEEP_ROUNDS", "3"))
_TRACE_REGION = 100_000_000
RESULTS_PATH = os.environ.get("QWEN3_TTS_SWEEP_OUT", "")
# Every candidate, not just the printed top-12 — needed to attribute a win to the
# program config vs fp32_dest_acc_en vs the sharding, one lever at a time.
CSV_PATH = os.environ.get("QWEN3_TTS_SWEEP_CSV", "")

# PCC floor for a candidate to be considered at all. fp32_dest_acc_en=False accumulates in
# fp16, so it is a real numerical change and must clear this before it is ever proposed.
PCC_FLOOR = 0.999

# A candidate must beat the baseline by more than this to be worth wiring in. Run-to-run
# spread on these shapes is ~2-3 %, so anything under 5 % is not a result.
WIN_THRESHOLD_PCT = 5.0


# Talker prefill matmuls, per-chip dims at TP=2 (hidden 2048, local fused QKV 2048, local
# head width 1024, local intermediate 3072). `baseline_us` is the in-model Tracy device time
# from qwen3_tts_n300_blocks_opt.txt and is used only as the harness sanity gate.
#
#   name          M    K     N     baseline_us  source op
@dataclass(frozen=True)
class Shape:
    name: str
    m: int
    k: int
    n: int
    baseline_us: float
    note: str


SHAPES = [
    Shape("s64_qkv", 64, 2048, 2048, 56.0, "attention wqkv"),
    Shape("s64_wo", 64, 1024, 2048, 29.0, "attention o_proj"),
    Shape("s64_gate_up", 64, 2048, 3072, 66.0, "mlp gate / up"),
    Shape("s64_down", 64, 3072, 2048, 82.0, "mlp down"),
    Shape("s128_qkv", 128, 2048, 2048, 76.0, "attention wqkv"),
    Shape("s128_wo", 128, 1024, 2048, 40.0, "attention o_proj"),
    Shape("s128_gate_up", 128, 2048, 3072, 76.0, "mlp gate / up"),
    Shape("s128_down", 128, 3072, 2048, 110.0, "mlp down"),
]


# --------------------------------------------------------------------------------------
# device
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def device():
    """A real 1x2 wormhole mesh — this sweep is N300-only by construction.

    Sweeping on a 1x1 mesh would tune against the wrong per-chip N and produce configs that
    do not apply, so the fixture opens the mesh itself rather than trusting MESH_DEVICE.
    """
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    d = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 2), l1_small_size=32768, trace_region_size=_TRACE_REGION)
    d.enable_program_cache()
    yield d
    ttnn.close_mesh_device(d)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _require_n300(device):
    from models.demos.qwen3_tts.tt.mesh_utils import is_n300

    if not is_n300(device):
        pytest.skip("N300-only sweep (needs a 2-chip wormhole mesh)")


# --------------------------------------------------------------------------------------
# candidate description
# --------------------------------------------------------------------------------------
@dataclass
class Candidate:
    label: str
    family: str
    program_config: Optional[object]
    in0_memcfg: object
    out_memcfg: object
    fp32_dest_acc_en: bool
    extra: dict = field(default_factory=dict)


def _compute_cfg(fp32_dest_acc_en: bool):
    # LoFi + packer_l1_acc=True mirror production. packer_l1_acc must never be False here.
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=True,
    )


def _divisors(n: int):
    return [i for i in range(1, n + 1) if n % i == 0]


def _subblocks(per_core_m: int, per_core_n: int, cap: int):
    """Legal (h, w) with h*w <= cap, h | per_core_M, w | per_core_N. Widest first."""
    out = []
    for h in range(1, min(cap, per_core_m) + 1):
        if per_core_m % h:
            continue
        for w in range(1, min(cap // h, per_core_n) + 1):
            if per_core_n % w == 0:
                out.append((h, w))
    out.sort(key=lambda hw: -(hw[0] * hw[1]))
    return out[:4]


def _block_sharded(m, k, gx, gy):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (m // gy, k // gx), ttnn.ShardOrientation.ROW_MAJOR),
    )


def _width_sharded(m, k, num_cores, gx):
    rows = math.ceil(num_cores / gx)
    cols = min(num_cores, gx)
    if rows * cols != num_cores:
        return None
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cols - 1, rows - 1))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (m, k // num_cores), ttnn.ShardOrientation.ROW_MAJOR),
    )


def build_candidates(s: Shape, gx: int, gy: int) -> list[Candidate]:
    """Every family x sharding x blocking combination worth trying for this shape."""
    m_t, k_t, n_t = s.m // TILE, s.k // TILE, s.n // TILE
    L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
    cands: list[Candidate] = []

    # ---- 0. production baseline: auto-routing, L1 interleaved, fp32 accumulate ----
    cands.append(Candidate("BASELINE auto/L1/fp32acc", "auto", None, L1, L1, True))

    # ---- 1. auto-routing variants: the zero-risk knobs ----
    cands.append(Candidate("auto/L1/fp16acc", "auto", None, L1, L1, False))
    cands.append(Candidate("auto/DRAM/fp32acc", "auto", None, DRAM, DRAM, True))
    cands.append(Candidate("auto/DRAM/fp16acc", "auto", None, DRAM, DRAM, False))

    for fp32 in (True, False):
        cap = 4 if fp32 else 8
        tag = "fp32acc" if fp32 else "fp16acc"

        # ---- 2. 1D mcast, mcast_in0=True (K split across cores; the repo's decode form) ----
        for ncores in sorted({gx * gy, gx * gy // 2, gx * 4, gx * 2}, reverse=True):
            if ncores < 1 or ncores > gx * gy:
                continue
            per_core_n = math.ceil(n_t / ncores)
            per_core_k = math.ceil(k_t / ncores)
            if per_core_n < 1 or per_core_k < 1:
                continue
            for h, w in _subblocks(m_t, per_core_n, cap):
                cands.append(
                    Candidate(
                        f"1D mcast_in0 c{ncores} ibw{per_core_k} sb{h}x{w}/{tag}",
                        "1d_mcast_in0",
                        ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                            compute_with_storage_grid_size=(gx, gy),
                            in0_block_w=per_core_k,
                            out_subblock_h=h,
                            out_subblock_w=w,
                            per_core_M=m_t,
                            per_core_N=per_core_n,
                            fuse_batch=True,
                            fused_activation=None,
                            mcast_in0=True,
                        ),
                        L1,
                        L1,
                        fp32,
                    )
                )

        # ---- 3. 2D mcast block-sharded. gy must divide M_tiles, which is only 2 or 4 here,
        #         so the grid is short and wide — the opposite of the usual encoder shape.
        for g_y in [g for g in _divisors(m_t) if g <= gy]:
            per_core_m = m_t // g_y
            for g_x in (gx,):
                if g_x < 1 or k_t % g_x or n_t % g_x:
                    continue
                per_core_n = n_t // g_x
                for ibw in [d for d in _divisors(k_t // g_x) if d in (2, 8)]:
                    for h, w in _subblocks(per_core_m, per_core_n, cap)[:2]:
                        for tm in (False,):
                            cands.append(
                                Candidate(
                                    f"2D block {g_x}x{g_y} ibw{ibw} sb{h}x{w} tm{int(tm)}/{tag}",
                                    "2d_block",
                                    ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                                        compute_with_storage_grid_size=(g_x, g_y),
                                        in0_block_w=ibw,
                                        out_subblock_h=h,
                                        out_subblock_w=w,
                                        per_core_M=per_core_m,
                                        per_core_N=per_core_n,
                                        transpose_mcast=tm,
                                        fused_activation=None,
                                    ),
                                    _block_sharded(s.m, s.k, g_x, g_y),
                                    L1,
                                    fp32,
                                )
                            )

        # ---- 4. 1D mcast with a width-sharded activation (explicit, not matmul-internal) ----
        for ncores in (gx * gy, gx * 4):
            if k_t % ncores:
                continue
            mc = _width_sharded(s.m, s.k, ncores, gx)
            if mc is None:
                continue
            per_core_n = math.ceil(n_t / ncores)
            for h, w in _subblocks(m_t, per_core_n, cap):
                cands.append(
                    Candidate(
                        f"1D wshard c{ncores} sb{h}x{w}/{tag}",
                        "1d_wshard",
                        ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                            compute_with_storage_grid_size=(gx, gy),
                            in0_block_w=k_t // ncores,
                            out_subblock_h=h,
                            out_subblock_w=w,
                            per_core_M=m_t,
                            per_core_N=per_core_n,
                            fuse_batch=True,
                            fused_activation=None,
                            mcast_in0=True,
                        ),
                        mc,
                        L1,
                        fp32,
                    )
                )

    return cands


# --------------------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------------------
def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    ac, bc = a - a.mean(), b - b.mean()
    d = ac.norm() * bc.norm()
    return 0.0 if d < 1e-12 else (ac * bc).sum().item() / d.item()


def _time_us(device, fn: Callable, reps: int, rounds: int) -> float:
    """us per matmul, measured by replaying a Metal trace. Median of `rounds` rounds.

    Eager enqueue-then-sync does NOT work here: host dispatch costs more per call than the
    matmul itself (a 29 us op measured 88 us), so every ranking it produces is host noise.
    Capturing `reps` back-to-back matmuls in one trace and replaying it removes dispatch and
    amortises the per-replay launch cost over `reps` ops.
    """
    warm = fn()  # compile + populate the program cache before capture
    ttnn.synchronize_device(device)
    ttnn.deallocate(warm)

    tid = ttnn.begin_trace_capture(device, cq_id=0)
    try:
        captured = [fn() for _ in range(reps)]
    finally:
        ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)

    samples = []
    try:
        for _ in range(rounds):
            t0 = time.perf_counter()
            for _ in range(REPLAYS):
                ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            t1 = time.perf_counter()
            samples.append((t1 - t0) / (REPLAYS * reps) * 1e6)
    finally:
        ttnn.release_trace(device, tid)
        for o in captured:
            ttnn.deallocate(o)
    samples.sort()
    return samples[len(samples) // 2]


def _run_candidate(device, s: Shape, c: Candidate, act_t, w_t, ref):
    """Returns (us, pcc) or raises. Caller catches unsupported configs."""
    act = ttnn.from_torch(
        act_t, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=c.in0_memcfg
    )
    w = ttnn.from_torch(
        w_t, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    kw = dict(
        memory_config=c.out_memcfg,
        compute_kernel_config=_compute_cfg(c.fp32_dest_acc_en),
        dtype=ttnn.bfloat16,
    )
    if c.program_config is not None:
        kw["program_config"] = c.program_config

    def fn():
        return ttnn.linear(act, w, **kw)

    out = fn()
    ttnn.synchronize_device(device)
    from models.demos.qwen3_tts.tt.mesh_utils import to_torch

    pcc = _pcc(ref, to_torch(out, device=device).float())
    ttnn.deallocate(out)

    us = _time_us(device, fn, REPS, ROUNDS)
    ttnn.deallocate(act)
    ttnn.deallocate(w)
    return us, pcc


def _emit(lines: list[str]):
    text = "\n".join(lines)
    print("\n" + text)
    if RESULTS_PATH:
        with open(RESULTS_PATH, "a") as f:
            f.write(text + "\n\n")


# --------------------------------------------------------------------------------------
# the sweep
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("shape", SHAPES, ids=[s.name for s in SHAPES])
def test_matmul_sweep(device, shape: Shape):
    _require_n300(device)
    grid = device.compute_with_storage_grid_size()
    gx, gy = grid.x, grid.y

    torch.manual_seed(0)
    act_t = torch.randn(1, 1, shape.m, shape.k, dtype=torch.bfloat16)
    w_t = torch.randn(1, 1, shape.k, shape.n, dtype=torch.bfloat16)
    ref = (act_t.float() @ w_t.float()).squeeze(0).squeeze(0)

    def _ref_cmp(t):
        return ref

    cands = build_candidates(shape, gx, gy)
    results = []
    baseline_us = None
    skipped = 0

    for c in cands:
        try:
            us, pcc = _run_candidate(device, shape, c, act_t, w_t, ref)
        except Exception as e:  # unsupported blocking / L1 clash — a normal sweep outcome
            skipped += 1
            if c.family == "auto" and c.label.startswith("BASELINE"):
                raise AssertionError(f"baseline itself failed: {e}") from e
            continue
        results.append((us, pcc, c))
        if c.label.startswith("BASELINE"):
            baseline_us = us

    assert baseline_us is not None, "baseline never ran"

    # --- harness sanity gate (methodology.md §4) ---------------------------------------
    # If the standalone baseline does not track the in-model Tracy number, the harness is
    # measuring something else and every "winner" below is noise.
    ratio = baseline_us / shape.baseline_us
    harness_ok = 0.7 <= ratio <= 1.6

    ok = [r for r in results if r[1] >= PCC_FLOOR]
    ok.sort(key=lambda r: r[0])

    lines = [
        f"### {shape.name}  ({shape.note})  M={shape.m} K={shape.k} N={shape.n}  grid {gx}x{gy}",
        "",
        f"harness baseline {baseline_us:.1f} us vs in-model Tracy {shape.baseline_us:.1f} us "
        f"-> ratio {ratio:.2f} {'OK' if harness_ok else 'SUSPECT — results not trustworthy'}",
        f"{len(results)} of {len(cands)} candidates ran ({skipped} rejected by the device), "
        f"{len(ok)} cleared PCC >= {PCC_FLOOR}",
        "",
        "| rank | us | vs baseline | PCC | family | config |",
        "|---:|---:|---:|---:|---|---|",
    ]
    for i, (us, pcc, c) in enumerate(ok[:12], 1):
        d = 100.0 * (us - baseline_us) / baseline_us
        lines.append(f"| {i} | {us:.1f} | {d:+.1f} % | {pcc:.5f} | {c.family} | {c.label} |")
    lines.append(f"| — | {baseline_us:.1f} | baseline | — | auto | BASELINE auto/L1/fp32acc |")

    if CSV_PATH:
        import csv

        new_file = not os.path.exists(CSV_PATH)
        with open(CSV_PATH, "a", newline="") as f:
            wcsv = csv.writer(f)
            if new_file:
                wcsv.writerow(
                    ["shape", "M", "K", "N", "us", "pcc", "family", "fp32_dest_acc_en", "label", "baseline_us"]
                )
            for us, pcc, c in sorted(results):
                wcsv.writerow(
                    [
                        shape.name,
                        shape.m,
                        shape.k,
                        shape.n,
                        f"{us:.2f}",
                        f"{pcc:.6f}",
                        c.family,
                        int(c.fp32_dest_acc_en),
                        c.label,
                        f"{baseline_us:.2f}",
                    ]
                )

    best_us, best_pcc, best_c = ok[0]
    win = 100.0 * (baseline_us - best_us) / baseline_us
    lines += ["", f"best: {best_c.label}  {best_us:.1f} us  ({win:+.1f} % vs baseline, PCC {best_pcc:.5f})"]
    _emit(lines)

    assert harness_ok, (
        f"{shape.name}: harness baseline {baseline_us:.1f} us is {ratio:.2f}x the in-model "
        f"{shape.baseline_us:.1f} us. Fix the harness before trusting any winner."
    )
    # "No config beats auto-routing here" is a real result, not a harness failure, so the
    # win assertion is opt-in: run with QWEN3_TTS_SWEEP_STRICT=1 once you expect a win.
    if os.environ.get("QWEN3_TTS_SWEEP_STRICT", "0") == "1":
        assert win > WIN_THRESHOLD_PCT, (
            f"{shape.name}: best candidate {best_c.label} is only {win:+.1f} % vs the "
            f"interleaved-L1 baseline — at or inside the noise floor."
        )
