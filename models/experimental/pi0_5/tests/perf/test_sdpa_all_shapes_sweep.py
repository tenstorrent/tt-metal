# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""SDPA config sweep across ALL three SDPA shapes in pi0.5.

Production shapes (from tracy of test_perf_ttnn_full_e2e.py):

  denoise  Sq=32   Skv=544  hd=256  num_heads=8   num_kv_heads=1   180 calls/inf
  vlm      Sq=512  Skv=512  hd=256  num_heads=8   num_kv_heads=1   18 calls
  siglip   Sq=256  Skv=256  hd=96   num_heads=16  num_kv_heads=16  27 calls

Sweeps (per shape):
  q_chunk_size               {32, 64, 128, 256, 512}  (capped at Sq)
  k_chunk_size               {32, 64, 128, 256, 512, Skv if not power of 2}
  max_cores_per_head_batch   {4, 8, 12, 16, 24, 32}
  exp_approx_mode            {False, True}
  fp32_dest_acc_en           {False, True}
  packer_l1_acc              {True, False}
  math_fidelity              {HiFi2, HiFi4}

PCC against torch.nn.functional.scaled_dot_product_attention.

Run a single shape:
    PI0_SDPA_ALL=1 PI0_SDPA_SHAPE=vlm pytest -xvs \\
      models/experimental/pi0_5/tests/perf/test_sdpa_all_shapes_sweep.py

Or all 3 in one pass:
    PI0_SDPA_ALL=1 pytest -xvs \\
      models/experimental/pi0_5/tests/perf/test_sdpa_all_shapes_sweep.py
"""

from __future__ import annotations

import os
import statistics
import time
from dataclasses import dataclass
from typing import List, Tuple

import pytest
import torch
import ttnn


BENCH_ENABLED = os.environ.get("PI0_SDPA_ALL") == "1"
pytestmark = pytest.mark.skipif(not BENCH_ENABLED, reason="set PI0_SDPA_ALL=1 to run the all-shapes SDPA sweep")

NUM_WARMUP = int(os.environ.get("PI0_SDPA_ALL_WARMUP", "10"))
NUM_ITER = int(os.environ.get("PI0_SDPA_ALL_ITER", "25"))
PCC_THRESHOLD = float(os.environ.get("PI0_SDPA_ALL_PCC", "0.999"))
SHAPE_FILTER = os.environ.get("PI0_SDPA_SHAPE", "").lower()


@dataclass(frozen=True)
class ShapeSpec:
    label: str
    batch: int
    num_heads: int
    num_kv_heads: int
    s_q: int
    s_kv: int
    head_dim: int


SHAPES: List[ShapeSpec] = [
    ShapeSpec("denoise", 1, 8, 1, 32, 544, 256),
    ShapeSpec("vlm", 1, 8, 1, 512, 512, 256),
    ShapeSpec("siglip", 1, 16, 16, 256, 256, 96),
]


@dataclass
class CandConfig:
    q_chunk: int
    k_chunk: int
    max_cores_per_head_batch: int
    exp_approx: bool
    fp32_dest_acc: bool
    packer_l1_acc: bool
    math_fidelity: "ttnn.MathFidelity"


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    t1 = a.flatten().float()
    t2 = b.flatten().float()
    if t1.numel() != t2.numel():
        return -1.0
    m1, m2 = torch.mean(t1), torch.mean(t2)
    s1, s2 = torch.std(t1), torch.std(t2)
    if s1 < 1e-9 or s2 < 1e-9:
        return 1.0 if torch.allclose(t1, t2, atol=1e-5) else 0.0
    return (torch.mean((t1 - m1) * (t2 - m2)) / (s1 * s2)).item()


def _torch_ref(q, k, v, mask, scale):
    if k.shape[1] != q.shape[1]:
        repeat = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
    return torch.nn.functional.scaled_dot_product_attention(
        q.float(),
        k.float(),
        v.float(),
        attn_mask=mask.float(),
        scale=scale,
    )


def _build_inputs(spec: ShapeSpec, device):
    torch.manual_seed(0)
    q = torch.randn(spec.batch, spec.num_heads, spec.s_q, spec.head_dim) * 0.1
    k = torch.randn(spec.batch, spec.num_kv_heads, spec.s_kv, spec.head_dim) * 0.1
    v = torch.randn(spec.batch, spec.num_kv_heads, spec.s_kv, spec.head_dim) * 0.1
    mask = torch.zeros(spec.batch, 1, spec.s_q, spec.s_kv)
    scale = 1.0 / (spec.head_dim**0.5)

    def _upload(t, mc):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)

    return (
        q,
        k,
        v,
        mask,
        scale,
        _upload(q, ttnn.L1_MEMORY_CONFIG),
        _upload(k, ttnn.L1_MEMORY_CONFIG),
        _upload(v, ttnn.L1_MEMORY_CONFIG),
        _upload(mask, ttnn.DRAM_MEMORY_CONFIG),
    )


def _run(q_tt, k_tt, v_tt, mask_tt, scale, c, grid):
    sdpa_cfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid,
        q_chunk_size=c.q_chunk,
        k_chunk_size=c.k_chunk,
        exp_approx_mode=c.exp_approx,
        max_cores_per_head_batch=c.max_cores_per_head_batch,
    )
    ck_cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=c.math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=c.fp32_dest_acc,
        packer_l1_acc=c.packer_l1_acc,
    )
    return ttnn.transformer.scaled_dot_product_attention(
        q_tt,
        k_tt,
        v_tt,
        attn_mask=mask_tt,
        is_causal=False,
        scale=scale,
        program_config=sdpa_cfg,
        compute_kernel_config=ck_cfg,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )


def _time(device, q_tt, k_tt, v_tt, mask_tt, scale, c, grid) -> Tuple[float, float]:
    for _ in range(NUM_WARMUP):
        out = _run(q_tt, k_tt, v_tt, mask_tt, scale, c, grid)
        ttnn.deallocate(out)
    ttnn.synchronize_device(device)
    samples = []
    for _ in range(NUM_ITER):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        out = _run(q_tt, k_tt, v_tt, mask_tt, scale, c, grid)
        ttnn.synchronize_device(device)
        samples.append((time.perf_counter() - t0) * 1000)
        ttnn.deallocate(out)
    return statistics.mean(samples) * 1000, min(samples) * 1000  # µs


def _build_candidates(spec: ShapeSpec) -> List[CandConfig]:
    """Build candidates respecting tile alignment and divisibility."""
    cands: List[CandConfig] = []
    # q_chunk must be ≤ Sq tile-aligned multiple, power of 2 OR exactly Sq
    q_aligned = ((spec.s_q + 31) // 32) * 32
    k_aligned = ((spec.s_kv + 31) // 32) * 32
    q_chunks = [c for c in (32, 64, 128, 256, 512) if c <= q_aligned]
    if q_aligned not in q_chunks:
        q_chunks.append(q_aligned)
    k_chunks = [c for c in (32, 64, 128, 256, 512) if c <= k_aligned]
    if k_aligned not in k_chunks:
        k_chunks.append(k_aligned)

    for q_chunk in q_chunks:
        for k_chunk in k_chunks:
            for max_cores in (4, 8, 16, 24, 32):
                for exp_approx in (False, True):
                    for fp32_dest in (False, True):
                        for fid in (ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4):
                            cands.append(
                                CandConfig(
                                    q_chunk=q_chunk,
                                    k_chunk=k_chunk,
                                    max_cores_per_head_batch=max_cores,
                                    exp_approx=exp_approx,
                                    fp32_dest_acc=fp32_dest,
                                    packer_l1_acc=True,  # keep packer_l1 fixed — sweep v1 showed no effect
                                    math_fidelity=fid,
                                )
                            )
    return cands


def _sweep_one(device, spec: ShapeSpec):
    print(f"\n{'=' * 80}")
    print(
        f"  SDPA SWEEP: {spec.label}  B={spec.batch} H={spec.num_heads} KVh={spec.num_kv_heads} "
        f"Sq={spec.s_q} Skv={spec.s_kv} hd={spec.head_dim}"
    )
    print(f"{'=' * 80}")

    grid = device.compute_with_storage_grid_size()
    q, k, v, mask, scale, q_tt, k_tt, v_tt, mask_tt = _build_inputs(spec, device)
    ref = _torch_ref(q, k, v, mask, scale)

    # Baseline = production picker's choice (divisor-aware)
    from models.experimental.pi0_5.tt.ttnn_common import sdpa_prefill_chunk_sizes

    q_chunk, k_chunk = sdpa_prefill_chunk_sizes(spec.s_q, spec.s_kv)
    base_c = CandConfig(
        q_chunk=q_chunk,
        k_chunk=k_chunk,
        max_cores_per_head_batch=16,
        exp_approx=False,
        fp32_dest_acc=True,
        packer_l1_acc=True,
        math_fidelity=ttnn.MathFidelity.HiFi2,
    )
    try:
        ref_out = ttnn.to_torch(_run(q_tt, k_tt, v_tt, mask_tt, scale, base_c, grid))
        base_pcc = _pcc(ref, ref_out.reshape(ref.shape))
        # Run the baseline TWICE — first time is JIT-cold, second is steady-state
        _ = _time(device, q_tt, k_tt, v_tt, mask_tt, scale, base_c, grid)  # warmup JIT
        base_mean, base_min = _time(device, q_tt, k_tt, v_tt, mask_tt, scale, base_c, grid)
        print(f"  Baseline (picker pick): q={q_chunk} k={k_chunk} max_cores=16 exp=0 fp32=1 HiFi2")
        print(f"    steady-state: mean={base_mean:.2f} µs  min={base_min:.2f}  pcc={base_pcc:.5f}")
    except RuntimeError as e:
        print(f"  Baseline FAILED: {str(e).splitlines()[0][:120]}")
        return

    cands = _build_candidates(spec)
    print(f"  Trying {len(cands)} configs ...")
    results: List[Tuple[CandConfig, float, float, float]] = []
    for c in cands:
        try:
            out = _run(q_tt, k_tt, v_tt, mask_tt, scale, c, grid)
            out_torch = ttnn.to_torch(out)
            ttnn.deallocate(out)
            pcc = _pcc(ref, out_torch.reshape(ref.shape))
        except RuntimeError:
            continue
        try:
            mean, mn = _time(device, q_tt, k_tt, v_tt, mask_tt, scale, c, grid)
        except RuntimeError:
            continue
        results.append((c, mean, mn, pcc))

    ttnn.deallocate(q_tt)
    ttnn.deallocate(k_tt)
    ttnn.deallocate(v_tt)
    ttnn.deallocate(mask_tt)

    good = [r for r in results if r[3] >= PCC_THRESHOLD]
    good.sort(key=lambda r: r[1])

    print(f"  Tried: {len(results)}   Passing PCC≥{PCC_THRESHOLD}: {len(good)}")
    print()
    print(
        f"  {'rank':<4}  {'mean µs':>8}  {'min µs':>8}  {'pcc':>8}  {'vs base':>8}  {'q':>4}  {'k':>4}  {'cores':>5}  {'exp':>3}  {'fp32':>4}  {'fid':>5}"
    )
    for rank, (c, mean, mn, pcc) in enumerate(good[:15], 1):
        fid = "HiFi2" if c.math_fidelity == ttnn.MathFidelity.HiFi2 else "HiFi4"
        delta = f"{(mean - base_mean):+6.2f}"
        print(
            f"  {rank:<4}  {mean:>8.2f}  {mn:>8.2f}  {pcc:>8.5f}  {delta:>8}  {c.q_chunk:>4}  {c.k_chunk:>4}  {c.max_cores_per_head_batch:>5}  {int(c.exp_approx):>3}  {int(c.fp32_dest_acc):>4}  {fid:>5}"
        )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_sdpa_all_shapes_sweep(device):
    print(f"\nSDPA sweep across all 3 production shapes (PCC≥{PCC_THRESHOLD})")
    if SHAPE_FILTER:
        print(f"shape filter: {SHAPE_FILTER}")
    for spec in SHAPES:
        if SHAPE_FILTER and SHAPE_FILTER != spec.label:
            continue
        _sweep_one(device, spec)
