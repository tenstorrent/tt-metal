# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Extended SDPA sweep at pi0.5 denoise shape (Sq=32, Skv=544, hd=256).

Wave 1 (test_denoise_sdpa_sweep.py) covered (q_chunk, k_chunk, exp_approx,
math_fidelity). This wave adds the missing knobs the user surfaced:

  - max_cores_per_head_batch  {4, 8, 12, 16, 24, 32}   default 16
  - exp_approx_mode           {False, True}             default False
  - fp32_dest_acc_en          {False, True}             default True (pi0.5)
  - packer_l1_acc             {False, True}             default True
  - math_fidelity             {HiFi2, HiFi4}            default HiFi2

PCC vs torch.nn.functional.scaled_dot_product_attention computed for
every candidate; configs falling below threshold (default 0.999) are
dropped from the leaderboard.

Fixed (chosen by the wave-1 winner):
  q_chunk_size = 32, k_chunk_size = 32   (divisor-aware pick for Skv=544)
  math_approx_mode = False                (off — matches pi0.5 default)

Run:
    PI0_SDPA2_SWEEP=1 pytest -xvs \\
      models/experimental/pi0_5/tests/perf/test_denoise_sdpa_sweep_v2.py
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


BENCH_ENABLED = os.environ.get("PI0_SDPA2_SWEEP") == "1"
pytestmark = pytest.mark.skipif(not BENCH_ENABLED, reason="set PI0_SDPA2_SWEEP=1 to run the extended SDPA sweep")

# pi0.5 expert denoise SDPA shape
BATCH = 1
NUM_HEADS = 8
NUM_KV_HEADS = 1
S_SUFFIX = 32  # action_horizon=10 padded to tile
HEAD_DIM = 256
KV_SEQ_LEN = int(os.environ.get("PI0_SDPA2_KV", "544"))  # prefix(512) + suffix(32)

NUM_WARMUP = int(os.environ.get("PI0_SDPA2_WARMUP", "10"))
NUM_ITER = int(os.environ.get("PI0_SDPA2_ITER", "30"))
PCC_THRESHOLD = float(os.environ.get("PI0_SDPA2_PCC", "0.999"))


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


def _torch_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, scale: float):
    """Reference SDPA. Mask is additive (0=allow, -inf=block)."""
    # Broadcast K, V along heads to match Q's num_heads (since num_kv_heads=1)
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


def _build_inputs(device):
    torch.manual_seed(0)
    q = torch.randn(BATCH, NUM_HEADS, S_SUFFIX, HEAD_DIM) * 0.1
    k = torch.randn(BATCH, NUM_KV_HEADS, KV_SEQ_LEN, HEAD_DIM) * 0.1
    v = torch.randn(BATCH, NUM_KV_HEADS, KV_SEQ_LEN, HEAD_DIM) * 0.1
    mask = torch.zeros(BATCH, 1, S_SUFFIX, KV_SEQ_LEN)
    scale = 1.0 / (HEAD_DIM**0.5)

    def _upload(t, mc):
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=mc,
        )

    q_tt = _upload(q, ttnn.L1_MEMORY_CONFIG)
    k_tt = _upload(k, ttnn.L1_MEMORY_CONFIG)
    v_tt = _upload(v, ttnn.L1_MEMORY_CONFIG)
    mask_tt = _upload(mask, ttnn.DRAM_MEMORY_CONFIG)
    return q, k, v, mask, scale, q_tt, k_tt, v_tt, mask_tt


def _run_sdpa(q_tt, k_tt, v_tt, mask_tt, scale, c, grid):
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


def _time_config(device, q_tt, k_tt, v_tt, mask_tt, scale, c, grid):
    for _ in range(NUM_WARMUP):
        out = _run_sdpa(q_tt, k_tt, v_tt, mask_tt, scale, c, grid)
        ttnn.deallocate(out)
    ttnn.synchronize_device(device)
    samples: List[float] = []
    for _ in range(NUM_ITER):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        out = _run_sdpa(q_tt, k_tt, v_tt, mask_tt, scale, c, grid)
        ttnn.synchronize_device(device)
        samples.append((time.perf_counter() - t0) * 1000)
        ttnn.deallocate(out)
    return statistics.mean(samples) * 1000, min(samples) * 1000  # µs


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_denoise_sdpa_sweep_v2(device):
    grid = device.compute_with_storage_grid_size()
    print("\n" + "=" * 80)
    print(f"  DENOISE SDPA SWEEP v2  shape=(B={BATCH}, H={NUM_HEADS}, Sq={S_SUFFIX}, Skv={KV_SEQ_LEN}, hd={HEAD_DIM})")
    print(f"  grid={grid.x}×{grid.y}  warmup={NUM_WARMUP}  iter={NUM_ITER}  PCC≥{PCC_THRESHOLD}")
    print("=" * 80)

    q, k, v, mask, scale, q_tt, k_tt, v_tt, mask_tt = _build_inputs(device)

    # Build torch reference once
    ref = _torch_sdpa(q, k, v, mask, scale)

    # Baseline = current production picker output (q=32, k=32 after divisor-aware fix)
    base_c = CandConfig(
        q_chunk=32,
        k_chunk=32,
        max_cores_per_head_batch=16,
        exp_approx=False,
        fp32_dest_acc=True,
        packer_l1_acc=True,
        math_fidelity=ttnn.MathFidelity.HiFi2,
    )
    try:
        ref_out = ttnn.to_torch(_run_sdpa(q_tt, k_tt, v_tt, mask_tt, scale, base_c, grid))
        base_pcc = _pcc(ref, ref_out.reshape(ref.shape))
        base_mean, base_min = _time_config(device, q_tt, k_tt, v_tt, mask_tt, scale, base_c, grid)
        print(f"\n  Baseline: q=32 k=32 max_cores=16 exp_approx=0 fp32_dest=1 packer_l1=1 HiFi2")
        print(f"            mean={base_mean:.2f} µs  min={base_min:.2f}  pcc={base_pcc:.5f}")
    except RuntimeError as e:
        print(f"  Baseline failed: {str(e).splitlines()[0]}")
        return

    # Sweep new knobs (q/k chunks fixed at divisor-aware winner)
    candidates: List[CandConfig] = []
    fidelities = [ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4]
    for max_cores in (4, 8, 12, 16, 24, 32):
        for exp_approx in (False, True):
            for fp32_dest in (False, True):
                for packer_l1 in (True, False):
                    for fid in fidelities:
                        candidates.append(
                            CandConfig(
                                q_chunk=32,
                                k_chunk=32,
                                max_cores_per_head_batch=max_cores,
                                exp_approx=exp_approx,
                                fp32_dest_acc=fp32_dest,
                                packer_l1_acc=packer_l1,
                                math_fidelity=fid,
                            )
                        )

    print(f"\n  Trying {len(candidates)} configs...")
    results: List[Tuple[CandConfig, float, float, float]] = []
    for c in candidates:
        try:
            out = _run_sdpa(q_tt, k_tt, v_tt, mask_tt, scale, c, grid)
            out_torch = ttnn.to_torch(out)
            ttnn.deallocate(out)
            pcc = _pcc(ref, out_torch.reshape(ref.shape))
        except RuntimeError:
            continue
        try:
            mean, mn = _time_config(device, q_tt, k_tt, v_tt, mask_tt, scale, c, grid)
        except RuntimeError:
            continue
        results.append((c, mean, mn, pcc))

    ttnn.deallocate(q_tt)
    ttnn.deallocate(k_tt)
    ttnn.deallocate(v_tt)
    ttnn.deallocate(mask_tt)

    good = [r for r in results if r[3] >= PCC_THRESHOLD]
    good.sort(key=lambda r: r[1])

    print(f"\n  Tried: {len(results)}   Passing PCC≥{PCC_THRESHOLD}: {len(good)}")
    print()
    print(
        f"  {'rank':<4}  {'mean µs':>8}  {'min µs':>8}  {'pcc':>8}  {'vs base':>8}  {'cores':>5}  {'exp':>3}  {'fp32':>4}  {'pkr':>3}  {'fid':>5}"
    )
    for rank, (c, mean, mn, pcc) in enumerate(good[:20], 1):
        fid = "HiFi2" if c.math_fidelity == ttnn.MathFidelity.HiFi2 else "HiFi4"
        delta = f"{(mean - base_mean):+6.2f}"
        print(
            f"  {rank:<4}  {mean:>8.2f}  {mn:>8.2f}  {pcc:>8.5f}  {delta:>8}  {c.max_cores_per_head_batch:>5}  {int(c.exp_approx):>3}  {int(c.fp32_dest_acc):>4}  {int(c.packer_l1_acc):>3}  {fid:>5}"
        )
