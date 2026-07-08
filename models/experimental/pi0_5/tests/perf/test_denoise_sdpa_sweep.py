# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sweep SDPA (q_chunk_size, k_chunk_size) for the pi0.5 denoise step.

Isolates one call to `ttnn.transformer.scaled_dot_product_attention` at the
exact shape the denoise loop hits each step:

    Q   shape = [B=1, num_heads=8,    S_suffix=32,  head_dim=256]
    K   shape = [B=1, num_kv_heads=1, kv_seq_len,    head_dim=256]
    V   shape = same as K
    msk shape = [B=1, 1, S_suffix, kv_seq_len]   (zeros = unmasked)

kv_seq_len defaults to 544 (prefix=512 + suffix=32). The current
production picker (`sdpa_prefill_chunk_sizes`) chooses q=64, k=128 for
this shape; this sweep walks a wider matrix to see if there's a better
config.

References:
- gtobarTT/bgem3_p150_demo/.../bge_m3/tt/attention.py picks q=128, k∈{256,128}
  for encoder shapes (with full grid + HiFi4 + fp32_dest_acc + packer_l1).
- pi0_5 production SDPA uses HiFi2 (PI0_SDPA_HIFI=2) by default.

Run:
    PI0_DENOISE_SDPA_SWEEP=1 pytest -xvs \\
      models/experimental/pi0_5/tests/perf/test_denoise_sdpa_sweep.py

Env knobs:
    PI0_DENOISE_SDPA_KV=544          kv_seq_len (must be %32). Default 544.
    PI0_DENOISE_SDPA_ITER=50         timed iterations per config.
    PI0_DENOISE_SDPA_WARMUP=10       warmup iterations per config.
    PI0_DENOISE_SDPA_Q_CHUNKS=32,64,128         q_chunk_size grid (comma list).
    PI0_DENOISE_SDPA_K_CHUNKS=32,64,128,256,544 k_chunk_size grid.
    PI0_DENOISE_SDPA_EXP_APPROX=0|1  also sweep exp_approx_mode (default off=just False).
    PI0_DENOISE_SDPA_FIDELITY=2|4    math fidelity (HiFi2 default).
"""

from __future__ import annotations

import os
import statistics
import time
from typing import List, Tuple

import pytest
import torch
import ttnn


ENABLED = os.environ.get("PI0_DENOISE_SDPA_SWEEP") == "1"
pytestmark = pytest.mark.skipif(not ENABLED, reason="set PI0_DENOISE_SDPA_SWEEP=1 to run the denoise SDPA sweep")

# ---- shape constants — pi0.5 expert at denoise step ----
BATCH = 1
NUM_HEADS = 8
NUM_KV_HEADS = 1
S_SUFFIX = 32  # action_horizon=10 padded to 32; 32 = 1 tile-row
HEAD_DIM = 256
KV_SEQ_LEN = int(os.environ.get("PI0_DENOISE_SDPA_KV", "544"))

NUM_WARMUP = int(os.environ.get("PI0_DENOISE_SDPA_WARMUP", "10"))
NUM_ITER = int(os.environ.get("PI0_DENOISE_SDPA_ITER", "50"))

_Q_CHUNKS = [int(x) for x in os.environ.get("PI0_DENOISE_SDPA_Q_CHUNKS", "32,64,128").split(",")]
_K_CHUNKS = [int(x) for x in os.environ.get("PI0_DENOISE_SDPA_K_CHUNKS", "32,64,128,256,544").split(",")]

# Toggle 'True' for exp_approx as a secondary axis; default: only False (matches prod)
SWEEP_EXP_APPROX = os.environ.get("PI0_DENOISE_SDPA_EXP_APPROX") == "1"
FIDELITY = (
    ttnn.MathFidelity.HiFi4 if os.environ.get("PI0_DENOISE_SDPA_FIDELITY", "2") == "4" else ttnn.MathFidelity.HiFi2
)


def _upload_tile_l1(t: torch.Tensor, device) -> "ttnn.Tensor":
    return ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )


def _upload_tile_dram(t: torch.Tensor, device) -> "ttnn.Tensor":
    return ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _make_inputs(device):
    """Build Q/K/V/mask matching denoise-step shape, on device."""
    torch.manual_seed(0)
    q = torch.randn(BATCH, NUM_HEADS, S_SUFFIX, HEAD_DIM) * 0.1
    k = torch.randn(BATCH, NUM_KV_HEADS, KV_SEQ_LEN, HEAD_DIM) * 0.1
    v = torch.randn(BATCH, NUM_KV_HEADS, KV_SEQ_LEN, HEAD_DIM) * 0.1
    mask = torch.zeros(BATCH, 1, S_SUFFIX, KV_SEQ_LEN)
    return (
        _upload_tile_l1(q, device),
        _upload_tile_l1(k, device),
        _upload_tile_l1(v, device),
        _upload_tile_dram(mask, device),  # SDPA requires DRAM mask
    )


def _time_sdpa(
    device,
    q_tt,
    k_tt,
    v_tt,
    mask_tt,
    q_chunk: int,
    k_chunk: int,
    exp_approx: bool,
    ck_cfg,
) -> Tuple[float, float, float, float]:
    """Returns (mean_ms, stdev_ms, min_ms, max_ms)."""
    grid = device.compute_with_storage_grid_size()
    sdpa_cfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid,
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        exp_approx_mode=exp_approx,
    )
    scale = 1.0 / (HEAD_DIM**0.5)

    for _ in range(NUM_WARMUP):
        out = ttnn.transformer.scaled_dot_product_attention(
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
        ttnn.deallocate(out)
    ttnn.synchronize_device(device)

    samples: List[float] = []
    for _ in range(NUM_ITER):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        out = ttnn.transformer.scaled_dot_product_attention(
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
        ttnn.synchronize_device(device)
        samples.append((time.perf_counter() - t0) * 1000)
        ttnn.deallocate(out)

    return (
        statistics.mean(samples),
        statistics.stdev(samples) if len(samples) > 1 else 0.0,
        min(samples),
        max(samples),
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_denoise_sdpa_sweep(device):
    fid_lbl = "HiFi4" if FIDELITY == ttnn.MathFidelity.HiFi4 else "HiFi2"
    print("\n" + "=" * 80)
    print(
        f"  DENOISE-STEP SDPA SWEEP  shape=(B={BATCH}, H={NUM_HEADS}, Sq={S_SUFFIX}, "
        f"Skv={KV_SEQ_LEN}, hd={HEAD_DIM})"
    )
    print(f"  warmup={NUM_WARMUP}  iter={NUM_ITER}  fidelity={fid_lbl}  " f"sweep_exp_approx={SWEEP_EXP_APPROX}")
    print(f"  q_chunks={_Q_CHUNKS}  k_chunks={_K_CHUNKS}")
    print("=" * 80)

    ck_cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=FIDELITY,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    q_tt, k_tt, v_tt, mask_tt = _make_inputs(device)

    exp_modes = (False, True) if SWEEP_EXP_APPROX else (False,)
    results: List[Tuple[int, int, bool, float, float, float, float]] = []

    for exp_approx in exp_modes:
        for q_chunk in _Q_CHUNKS:
            for k_chunk in _K_CHUNKS:
                # Tile-align is hard requirement; chunk > seq_len lets the kernel
                # decide whether to pad (run as single chunk) or reject.
                if q_chunk % 32 or k_chunk % 32:
                    continue
                tag = f"q={q_chunk:>4}  k={k_chunk:>4}  exp_approx={int(exp_approx)}"
                try:
                    mean, std, mn, mx = _time_sdpa(
                        device,
                        q_tt,
                        k_tt,
                        v_tt,
                        mask_tt,
                        q_chunk,
                        k_chunk,
                        exp_approx,
                        ck_cfg,
                    )
                except RuntimeError as e:
                    msg = str(e).splitlines()[0][:120]
                    print(f">> {tag}   FAILED: {msg}")
                    continue
                print(f">> {tag}   mean={mean:6.3f} ms  std={std:.4f}  min={mn:6.3f}  max={mx:6.3f}")
                results.append((q_chunk, k_chunk, exp_approx, mean, std, mn, mx))

    ttnn.deallocate(q_tt)
    ttnn.deallocate(k_tt)
    ttnn.deallocate(v_tt)
    ttnn.deallocate(mask_tt)

    if not results:
        pytest.fail("no SDPA configs ran successfully")

    print("\n" + "=" * 80)
    print("  SUMMARY (sorted by mean ms ASC; best first)")
    print("=" * 80)
    results_sorted = sorted(results, key=lambda r: r[3])
    best = results_sorted[0]
    print(
        f"  {'rank':<4}  {'q_chunk':>8}  {'k_chunk':>8}  {'exp_approx':>10}  {'mean ms':>9}  {'min':>7}  {'speedup':>8}"
    )
    for rank, r in enumerate(results_sorted, start=1):
        q, k, ea, mean, std, mn, mx = r
        speedup = best[3] / mean if mean > 0 else 0.0
        print(f"  {rank:<4}  {q:>8}  {k:>8}  {int(ea):>10}  {mean:>9.3f}  {mn:>7.3f}  {speedup:>8.3f}x")
    print("=" * 80)
    print(
        f"  Production default: q=32, k=128, exp_approx=0  "
        f"(picker base_q=64 capped to q_aligned=Sq=32; see sdpa_prefill_chunk_sizes in ttnn_common.py)"
    )
    print("=" * 80)
