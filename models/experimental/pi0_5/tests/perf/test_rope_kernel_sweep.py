# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Compare RoPE kernel variants at the pi0.5 denoise-step shape.

The pi0.5 expert at denoise calls Q-RoPE and K-RoPE 360× total per inference
(18 layers × 10 steps × 2). Each `ttnn.experimental.rotary_embedding` call
is ~12.4 µs and the workload is tiny (Q: 64 tiles, K: 8 tiles) so the
kernel is dispatch-bound. Total RoPE cost on denoise: ~4.5 ms.

This bench tries each available RoPE kernel variant at the denoise shape:
  1. ttnn.experimental.rotary_embedding              (current — split-half pattern)
  2. ttnn.experimental.rotary_embedding_llama         (matrix-mul rotation)
  3. ttnn.experimental.rotary_embedding_llama_fused_qk (fuses Q+K into 1 call,
                                                       seq_len=1 only — expected fail)

The llama variant requires fp32_dest_acc_en=False when head_dim > 128
(pi0.5 expert has head_dim=256). That's a precision tradeoff.

Run:
    PI0_ROPE_SWEEP=1 pytest -xvs \\
      models/experimental/pi0_5/tests/perf/test_rope_kernel_sweep.py
"""

from __future__ import annotations

import os
import statistics
import time
from typing import List, Tuple

import pytest
import torch
import ttnn

from models.common.tensor_utils import get_rot_transformation_mat


BENCH_ENABLED = os.environ.get("PI0_ROPE_SWEEP") == "1"
pytestmark = pytest.mark.skipif(not BENCH_ENABLED, reason="set PI0_ROPE_SWEEP=1 to run the RoPE kernel sweep")

# pi0.5 expert denoise shape
BATCH = 1
NUM_HEADS_Q = 8
NUM_HEADS_K = 1
SEQ_LEN = 32  # action_horizon=10 padded to 32 (1 tile)
HEAD_DIM = 256
MAX_SEQ_LEN = 512

NUM_WARMUP = 10
NUM_ITER = 50


def _upload(t: torch.Tensor, device, memory_config=None) -> "ttnn.Tensor":
    return ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config if memory_config else ttnn.L1_MEMORY_CONFIG,
    )


def _precompute_cos_sin_splithalf(head_dim: int, max_seq_len: int):
    """For ttnn.experimental.rotary_embedding (the split-half kernel pi0.5 uses).

    Returns (cos, sin) each shaped [1, 1, max_seq_len, head_dim] in split-half
    layout: [c0, c1, ..., c_{half-1}, c0, c1, ..., c_{half-1}].
    """
    half = head_dim // 2
    freqs = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    fo = torch.outer(t, freqs)
    cos = torch.cos(fo)
    sin = torch.sin(fo)
    cos_full = torch.cat([cos, cos], dim=-1)  # [max_seq_len, head_dim]
    sin_full = torch.cat([sin, sin], dim=-1)
    return cos_full.unsqueeze(0).unsqueeze(0), sin_full.unsqueeze(0).unsqueeze(0)


def _precompute_cos_sin_llama(head_dim: int, max_seq_len: int):
    """For rotary_embedding_llama. cos/sin in standard non-doubled layout."""
    freqs = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    fo = torch.outer(t, freqs)
    # The llama kernel still expects head_dim cos/sin (pairs the half columns
    # via the trans_mat). So we still need a head_dim-wide cos/sin.
    cos = torch.cos(fo).repeat_interleave(2, dim=-1)
    sin = torch.sin(fo).repeat_interleave(2, dim=-1)
    return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)


def _time_kernel(label: str, fn, n_warmup=NUM_WARMUP, n_iter=NUM_ITER) -> Tuple[float, float, float]:
    for _ in range(n_warmup):
        out = fn()
        ttnn.deallocate(out)
    ttnn.synchronize_device(_DEVICE_REF[0])

    samples: List[float] = []
    for _ in range(n_iter):
        ttnn.synchronize_device(_DEVICE_REF[0])
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(_DEVICE_REF[0])
        samples.append((time.perf_counter() - t0) * 1000)
        ttnn.deallocate(out)
    return statistics.mean(samples), min(samples), statistics.stdev(samples) if len(samples) > 1 else 0.0


_DEVICE_REF: List["ttnn.Device"] = [None]


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_rope_kernel_sweep(device):
    _DEVICE_REF[0] = device

    print("\n" + "=" * 80)
    print(
        f"  RoPE kernel sweep @ denoise shape  Q=[{BATCH},{NUM_HEADS_Q},{SEQ_LEN},{HEAD_DIM}]  "
        f"K=[{BATCH},{NUM_HEADS_K},{SEQ_LEN},{HEAD_DIM}]"
    )
    print(f"  warmup={NUM_WARMUP}  iter={NUM_ITER}")
    print("=" * 80)

    # Build random Q, K tensors
    torch.manual_seed(0)
    q_torch = torch.randn(BATCH, NUM_HEADS_Q, SEQ_LEN, HEAD_DIM) * 0.1
    k_torch = torch.randn(BATCH, NUM_HEADS_K, SEQ_LEN, HEAD_DIM) * 0.1
    q_tt = _upload(q_torch, device)
    k_tt = _upload(k_torch, device)

    # Compute kernel config
    ck_cfg_default = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,  # required by llama for head_dim > 128
        packer_l1_acc=True,
    )

    # --- 1) rotary_embedding (current) ---
    cos_sh, sin_sh = _precompute_cos_sin_splithalf(HEAD_DIM, MAX_SEQ_LEN)
    # Slice to actual seq_len
    cos_sh_seq = cos_sh[:, :, :SEQ_LEN, :]
    sin_sh_seq = sin_sh[:, :, :SEQ_LEN, :]
    cos_sh_tt = _upload(cos_sh_seq, device)
    sin_sh_tt = _upload(sin_sh_seq, device)

    def run_orig_q():
        return ttnn.experimental.rotary_embedding(q_tt, cos_sh_tt, sin_sh_tt)

    def run_orig_k():
        return ttnn.experimental.rotary_embedding(k_tt, cos_sh_tt, sin_sh_tt)

    mean_orig_q, min_orig_q, std_orig_q = _time_kernel("orig_Q", run_orig_q)
    mean_orig_k, min_orig_k, std_orig_k = _time_kernel("orig_K", run_orig_k)
    print(f"\n  [orig] rotary_embedding (current pi0.5 path)")
    print(
        f"     Q (Z={NUM_HEADS_Q}): mean={mean_orig_q*1000:.2f} µs  min={min_orig_q*1000:.2f}  std={std_orig_q*1000:.2f}"
    )
    print(
        f"     K (Z={NUM_HEADS_K}): mean={mean_orig_k*1000:.2f} µs  min={min_orig_k*1000:.2f}  std={std_orig_k*1000:.2f}"
    )
    print(f"     Q+K combined: {(mean_orig_q + mean_orig_k)*1000:.2f} µs")
    ttnn.deallocate(cos_sh_tt)
    ttnn.deallocate(sin_sh_tt)

    # --- 2) rotary_embedding_llama ---
    cos_ll, sin_ll = _precompute_cos_sin_llama(HEAD_DIM, MAX_SEQ_LEN)
    cos_ll_seq = cos_ll[:, :, :SEQ_LEN, :]
    sin_ll_seq = sin_ll[:, :, :SEQ_LEN, :]
    cos_ll_tt = _upload(cos_ll_seq, device, ttnn.DRAM_MEMORY_CONFIG)
    sin_ll_tt = _upload(sin_ll_seq, device, ttnn.DRAM_MEMORY_CONFIG)
    # For prefill mode (seq_len > 1), trans_mat is head_dim × head_dim
    trans_mat_torch = get_rot_transformation_mat()
    trans_mat_tt = _upload(trans_mat_torch, device, ttnn.DRAM_MEMORY_CONFIG)

    def run_llama_q():
        return ttnn.experimental.rotary_embedding_llama(
            q_tt,
            cos_ll_tt,
            sin_ll_tt,
            trans_mat_tt,
            is_decode_mode=False,
            compute_kernel_config=ck_cfg_default,
        )

    def run_llama_k():
        return ttnn.experimental.rotary_embedding_llama(
            k_tt,
            cos_ll_tt,
            sin_ll_tt,
            trans_mat_tt,
            is_decode_mode=False,
            compute_kernel_config=ck_cfg_default,
        )

    try:
        mean_llama_q, min_llama_q, std_llama_q = _time_kernel("llama_Q", run_llama_q)
        mean_llama_k, min_llama_k, std_llama_k = _time_kernel("llama_K", run_llama_k)
        print(f"\n  [llama] rotary_embedding_llama (matrix-mul rotation)")
        print(
            f"     Q (Z={NUM_HEADS_Q}): mean={mean_llama_q*1000:.2f} µs  min={min_llama_q*1000:.2f}  std={std_llama_q*1000:.2f}"
        )
        print(
            f"     K (Z={NUM_HEADS_K}): mean={mean_llama_k*1000:.2f} µs  min={min_llama_k*1000:.2f}  std={std_llama_k*1000:.2f}"
        )
        print(f"     Q+K combined: {(mean_llama_q + mean_llama_k)*1000:.2f} µs")
        print(
            f"     Δ vs orig:    {((mean_llama_q + mean_llama_k) - (mean_orig_q + mean_orig_k))*1000:+.2f} µs/layer-step"
        )
        print(
            f"     × 180 calls:  {((mean_llama_q + mean_llama_k) - (mean_orig_q + mean_orig_k))*180*1000:+.0f} µs total over denoise"
        )
    except RuntimeError as e:
        print(f"\n  [llama] FAILED:")
        for line in str(e).splitlines()[:6]:
            print(f"    {line}")
    finally:
        try:
            ttnn.deallocate(cos_ll_tt)
            ttnn.deallocate(sin_ll_tt)
            ttnn.deallocate(trans_mat_tt)
        except Exception:
            pass

    # --- 3) rotary_embedding_llama_fused_qk (expected to fail for seq_len > 1) ---
    try:
        cos_ll2, sin_ll2 = _precompute_cos_sin_llama(HEAD_DIM, MAX_SEQ_LEN)
        cos_ll2_tt = _upload(cos_ll2[:, :, :SEQ_LEN, :], device, ttnn.DRAM_MEMORY_CONFIG)
        sin_ll2_tt = _upload(sin_ll2[:, :, :SEQ_LEN, :], device, ttnn.DRAM_MEMORY_CONFIG)
        trans_mat2_tt = _upload(get_rot_transformation_mat(), device, ttnn.DRAM_MEMORY_CONFIG)

        def run_fused():
            return ttnn.experimental.rotary_embedding_llama_fused_qk(
                q_tt,
                k_tt,
                cos_ll2_tt,
                sin_ll2_tt,
                trans_mat2_tt,
                compute_kernel_config=ck_cfg_default,
            )

        out = run_fused()
        # If it didn't raise, it works — measure
        ttnn.deallocate(out[0])
        ttnn.deallocate(out[1])
        print("\n  [fused_qk] SUPPORTED at seq_len=32 — should benchmark")
    except RuntimeError as e:
        msg = str(e).splitlines()[0][:150]
        print(f"\n  [fused_qk] FAILED (expected): {msg[:200]}")

    print()
