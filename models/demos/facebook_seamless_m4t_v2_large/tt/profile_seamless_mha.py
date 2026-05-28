#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tracy harness for the SeamlessM4T-v2 SeamlessMHA block.

Exercises three production-representative scenarios:

  1) ENC_SELF : encoder self-attention (T=128 prefill, no encoder_hidden_states).
  2) DEC_CROSS: decoder cross-attention DECODE step (T=1 Q, S=128 cached K/V).
  3) DEC_SELF : decoder self-attention DECODE step (T=1 Q + KV-cache append).

Run untraced (host dispatch dominates, useful for sanity):

    source python_env/bin/activate && export TT_METAL_HOME=$(pwd) && \\
        export PYTHONPATH=$(pwd) && export ARCH_NAME=blackhole
    python models/demos/facebook_seamless_m4t_v2_large/tt/profile_seamless_mha.py

Run under tracy with metal trace (the production path inside the AR loop):

    python -m tracy -p -v -r --no-device-data-capture \\
        -o generated/profiler/reports/seamless_mha \\
        models/demos/facebook_seamless_m4t_v2_large/tt/profile_seamless_mha.py --traced

The CSV at generated/profiler/.logs/cpp_device_perf_report.csv is the
authoritative artifact for hotspot analysis.
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import List

import torch

import ttnn

# Production shapes (large variant, FP32 reference).
EMBED_DIM = 1024
NUM_HEADS = 16
HEAD_DIM = 64
ENC_SRC_LEN = 128  # encoder cache T (tile-padded; encoder prefill uses 128)
DEC_TGT_LEN = 1  # AR decode step Q length


def _make_state_dict(seed: int = 0) -> dict:
    """Create a representative state dict mirroring HF SeamlessM4Tv2Attention.

    All four projections (q/k/v/out) are nn.Linear(embed_dim, embed_dim, bias=True).
    """
    g = torch.Generator().manual_seed(seed)
    sd = {}
    for n in ("q_proj", "k_proj", "v_proj", "out_proj"):
        sd[n] = {
            "weight": (torch.randn(EMBED_DIM, EMBED_DIM, generator=g) * (1.0 / EMBED_DIM**0.5)),
            "bias": torch.zeros(EMBED_DIM),
        }
    return sd


def _to_tt_bf16(device, t: torch.Tensor) -> ttnn.Tensor:
    return ttnn.from_torch(
        t,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _run_enc_self(device, mha, n_iter: int, use_trace: bool) -> List[float]:
    """Encoder self-attention forward — B=1, T=ENC_SRC_LEN."""
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, ENC_SRC_LEN, EMBED_DIM, generator=g) * 0.02
    tt_x = _to_tt_bf16(device, x)

    # Warmup outside trace
    _ = mha.forward(tt_x)
    ttnn.synchronize_device(device)

    times = []
    if use_trace:
        # Capture a single trace covering one forward.
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        out = mha.forward(tt_x)
        ttnn.end_trace_capture(device, tid, cq_id=0)
        ttnn.synchronize_device(device)
        # Replay n_iter times — only replay times count as steady-state.
        for _ in range(n_iter):
            t0 = time.perf_counter()
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
        ttnn.release_trace(device, tid)
    else:
        for _ in range(n_iter):
            t0 = time.perf_counter()
            out = mha.forward(tt_x)
            ttnn.synchronize_device(device)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
    return times


def _run_dec_cross(device, mha, n_iter: int, use_trace: bool) -> List[float]:
    """Decoder cross-attention DECODE step — Q is T=1, K/V are cached at S=ENC_SRC_LEN.

    This uses project_q + attend_and_out_project (the production cached path).
    """
    g = torch.Generator().manual_seed(2)
    h = torch.randn(1, DEC_TGT_LEN, EMBED_DIM, generator=g) * 0.02
    enc = torch.randn(1, ENC_SRC_LEN, EMBED_DIM, generator=g) * 0.02

    tt_h = _to_tt_bf16(device, h)
    tt_enc = _to_tt_bf16(device, enc)

    # Pre-project K/V once (cache); these stay alive across replays.
    k_cache, v_cache = mha.project_kv(tt_enc)

    # Warmup
    q = mha.project_q(tt_h)
    out = mha.attend_and_out_project(q, k_cache, v_cache, attention_mask=None)
    ttnn.synchronize_device(device)

    times = []
    if use_trace:
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        q2 = mha.project_q(tt_h)
        out2 = mha.attend_and_out_project(q2, k_cache, v_cache, attention_mask=None)
        ttnn.end_trace_capture(device, tid, cq_id=0)
        ttnn.synchronize_device(device)
        for _ in range(n_iter):
            t0 = time.perf_counter()
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
        ttnn.release_trace(device, tid)
    else:
        for _ in range(n_iter):
            t0 = time.perf_counter()
            q3 = mha.project_q(tt_h)
            out3 = mha.attend_and_out_project(q3, k_cache, v_cache, attention_mask=None)
            ttnn.synchronize_device(device)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
    return times


def _run_dec_self(device, mha, n_iter: int, use_trace: bool) -> List[float]:
    """Decoder self-attention DECODE step — single-token Q + cached K/V (S=ENC_SRC_LEN).

    Mimics the AR loop self-attn step where every position projects QKV for
    the new token then attends against the full history.
    """
    g = torch.Generator().manual_seed(3)
    h = torch.randn(1, DEC_TGT_LEN, EMBED_DIM, generator=g) * 0.02
    history = torch.randn(1, ENC_SRC_LEN, EMBED_DIM, generator=g) * 0.02

    tt_h = _to_tt_bf16(device, h)
    tt_hist = _to_tt_bf16(device, history)

    # Build static cache via project_kv on history.
    k_hist, v_hist = mha.project_kv(tt_hist)

    # Warmup
    q, k_new, v_new = mha.project_qkv_single_token(tt_h)
    out = mha.attend_and_out_project(q, k_hist, v_hist, attention_mask=None)
    ttnn.deallocate(k_new)
    ttnn.deallocate(v_new)
    ttnn.synchronize_device(device)

    times = []
    if use_trace:
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        q2, k_new2, v_new2 = mha.project_qkv_single_token(tt_h)
        ttnn.deallocate(k_new2)
        ttnn.deallocate(v_new2)
        out2 = mha.attend_and_out_project(q2, k_hist, v_hist, attention_mask=None)
        ttnn.end_trace_capture(device, tid, cq_id=0)
        ttnn.synchronize_device(device)
        for _ in range(n_iter):
            t0 = time.perf_counter()
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
        ttnn.release_trace(device, tid)
    else:
        for _ in range(n_iter):
            t0 = time.perf_counter()
            q3, k_new3, v_new3 = mha.project_qkv_single_token(tt_h)
            ttnn.deallocate(k_new3)
            ttnn.deallocate(v_new3)
            out3 = mha.attend_and_out_project(q3, k_hist, v_hist, attention_mask=None)
            ttnn.synchronize_device(device)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
    return times


def _stat(name: str, xs: List[float]) -> str:
    if not xs:
        return f"{name}=<empty>"
    return (
        f"{name}: n={len(xs)}, min={min(xs):.3f}, "
        f"p50={statistics.median(xs):.3f}, mean={statistics.mean(xs):.3f}, "
        f"max={max(xs):.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traced", action="store_true", help="Use metal trace capture+replay (production path).")
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        choices=["all", "enc_self", "dec_cross", "dec_self"],
        help="Which scenario(s) to run.",
    )
    parser.add_argument("--n-iter", type=int, default=20, help="Steady-state iteration count per scenario.")
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()

    from models.demos.facebook_seamless_m4t_v2_large.tt.seamless_mha import SeamlessMHA

    print(f"[profile_seamless_mha] embed_dim={EMBED_DIM} num_heads={NUM_HEADS} head_dim={HEAD_DIM}")
    print(f"[profile_seamless_mha] ENC_SRC_LEN={ENC_SRC_LEN} DEC_TGT_LEN={DEC_TGT_LEN}")
    print(f"[profile_seamless_mha] traced={args.traced} scenario={args.scenario} n_iter={args.n_iter}")

    device = ttnn.open_device(
        device_id=args.device_id,
        l1_small_size=16384,
        trace_region_size=64_000_000 if args.traced else 0,
    )
    try:
        sd = _make_state_dict()
        mha = SeamlessMHA(
            device=device,
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            head_dim=HEAD_DIM,
            state_dict=sd,
            weight_dtype=ttnn.bfloat16,
            weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        scenarios = ["enc_self", "dec_cross", "dec_self"] if args.scenario == "all" else [args.scenario]
        results = {}
        for s in scenarios:
            if s == "enc_self":
                t = _run_enc_self(device, mha, args.n_iter, args.traced)
            elif s == "dec_cross":
                t = _run_dec_cross(device, mha, args.n_iter, args.traced)
            else:
                t = _run_dec_self(device, mha, args.n_iter, args.traced)
            results[s] = t
            print(_stat(f"[{s}] step_ms", t))

        print("[profile_seamless_mha] DONE")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
