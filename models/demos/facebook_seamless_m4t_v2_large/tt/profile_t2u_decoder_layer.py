#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tracy harness for the SeamlessM4T-v2 T2U decoder layer (NAR text-to-unit).

The T2U decoder layer is the per-layer building block of the
non-autoregressive Text-to-Unit decoder (6 stacked instances in
production). It runs ONCE per ``synthesize()`` call over the full
upsampled unit sequence, so optimising it benefits the T2ST and S2ST
audio-out pipelines.

Unlike the AR text decoder layer (which runs single-token, ``T=1``),
this NAR layer runs at the FULL unit sequence length at once -- typical
production shape after character-expansion is several tens to a couple
hundred tokens. This makes ``M_tiles >> 1`` and matmul sharding /
larger-grid LN are more impactful here than in AR decode.

Production shapes (SeamlessM4T-v2-Large config + golden):
  - hidden_size = 1024
  - decoder_attention_heads = 16, head_dim = 64
  - conv_kernel_size = 7 (symmetric pad=3, NOT causal)
  - layer_norm_eps = 1e-5
  - seq_len = 128 (tile-aligned representative; goldens are T=64)

Run untraced (host dispatch dominates, useful only for sanity):

    source python_env/bin/activate && export TT_METAL_HOME=$(pwd) && \\
        export PYTHONPATH=$(pwd) && export ARCH_NAME=blackhole
    python models/demos/facebook_seamless_m4t_v2_large/tt/profile_t2u_decoder_layer.py

Run under tracy with metal trace (the production path inside
``t2u_generator.synthesize_units``):

    python -m tracy -p -v -r --no-device-data-capture \\
        -o generated/profiler/reports/t2u_decoder_layer_traced \\
        models/demos/facebook_seamless_m4t_v2_large/tt/profile_t2u_decoder_layer.py --traced

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
from models.demos.facebook_seamless_m4t_v2_large.tt.t2u_decoder_layer import T2UDecoderLayer

# Production shapes (SeamlessM4T-v2-Large).
EMBED_DIM = 1024
NUM_HEADS = 16
HEAD_DIM = 64
SEQ_LEN = 128  # NAR full-sequence length (tile-aligned).
CONV_KSIZE = 7
EPS = 1e-5
BATCH = 1


def _make_state_dict(seed: int = 0) -> dict:
    """Build a representative state dict for one T2U decoder layer."""
    g = torch.Generator().manual_seed(seed)

    def _ln():
        return {
            "weight": torch.ones(EMBED_DIM),
            "bias": torch.zeros(EMBED_DIM),
        }

    def _attn(prefix_seed):
        gp = torch.Generator().manual_seed(prefix_seed)
        sd = {}
        for n in ("q_proj", "k_proj", "v_proj", "out_proj"):
            sd[n] = {
                "weight": torch.randn(EMBED_DIM, EMBED_DIM, generator=gp) * (1.0 / EMBED_DIM**0.5),
                "bias": torch.zeros(EMBED_DIM),
            }
        return sd

    def _conv(prefix_seed):
        gp = torch.Generator().manual_seed(prefix_seed)
        # Conv1d weight: [out_C, in_C, K]; HF stores bias [out_C].
        return {
            "weight": torch.randn(EMBED_DIM, EMBED_DIM, CONV_KSIZE, generator=gp)
            * (1.0 / (EMBED_DIM * CONV_KSIZE) ** 0.5),
            "bias": torch.zeros(EMBED_DIM),
        }

    return {
        "self_attn": _attn(seed + 1),
        "self_attn_layer_norm": _ln(),
        "conv1": _conv(seed + 2),
        "conv2": _conv(seed + 3),
        "conv_layer_norm": _ln(),
    }


def _to_tt_bf16(device, t: torch.Tensor) -> ttnn.Tensor:
    return ttnn.from_torch(
        t,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _run_layer(device, layer, n_iter: int, use_trace: bool) -> List[float]:
    """Drive one T2UDecoderLayer forward at production shapes (B=1, T=SEQ_LEN).

    ``T2UDecoderLayer.forward`` deallocates its residual inputs internally,
    so for the traced path we re-upload host data into the persistent
    buffer before each replay (the trace re-runs the same op sequence
    starting from that buffer).
    """
    g = torch.Generator().manual_seed(11)
    x_host = torch.randn(BATCH, SEQ_LEN, EMBED_DIM, generator=g) * 0.02
    # Padding-mask: all-ones float multiplier of shape [B, T, 1] -- exercises
    # the masked path that production uses inside the conv branch.
    pad_host = torch.ones(BATCH, SEQ_LEN, 1, dtype=torch.float32)
    # Attention mask: BART-style additive log-mask of shape [B, 1, T, T] (zeros
    # = no masking; exercises the masked self-attention path).
    attn_host = torch.zeros(BATCH, 1, SEQ_LEN, SEQ_LEN, dtype=torch.float32)

    tt_pad = _to_tt_bf16(device, pad_host)
    tt_attn = _to_tt_bf16(device, attn_host)

    times: List[float] = []
    if use_trace:
        # Persistent input buffer used for capture AND replay.
        x_pers = _to_tt_bf16(device, x_host)

        # Warmup -- compiles all kernels into the program cache.
        warm = _to_tt_bf16(device, x_host)
        _ = layer.forward(warm, attention_mask=tt_attn, padding_mask=tt_pad)
        ttnn.synchronize_device(device)

        # Capture.
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        out = layer.forward(x_pers, attention_mask=tt_attn, padding_mask=tt_pad)
        ttnn.end_trace_capture(device, tid, cq_id=0)
        ttnn.synchronize_device(device)

        # Replay loop -- reallocate persistent input so the trace's internal
        # deallocate() doesn't poison the next iteration.
        for _ in range(n_iter):
            x_pers = _to_tt_bf16(device, x_host)
            t0 = time.perf_counter()
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
        ttnn.release_trace(device, tid)
    else:
        # Untraced -- host dispatch dominates.
        warm = _to_tt_bf16(device, x_host)
        _ = layer.forward(warm, attention_mask=tt_attn, padding_mask=tt_pad)
        ttnn.synchronize_device(device)

        for _ in range(n_iter):
            tt_x = _to_tt_bf16(device, x_host)
            t0 = time.perf_counter()
            out = layer.forward(tt_x, attention_mask=tt_attn, padding_mask=tt_pad)
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
    global SEQ_LEN
    parser = argparse.ArgumentParser()
    parser.add_argument("--traced", action="store_true", help="Use metal trace capture+replay (production path).")
    parser.add_argument("--n-iter", type=int, default=20, help="Steady-state iteration count.")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    args = parser.parse_args()
    SEQ_LEN = args.seq_len

    print(
        f"[profile_t2u_decoder_layer] embed={EMBED_DIM} heads={NUM_HEADS} "
        f"head_dim={HEAD_DIM} T={args.seq_len} k={CONV_KSIZE} "
        f"traced={args.traced} n_iter={args.n_iter}"
    )

    device = ttnn.open_device(
        device_id=args.device_id,
        l1_small_size=16384,
        trace_region_size=64_000_000 if args.traced else 0,
    )
    try:
        sd = _make_state_dict()
        layer = T2UDecoderLayer(
            device=device,
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            head_dim=HEAD_DIM,
            state_dict=sd,
            conv_kernel_size=CONV_KSIZE,
            eps=EPS,
            weight_dtype=ttnn.bfloat16,
        )

        times = _run_layer(device, layer, args.n_iter, args.traced)
        print(_stat("[t2u_decoder_layer] step_ms", times))
        print("[profile_t2u_decoder_layer] DONE")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
