#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tracy harness for the SeamlessM4T-v2 ConformerEncoderLayer block.

The Conformer encoder layer is the per-layer building block of the speech
encoder (24 stacked instances in production). Optimising it benefits the
S2TT, S2ST, and ASR pipelines.

In production, the layer runs inside :class:`ConformerEncoder` (see
``tt/speech_encoder.py``) which is invoked under the model-level metal
trace captured by ``tt/profile_s2tt.py`` (and friends). To make the
device-kernel time the dominant signal in tracy — not host dispatch —
this harness captures a per-layer metal trace and replays it.

Production shapes:
  - hidden_size = 1024
  - speech_encoder_attention_heads = 16, head_dim = 64
  - speech encoder operates on seq_len = 256 (the
    ``DEFAULT_AUDIO_SEQ_LEN`` post-feature-extraction time axis budget
    enforced by ``tt/speech_to_text_model.py``).
  - position_embeddings_type = "relative_key"
  - conv_depthwise_kernel_size = 31
  - layer_norm_eps = 1e-5

Run untraced (host dispatch dominates, useful only for sanity):

    source python_env/bin/activate && export TT_METAL_HOME=$(pwd) && \\
        export PYTHONPATH=$(pwd) && export ARCH_NAME=blackhole
    python models/demos/facebook_seamless_m4t_v2_large/tt/profile_conformer_encoder_layer.py

Run under tracy with metal trace (the production path inside the
speech encoder forward):

    python -m tracy -p -v -r --no-device-data-capture \\
        -o generated/profiler/reports/conformer_encoder_layer_traced \\
        models/demos/facebook_seamless_m4t_v2_large/tt/profile_conformer_encoder_layer.py --traced

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
from models.demos.facebook_seamless_m4t_v2_large.tt.conformer_encoder_layer import ConformerEncoderLayer

# Production shapes (SeamlessM4T-v2-Large).
EMBED_DIM = 1024
NUM_HEADS = 16
HEAD_DIM = 64
SEQ_LEN = 256  # DEFAULT_AUDIO_SEQ_LEN — post-feature-extraction time axis
LEFT_MAX = 64
RIGHT_MAX = 8
CONV_KSIZE = 31
EPS = 1e-5


def _make_state_dict(seed: int = 0) -> dict:
    """Build a representative state dict for one conformer encoder layer.

    Mirrors :class:`SeamlessM4Tv2ConformerEncoderLayer` weight layout.
    Shapes follow the v2-Large config: hidden=1024, intermediate=4096,
    conv kernel=31.
    """
    g = torch.Generator().manual_seed(seed)
    intermediate = 4096

    def _ln():
        return {
            "weight": torch.ones(EMBED_DIM),
            "bias": torch.zeros(EMBED_DIM),
        }

    def _ffn(prefix_seed):
        gp = torch.Generator().manual_seed(prefix_seed)
        return {
            "intermediate_dense": {
                "weight": torch.randn(intermediate, EMBED_DIM, generator=gp) * (1.0 / EMBED_DIM**0.5),
                "bias": torch.zeros(intermediate),
            },
            "output_dense": {
                "weight": torch.randn(EMBED_DIM, intermediate, generator=gp) * (1.0 / intermediate**0.5),
                "bias": torch.zeros(EMBED_DIM),
            },
        }

    def _attn(prefix_seed):
        gp = torch.Generator().manual_seed(prefix_seed)
        sd = {}
        for n in ("linear_q", "linear_k", "linear_v", "linear_out"):
            sd[n] = {
                "weight": torch.randn(EMBED_DIM, EMBED_DIM, generator=gp) * (1.0 / EMBED_DIM**0.5),
                "bias": torch.zeros(EMBED_DIM),
            }
        return sd

    def _conv():
        gp = torch.Generator().manual_seed(seed + 100)
        return {
            "layer_norm": _ln(),
            "pointwise_conv1": {
                # HF SeamlessM4Tv2ConformerSelfAttention conv: pointwise_conv1 is 1024 -> 2048
                "weight": torch.randn(2 * EMBED_DIM, EMBED_DIM, 1, generator=gp)
                * (1.0 / EMBED_DIM**0.5),
            },
            "depthwise_conv": {
                "weight": torch.randn(EMBED_DIM, 1, CONV_KSIZE, generator=gp) * (1.0 / CONV_KSIZE**0.5),
            },
            "depthwise_layer_norm": _ln(),
            "pointwise_conv2": {
                "weight": torch.randn(EMBED_DIM, EMBED_DIM, 1, generator=gp) * (1.0 / EMBED_DIM**0.5),
            },
        }

    sd = {
        "ffn1_layer_norm": _ln(),
        "ffn1": _ffn(seed + 1),
        "self_attn_layer_norm": _ln(),
        "self_attn": _attn(seed + 2),
        "conv_module": _conv(),
        "ffn2_layer_norm": _ln(),
        "ffn2": _ffn(seed + 3),
        "final_layer_norm": _ln(),
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


def _run_layer(device, layer, n_iter: int, use_trace: bool) -> List[float]:
    """Drive one ConformerEncoderLayer forward at production shapes (B=1, T=SEQ_LEN).

    Note: ConformerEncoderLayer.forward consumes its input (the first
    residual gets deallocated). For the untraced path we re-upload the
    activation per iteration. For the traced path we keep a single
    persistent input buffer alive and re-copy host data into it before
    each replay (the trace re-runs the same op sequence starting from
    that buffer).
    """
    g = torch.Generator().manual_seed(11)
    x_host = torch.randn(1, SEQ_LEN, EMBED_DIM, generator=g) * 0.02

    times: List[float] = []
    if use_trace:
        # Persistent input buffer used for capture AND replay.
        x_pers = _to_tt_bf16(device, x_host)

        # Warmup (compiles all kernels into the program cache). The
        # warmup MUST itself preserve x_pers because forward() runs a
        # deallocate on the residual. To survive, we shadow-copy the
        # warmup data via from_torch directly into a fresh tensor.
        warm = _to_tt_bf16(device, x_host)
        _ = layer.forward(warm)
        ttnn.synchronize_device(device)

        # Capture: x_pers is the input the trace will replay against.
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        out = layer.forward(x_pers)
        ttnn.end_trace_capture(device, tid, cq_id=0)
        ttnn.synchronize_device(device)

        # Replay loop: copy fresh data into x_pers each iter to keep it allocated.
        for _ in range(n_iter):
            x_pers = _to_tt_bf16(device, x_host)  # reallocate (trace replays the deallocate)
            t0 = time.perf_counter()
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
        ttnn.release_trace(device, tid)
    else:
        # Untraced: re-upload per iteration (forward consumes the input).
        warm = _to_tt_bf16(device, x_host)
        _ = layer.forward(warm)
        ttnn.synchronize_device(device)

        for _ in range(n_iter):
            tt_x = _to_tt_bf16(device, x_host)
            t0 = time.perf_counter()
            out = layer.forward(tt_x)
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
    parser.add_argument("--n-iter", type=int, default=20, help="Steady-state iteration count.")
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()

    print(
        f"[profile_conformer_encoder_layer] embed={EMBED_DIM} heads={NUM_HEADS} "
        f"head_dim={HEAD_DIM} T={SEQ_LEN} traced={args.traced} n_iter={args.n_iter}"
    )

    device = ttnn.open_device(
        device_id=args.device_id,
        l1_small_size=16384,
        trace_region_size=128_000_000 if args.traced else 0,
    )
    try:
        sd = _make_state_dict()
        # Distance-embedding table for relative_key: shape [L+R+1, head_dim].
        de_weight = torch.zeros(LEFT_MAX + RIGHT_MAX + 1, HEAD_DIM)

        layer = ConformerEncoderLayer(
            device=device,
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            head_dim=HEAD_DIM,
            seq_len=SEQ_LEN,
            state_dict=sd,
            distance_embedding_weight=de_weight,
            left_max_position_embeddings=LEFT_MAX,
            right_max_position_embeddings=RIGHT_MAX,
            position_embeddings_type="relative_key",
            conv_kernel_size=CONV_KSIZE,
            eps=EPS,
            batch_size=1,
            weight_dtype=ttnn.bfloat16,
        )

        times = _run_layer(device, layer, args.n_iter, args.traced)
        print(_stat("[conformer_encoder_layer] step_ms", times))
        print("[profile_conformer_encoder_layer] DONE")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
