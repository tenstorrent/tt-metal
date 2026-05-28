#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tracy harness for the SeamlessM4T-v2 NLLB text decoder layer (AR-decode path).

The NLLB text decoder layer is the per-layer building block of the text decoder
(24 stacked instances in production). It runs inside the autoregressive decode
loop on EVERY decode step for all 5 use cases (T2TT, S2TT, ASR, T2ST, S2ST), so
optimising it multiplies across the entire model.

In production, the layer runs in its ``_forward_cached`` path:
  - hidden_size = 1024
  - decoder_attention_heads = 16, head_dim = 64
  - seq_len = 1 (single new decoder token per step)
  - self-attn KV-cache capacity = 128, cross-attn cache encoder_seq_len = 128
  - activation_function = "relu"  -- 1024 -> 8192 -> 1024 FFN
  - layer_norm_eps = 1e-5

The production critical path uses metal trace, so this harness MUST profile under
``--traced`` to get realistic device-kernel time (not host dispatch noise).

Run untraced (host dispatch dominates, useful only for sanity):

    source python_env/bin/activate && export TT_METAL_HOME=$(pwd) && \\
        export PYTHONPATH=$(pwd) && export ARCH_NAME=blackhole
    python models/demos/facebook_seamless_m4t_v2_large/tt/profile_text_decoder_layer.py

Run under tracy with metal trace (the production path inside the AR loop):

    python -m tracy -p -v -r --no-device-data-capture \\
        -o generated/profiler/reports/text_decoder_layer_traced \\
        models/demos/facebook_seamless_m4t_v2_large/tt/profile_text_decoder_layer.py --traced

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
from models.demos.facebook_seamless_m4t_v2_large.tt.kv_cache import CrossAttentionKVCache, SelfAttentionKVCache
from models.demos.facebook_seamless_m4t_v2_large.tt.text_decoder_layer import TextDecoderLayer

# Production shapes (SeamlessM4T-v2-Large).
EMBED_DIM = 1024
NUM_HEADS = 16
HEAD_DIM = 64
FFN_DIM = 8192
DEC_TGT_LEN = 1  # AR decode step Q length
ENC_SRC_LEN = 128  # encoder cache S (tile-padded)
MAX_DECODE_SEQ_LEN = 128  # self-attn KV cache capacity (tile-aligned)
EPS = 1e-5
BATCH = 1
LAYER_IDX = 0


def _make_state_dict(seed: int = 0) -> dict:
    """Build a representative state dict for one NLLB text decoder layer."""
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

    def _ffn(prefix_seed):
        gp = torch.Generator().manual_seed(prefix_seed)
        return {
            "fc1": {
                "weight": torch.randn(FFN_DIM, EMBED_DIM, generator=gp) * (1.0 / EMBED_DIM**0.5),
                "bias": torch.zeros(FFN_DIM),
            },
            "fc2": {
                "weight": torch.randn(EMBED_DIM, FFN_DIM, generator=gp) * (1.0 / FFN_DIM**0.5),
                "bias": torch.zeros(EMBED_DIM),
            },
        }

    return {
        "self_attn_layer_norm": _ln(),
        "self_attn": _attn(seed + 1),
        "cross_attention_layer_norm": _ln(),
        "cross_attention": _attn(seed + 2),
        "ffn_layer_norm": _ln(),
        "ffn": _ffn(seed + 3),
    }


def _to_tt_bf16(device, t: torch.Tensor) -> ttnn.Tensor:
    return ttnn.from_torch(
        t,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _populate_cross_attn(device, layer: TextDecoderLayer, cross_kv: CrossAttentionKVCache) -> None:
    """Pre-populate the cross-attn cache by projecting random encoder hidden states.

    Mirrors what the encoder-prefill step does in production: project the
    encoder hidden states through the layer's cross-attention K/V weights,
    then store via ``ttnn.fill_cache``.
    """
    g = torch.Generator().manual_seed(7)
    enc = torch.randn(BATCH, ENC_SRC_LEN, EMBED_DIM, generator=g) * 0.02
    tt_enc = _to_tt_bf16(device, enc)
    k, v = layer.cross_attention.project_kv(tt_enc)
    ttnn.deallocate(tt_enc)
    cross_kv.populate(LAYER_IDX, k, v)


def _run_dec_step(
    device,
    layer: TextDecoderLayer,
    self_kv: SelfAttentionKVCache,
    cross_kv,
    n_iter: int,
    use_trace: bool,
    position: int = 0,
) -> List[float]:
    """One traced AR decode step (the production critical path).

    Bundles self-attn + KV-cache write + cross-attn (cache read) + FFN — exactly
    what runs per token inside the AR loop.
    """
    g = torch.Generator().manual_seed(11)
    h_host = torch.randn(BATCH, DEC_TGT_LEN, EMBED_DIM, generator=g) * 0.02

    # Bundle for the cached path. The text_decoder_layer expects an object with
    # ``.self_attn`` and ``.cross_attn`` slots that have ``update``/``read``.
    class _PKVBundle:
        pass

    pkv = _PKVBundle()
    pkv.self_attn = self_kv
    pkv.cross_attn = cross_kv

    # Write the position once into the persistent pos buffer. Inside the
    # captured trace we pass the buffer itself (ttnn.Tensor) so no H2D copy
    # happens during trace capture / replay — required because the SelfAttn
    # KV-cache ``update()`` would otherwise do a ``copy_host_to_device_tensor``
    # to materialize the int into a device int32 tensor, and writes are
    # disallowed inside trace capture.
    pos_host = ttnn.from_torch(
        torch.tensor([int(position)] * BATCH, dtype=torch.int32),
        dtype=ttnn.int32,
    )
    ttnn.copy_host_to_device_tensor(pos_host, self_kv.get_persistent_pos_buffer())
    pos_tt = self_kv.get_persistent_pos_buffer()

    times: List[float] = []
    if use_trace:
        # Persistent input buffer.
        h_pers = _to_tt_bf16(device, h_host)

        # Warmup once — compiles all kernels into the program cache.
        out_warm = layer.forward(
            h_pers,
            encoder_hidden_states=None,
            self_attention_mask=None,
            encoder_attention_mask=None,
            past_key_values=pkv,
            position=pos_tt,
            layer_idx=LAYER_IDX,
        )
        ttnn.synchronize_device(device)
        ttnn.deallocate(out_warm)
        h_pers = _to_tt_bf16(device, h_host)  # re-alloc (forward consumes)

        # Capture.
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        out = layer.forward(
            h_pers,
            encoder_hidden_states=None,
            self_attention_mask=None,
            encoder_attention_mask=None,
            past_key_values=pkv,
            position=pos_tt,
            layer_idx=LAYER_IDX,
        )
        ttnn.end_trace_capture(device, tid, cq_id=0)
        ttnn.synchronize_device(device)

        # Replay loop.
        for _ in range(n_iter):
            h_pers = _to_tt_bf16(device, h_host)
            t0 = time.perf_counter()
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
        ttnn.release_trace(device, tid)
    else:
        # Untraced loop — host dispatch dominates.
        warm = _to_tt_bf16(device, h_host)
        _ = layer.forward(
            warm,
            encoder_hidden_states=None,
            self_attention_mask=None,
            encoder_attention_mask=None,
            past_key_values=pkv,
            position=pos_tt,
            layer_idx=LAYER_IDX,
        )
        ttnn.synchronize_device(device)

        for _ in range(n_iter):
            tt_h = _to_tt_bf16(device, h_host)
            t0 = time.perf_counter()
            out = layer.forward(
                tt_h,
                encoder_hidden_states=None,
                self_attention_mask=None,
                encoder_attention_mask=None,
                past_key_values=pkv,
                position=pos_tt,
                layer_idx=LAYER_IDX,
            )
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
        f"[profile_text_decoder_layer] embed={EMBED_DIM} heads={NUM_HEADS} "
        f"head_dim={HEAD_DIM} ffn={FFN_DIM} T={DEC_TGT_LEN} S={ENC_SRC_LEN} "
        f"max_dec_seq={MAX_DECODE_SEQ_LEN} traced={args.traced} n_iter={args.n_iter}"
    )

    device = ttnn.open_device(
        device_id=args.device_id,
        l1_small_size=16384,
        trace_region_size=64_000_000 if args.traced else 0,
    )
    try:
        sd = _make_state_dict()
        layer = TextDecoderLayer(
            device=device,
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            head_dim=HEAD_DIM,
            state_dict=sd,
            eps=EPS,
            weight_dtype=ttnn.bfloat16,
        )

        # One-layer self-attn and cross-attn caches.
        self_kv = SelfAttentionKVCache(
            device=device,
            num_layers=1,
            batch=BATCH,
            num_heads=NUM_HEADS,
            max_seq_len=MAX_DECODE_SEQ_LEN,
            head_dim=HEAD_DIM,
        )
        cross_kv = CrossAttentionKVCache(
            device=device,
            num_layers=1,
            batch=BATCH,
            num_heads=NUM_HEADS,
            encoder_seq_len=ENC_SRC_LEN,
            head_dim=HEAD_DIM,
        )
        _populate_cross_attn(device, layer, cross_kv)

        times = _run_dec_step(device, layer, self_kv, cross_kv, args.n_iter, args.traced, position=0)
        print(_stat("[text_decoder_layer dec_step] step_ms", times))
        print("[profile_text_decoder_layer] DONE")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
