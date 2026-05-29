# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Tracy profiling harness for the dots.ocr Qwen2 LM decoder layer (composite).

Profiles :class:`TtDecoderLayer` -- the pre-norm residual composite
(TtRMSNorm x2 + TtAttention GQA 12/2 + TtMLP SwiGLU + two residual adds) in
isolation under metal trace so the CSV reflects device-kernel time rather than
host dispatch. Uses the production-representative shapes from the seed-0 golden:
seq_len=128, hidden=1536, 12 query heads, 2 kv heads, head_dim 128,
intermediate 8960, rms_norm_eps 1e-6, RoPE theta 1e6, causal mask.

The leaf modules (attention -21.8 percent, mlp -7.5 percent, rmsnorm at-ceiling)
were already optimized; decoder_layer inherits those wins because it composes the
exact same modules by file-path import. This harness exists to check the
composite boundaries: the two residual ttnn.add ops and the norm-to-attn /
attn-to-mlp reshards.

Run under tracy::

    python3 -m tracy -p -v -r --op-support-count 50000 \
      models/demos/rednote_hilab_dots.ocr/tt/profile_decoder_layer.py --traced

The ops CSV lands in generated/profiler/reports/<TIMESTAMP>/ops_perf_results_*.csv.
"""
import argparse
import importlib.util
import os

import torch

import ttnn

_TT_DIR = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location(
    "dots_tt_decoder_layer_profile", os.path.join(_TT_DIR, "decoder_layer.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
TtDecoderLayer = _mod.TtDecoderLayer

# Production-representative shapes from the seed-0 golden.
SEQ = 128
HIDDEN = 1536
NUM_HEADS = 12
NUM_KV_HEADS = 2
HEAD_DIM = 128
INTERMEDIATE = 8960
EPS = 1e-6
ROPE_THETA = 1e6


def _rope_tables(seq, head_dim, theta):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(seq, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # [seq, head_dim//2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [seq, head_dim]
    return emb.cos(), emb.sin()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traced", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=50_000_000)
    try:
        input_layernorm_weight = torch.randn(HIDDEN, dtype=torch.float32)
        q_weight = torch.randn(NUM_HEADS * HEAD_DIM, HIDDEN, dtype=torch.float32) * 0.02
        k_weight = torch.randn(NUM_KV_HEADS * HEAD_DIM, HIDDEN, dtype=torch.float32) * 0.02
        v_weight = torch.randn(NUM_KV_HEADS * HEAD_DIM, HIDDEN, dtype=torch.float32) * 0.02
        q_bias = torch.randn(NUM_HEADS * HEAD_DIM, dtype=torch.float32) * 0.02
        k_bias = torch.randn(NUM_KV_HEADS * HEAD_DIM, dtype=torch.float32) * 0.02
        v_bias = torch.randn(NUM_KV_HEADS * HEAD_DIM, dtype=torch.float32) * 0.02
        o_weight = torch.randn(HIDDEN, NUM_HEADS * HEAD_DIM, dtype=torch.float32) * 0.02
        post_attention_layernorm_weight = torch.randn(HIDDEN, dtype=torch.float32)
        gate_weight = torch.randn(INTERMEDIATE, HIDDEN, dtype=torch.float32) * 0.02
        up_weight = torch.randn(INTERMEDIATE, HIDDEN, dtype=torch.float32) * 0.02
        down_weight = torch.randn(HIDDEN, INTERMEDIATE, dtype=torch.float32) * 0.02

        cos, sin = _rope_tables(SEQ, HEAD_DIM, ROPE_THETA)
        causal = torch.full((SEQ, SEQ), float("-inf"), dtype=torch.float32)
        causal = torch.triu(causal, diagonal=1)

        layer = TtDecoderLayer(
            device=device,
            input_layernorm_weight=input_layernorm_weight,
            q_weight=q_weight,
            k_weight=k_weight,
            v_weight=v_weight,
            q_bias=q_bias,
            k_bias=k_bias,
            v_bias=v_bias,
            o_weight=o_weight,
            post_attention_layernorm_weight=post_attention_layernorm_weight,
            gate_weight=gate_weight,
            up_weight=up_weight,
            down_weight=down_weight,
            cos=cos,
            sin=sin,
            attention_mask=causal,
            seq_len=SEQ,
            num_heads=NUM_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            eps=EPS,
        )

        host_in = torch.randn(SEQ, HIDDEN, dtype=torch.float32)
        x = ttnn.from_torch(
            host_in,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Warmup: compile the kernels into the program cache.
        for _ in range(3):
            out = layer(x)
        ttnn.synchronize_device(device)

        if args.traced:
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            out = layer(x)
            ttnn.end_trace_capture(device, tid, cq_id=0)
            ttnn.synchronize_device(device)
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(device)
            ttnn.release_trace(device, tid)
        else:
            out = layer(x)
            ttnn.synchronize_device(device)

        print("profile_decoder_layer done; out shape", tuple(out.shape))
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
