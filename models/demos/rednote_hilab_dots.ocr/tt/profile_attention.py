# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Tracy profiling harness for the dots.ocr Qwen2 LM self-attention leaf block.

Profiles :class:`TtAttention` (fused QKV linear -> manual GQA head split via
slice/reshape/permute -> 1D RoPE -> repeat_kv expansion 2->12 -> causal SDPA
chain -> head merge -> output proj) in isolation under metal trace so the CSV
reflects device-kernel time rather than host dispatch. Uses the production
shapes from the seed-0 golden: seq_len=128, 12 query heads, 2 KV heads,
head_dim 128, hidden 1536, theta 1e6, causal additive mask.

Run under tracy::

    python3 -m tracy -p -v -r --op-support-count 50000 \
      models/demos/rednote_hilab_dots.ocr/tt/profile_attention.py --traced

The ops CSV lands in generated/profiler/reports/<TIMESTAMP>/ops_perf_results_*.csv.
"""
import argparse
import importlib.util
import os

import torch

import ttnn

_TT_DIR = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("dots_tt_attention_profile", os.path.join(_TT_DIR, "attention.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
TtAttention = _mod.TtAttention

# Production-representative shapes from the seed-0 golden.
SEQ = 128
HIDDEN = 1536
NUM_HEADS = 12
NUM_KV_HEADS = 2
HEAD_DIM = 128
THETA = 1e6


def _rope_tables(seq, head_dim, theta):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    pos = torch.arange(seq, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)  # [seq, head_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [seq, head_dim]
    return emb.cos(), emb.sin()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traced", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=50_000_000)
    try:
        kv_dim = NUM_KV_HEADS * HEAD_DIM
        q_weight = torch.randn(HIDDEN, HIDDEN, dtype=torch.float32) * 0.02
        k_weight = torch.randn(kv_dim, HIDDEN, dtype=torch.float32) * 0.02
        v_weight = torch.randn(kv_dim, HIDDEN, dtype=torch.float32) * 0.02
        q_bias = torch.randn(HIDDEN, dtype=torch.float32) * 0.02
        k_bias = torch.randn(kv_dim, dtype=torch.float32) * 0.02
        v_bias = torch.randn(kv_dim, dtype=torch.float32) * 0.02
        o_weight = torch.randn(HIDDEN, HIDDEN, dtype=torch.float32) * 0.02

        cos, sin = _rope_tables(SEQ, HEAD_DIM, THETA)

        # Causal additive mask [seq, seq] (0 on/below diagonal, -inf above).
        mask = torch.full((SEQ, SEQ), float("-inf"), dtype=torch.float32)
        mask = torch.triu(mask, diagonal=1)

        attn = TtAttention(
            device=device,
            q_weight=q_weight,
            k_weight=k_weight,
            v_weight=v_weight,
            q_bias=q_bias,
            k_bias=k_bias,
            v_bias=v_bias,
            o_weight=o_weight,
            cos=cos,
            sin=sin,
            attention_mask=mask,
            seq_len=SEQ,
            num_heads=NUM_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
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
            out = attn(x)
        ttnn.synchronize_device(device)

        if args.traced:
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            out = attn(x)
            ttnn.end_trace_capture(device, tid, cq_id=0)
            ttnn.synchronize_device(device)
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(device)
            ttnn.release_trace(device, tid)
        else:
            out = attn(x)
            ttnn.synchronize_device(device)

        print("profile_attention done; out shape", tuple(out.shape))
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
