#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill goldens for deepseek-ai/DeepSeek-V4-Pro-0813, layer by layer (engine.py), from transformers' DeepseekV4
code. Same architecture as V4-Flash: see generate_golden_deepseek_v4_flash.py for the layer kinds, mHC, hash MoE,
the dequantization and the bf16 dtype plan.

Stored tensors (N = isl, i = layer, E = 384 routed experts, k = min(1024, N // 4); 61 layers: 2
heavily_compressed_attention (HCA, ratio 128), then compressed_sparse_attention (CSA, ratio 4) and HCA alternating,
the last one CSA; no sliding-only layer; layers 0-2 hash_moe, 3-60 moe):

  hidden/embed.safetensors
    model.embed_tokens.output                              [1, N, 7168]       bf16  embeddings (no scale); the
                                                                                    model copies them into hc_mult=4
                                                                                    residual streams before layer 0
  hidden/layer_XX.safetensors
    model.layers.{i}.output                                [1, N, 4, 7168]    bf16  all   the 4 mHC residual streams
                                                                                    after attention and MoE
    model.layers.{i}.self_attn.output                      [1, N, 7168]       bf16  all   --sublayers: attention
                                                                                    after o_b_proj
    model.layers.{i}.mlp.output                            [1, N, 7168]       bf16  all   --sublayers: routed experts
                                                                                    + shared expert
    model.layers.{i}.attn_hc.post                          [1, N, 4]          fp32  all   --sublayers: mHC weights
    model.layers.{i}.attn_hc.comb                          [1, N, 4, 4]       fp32  all   placing the attention
    model.layers.{i}.attn_hc.collapsed                     [1, N, 7168]       bf16  all   output / mixing streams,
                                                                                    and its input (pre-norm)
    model.layers.{i}.ffn_hc.{post,comb,collapsed}          as attn_hc               all   --sublayers: same, MoE site
  cache/layer_XX.safetensors
    model.layers.{i}.self_attn.kv_norm.output              [1, N, 512]        bf16  all   shared K=V of every
                                                                                    position, pre-RoPE
    past_key_values.layers.{i}.keys                        [1, 1, 127, 512]   bf16  all   last sliding_window - 1
                                                                                    K=V, post RoPE (min(N, 127))
    past_key_values.layers.{i}.compressed_kv.compressor    [1, N // 4, 512]   bf16  CSA   compressed KV entries
                                                           [1, N // 128, 512] bf16  HCA
    past_key_values.layers.{i}.buffer_kv.compressor        [1, N % 4, 1024]   bf16  CSA   kv_proj of the tokens
                                                           [1, N % 128, 512]  bf16  HCA   after the last full window
    past_key_values.layers.{i}.buffer_gate.compressor      as buffer_kv       bf16  CSA, HCA  gate_proj of the same
    past_key_values.layers.{i}.overlap_kv.compressor       [1, 4, 512]        bf16  CSA   last window's Ca half
    past_key_values.layers.{i}.overlap_gate.compressor     [1, 4, 512]        fp32  CSA   its gate + position_bias
    past_key_values.layers.{i}.compressed_kv.indexer       [1, N // 4, 128]   bf16  CSA   indexer keys
    past_key_values.layers.{i}.buffer_kv.indexer           [1, N % 4, 256]    bf16  CSA   (the indexer's own
    past_key_values.layers.{i}.buffer_gate.indexer         [1, N % 4, 256]    bf16  CSA   compressor, same scheme
    past_key_values.layers.{i}.overlap_kv.indexer          [1, 4, 128]        bf16  CSA   at index_head_dim)
    past_key_values.layers.{i}.overlap_gate.indexer        [1, 4, 128]        fp32  CSA
    model.layers.{i}.self_attn.compressor.indexer.scorer.output
                                                           [1, N, N // 4]     fp32  CSA   index scores, pre mask
    model.layers.{i}.self_attn.compressor.indexer.output   [1, N, k]          int32 CSA   top-k compressed entries
                                                                                    per query, -1 = not causal
    model.layers.{i}.mlp.gate.output                       [N, 384]           fp32  all   router logits
    model.layers.{i}.mlp.experts.top_k_index               [N, 6]             int32 all   chosen experts (hash_moe:
                                                                                    tid2eid[token]), ascending
    model.layers.{i}.mlp.experts.top_k_weights             [N, 6]             fp32  all   their weights
  hidden/final.safetensors
    model.last_hidden_state                                [1, N, 7168]       bf16  norm(hc_head(streams))
    logits                                                 [1, 1, 129280]     fp32  last position, lm_head (untied)

What differs from Flash: only sizes (hidden 7168, 128 heads in 16 output groups, q_lora_rank 1536, 384 experts of
width 3072, index_topk 1024, routed_scaling_factor 2.5) and the layer schedule above. The checkpoint's compress_ratios
has 64 entries: 61 layers + 3 DSpark draft stages (mtp.0-2, a speculative-decoding head fed by the stream-mean of
layers dspark_target_layer_ids = 58, 59, 60). transformers builds neither, nor does this script: those inputs are
model.layers.{58,59,60}.output.mean(dim=2) of a full-depth trace. The weights are stored like Flash's (FP8 e4m3
128x128 + FP4 e2m1 experts 1x32, ue8m0 scales), one shard per layer, ~13.9 GB each, streamed and deleted.

    python generate_golden_deepseek_v4_pro_0813.py --isl 10240
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_golden_deepseek_v4_flash import DeepseekV4Golden  # noqa: E402


class DeepseekV4ProGolden(DeepseekV4Golden):
    hf_repo = "deepseek-ai/DeepSeek-V4-Pro-0813"
    recipe = dataclasses.replace(
        DeepseekV4Golden.recipe,
        config=(*DeepseekV4Golden.recipe.config, "dspark_target_layer_ids", "dspark_block_size"),
    )


if __name__ == "__main__":
    sys.exit(DeepseekV4ProGolden().main())
