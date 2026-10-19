#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill goldens for XiaomiMiMo/MiMo-V2.6-Pro-RL, layer by layer (engine.py), from the repo's own remote code
(modeling_mimo_v2.py at the pinned commit, trust_remote_code).

The Pro model is MiMo-V2.6-Flash-RL scaled up, with byte-identical remote code (modeling / configuration
mimo_v2) and the same checkpoint formats, so this is generate_golden_mimo_v2_6_flash_rl.py with the Pro repo; see
its header for the model's peculiarities (partial RoPE, sinks on sliding layers, value scale, MoE gate, FP8 / MXFP4
formats, eager attention, the cache). What differs:

  * 70 layers, hidden 6144, 128 query heads and 8 KV heads on BOTH attention kinds (Flash: 64 / 4 full, 8 sliding),
    head_dim 192 / v 128 as Flash; 384 routed experts top-8, no shared expert; attention_value_scale 0.612;
    RMSNorm eps 1e-5. hybrid_layer_pattern puts full attention on 0, 7, 15, 23, 31, 39, 47, 55, 62, 69.
  * qkv_proj is stored as 8 tensor-parallel shards (Flash: 4), each [16 q | 1 k | 1 v] heads with its own 128-row
    block grid: [216, 48] scales for 27136 rows, which only 8 shards give (1, 2 or 4 shards: 212; 16: 224).

Stored tensors (N = isl, i = layer):

  hidden/embed.safetensors
    model.embed_tokens.output                  [1, N, 6144]       bf16  token embeddings (no scale)
  hidden/layer_XX.safetensors
    model.layers.{i}.output                    [1, N, 6144]       bf16  all   h + mlp(post_attn_norm(h)),
                                                                        h = x + attn(input_norm(x))
    model.layers.{i}.self_attn.qkv_proj.output [1, N, 27136]      bf16  all   --sublayers: fused projection
                                                                        [q | k | v], before RoPE / v scale
    model.layers.{i}.self_attn.o_proj.output   [1, N, 6144]       bf16  all   --sublayers: attention after o_proj
    model.layers.{i}.mlp.output                [1, N, 6144]       bf16  all   --sublayers: dense MLP (layer 0) or
                                                                        routed-experts sum (no shared expert)
  cache/layer_XX.safetensors
    past_key_values.layers.{i}.keys            [1, 8, N, 192]     bf16  full     RoPE theta 1e7 on dims [0:64] only
                                                                  sliding  RoPE theta 1e4 on dims [0:64] only;
                                                                        every position (the device keeps the last 128)
    past_key_values.layers.{i}.values          [1, 8, N, 128]     bf16  all   v_proj output times
                                                                        attention_value_scale (0.612), no RoPE
    model.layers.{i}.mlp.gate.router_logits    [N, 384]           fp32  1-69  hidden @ gate.weight.T in fp32, before
                                                                        sigmoid / e_score_correction_bias
    model.layers.{i}.mlp.gate.top_k_index      [N, 8]             int32 1-69  chosen experts, sorted by expert id
    model.layers.{i}.mlp.gate.top_k_weights    [N, 8]             fp32  1-69  their weights, same order: sigmoid
                                                                        scores renormalized to sum 1
  hidden/final.safetensors
    model.last_hidden_state                    [1, N, 6144]       bf16  final norm
    logits                                     [1, 1, 152576]     fp32  last position, untied lm_head

MTP layers (model.mtp.*), the vision and audio towers and the speech embeddings are never built nor read.

Long ISL: eager attention materializes [128 heads, query rows, N] scores, so pass --attn-chunk (Flash, with 64
heads, used 1024 at isl 56320; here half that per row budget).

    python generate_golden_mimo_v2_6_pro_rl.py --isl 10240 --prefill-chunk 5120
    python generate_golden_mimo_v2_6_pro_rl.py --isl 256 --num-layers 2 --root /tmp/smoke   # dense+full, MoE+SWA
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_golden_mimo_v2_6_flash_rl import NOTES as FLASH_NOTES  # noqa: E402
from generate_golden_mimo_v2_6_flash_rl import MiMoV2Golden  # noqa: E402

NOTES = {
    **FLASH_NOTES,
    "model.layers.{i}.self_attn.qkv_proj.output": (
        "Fused attention projection (attention_projection_layout fused_qkv), split [q | k | v] = "
        "[128*192 | 8*192 | 8*128] on both attention kinds; before RoPE and the value scale. The checkpoint stores "
        "this weight as 8 TP shards [q_r | k_r | v_r]; this is the regrouped layout."
    ),
    "past_key_values.layers.{i}.keys": (
        "K for every position, after RoPE. RoPE (half-split rotate_half) covers only dims [0:64] = int(head_dim 192 "
        "* partial_rotary_factor 0.334) of each head, dims [64:192] pass through. [1, 8, N, 192] on both kinds: "
        "full_attention rope_theta 1e7, sliding_attention swa_rope_theta 1e4 (the device only needs the last "
        "sliding_window (128) positions). Attention scaling is 192**-0.5. Sliding layers add an attention sink: "
        "self_attn.attention_sink_bias [128] (checkpoint weight, one logit per query head) is appended as an extra "
        "key column to the scaled, masked scores, the row max is subtracted, softmax in fp32, then the sink column "
        "is dropped (probabilities do not renormalize). Full layers have no sink."
    ),
    "past_key_values.layers.{i}.values": (
        "V for every position, the v slice of qkv_proj times attention_value_scale (0.612) in the compute dtype, no "
        "RoPE. [1, 8, N, 128] on both attention kinds."
    ),
}


class MiMoV26ProGolden(MiMoV2Golden):
    hf_repo = "XiaomiMiMo/MiMo-V2.6-Pro-RL"
    qkv_tp = 8
    # The chat template up to the assistant turn (thinking off, as the template renders it), the user turn naming the
    # WikiText format. Wikitext ppl over the same 5000 tokens at isl 5120: 2.087 (top-1 80.7%) with it, 2.114 after
    # <|endoftext|> (the checkpoint has no BOS), 2.125 raw. Flash-RL used none (4.779 raw, 4.780 chat).
    prefix = (
        "<|im_start|>user\nContinue this WikiText-2 dump verbatim: a sequence of Wikipedia articles, one after "
        "another, each starting with a = Title = line, in raw formatting ( @-@ , = = headings = = ).<|im_end|>"
        "<|im_start|>assistant\n<think></think>"
    )
    recipe = dataclasses.replace(MiMoV2Golden.recipe, notes=NOTES)


if __name__ == "__main__":
    sys.exit(MiMoV26ProGolden().main())
