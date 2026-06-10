<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
SPDX-License-Identifier: Apache-2.0
-->

# dots.ocr (rednote-hilab/dots.ocr) Architecture Analysis

## Model Family

Vision-Language model (VLM) for document OCR. Single HF class
`DotsOCRForCausalLM` = Qwen2ForCausalLM (1.5B-class decoder) + custom
`DotsVisionTransformer` (NaViT-style, ~1.26B). Total ~3.04B params,
~6.1 GB bf16.

## Config summary

| | Vision tower | Text decoder |
|---|---|---|
| Layers | 42 | 28 |
| Hidden | 1536 (embed_dim) | 1536 |
| Heads | 12 MHA, head_dim 128 | 12 Q / 2 KV (GQA), head_dim 128 |
| FFN | SwiGLU 4224, no bias | SwiGLU 8960, no bias |
| Norm | RMSNorm eps 1e-5 (post_norm on) | RMSNorm eps 1e-6 |
| Pos enc | 2D vision RoPE | RoPE theta 1e6, 1D positions |
| Other | patch 14, spatial_merge 2, temporal 1, qkv fused no bias | QKV bias=true, vocab 151936, untied lm_head |

## Similar Implementations

| Component | Reference Implementation | Similarity |
|-----------|-------------------------|------------|
| vision_patch_embed | models/demos/qwen25_vl/tt/model.py | Conv patchify + norm; dots uses Conv2d (temporal=1) + RMSNorm |
| vision_attention | models/demos/qwen25_vl/tt/vision_attention.py | fused QKV + 2D RoPE + cu_seqlens mask; dots is full attention every layer, no bias |
| vision_mlp | models/demos/qwen25_vl/tt/vision_mlp.py | identical SwiGLU pattern |
| vision_rmsnorm | models/demos/qwen25_vl/tt/vision_rmsnorm.py | identical |
| vision_block | models/demos/qwen25_vl/tt/vision_block.py | pre-norm residual block |
| patch_merger | models/demos/qwen25_vl/tt/patch_merger.py | LayerNorm + 2-layer GELU MLP, merge 2x2 |
| embedding | models/tt_transformers/tt/embedding.py | standard vocab embedding |
| text_attention | models/tt_transformers/tt/attention.py | Qwen2 GQA w/ QKV bias |
| text_mlp | models/tt_transformers/tt/mlp.py | SwiGLU |
| text_rmsnorm | models/common/rmsnorm.py | RMSNorm |
| decoder_layer | models/tt_transformers/tt/decoder.py | pre-norm decoder |
| lm_head | models/tt_transformers/tt/lm_head.py | vocab-sharded head |

## Key Differences

- Vision tower is **full attention in all 42 layers** (qwen25_vl alternates
  windowed/full) — simpler: one mask shape per image grid.
- Vision uses **RMSNorm** (not LayerNorm) and **SwiGLU** FFN with no biases;
  merger pre-norm is LayerNorm eps 1e-6.
- Text decoder is plain Qwen2: **QKV bias true**, no QK-norm, 12/2 GQA;
  multimodal fusion is masked-scatter of merged vision embeds at
  `image_token_id=151665` (host-side scatter, prefill input).

## Weight Mapping

| HuggingFace Key | TTNN Key |
|-----------------|----------|
| vision_tower.patch_embed.patchifier.proj.{weight,bias} | vision.patch_embed.proj |
| vision_tower.patch_embed.patchifier.norm.weight | vision.patch_embed.norm |
| vision_tower.blocks.{i}.attn.qkv.weight | vision.blocks.{i}.attn.wqkv |
| vision_tower.blocks.{i}.attn.proj.weight | vision.blocks.{i}.attn.wo |
| vision_tower.blocks.{i}.mlp.fc1/fc3/fc2.weight | vision.blocks.{i}.mlp.w1/w3/w2 |
| vision_tower.blocks.{i}.norm1/norm2.weight | vision.blocks.{i}.norm1/norm2 |
| vision_tower.post_trunk_norm.weight | vision.post_norm |
| vision_tower.merger.ln_q.{weight,bias} | vision.merger.ln_q |
| vision_tower.merger.mlp.0/2.{weight,bias} | vision.merger.fc1/fc2 |
| model.embed_tokens.weight | embedding.weight |
| model.layers.{i}.self_attn.{q,k,v}_proj.{weight,bias} | layers.{i}.attention.w{q,k,v} |
| model.layers.{i}.self_attn.o_proj.weight | layers.{i}.attention.wo |
| model.layers.{i}.mlp.{gate,up,down}_proj.weight | layers.{i}.feed_forward.w{1,3,2} |
| model.layers.{i}.input_layernorm / post_attention_layernorm | layers.{i}.attention_norm / ffn_norm |
| model.norm.weight / lm_head.weight | norm / lm_head |

## Parallelism plan (qb, 1x4 Blackhole mesh, 34.2 GB DRAM/chip)

Computed via `skills.orchestrator.lib.parallelism.plan_parallelism`
(total 6.08 GB bf16 vs 17.1 GB per-chip budget → fits replicated;
no judgments returned).

| Component group | Placement | Rationale |
|---|---|---|
| vision_tower (all vision_* + patch_merger) | replicate | run-once per input; cannot amortize CCL; replicated output = no boundary CCL into TP decoder |
| embedding | shard (vocab dim) | per-token cadence |
| decoder_stack (text_attn/mlp/norm x28) | shard 4-way, kv_replication=2 | per-token; 12 Q heads / 4 chips = 3 each; 2 KV heads < 4 devices → replicate each KV head x2, chip-local SDPA |
| lm_head | shard (vocab dim) | per-token; concat logits on host or all_gather |

## Implementation Order

1. vision_patch_embed, vision_rmsnorm, embedding, text_rmsnorm (leaves)
2. vision_attention, vision_mlp / text_attention, text_mlp
3. vision_block / decoder_layer
4. patch_merger, lm_head
5. vision_transformer (full tower)

## Use cases

| Name | Input | Output | needs_ar | HF class | Metric | Threshold | Components used |
|------|-------|--------|----------|----------|--------|-----------|------------------|
| ocr | image | text | true | DotsOCRForCausalLM | wer | HF + 0.05 | all 13 components |
