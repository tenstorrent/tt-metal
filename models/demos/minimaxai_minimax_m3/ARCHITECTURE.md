# MiniMax-M3 (`MiniMaxAI/MiniMax-M3`) Architecture Analysis

## Model Family
Multimodal **Vision-Language sparse Mixture-of-Experts** decoder (`model_type = minimax_m3_vl`,
top class `MiniMaxM3SparseForConditionalGeneration`). A CLIP-style vision tower + multimodal
projector feed a 60-layer DeepSeek-V3-style sparse-MoE text decoder.

> **Snapshot caveat:** the HF snapshot has **no `modeling_*.py`** (`auto_map` only declares
> `AutoConfig`). The forward below is reconstructed from `config.json` + the published
> MiniMax-M3 / DeepSeek-V3 architecture and the weight layout in
> `model.safetensors.index.json` (23,416 tensors, 869 GB bf16, 59 shards).
> **MTP modules** (`num_mtp_modules=7`) are declared in config but **no MTP/nextn weights
> exist in the index** — the MTP head is config-only and not buildable from this checkpoint.

## Text backbone (`text_config`)
- 60 layers, hidden 6144, 64 attn heads, GQA `num_kv_heads=4`, head_dim 128.
- Partial rotary: `rotary_dim 64`, `partial_rotary_factor 0.5`, `rope_theta 5e6`, max ctx 1,048,576.
- Per-head QK-norm (`use_qk_norm`, `qk_norm_type=per_head`), gemma-style RMSNorm (`use_gemma_norm`, eps 1e-6).
- Activation **swigluoai** (`alpha 1.702`, `limit 7.0`): `down((clamp(g,max=L)·σ(α·g))·(clamp(u,±L)+1))`.
- **First 3 layers DENSE** (`moe_layer_freq[0:3]=0`, dense intermediate 12288, full GQA attention);
  **layers 3..59 MoE** (`block_sparse_moe`) with **block-sparse/lightning attention**.
- MoE: 128 routed experts (intermediate 3072), top-4, 1 shared expert (3072), sigmoid scoring +
  `e_score_correction_bias` (routing bias), `routed_scaling_factor 2.0`.
- Sparse attention (layers 3..59): learned indexer heads (`index_q_proj/index_k_proj` +
  `index_q_norm/index_k_norm`, `sparse_num_index_heads 4`, `sparse_index_dim 128`) pick top
  `sparse_topk_blocks 16` blocks of `sparse_block_size 128` (score `max`, init_block 0, local_block 1).
- vocab 200064, `tie_word_embeddings=false` (separate `lm_head`).

## Vision tower (`vision_config`, `clip_vision_model`)
- 32 layers, hidden 1280, 16 heads, patch 14, image 2016, intermediate 5120, gelu, bias on q/k/v/out & MLP.
- **3D RoPE** (`rope_mode=3d`, theta 1e4) over temporal/H/W. patch_embedding (Conv2d) + `pre_layrnorm`.
- patch_merge compression (`spatial_merge_size 2`, `temporal_patch_size 2`) -> `patch_merge_mlp` ->
  `multi_modal_projector` (linear_1 -> gelu -> linear_2, into the 6144 text space). `image_seq_length 576`.

## Similar Implementations (reference reuse)
| Component | Reference Implementation | Similarity / Notes |
|-----------|--------------------------|--------------------|
| RMSNorm / QK-norm / final norm | `models/demos/deepseek_v3/tt/rms_norm` | gemma-style RMSNorm, per-head variant for QK |
| Embedding | `models/demos/deepseek_v3/tt/embedding` | 200064 x 6144 |
| Partial RoPE | `models/demos/deepseek_v3/tt/rope.py` | rotary_dim 64 split (MLA does partial rope) |
| GQA (dense layers) | `models/demos/llama3_70b_galaxy/tt/llama_attention.py` | 64q/4kv, partial rope, KV cache |
| SwiGLU-OAI MLP / experts | `models/demos/gpt_oss/tt/mlp.py`, `.../tt/experts` | exact `α=1.702`, clamp `limit` GLU + expert/TP allreduce |
| MoE router | `models/demos/deepseek_v3/tt/moe_gate.py` | sigmoid + correction bias + routed_scaling (2.0 vs 2.5) |
| Shared expert | `models/demos/deepseek_v3/tt/mlp` | always-on expert summed with routed |
| MoE decoder block | `models/demos/deepseek_v3/tt/decoder_block` | gate+experts+shared (attention sub-block differs) |
| LM head | `models/demos/deepseek_v3/tt/lm_head1d.py` | untied 6144 -> 200064 |
| Vision attn / MLP / LN / block / merger | `models/demos/qwen3_vl/tt/{vision_attention,vision_mlp,vision_layernorm,vision_block,patch_merger}.py` | CLIP MHA + gelu MLP + LayerNorm + patch merge |
| MTP head (architecture only) | `models/demos/deepseek_v3/tt/mtp.py` | DeepSeek-V3-style; **no weights in this checkpoint** |

## Key Differences / Blocked primitives
- **`sparse_lightning_attention` (BLOCKED, layers 3..59):** learned block-sparse indexer + top-k block
  SDPA. No existing ttnn block-sparse/lightning reference with a learned indexer. Bring-up fallback:
  run full GQA on these layers for numerical parity, accepting reduced long-context efficiency.
- **`vision_rope_3d` (BLOCKED):** temporal+spatial 3D RoPE; qwen3_vl/qwen25_vl have 2D/mrope only.
- **`vision_encoder` (BLOCKED, composite):** the assembled tower can't point at one reference because
  3D RoPE is missing, even though most sub-blocks reuse qwen3_vl.
- **`mtp_head` (BLOCKED + ABSENT):** declared in config, no weights in the safetensors index.
- swigluoai is GPT-OSS's exact activation (reuse gpt_oss); MoE router is DeepSeek-V3's family (value diffs only).

## Tensor-Parallel plan (bh_galaxy, TP=32, mesh (1,32), Linear topology)
Computed via `skills/orchestrator/lib/parallelism.plan_parallelism(facts, 32, dram=34.2 GB/chip)`.

**Param budget (bf16):** model **869 GB** (index `total_size`) vs aggregate DRAM 32 × 34.23 = **1095 GB**,
but per-chip is only 34.23 GB. MoE dominates: ~14.5 GB experts per MoE-layer × 57 ≈ 846 GB.
=> `fits_replicated = false` — **everything must shard**, including the run-once vision encoder.

| Component | Cadence | Placement | Rationale |
|-----------|---------|-----------|-----------|
| embedding | per_token | shard | CCLs amortize over decode loop |
| dense_decoder_layer (×3) | per_token | shard + **kv_replication ×8** | per-token; `kv_heads=4 < 32` -> replicate each KV head ×8 so SDPA is chip-local; 64 q-heads shard 2/chip |
| moe_decoder_layer (×57) | per_token | shard + **kv_replication ×8** | per-token; experts **expert-parallel ≈4 experts/chip**; same KV replication |
| lm_head | per_token | shard | per-token; column-parallel, all-gather logits |
| vision_encoder | per_input | **shard (forced)** | run-once, but replicated model exceeds per-chip DRAM -> sharding forced |

**Key TP outcomes:**
- **(a) GQA KV replication:** 4 KV heads < 32 devices -> each KV head + its cache replicated across an
  8-device group; the 64 Q heads shard 32-ways so SDPA stays chip-local (no cross-chip attention CCL).
- **(b) MoE:** 128 experts / 32 chips -> **expert-parallel ≈4 experts/chip**. Within an expert the
  gate/up are column-parallel and down (w2) is row-parallel; chosen over pure TP-within-expert because
  expert-parallel keeps each expert's matmul chip-local and only the routed combine needs an all-to-all
  + the down-proj all-reduce. The `gpt_oss/tt/experts` path supports both expert- and tensor-parallel
  allreduce — perf phase A/Bs the routed combine.
- **(c) Vision encoder:** normally a run-once `per_input` block one would replicate, but here it is
  forced to shard because the replicated model does not fit per chip; perf phase confirms shard vs the
  (infeasible) replicate.
- **(d) Row-parallel reductions:** `o_proj`, dense/shared `down_proj`, and expert `w2` produce partial
  sums needing an **all-reduce (or reduce-scatter + all-gather)** over the 32-chip Linear fabric; the
  `lm_head` needs an all-gather of column-parallel logits.

## Weight Mapping (HF -> TTNN intent)
| HuggingFace key | TTNN intent |
|-----------------|-------------|
| `language_model.model.embed_tokens.weight` | embedding |
| `language_model.model.layers.{i}.self_attn.{q,k,v,o}_proj.weight` | attention wq/wk/wv/wo |
| `language_model.model.layers.{i}.self_attn.{q,k}_norm.weight` | per-head QK-norm |
| `language_model.model.layers.{i}.self_attn.index_{q,k}_proj.weight`, `index_{q,k}_norm.weight` | sparse indexer (blocked) |
| `language_model.model.layers.{0,1,2}.mlp.{gate,up,down}_proj.weight` | dense SwiGLU-OAI MLP |
| `....block_sparse_moe.gate.weight`, `.e_score_correction_bias` | router gate + bias |
| `....block_sparse_moe.experts.{e}.{w1,w3,w2}.weight` | routed expert gate/up/down |
| `....block_sparse_moe.shared_experts.{gate,up,down}_proj.weight` | shared expert |
| `language_model.model.norm.weight` | final RMSNorm |
| `language_model.lm_head.weight` | LM head |
| `vision_tower.vision_model.embeddings.patch_embedding.weight` | conv patch embed |
| `vision_tower.vision_model.encoder.layers.{i}.{layer_norm1,layer_norm2,self_attn.*,mlp.*}` | CLIP encoder layer |
| `patch_merge_mlp.linear_{1,2}.{weight,bias}` | patch-merge MLP |
| `multi_modal_projector.linear_{1,2}.{weight,bias}` | multimodal projector |

## Implementation Order (bottom-up)
1. rms_norm, vision_layernorm, embedding, rope (partial), swigluoai_mlp (reuse gpt_oss), moe_gate (reuse deepseek).
2. qk_norm, gqa_attention (dense layers), moe_experts + shared_expert, vision_mlp/attention.
3. dense_decoder_layer (×3) end-to-end (no blocked deps) — earliest verifiable text path.
4. **BLOCKED**: sparse_lightning_attention, vision_rope_3d (new primitives) — fall back to full GQA /
   stub vision rope for first bring-up.
5. moe_decoder_layer, vision_encoder_layer, patch_merge_mlp, multimodal_projector, vision_encoder.
6. final_norm, lm_head -> text_generation use case; then image/video use cases.
7. mtp_head: skip (no weights in checkpoint).

## Use cases

| Name | Input | Output | needs_ar | HF class | Metric | Threshold | Components used |
|------|-------|--------|----------|----------|--------|-----------|------------------|
| text_generation | text | text | true | MiniMaxM3SparseForConditionalGeneration | perplexity | HF + 0.10 | embedding, rms_norm, rope, qk_norm, gqa_attention, sparse_lightning_attention, swigluoai_mlp, moe_gate, moe_experts, shared_expert, dense_decoder_layer, moe_decoder_layer, final_norm, lm_head |
| image_text_to_text | image | text | true | MiniMaxM3SparseForConditionalGeneration | perplexity | HF + 0.10 | vision_* + patch_merge_mlp + multimodal_projector + vision_encoder + (full text stack) |
| video_text_to_text | video | text | true | MiniMaxM3SparseForConditionalGeneration | perplexity | HF + 0.10 | vision_* (temporal axis) + projector + (full text stack) |

(See `architecture_inventory.json` for the exact `components_used[]` per use case.)
