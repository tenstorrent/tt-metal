# rednote-hilab/dots.ocr Architecture Analysis

## Model Family
**Vision-Language Model (VLM)** for OCR / document parsing.
HF arch `DotsOCRForCausalLM` (custom, `trust_remote_code=True`, `model_type="dots_ocr"`).

It is a **Qwen2.5-VL-style** VLM:
- **Vision tower** (`DotsVisionTransformer`, `model_type="dots_vit"`): a NaViT/Qwen2.5-VL-style ViT operating on packed variable-resolution patches (`cu_seqlens` block-diagonal attention), with **2D rotary position embeddings**, **fused QKV**, **RMSNorm**, **SwiGLU vision FFN**, and a **PatchMerger** (LayerNorm + GELU MLP, spatial_merge 2) projecting to the LM hidden size.
- **Language model** (`Qwen2ForCausalLM`): a standard **Qwen2** decoder — token embedding, 28 decoder layers (GQA, RoPE, RMSNorm, SwiGLU SiLU MLP, QKV bias), final RMSNorm, untied lm_head.
- The vision embeddings are scattered into the text embedding stream at `<|imgpad|>` (`image_token_id=151665`) positions via `masked_scatter`, then the merged sequence is decoded autoregressively.

## Config (verified from HF repo, snapshot c0111ce6)

### Language model (Qwen2)
| Field | Value |
|-------|-------|
| hidden_size | 1536 |
| intermediate_size | 8960 |
| num_hidden_layers | 28 |
| num_attention_heads | 12 |
| num_key_value_heads | 2 (GQA, head_dim 128) |
| vocab_size | 151936 |
| max_position_embeddings | 131072 |
| rope_theta | 1000000 |
| rms_norm_eps | 1e-6 |
| hidden_act | silu (SwiGLU) |
| attention_bias | true (q/k/v_proj carry bias) |
| tie_word_embeddings | false |
| torch_dtype | bfloat16 |

### Vision tower (dots_vit)
| Field | Value |
|-------|-------|
| embed_dim | 1536 |
| hidden_size (post-merge) | 1536 |
| intermediate_size | 4224 (SwiGLU FFN) |
| num_hidden_layers | 42 |
| num_attention_heads | 12 (head_dim 128, full MHA, fused QKV) |
| patch_size | 14 |
| spatial_merge_size | 2 |
| temporal_patch_size | 1 |
| num_channels | 3 |
| rms_norm_eps | 1e-5 |
| use_bias | false (vision qkv/proj/FFN unbiased) |
| post_norm | true (RMSNorm after trunk) |
| is_causal | false (bidirectional, block-diagonal per image) |
| rotary | 2D VisionRotaryEmbedding(head_dim//2, theta 10000) |

## Similar Implementations (TTNN references)

### Vision tower — reference: `models/demos/qwen25_vl`
Qwen2.5-VL's vision encoder is architecturally the closest existing TTNN
implementation: packed patches with `cu_seqlens`, 2D RoPE, fused QKV, RMSNorm,
SwiGLU vision FFN, and a PatchMerger. `models/demos/qwen3_vl` is an alternate
reference with the same block layout.

| Component | Reference Implementation | Similarity / Delta |
|-----------|--------------------------|--------------------|
| Patch embed | `models/demos/qwen25_vl/tt/model.py` | Conv2d(3,1536,k14,s14)+RMSNorm; Qwen-VL patchify pattern (host-side im2col). dots has temporal_patch_size=1. |
| Vision RMSNorm | `models/demos/qwen25_vl/tt/vision_rmsnorm.py` | RMSNorm eps 1e-5 (custom RMSNorm class, not LayerNorm). |
| Vision attention | `models/demos/qwen25_vl/tt/vision_attention.py` | Full bidirectional MHA, fused QKV, 2D RoPE, block-diagonal mask via cu_seqlens; **no bias** (use_bias=false). |
| Vision MLP | `models/demos/qwen25_vl/tt/vision_mlp.py` | SwiGLU `fc2(silu(fc1(x))*fc3(x))`, no bias. Qwen2.5-VL vision uses gated MLP — match. |
| Vision block | `models/demos/qwen25_vl/tt/vision_block.py` | Pre-norm: `h+=attn(norm1(h)); h+=mlp(norm2(h))`. |
| Patch merger | `models/demos/qwen25_vl/tt/patch_merger.py` | **ln_q is LayerNorm (with bias)** + `Linear→GELU→Linear` (mlp.0/mlp.2, with bias). spatial_merge 2 → 4x concat. NOTE: dots merger uses LayerNorm not RMSNorm; see qwen3_vl/tt/vision_layernorm.py if a LayerNorm primitive is needed. |
| Vision RoPE | `models/demos/qwen25_vl/tt/rope.py` | 2D vision rotary, theta 10000, head_dim//2. |
| Vision tower glue | `models/demos/qwen25_vl/tt/model.py` | trunk loop + post_trunk_norm (RMSNorm) + merger. |

### Language model — reference: `models/tt_transformers`
The LM is plain Qwen2; the `tt_transformers` library is the canonical
Qwen2/Llama-family decoder backbone (the qwen demos build on it).

| Component | Reference Implementation | Similarity / Delta |
|-----------|--------------------------|--------------------|
| Embedding | `models/tt_transformers/tt/embedding.py` | vocab 151936 → 1536. |
| Decoder RMSNorm | `models/common/rmsnorm.py` | RMSNorm eps 1e-6. |
| RoPE | `models/tt_transformers/tt/rope.py` | 1D rotary, theta 1e6, no scaling. |
| Attention | `models/tt_transformers/tt/attention.py` | GQA 12/2, head_dim 128, **QKV bias (attention_bias=true)**, o_proj no bias, KV-cache. |
| MLP | `models/tt_transformers/tt/mlp.py` | SwiGLU SiLU, gate/up/down 1536↔8960, no bias. |
| Decoder layer | `models/tt_transformers/tt/decoder.py` | input_layernorm + attn + post_attention_layernorm + mlp. |
| LM head | `models/tt_transformers/tt/lm_head.py` | untied, 1536 → 151936. |
| LM glue + generate | `models/tt_transformers/tt/model.py` | full prefill/decode + KV cache + AR generation. |

## Key Differences vs references
- **dots.ocr LM carries QKV bias** (Qwen2 `attention_bias=true`); ensure the
  tt_transformers attention loads q/k/v_proj.bias.
- **Vision tower has no bias anywhere except the PatchMerger** (qkv/proj/fc1-3
  unbiased; merger ln_q + mlp.0/mlp.2 biased).
- **PatchMerger uses LayerNorm (`ln_q`, eps 1e-6), not RMSNorm.** Most other
  vision norms in this model are RMSNorm.
- Vision attention is **bidirectional, block-diagonal per packed image**
  (`cu_seqlens`), `is_causal=false` — not causal.
- Two RoPE schemes coexist: 2D vision RoPE (theta 1e4) and 1D LM RoPE (theta 1e6).

## Weight Mapping (verified from model.safetensors.index.json)
| HuggingFace Key | Component |
|-----------------|-----------|
| `vision_tower.patch_embed.patchifier.proj.{weight,bias}` / `.norm.weight` | vision_patch_embed |
| `vision_tower.blocks.{i}.norm1.weight` / `.norm2.weight` | vision_rmsnorm |
| `vision_tower.blocks.{i}.attn.qkv.weight` / `.attn.proj.weight` | vision_attention |
| `vision_tower.blocks.{i}.mlp.fc1/fc2/fc3.weight` | vision_mlp |
| `vision_tower.post_trunk_norm.weight` | vision_tower (post norm) |
| `vision_tower.merger.ln_q.{weight,bias}` / `.mlp.0.{w,b}` / `.mlp.2.{w,b}` | vision_patch_merger |
| `model.embed_tokens.weight` | embedding |
| `model.layers.{i}.self_attn.{q,k,v}_proj.{weight,bias}` / `o_proj.weight` | attention |
| `model.layers.{i}.mlp.{gate,up,down}_proj.weight` | mlp |
| `model.layers.{i}.{input_layernorm,post_attention_layernorm}.weight` | rmsnorm |
| `model.norm.weight` | rmsnorm (final) |
| `lm_head.weight` | lm_head |

## Implementation Order (topological)
1. vision_patch_embed, vision_rmsnorm, vision_mlp (leaves)
2. vision_attention (needs vision_rmsnorm), embedding, rmsnorm, rope
3. vision_block; attention, mlp (LM)
4. vision_patch_merger; decoder_layer, lm_head
5. vision_tower (full encoder); language_model (full decoder)
6. End-to-end OCR use case (vision_tower → masked_scatter → language_model AR decode)

## Use cases

| Name | Input | Output | needs_ar | HF class | Metric | Threshold | Components used |
|------|-------|--------|----------|----------|--------|-----------|------------------|
| ocr | image | text | true | DotsOCRForCausalLM | accuracy | HF - 1.0 | vision_patch_embed, vision_rmsnorm, vision_attention, vision_mlp, vision_block, vision_patch_merger, vision_tower, embedding, rmsnorm, rope, attention, mlp, decoder_layer, lm_head, language_model |

**Hybrid notes:** Image preprocessing (resize → `pixel_values` + `image_grid_thw`)
and tokenizer/chat-template prompt construction stay on the HF host via
`DotsVLProcessor` (a `Qwen2_5_VLProcessor` subclass). The `masked_scatter` of
vision embeddings into the text embedding stream is host glue. Validation is
text-match accuracy against HF greedy decode on a representative document image.
