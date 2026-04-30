# Molmo2-8B Architecture Analysis

## Model Family
Vision-Language Model (VLM): Qwen2.5-7B-style text decoder + SigLIP ViT-L/14@378 vision encoder + adapter (image pooling cross-attention + SwiGLU projector). Supports both image and video input (up to 384 frames).

HuggingFace class: `Molmo2ForConditionalGeneration`
Config snapshot: `~/.cache/huggingface/hub/models--allenai--Molmo2-8B/snapshots/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b/`

**Preprocessing policy**: Use HF `Molmo2ImageProcessor` and `Molmo2VideoProcessor` as-is (no custom TTNN preprocessing needed).

---

## Complete Component Inventory

| Component | Weight Prefix | Tensor Count | Required For |
|-----------|--------------|--------------|--------------|
| Text Embedding (dual) | `model.transformer.wte` | 2 | Token → hidden |
| Text Decoder (36 blocks) | `model.transformer.blocks.*` | 288 | Text generation |
| Text LN Final | `model.transformer.ln_f` | 1 | Post-transformer RMSNorm |
| LM Head | `lm_head` | 1 | Token logits |
| ViT (25 resblocks built) | `model.vision_backbone.image_vit` | 403 | Image/video encoding |
| Image Pooling 2D | `model.vision_backbone.image_pooling_2d` | 8 | ViT features → adapter tokens |
| Image Projector | `model.vision_backbone.image_projector` | 3 | Adapter dim → text dim |

**Total: 706 tensors across 8 safetensors shards**

---

## Sub-Component Details

### 1. Text Decoder (LLM backbone)

**Config** (`text_config`):
| Parameter | Value |
|-----------|-------|
| `hidden_size` | 4096 |
| `intermediate_size` | 12288 |
| `num_hidden_layers` | 36 |
| `num_attention_heads` | 32 (Q) |
| `num_key_value_heads` | 8 (KV) — GQA, 4 Q heads per KV head |
| `head_dim` | 128 |
| `rope_theta` | 1,000,000 |
| `vocab_size` | 151,936 base + 128 additional |
| `norm` | RMSNorm, no bias (`layer_norm_eps=1e-6`) |
| `activation` | SiLU (SwiGLU gating) |
| `qkv_bias` | false |
| `use_qk_norm` | true — Qwen3-style per-head RMSNorm on Q and K |
| `qk_norm_type` | `"qwen3"` — norm applied AFTER reshape to `[batch, seq, n_heads, head_dim]` |
| `norm_after` | false → pre-norm architecture |
| `rope_scaling` | null (standard RoPE) |
| `max_position_embeddings` | 36,864 |

**Per-block weight keys** (8 tensors, no biases in text decoder):
```
blocks.{i}.attn_norm.weight             # RMSNorm before attention, shape [4096]
blocks.{i}.ff_norm.weight               # RMSNorm before MLP, shape [4096]
blocks.{i}.self_attn.att_proj.weight    # Fused Q+K+V: shape [6144, 4096]
                                        #   Q slice: [:4096, :] = [4096, 4096]
                                        #   K slice: [4096:5120, :] = [1024, 4096]
                                        #   V slice: [5120:, :] = [1024, 4096]
blocks.{i}.self_attn.attn_out.weight    # Output proj: shape [4096, 4096]
blocks.{i}.self_attn.q_norm.weight      # QK-norm for Q: shape [128] (head_dim)
blocks.{i}.self_attn.k_norm.weight      # QK-norm for K: shape [128] (head_dim)
blocks.{i}.mlp.ff_proj.weight           # Fused gate+up: shape [24576, 4096]
                                        #   "value" (x): first half [:12288, :]
                                        #   "gate": second half [12288:, :]
blocks.{i}.mlp.ff_out.weight            # Down proj: shape [4096, 12288]
```

**Attention forward (from modeling_molmo2.py:Molmo2Attention.forward)**:
```python
qkv = att_proj(hidden_states)                          # [B, S, 6144]
q, k, v = qkv.split([4096, 1024, 1024], dim=-1)
q = q.view(B, S, 32, 128)                              # reshape to heads
k = k.view(B, S, 8, 128)
# QK-norm AFTER reshape (qk_norm_type=="qwen3")
q = q_norm(q)                                          # RMSNorm per-head
k = k_norm(k)
q, k = apply_rotary_pos_emb(q.T, k.T, cos, sin)       # RoPE
# GQA: repeat k, v 4x
attn_out = scaled_dot_product_attention(q, k, v, mask)
attn_out = attn_out.reshape(B, S, 4096)
out = attn_out(attn_out)                               # output proj
```

**MLP forward (from modeling_molmo2.py:LanguageModelMLP.forward)**:
```python
ff = ff_proj(x)                   # [B, S, 24576]
x, gate = ff.chunk(2, dim=-1)     # x=[..., 12288] (value/first half), gate=[..., 12288] (second half)
x = silu(gate) * x                # IMPORTANT: gate is second half, value is first half
out = ff_out(x)                   # [B, S, 4096]
```
> **Critical**: Unlike standard Llama SwiGLU where `act(W1*x) * W3*x` (gate first), Molmo2 stores gate in the SECOND half of `ff_proj`. At load time: `w_gate = ff_proj[:12288]`, `w_up = ff_proj[12288:]`.

### 2. ViT Vision Encoder

**Config** (`vit_config`):
| Parameter | Value |
|-----------|-------|
| `hidden_size` | 1152 |
| `intermediate_size` | 4304 |
| `num_hidden_layers` | 27 (but only 25 built — see below) |
| `num_attention_heads` | 16 (full MHA, `n_kv_heads=16`) |
| `head_dim` | 72 (non-standard — pad to 96 for TTNN tile alignment) |
| `activation` | `gelu_pytorch_tanh` |
| `norm` | LayerNorm with weight+bias (`layer_norm_eps=1e-6`) |
| `image_default_input_size` | [378, 378] |
| `image_patch_size` | 14 |
| `image_num_pos` | 729 (27×27 patches) |
| `float32_attention` | true (fp32 during attention computation) |
| Position encoding | Absolute learned `positional_embedding [729, 1152]` with bicubic interpolation for non-default sizes |
| No RoPE | ViT uses absolute pos embed only |
| No causal mask | `is_causal=False` |

**Intermediate layer extraction** (from `Molmo2VisionBackbone.__init__`):
```python
vit_layers = [-3, -9]  # from adapter_config
# → absolute indices: 27-3=24, 27-9=18
last_layer_needed = max(24, 18) + 1 = 25
# Only 25 ViT resblocks are built (not 27)!
```
After running 25 blocks, features from layers 18 and 24 are concatenated:
```python
image_features = cat([hidden_states[18], hidden_states[24]], dim=-1)
# shape: [B*n_crops, 729, 2304]  (2 * 1152)
```

**Per-resblock weight keys** (16 tensors each, all have bias):
```
resblocks.{i}.attention_norm.weight / .bias    # LayerNorm before attention, [1152]
resblocks.{i}.ffn_norm.weight / .bias          # LayerNorm before MLP, [1152]
resblocks.{i}.attention.wq.weight / .bias      # Q: [1152, 1152]
resblocks.{i}.attention.wk.weight / .bias      # K: [1152, 1152]
resblocks.{i}.attention.wv.weight / .bias      # V: [1152, 1152]
resblocks.{i}.attention.wo.weight / .bias      # Out: [1152, 1152]
resblocks.{i}.feed_forward.w1.weight / .bias   # Up: [4304, 1152]
resblocks.{i}.feed_forward.w2.weight / .bias   # Down: [1152, 4304]
```

**ViT block forward**:
```python
# Pre-norm, no causal mask, no RoPE
x = x + attention(attention_norm(x))
x = x + mlp(ffn_norm(x))
```

**Patch embedding**: `nn.Linear(14*14*3=588, 1152, bias=True)` — input is flattened patches `[n_crops, 729, 588]`.

### 3. Image Pooling 2D (adapter cross-attention)

**Config**: from `adapter_config` (head_dim=72, hidden_size=1152, n_heads=16, n_kv_heads=16)

**Input dim**: `pool_dim = 1152 * 2 = 2304` (concatenated ViT features from 2 layers)

**Weight shapes** (all have bias):
```
image_pooling_2d.wq.weight  [1152, 2304]   # Q from mean-pooled window
image_pooling_2d.wk.weight  [1152, 2304]   # K from full window
image_pooling_2d.wv.weight  [1152, 2304]   # V from full window
image_pooling_2d.wo.weight  [1152, 1152]   # Output
```

**Forward** (from `Molmo2VisionBackbone.forward`):
```python
# For each pooling window (pool_size × pool_size patches):
denom = valid_patches_count_in_window
query = to_pool.sum(dim=-2, keepdim=True) / denom  # masked mean [N_windows, 1, 2304]
attn_mask = valid_mask  # [N_windows, 1, 1, pool_h*pool_w]
pooled = image_pooling_2d(inputs_q=query, inputs_kv=to_pool, attn_mask=attn_mask)
# pooled: [N_windows, 1, 1152]
```
> **Note**: Query is computed from the patches themselves (masked mean), not learned queries. The `pooling_attention_mask=True` is set in adapter_config, so invalid/padded patches are masked out.

### 4. Image Projector (SwiGLU MLP)

**Weight shapes** (NO biases):
```
image_projector.w1.weight  [12288, 1152]   # gate projection
image_projector.w2.weight  [1152, 12288]   # down projection (NOT 4096 — intermediate!)
image_projector.w3.weight  [12288, 1152]   # up projection
```
Wait — re-reading `ImageProjectorMLP`: `w1` and `w3` both project `[input_dim=1152 → hidden_dim=12288]`, and `w2` projects `[hidden_dim=12288 → output_dim=4096]`.
```python
return w2(act(w1(x)) * w3(x))  # standard SwiGLU: act(gate)*up, then down
# w1=gate [12288, 1152], w3=up [12288, 1152], w2=down [4096, 12288]
```
Corrected shapes from adapter_config: `hidden_dim=intermediate_size=12288`, `output_dim=text_hidden_size=4096`:
```
image_projector.w1.weight  [12288, 1152]   # gate
image_projector.w3.weight  [12288, 1152]   # up
image_projector.w2.weight  [4096, 12288]   # down → text_hidden_size=4096
```

### 5. Dual Vocabulary Embedding

```python
# forward (Molmo2Embedding):
embedding = cat([self.embedding, self.new_embedding], dim=0)  # [151936+128, 4096]
return F.embedding(x, embedding)
```
Weight keys: `wte.embedding [151936, 4096]`, `wte.new_embedding [128, 4096]`

---

## CRITICAL: Masking Mechanism (Image Token Bidirectional Attention)

**Confirmed**: Molmo2 applies **bidirectional attention among image tokens at prefill**, overlaid on top of the standard causal mask. This is the key architectural difference from Qwen3-VL.

### How it works (from `modeling_molmo2.py` lines 1086-1111, 1515-1544)

**Step 1 — token_type_ids creation** (in `Molmo2Processor.__call__`):
```python
# All 8 image/video special tokens get type_id=1
IMAGE_TOKENS = [IMAGE_PATCH_TOKEN, IM_COL_TOKEN, IM_START_TOKEN, LOW_RES_IMAGE_START_TOKEN,
                FRAME_START_TOKEN, IM_END_TOKEN, FRAME_END_TOKEN, IMAGE_LOW_RES_TOKEN]
token_type_ids = (input_ids in IMAGE_TOKENS)  # bool, shape [B, S]
# 1 = image/video token, 0 = text token
```

**Step 2 — mask construction at prefill** (in `Molmo2Model.forward`):
```python
is_prefill = (
    not use_cache
    or past_key_values is None
    or not past_key_values.is_initialized
    or images is not None  # ← always prefill when new images present
)
if token_type_ids is not None and is_prefill:
    mask_kwargs["or_mask_function"] = token_type_ids_mask_function(token_type_ids)
causal_mask = create_causal_mask(**mask_kwargs)
```

**Step 3 — mask function logic**:
```python
def token_type_ids_mask_function(token_type_ids):
    def inner_mask(batch_idx, head_idx, q_idx, kv_idx) -> bool:
        # Returns True if BOTH q and kv are image tokens
        # This is OR'd with causal mask: attend if causal OR both-image
        is_image_q = (token_type_ids[batch_idx, q_idx] == 1)
        is_image_kv = (token_type_ids[batch_idx, kv_idx] == 1)
        return is_image_q & is_image_kv
    return inner_mask
```

**Effect**: An image token at position `q` can attend to any image token at position `kv` — whether `kv > q` or `kv < q`. This is **bidirectional within image token blocks**, while text tokens remain strictly causal.

**Also at generate step**: The same `create_masks_for_generate` override applies during the first generation step (when `input_embeds.shape[1] != 1`).

### Qwen3-VL comparison
| Aspect | Molmo2 | Qwen3-VL |
|--------|--------|----------|
| ViT attention | No mask, non-causal (all attend all) | No mask, non-causal |
| Text decoder — text tokens | Standard causal mask | Standard causal mask |
| Text decoder — image tokens at prefill | **Bidirectional among image tokens** (via token_type_ids or_mask) | Full bidirectional for all ViT-merged tokens |
| Text decoder — image tokens at decode | Standard causal (new single token sees all past) | Standard causal |
| Mask source | `token_type_ids` passed from processor | Position ranges from patch merger |

**TTNN implementation implication**: The prefill attention kernel must accept a custom 4D mask that combines causal lower-triangular with a bidirectional block for image token positions. This is **not** the same as Qwen3-VL. The mask must be constructed from `token_type_ids` at inference time.

---

## Image Embedding Injection

Image features are **added** to (not replacing) text token embeddings at `image_patch_id` positions:
```python
# from build_input_embeddings()
x = self.transformer.wte(input_ids)          # text embeddings
is_image_patch = (input_ids.view(-1) == config.image_patch_id)  # image_patch_id=151938
x.view(-1, hidden)[is_image_patch] += image_features            # additive, not replacement
```
> This means the `<im_patch>` token embedding contributes to the final representation alongside the ViT feature. Important for weight initialization in the TTNN embedding lookup.

---

## Preprocessing Policy

**Use HF preprocessors as-is. Run entirely on CPU before transferring tensors to device.**

- `Molmo2ImageProcessor` — handles crop tiling, overlap margins, pooling index construction, normalization
- `Molmo2VideoProcessor` — handles frame sampling, per-frame resize, video grid construction
- `Molmo2Processor` — combines both, builds token strings, assembles `token_type_ids`

Do **not** reimplement any of this in TTNN. The processors produce the exact tensors the model expects; reimplementing introduces subtle bugs (wrong `pooling_size` for video vs image, off-by-one in overlap margins, wrong token string ordering, etc.).

**CPU preprocessing outputs** are NumPy arrays or PyTorch CPU tensors. Upload to device with `ttnn.as_tensor(..., device=mesh_device)` only after the full preprocessing pipeline completes.

---

## Image Preprocessing (HF-managed, CPU, use as-is)

**Processor outputs for images** (`Molmo2ImageProcessor.preprocess`):
| Output key | Shape | Description |
|------------|-------|-------------|
| `pixel_values` | `[n_total_crops, 729, 588]` | All crops for all images, flattened patches |
| `image_token_pooling` | `[n_total_pooled_tokens, pool_h*pool_w]` | Patch indices per pooled token (−1 = invalid) |
| `image_grids` | `[n_images, 4]` | Per-image: `[resized_h, resized_w, hr_h, hr_w]` after pooling |
| `image_num_crops` | `[n_images]` | How many crops per image (global + high-res) |

**Image tiling**:
- 1 global crop (378×378 resize) + up to 8 high-res crops with overlap margins `[4,4]` patches
- Each crop → 27×27 = 729 patches of 14×14×3 = 588 pixels
- After pooling `[2,2]`: 729 patches → ~182 pooled tokens per crop
- Normalization: mean=std=[0.5,0.5,0.5] (maps [0,1] pixels to [−1,1])

**Processor outputs for video** (`Molmo2VideoProcessor.preprocess`):
| Output key | Shape | Description |
|------------|-------|-------------|
| `pixel_values_videos` | `[n_total_frames, 729, 588]` | All frames, each as single 378×378 crop |
| `video_token_pooling` | `[n_total_pooled_tokens, pool_h*pool_w=9]` | Patch indices per pooled token |
| `video_grids` | `[n_videos, 3]` | Per-video: `[num_frames, h_pool, w_pool]` |
| `video_metadata` | list | Contains timestamps, fps, etc. |

**Video tiling**: No multi-crop for video — each frame is a single 378×378 resize.
- After pooling `[3,3]`: 729 patches → 9×9 = 81 pooled tokens per frame
- Max 384 frames × 81 tokens = 31,104 video tokens maximum
- Frame sampling: `uniform_last_frame` mode, max_fps=2, sampling_fps=2

**Video token string format** (per frame, from `get_video_string`):
```
"{timestamp:.1f} <im_start|frame_start> <im_patch>×(w_pool×h_pool) <im_end|frame_end>"
```
With `use_frame_special_tokens=False` (from `processor_config.json`): uses `<im_start>/<im_end>`.

**Image token string format** (from `get_image_tokens`):
```
<low_res_im_start> <im_patch>×(resized_w) <im_col> ... <im_end>  ← global view
<im_start> <im_patch>×(w) <im_col> ... <im_end>                  ← high-res tiled view
```

---

## Model Forward Pass Flow

```
Input: input_ids, pixel_values (images) or pixel_values_videos + token_type_ids

1. merge_visual_inputs()
   ├── build_batched_images() or build_batched_videos()
   └── returns: images [B, max_crops, 729, 588], token_pooling [B, max_pooled, pool_window]

2. build_input_embeddings()
   ├── x = wte(input_ids)                     # [B, S, 4096]
   ├── image_features = vision_backbone(images, token_pooling)
   │   ├── encode_image()
   │   │   ├── image_vit(patches) → all_hidden_states (25 layers)
   │   │   └── concat(hidden[18], hidden[24], dim=-1) → [B*crops, 729, 2304]
   │   ├── image_pooling_2d(query, kv, mask) → [total_tokens, 1, 1152]
   │   └── image_projector(pooled) → [valid_tokens, 4096]
   └── x[is_image_patch] += image_features    # additive injection

3. create_causal_mask() with token_type_ids or_mask (prefill only)

4. transformer(x, causal_mask, position_ids)  # 36 decoder blocks

5. lm_head(ln_f(hidden_states)) → logits
```

---

## Similar TTNN Implementations

| Component | Reference Implementation | Key Difference |
|-----------|-------------------------|----------------|
| Text GQA attention | `models/demos/qwen3_vl/tt/attention.py` | Fused att_proj (split needed); QK-norm after reshape |
| Text SwiGLU MLP | `models/tt_transformers/tt/mlp.py` | Fused ff_proj; gate is SECOND half (not first) |
| Text RMSNorm | `models/common/rmsnorm.py` | Direct reuse |
| QK-norm (per-head) | `models/demos/qwen3_vl/tt/attention.py` (q/k_norm) | Same Qwen3-style, head_dim=128 |
| RoPE | `models/demos/qwen3_vl/tt/rope.py` | rope_theta=1e6; no rope_scaling |
| ViT MHA (no RoPE) | `models/demos/qwen3_vl/tt/vision_attention.py` | head_dim=72 (pad to 96); biases; no RoPE; fp32 attn |
| ViT LayerNorm | `models/demos/qwen3_vl/tt/vision_layernorm.py` | Direct reuse |
| ViT GELU MLP | `models/demos/qwen3_vl/tt/vision_mlp.py` | GELU (gelu_pytorch_tanh); biases |
| Image pooling cross-attn | `models/demos/qwen3_vl/tt/patch_merger.py` | Query=masked-mean of patches (not learned); input_dim=2304; biases |
| Image projector SwiGLU | `models/tt_transformers/tt/mlp.py` | No bias; act(w1)*w3 ordering (standard) |
| Dual embedding | `models/tt_transformers/tt/embedding.py` | Must concat [151936, 4096]+[128, 4096] at load |
| LM head | `models/tt_transformers/tt/lm_head.py` | Direct reuse |
| **Prefill mask** | **No direct equivalent** | Must implement token_type_ids bidirectional override |

---

## Key Differences from Reference Implementations (Summary)

1. **Fused att_proj**: Combined Q+K+V `[6144, 4096]` — split `[4096|1024|1024]` along dim 0 at load.
2. **Fused ff_proj with reversed gate order**: `[24576, 4096]` — first half is value (`w_up`), second half is gate (`w_gate`). Unlike Llama where gate is first.
3. **QK-norm applied after head reshape**: `q_norm` and `k_norm` have shape `[128]` (head_dim). Applied to `[B, S, n_heads, head_dim]` tensor after splitting.
4. **ViT head_dim=72**: Requires padding to 96 (next mult of 32) for TTNN tile alignment.
5. **ViT runs only 25 of 27 layers**: `last_layer_needed = max(vit_layers) + 1 = 25`.
6. **ViT intermediate feature extraction**: Features from ViT layers 18 and 24 concatenated → 2304-dim input to pooling cross-attention.
7. **Pooling query = masked mean of patches**: Not learned queries. Computed dynamically per inference.
8. **Image pooling wq/wk/wv take 2304-dim input**: Because input is concat of 2 ViT layers.
9. **Image features added (not replacing) embeddings** at `image_patch_id` token positions.
10. **Bidirectional attention among image tokens at prefill**: Requires custom 4D mask from `token_type_ids`. Not present in any existing TTNN VLM demo.
11. **Dual vocabulary**: `wte.embedding [151936]` + `wte.new_embedding [128]` concatenated at runtime.
12. **Video uses pooling_size=[3,3]** (9 patches per window) vs image `[2,2]` (4 patches per window) → 81 tokens/frame vs ~182 tokens/crop.
13. **No multi-crop for video**: Each video frame is a single 378×378 resize (no overlapping crops).
14. **Absolute positional embedding in ViT with bicubic interpolation**: When patch grid ≠ 27×27, positional embedding is interpolated.

---

## Weight Mapping: HuggingFace → TTNN

### Text Decoder
| HuggingFace Key | Load-time operation | TTNN usage |
|-----------------|---------------------|------------|
| `model.transformer.wte.embedding` | concat with new_embedding | token embedding table |
| `model.transformer.wte.new_embedding` | concat with embedding | additional 128 tokens |
| `blocks.{i}.attn_norm.weight` | direct | pre-attn RMSNorm |
| `blocks.{i}.ff_norm.weight` | direct | pre-MLP RMSNorm |
| `blocks.{i}.self_attn.att_proj.weight` | split dim 0: `[0:4096]`=wq, `[4096:5120]`=wk, `[5120:]`=wv | Q, K, V projections |
| `blocks.{i}.self_attn.attn_out.weight` | direct | output projection |
| `blocks.{i}.self_attn.q_norm.weight` | direct | per-head Q norm |
| `blocks.{i}.self_attn.k_norm.weight` | direct | per-head K norm |
| `blocks.{i}.mlp.ff_proj.weight` | split dim 0: `[0:12288]`=w_value, `[12288:]`=w_gate | value and gate weights |
| `blocks.{i}.mlp.ff_out.weight` | direct | down projection |
| `model.transformer.ln_f.weight` | direct | final RMSNorm |
| `lm_head.weight` | direct | logit projection |

### Vision Backbone
| HuggingFace Key | Shape | Notes |
|-----------------|-------|-------|
| `image_vit.patch_embedding.weight` | [1152, 588] | Linear, not Conv2d |
| `image_vit.patch_embedding.bias` | [1152] | |
| `image_vit.positional_embedding` | [729, 1152] | Absolute pos embed; bicubic interp if needed |
| `image_vit.transformer.resblocks.{i}.attention.w{q,k,v,o}.weight` | [1152, 1152] | All with bias |
| `image_pooling_2d.w{q,k,v}.weight` | [1152, 2304] | 2304 = 2×ViT_hidden |
| `image_pooling_2d.wo.weight` | [1152, 1152] | |
| `image_projector.w1.weight` | [12288, 1152] | gate |
| `image_projector.w3.weight` | [12288, 1152] | up |
| `image_projector.w2.weight` | [4096, 12288] | down → text dim |

---

## Implementation Order

Follow the Relay Race flow: Architecture → Reference → TTNN → Debug → Opt. Complete each phase fully before proceeding.

### Phase 1: Reference
1. Run `Molmo2ForConditionalGeneration` end-to-end on HF to collect per-block golden outputs
2. Verify token_type_ids masking effect on attention patterns (log attention weights)
3. Verify image feature injection shape: `image_features[valid_tokens] → [N, 4096]`

### Phase 2: TTNN Implementation Order
1. **Text decoder (text-only first)**:
   - Weight loader: split `att_proj` and `ff_proj`
   - Text attention: adapt `qwen3_vl/tt/attention.py` — att_proj split + QK-norm after reshape
   - Text MLP: adapt `tt_transformers/tt/mlp.py` — ff_proj split, reversed gate/value order
   - RMSNorm + RoPE (rope_theta=1e6): reuse from qwen3_vl
   - Dual embedding: concat at load time
   - Verify text-only PCC > 0.99

2. **Prefill attention mask**:
   - Implement `token_type_ids` bidirectional override for image tokens
   - Test: image tokens must attend to all other image tokens in both directions

3. **ViT encoder**:
   - LayerNorm: reuse `qwen3_vl/tt/vision_layernorm.py` directly
   - MHA (no RoPE, head_dim=72 padded to 96, biases): adapt `qwen3_vl/tt/vision_attention.py`
   - GELU MLP (biases): adapt `qwen3_vl/tt/vision_mlp.py`
   - Absolute pos embed with bicubic interpolation
   - Run only 25 blocks; capture hidden_states[18] and hidden_states[24]

4. **Adapter**:
   - Image pooling 2D: cross-attention with 2304-dim input (concat of two ViT layers), masked-mean query
   - Image projector: SwiGLU MLP, no bias

5. **End-to-end image inference**:
   - Image feature injection (additive)
   - Combined prefill with token_type_ids mask
   - PCC > 0.99 on full forward pass

6. **Video inference**:
   - Same ViT path, pooling_size=[3,3] (9 patches/window)
   - No multi-crop; frames processed sequentially
   - Update token_type_ids mask for video tokens

---

## T3K Parallelization Strategy (8 Wormhole B0 Chips)

### Hardware Topology
- **Mesh**: 1×8 (1 row, 8 columns) — `cluster_shape = [1, 8]`
- **CCL topology**: Ring (T3K ≥ 8 devices → Ring, see `tt_transformers/tt/ccl.py`)
- **CCL operations**: Ring AllGather, ReduceScatter (semaphore-based, double-buffered)
- **Reference**: T3K mesh defined in `tt_transformers/conftest.py` as `(1, 8)`

### Divisibility Constraints (T3K = 8 devices)
| Component | Param | Value | ÷8 | OK? |
|-----------|-------|-------|----|-----|
| Text Q heads | `num_attention_heads` | 32 | 4/dev | ✓ |
| Text KV heads | `num_key_value_heads` | 8 | 1/dev | ✓ |
| Text intermediate | `intermediate_size` | 12288 | 1536/dev | ✓ |
| Text hidden | `hidden_size` | 4096 | 512/dev | ✓ |
| ViT heads | `num_attention_heads` | 16 | (data-parallel; no head shard needed) | ✓ |
| Projector intermediate | 12288 | — | 1536/dev if sharded | ✓ |

---

### Component-by-Component Parallelism Plan

#### 1. Text Decoder — Tensor Parallel (column-row)

**Mesh mapper**: `ShardTensor2dMesh(mesh_device, dims=(None, -1), mesh_shape=[1,8])`

| Weight | Full shape | Per-device shape | Strategy | CCL needed |
|--------|-----------|-----------------|----------|------------|
| `att_proj.weight` | [6144, 4096] | [768, 4096] | Column-parallel (output dim) | AllGather on hidden-state before mm |
| `attn_out.weight` | [4096, 4096] | [4096, 512] | Row-parallel (input dim) | ReduceScatter after mm |
| `q_norm.weight` | [128] | [128] | Replicated | — |
| `k_norm.weight` | [128] | [128] | Replicated | — |
| `attn_norm.weight` | [4096] | [4096] | Replicated | — |
| `ff_proj.weight` | [24576, 4096] | [3072, 4096] | Column-parallel | AllGather on hidden-state |
| `ff_out.weight` | [4096, 12288] | [4096, 1536] | Row-parallel | ReduceScatter after mm |
| `ff_norm.weight` | [4096] | [4096] | Replicated | — |
| `ln_f.weight` | [4096] | [4096] | Replicated | — |

**Per-device attention breakdown**:
- `att_proj` shard = Q:512 + K:128 + V:128 = 768 output dims = 4 Q-heads + 1 K-head + 1 V-head
- No KV replication needed (1 KV head/device matches 4 Q heads/device, group size = 4)
- Prefill: AllGather full Q across devices for SDPA, then ReduceScatter on attention output
- Decode: standard single-token GQA with sharded heads

**Gate/value ordering in ff_proj shard**: the `[24576/8=3072, 4096]` per-device slice contains:
`[:1536]` = value (w_up) portion for this device's 1536 channels
`[1536:]` = gate (w_gate) portion for this device's 1536 channels
The `x, gate = ff_proj_out.chunk(2)` split still works correctly within each device's shard.

#### 2. Embeddings and LM Head — Replicated

| Weight | Shape | Strategy | Reason |
|--------|-------|----------|--------|
| `wte.embedding` + `wte.new_embedding` | [152064, 4096] | `ReplicateTensorToMesh` | Vocab lookup is sparse; sharding complicates index scatter |
| `lm_head.weight` | [152064, 4096] | `ReplicateTensorToMesh` | Logit computation straightforward; 1.2GB fits on device |

> Future optimization: vocab-parallel lm_head ([19008, 4096]/device) with AllGather on logits at decode step.

#### 3. ViT Encoder — Replicated Weights + Data-Parallel Input

**Principle**: Each device holds a full copy of all 25 ViT blocks. Input crops/frames are sharded along the batch dimension so each device processes a different subset. No CCL within the ViT forward pass.

**Weight distribution**: `ReplicateTensorToMesh` for all ViT tensors:
- resblock attention wq/wk/wv/wo [1152, 1152] + biases [1152]
- resblock MLP w1/w2 [4304, 1152] / [1152, 4304] + biases
- patch_embedding.weight [1152, 588], .bias [1152]
- positional_embedding [729, 1152]

**Input distribution**: `ShardTensorToMesh(dim=0)` — shard along crop/frame batch dim

| Input | Full shape | Per-device shape | Frames/device |
|-------|-----------|-----------------|---------------|
| Image crops | [≤9, 729, 588] | [⌈9/8⌉, 729, 588] | ≤2 crops/dev |
| Video frames | [384, 729, 588] | [48, 729, 588] | 48 frames/dev |

**ViT forward** (data-parallel, no CCL):
```
Device i: ViT(crops[i*K:(i+1)*K]) -> hidden_states_slice (25 layers, capture [18] and [24])
                                   -> [K_i, 729, 2304]  (concat of layers 18+24)
```

**After ViT**: AllGather dim=0 across all 8 devices to reconstruct full `image_features`:
```
AllGather([K_0,...,K_7]) -> [N_total_crops, 729, 2304] on all devices
```

**Why data-parallel (not tensor-parallel for ViT heads)**:
- Data-parallel: zero CCL inside ViT, only 1 AllGather at the end → minimal communication
- Tensor-parallel heads: needs AllGather after every attention block (25× more CCL)
- ViT weight memory (762MB replicated) is acceptable given 12GB device DRAM
- Video with 384 frames: 48 frames/device reduces compute 8× vs single-device

#### 4. Image Pooling 2D — Replicated

After AllGather, `image_features [N_crops, 729, 2304]` is identical on all 8 devices. Pooling operates on this replicated tensor.

**Weight distribution**: `ReplicateTensorToMesh`

| Weight | Shape | Notes |
|--------|-------|-------|
| `wq.weight` | [1152, 2304] | 2304 = 2× ViT hidden (concat of 2 ViT layers) |
| `wk.weight` | [1152, 2304] | — |
| `wv.weight` | [1152, 2304] | — |
| `wo.weight` | [1152, 1152] | — |
| All biases | [1152] | — |

**Why replicated** (not head-sharded):
- 16 heads divisible by 8 → head-sharding is feasible in principle
- But the masked-mean query construction involves complex batch indexing (`pooled_patches_idx`)
  which spans the full batch — sharding this adds significant implementation complexity
- Pooling is small (N_pooled_tokens × pool_window_size = at most ~14,000 × 9 for video)
- Output: `[N_valid_tokens, 1152]` replicated on all 8 devices, `valid_token` mask applied

> Future opt: Shard pooled tokens along batch dim (each device handles a disjoint subset of windows) — avoids any CCL, requires splitting `pooled_patches_idx` first.

#### 5. Image Projector — Replicated (Phase 1), Tensor-Parallel (Phase 2)

**Phase 1 (initial)**: `ReplicateTensorToMesh` for all weights
- w1 [12288, 1152], w3 [12288, 1152], w2 [4096, 12288]
- Output: `[N_valid, 4096]` replicated → feeds directly into additive embedding injection
- Image feature injection `x[is_image_patch] += image_features` works on replicated `x`

**Phase 2 (optimization)**: Column-row parallel
- w1, w3: column-parallel → [1536, 1152]/device
- w2: row-parallel → [4096, 1536]/device, ReduceScatter on output
- Output: `[N_valid, 4096]` replicated after gather

---

### Full Inference Data Flow on T3K

```
PREFILL (image example, 9 crops):

[CPU] pixel_values [9, 729, 588]
         |
  ShardTensorToMesh(dim=0)   → each device gets [2 or 1, 729, 588]
         |
  [8x ViT forward] (replicated weights, no CCL)
  Device 0: ViT(crops[0:2]) → [2, 729, 2304]
  Device 1: ViT(crops[2:4]) → [2, 729, 2304]
  ...
  Device 7: ViT(crops[8:9]) → [1, 729, 2304]  (padded)
         |
  AllGather(dim=0)            → all devices: [9, 729, 2304]
         |
  [8x pooling_2d] (replicated, all identical)
         → [N_pooled, 1, 1152] on all devices
         |
  [8x projector] (replicated, all identical)
         → [N_valid, 4096] on all devices
         |
  [8x wte(input_ids)] + [N_valid, 4096] at image_patch positions
         → inputs_embeds [1, S, 4096] replicated
         |
  create_causal_mask with token_type_ids or_mask
         |
  [36x Decoder Block] (tensor-parallel)
  ┌─ attn_norm (replicated) → hidden [1, S, 4096]
  ├─ att_proj [768, 4096]/dev → AllGather Q → SDPA → ReduceScatter
  ├─ attn_out [4096, 512]/dev → partial; AllReduce
  ├─ ff_norm (replicated) → hidden [1, S, 4096]
  ├─ ff_proj [3072, 4096]/dev → SwiGLU activation (local chunk)
  └─ ff_out [4096, 1536]/dev → AllReduce
         |
  ln_f (replicated) → [1, S, 4096]
  lm_head (replicated) → logits [1, 1, vocab_size]

DECODE (single token):
  Same as prefill but S=1, pixel_values=None, KV cache lookup
  att_proj: [768, 4096]/dev → single-token decode (no AllGather needed for Q in decode)
  KV cache per device: 1 KV-head × head_dim=128 × 36 layers
```

---

### KV Cache Memory on T3K

| Parameter | Value |
|-----------|-------|
| KV heads per device | 1 (8 KV heads / 8 devices) |
| head_dim | 128 |
| Layers | 36 |
| Bytes per token per device | 36 × 2 × 1 × 128 × 2 = 18,432 bytes |
| Max seq len | 36,864 tokens |
| KV cache at max seq | 36,864 × 18,432 ≈ **679 MB/device** |
| Max video prefill (384f) | ~35,000 tokens × 18,432 ≈ **644 MB/device** |

---

### Memory Budget Per Device (fp16 weights)

| Component | Strategy | MB/device |
|-----------|----------|-----------|
| Text decoder (36 blocks, 1/8 sharded) | Tensor-parallel | 1,737 |
| ViT (25 blocks, replicated) | Data-parallel input | 762 |
| Image pooling 2D (replicated) | — | 19 |
| Image projector (replicated) | — | 157 |
| Embedding + LM head (replicated) | — | 2,491 |
| **Total weights** | | **5,166** |
| KV cache (max seq 36,864) | | ~679 |
| Activations (est.) | | ~500 |
| **Total peak** | | **~6,345** |
| **Available (T3K device DRAM)** | | **12,288** |
| **Headroom** | | **~5,943** |

---

### Implementation Notes for Model Config

```python
# Molmo2ModelArgs (model_config.py)
cluster_shape = [1, 8]   # T3K

# Text attention
n_local_heads = 32 // 8 = 4        # Q heads per device
n_local_kv_heads = 8 // 8 = 1      # KV heads per device
# ShardTensor2dMesh(dims=(None, -1)) for att_proj, attn_out, ff_proj, ff_out

# ViT
vit_num_devices_per_group = 1      # data-parallel, full ViT on each device
# ReplicateTensorToMesh for all ViT weights
# ShardTensorToMesh(dim=0) for pixel_values / pixel_values_videos input

# Vision head_dim padding
vit_padded_head_dim = 96           # ceil(72/32)*32 = 96

# Pooling + projector
# ReplicateTensorToMesh for all pooling and projector weights

# CCL
ccl_topology = ttnn.Topology.Ring  # T3K with 8 devices
```

### Key Reference Files
| Component | Closest reference | Key difference |
|-----------|------------------|----------------|
| Text attention TP | `tt_transformers/tt/attention.py:38-60` | att_proj fused; QK-norm after reshape |
| Text MLP TP | `tt_transformers/tt/mlp.py` | ff_proj fused; gate is 2nd half |
| ViT data-parallel input | `qwen25_vl/tt/model_config.py:84` | `ShardTensorToMesh(dim=0)` |
| ViT replicated weights | `qwen3_vl/tt/vision_attention.py:70` | `num_devices_per_group=1` |
| AllGather after ViT | `qwen25_vl` pattern | AllGather before pooling |
| Pooling replicated | `qwen3_vl/tt/patch_merger.py` | 2304-dim input; masked-mean query |

---

## Attention Mask at High ISL: Problem and Solution

### Maximum Context and Sequence Lengths

| Scenario | Vision tokens | Text budget | Total ISL |
|----------|--------------|-------------|-----------|
| Text only | 0 | 36,864 | 36,864 |
| Single image (9 crops) | 1,764 (9×196, pool 2×2) | ~35,100 | ~36,864 |
| Max video (384 frames) | 32,256 (384×84, pool 3×3) | 4,608 | **36,864** |

`max_position_embeddings = 36,864` is the hard ceiling. Max video fills it entirely.

### Mask Dtype: bfloat4_b

Use `ttnn.bfloat4_b` (4-bit, 0.5 bytes/element) for the combined attention mask. This is 4× smaller than BF16:

| S | bfloat4_b 2D `[S,S]` | bfloat4_b 4D `[1,4,S,S]`/dev | Verdict |
|---|---------------------|------------------------------|---------|
| 8,192 | 32 MB | 128 MB | ✓ |
| 16,384 | 128 MB | 512 MB | ✓ |
| 32,768 | 512 MB | 2,048 MB | ✓ |
| **36,864** | **648 MB** | **2,592 MB** | **✓ fits in ~7.1 GB headroom** |

With `bfloat4_b`, a **precomputed full `[S, S]` combined mask** (causal + image-bidir) fits at all ISLs up to max context. No per-chunk construction needed.

### Combined Mask Construction

Build once at prefill entry, store in DRAM as `bfloat4_b`:

```python
# mask[q, kv] = 0.0 (attend) or -inf (block)
causal_block   = (kv_pos > q_pos)                             # True = causal block
image_override = is_image[q_pos] & is_image[kv_pos]           # True = force allow (image-bidir)
block          = causal_block & ~image_override
combined_mask  = torch.where(block, -inf, 0.0).to(bfloat4_b) # [S, S]
```

Inputs: `is_image [S]` bool (36 KB at S=36,864) derived from `token_type_ids` produced by the HF processor.

**Decode step**: single new token is always a text token; standard causal decode, no mask needed.

### Video Attention Pattern at Max ISL

At max video (32,256 video + 4,608 text = 36,864 tokens):
- 87.5% of tokens are image/video → bidirectional among nearly all pairs
- The combined mask is nearly all-attend; only 4,608 text-token rows remain strictly causal
- Precomputing `[S, S]` bfloat4_b (648 MB) once is the right approach for this case

---

## Performance Bottleneck Analysis (T3K, 8 Devices)

**Assumptions**: Wormhole B0 peak 236 TFLOPS bf16; DRAM BW 768 GB/s/device; Ring CCL effective BW 160 GB/s; MFU 40-50%.

### Estimated Latency by Phase (max video ISL = 36,864 tokens)

| Phase | Operation | Est. time | Bound by |
|-------|-----------|-----------|----------|
| Vision | ViT prefill (48 frames/dev, 25 blocks) | ~313 ms | Compute |
| Vision | AllGather image_features after ViT | ~7 ms | CCL BW |
| Vision | Pooling gather (non-contiguous) | ~2 ms | DRAM BW |
| Vision | Pooling cross-attn + projector | ~5 ms | Compute |
| Text | QKV matmul (36 layers) | ~71 ms | Compute |
| Text | SDPA (36 layers, flash-attn, S=36K) | ~1,061 ms | Compute |
| Text | MLP (36 layers) | ~425 ms | Compute |
| Text | CCL AllGather+RS (36 layers) | ~238 ms | Ring BW |
| **Total prefill** | | **~2,122 ms** | |
| Decode | Single token (KV cache full) | ~4 ms/tok | DRAM BW |
| Decode | Throughput | ~261 tok/s | DRAM BW |

For image-only (ISL ~5K tokens): SDPA drops to ~20 ms total, prefill ~500 ms.

---

### Bottleneck 1 — Text SDPA at High ISL (CRITICAL)

**Dominant cost at max video ISL. ~1,061 ms / 50% of total prefill.**

| S | SDPA FLOP/layer | Est. time/layer | 36 layers |
|---|-----------------|-----------------|-----------|
| 4,096 | 34 GFLOP | 0.36 ms | 13 ms |
| 8,192 | 137 GFLOP | 1.45 ms | 52 ms |
| 16,384 | 549 GFLOP | 5.8 ms | 209 ms |
| 36,864 | 2,783 GFLOP | 29.5 ms | **1,061 ms** |

SDPA is O(S²·H·n_heads_local). Flash-attention (TTNN SDPA) avoids materialising the full S×S matrix, but the compute remains quadratic. At 36K tokens this is fundamental.

**Mitigations**:
- Practical ISL for most video queries is ≪ 36K (e.g. 30-second clip at 2fps = 60 frames = 4,860 tokens)
- GQA already reduces KV memory 8× (1 KV-head per device)
- `bfloat4_b` mask avoids 24 GB of mask rereads per pass
- If needed: per-frame chunked attention (process video frames in groups, build KV incrementally)

---

### Bottleneck 2 — ViT Prefill for Video (HIGH)

**~313 ms for 384 frames even with data-parallel across 8 devices.**

Per device: 48 frames × 25 blocks × 24.6 GFLOP/block/frame = 29.6 TFLOP.
At 40% MFU → 313 ms. ViT runs BEFORE text decoder so it is not on the decode critical path.

**Key sub-issue — ViT `float32_attention=True`**:
The ViT config requires Q and K cast to fp32 before SDPA. This doubles Q/K memory traffic and requires HiFi4 compute kernel in TTNN. Verify `ttnn.transformer.scaled_dot_product_attention` supports fp32 compute. If not available, use `compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4)`.

**Mitigations**:
- 8-way data-parallel already gives 8× speedup vs single device
- Can further pipeline: overlap ViT compute on frames already uploaded while text decoder runs on earlier frames (for streaming video)

---

### Bottleneck 3 — Text MLP at High ISL (MEDIUM-HIGH)

**~425 ms across 36 layers at S=36,864.**

`ff_proj [36864,4096]×[4096,3072]` = 0.93 TFLOP/layer. Compute-bound.

**Mitigations**:
- `bfloat4_b` weights for MLP (`four_bit_mlp` optimisation from qwen3_vl) reduce weight reads 4× and may improve MFU
- Already column-parallel (3,072-wide per device, 8× less than full 24,576)

---

### Bottleneck 4 — CCL at Text Prefill (MEDIUM)

**~238 ms across 36 layers at S=36,864. 38 GB total ring traffic.**

4 collectives (2 AG + 2 RS) per layer on 302 MB hidden-state tensors.

**Mitigations**:
- Ring topology (already configured for T3K ≥ 8 devices)
- AG-matmul fusion: `ttnn.linear` with `all_gather_matmul=True` where available (see `tt_transformers/tt/attention.py:1797` — AG-matmul only on T3K currently)
- CCL is already overlapped with compute in the tt_transformers framework using prefetcher

---

### Bottleneck 5 — Non-Standard TTNN Operations (IMPLEMENTATION RISK)

These operations have no direct TTNN equivalent and require careful reformulation:

#### a) Image feature injection: scatter-add

```python
# Reference:
x.view(-1, H)[is_image_patch] += image_features   # [N_valid, H] scattered into [B*S, H]
```

**TTNN approach** — pad image_features to full sequence length, then add:
```python
# Precompute: image_positions [N_valid] — index of each image token in [0, S)
# Build dense_image_feats [1, S, H] = zeros, filled at image positions
# Use ttnn.scatter or construct via index-based matmul:
#   scatter_matrix [S, N_valid] (binary, 1 at [image_pos_i, i]) — sparse
#   dense_image_feats = scatter_matrix @ image_features  → [S, H]
# Then: inputs_embeds = text_embeds + dense_image_feats
```
For typical ISL (≤ 36K tokens, ≤ 1764 image tokens), the scatter matrix is small.

#### b) Pooling gather: batched irregular index-select

```python
# Reference:
to_pool = image_features.reshape(B, -1, dim)[batch_idx, clip(pooled_patches_idx)]
# pooled_patches_idx: [N_windows, pool_window] — sparse indices into features
```

**TTNN approach** — flatten and use `ttnn.gather` or reindex during preprocessing:
```python
# Preprocess pooled_patches_idx into a flat gather index on CPU
# features_flat [N_total_patches, 2304]
# gather_idx [N_windows * pool_window]  — precomputed flat indices
# to_pool = ttnn.gather(features_flat, gather_idx).reshape(N_windows, pool_window, 2304)
```
The `pooled_patches_idx` is static per input shape — precompute the gather index in the processor and keep it on device.

#### c) valid_token boolean select

```python
# Reference:
return pooled_features.view(-1, H)[valid_token.flatten()]  # variable-length output
```

**TTNN approach** — keep padded throughout:
```python
# Pad pooled_features to [max_pooled_tokens, H] (max known per input config)
# Return padded tensor + count; text decoder pads image_patch positions accordingly
# Avoids variable-length tensors entirely
```

---

### Bottleneck 6 — Decode at Full KV Cache (LOW — acceptable)

**~4 ms/token (~250 tok/s) at max KV cache (S=36,864).**

Breakdown: KV reads 0.4 ms + MLP weights 1.8 ms + LM head 1.6 ms = 3.8 ms.
Memory-bandwidth bound; arithmetic intensity = 1 at decode.

LM head weight `[152,064 × 4,096]` = 1.2 GB: replicated across all 8 devices. Each device reads the full 1.2 GB on every decode step. If this becomes a bottleneck, vocab-parallel sharding reduces it to 150 MB/device.

---

### Summary: Risk-Ranked Bottleneck List

| Rank | Item | Risk | Action required |
|------|------|------|-----------------|
| 1 | SDPA compute at S=36K (video) | High | Accept; use flash-attn; note practical ISL is lower |
| 2 | ViT fp32 attention config | High | Verify TTNN HiFi4 SDPA; may need kernel override |
| 3 | Pooling gather (irregular indices) | High | Precompute flat gather index in processor; use `ttnn.gather` |
| 4 | Image feature scatter-add | High | Reformulate as padded-dense add; design early |
| 5 | ViT prefill compute (384 frames) | Medium | Data-parallel 8× already applied; 313 ms acceptable |
| 6 | CCL at max ISL | Medium | AG-matmul fusion where available; use prefetcher overlap |
| 7 | valid_token index-select | Medium | Keep padded throughout; no variable-length tensors |
| 8 | Intermediate ViT layer retention | Low | 162 MB storage for layers 18+24; fine |
| 9 | bfloat4_b mask construction | Low | One-time precompute; 648 MB DRAM |
| 10 | Decode at full KV cache | Low | ~250 tok/s acceptable; vocab-shard lm_head if needed |
