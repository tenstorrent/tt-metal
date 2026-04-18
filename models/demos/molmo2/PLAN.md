# Molmo2-8B TTNN Implementation Plan

## Model Overview

[Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B) (Allen AI) is a vision-language model with three sub-systems:

| Sub-system | Config |
|---|---|
| ViT encoder (`vit_config`) | 27 layers (25 used), hidden=1152, heads=16, head_dim=72, patch_size=14, input 378×378 → 729 patches, LayerNorm, GELU, learned pos. emb. |
| Vision adapter (`adapter_config`) | Multi-scale concat (layers 18+24) → attention-based pooling → SwiGLU projector (1152→12288→4096) |
| Language model (`text_config`) | 36 layers, hidden=4096, GQA 32/8, head_dim=128, SwiGLU (intermediate=12288), RMSNorm, QK-norm (qwen3-style), RoPE θ=1M, vocab=152064 |

Total: **8.66B parameters** (~17.3 GB @ BF16)
- ViT: 383M (4.4%)
- Adapter: 88M (1.0%)
- Language Model: 8,192M (94.6%)

## Architecture Data Flow

```
Image (378x378)
    ↓ patch_embedding (14×14 patches → 729 tokens, linear+bias)
    ↓ + learned positional embedding [729, 1152] (bicubic interp for non-native sizes)
    ↓
ViT layers 1–25  [383M params, LayerNorm, GELU, separate wq/wk/wv/wo + bias]
    ↓ collect hidden states at layers 18 and 24
    ↓ concat on hidden dim: [B*T, 729, 1152] × 2 → [B*T, 729, 2304]
    ↓
image_pooling_2d [9.3M]
    cross-attention: query = mean(pooled_patches_idx neighborhood)
    keys/values = gathered patch features (pooled_patches_idx from preprocessor)
    wq/wk/wv: 2304→1152, wo: 1152→1152 (all with bias)
    ↓
image_projector [78.6M]
    SwiGLU: w1/w3: 1152→12288, w2: 12288→4096 (no bias)
    ↓ [valid_tokens, 4096]
    ↓
splice into token sequence at <image_patch_id> positions
    ↓
Language Model [8,192M, 36 layers, GQA 32/8, QK-norm]
    ↓
Logits
```

## Reuse Map

| Component | Reuse Source | New Work |
|---|---|---|
| Language backbone | `models/tt_transformers/tt/` (Qwen3-VL path) | Vocab=152064, special token splicing |
| ViT blocks | `models/demos/vision/classification/vit/common/tt/ttnn_functional_vit.py` | patch_size=14, multi-layer output collection |
| Pos. emb. interpolation | `models/tt_dit/layers/embeddings.py` `_cropped_pos_embed` | Wire into ViT forward |
| Vision adapter cross-attn | `models/tt_transformers/tt/multimodal/llama_cross_attention.py` | `pooled_patches_idx` gather, input_dim=2304 |
| SwiGLU projector | `models/tt_transformers/tt/mlp.py` | Dims 1152→12288→4096 |
| Generator / prefill-decode | `models/demos/qwen3_vl/tt/generator.py` | Vision token splicing |
| Checkpoint loading | `models/tt_transformers/tt/load_checkpoints.py` | Molmo2 key remapping table |

---

## Directory Structure

```
models/demos/molmo2/
├── PLAN.md                       ← this file
├── README.md
├── requirements.txt
├── conftest.py
├── reference/
│   └── model.py                  # thin wrapper: loads HF Molmo2ForConditionalGeneration
├── tt/
│   ├── model_config.py           # Molmo2ModelArgs extends ModelArgs
│   ├── vision_block.py           # LayerNorm + attn + MLP (pre-norm)
│   ├── vision_attention.py       # bidirectional, separate wq/wk/wv/wo + bias
│   ├── vision_mlp.py             # GELU MLP with bias
│   ├── vision_transformer.py     # patch_embed + pos_embed + N blocks + multi-layer output
│   ├── image_pooling.py          # pooled_patches_idx gather + cross-attn
│   ├── image_projector.py        # SwiGLU: 1152→12288→4096
│   ├── vision_backbone.py        # Molmo2VisionBackbone: ViT + pooling + projector
│   ├── model.py                  # Molmo2TextModel (wraps tt_transformers Transformer)
│   ├── generator.py              # prefill/decode loop with vision splicing
│   └── load_weights.py           # weight name remapping for Molmo2 HF keys
├── tests/
│   ├── conftest.py
│   ├── test_vision_block.py      # Phase 1
│   ├── test_vision_transformer.py # Phase 1
│   ├── test_image_pooling.py     # Phase 2
│   ├── test_image_projector.py   # Phase 2
│   ├── test_vision_backbone.py   # Phase 2
│   ├── test_language_model.py    # Phase 3
│   └── test_full_model.py        # Phase 4
└── demo/
    ├── demo.py
    └── sample_prompts/
        └── demo.json
```

---

## Phase 0 — Setup & Reference Model (~1 day)

**Goal:** Working CPU reference baseline to diff against throughout all phases.

### Files to create

**`reference/model.py`** — thin wrapper around HF `Molmo2ForConditionalGeneration` that exposes each sub-module for isolated testing:
```python
from transformers import AutoModelForImageTextToText, AutoProcessor

class Molmo2Reference:
    def __init__(self, ckpt_dir):
        self.model = AutoModelForImageTextToText.from_pretrained(
            ckpt_dir, trust_remote_code=True, torch_dtype=torch.float32
        )
        self.processor = AutoProcessor.from_pretrained(ckpt_dir, trust_remote_code=True)

    @property
    def image_vit(self):
        return self.model.model.vision_backbone.image_vit

    @property
    def image_pooling_2d(self):
        return self.model.model.vision_backbone.image_pooling_2d

    @property
    def image_projector(self):
        return self.model.model.vision_backbone.image_projector

    @property
    def text_model(self):
        return self.model.model.model
```

**`tt/model_config.py`** — `Molmo2ModelArgs` extends `ModelArgs` from `models/tt_transformers/tt/model_config.py`:
- Reads `vit_config`, `adapter_config`, `text_config` from HF config JSON
- Implements `get_state_dict_prefix(module, layer_num=None)` for Molmo2 key scheme
- Provides `reference_vision_block(layer_num)`, `reference_image_pooling()`, etc.

**`tt/load_weights.py`** — key remapping table:
```python
MOLMO2_VIT_KEY_MAP = [
    ("model.vision_backbone.image_vit.patch_embedding.",  "image_vit.patch_embedding."),
    ("model.vision_backbone.image_vit.positional_embedding", "image_vit.positional_embedding"),
    ("model.vision_backbone.image_vit.transformer.resblocks.{N}.", "image_vit.transformer.resblocks.{N}."),
    ("model.vision_backbone.image_pooling_2d.", "image_pooling_2d."),
    ("model.vision_backbone.image_projector.", "image_projector."),
]
MOLMO2_TEXT_KEY_MAP = [
    ("model.model.wte.",                          "model.tok_embeddings."),
    ("model.model.ln_f.",                         "model.norm."),
    ("model.model.blocks.{N}.attention_norm.",    "model.layers.{N}.attention_norm."),
    ("model.model.blocks.{N}.self_attn.att_proj.","model.layers.{N}.attention.wqkv."),
    ("model.model.blocks.{N}.self_attn.attn_out.","model.layers.{N}.attention.wo."),
    ("model.model.blocks.{N}.ffn_norm.",          "model.layers.{N}.ffn_norm."),
    ("model.model.blocks.{N}.mlp.ff_proj.",       "model.layers.{N}.feed_forward.w13."),
    ("model.model.blocks.{N}.mlp.ff_out.",        "model.layers.{N}.feed_forward.w2."),
    ("model.lm_head.",                            "model.output."),
]
```

**Testing (CPU-only, no TTNN):**
- `assert` that `Molmo2Reference` outputs match `transformers` pipeline directly
- Verify `image_vit`, `image_pooling_2d`, `image_projector`, `text_model` are accessible

---

## Phase 1 — Vision Transformer Encoder (~2–3 days)

**Goal:** TTNN ViT encoder matching `Molmo2VisionTransformer` (25 layers, hidden=1152, LayerNorm, GELU, learned interpolatable pos. emb., separate wq/wk/wv/wo + bias).

### Key design decisions
- Base `vision_block.py` on `models/demos/vision/classification/vit/common/tt/ttnn_functional_vit.py` — already has separate QKV + bias + LayerNorm + GELU
- Borrow positional embedding interpolation from `models/tt_dit/layers/embeddings.py` `_cropped_pos_embed` pattern (CPU-side bicubic via `torch.nn.functional.interpolate`, matching `models/experimental/pi0/tt/ttnn_siglip.py` fallback approach)
- `vision_transformer.py` collects hidden states from **all layers** (not just final) and returns the full list; the backbone selects indices [18, 24]

### `tt/vision_attention.py`
```python
class Molmo2VisionAttention:
    # Separate wq, wk, wv, wo projections (all bias=True)
    # ttnn.linear for each projection
    # ttnn.transformer.split_query_key_value_and_split_heads
    # ttnn.transformer.scaled_dot_product_attention (is_causal=False)
    # ttnn.transformer.concatenate_heads + ttnn.linear for wo
```

### `tt/vision_mlp.py`
```python
class Molmo2VisionMLP:
    # w1: hidden→intermediate (bias=True), GELU, w2: intermediate→hidden (bias=True)
    # ttnn.linear + ttnn.gelu + ttnn.linear
```

### `tt/vision_block.py`
```python
class Molmo2VisionBlock:
    # LayerNorm (attention_norm) → VisionAttention → residual add
    # LayerNorm (ffn_norm)       → VisionMLP       → residual add
```

### `tt/vision_transformer.py`
```python
class Molmo2VisionTransformer:
    def forward(self, x, patch_num=(27,27)):
        x = self.patch_embedding(x)        # ttnn.fold() + ttnn.linear(bias=True)
        x = self.add_pos_emb(x, patch_num) # interpolate if patch_num != (27,27)
        hidden_states = []
        for block in self.blocks[:self.num_layers]:  # num_layers=25
            x = block(x)
            hidden_states.append(x)
        return hidden_states               # list of 25 tensors [B, 729, 1152]
```

### Tests

**`tests/test_vision_block.py`:**
```python
@pytest.mark.parametrize("layer_num", [0, 12, 24])
def test_vision_block(layer_num, mesh_device, reset_seeds, ensure_gc):
    # Input: torch.randn(1, 729, 1152)
    # Reference: Molmo2VisionBlock from HF config
    # PCC >= 0.99
    passing, pcc_msg = comp_pcc(ref_output, tt_output, pcc=0.99)
    assert passing
```

**`tests/test_vision_transformer.py`:**
```python
@pytest.mark.parametrize("num_layers", [1, 5, 25], ids=["1L", "5L", "full"])
@pytest.mark.parametrize("patch_num", [(27,27), (14,14)], ids=["native", "interp"])
def test_vision_transformer(num_layers, patch_num, mesh_device, reset_seeds, ensure_gc):
    # PCC >= 0.99 for num_layers in [1, 5]
    # PCC >= 0.91 for num_layers == 25
    # Extra assertions when num_layers == 25:
    #   comp_pcc(ref_hidden[18], tt_hidden[18], pcc=0.91)
    #   comp_pcc(ref_hidden[24], tt_hidden[24], pcc=0.91)
```

**Gate:** All PCC thresholds met before starting Phase 2.

---

## Phase 2 — Vision Adapter (~5–8 days, most novel component)

**Goal:** TTNN `Molmo2VisionBackbone` — multi-scale feature extraction + attention pooling + SwiGLU projection.

### `tt/image_pooling.py` — `Molmo2ImagePooling`

This is the most novel component. The `pooled_patches_idx` tensor (shape `[B, N_out, K_pool]`) is computed by the image processor CPU-side and maps each output visual token to a neighborhood of K_pool source ViT patches.

```python
class Molmo2ImagePooling:
    # wq: Linear(2304→1152, bias=True)  — input_dim = 1152 * len(vit_layers) = 2304
    # wk: Linear(2304→1152, bias=True)
    # wv: Linear(2304→1152, bias=True)
    # wo: Linear(1152→1152, bias=True)

    def forward(self, features_2304, pooled_patches_idx, attn_mask=None):
        # 1. Gather: index features_2304 [B, T*N, 2304] with pooled_patches_idx [B, N_out, K_pool]
        #    → to_pool [B*N_out, K_pool, 2304]   (ttnn.gather or CPU-side index + to_device)
        # 2. Query: masked mean of to_pool → [B*N_out, 1, 2304]
        # 3. Cross-attention: Q=query, KV=to_pool
        #    → [B*N_out, 1, 1152] → reshape [B, N_out, 1152]
        # Base on models/tt_transformers/tt/multimodal/llama_cross_attention.py structure
```

### `tt/image_projector.py` — `Molmo2ImageProjector`

```python
class Molmo2ImageProjector:
    # SwiGLU: same pattern as models/tt_transformers/tt/mlp.py
    # w1: Linear(1152→12288, bias=False)  — gate
    # w3: Linear(1152→12288, bias=False)  — up
    # w2: Linear(12288→4096, bias=False)  — down
    # output = w2(silu(w1(x)) * w3(x))
```

### `tt/vision_backbone.py` — `Molmo2VisionBackbone`

```python
class Molmo2VisionBackbone:
    def forward(self, images, pooled_patches_idx):
        # images: [B, T, N_patches, pixel_dim]  T=num_crops
        # 1. ViT forward → hidden_states list[25]
        # 2. Extract [hidden_states[18], hidden_states[24]]
        # 3. Concat on last dim → [B*T, N, 2304]
        # 4. Reshape to [B, T*N, 2304] for gather
        # 5. image_pooling_2d(features_2304, pooled_patches_idx)  → [B, N_out, 1152]
        # 6. image_projector(pooled) → [B, N_out, 4096]
        # 7. Filter by valid_token mask → [valid_tokens, 4096]
```

### Tests

**`tests/test_image_pooling.py`:**
```python
def test_image_pooling(mesh_device, reset_seeds, ensure_gc):
    # Construct synthetic features [1, 729, 2304] and random pooled_patches_idx [1, 64, 9]
    # Reference: ViTMultiHeadDotProductAttention.forward(query, kv)
    # PCC >= 0.99
    # Also test with pooling_attention_mask=True (masked mean query)
```

**`tests/test_image_projector.py`:**
```python
@pytest.mark.parametrize("num_tokens", [256, 729])
def test_image_projector(num_tokens, mesh_device, reset_seeds, ensure_gc):
    # Input: [num_tokens, 1152], reference: ImageProjectorMLP
    # PCC >= 0.99
```

**`tests/test_vision_backbone.py`:**
```python
@pytest.mark.parametrize("num_images,num_crops", [(1,1), (1,4)])
def test_vision_backbone(num_images, num_crops, mesh_device, reset_seeds, ensure_gc):
    # Use real image preprocessed with Molmo2Processor
    # Reference: Molmo2VisionBackbone.forward(images, pooled_patches_idx)
    # PCC >= 0.95 on final [valid_tokens, 4096] output
```

**Gate:** All PCC thresholds met before starting Phase 3.

---

## Phase 3 — Language Model (~2–3 days)

**Goal:** TTNN `Molmo2TextModel` using `models/tt_transformers/tt/model.py` `Transformer`, configured for Molmo2's text config.

**Key parameters:** 36 layers, hidden=4096, GQA 32/8, head_dim=128, SwiGLU (intermediate=12288), RMSNorm, QK-norm `qk_norm_type="qwen3"` (per-head normalization, same path as Qwen3-VL), RoPE θ=1M, total vocab 152064.

### `tt/model.py`
```python
class Molmo2TextModel:
    # Instantiates models/tt_transformers/tt/model.py Transformer
    # with Molmo2TextConfig parameters
    # Key differences from standard LLaMA:
    #   - vocab_size = 151936 + 128 (extra image special tokens)
    #   - use_qk_norm = True, qk_norm_type = "qwen3"
    #   - rope_theta = 1_000_000
    #   - 36 layers (not 32)
    #   - intermediate_size = 12288 (not 18944 — this is Molmo2-specific)
```

The `qk_norm_type="qwen3"` path applies RMSNorm per head-dim (not per full head projection), which is already implemented in the Qwen3-VL attention module — confirm it is exercised via the existing `use_qk_norm` config flag.

### Tests

**`tests/test_language_model.py`:**
```python
@pytest.mark.parametrize("num_layers", [1, 4, 36], ids=["1L", "4L", "full"])
def test_language_model_text_only(num_layers, mesh_device, reset_seeds, ensure_gc):
    # Random token IDs, text-only sequence
    # Reference: Molmo2TextModel first N blocks
    # PCC >= 0.99 for num_layers in [1, 4]; >= 0.95 for full 36

@pytest.mark.parametrize("vision_token_fraction", [0.0, 0.3])
def test_language_model_mixed_tokens(vision_token_fraction, mesh_device, reset_seeds, ensure_gc):
    # Mix: real text token embeddings + mock [N, 4096] vision embeddings spliced in
    # PCC >= 0.95
```

**Gate:** All PCC thresholds met before starting Phase 4.

---

## Phase 4 — E2E Integration + Generator (~4–5 days)

**Goal:** Full `Molmo2ForConditionalGeneration` — image preprocessing → vision backbone → token splicing → LM prefill → autoregressive decode.

### `tt/generator.py`

Follows the pattern of `models/demos/qwen3_vl/tt/generator.py`:

```python
class Molmo2Generator:
    def prefill_forward_vision(self, images, pooled_patches_idx):
        # → [valid_tokens, 4096] visual token embeddings

    def prefill_forward_text(self, input_ids, visual_embeds, image_positions):
        # 1. Embed input_ids via wte → [B, S, 4096]
        # 2. Splice visual_embeds at image_patch_id token positions
        # 3. Run LM prefill with merged embeddings
        # 4. Populate KV cache

    def decode_forward(self, token_ids, position_ids):
        # Standard single-token decode with KV cache
```

**Special token handling:** `image_patch_id=151938` marks positions in `input_ids` where visual embeddings should replace text embeddings. The processor pre-fills these positions with `image_patch_id` values; the generator detects them and substitutes the visual embeddings before the LM forward pass.

### Tests

**`tests/test_full_model.py`:**
```python
@pytest.mark.parametrize("prompt_file", ["single_image_qa.json"])
def test_full_model_prefill(prompt_file, mesh_device, reset_seeds, ensure_gc):
    # Load real image + question from prompt_file
    # Run full model (vision backbone → token splice → LM prefill)
    # comp_pcc(ref_lm_hidden, tt_lm_hidden, pcc=0.90) before lm_head
    # Assert first 20 greedy-decoded tokens == HF reference tokens

def test_full_model_decode(mesh_device, reset_seeds, ensure_gc):
    # Prefill + decode 50 tokens
    # Assert generated text contains expected answer keywords (e.g. "dog", "cat")
```

**Gate:** Greedy decode produces coherent answers matching HF reference before Phase 5.

---

## Phase 5 — Demo & Performance (~5–7 days)

**Goal:** Polished demo script, multi-image support, performance profiling.

### Files
- `demo/demo.py` — follows `models/demos/qwen3_vl/demo/demo.py` pattern
  - Supports single image, multi-image, and video-frame inputs
  - BERTScore validation for CI
- `demo/sample_prompts/demo.json` — example image+question pairs

### Performance targets (N150 single-chip baseline)
| Metric | Target |
|---|---|
| ViT prefill (1 image, 729 tokens) | < 50 ms |
| LM prefill throughput | Track tokens/s |
| Decode throughput | Track tokens/s |

---

## Testing Criteria Summary

| Phase | Test File | Test | PCC Threshold | Additional Criteria |
|---|---|---|---|---|
| 1 | `test_vision_block.py` | Single ViT block | ≥ 0.99 | Layers 0, 12, 24 |
| 1 | `test_vision_transformer.py` | ViT 1–5 layers | ≥ 0.99 | — |
| 1 | `test_vision_transformer.py` | ViT full 25 layers | ≥ 0.91 | Layer 18 & 24 hidden states verified; pos. emb. interpolation tested |
| 2 | `test_image_pooling.py` | image_pooling_2d | ≥ 0.99 | With and without attention mask |
| 2 | `test_image_projector.py` | image_projector | ≥ 0.99 | Token counts 256 and 729 |
| 2 | `test_vision_backbone.py` | vision_backbone | ≥ 0.95 | 1 crop and 4 crops |
| 3 | `test_language_model.py` | LM 1–4 layers | ≥ 0.99 | Text-only |
| 3 | `test_language_model.py` | LM full 36 layers | ≥ 0.95 | Mixed text + vision embeddings |
| 4 | `test_full_model.py` | Full prefill | ≥ 0.90 | On hidden states before lm_head |
| 4 | `test_full_model.py` | Full decode | — | First 20 tokens exact match vs HF |

All tests follow the fixture pattern from `models/demos/qwen3_vl/tests/conftest.py`:
- `mesh_device` (indirect, parametrized for N150/N300/T3K/TG)
- `reset_seeds` and `ensure_gc` (autouse)
- `comp_pcc` from `models.utility_functions`

---

## Effort Estimate

| Phase | Effort | Primary New Work |
|---|---|---|
| Phase 0: Setup & reference | ~1 day | Directory scaffold, HF wrapper, key map |
| Phase 1: ViT encoder | ~2–3 days | Multi-layer output, pos. emb. interpolation |
| Phase 2: Vision adapter | ~5–8 days | `pooled_patches_idx` gather, cross-attn pooling |
| Phase 3: Language model | ~2–3 days | Molmo2TextConfig wiring into tt_transformers |
| Phase 4: E2E integration | ~4–5 days | Token splicing, generator, checkpoint loading |
| Phase 5: Demo & perf | ~5–7 days | Demo script, profiling, multi-image |
| **Total** | **~4–6 weeks** | |
