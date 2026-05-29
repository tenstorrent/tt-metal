---
name: architecture-mapping
description: Identify existing TTNN implementations that share architectural blocks with the target model, enabling code reuse and faster bring-up. Use when starting a new model bring-up, analyzing model architecture, or mapping HuggingFace blocks to TTNN equivalents.
---

# SKILL: Architecture Mapping

## Purpose
Identify existing TTNN implementations that share architectural blocks with the target model, enabling code reuse and faster bring-up.

## Step-by-Step Process

### 1. Identify Model Family
Classify the target model into one of these categories:
- **Decoder-only LLM**: Llama, Falcon, Qwen, DeepSeek, Mistral
- **Encoder-only**: BERT, SentenceBERT, SqueezeBERT
- **Vision-Language (VLM)**: Qwen-VL, Molmo, LLaVA
- **Vision**: ViT, CLIP vision encoder
- **Speech/Audio (ASR)**: Whisper, Wav2Vec2
- **Text-to-Speech (TTS)**: SpeechT5, Qwen3-TTS, VITS, Bark

### 2. Map Architectural Blocks
For each block in the target model, find TTNN equivalents:

| Block Type | Reference Implementations |
|------------|--------------------------|
| Multi-Head Attention (MHA) | `models/demos/bert/tt/`, `models/demos/metal_BERT_large_11/tt/mha.py` |
| Grouped Query Attention (GQA) | `models/demos/llama3_70b_galaxy/tt/llama_attention.py` |
| Multi-Query Attention (MQA) | `models/demos/falcon7b_common/tt/falcon_attention.py` |
| RoPE (Rotary Position Embedding) | `models/demos/llama3_70b_galaxy/tt/llama_rope.py`, `models/demos/deepseek_v3/tt/rope.py` |
| RMSNorm | `models/common/rmsnorm.py`, `models/demos/gpt_oss/tt/rms_norm.py` |
| LayerNorm | `models/demos/bert/tt/`, `models/demos/qwen3_vl/tt/vision_layernorm.py` |
| SwiGLU MLP | `models/demos/llama3_70b_galaxy/tt/llama_mlp.py` |
| GELU MLP | `models/demos/bert/tt/`, `models/demos/falcon7b_common/tt/falcon_mlp.py` |
| MoE (Mixture of Experts) | `models/demos/deepseek_v3/tt/moe.py`, `models/demos/deepseek_v3/tt/experts.py` |
| KV-Cache | `models/demos/llama3_70b_galaxy/tt/llama_attention.py` |
| Paged Attention | `models/demos/llama3_70b_galaxy/tt/` (paged_attention_config) |
| Vision Attention | `models/demos/qwen3_vl/tt/vision_attention.py`, `models/demos/qwen25_vl/tt/vision_attention.py` |
| Patch Embedding | `models/demos/qwen3_vl/tt/patch_merger.py` |
| **TTS/Audio Blocks** | |
| Audio Codec Encoder | (audio → tokens) |
| Audio Codec Decoder | (tokens → audio waveform) |
| VQ/RVQ Codebook Lookup | Residual Vector Quantization embedding lookup |
| Conv1d / ConvTranspose1d | 1D convolutions for audio processing |
| ConvNeXt Block | Depthwise separable conv blocks for upsampling |

### 3. Identify Weight Naming Conventions
Map HuggingFace weight names to TTNN patterns:

```python
# Common mappings:
# HuggingFace → TTNN state_dict key
"model.layers.{i}.self_attn.q_proj.weight" → layer_name + "attention.wq.weight"
"model.layers.{i}.self_attn.k_proj.weight" → layer_name + "attention.wk.weight"
"model.layers.{i}.self_attn.v_proj.weight" → layer_name + "attention.wv.weight"
"model.layers.{i}.self_attn.o_proj.weight" → layer_name + "attention.wo.weight"
"model.layers.{i}.mlp.gate_proj.weight"   → layer_name + "feed_forward.w1.weight"
"model.layers.{i}.mlp.up_proj.weight"     → layer_name + "feed_forward.w3.weight"
"model.layers.{i}.mlp.down_proj.weight"   → layer_name + "feed_forward.w2.weight"
```

### 4. Document Similarities and Differences
Create ARCHITECTURE.md in your model folder with:

```markdown
# {Model Name} Architecture Analysis

## Model Family
{decoder-only LLM / VLM / encoder / etc.}

## Similar Implementations
| Component | Reference Implementation | Similarity |
|-----------|-------------------------|------------|
| Attention | models/demos/llama3_70b_galaxy/tt/llama_attention.py | GQA with RoPE |
| MLP | models/demos/llama3_70b_galaxy/tt/llama_mlp.py | SwiGLU activation |
| Norm | models/common/rmsnorm.py | RMSNorm |

## Key Differences
- {List differences from reference implementations}
- {e.g., QK-norm in attention, different head dimensions}

## Weight Mapping
| HuggingFace Key | TTNN Key |
|-----------------|----------|
| model.layers.{i}.self_attn.q_proj | layers.{i}.attention.wq |

## Implementation Order
1. {First block to implement}
2. {Second block}
...
```

## Common Architectural Patterns

### Attention Patterns
- **MHA**: All heads compute Q, K, V (BERT-style)
- **MQA**: Single K, V head shared across Q heads (Falcon)
- **GQA**: K, V heads grouped, fewer than Q heads (Llama, Qwen)

### Position Encoding
- **RoPE**: Llama, Qwen, Mistral, DeepSeek (rotary embeddings)
- **Absolute**: BERT-style learned embeddings
- **ALiBi**: Falcon-style attention bias

### Normalization
- **Pre-norm**: Norm before attention/MLP (Llama, GPT)
- **Post-norm**: Norm after attention/MLP (BERT)
- **QK-norm**: Additional norm on Q, K in attention (Molmo, some VLMs)

### TTS Model Architecture Patterns

TTS models typically have multiple stages that ALL need implementation:

```
Text Input
    ↓
[Text Encoder / Tokenizer]
    ↓
[Main Decoder] ← Transformer layers
    ↓
[Code Predictor] ← Predicts audio codec tokens
    ↓
[Codec Tokens] ← Multiple code groups
    ↓
[Audio Codec Decoder] ← Converts tokens to audio (REQUIRED for audio output!)
    ↓
Audio Waveform Output
```

**CRITICAL**: The Audio Codec Decoder is required to produce actual audio output!

#### Audio Codec Decoder Components
| Component | Purpose |
|-----------|---------|
| Codebook Lookup | Token IDs → Embeddings (RVQ) |
| Pre-Transformer | Process concatenated embeddings |
| Upsampler | Increase temporal resolution (ConvTranspose1d / ConvNeXt) |
| Conv Decoder | Generate final waveform |

## CRITICAL: Complete Component Inventory

**Before implementing ANY code, you MUST create a complete inventory of ALL model components.**

### Component Inventory Checklist

Create a table listing EVERY neural network component in the model:

```markdown
## Complete Component Inventory

| Component | Weight File | Tensor Count | Required For | Implementation Status |
|-----------|-------------|--------------|--------------|----------------------|
| Talker | model.safetensors | 300 | Token generation | Not started |
| Code Predictor | model.safetensors | 100 | RVQ codebook prediction | Not started |
| Speaker Encoder | model.safetensors | 76 | Voice cloning | Not started |
| Speech Tokenizer Encoder | speech_tokenizer/model.safetensors | 225 | Audio → RVQ codes | Not started |
| Speech Tokenizer Decoder | speech_tokenizer/model.safetensors | 271 | RVQ codes → Audio | Not started |
```

**CRITICAL RULES:**
1. List ALL components - not just the "main" transformer
2. For each component, note what it's REQUIRED FOR
3. If ANY component is missing, the model won't produce correct end-to-end output
4. Don't proceed to Reference phase until this inventory is COMPLETE

### Verification Command

Run this to discover all components:
```python
from safetensors.torch import load_file
state_dict = load_file("model.safetensors")
prefixes = {}
for k in state_dict.keys():
    prefix = k.split('.')[0]
    prefixes[prefix] = prefixes.get(prefix, 0) + 1
print(prefixes)  # Shows all top-level components
```

### For TTS/Audio Models - REQUIRED Components

TTS models have multiple REQUIRED neural network components:
- **Text Encoder/Embeddings** - text → embeddings
- **Main Decoder (Talker)** - generates codec tokens
- **Code Predictor** - predicts multiple RVQ codebooks
- **Speaker Encoder** - extracts speaker embedding (for voice cloning)
- **Speech Tokenizer Encoder** - audio → RVQ codes
- **Speech Tokenizer Decoder** - RVQ codes → audio waveform

**ALL of these are neural networks that need implementation, not just "infrastructure"!**

## Use case inventory

In addition to the component DAG, the architecture worker emits a `use_cases[]`
array describing every distinct inference path the model exposes. This is what
the orchestrator uses to dispatch per-use-case work in the post-bringup phases
(see `skills/orchestrator/SPEC_post_bringup.md`).

### How to discover use cases (model-agnostic)

1. **Inspect the HF model's class hierarchy.** Look at
   `transformers.<model>.modeling_<model>` for classes that have a `.generate()`
   method or are top-level inference classes:
   - Multiple task-specific classes (`XxxForAToB`, `XxxForCToD`) → one use case
     per class.
   - Single class with a task/modality arg → one use case per task value.
   - Single class with no `.generate()` → one inference use case (encoder-only,
     classifier, etc.).

2. **For each use case, derive the model-agnostic fields:**
   - `name`: lowercase short token from the HF class name (opaque identifier;
     e.g. `t2tt`, `text_generation`, `classification`). The orchestrator treats
     this as an opaque string.
   - `description`: one sentence.
   - `input_modality` / `output_modality`: pick from `{text, audio, image, video, none}`.
   - `components_used`: subset of `components[]` the use case's forward path touches.
     Read the HF class's `__init__` to see which sub-modules it instantiates.
   - `needs_ar`: true iff the class has `.generate()` (or inherits from
     `GenerationMixin`).
   - `needs_audio_out`: true iff `output_modality == "audio"`.
   - `hf_class`: full HF class name (string).
   - `validation_metric`: pick from the orchestrator's known set:
     - `bleu` — text out (translation / open-ended generation), `sacrebleu`
     - `wer` — text out (transcription / classification), `jiwer`
     - `ecapa_cos` — audio out (primary); falls back to re-ASR `char_similarity`
       if ECAPA scorer not available
     - `perplexity` — text out (language modeling)
     - `accuracy` — classification
     - `mse` / `pcc` — encoder-only embeddings against HF
   - `validation_threshold`: expressed in normalized form. The generation worker
     parses these:
     - Parity-relative for additive metrics: `"HF - 1.0"` (BLEU/accuracy allow
       1pt drift), `"HF + 0.05"` (WER allow 5% drift)
     - Absolute for cosine/sim metrics: `"≥ 0.95"`
   - `hybrid_notes`: optional string documenting parts of the pipeline that
     legitimately stay on the HF host (e.g. tokenizer-bound char prep for TTS).

3. **For models with no clear use case beyond "inference"** (BERT classification,
   ViT embeddings): emit one `use_case` entry with `name="inference"`,
   `components_used = ALL components`, `needs_ar` inferred from class, and metric
   `pcc` (encoder-only) or `accuracy` (classifier) against HF on a representative input.

### Schema added to `architecture_inventory.json`

```json
{
  "components": [ ... existing ... ],
  "use_cases": [
    {
      "name": "<short_token>",
      "description": "<one-sentence>",
      "input_modality": "text|audio|image|video|none",
      "output_modality": "text|audio|image|video|none",
      "components_used": ["<comp>", ...],
      "needs_ar": true,
      "needs_audio_out": false,
      "hf_class": "<HF class name>",
      "validation_metric": "<bleu|wer|ecapa_cos|perplexity|accuracy|mse|pcc>",
      "validation_threshold": "<expression>",
      "hybrid_notes": null
    }
  ]
}
```

### `ARCHITECTURE.md` additions

Add a `## Use cases` markdown table to the model's ARCHITECTURE.md listing each
entry plus the "Components used" column. Example shape:

```markdown
## Use cases

| Name | Input | Output | needs_ar | HF class | Metric | Threshold | Components used |
|------|-------|--------|----------|----------|--------|-----------|------------------|
| <name> | <mod> | <mod> | true | <Class> | bleu | HF - 1.0 | <comp_a>, <comp_b>, ... |
```

## Output
Create `models/demos/{model_name}/ARCHITECTURE.md` documenting the analysis.
Create `models/demos/{model_name}/architecture_inventory.json` with both
`components[]` AND `use_cases[]` populated per the schemas above.
