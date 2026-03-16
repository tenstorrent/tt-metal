---
name: architecture-mapping
description: >-
  Identify existing TTNN implementations that share architectural blocks with
  the target model, enabling code reuse and faster bring-up. Use when starting
  a new model bring-up, mapping model architecture, creating ARCHITECTURE.md,
  identifying similar TTNN implementations, or inventorying model components.
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

## CRITICAL: Complete Component Inventory

**Before implementing ANY code, you MUST create a complete inventory of ALL model components.**

### Component Inventory Checklist

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

## Output
Create `models/demos/{model_name}/ARCHITECTURE.md` documenting the analysis.
