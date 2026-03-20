# Qwen3-TTS-12Hz-1.7B Architecture Analysis

## Model Source
- HuggingFace: `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- Config class: `Qwen3TTSForConditionalGeneration`
- Model Type: `qwen3_tts` (custom, not in transformers yet)

## Target Devices
- **Primary:** N150, N300 (single chip - model fits comfortably)
- **Extended:** T3K for batch inference (optional)

## Model Overview
Qwen3-TTS is a Text-to-Speech model with multiple neural network components:
1. **Talker** - Main text-to-audio decoder (28 layers)
2. **Code Predictor** - Audio code prediction head (5 layers)
3. **Speaker Encoder** - Speaker embedding extraction for voice cloning
4. **Speech Tokenizer Encoder** - Converts audio waveform to RVQ codes
5. **Speech Tokenizer Decoder** - Converts RVQ codes back to audio waveform

## CRITICAL: Complete Component Inventory

**ALL components below are neural networks that must be implemented for full functionality.**

### Main Model (`model.safetensors`) - 480 tensors

| Component | Sub-component | Tensors | Purpose | Status |
|-----------|---------------|---------|---------|--------|
| **speaker_encoder** | asp | 4 | Attentive Statistics Pooling | ✅ Reference |
| | blocks | 68 | Res2Net conv blocks | ✅ Reference |
| | fc | 2 | Final projection to 2048-dim | ✅ Reference |
| | mfa | 2 | Multi-scale Feature Aggregation | ✅ Reference |
| **talker** | model | 311 | 28-layer transformer decoder | ✅ PCC 0.978 |
| | code_predictor | 88 | 5-layer decoder + 15 LM heads | ✅ Working |
| | codec_head | 1 | Codec token prediction head | ✅ Working |
| | text_projection | 4 | Text embedding projection | ⚠️ Not tested |

### Speech Tokenizer (`speech_tokenizer/model.safetensors`) - 496 tensors

| Component | Sub-component | Tensors | Purpose | Status |
|-----------|---------------|---------|---------|--------|
| **encoder** | encoder | 28 | Conv encoder layers | ✅ Reference |
| | encoder_transformer | 96 | 8-layer transformer | ✅ Reference |
| | quantizer | 100 | RVQ quantization (16 codebooks) | ✅ Reference |
| | downsample | 1 | Temporal downsampling | ✅ Reference |
| **decoder** | quantizer | 36 | RVQ codebook lookup | ✅ FIXED |
| | pre_transformer | 93 | 8-layer pre-transformer | ✅ FIXED |
| | pre_conv | 2 | Pre-convolution layer | ✅ FIXED |
| | upsample | 22 | ConvNeXt upsampler (4×) | ✅ FIXED |
| | decoder | 118 | Conv decoder (480×) | ✅ FIXED |

### Implementation Status Summary

| Category | Component | Status | Notes |
|----------|-----------|--------|-------|
| ✅ **Working** | Talker (28 layers) | PCC 0.978 | Core transformer works |
| ✅ **Working** | Code Predictor (5 layers) | Working | 15 LM heads |
| ✅ **Working** | Speech Tokenizer Decoder | PCC 1.0 | 271 tensors, FIXED - outputs correct audio |
| ✅ **Reference** | Speaker Encoder | Reference done | 76 tensors, TTNN not started |
| ✅ **Reference** | Speech Tokenizer Encoder | Reference done | 225 tensors, TTNN not started |

### Verification Requirements

Before marking as complete:
- [x] Official `qwen_tts` package produces correct audio (verified via clone_ref.wav)
- [x] Reference decoder matches official (Energy PCC 0.93)
- [x] TTNN decoder matches reference exactly (PCC 1.0)
- [x] Reference encoder produces valid RVQ codes
- [x] Reference speaker encoder produces valid 2048-dim embeddings
- [x] End-to-end demo produces intelligible speech (pure reference pipeline complete)

### End-to-End Pipeline
```
Text Input
    ↓
[Talker] (28 layers, 2048 dim)
    ↓
Hidden States
    ↓
[Code Predictor] (5 layers, 1024 dim)
    ↓
Logits (15 code groups × 2048 vocab)
    ↓
Argmax/Sampling
    ↓
Codec Tokens [batch, 15, seq_len]
    ↓
[Speech Tokenizer Decoder] ← REQUIRED for audio output!
    ↓
Audio Waveform [batch, 1, num_samples] @ 24kHz
```

## Talker Specifications (Main Model)
| Parameter | Value |
|-----------|-------|
| hidden_size | 2048 |
| num_layers | 28 |
| num_attention_heads | 16 |
| num_kv_heads | 8 |
| head_dim | 128 |
| intermediate_size | 6144 |
| text_vocab_size | 151936 |
| audio_vocab_size | 3072 |
| max_position_embeddings | 32768 |
| rope_theta | 1000000 |
| rms_norm_eps | 1e-06 |
| num_code_groups | 16 |

## Code Predictor Specifications
| Parameter | Value |
|-----------|-------|
| hidden_size | 1024 |
| num_layers | 5 |
| num_attention_heads | 16 |
| num_kv_heads | 8 |
| head_dim | 128 |
| intermediate_size | 3072 |
| vocab_size | 2048 |
| num_lm_heads | 15 |

## Speech Tokenizer Decoder Specifications (REQUIRED for audio output)
The Speech Tokenizer converts codec tokens back to audio waveforms.

| Parameter | Value |
|-----------|-------|
| input_sample_rate | 24000 Hz |
| output_sample_rate | 24000 Hz |
| encode_downsample_rate | 1920 (12.5 Hz tokens) |
| decode_upsample_rate | 1920 |
| num_quantizers | 16 |
| codebook_size | 2048 |
| codebook_dim | 256 |

### Decoder Architecture
| Component | Details |
|-----------|---------|
| **Quantizer** | RVQ with 16 codebooks (2048 entries × 256 dim) |
| **Pre-Transformer** | 8 layers, 512 hidden, 1024 intermediate, 16 heads |
| **Upsampler** | 2× + 2× = 4× via ConvNeXt blocks |
| **Conv Decoder** | 8× + 5× + 4× + 3× = 480× via ConvTranspose1d |
| **Total Upsample** | 4× × 480× = 1920× (12.5 Hz → 24 kHz) |

### Decoder Pipeline
```
Codec Tokens [batch, 16, seq_len]
    ↓
Codebook Lookup (16 codebooks × 2048 × 256)
    ↓
Sum Embeddings [batch, seq_len, 512]
    ↓
Pre-Transformer (8 layers, 512 dim)
    ↓
Upsampler (ConvNeXt: 2× → 2×)
    ↓
Conv Decoder (ConvTranspose1d: 8× → 5× → 4× → 3×)
    ↓
Audio Waveform [batch, 1, num_samples]
```

## Architecture Classification
- **Attention**: GQA (16 heads, 8 kv_heads - 2:1 ratio)
- **Position Encoding**: MROPE (Multi-dimensional RoPE)
  - `rope_type`: default
  - `interleaved`: true
  - `mrope_section`: [24, 20, 20] (64 total = 2 * head_dim/4 for cos+sin)
- **Normalization**: RMSNorm (eps=1e-06), Pre-norm
- **MLP**: SwiGLU (silu activation)
- **Special Features**:
  - Multi-modal position encoding for audio/text
  - 16 code groups for audio token prediction
  - Speaker conditioning via speaker encoder

## MROPE (Multi-dimensional RoPE) Details
The model uses Multi-dimensional Rotary Position Embeddings with sections:
- Section 1: 24 dimensions (text position)
- Section 2: 20 dimensions (audio time position)
- Section 3: 20 dimensions (audio frequency position)

Total: 64 dimensions = head_dim / 2 (for cos/sin pairs)

## Reference Implementations
| Block | Reference Path | Notes |
|-------|---------------|-------|
| Attention | `models/tt_transformers/tt/attention.py` | GQA with RoPE, similar to Qwen3-VL |
| MLP | `models/tt_transformers/tt/mlp.py` | Standard SwiGLU (w1=gate, w2=down, w3=up) |
| RMSNorm | `models/tt_transformers/tt/rmsnorm.py` | Standard RMSNorm |
| RoPE | `models/demos/qwen3_vl/tt/rope.py` | MROPE variant needed |
| Full Model | `models/tt_transformers/tt/model.py` | Base transformer |
| Code Predictor | `models/demos/qwen3_tts/tt/code_predictor.py` | 5 layers + 15 LM heads |
| Speech Tokenizer | N/A | Requires Conv1d ops - see below |

### Speech Tokenizer Decoder Implementation Notes
The decoder uses Conv1d/ConvTranspose1d which may need:
- **Option A**: Use `ttnn.conv1d` if available and performant
- **Option B**: Reshape to 2D and use `ttnn.conv2d` with kernel height=1
- **Option C**: Fallback to PyTorch for conv layers, TTNN for transformer

The pre-transformer (8 layers) can use standard TTNN attention/MLP patterns.

## Weight Mapping (Talker)
| HuggingFace Key | TTNN Key |
|-----------------|----------|
| `talker.layers.{i}.self_attn.q_proj.weight` | `layers.{i}.attention.wq.weight` |
| `talker.layers.{i}.self_attn.k_proj.weight` | `layers.{i}.attention.wk.weight` |
| `talker.layers.{i}.self_attn.v_proj.weight` | `layers.{i}.attention.wv.weight` |
| `talker.layers.{i}.self_attn.o_proj.weight` | `layers.{i}.attention.wo.weight` |
| `talker.layers.{i}.mlp.gate_proj.weight` | `layers.{i}.feed_forward.w1.weight` |
| `talker.layers.{i}.mlp.up_proj.weight` | `layers.{i}.feed_forward.w3.weight` |
| `talker.layers.{i}.mlp.down_proj.weight` | `layers.{i}.feed_forward.w2.weight` |
| `talker.layers.{i}.input_layernorm.weight` | `layers.{i}.attention_norm.weight` |
| `talker.layers.{i}.post_attention_layernorm.weight` | `layers.{i}.ffn_norm.weight` |
| `talker.norm.weight` | `norm.weight` |
| `talker.embed_tokens.weight` | `tok_embeddings.weight` |

## Weight Mapping (Code Predictor)
| HuggingFace Key | TTNN Key |
|-----------------|----------|
| `talker.code_predictor.layers.{i}.self_attn.q_proj.weight` | `code_predictor.layers.{i}.attention.wq.weight` |
| `talker.code_predictor.layers.{i}.self_attn.k_proj.weight` | `code_predictor.layers.{i}.attention.wk.weight` |
| `talker.code_predictor.layers.{i}.self_attn.v_proj.weight` | `code_predictor.layers.{i}.attention.wv.weight` |
| `talker.code_predictor.layers.{i}.self_attn.o_proj.weight` | `code_predictor.layers.{i}.attention.wo.weight` |
| `talker.code_predictor.layers.{i}.mlp.gate_proj.weight` | `code_predictor.layers.{i}.feed_forward.w1.weight` |
| `talker.code_predictor.layers.{i}.mlp.up_proj.weight` | `code_predictor.layers.{i}.feed_forward.w3.weight` |
| `talker.code_predictor.layers.{i}.mlp.down_proj.weight` | `code_predictor.layers.{i}.feed_forward.w2.weight` |
| `talker.code_predictor.lm_head.{g}.weight` | `code_predictor.lm_heads.{g}.weight` |

## Weight Mapping (Speech Tokenizer Decoder)
Located in `speech_tokenizer/model.safetensors`:

| HuggingFace Key | Purpose |
|-----------------|---------|
| `decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum` | First codebook (semantic) |
| `decoder.quantizer.rvq_rest.vq.layers.{i}._codebook.embedding_sum` | Codebooks 1-15 (acoustic) |
| `decoder.pre_transformer.layers.{i}.self_attn.*` | Pre-transformer attention |
| `decoder.pre_transformer.layers.{i}.mlp.*` | Pre-transformer MLP |
| `decoder.upsample.{i}.0.conv.weight` | Upsample ConvTranspose1d |
| `decoder.upsample.{i}.1.dwconv.conv.weight` | ConvNeXt depthwise conv |
| `decoder.upsample.{i}.1.pwconv1.weight` | ConvNeXt pointwise conv 1 |
| `decoder.upsample.{i}.1.pwconv2.weight` | ConvNeXt pointwise conv 2 |
| `decoder.decoder.{i}.block.*.conv.weight` | Conv decoder blocks |
| `decoder.decoder.6.conv.weight` | Final output conv (1 channel) |

## Special Token IDs
| Token | ID |
|-------|-----|
| tts_bos_token_id | 151672 |
| tts_eos_token_id | 151673 |
| tts_pad_token_id | 151671 |
| codec_bos_id | 2149 |
| codec_eos_token_id | 2150 |
| codec_pad_id | 2148 |
| codec_think_id | 2154 |
| codec_nothink_id | 2155 |

## Speaker Encoder Architecture (76 tensors)

Extracts speaker embedding from reference audio for voice cloning.

| Parameter | Value |
|-----------|-------|
| **Input** | Mel-spectrogram from reference audio |
| **Output** | 2048-dim speaker embedding |
| **Sample Rate** | 24000 Hz |

### Architecture Components
| Component | Details |
|-----------|---------|
| **Blocks** | 6 Res2Net blocks with SE attention |
| **ASP** | Attentive Statistics Pooling |
| **MFA** | Multi-scale Feature Aggregation |
| **FC** | Final projection to 2048-dim |

### Weight Keys
- `speaker_encoder.blocks.{i}.*` - Res2Net blocks (0-5)
- `speaker_encoder.asp.*` - Attentive Statistics Pooling
- `speaker_encoder.mfa.*` - Multi-scale Feature Aggregation
- `speaker_encoder.fc.*` - Final projection

### Reference Implementations
- Similar to ECAPA-TDNN speaker encoder
- Conv1d based, may need PyTorch fallback initially

## Speech Tokenizer Encoder Architecture (225 tensors)

Converts audio waveform to RVQ codes. **REQUIRED for voice cloning.**

| Parameter | Value |
|-----------|-------|
| **Input** | Audio waveform [batch, 1, samples] @ 24kHz |
| **Output** | RVQ codes [batch, 16, seq_len] @ 12.5Hz |
| **Downsample Rate** | 1920× (24kHz → 12.5Hz) |

### Architecture Components
| Component | Details |
|-----------|---------|
| **Conv Encoder** | 7 conv layers, progressively downsample audio |
| **Encoder Transformer** | 8 layers, 512 hidden, 16 heads, bidirectional |
| **Quantizer** | RVQ with 16 codebooks (2048 × 256 each) |
| **Downsample** | Final temporal downsampling |

### Pipeline
```
Audio Waveform [batch, 1, samples] @ 24kHz
    ↓
Conv Encoder (7 layers, downsample)
    ↓
Encoder Transformer (8 layers, 512 hidden)
    ↓
RVQ Quantization (16 codebooks × 2048 × 256)
    ↓
RVQ Codes [batch, 16, seq_len] @ 12.5Hz
```

### Weight Keys
- `encoder.encoder.layers.{i}.*` - Conv encoder layers (0-6)
- `encoder.encoder_transformer.layers.{i}.*` - Transformer layers (0-7)
- `encoder.quantizer.rvq_first.*` - First codebook (semantic)
- `encoder.quantizer.rvq_rest.*` - Codebooks 1-15 (acoustic)
- `encoder.downsample.*` - Temporal downsampling

### Voice Cloning Pipeline
```
Reference Audio
    ↓
[Speech Tokenizer ENCODER] ← audio → RVQ codes (REQUIRED!)
    ↓
RVQ Codes
    ↓
[Talker + Code Predictor] ← generates new codes
    ↓
New RVQ Codes
    ↓
[Speech Tokenizer DECODER] ← RVQ codes → audio
    ↓
Generated Audio
```

**NOTE:** Without the Speech Tokenizer Encoder, voice cloning cannot work because there's no way to extract RVQ codes from reference audio.

## Implementation Order

### Phase 1: Core Transformer (COMPLETE)
1. ✅ **RMSNorm** - Use existing `models/tt_transformers/tt/rmsnorm.py`
2. ✅ **MROPE Setup** - Adapt from `models/demos/qwen3_vl/tt/rope.py` for TTS sections
3. ✅ **Attention** - Use `models/tt_transformers/tt/attention.py` (GQA with MROPE)
4. ✅ **MLP** - Use `models/tt_transformers/tt/mlp.py` (SwiGLU)
5. ✅ **Talker Decoder Layer** - Compose attention + MLP with pre-norm
6. ✅ **Full Talker Model** - 28 decoder layers + embeddings (PCC 0.978)
7. ✅ **Code Predictor** - 5-layer decoder + 15 LM heads

### Phase 2: Audio Pipeline (COMPLETE)
8. ✅ **Speech Tokenizer Decoder** - RVQ codes → audio waveform
   - **Status:** FIXED - TTNN matches reference exactly (PCC 1.0)
   - Components:
     - ✅ Codebook embedding lookup (RVQ with cluster_usage normalization)
     - ✅ Pre-transformer (8 layers)
     - ✅ Upsampler (ConvNeXt blocks)
     - ✅ Conv decoder (ConvTranspose1d with causal padding)

### Phase 3: Voice Cloning (REFERENCE COMPLETE)
9. ✅ **Speech Tokenizer Encoder** - audio → RVQ codes (225 tensors)
   - **Status:** Reference implementation complete, TTNN not started
   - Components:
     - ✅ Conv encoder (7 layers)
     - ✅ Encoder transformer (8 layers)
     - ✅ RVQ quantization (16 codebooks)

10. ✅ **Speaker Encoder** - audio → speaker embedding (76 tensors)
    - **Status:** Reference implementation complete, TTNN not started
    - Components:
      - ✅ Res2Net blocks with TDNN + SE (3 blocks)
      - ✅ ASP (Attentive Statistics Pooling)
      - ✅ MFA (Multi-scale Feature Aggregation)
    - Note: Expects 128 mel channels, not 80!

### Phase 4: Integration
11. ✅ **Demo with Audio Generation** - Full pipeline producing audio files
    - Pure reference implementation: `demo_pure_reference_tts.py`
    - All 5 components working: Speaker Encoder, Speech Encoder, Talker, Code Predictor, Decoder
    - Key fix: `trailing_text_hidden` handling for proper EOS generation

## Key Implementation Considerations

### ICL Mode Generation (Critical)
Voice cloning requires In-Context Learning (ICL) mode with proper handling of text embeddings during autoregressive generation.

**Key Algorithm (from official qwen_tts):**
```python
# During generation, for each step:
if generation_step < trailing_text_hidden.shape[1]:
    # Add remaining text embeddings
    inputs_embeds = inputs_embeds + trailing_text_hidden[:, generation_step].unsqueeze(1)
else:
    # After exhausting text embeddings, use tts_pad
    inputs_embeds = inputs_embeds + tts_pad_embed
```

**Critical Details:**
1. `trailing_text_hidden` = remaining text embeddings when `text_lens > codec_lens`
2. When `text_lens <= codec_lens`, `trailing_text_hidden = tts_pad_embed` (single token)
3. Without this logic, the model won't properly hit EOS tokens

**Input Construction:**
```
ICL Input = role_tokens + prefix + icl_embed
Where:
  - role_tokens = <|im_start|>assistant\n (3 tokens)
  - prefix = think + think_bos + lang_id + think_eos + speaker_embed + pad + bos
  - icl_embed = text_embed[:, :codec_lens] + codec_embed (or padded if text < codec)
```

**Next Step Embedding:**
```python
next_embed = sum(all_16_codebook_embeds) + trailing_text_hidden[step]
```

### MROPE Handling
The MROPE with sections [24, 20, 20] requires:
- Splitting position IDs into 3 dimensions
- Applying different RoPE frequencies to each section
- Reference: `transformers/models/qwen3_vl/modeling_qwen3_vl.py`

### Code Group Prediction
- 16 parallel code groups for audio token prediction
- Each group has its own LM head
- Requires special decoding logic for audio generation

### Autoregressive Generation
- Text tokens use standard causal attention
- Audio codes generated in parallel across code groups
- Position ID per seconds: 13 (audio temporal resolution)

## Memory Estimates (N150/N300)
| Component | Parameters | Memory (bfloat16) |
|-----------|------------|-------------------|
| Talker (28L) | ~1.5B | ~3 GB |
| Code Predictor (5L) | ~100M | ~200 MB |
| Speech Tokenizer Decoder | ~80M | ~160 MB |
| Total | ~1.8B | ~3.6 GB |

## Target Device Support
| Device | Support Level | Notes |
|--------|---------------|-------|
| **N150** | Primary | Single chip, full model fits in DRAM (~3.4 GB) |
| **N300** | Primary | Single chip, full model fits in DRAM (~3.4 GB) |
| T3K | Extended | For batch inference, data parallelism |

The 1.7B parameter model (~3.4 GB in bfloat16) fits comfortably on N150/N300 devices which have 12 GB DRAM.
