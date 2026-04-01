# Inworld TTS Architecture Analysis

**Source**: https://github.com/inworld-ai/tts
**Paper**: arXiv:2507.21138
**Codec**: xcodec2-compatible (FSQ, not RVQ)

## Model Family
**SpeechLM-based TTS** -- A fine-tuned LLaMA causal LM that autoregressively generates audio codec tokens from text + audio prompt, followed by a codec decoder that converts tokens to waveforms.

## High-Level Pipeline

```
Text + Audio Prompt
       |
[Audio Encoder] -- encodes prompt audio to FSQ codes (50 tokens/sec)
       |
[Tokenizer] -- maps text + speech codes into LLM token IDs (vocab: 193,856)
       |
[LLaMA LLM] -- autoregressively generates speech token IDs
       |
[Extract Speech IDs] -- parse <|s_X|> tokens -> integer code IDs
       |
[Codec Decoder] -- FSQ dequantize -> VocosBackbone (12 transformer layers) -> ISTFT -> waveform
       |
Audio Output (16kHz)
```

## Complete Component Inventory

| Component | Source | Tensor Count | Required For | Implementation Status |
|-----------|--------|-------------|--------------|----------------------|
| **LLaMA LLM** (1B or 8B) | HuggingFace checkpoint | ~1B-8B params | Autoregressive speech token generation | Existing (models/demos/llama3_70b_galaxy/) |
| **Codec Encoder: AcousticEncoder** | codec checkpoint | ~15M params | Prompt audio -> acoustic features | Not started |
| **Codec Encoder: Wav2Vec2-BERT** | facebook/w2v-bert-2.0 | ~580M params | Prompt audio -> semantic features | Not started (frozen pretrained) |
| **Codec Encoder: SemanticEncoder** | codec checkpoint | ~6M params | Wav2Vec2 features -> semantic hidden | Not started |
| **Codec Encoder: Fusion Layer** | codec checkpoint | Linear(2048,2048) | Fuse semantic + acoustic | Not started |
| **Codec Encoder: FSQ Quantizer** | codec checkpoint | FSQ levels=[4]x8 | Hidden states -> VQ codes | Not started |
| **Codec Decoder: FSQ Dequantizer** | codec checkpoint | Same FSQ | VQ codes -> embeddings (2048-dim) | Not started |
| **Codec Decoder: fc_post_a** | codec checkpoint | Linear(2048,1024) | Project FSQ embeddings down | Not started |
| **Codec Decoder: VocosBackbone** | codec checkpoint | ~75M params | Embeddings -> hidden states | Not started |
| **Codec Decoder: ISTFTHead** | codec checkpoint | Linear(1024,1282) | Hidden states -> waveform | Not started |

## Component Details

### 1. LLaMA LLM (Main Generation Model)
- **Base**: `meta-llama/Llama-3.2-1B-Instruct` (or 8B variant)
- **Vocab size**: 193,856 (128k LLaMA base + 65,536 speech tokens + special tokens)
- **Attention**: Flash Attention 2 (standard causal LM)
- **Generation**: temperature=0.8, top_k=50, repetition_penalty=1.1-1.4, max_tokens=1792
- **Reference Implementation**: `models/demos/llama3_70b_galaxy/tt/` (GQA with RoPE, SwiGLU MLP, RMSNorm)

### 2. Audio Codec Encoder

#### 2a. AcousticEncoder
- Input: raw waveform [batch, 1, samples] at 16kHz
- Initial Conv1d(1, 48, kernel_size=7)
- 5 EncoderBlocks with `up_ratios=[2, 2, 4, 4, 5]` (total downsampling = 320x)
  - Each block: 3 ResidualUnits (SnakeBeta + Conv1d) -> stride Conv1d (doubles channels)
  - Channel progression: 48 -> 96 -> 192 -> 384 -> 768 -> 1536
- Final: SnakeBeta + Conv1d(1536, 1024, kernel_size=3)
- Output: [batch, seq_len, 1024]
- **Activations**: SnakeBeta (x + (1/beta) * sin^2(alpha * x)) -- custom, not standard

#### 2b. Wav2Vec2-BERT (Frozen)
- Model: `facebook/w2v-bert-2.0`
- Extracts hidden_states[16] (layer 16 features)
- Input: preprocessed audio features from AutoFeatureExtractor
- Output: [batch, seq_len, 1024] semantic features

#### 2c. SemanticEncoder
- Conv1d(1024, 1024, k=3) -> ResidualBlock(ReLU+Conv+ReLU+Conv) -> Conv1d(1024, 1024, k=3)
- Output: [batch, 1024, seq_len]

#### 2d. Fusion + Quantization
- Concatenate semantic (1024) + acoustic (1024) -> [batch, seq_len, 2048]
- Linear(2048, 2048) fusion
- **FSQ (Finite Scalar Quantization)**: levels=[4,4,4,4,4,4,4,4], num_quantizers=1
  - Codebook size: 4^8 = 65,536 entries
  - Single code per timestep (flat, not multi-stream like RVQ)
- Output: [batch, 1, seq_len] integer VQ codes

### 3. Audio Codec Decoder

#### 3a. FSQ Dequantizer
- Same FSQ config as encoder
- VQ code indices -> 2048-dim embeddings via `quantizer.get_output_from_indices()`

#### 3b. fc_post_a
- Linear(2048, 1024) -- projects embeddings down for VocosBackbone

#### 3c. VocosBackbone (12 Transformer Layers)
- `embed`: Conv1d(1024, 1024, kernel_size=7, padding=3)
- `prior_net`: 2x ResnetBlock(1024, GroupNorm(32), swish, Conv1d(k=3), dropout=0.1)
- **12 TransformerBlocks** (pre-norm, NON-CAUSAL):
  - `att_norm`: RMSNorm(1024)
  - `att`: MHA(dim=1024, heads=16, head_dim=64) with RoPE (pos_emb_dim=64)
    - Fused QKV: Linear(1024, 3072, bias=False) -> split into Q,K,V
    - RoPE from `torchtune.modules.RotaryPositionalEmbeddings`
    - `scaled_dot_product_attention(is_causal=False)` -- BIDIRECTIONAL
    - Output: Linear(1024, 1024, bias=False)
  - `ffn_norm`: RMSNorm(1024)
  - `mlp`: Linear(1024, 4096) -> SiLU -> Linear(4096, 1024) (NO gate, just SiLU MLP)
- `final_layer_norm`: LayerNorm(1024, eps=1e-6)
- `post_net`: 2x ResnetBlock(1024, same as prior_net)

#### 3d. ISTFTHead
- Linear(1024, 1282) -- predicts n_fft+2 = 1282 values
- Split into magnitude (641) + phase (641)
- magnitude = exp(clamp(mag, max=1e2))
- Complex: S = mag * (cos(phase) + j*sin(phase))
- ISTFT: n_fft=1280, hop_length=320, win_length=1280, padding="same"
- Output: [batch, 1, num_samples] audio waveform

## Similar Implementations

| Component | Reference Implementation | Similarity |
|-----------|-------------------------|------------|
| LLaMA LLM | `models/demos/llama3_70b_galaxy/tt/llama_attention.py` | GQA with RoPE, SwiGLU MLP, RMSNorm -- direct reuse |
| LLaMA Embedding | `models/demos/llama3_70b_galaxy/tt/llama_embedding.py` | Token embedding -- need expanded vocab |
| LLaMA LM Head | `models/demos/llama3_70b_galaxy/tt/lm_head.py` | Output projection -- need expanded vocab |
| VocosBackbone Attention | `models/demos/bert/tt/` (MHA pattern) | MHA but NON-CAUSAL, with RoPE |
| VocosBackbone RoPE | `models/demos/llama3_70b_galaxy/tt/llama_rope.py` | Same RoPE concept but pos_emb_dim=64 |
| VocosBackbone RMSNorm | `models/common/rmsnorm.py` | Identical RMSNorm |
| VocosBackbone MLP | `models/demos/bert/tt/` (GELU MLP pattern) | SiLU MLP (not SwiGLU -- no gate) |
| Conv1d operations | `models/experimental/speecht5_tts/tt/ttnn_speecht5_postnet.py` | Conv1d patterns |
| Postnet ResnetBlock | `models/experimental/speecht5_tts/tt/ttnn_speecht5_postnet.py` | GroupNorm + Conv1d residual blocks |

## Key Differences from Standard LLaMA

| Feature | Standard LLaMA | Inworld TTS LLaMA |
|---------|---------------|-------------------|
| Vocab size | 128,256 | 193,856 (+ 65,536 speech + special tokens) |
| Task | Text generation | Speech token generation |
| Output | Text tokens | `<\|s_X\|>` speech tokens -> codec decoder |

## Key Differences: VocosBackbone vs Standard Transformer

| Feature | VocosBackbone | Standard LLaMA |
|---------|--------------|----------------|
| Attention | Bidirectional (is_causal=False) | Causal |
| MLP | SiLU (no gate) | SwiGLU (gated) |
| Norm | Pre-norm RMSNorm + final LayerNorm | Pre-norm RMSNorm only |
| Position | RoPE (dim=64) | RoPE (dim=head_dim) |
| QKV | Fused single Linear(d, 3d) | Separate Q, K, V projections |
| Depth | 12 layers | 16-80+ layers |
| Surrounding | Conv1d prior_net/post_net | None |

## Weight Mapping

### LLaMA Weights
Standard HuggingFace LLaMA checkpoint -- existing weight loading should work with expanded vocab.

### Codec Decoder Weights (from xcodec2 checkpoint)
| Checkpoint Key | Component |
|---------------|-----------|
| `generator.quantizer.*` | FSQ quantizer (dequantizer) |
| `generator.backbone.embed.*` | VocosBackbone initial Conv1d |
| `generator.backbone.prior_net.*` | VocosBackbone prior ResnetBlocks |
| `generator.backbone.transformers.{i}.att_norm.*` | Transformer layer i attention RMSNorm |
| `generator.backbone.transformers.{i}.att.c_attn.*` | Transformer layer i fused QKV (1024 -> 3072) |
| `generator.backbone.transformers.{i}.att.c_proj.*` | Transformer layer i output proj (1024 -> 1024) |
| `generator.backbone.transformers.{i}.ffn_norm.*` | Transformer layer i FFN RMSNorm |
| `generator.backbone.transformers.{i}.mlp.fc1.*` | Transformer layer i MLP up (1024 -> 4096) |
| `generator.backbone.transformers.{i}.mlp.fc2.*` | Transformer layer i MLP down (4096 -> 1024) |
| `generator.backbone.final_layer_norm.*` | Final LayerNorm |
| `generator.backbone.post_net.*` | VocosBackbone post ResnetBlocks |
| `generator.head.out.*` | ISTFTHead Linear(1024, 1282) |
| `fc_post_a.*` | Post-FSQ projection Linear(2048, 1024) |

### Codec Encoder Weights (from xcodec2 checkpoint)
| Checkpoint Key | Component |
|---------------|-----------|
| `CodecEnc.*` | AcousticEncoder |
| `SemanticEncoder_module.*` | SemanticEncoder |
| `fc_prior.*` | Fusion layer Linear(2048, 2048) |
| `generator.quantizer.*` | FSQ quantizer (shared with decoder) |

## Implementation Order

### Priority 1: Codec Decoder (required for audio output)
1. FSQ dequantizer (codebook lookup)
2. fc_post_a Linear(2048, 1024)
3. VocosBackbone: Conv1d embed
4. VocosBackbone: ResnetBlocks (prior_net)
5. VocosBackbone: 12 TransformerBlocks (attention + MLP)
6. VocosBackbone: ResnetBlocks (post_net) + final LayerNorm
7. ISTFTHead (Linear + ISTFT signal processing)

### Priority 2: LLaMA LLM (reuse existing)
8. Expand existing LLaMA implementation for 193,856 vocab
9. Verify autoregressive generation produces valid speech tokens

### Priority 3: Codec Encoder (for voice cloning / prompt encoding)
10. AcousticEncoder (Conv1d chain with SnakeBeta)
11. Wav2Vec2-BERT integration (frozen pretrained)
12. SemanticEncoder (Conv1d chain)
13. Fusion layer + FSQ quantizer

## Audio Constants

| Parameter | Value |
|-----------|-------|
| Sample rate | 16,000 Hz |
| Token rate | 50 tokens/sec |
| Hop length | 320 samples |
| FSQ levels | [4,4,4,4,4,4,4,4] |
| FSQ codebook size | 65,536 (4^8) |
| FSQ num_quantizers | 1 |
| ISTFT n_fft | 1280 |
| ISTFT hop_length | 320 |
| ISTFT win_length | 1280 |

## Inference Settings (Defaults)

| Parameter | Value |
|-----------|-------|
| temperature | 0.8 |
| max_tokens | 1,792 |
| top_k | 50 |
| top_p | 1.0 |
| repetition_penalty | 1.1-1.4 |
| frequency_penalty | 0.3-0.4 |
| seed | 42 |
