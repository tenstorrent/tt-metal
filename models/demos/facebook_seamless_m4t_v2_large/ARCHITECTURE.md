<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
SPDX-License-Identifier: Apache-2.0
-->

# SeamlessM4T-v2-Large Architecture Analysis

**HF model id:** `facebook/seamless-m4t-v2-large`
**Target device:** p150 (Blackhole, single chip)
**HF transformers reference:** `transformers/models/seamless_m4t_v2/modeling_seamless_m4t_v2.py`
**HF config reference:** `transformers/models/seamless_m4t_v2/configuration_seamless_m4t_v2.py`

## Model Family

**Multi-modal speech-and-text encoder-decoder translation model.** A single
`SeamlessM4Tv2Model` checkpoint contains FIVE distinct neural networks that
ALL must be implemented in TTNN to support the full S2ST pipeline (and four
task-specific entry-points: T2TT, S2TT, T2ST, S2ST, ASR):

```
            +-------------------+        +-------------------+
  audio --> | Speech Encoder    |        | Text Encoder      | <-- text
            | (Conformer x24,   |        | (NLLB-style x24,  |
            |  W2v-BERT-2.0     |        |  Transformer enc) |
            |  style)           |        +---------+---------+
            +---------+---------+                  |
                      | + Conformer Adapter        |
                      +------------+---------------+
                                   |
                            +------v------+
                            | Text Decoder|   (token-level text output;
                            | (NLLB-style |    S2TT / T2TT / ASR stop here)
                            |  x24)       |
                            +------+------+
                                   | last_hidden_state + chars
                                   v
                        +----------+----------+
                        | T2U Encoder x6      |
                        +----------+----------+
                                   |
                        +----------v----------+
                        | T2U Decoder x6      |  conv1d-residual decoder layers
                        | + char-duration     |  + duration predictor
                        |   variance pred     |
                        +----------+----------+
                                   | unit ids
                                   v
                        +----------+----------+
                        | Code HiFi-GAN Vocoder|  unit + spkr + lang embed
                        | (HiFi-GAN with       |  + duration predictor
                        |  ConvTranspose1d     |  + ConvTranspose1d upsamplers
                        |  upsamplers +        |  + Residual conv blocks
                        |  resblocks)          |  -> 16 kHz waveform
                        +----------------------+
```

## Key dimensions (defaults from `SeamlessM4Tv2Config`)

| Parameter | Value |
| :--- | :--- |
| `hidden_size` (shared) | 1024 |
| `vocab_size` (text) | 256,102 |
| `t2u_vocab_size` (units) | 10,082 |
| `char_vocab_size` (T2U char input) | 10,943 |
| `unit_hifi_gan_vocab_size` | 10,000 |
| Speech encoder: layers / heads / ffn | 24 / 16 / 4096 (Conformer; act=swish) |
| Text encoder: layers / heads / ffn | 24 / 16 / 8192 (ReLU FFN) |
| Text decoder: layers / heads / ffn | 24 / 16 / 8192 |
| T2U encoder: layers / heads / ffn | 6 / 16 / 8192 |
| T2U decoder: layers / heads / ffn | 6 / 16 / 8192 (conv layers, no cross-attn) |
| Conformer convolution kernel | 31 (causal depthwise) |
| Adapter kernel / stride | 8 / 8 |
| Relative position embeddings | `relative_key`, left=64 / right=8 |
| Vocoder upsample rates / kernels | [5,4,4,2,2] / [11,8,8,4,4] |
| Vocoder resblock kernels / dilations | [3,7,11] / [[1,3,5]]*3 |
| `unit_embed_dim` / `lang_embed_dim` / `spkr_embed_dim` | 1280 / 256 / 256 |
| Vocoder languages / speakers | 36 / 200 |
| Sampling rate | 16 kHz |
| `layer_norm_eps` | 1e-5 |

## Detailed Block Inventory

### 1. Speech Encoder (Conformer, W2v-BERT-2.0 style)

Class: `SeamlessM4Tv2SpeechEncoder`. Composed of:

| Sub-block | HF Class | Notes |
| :--- | :--- | :--- |
| Feature projection | `SeamlessM4Tv2ConformerFeatureProjection` | LayerNorm(160) + Linear(160 -> 1024) |
| Conformer encoder x24 | `SeamlessM4Tv2ConformerEncoderLayer` | FFN macaron + MHA + Convolution module + FFN macaron + final LN |
| Conformer self-attn | `SeamlessM4Tv2ConformerSelfAttention` | MHA with `relative_key` positional bias (left=64, right=8), separate Q/K/V Linears (no fused QKV) |
| Conformer FFN | `SeamlessM4Tv2ConformerFeedForward` | Linear(1024->4096), Swish, Linear(4096->1024) |
| Conformer conv module | `SeamlessM4Tv2ConformerConvolutionModule` | LayerNorm -> pointwise Conv1d 1024->2048 -> GLU -> *causal* depthwise Conv1d k=31, groups=1024 -> LayerNorm -> Swish -> pointwise Conv1d 1024->1024 |
| Intermediate FFN | `SeamlessM4Tv2ConformerFeedForward` (`act=relu`) | half-add wrapper around encoder output |
| Adapter | `SeamlessM4Tv2ConformerAdapter` (1 layer) | LayerNorm + Conv1d(1024->2048, k=8, s=8) + GLU residual; LayerNorm + Conv1d(1024->2048, k=8, s=8) + GLU + self-attention (no relpos) + FFN |
| Final LN | `nn.LayerNorm(1024)` | `inner_layer_norm` |

The encoder also applies a *chunked attention mask* (`speech_encoder_chunk_size=20000`, left chunks=128) per-step — host-side mask materialization.

### 2. Text Encoder (NLLB-style)

Class: `SeamlessM4Tv2Encoder` with `is_t2u_encoder=False`. Pre-norm transformer:

| Sub-block | HF Class | Notes |
| :--- | :--- | :--- |
| Token embed | `SeamlessM4Tv2ScaledWordEmbedding` | shared with text decoder; scale by sqrt(1024) |
| Position embed | `SeamlessM4Tv2SinusoidalPositionalEmbedding` | sinusoidal, fixed (no params); built host-side and added |
| Encoder layer x24 | `SeamlessM4Tv2EncoderLayer` | LN -> MHA(`SeamlessM4Tv2Attention`) -> residual ; LN -> FFN -> residual |
| Attention | `SeamlessM4Tv2Attention` | Standard BART-style MHA: k_proj/v_proj/q_proj/out_proj, bias=True, no rotary |
| FFN | `SeamlessM4Tv2FeedForwardNetwork` | Linear(1024->8192), ReLU, Linear(8192->1024) |
| Final LN | `nn.LayerNorm(1024)` | over output |

### 3. Text Decoder (NLLB-style with cross-attention)

Class: `SeamlessM4Tv2Decoder`. Same scaled embedding + sinusoidal pos embed.

| Sub-block | HF Class | Notes |
| :--- | :--- | :--- |
| Decoder layer x24 | `SeamlessM4Tv2DecoderLayer` | LN -> causal MHA self-attn -> residual ; LN -> MHA cross-attn -> residual ; LN -> FFN -> residual |
| Self-attn | `SeamlessM4Tv2Attention(is_decoder=True)` | with KV-cache (`EncoderDecoderCache.self_attention_cache`) |
| Cross-attn | `SeamlessM4Tv2Attention(is_decoder=True)` | with KV-cache (`cross_attention_cache`); keys/values projected once from encoder output |
| FFN | `SeamlessM4Tv2FeedForwardNetwork` | Linear(1024->8192), ReLU, Linear(8192->1024) |
| LM head | `nn.Linear(1024, 256102, bias=False)` | tied to scaled embedding |

### 4. Text-to-Unit (T2U) Encoder

Class: `SeamlessM4Tv2Encoder` with `is_t2u_encoder=True` (6 layers, no token/pos embed — takes hidden states from the text decoder directly). Layer structure identical to text encoder.

### 5. Text-to-Unit (T2U) Decoder (non-autoregressive)

Class: `SeamlessM4Tv2TextToUnitDecoder`. Non-autoregressive decoder with a duration predictor and hard upsampling.

| Sub-block | HF Class | Notes |
| :--- | :--- | :--- |
| Token embed (unit) | `nn.Embedding(10082, 1024)` | scaled by sqrt(1024) |
| Character embed | `nn.Embedding(10943, 1024)` | character-level conditioning |
| Char-position embed | `SeamlessM4Tv2SinusoidalPositionalEmbedding` | with learnable scalar alpha (`pos_emb_alpha_char`) |
| Position embed | `SeamlessM4Tv2SinusoidalPositionalEmbedding` | post-upsample, with learnable scalar alpha (`pos_emb_alpha`) |
| Duration predictor | `SeamlessM4Tv2VariancePredictor` (embed=1024, hidden=256, k=3) | Conv1d -> ReLU -> LN -> Conv1d -> ReLU -> LN -> Linear(256, 1) |
| T2U decoder layer x6 | `SeamlessM4Tv2TextToUnitDecoderLayer` | causal MHA self-attn + LN, then Conv1d(k=7, padding=same) -> ReLU -> Conv1d(k=7, padding=same) + LN. **No cross-attention** (encoder context is fused via char-upsampling). |
| Final LN | `nn.LayerNorm(1024)` | |
| LM head | `nn.Linear(1024, 10082, bias=False)` | tied to unit embedding |

### 6. Code HiFi-GAN Vocoder (units -> waveform)

Class: `SeamlessM4Tv2CodeHifiGan`. Wraps a `SeamlessM4Tv2HifiGan` plus side embeddings.

| Sub-block | HF Class | Notes |
| :--- | :--- | :--- |
| Unit embedding | `nn.Embedding(10000, 1280)` | |
| Speaker embedding | `nn.Embedding(200, 256)` | host-side gather by speaker id |
| Language embedding | `nn.Embedding(36, 256)` | host-side gather by lang id |
| Duration predictor | `SeamlessM4Tv2VariancePredictor` (embed=1280, hidden=1280, k=3) | identical conv structure to T2U dur predictor but wider |
| Pre-conv | `nn.Conv1d(1792, 512, k=7, padding=3)` | (unit+lang+spkr concatenation) |
| Upsamplers x5 | `nn.ConvTranspose1d` | rates [5,4,4,2,2], kernels [11,8,8,4,4] |
| ResBlocks x5x3 | `HifiGanResidualBlock` | per-upsample MRF; 3 parallel kernels [3,7,11], dilations [[1,3,5]]*3 |
| Post-conv | `nn.Conv1d(channels, 1, k=7, padding=3)` | + tanh -> waveform |

Uses `leaky_relu(0.1)` throughout (no plain ReLU in the vocoder).

## Similar Implementations (Reference TTNN code in this repo)

| Component | Reference Implementation | Similarity |
| :--- | :--- | :--- |
| `SeamlessM4Tv2Attention` (encoder MHA) | `models/demos/bert/tt/ttnn_optimized_bert.py`, `models/demos/audio/whisper/tt/ttnn_optimized_functional_whisper.py` (`whisper_attention`, `encoder_layer`) | Vanilla MHA with bias, k/v/q/out_proj — Whisper is the closest match (same BART-derived layout, sinusoidal pos, scaled embed). |
| `SeamlessM4Tv2DecoderLayer` (text decoder w/ cross-attn + KV-cache) | `models/demos/audio/whisper/tt/ttnn_optimized_functional_whisper.py` (`decoder_layer`, KV cache + cross SDPA) | Whisper is essentially the same NLLB-style decoder layer; KV-cache shape and cross-attn precompute are identical. |
| `SeamlessM4Tv2FeedForwardNetwork` (ReLU FFN, dim 1024->8192) | `models/demos/bert/tt/`, `models/demos/falcon7b_common/tt/falcon_mlp.py` | Two-linear ReLU FFN (no gating). BERT is the closest. |
| `SeamlessM4Tv2ConformerFeedForward` (Swish FFN) | `models/demos/llama3_70b_galaxy/tt/llama_mlp.py` (Swish/SiLU op), `models/demos/audio/whisper/tt/` | Standard two-linear with Swish in the middle (NOT gated SwiGLU). |
| `nn.LayerNorm` (eps 1e-5, ubiquitous) | `models/demos/qwen3_vl/tt/vision_layernorm.py`, `models/demos/bert/tt/` | LayerNorm with both weight and bias. |
| Sinusoidal positional embedding | (Host-side precomputed buffer; no TTNN reference needed — register as host tensor and add.) | Implement as host-resident constant. Same approach as Whisper. |
| Scaled word embedding | `models/demos/audio/whisper/tt/` (embed + add positional) | `ttnn.embedding` + scalar multiply. |
| `SeamlessM4Tv2ConformerSelfAttention` (relative_key MHA) | `models/demos/bert/tt/ttnn_optimized_bert.py` (MHA base) + custom rel-pos bias gather | The relative-position bias term needs a custom op (gather `distance_embedding` then einsum), no direct TTNN reference. |
| `SeamlessM4Tv2ConformerConvolutionModule` (LN + pw conv + GLU + dw conv + LN + Swish + pw conv) | `models/demos/qwen3_tts/tt/ttnn_conv_decoder.py`, `models/demos/qwen3_tts/tt/ttnn_speech_decoder.py` | Conv1d + depthwise Conv1d are used in the qwen3_tts speech decoder and TTNN-conv-decoder paths; GLU = elementwise multiply of two halves. |
| `SeamlessM4Tv2ConformerEncoderLayer` (full Conformer block) | (Compose the four sub-blocks above; no exact reference — closest cousin is the Conformer-style block hinted at in `models/demos/qwen3_tts/tt/ttnn_speech_decoder.py` for ConvNeXt+attn stacks.) | Macaron FFN scaling (x0.5) is a peculiarity worth flagging in tests. |
| `SeamlessM4Tv2ConformerAdapter` (LN + Conv1d/stride 8 + GLU + attn + FFN) | `models/demos/qwen3_tts/tt/ttnn_conv_decoder.py` (strided Conv1d + GLU patterns) | Strided Conv1d collapses time x8; needs host-padded inputs or a fold-in. |
| `SeamlessM4Tv2VariancePredictor` (Conv1d -> LN -> Conv1d -> LN -> Linear) | `models/demos/qwen3_tts/tt/ttnn_conv_decoder.py` | Same conv1d+LN+linear pattern. |
| `SeamlessM4Tv2TextToUnitDecoderLayer` (causal MHA + Conv1d k=7 x2 + LN) | `models/demos/audio/whisper/tt/ttnn_optimized_functional_whisper.py` (decoder MHA) + `models/demos/qwen3_tts/tt/ttnn_conv_decoder.py` (Conv1d k=7) | Mix of decoder MHA and conv-decoder building blocks. |
| `SeamlessM4Tv2HifiGan` (ConvTranspose1d upsamplers + MRF ResBlocks) | `models/demos/qwen3_tts/tt/ttnn_speech_decoder.py`, `models/demos/qwen3_tts/tt/ttnn_conv_decoder.py` | qwen3_tts already implements HiFi-GAN-style decoders end-to-end on Blackhole. The closest match in the entire repo. |
| `SeamlessM4Tv2CodeHifiGan` wrapper (embed lookup + dur predictor + HiFi-GAN) | `models/demos/qwen3_tts/tt/ttnn_speech_decoder.py` (full vocoder wrapper), `models/demos/qwen3_tts/tt/code_predictor.py` (embedding/duration patterns) | Direct analogue: qwen3_tts vocoder takes RVQ tokens + speaker conditioning -> waveform. |

## Key Differences vs. References

- **Conformer relative_key attention** (`SeamlessM4Tv2ConformerSelfAttention`):
  bias term is `einsum("bhld,lrd->bhlr", query, positional_embedding) / sqrt(head_dim)`
  with `positional_embedding` produced by gathering `nn.Embedding(num_positions, head_size)`
  by `clamp(j-i, -L, R) + L`. No existing TTNN model in the repo implements
  this exact relative-key bias — needs custom code.
- **Chunked attention mask** in the Conformer encoder: built host-side per
  forward (chunk_size=20000, left chunks=128). Should be precomputed once
  for the canonical max sequence length used in bring-up.
- **Causal depthwise Conv1d** in the Conformer conv module: padding is
  `(kernel-1, 0)` on the left only. TTNN's standard `conv2d` wrappers must
  be configured for asymmetric/left padding.
- **GLU** along the channel dim after a 1x1 conv: implemented as
  `chunk-then-sigmoid-multiply`. No prebuilt TTNN GLU; use elementwise ops.
- **Sinusoidal positional embedding with `+ offset=2`**: precompute host-side
  to bf16 buffer. Same convention used by NLLB/M2M-100.
- **Two scalar-tensor learnable scales** in T2U decoder (`pos_emb_alpha`,
  `pos_emb_alpha_char`): just two single-element fp32 parameters; load as
  scalars and broadcast-multiply.
- **`SeamlessM4Tv2TextToUnitDecoder` has NO cross-attention** — it consumes
  the (hard-upsampled) char-encoded encoder hidden states as the residual
  stream itself. This is a NON-autoregressive decoder.
- **The text decoder's KV-cache** is an `EncoderDecoderCache` (separate self
  and cross caches), and cross-attn KV is computed only once. Same as
  Whisper.
- **`SeamlessM4Tv2CodeHifiGan` uses `leaky_relu(0.1)`** consistently; the
  qwen3_tts vocoder code uses `silu`/`gelu` in places — leaky_relu may need
  an explicit ttnn op.
- **Variance/Duration predictor** uses `nn.Conv1d(..., padding="same")`
  with odd kernel sizes — translate to `kernel_size // 2` explicit padding.
- **Speaker / language embedding lookup** is keyed by a single integer per
  sample — host-resident lookup table OK if device memory becomes tight.

## Weight Loading Notes

The HF state-dict groups weights by sub-model. Top-level prefixes for
`SeamlessM4Tv2Model` (the all-in-one) are:

- `shared.` — the shared text token embedding (used by text encoder, text
  decoder, and T2U encoder/decoder via tying).
- `speech_encoder.` — Conformer speech encoder (incl. `feature_projection`,
  `encoder.layers.{0..23}.…`, `intermediate_ffn`, `adapter.layers.0.…`,
  `inner_layer_norm`).
- `text_encoder.` — Text encoder (`embed_tokens` may be tied to `shared`,
  `embed_positions` is a buffer in code but no weight in state-dict for
  sinusoidal pos, `layers.{0..23}.…`, `layer_norm`).
- `text_decoder.` — Text decoder (`embed_tokens`, `layers.{0..23}.…`,
  `layer_norm`).
- `lm_head.weight` — text decoder LM head (tied to text embedding).
- `t2u_model.encoder.…` — T2U encoder (no `embed_tokens`).
- `t2u_model.decoder.…` — T2U decoder (incl. `embed_tokens` (unit),
  `embed_char`, `pos_emb_alpha`, `pos_emb_alpha_char`, `duration_predictor.…`,
  `layers.{0..5}.…`, `layer_norm`).
- `t2u_model.lm_head.weight` — unit LM head.
- `vocoder.dur_predictor.…` — vocoder duration predictor.
- `vocoder.unit_embedding.weight`, `vocoder.speaker_embedding.weight`,
  `vocoder.language_embedding.weight`.
- `vocoder.hifi_gan.conv_pre.{weight,bias}`, `vocoder.hifi_gan.upsampler.{0..4}.…`,
  `vocoder.hifi_gan.resblocks.{0..14}.…`, `vocoder.hifi_gan.conv_post.…`.

All Linear layers have a `bias` (unlike Llama-style). `Conv1d` weights are
3D `(out_channels, in_channels/groups, kernel_size)`. The HiFi-GAN weights
are stored in `weight_norm`-applied form on the original checkpoint;
HF stores the materialized weight directly (handled by `transformers`).

Suggested HF -> TTNN key mapping pattern (preserve the HF names — there is
no advantage to renaming for this model):

```text
text_encoder.layers.{i}.self_attn.{q,k,v,out}_proj.{weight,bias}
text_encoder.layers.{i}.self_attn_layer_norm.{weight,bias}
text_encoder.layers.{i}.ffn.fc1.{weight,bias}
text_encoder.layers.{i}.ffn.fc2.{weight,bias}
text_encoder.layers.{i}.ffn_layer_norm.{weight,bias}

text_decoder.layers.{i}.self_attn.{q,k,v,out}_proj.{weight,bias}
text_decoder.layers.{i}.self_attn_layer_norm.{weight,bias}
text_decoder.layers.{i}.cross_attention.{q,k,v,out}_proj.{weight,bias}
text_decoder.layers.{i}.cross_attention_layer_norm.{weight,bias}
text_decoder.layers.{i}.ffn.fc1.{weight,bias}
text_decoder.layers.{i}.ffn.fc2.{weight,bias}
text_decoder.layers.{i}.ffn_layer_norm.{weight,bias}

speech_encoder.encoder.layers.{i}.ffn1.intermediate_dense.{weight,bias}
speech_encoder.encoder.layers.{i}.ffn1.output_dense.{weight,bias}
speech_encoder.encoder.layers.{i}.ffn1_layer_norm.{weight,bias}
speech_encoder.encoder.layers.{i}.self_attn.linear_{q,k,v,out}.{weight,bias}
speech_encoder.encoder.layers.{i}.self_attn.distance_embedding.weight
speech_encoder.encoder.layers.{i}.self_attn_layer_norm.{weight,bias}
speech_encoder.encoder.layers.{i}.conv_module.pointwise_conv1.weight
speech_encoder.encoder.layers.{i}.conv_module.depthwise_conv.weight
speech_encoder.encoder.layers.{i}.conv_module.pointwise_conv2.weight
speech_encoder.encoder.layers.{i}.conv_module.layer_norm.{weight,bias}
speech_encoder.encoder.layers.{i}.conv_module.depthwise_layer_norm.{weight,bias}
speech_encoder.encoder.layers.{i}.ffn2.intermediate_dense.{weight,bias}
speech_encoder.encoder.layers.{i}.ffn2.output_dense.{weight,bias}
speech_encoder.encoder.layers.{i}.ffn2_layer_norm.{weight,bias}
speech_encoder.encoder.layers.{i}.final_layer_norm.{weight,bias}
```

## Host-Resident Exceptions

None are *required*. Two candidates if device memory becomes tight at
bring-up time:

- Sinusoidal positional embedding buffers (text encoder/decoder, T2U) —
  these are statically computed, never trained, and small (4098 x 1024 fp32 -> ~16 MB).
  Cheapest to keep host-side and only transfer per forward, but they are
  small enough to live in DRAM permanently. **No exception requested.**
- Speaker/language embedding tables in `vocoder` (200x256 + 36x256 = ~256 KB)
  — trivially small; keep on device.

## Implementation Order (suggested, leaves first)

Each bullet should be implementable in roughly one focused TTNN
implementation session (one block_hash entry). The orchestrator can fan
out the *reference* phase across these in parallel:

1. **layernorm** — shared LayerNorm helper (eps 1e-5; weight+bias).
2. **scaled_word_embedding** — embedding lookup + scalar multiply.
3. **sinusoidal_positional_embedding** — host-side precomputed buffer add.
4. **seamless_mha** — `SeamlessM4Tv2Attention` core (BART-style MHA w/ bias).
5. **seamless_ffn** — `SeamlessM4Tv2FeedForwardNetwork` (Linear-ReLU-Linear).
6. **conformer_ffn** — `SeamlessM4Tv2ConformerFeedForward` (Linear-Swish-Linear).
7. **conformer_self_attention** — Conformer MHA with relative_key bias.
8. **conformer_convolution_module** — LN + pw conv + GLU + dw causal conv + LN + Swish + pw conv.
9. **variance_predictor** — Conv1d->ReLU->LN->Conv1d->ReLU->LN->Linear.
10. **hifigan_residual_block** — three-dilation conv pair w/ leaky_relu.
11. **conformer_feature_projection** — LN(160) + Linear(160->1024).
12. **conformer_encoder_layer** — composes (6)+(7)+(8)+(6) macaron block.
13. **text_encoder_layer** — composes (4)+(5) NLLB-style encoder layer.
14. **text_decoder_layer** — composes (4)+(4 cross)+(5) NLLB-style decoder layer w/ KV-cache.
15. **t2u_decoder_layer** — composes (4) + two Conv1d(k=7) + LN.
16. **conformer_adapter_layer** — strided conv + GLU + attn + FFN.
17. **speech_encoder** — full stack: feature_projection -> 24 conformer layers -> intermediate_ffn -> adapter -> inner_layer_norm.
18. **text_encoder** — embed + sinusoidal pos + 24 layers + LN.
19. **text_decoder** — embed + sinusoidal pos + 24 layers + LN + LM head; supports KV-cache and `EncoderDecoderCache` semantics.
20. **t2u_encoder** — 6 encoder layers (no embed) + LN.
21. **t2u_decoder** — unit/char embeddings + sinusoidal pos + duration predictor + hard-upsample + 6 t2u-decoder layers + LN + LM head.
22. **hifigan_vocoder** — `conv_pre` + 5 ConvTranspose1d upsamplers + 15 resblocks + `conv_post` + tanh.
23. **code_hifigan_vocoder** — unit_embedding + speaker_embedding + language_embedding + dur_predictor + `hifigan_vocoder`.
24. **seamless_m4t_v2** — top-level glue: speech_encoder | text_encoder -> text_decoder -> t2u_encoder -> t2u_decoder -> code_hifigan_vocoder + sub-pipelines for {T2TT, S2TT, T2ST, S2ST, ASR}.

## Summary

SeamlessM4T-v2-Large is **five neural networks in one checkpoint**: a
Conformer speech encoder, an NLLB-style text encoder, an NLLB-style text
decoder (KV-cached, autoregressive), a non-autoregressive T2U
encoder-decoder with a duration predictor, and a Code HiFi-GAN vocoder
that converts unit ids to a 16 kHz waveform. The closest TTNN references
already in this repo are:

- **Whisper** (`models/demos/audio/whisper/`) — for the BART-style MHA,
  cross-attention, KV-cache, sinusoidal-positional encoder/decoder layout.
- **BERT** (`models/demos/bert/`) — for the canonical MHA-with-bias and
  ReLU FFN building blocks.
- **qwen3_tts** (`models/demos/qwen3_tts/`) — for the HiFi-GAN-style conv
  decoder (ConvTranspose1d upsamplers + multi-receptive-field ResBlocks)
  and Conv1d-based decoder/variance-predictor patterns.
- **llama3_70b_galaxy** (`models/demos/llama3_70b_galaxy/`) — for general
  Swish/SiLU FFN op patterns (the Conformer feed-forward uses Swish but is
  not gated).

The only block with **no direct TTNN reference** is the Conformer
**relative_key** positional-bias term — that needs a custom embedding
gather + einsum implemented from scratch in TTNN ops, similar in spirit
to (but smaller than) RoPE.
