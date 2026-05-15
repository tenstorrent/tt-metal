# LTX-2 to TT_DIT Porting Plan

## Architecture Summary

LTX-2 (video-only) is structurally very similar to Wan 2.2. Both use an AdaLN-modulated DiT with self-attention (RMSNorm Q/K + RoPE), text cross-attention, and gated feed-forward. The key differences are:

- **Dimensions**: LTX uses 4096 dim (32 heads x 128), 48 layers, 128-channel latents vs Wan's 5120 dim (40 heads x 128), 40 layers, 16-channel latents
- **Normalization**: LTX uses `rms_norm` before attention blocks (not LayerNorm)
- **RoPE**: LTX uses fractional position encoding with custom frequency grids and interleaved/split modes
- **Patch embedding**: LTX uses a simple Linear(128, 4096) instead of Wan's 3D conv + linear
- **Timestep conditioning**: LTX uses `PixArtAlphaCombinedTimestepSizeEmbeddings` with `AdaLayerNormSingle`
- **Text encoder**: Gemma (not UMT5)
- **Optional features**: Per-head gating, cross-attention AdaLN (22B), perturbation masking (STG)

## Reusable Components from Wan

Nearly all of Wan's TT infrastructure can be reused:

- `DistributedRMSNorm` with fused RoPE ([layers/normalization.py](models/tt_dit/layers/normalization.py))
- `DistributedLayerNorm` for output norm
- `ColParallelLinear` / `RowParallelLinear` ([layers/linear.py](models/tt_dit/layers/linear.py))
- `ParallelFeedForward` ([layers/feedforward.py](models/tt_dit/layers/feedforward.py))
- Fused QKV/KV attention pattern from `WanAttention` ([attention_wan.py](models/tt_dit/models/transformers/wan2_2/attention_wan.py))
- SDPA / ring attention infrastructure
- `Module`, `Parameter`, `ModuleList` base classes
- `Timesteps`, `TimestepEmbedding`, `PixArtAlphaTextProjection` from [embeddings.py](models/tt_dit/layers/embeddings.py)
- Pipeline patterns, weight caching, TP/SP/FSDP infrastructure

## Testing Strategy

All tests follow the pattern established by Wan tests in [tests/models/wan2_2/](models/tt_dit/tests/models/wan2_2/). Each test:

1. Loads the PyTorch reference model (from LTX checkpoint or random init)
2. Creates the corresponding TT model and loads weights via `load_torch_state_dict()`
3. Creates matching random inputs, runs both models
4. Compares outputs using `assert_quality(torch_out, tt_out, pcc=MIN_PCC, relative_rmse=MAX_RMSE)`
5. Parametrizes across mesh configurations: `(2,2)`, `(2,4)`, `(4,8)` with various SP/TP axis assignments

Test files live in `models/tt_dit/tests/models/ltx/`.

## Phase 1: LTX Video-Only DiT Transformer

### 1.1 Directory structure

Create `models/tt_dit/models/transformers/ltx/` with `__init__.py`, `attention_ltx.py`, `transformer_ltx.py`, `rope_ltx.py`.
Create `models/tt_dit/tests/models/ltx/` for all test files.

### 1.2 LTX RoPE (`rope_ltx.py`)

LTX RoPE differs from Wan: fractional positions (`indices_grid / max_pos`), custom freq grid (`theta^linspace * pi/2`), interleaved vs split modes. Reference implementation: [LTX-2 rope.py](LTX-2/packages/ltx-core/src/ltx_core/model/transformer/rope.py). Compute cos/sin on CPU, convert to ttnn (same as Wan's `TorchWanRotaryPosEmbed` fallback pattern in [transformer_wan.py](models/tt_dit/models/transformers/wan2_2/transformer_wan.py) lines 365-411).

**Unit Test** (`test_rope_ltx.py`): Following [test_rope.py](models/tt_dit/tests/models/wan2_2/test_rope.py) pattern:

- Instantiate LTX `precompute_freqs_cis` on CPU with sample `indices_grid` and `max_pos=[20,2048,2048]`
- Apply `apply_rotary_emb` (PyTorch reference) to random input tensor
- Convert cos/sin to ttnn, apply `ttnn.experimental.rotary_embedding_llama` with transformation matrix
- Compare output, **PCC >= 0.99**
- Test both `INTERLEAVED` and `SPLIT` rope modes
- Parametrize across mesh configs

### 1.3 Timestep Conditioning

Add `LTXAdaLayerNormSingle` to [embeddings.py](models/tt_dit/layers/embeddings.py):

- Reuses existing `Timesteps` (sinusoidal) and `TimestepEmbedding` (Linear -> SiLU -> Linear)
- Adds: SiLU -> Linear(dim, 6*dim) to produce per-block modulation params
- `timestep_scale_multiplier = 1000`

**Unit Test** (`test_embeddings_ltx.py`):

- Load PyTorch `AdaLayerNormSingle` weights into TT `LTXAdaLayerNormSingle`
- Feed same timestep tensor to both
- Compare output (modulation params) and `embedded_timestep`, **PCC >= 0.999**

### 1.4 Patch Embedding

Simple `Linear(128, 4096)` with bias. Use existing `Linear` from [linear.py](models/tt_dit/layers/linear.py).

**Unit Test** (`test_embeddings_ltx.py`):

- Load PyTorch `patchify_proj` weights into TT Linear
- Feed random latent input `(B, N, 128)`, compare output, **PCC >= 0.999**

### 1.5 LTX Attention (`attention_ltx.py`)

Based heavily on [attention_wan.py](models/tt_dit/models/transformers/wan2_2/attention_wan.py):

- Fused QKV for self-attn, fused KV for cross-attn (same `ColParallelLinear` with `chunks` pattern)
- `DistributedRMSNorm` on Q/K with RoPE (same fused kernel)
- SDPA / ring attention (same infrastructure)
- **New**: Optional per-head gating via `to_gate_logits` Linear + `2 * sigmoid(gate)` scaling
- **New**: Perturbation masking support (blend attn output with raw value projection)
- Dimensions: 32 heads x 128 head_dim = 4096

**Unit Test** (`test_attention_ltx.py`): Following [test_attention_wan.py](models/tt_dit/tests/models/wan2_2/test_attention_wan.py) pattern:

- Load LTX PyTorch `Attention` weights (from checkpoint or random init) into TT `LTXAttention`
- Test self-attention: random spatial input + RoPE cos/sin, compare output, **PCC >= 0.988**
- Test cross-attention: random spatial + prompt input (no RoPE), compare output, **PCC >= 0.988**
- Parametrize: mesh configs `(2,4)`, `(4,8)` with SP/TP axes; sequence lengths `(short, medium, long)`; prompt lengths `(None, 26, 126)`

### 1.6 LTX Transformer Block (`transformer_ltx.py`)

Based on `WanTransformerBlock` ([transformer_wan.py](models/tt_dit/models/transformers/wan2_2/transformer_wan.py) lines 29-224):

- `scale_shift_table` (6, dim) + timestep -> 6 chunks: shift/scale/gate for self-attn and FF
- `rms_norm` + AdaLN (scale/shift) -> self-attn + gate residual (fused addcmul)
- `rms_norm` -> cross-attn + residual
- `rms_norm` + AdaLN (scale/shift) -> FF + gate residual (addcmul)
- **Key difference**: Uses `rms_norm` for pre-attention norms (Wan uses `DistributedLayerNorm`). Use `DistributedRMSNorm` here.
- **22B variant**: Add optional cross-attention AdaLN (3 extra params: shift_q, scale_q, gate)

**Unit Test** (`test_transformer_ltx.py`): Following [test_transformer_wan.py](models/tt_dit/tests/models/wan2_2/test_transformer_wan.py) `test_wan_transformer_block` pattern:

- Load LTX PyTorch `BasicAVTransformerBlock` (video config only) weights into TT `LTXTransformerBlock`
- Create random spatial `(B, N, 4096)`, prompt `(B, L, 4096)`, temb `(B, 6, 4096)`, RoPE cos/sin inputs
- Run both models, compare spatial output, **PCC >= 0.999_500, RMSE <= 0.032**
- Parametrize: mesh configs, sequence lengths `(short, 480p, 720p)`

### 1.7 Main Model (`LTXTransformerModel`) + Weight Loading

Based on `WanTransformer3DModel` ([transformer_wan.py](models/tt_dit/models/transformers/wan2_2/transformer_wan.py) lines 227-638):

- `patchify_proj`, `adaln_single`, optional `caption_projection`, 48 blocks, `scale_shift_table(2, dim)`, `norm_out`, `proj_out`
- `forward()`, `inner_step()`, `prepare_conditioning()`, `prepare_rope_features()`
- Pre/post-processing for spatial input (patchify -> pad for SP -> device, and reverse)

`_prepare_torch_state()` methods for each module:

- Fuse Q/K/V -> QKV with head interleaving (same pattern as [attention_wan.py](models/tt_dit/models/transformers/wan2_2/attention_wan.py) lines 166-211)
- Rename `to_out.0` -> `to_out`, `ffn.net.0.proj` -> `ffn.ff1`, `ffn.net.2` -> `ffn.ff2`
- Unsqueeze `scale_shift_table`

**Unit Test - Full Model** (`test_transformer_ltx.py`): Following `test_wan_transformer_model` pattern:

- Load LTX PyTorch `LTXModel` (VideoOnly, truncated to 1 layer for fast test + full model for accuracy test)
- Create random video latent `(B, 128, F, H, W)`, timestep, context inputs
- Call `tt_model.forward()`, compare output against PyTorch `LTXModel.forward()`, **PCC >= 0.992, RMSE <= 0.15**
- Parametrize: mesh configs, video shapes

**Unit Test - Inner Step** (`test_transformer_ltx.py`): Following `test_wan_transformer_inner_step` pattern:

- Load weights, pre-cache prompt/RoPE on device
- Call `tt_model.inner_step()` with patchified spatial input (torch tensor) and cached device tensors
- Compare against PyTorch full forward, **PCC >= 0.992, RMSE <= 0.15**
- Validates the denoising loop path used by the pipeline

## Phase 2: Gemma Text Encoder

Create `models/tt_dit/encoders/gemma/` with `model_gemma.py` and `encoder_pair.py`. Can leverage existing Gemma code from `models/experimental/pi0/tt/ttnn_gemma.py` (has `GemmaAttentionTTNN`, `GemmaMLPTTNN`, `GemmaBlockTTNN`) or `models/demos/multimodal/gemma3/`. Follow the `*TokenizerEncoderPair` pattern from [UMT5](models/tt_dit/encoders/umt5/).

**Unit Test** (`test_gemma_encoder.py`):

- Load HuggingFace Gemma model weights into TT `GemmaEncoder`
- Tokenize sample prompts, run both models
- Compare output embeddings, **PCC >= 0.99**
- Test `GemmaTokenizerEncoderPair.encode()` end-to-end

## Phase 3: LTX Pipeline

Create `models/tt_dit/pipelines/ltx/pipeline_ltx.py` based on [pipeline_wan.py](models/tt_dit/pipelines/wan/pipeline_wan.py):

- Load Gemma encoder, LTX transformer, Video VAE
- `LTX2Scheduler` (sigma schedule with token-dependent shifting) - pure math, runs on CPU
- `EulerDiffusionStep` (velocity-based Euler) - runs on CPU
- CFG with `ttnn.lerp`, caching of prompt/RoPE/positions

**Unit Test** (`test_pipeline_ltx.py`): Following [test_pipeline_wan.py](models/tt_dit/tests/models/wan2_2/test_pipeline_wan.py) pattern:

- Run full end-to-end text-to-video generation with a short prompt and small resolution
- Compare denoised latent output against PyTorch reference pipeline on same seed/inputs
- Validate output shape and value ranges are correct
- Parametrize across mesh configs `(2,2)`, `(2,4)`, `(4,8)`

## Phase 4: LTX Video VAE

128-channel latent VAE with 8x temporal / 32x spatial compression and PixelNorm. Significantly different from Wan VAE - plan separately once DiT works.

**Unit Test** (`test_vae_ltx.py`): Following [test_vae_wan2_1.py](models/tt_dit/tests/models/wan2_2/test_vae_wan2_1.py) pattern:

- Test encoder: random video input -> TT VideoEncoder vs PyTorch VideoEncoder, compare latents, **PCC >= 0.99**
- Test decoder: random latent input -> TT VideoDecoder vs PyTorch VideoDecoder, compare output, **PCC >= 0.99**
- Test round-trip: encode -> decode, verify reconstruction quality

## Phase 5: Audio-Video Extension

Add audio path: audio self/cross-attention blocks, bidirectional audio-video cross-attention with separate AdaLN, 1D audio RoPE, audio VAE.

**Unit Test** (`test_audio_ltx.py`):

- Test audio self-attention and cross-attention individually vs PyTorch reference
- Test bidirectional audio-video cross-attention: feed video + audio hidden states, compare both output streams
- Test full AudioVideo model (`LTXModelType.AudioVideo`) forward pass vs PyTorch reference
- Test audio VAE encode/decode independently

## New Files

**Model implementation:**

- `models/tt_dit/models/transformers/ltx/__init__.py`
- `models/tt_dit/models/transformers/ltx/attention_ltx.py`
- `models/tt_dit/models/transformers/ltx/transformer_ltx.py`
- `models/tt_dit/models/transformers/ltx/rope_ltx.py`
- `models/tt_dit/encoders/gemma/model_gemma.py`
- `models/tt_dit/encoders/gemma/encoder_pair.py`
- `models/tt_dit/pipelines/ltx/pipeline_ltx.py`

**Tests (one per implementation step, verifying against PyTorch reference):**

- `models/tt_dit/tests/models/ltx/__init__.py`
- `models/tt_dit/tests/models/ltx/test_rope_ltx.py` - RoPE correctness (after step 1.2)
- `models/tt_dit/tests/models/ltx/test_embeddings_ltx.py` - Timestep conditioning + patch embedding (after steps 1.3-1.4)
- `models/tt_dit/tests/models/ltx/test_attention_ltx.py` - Self/cross attention (after step 1.5)
- `models/tt_dit/tests/models/ltx/test_transformer_ltx.py` - Block, full model, inner_step (after steps 1.6-1.7)
- `models/tt_dit/tests/models/ltx/test_gemma_encoder.py` - Gemma text encoder (after Phase 2)
- `models/tt_dit/tests/models/ltx/test_pipeline_ltx.py` - End-to-end pipeline (after Phase 3)
- `models/tt_dit/tests/models/ltx/test_vae_ltx.py` - Video VAE encode/decode (after Phase 4)
- `models/tt_dit/tests/models/ltx/test_audio_ltx.py` - Audio extension (after Phase 5)

## Modified Files

- `models/tt_dit/layers/embeddings.py` - Add `LTXAdaLayerNormSingle`, `LTXPatchEmbed`
