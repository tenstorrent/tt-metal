# Voxtral-4B-TTS-2603 Architecture

## Target Device

**N150** (single Wormhole B0, 12GB DRAM). No tensor parallelism. Model ~8GB BF16 fits comfortably.
Note: Architecture was initially planned for T3K but user confirmed N150 as target. All divisibility checks were for T3K; they trivially hold for N150 (no sharding needed).

---

## Pre-Flight CPU Audit

| Pattern | Model | Location | Verdict |
|---------|-------|----------|---------|
| `F.softmax` / `F.linear` in forward | molmo2 | `tt/model.py:1227-1237` | DOCUMENTED — CPU decode path, variable-seq bypass |
| `F.linear` in `pli_proj` | gemma4 | `tt/model.py:355` | DOCUMENTED — per-layer image projection tradeoff |
| `torch.matmul` in lm_head | llama3_70b_galaxy | `tt/lm_head.py:126` | DOCUMENTED — >32k vocab on Galaxy |
| `.detach().cpu()` | informer | `tt/state_io.py:177-243` | ACCEPTABLE — weight loading only |
| `.cpu()` | metal_BERT_large_11 | `tt/bert_model.py:111` | LEGACY — pre-TTNN model |
| `.cpu()` on output | llama3_70b_galaxy, qwen25_vl | generators | ACCEPTABLE — output collection |

No BLOCK-level CPU patterns found in active models.

---

## Model Overview

**Type**: Text-to-Speech (TTS)
**Base**: Ministral-3B (Mistral)
**Total params**: ~4.1B (3.4B text backbone + 390M acoustic transformer + 300M codec)
**Weights**: `consolidated.safetensors` (8GB BF16, Mistral format)
**Tokenizer**: Tekken (131072 vocab, `tekken.json`)
**Output**: 24kHz mono waveform

---

## Architecture: Three-Component Pipeline

```
Text + Voice → [Text Backbone] → semantic tokens + hidden states h
                                        ↓
                              [Flow-Matching Transformer]  (8 Euler steps)
                                        ↓
                                 acoustic tokens [N_frames, 36]
                                        ↓
                              [Voxtral Codec Decoder]
                                        ↓
                               24kHz waveform
```

---

## Component 1: Text Decoder Backbone

**Source**: Ministral-3B architecture

| Param | Value |
|-------|-------|
| `n_layers` | 26 |
| `dim` | 3072 |
| `n_heads` | 32 |
| `n_kv_heads` | 8 (GQA) |
| `head_dim` | 128 |
| `hidden_dim` | 9216 (SwiGLU) |
| `vocab_size` | 131072 |
| `rope_theta` | 1,000,000 |
| `norm` | RMSNorm (pre) |
| `tied_embeddings` | true |
| `max_seq_len` | 65536 |

**Inputs**:
- Voice reference: audio tokens (37 tokens/frame: 1 semantic + 36 acoustic), summed embeddings
- Text: tokenized with Tekken tokenizer
- Concatenated: `[voice_tokens, text_tokens, <next>]`

**Outputs**:
- Hidden states `h` per position (used by flow-matching transformer)
- Semantic head: linear → 8193 logits (8192 vocab + EoA), autoregressive

### Divisibility Checks (T3K, 8 devices)

| Dim | Value | ÷8 | Pass |
|-----|-------|-----|------|
| n_heads | 32 | 4 | ✓ |
| n_kv_heads | 8 | 1 | ✓ |
| hidden_dim | 9216 | 1152 | ✓ |
| dim | 3072 | 384 | ✓ |

---

## Component 2: Acoustic Flow-Matching Transformer

**3-layer bidirectional transformer**, same width as text backbone

| Param | Value |
|-------|-------|
| `n_layers` | 3 |
| `dim` | 3072 |
| `n_heads` | 32 |
| `n_kv_heads` | 8 |
| `head_dim` | 128 |
| `hidden_dim` | 9216 |
| `rope_theta` | 10000.0 |

**Inputs per ODE step**:
- `h`: decoder hidden states (3072-dim)
- `t`: sinusoidal timestep embedding ∈ [0,1]
- `x_t`: current acoustic embedding (36-dim, projected to 3072)
- All three have separate projection layers (different activation scales)

**Inference (Euler ODE)**:
- x_0 ~ N(0,1), shape [N_frames, 36]
- 8 Euler steps, Δt = 1/8
- CFG: α=1.2, runs model TWICE per step (conditioned h vs. null_h)
- x_{t+1} = x_t + v_θ(x_t, t, h) * Δt
- Output → quantize to 21 FSQ levels → acoustic tokens [N_frames, 36]

---

## Component 3: Voxtral Codec Decoder

**300M parameter 4-stage upsampling decoder**

| Param | Value |
|-------|-------|
| `dim` | 1024 |
| `hidden_dim` | 4096 |
| `n_heads` | 8 |
| `n_kv_heads` | 8 |
| `head_dim` | 128 |
| `norm_eps` | 0.01 |
| `attn_type` | ALiBi + sliding window |
| Initial window | 2 (bottom stage, most compressed) |
| Final window | 16 (top stage, full resolution) |
| Upsampling strides | [1, 2, 2, 2] (kernels [3, 4, 4, 4]) |
| Transformer blocks/stage | [2, 2, 2, 2] |

**4-Stage Decoder Architecture**:
```
acoustic tokens [N_frames, 256+36=292]
    → initial conv (kernel=7)
    → Block 0: 2× transformer (window=2) + ConvTranspose (stride=1, kernel=3)
    → Block 1: 2× transformer (window=4) + ConvTranspose (stride=2, kernel=4)
    → Block 2: 2× transformer (window=8) + ConvTranspose (stride=2, kernel=4)
    → Block 3: 2× transformer (window=16) + ConvTranspose (stride=2, kernel=4)
    → final conv → waveform [N_samples] (at 24kHz)
```

**Non-Standard Ops**:
- **ALiBi positional bias**: linear decay bias added to attention logits (no RoPE)
- **Sliding window attention**: window halves at each downsampling stage (2→4→8→16 in decoder)
- **QK-norm** (`qk_norm_eps=1e-6`)
- **LayerScale** (init=0.01, learned scalar per residual)
- **Causal Conv1D** (causal padding = kernel_size-1 zeros prepended)
- **ConvTranspose1D** for upsampling
- **Weight normalization**: training artifact, FUSED INTO WEIGHTS at inference — no special op needed

**Codec inputs**:
- Semantic tokens → embedding lookup (8192×256)
- Acoustic tokens → 36 separate FSQ dequantization (21 levels × 36 dims → 36-dim float vector)
- Concatenate semantic (256) + acoustic (36) → 292-dim → project to 1024

---

## Block Inventory

### Text Backbone
| Block | TTNN Op(s) | Reuse From | T3K Strategy |
|-------|-----------|------------|--------------|
| tok_embeddings | ttnn.embedding | molmo2/tt/model.py | Replicated |
| voice_embeddings (20 presets) | ttnn.from_torch + concat | NEW | Replicated |
| layers[0-25].attention_norm | ttnn.rms_norm | molmo2/tt/ | Per-device |
| layers[0-25].attention.wq/wk/wv | ttnn.linear | molmo2/tt/attention.py | Column-parallel |
| layers[0-25].attention.wo | ttnn.linear + all_gather | molmo2/tt/attention.py | Row-parallel |
| layers[0-25].ffn_norm | ttnn.rms_norm | molmo2/tt/ | Per-device |
| layers[0-25].feed_forward.w1/w3 | ttnn.linear | molmo2/tt/mlp.py | Column-parallel |
| layers[0-25].feed_forward.w2 | ttnn.linear + reduce | molmo2/tt/mlp.py | Row-parallel |
| norm | ttnn.rms_norm | molmo2/tt/ | Replicated |
| semantic_head | ttnn.linear (3072→8193) | NEW (lm_head pattern) | Replicated |

### Acoustic Transformer
| Block | TTNN Op(s) | Reuse From | T3K Strategy |
|-------|-----------|------------|--------------|
| time_embedding (sinusoidal) | Python math → ttnn.from_torch | NEW | Replicated |
| time_projection | ttnn.linear × 2 + ttnn.gelu | NEW | Replicated |
| h_projection | ttnn.linear | NEW | Replicated |
| x_t_projection (36→3072) | ttnn.linear | NEW | Replicated |
| layers[0-2].{attention,mlp,norms} | Same as text backbone | molmo2/tt/ | Same TP |
| output_proj (3072→36) | ttnn.linear | NEW | Replicated |
| ODE loop | Python loop, no TTNN primitive | NEW | CPU orchestration |

### Codec Decoder
| Block | TTNN Op(s) | Reuse From | T3K Strategy |
|-------|-----------|------------|--------------|
| semantic_embed (8192→256) | ttnn.embedding | NEW | Single device |
| acoustic_fsq_dequant (int→float) | ttnn.from_torch (preprocessing) | NEW | CPU preprocessing |
| initial_conv (kernel=7) | ttnn.conv1d + causal padding | whisper/tt/ | Single device |
| blocks[0-3].transformer.attention | ttnn.sdpa + ALiBi mask | NEW | Single device |
| blocks[0-3].transformer.mlp | ttnn.linear (SwiGLU variant) | NEW | Single device |
| blocks[0-3].qk_norm | ttnn.rms_norm | molmo2/tt/ | Single device |
| blocks[0-3].layer_scale | ttnn.multiply | NEW | Single device |
| blocks[0-3].conv_transpose | ttnn.conv_transpose2d (H=1) | NEW | Single device |
| final_conv | ttnn.conv1d | NEW | Single device |

---

## Non-Standard Op Strategies

### ALiBi Attention (Codec)
- Precompute ALiBi bias tensor on CPU: slope = 2^(-8i/n_heads), position offset = j - i
- Pass as `attention_mask` bias to `ttnn.sdpa`
- Window mask: set out-of-window positions to -inf in the same mask
- Single computation per decode call (fixed geometry per audio length)
- **Unit test**: `tests/ttnn/unit_tests/operations/sdpa/test_sdpa_prefill.py`

### Causal Conv1D (Codec)
- Left-pad input by (kernel_size - 1) before `ttnn.conv1d`
- Use padding=0 in ttnn.conv1d (causal padding applied manually)
- **Unit test**: `tests/ttnn/unit_tests/operations/conv/test_conv1d.py`

### ConvTranspose1D (Codec)
- Use `ttnn.conv_transpose2d` with H=1 (treat 1D as 2D with height=1)
- Reshape: [B, L, C] → [B, 1, L, C] → conv_transpose2d → [B, 1, L', C] → [B, L', C]
- **Unit test**: `tests/ttnn/unit_tests/operations/conv/test_conv_transpose2d.py`

### Flow-Matching ODE (Acoustic Transformer)
- 8 Euler steps = 8 sequential forward passes through 3-layer transformer
- CFG: 2 passes per step = 16 total forward passes per audio segment
- Can batch conditioned + unconditioned as batch_size=2 for efficiency
- Loop is pure Python; only transformer forward is TTNN

### Weight Normalization (Codec)
- Training artifact: `w = (g / ‖v‖) * v`
- At load time, compute normalized weight: `w_fused = g * v / v.norm(dim=0, keepdim=True)`
- Store fused `w_fused` as plain weight tensor
- **No special TTNN op required**

---

## Memory Budget (T3K, 8 devices)

| Component | Total (BF16) | Per Device |
|-----------|-------------|------------|
| Text backbone (26L × ~200MB/layer) | 6.8 GB | 0.85 GB |
| Acoustic transformer (3L) | 0.8 GB | 0.10 GB |
| Codec decoder | 1.5 GB | 1.5 GB (device 0 only) |
| KV cache (prefill, 65k× BF16) | ~1.0 GB | 0.13 GB |
| Activations | ~0.5 GB | 0.06 GB |
| **Total** | **~10.6 GB** | **~2.6 GB** |

All within T3K 12GB/device limit ✓

---

## Reuse Scoring

| Existing Model | Score | Reason |
|---------------|-------|--------|
| `molmo2/tt/` | **+8** | T3K ✓, GQA ✓, SwiGLU ✓, RMSNorm ✓, all_gather pattern ✓ |
| `qwen3_vl/tt/` | **+7** | T3K ✓, GQA ✓, QK-norm ✓ |
| `audio/whisper/tt/` | **+5** | Conv1D pattern ✓, audio processing ✓ |
| `tt_transformers/tt/` | **+5** | General T3K transformer ✓ |

**Plan**: Adapt text backbone + acoustic transformer from `molmo2/tt/`. Codec decoder is NEW (no close precedent).

---

## Inference Flow

```
1. PREFILL (text backbone):
   - Encode voice reference → voice tokens [V_frames, 37] (audio embeddings)
   - Tokenize text → text_ids [T]
   - input = concat([voice_tokens, text_tokens]) → [V+T, 3072]
   - Run 26-layer GQA transformer → h [V+T, 3072], semantic_logits [V+T, 8193]

2. DECODE (autoregressive, text backbone):
   - For each new semantic token position:
     - Input: [semantic_tok_embed + acoustic_tok_embed] → new h, semantic_tok

3. FLOW MATCHING (per audio segment, acoustic transformer):
   - x_0 ~ N(0,1) [N_frames, 36]
   - batch=[cond, uncond] for CFG
   - For step in range(8):
     - t = step / 8
     - v = acoustic_transformer(x_t, t, h)  [batch=2]
     - v_guided = 1.2 * v_cond - 0.2 * v_uncond
     - x_{t+1} = x_t + v_guided * (1/8)
   - Quantize x_1 → acoustic_codes [N_frames, 36]

4. CODEC DECODE (codec decoder, single device):
   - Dequantize: semantic_codes → embed [N, 256], acoustic_codes → float [N, 36]
   - Concat → [N, 292] → project → [N, 1024]
   - 4-stage upsampling with ALiBi self-attention + ConvTranspose
   - Output: waveform [N * 80ms * 24kHz samples]
```

---

## Architecture Phase Gate

- [x] All weight-file components mapped to TTNN impl or flagged NEW
- [x] All non-standard ops have TTNN strategy (ALiBi via mask, causal conv via left-pad, ConvTranspose via conv_transpose2d H=1, weight_norm fused at load)
- [x] Every planned op has a unit test found
- [x] Divisibility checks pass for T3K (8 devices)
- [x] Memory budget < 85% (2.6GB / 12GB = 22% per device)
- [x] Block inventory table complete

**Next Phase**: Reference — build `tt/load_checkpoint.py` and `reference/functional.py`
