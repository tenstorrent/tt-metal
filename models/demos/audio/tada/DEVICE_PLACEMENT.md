# TADA TTS Pipeline: Device Placement Map

## Pipeline Overview

```
Text Input
  | (CPU: tokenize)
[Llama 3.2 1B AR Loop] <- TT device (16 transformer layers)
  |-- Embedding construction <- TT (5x ttnn.embedding + ttnn.add)
  |-- Llama forward <- TT (tt_transformers, KV cache on device)
  |-- LM head <- TT (ttnn.linear)
  |-- Text sampling <- CPU (top-k/p, temperature, repetition penalty)
  +-- VibeVoice flow matching <- TT (6 HeadLayers) + CPU (ODE solver, CFG scaling)
  |
[Decoder] <- TT (6-layer transformer) + CPU (DACDecoder CNN)
  |
Waveform Output (24kHz)
```

---

## Ops on Tenstorrent (TTNN)

| Module | TTNN Ops Used | Notes |
|--------|--------------|-------|
| **Llama 3.2 1B (16 layers)** | Full transformer via `tt_transformers` | Reuses existing implementation; two KV caches for CFG |
| **Input Embedding Construction** | `ttnn.embedding` (x4: token, mask, time_start, time_end), `ttnn.linear` (acoustic proj), `ttnn.add` (x5 to sum) | All 5 embeddings summed on device |
| **LM Head** | `ttnn.linear` | (B, 1, 2048) -> (B, 1, vocab_size) |
| **VibeVoice Timestep MLP** | `ttnn.linear` (x2), `ttnn.silu` | Sinusoidal embedding done on CPU, MLP on TT |
| **VibeVoice Projections** | `ttnn.linear` (noisy_images_proj, cond_proj), `ttnn.add` | Project inputs to hidden dim |
| **VibeVoice HeadLayer (x6)** | `ttnn.silu`, `ttnn.linear` (adaLN), `ttnn.rms_norm`, `ttnn.mul`, `ttnn.add`, `ttnn.ones_like`, SwiGLU FFN: `ttnn.linear` (x3 gate/up/down), `ttnn.silu`, `ttnn.mul` | Full diffusion head on device |
| **VibeVoice FinalLayer** | `ttnn.rms_norm`, `ttnn.linear` (adaLN + output), `ttnn.mul`, `ttnn.add` | Velocity output (B, 1, 528) |
| **Encoder LocalAttentionEncoder (6 layers)** | Per layer: `ttnn.linear` (QKV, out_proj, FFN x2), `ttnn.experimental.nlp_create_qkv_heads`, `ttnn.permute`, `ttnn.matmul` (Q@K^T, attn@V), `ttnn.mul` (scale), `ttnn.add` (mask, residual), `ttnn.softmax`, `ttnn.experimental.nlp_concat_heads`, `ttnn.gelu`, `ttnn.layer_norm` (x2) | HiFi4 + fp32 accumulation for precision |
| **Decoder Projection** | `ttnn.linear` | (B, T, 512) -> (B, T, 1024) |
| **Decoder LocalAttentionEncoder (6 layers)** | Same as encoder transformer | Identical architecture |
| **Encoder Hidden Linear** | `ttnn.linear` | (B, T, 1024) -> (B, T, 512) |
| **Attention Mask (on device)** | `ttnn.from_torch` of float mask | Bool->float(-inf) conversion done on CPU, mask stored on device |

---

## Ops on CPU (PyTorch)

| Module | Why on CPU |
|--------|-----------|
| **Text normalization + tokenization** | String/regex ops, Llama tokenizer |
| **WavEncoder CNN** (encoder) | Snake1d activation + weight-normed Conv1d with strides [6,5,4,4]; not native TTNN ops |
| **Positional embedding indexing** | `pos_emb_weight[token_masks]` -- index-based lookup before device transfer |
| **Segment attention mask creation** | Graph/boolean logic (`cumsum`, comparisons); computed once, transferred to TT |
| **RoPE (Rotary Position Embeddings)** | Reference uses interleaved pairs requiring 5D reshape `(..., D//2, 2)` -- TTNN tile layout only supports 4D. Q/K transferred to CPU (~1MB each), rotated, transferred back |
| **Text token sampling** | Repetition penalty, temperature scaling, top-k/p filtering, `torch.multinomial` |
| **Sinusoidal timestep embedding** | Small (256-dim), computed once per ODE step |
| **ODE solver (Euler method)** | Scalar time schedule, `speech = speech + dt * velocity` |
| **CFG velocity scaling** | Split doubled batch, apply separate acoustic/duration CFG scales |
| **Flow matching noise generation** | `torch.randn()` with seed control |
| **Duration expansion** | Repeat acoustic features by predicted duration (loop + cat) |
| **DACDecoder CNN** (decoder) | ConvTranspose1d with strides [4,4,5,6]; not natively supported in TTNN |
| **Waveform silence trimming** | Leading zeros detection |

---

## Key Files

| File | Role |
|------|------|
| `tt/tada_generator.py` | Main generator: AR loop orchestration, flow matching, encoding, decoding |
| `tt/ttnn_functional_common.py` | Shared: RoPE, LocalSelfAttention, LocalAttentionEncoder, Snake1d |
| `tt/ttnn_functional_vibevoice.py` | VibeVoice diffusion head (6 HeadLayers + FinalLayer) |
| `tt/ttnn_functional_encoder.py` | Encoder: WavEncoder CNN (CPU) + transformer (TT) |
| `tt/ttnn_functional_decoder.py` | Decoder: projection (TT) + transformer (TT) + DACDecoder CNN (CPU) |
| `tt/ttnn_functional_tada.py` | Embedding construction + LM head |
