# Voxtral-4B-TTS-2603 Bringup Log

## Current Status

Phase: End-to-End Verification | Status: IN PROGRESS
Speech detected: 4/5 prompts (Whisper). Words are wrong (text conditioning limited).
Next: Fix text conditioning so generated speech matches input text.

---

## What Is Done

### Phase 1 — Architecture (COMPLETE)
- Read all HF artifacts: `consolidated.safetensors` (8GB, 386 tensors), `params.json`, `tekken.json`, `voice_embedding/*.pt`
- Identified 3-component TTS pipeline: text decoder backbone + acoustic transformer + codec decoder
- Mapped all non-standard ops: ALiBi attention, causal Conv1D (weight_norm fused), ConvTranspose1D, FSQ quantization, ODE solver
- Fetched safetensors header via HTTP range request — no download needed for architecture analysis
- **Files created:** `ARCHITECTURE.md`

### Phase 2 — Reference Implementation (COMPLETE)
- `tt/load_checkpoint.py` — loads weights, fuses weight_norm parametrizations, materializes VQ codebook
- `reference/functional.py` — complete PyTorch reference for all 3 components
- `reference/save_goldens.py` + `reference/golden/*.pt` — 26 golden tensors for PCC comparison
- `reference/test_functional.py` — 20 reference tests, all pass
- **End-to-end**: "Hello." → 9600 samples (0.40s) at 24kHz

### Phase 3 — TTNN on N150 (COMPLETE, all PCC tests pass)

**Target:** N150 (single Wormhole B0, 12GB DRAM). No tensor parallelism.

**Files:**
- `tt/model_config.py` — VoxtralTTSConfig (N150 settings, max_seq_len=4096)
- `tt/attention.py` — GQA attention, fused wqkv, RoPE, causal SDPA
- `tt/mlp.py` — SwiGLU MLP, BF8 weights to save DRAM
- `tt/acoustic_transformer.py` — 3-layer flow-matching transformer + ODE solver
- `tt/codec_decoder.py` — CPU Phase 1 (causal Conv1D, ALiBi, ConvTranspose — complex TTNN ops)
- `tt/model.py` — Full autoregressive pipeline: prefill → ODE → codec → waveform

**PCC results (N150, all PASS):**

| Block | PCC | p99_diff |
|-------|-----|---------|
| text_attention_layer0 | 0.998867 | 0.001434 |
| text_mlp_layer0 | 0.999927 | 0.001709 |
| decoder_block_layer0 | 0.999996 | 0.007812 |
| acoustic_transformer_velocity | 0.999669 | 0.050781 |
| codec_decoder_waveform | 1.000000 | — |

**Key N150 implementation details:**
- `max_seq_len=4096` (not 65536) — 26-layer KV cache at 65536 causes OOM (7GB)
- MLP weights in BF8, attention in BF16 — fits in 12GB
- N150 SDPA grid: `(8,4)=32 cores` (not `(8,8)=64` — N150 has 56-core limit)
- SDPA prefill: use current k/v (Sq==Sk required), then `fill_cache`
- Decode RoPE: `to_memory_config(q, L1)` before reshape (prevents WIDTH_SHARDED crash)

**Test suite: 36/36 pass** (20 reference + 16 TTNN)

### Phase 4 — End-to-End Verification (IN PROGRESS)

**Bugs found and fixed** (by reading vllm-omni source code):

| Bug | Was Wrong | Fixed To |
|-----|-----------|---------|
| Acoustic transformer input | SUM of 3 projections (1 token) | CONCATENATE to 3-token sequence |
| Semantic prediction source | From transformer output | From `W_semantic @ h_llm` directly |
| Velocity token position | From combined hidden | From position 0 (x_t token) |
| Acoustic transformer RoPE | Applied RoPE | No positional encoding (bidirectional) |
| Token offset (+2 special) | `argmax(sem[:8192])` | Mask pos 0, EoA at pos 1, codes at 2..8193 |
| Audio embedding stride | `8192 + k*21 + v` | `8194 + k*23 + v + 2` (stride=23 with special tokens) |
| FSQ quantization | `(x*10+10).round()` | `clamp(-1,1) → ((x+1)/2)*20 → round` |
| TTS tokenization | ChatCompletion (adds BOS/EOS/markers) | SpeechRequest raw tokens (no BOS/EOS) |
| Input format | `[voice][text][begin_audio]` | `[BOS][begin_audio][voice][text_to_audio=36][text][audio_to_text=35][begin_audio]` |
| Semantic h source | `zeros` instead of real h | Actual h_text_tt from prefill |
| Decode step ordering | Extra begin_audio decode step | Use h_prefill[-1] directly for frame 0 |

**Current state:**
- Whisper detects English speech in 4/5 prompts (was 0/5 before vllm-omni fixes)
- Semantic entropy: 0.55 (was 0.9911 before acoustic transformer fix)
- WER still high (~4.2) — semantic codes not yet text-conditioned

---

## What Is Remaining

### Remaining Issue: Text Conditioning

**Problem:** Generated speech does not say the target text. Only 2 unique semantic codes
(code 8 and code 853) appear across all frames regardless of input text. The speech sounds
like repetitive speech syllables, not the target words.

**Root cause:** The semantic prediction `W_semantic @ h_llm` produces near-concentrated but
text-insensitive outputs. The LLM hidden state `h_llm` at the last prefill position has
cosine similarity ~0.99 for different texts of the same length (147 voice frames dominate
over 4-7 text tokens, diluting the text signal).

**Evidence:**
- Voice reference (casual_male.pt): 117 unique semantic codes across 147 frames ✓
- Our generation: 2 unique codes regardless of text ✗
- Entropy: 0.55 (was 0.9911 after transformer fix, but codes are still text-independent)

**What needs to happen for WER < 30%:**

1. **Understand text conditioning**: The LLM needs to attend more strongly to text tokens
   relative to the 147 voice frames. This may require:
   - Different attention masking/weighting
   - A different prefill strategy (e.g., text-first, then voice)
   - Checking if vllm-omni applies any reweighting

2. **Verify weight loading**: The `mm_audio_embeddings` embedding table layout must exactly
   match vllm-omni's `MultiVocabEmbeddings` scheme. The offsets [0, 8194, 8217, ...] need
   to be verified end-to-end.

3. **Full autoregressive context**: As more audio frames are generated, the model builds up
   audio context. It's possible that after ~20-30 frames, the semantic codes begin to
   diversify. Need to test longer generation windows.

4. **CFG (Classifier-Free Guidance)**: The current implementation runs 2 forward passes
   per ODE step (conditioned + unconditioned). Verify `cfg_alpha=1.2` is correctly applied.

### Remaining Work by Phase

#### Phase 4 (E2E) — Remaining
- [ ] Fix text conditioning so generated speech says the target words
- [ ] WER < 30% on 5 standard test prompts using Whisper CPU
- [ ] Test with multiple voices (neutral_male, fr_female, etc.)
- [ ] Test with longer texts (> 20 words)

#### Phase 5 (tt-inference-server) — Not Started
- [ ] Implement `generator_vllm.py` for Voxtral TTS
- [ ] Register in `tt-vllm-plugin/__init__.py`
- [ ] Add `DeviceModelSpec` to `workflows/model_spec.py`
- [ ] Serve via `/audio/speech` OpenAI-compatible endpoint
- [ ] Test suite: measure latency, RTF, and accuracy

#### Phase 6 (Optimization) — Not Started
- [ ] Move codec decoder transformer blocks to TTNN (currently CPU)
  - ALiBi + sliding-window attention
  - QK-norm, LayerScale
  - Causal Conv1D + ConvTranspose1D
- [ ] Tracy profiling for text decoder and acoustic transformer
- [ ] Profile xlsx from `run_block_profiles.sh`
- [ ] Measure RTF vs GPU reference (H200 baseline: 0.103 at concurrency=1)

---

## Architecture Summary

**Model:** `mistralai/Voxtral-4B-TTS-2603` | 4.1B params | CC BY-NC 4.0
**Target:** N150 (single Wormhole B0, 12GB DRAM) | No tensor parallelism

**3-component pipeline:**
```
Text + Voice → [Text Decoder 26L] → hidden states h
                     ↓
              [Acoustic Transformer 3L] (8 Euler ODE steps × CFG α=1.2)
              Semantic: W_semantic @ h  [8192 codes]
              Acoustic: ODE on [x_t, t, h] → 36 acoustic codes
                     ↓
              [Voxtral Codec Decoder 4-stage]  ← CPU Phase 1
              (ALiBi attention + causal Conv1D + ConvTranspose1D)
                     ↓
              24kHz waveform
```

**Input format** (from `mistral_common.encode_speech_request`):
```
[BOS=1] [begin_audio=25] [voice×V] [text_to_audio=36] [text×T] [audio_to_text=35] [begin_audio=25]
```
Voice frames: pre-computed in `voice_embedding/*.pt` [V, 3072]

**Audio token embedding layout** (MultiVocabEmbeddings):
- Semantic codebook: rows 0..8193 (0=EMPTY, 1=EoA, 2..8193=codes 0..8191)
- Acoustic codebook k: rows 8194 + k×23 + v + 2 (stride=23 with 2 special tokens)
- Total: 8194 + 36×23 = 9022, padded to 9088

**Weights at:**
`/home/ttuser/.cache/huggingface/hub/models--mistralai--Voxtral-4B-TTS-2603/snapshots/b81be46c3777f88621676791b512bb01dc1cb970/`

---

## 2026-05-02

### Architecture — COMPLETE

Model: `mistralai/Voxtral-4B-TTS-2603`
Type: Text-to-Speech (3-component pipeline)
Target device: N150 (1×1, single Wormhole B0, 12GB DRAM)

**Components identified:**
1. **Text Decoder Backbone** — 26-layer GQA transformer (Ministral-3B), dim=3072, n_heads=32, n_kv_heads=8, hidden_dim=9216, vocab=131072
2. **Acoustic Flow-Matching Transformer** — 3 layers, same dims as backbone, 8 Euler ODE steps, CFG α=1.2
3. **Voxtral Codec Decoder** — 300M, dim=1024, 4-stage ConvTranspose + ALiBi attention, window=2→4→8→16

**Files created:** ARCHITECTURE.md

---

### Reference — ALL BLOCKS — PASS

**Test results (20 tests total):** 12 helper unit tests + 4 mock-weight + 8 real-weight + end-to-end — ALL PASS

**Confirmed architecture details from weight keys:**
- `acoustic_transformer.*` contains BOTH semantic head AND flow-matching module
- Voice embeddings are PRE-COMPUTED [V, 3072] — no codec encoder at inference
- Codec decoder block pattern: 0=initial_conv, 1/3/5/7=attn+mlp, 2/4/6=conv_transpose

**Golden tensors:** 26 files in `reference/golden/`

**Files created:** tt/load_checkpoint.py, reference/functional.py, reference/test_functional.py, reference/save_goldens.py

---

### TTNN Phase 3 — ALL PCC TESTS PASS

**Target device confirmed: N150** (not T3K — per user direction)

**Debug notes:**
1. SDPA `Sq==Sk` error — use current k/v for SDPA, then fill_cache
2. N150 SDPA grid (8,4) not (8,8) — N150 has 7×8=56 cores
3. `ttnn.rms_norm` takes `weight=` as keyword arg
4. Integration OOM — max_seq_len=4096 (not 65536) to fit 26-layer KV cache
5. BF8 for MLP weights to save DRAM

**Total: 36/36 tests pass** (20 reference + 16 TTNN incl. 2 integration)

**Files created:** tt/model_config.py, tt/attention.py, tt/mlp.py, tt/acoustic_transformer.py, tt/codec_decoder.py, tt/model.py, tests/test_tt_text_decoder.py, tests/test_tt_acoustic_transformer.py, tests/test_tt_codec_decoder.py, tests/test_integration.py

---

### Phase 4 — E2E Verification (IN PROGRESS)

**Bugs fixed (chronological):**
1. FSQ: `x_t.round()` → `((clamp(x,-1,1)+1)/2)*20` (from vllm-omni)
2. Semantic prediction: zeros used instead of real h_text_tt
3. Autoregressive loop: prefill + per-frame decode with KV cache
4. Decode RoPE crash: `to_memory_config(q, L1)` before reshape
5. Redundant decode: use h_prefill[-1] directly for frame 0
6. TTS tokenization: SpeechRequest raw tokens (not ChatCompletion)
7. Input format: missing [BOS, begin_audio, text_to_audio=36, audio_to_text=35]
8. Token offset (+2): EoA=1, codes at 2..8193 (from vllm-omni source)
9. Embedding stride: 8194 + k×23 + v+2 (not 8192 + k×21 + v)
10. **Acoustic transformer**: CONCATENATE 3 tokens, semantic from h_llm directly, no RoPE

**Results before vllm-omni source analysis:** entropy=0.9911, 1/5 speech detected
**Results after:** entropy=0.5516, 4/5 speech detected, Whisper transcribes English words

**Whisper WER summary (Whisper small on CPU):**
| Input | Transcript | WER |
|-------|-----------|-----|
| "Hello, world." | "If we" | high |
| "One two three four five." | "I am going to finish it. Thank you." | high |
| "Good morning." | "" | 1.00 |
| "Paris is a beautiful city." | "In one push." | high |
| "The quick brown fox jumps." | "Thank you." | high |

**SpeechT5 reference (same infrastructure):** WER=0.00 for 4/5 prompts ✓ (confirms TTNN infrastructure is working)

**Files added:** tests/test_e2e_whisper_verification.py, vllm-omni/ (cloned for source reference)
