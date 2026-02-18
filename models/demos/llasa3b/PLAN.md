# Llasa-3B Model Bring-Up Plan

## Current Status (Feb 18, 2026)
**Status: ✅ Full End-to-End Pipeline Working (Zero-Shot + Prompted TTS)**

The Llasa-3B model has been successfully brought up on Tenstorrent N300 hardware with full audio output.
Both zero-shot TTS and prompted TTS (voice cloning) are working in the PyTorch reference and on TTNN.

### Achievements
- **TTNN Inference**: Llasa-3B (193k vocab) running on N300 via `tt_transformers`.
- **Zero-Shot Audio Generation**: Full pipeline from text → TTNN inference → speech tokens → XCodec2 decode → WAV file.
- **Prompted TTS (Voice Cloning)**: Full pipeline matching official reference with robust alignment.
  - Encoder inputs: ~795 prompt tokens (16s) → TTNN prefill → ~650 generated tokens → XCodec2 decode → voice-cloned WAV.
  - **Clean Output**: Output WAV contains *only* the generated target speech (prompt audio removed).
- **Performance**: ~1360 tok/s prefill, ~16.4 tok/s decode on TTNN.
- **Code Restructured**: Clean separation — utilities in `tt/llasa_utils.py`, demo harness in `demo/llasa_demo.py`.

### Issues Resolved
1. **L1 OOM (LM Head)**: Large vocab concat exceeded L1. Fixed by routing to DRAM in `lm_head.py`.
2. **KV Cache Shape**: Fixed list-of-lists structure for `Generator.prefill_forward_text`.
3. **XCodec2 Import Failure**: Unused `import torchaudio` in `xcodec2` caused `OSError`. Fixed.
4. **XCodec2 Encoder Weight Mismatch (SnakeBeta)**: Fixed critical issue where `xcodec2` package expects `bias` but checkpoint has `beta`. Renamed parameter in `activations.py` to fix encoder output.
5. **Prompt Alignment**: Model was regenerating prompt text because 3s audio didn't match 16s text. **Fix**: Increased `MAX_PROMPT_SPEECH_TOKENS` to 1000 in `llasa_utils.py` to allow full 16s prompt audio.
6. **Output Cleaning**: Modified `llasa_demo.py` to exclude prompt tokens from the final output WAV, ensuring the user hears only the new content.

### Remaining Work
- [ ] **Performance optimization**: RTF ~5.0 (target < 0.5). Needs Trace mode / 2CQ.
- [ ] **PCC validation**: Module-level PCC tests vs PyTorch reference.
- [ ] **MAX_PREFILL_CHUNK_SIZE**: Currently defaulting to 4 (unknown model). Tune for better prefill.

---

## 1. Architecture Overview

### Llasa-3B = LLaMA-3.2-3B + Speech Tokens
```
                    ┌───────────────────────────────┐
                    │       Llasa-3B Pipeline        │
                    └───────────────────────────────┘
                                  │
            ┌─────────────────────┼─────────────────────┐
            │                     │                     │
   ┌────────▼────────┐  ┌────────▼────────┐  ┌────────▼────────┐
   │   Tokenizer +   │  │   LLaMA-3.2-3B  │  │    XCodec2      │
   │   Chat Template │  │   (28 layers)   │  │  Encode/Decode  │
   │                 │  │   TTNN on WH    │  │   (CPU only)    │
   └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
            │                     │                     │
   Text → Token IDs      Token IDs → Speech     Encode: WAV → VQ codes
   (incl. speech         Token IDs              Decode: VQ codes → WAV
    special tokens)      (autoregressive)
```

### Model Specifications
| Parameter | Value |
|-----------|-------|
| Base Model | LLaMA-3.2-3B |
| Hidden Dim | 3072 |
| Num Layers | 28 |
| Attention Heads | 24 |
| KV Heads | 8 (GQA) |
| Head Dim | 128 |
| Intermediate Size | 8192 |
| Base Vocab Size | 128,256 (LLaMA) |
| Speech Tokens | 65,536 (XCodec2) |
| **Total Vocab Size** | **~193,792** |
| Max Seq Len | 2048 (training limit) |
| Max Position Embeddings | 131,072 (from LLaMA 3.2) |
| RoPE | llama3 style |
| Target Hardware | N300 (Wormhole) |

### Two Operating Modes
1. **Zero-shot TTS**: Text only → Speech tokens → Audio ✅ Working (TTNN + Reference)
2. **Prompted TTS** (Voice Cloning): Audio prompt → XCodec2 encode → prepend speech tokens + text → Speech tokens → Audio ✅ Working (TTNN + Reference)

### Token Budget (max_length=2048)
The model was trained with `max_length=2048`. This total budget is shared:
```
~30 tokens   — chat template overhead
~N tokens    — input text (~1 token per word)
~M tokens    — prompt speech tokens (50 tokens/sec of prompt audio)
~R tokens    — generated speech tokens (50 tokens/sec of output audio)
──────────────
Total ≤ 2048
```

For prompted TTS, prompt audio is truncated to 150 tokens (~3 seconds) to leave room for generation.

---

## 2. Strategy — Leveraging Existing Infrastructure

### Key Insight: Llasa-3B IS a LLaMA-3.2-3B model
The critical insight is that **Llasa-3B uses the exact same architecture as LLaMA-3.2-3B** (`LlamaForCausalLM`). The only differences are:
1. **Extended vocabulary** (65,536 additional speech tokens)
2. **Different tokenizer** (with speech-specific special tokens)
3. **Post-processing**: speech token extraction + XCodec2 decoding

### Components Reused (from `models/tt_transformers/tt/`)
| Component | File | Status |
|-----------|------|--------|
| Transformer model | `model.py` | ✅ Reused directly |
| Attention (GQA) | `attention.py` | ✅ Reused directly |
| MLP (SwiGLU) | `mlp.py` | ✅ Reused directly |
| Decoder layers | `decoder.py` | ✅ Reused directly |
| RoPE (llama3) | `rope.py` | ✅ Reused directly |
| Embedding | `embedding.py` | ✅ Works with larger vocab |
| LM Head | `lm_head.py` | ✅ Fixed for large vocab (DRAM concat) |
| Model Config | `model_config.py` | ✅ Works with Llasa config |
| Generator | `generator.py` | ✅ Used for prefill + decode |
| Checkpoint loading | `load_checkpoints.py` | ✅ Works with HF safetensors |

---

## 3. Implementation Phases

### Phase 1: PyTorch Reference & Validation ✅
- [x] Install dependencies (`xcodec2==0.1.5`, `soundfile`)
- [x] Reference implementation with both zero-shot and prompted modes
- [x] Matches official HuggingFace examples exactly
- [x] Prompted TTS tested with official Anna.wav prompt — produces valid voice-cloned audio

### Phase 2: Model Config & Weight Loading ✅
- [x] `models/demos/llasa3b/model_params/config.json` with vocab_size=193800
- [x] ModelArgs handles larger vocab automatically
- [x] Weight cache at `model_cache/HKUSTAudio/Llasa-3B/N300`
- [x] HF safetensors checkpoint loading verified

### Phase 3: TTNN Model Bring-Up ✅
- [x] Model initialization via `create_tt_model`
- [x] Prefill with tokenized text input (~1360 tok/s)
- [x] Autoregressive decode generating speech tokens (~16.4 tok/s)
- [x] L1 OOM fix for large vocab LM head concat

### Phase 4: End-to-End Pipeline ✅
- [x] Tokenize input text with Llasa chat template
- [x] Prefill on TTNN
- [x] Autoregressive decode until `<|SPEECH_GENERATION_END|>` or max_length
- [x] Extract speech token IDs from generated tokens
- [x] Decode speech tokens via XCodec2 on CPU
- [x] Save audio as WAV file (16kHz)

### Phase 5: Code Restructuring ✅
- [x] Clean utility module: `tt/llasa_utils.py` (speech token manipulation, chat template, XCodec2 decode)
- [x] Slim demo harness: `demo/llasa_demo.py` (prepare_llasa_generator, run_llasa_tts, test_llasa_tts)
- [x] Sample prompts: `demo/input_data.json`
- [x] Removed dead code (`tt/llasa_utils.py` old LlasaUtils class)

### Phase 6: XCodec2 Fixes ✅
- [x] Fix dead `import torchaudio` in `xcodec2/vq/bs_roformer5.py` (commented out)
- [x] Fix SnakeBeta weight name mismatch in `xcodec2/vq/activations.py` (`self.bias` → `self.beta`)
- [x] Clear `.pyc` caches after fixes

### Phase 7: Testing & Optimization
- [x] Prompted TTS mode on TTNN (encode prompt audio → prepend speech tokens)
- [ ] PCC tests per module vs PyTorch reference
- [ ] Performance optimization (Trace, 2CQ)
- [ ] Batch support

---

## 4. Directory Structure

```
models/demos/llasa3b/
├── PLAN.md                         # This file (development notes)
├── README.md                       # End-user documentation
├── model_params/
│   └── config.json                 # Llasa-3B model config (vocab_size=193800)
├── reference/
│   └── llasa_reference.py          # PyTorch reference (zero-shot + prompted TTS)
├── tt/
│   ├── __init__.py                 # Package init
│   └── llasa_utils.py              # Speech token utils, chat template, XCodec2 decode
├── demo/
│   ├── llasa_demo.py               # Main TTNN demo (pytest entry point)
│   ├── input_data.json             # Sample text prompts
│   └── output/
│       └── llasa_output.wav        # Generated audio output
└── tests/                          # (future) PCC and accuracy tests
```

---

## 5. Key Challenges & Solutions

### Challenge 1: Large Vocabulary (~194K tokens) → L1 OOM
**Impact**: The `LMHead` concat for 193k tokens exceeded L1 buffer capacity.
**Root Cause**: `ttnn.concat` in `lm_head.py` was hardcoded to use `ttnn.L1_MEMORY_CONFIG`.
**Fix**: Modified `lm_head.py` to use `ttnn.DRAM_MEMORY_CONFIG` when `padded_vocab_size > 128256`:
```python
LLAMA_VOCAB_SIZE = 128256
use_dram_concat = use_prefetcher or self.padded_vocab_size > LLAMA_VOCAB_SIZE
output = ttnn.concat(outputs, dim=-1,
    memory_config=ttnn.DRAM_MEMORY_CONFIG if use_dram_concat else ttnn.L1_MEMORY_CONFIG, ...)
```

### Challenge 2: XCodec2 Import Failure (torchaudio)
**Impact**: `from xcodec2.modeling_xcodec2 import XCodec2Model` failed with `OSError: libtorch_cuda.so`.
**Root Cause**: `xcodec2/vq/bs_roformer5.py` line 5 had `import torchaudio` — a **dead import** (never used in the file). `torchaudio` tried to load CUDA libs that don't exist in the CPU-only environment.
**Fix**: Commented out the unused import in `bs_roformer5.py`.

### Challenge 3: KV Cache Shape
**Impact**: `ttnn.prim.paged_fill_cache` expected list-of-lists for KV cache.
**Fix**: Wrapped `kv_cache` in `[kv_cache]` in `llasa_demo.py`.

### Challenge 4: XCodec2 Encoder Weight Mismatch (SnakeBeta)
**Impact**: Prompted TTS (voice cloning) produced garbage audio — unintelligible output followed by silence.
**Root Cause**: The `xcodec2==0.1.5` package renamed the `SnakeBeta` activation's magnitude parameter from `self.beta` to `self.bias` in `vq/activations.py`. However, the HuggingFace checkpoint (`HKUSTAudio/xcodec2`) stores the weights under the old name `.act.beta`. On loading:
- `.act.beta` weights from checkpoint → "not used" (no matching parameter)
- `.act.bias` in code → "newly initialized" (random values)
This affected all `CodecEnc.conv_blocks.*.act` parameters (the encoder's SnakeBeta activations). The **decoder** was unaffected because it does not use SnakeBeta, so zero-shot TTS worked fine. But the **encoder** (needed for prompted TTS to convert prompt WAV → VQ codes) produced garbage.
**Fix**: Renamed `self.bias` back to `self.beta` in `xcodec2/vq/activations.py` (lines 97, 100, 103, 114).
**Detection**: The HuggingFace model loading warnings were the clue — "Some weights not used" + "Some weights newly initialized" indicated a parameter name mismatch.

---

## 6. How to Run

### Prerequisites
```bash
# Install dependencies
source python_env/bin/activate
uv pip install xcodec2==0.1.5 soundfile

# Fix 1: Dead torchaudio import in xcodec2 (CPU-only environments)
# File: python_env/lib/python3.10/site-packages/xcodec2/vq/bs_roformer5.py
# Comment out line 5: import torchaudio → # import torchaudio

# Fix 2: SnakeBeta weight name mismatch (required for prompted TTS / voice cloning)
# File: python_env/lib/python3.10/site-packages/xcodec2/vq/activations.py
# In the SnakeBeta class, rename all occurrences of self.bias → self.beta
# (lines 97, 100, 103, 114)

# Clear Python bytecode cache after fixes
find python_env/lib/python3.10/site-packages/xcodec2 -name "*.pyc" -delete
```

### Run the TTNN Demo (Zero-Shot TTS)
```bash
export HF_MODEL=HKUSTAudio/Llasa-3B
pytest models/demos/llasa3b/demo/llasa_demo.py -s -k "test_llasa_tts"
# Output: models/demos/llasa3b/demo/output/llasa_output.wav
```

### Run the TTNN Demo (Prompted / Voice Cloning)
```bash
export HF_MODEL=HKUSTAudio/Llasa-3B
pytest models/demos/llasa3b/demo/llasa_demo.py -s -k "test_llasa_tts_prompted"
# Output: models/demos/llasa3b/demo/output/llasa_output_prompted.wav
```

### Run the PyTorch Reference
```bash
# Zero-shot TTS
python models/demos/llasa3b/reference/llasa_reference.py \
    --text "Hello, this is a test." \
    --output_dir reference_output

# Prompted TTS (voice cloning)
python models/demos/llasa3b/reference/llasa_reference.py \
    --text "Target text to generate." \
    --prompt_text "Text spoken in the prompt audio." \
    --prompt_wav prompt.wav \
    --output_dir reference_output
```

---

## 7. Performance

### TTNN on N300 (Zero-Shot TTS)
| Metric | Value |
|--------|-------|
| Prefill throughput | ~1,360 tok/s |
| Decode throughput | ~16.4 tok/s/user |
| Typical speech tokens | 300–500 |
| Audio output | 16kHz WAV |
| RTF (Real-Time Factor) | ~5.0 (target < 0.5) |

### PyTorch Reference on CPU
| Metric | Value |
|--------|-------|
| Decode throughput | ~1.6 tok/s |
| TTNN speedup | ~10x |

---

## 8. Success Criteria

### Must-Have (Stage 1 Bring-Up) ✅
- [x] Model runs on N300 hardware with no errors
- [x] Zero-shot TTS mode works (text → speech → audio) — TTNN + Reference
- [x] Prompted TTS mode works (voice cloning) — TTNN + Reference
- [x] XCodec2 encoder and decoder produce valid audio
- [x] XCodec2 dependency issues documented and fixed
- [x] Decode throughput verified (~16.4 tok/s on TTNN)
- [x] Clear setup and running instructions
- [x] Code restructured for maintainability

### Nice-to-Have (Future Optimization)
- [ ] Trace mode for decode
- [ ] 2CQ optimization
- [ ] Chunked prefill for long contexts
- [ ] Batch support (multiple users)
- [ ] Streaming audio output
- [ ] On-device sampling
- [ ] Performance profiling
