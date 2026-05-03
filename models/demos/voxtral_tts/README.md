# Voxtral-4B-TTS-2603 on Tenstorrent N150

Text-to-Speech model by Mistral AI running on Tenstorrent N150 (single Wormhole B0).

## Status

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1: Architecture | ✅ COMPLETE | All components mapped |
| Phase 2: Reference | ✅ COMPLETE | 20/20 tests pass |
| Phase 3: TTNN N150 | ✅ COMPLETE | 36/36 tests pass, all PCC > 0.99 |
| Phase 4: E2E Verification | 🔄 IN PROGRESS | Speech generated, wrong words (WER high) |
| Phase 5: tt-inference-server | ⏳ NOT STARTED | vllm TTS endpoint |
| Phase 6: Optimization | ⏳ NOT STARTED | Codec decoder TTNN, Tracy profiling |

See [BRINGUP_LOG.md](BRINGUP_LOG.md) for full details and debug history.

---

## What Works

- **TTNN blocks** — text decoder, acoustic transformer, codec decoder all pass PCC > 0.99
- **Audio generation** — produces non-silent audio at 24kHz
- **Speech detection** — Whisper detects English speech in 4/5 test prompts
- **Correct architecture** — verified against vllm-omni source code
- **SpeechT5 TTS** — separate working TTS model on N150 (WER=0.00 for common sentences)

## What Doesn't Work Yet

- **Generated words don't match input text** — semantic codes are text-insensitive;
  only 2 unique codes appear regardless of input (code 8 and 853 dominate).
  Root cause: LLM hidden state at last prefill position is nearly identical for
  different texts of same length (147 voice frames dilute text signal).

---

## Architecture

Three-component TTS pipeline:

```
Text + Voice → [Text Decoder 26L] → h (3072-dim hidden states)
                     ↓
              [Acoustic Transformer 3L] — 8 Euler ODE steps, CFG α=1.2
              Semantic codes: W_semantic @ h  (8192-entry codebook)
              Acoustic codes: ODE on concat[x_t, t_emb, h] → 36 FSQ codes
                     ↓
              [Voxtral Codec Decoder]  — 4-stage upsampler (CPU Phase 1)
                     ↓
              24kHz waveform
```

**Input format** (from `mistral_common.encode_speech_request`):
```
[BOS=1] [begin_audio=25] [voice×V] [text_to_audio=36] [text×T] [audio_to_text=35] [begin_audio=25]
```
Voice frames: pre-computed `voice_embedding/casual_male.pt` [147, 3072] (no codec encoder needed).

**Model weights:** `mistralai/Voxtral-4B-TTS-2603` (~8GB BF16)
**Hardware:** N150 (single Wormhole B0, 12GB DRAM)

---

## Requirements

```bash
pip install openai-whisper soundfile scipy num2words
```

Model weights must be at:
```
~/.cache/huggingface/hub/models--mistralai--Voxtral-4B-TTS-2603/snapshots/<hash>/
```

---

## Setup

```bash
cd tt-metal
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd):$(pwd)/models
export ARCH_NAME=wormhole_b0
export VOXTRAL_MODEL_DIR=~/.cache/huggingface/hub/models--mistralai--Voxtral-4B-TTS-2603/snapshots/b81be46c3777f88621676791b512bb01dc1cb970
source python_env/bin/activate
```

---

## Test Commands

### Reference tests (CPU, no device needed):
```bash
pytest models/demos/voxtral_tts/reference/test_functional.py -v
```

### TTNN block PCC tests (N150):
```bash
pytest models/demos/voxtral_tts/tests/test_tt_text_decoder.py -v
pytest models/demos/voxtral_tts/tests/test_tt_acoustic_transformer.py -v
pytest models/demos/voxtral_tts/tests/test_tt_codec_decoder.py -v
```

### Integration + E2E (N150):
```bash
pytest models/demos/voxtral_tts/tests/test_integration.py -v
pytest models/demos/voxtral_tts/tests/test_e2e_whisper_verification.py -v
```

### Full suite:
```bash
pytest models/demos/voxtral_tts/ -v --timeout=600
```

---

## Demo

```bash
python3 models/demos/voxtral_tts/demo/demo.py \
    --text "Hello, this is a test." \
    --voice casual_male \
    --output output.wav
```

**Note:** The demo generates audio but the words do not yet match the input text (Phase 4 incomplete). Audio sounds like English speech in the casual_male voice but with different content.

---

## PCC Results (Phase 3, N150)

| Block | PCC | Status |
|-------|-----|--------|
| text_attention_layer0 | 0.998867 | ✅ |
| text_mlp_layer0 | 0.999927 | ✅ |
| decoder_block_layer0 | 0.999996 | ✅ |
| acoustic_transformer_velocity | 0.999669 | ✅ |
| codec_decoder_waveform | 1.000000 | ✅ (CPU) |
| integration "Hello." | — | ✅ 9600 samples |

---

## Key Implementation Details

### N150-Specific Issues Found
- **max_seq_len=4096** (not 65536) — KV cache at full length = 7GB OOM
- **SDPA grid `(8,4)`** — N150 has 7×8=56 cores; `(8,8)=64` crashes
- **BF8 MLP weights** — saves ~2GB vs BF16; attention stays BF16 for precision
- **SDPA prefill** — run SDPA on current k/v, then `fill_cache` (Sq==Sk required)
- **Decode RoPE** — `to_memory_config(q, L1_MEMORY_CONFIG)` before reshape

### Acoustic Transformer Architecture (from vllm-omni source)
- **3 separate input tokens**: concatenate `[x_proj, t_proj, h_proj]` → seq_len=3
- **Semantic prediction**: `W_semantic @ h_llm` (raw LLM hidden state, before transformer)
- **Velocity prediction**: from token position 0 (x_t token) of transformer output
- **No RoPE** in bidirectional attention

### Token Layout (from vllm-omni MultiVocabEmbeddings)
- `EMPTY_AUDIO=0`, `END_AUDIO=1` (EoA), semantic codes at positions `2..8193`
- Audio embedding rows: `8194 + k×23 + v+2` for acoustic codebook k, level v

---

## Known Limitations

1. **Phase 4 incomplete**: Generated speech doesn't say the target text. Only 2 unique
   semantic codes regardless of input. Requires fixing text conditioning in the LLM.

2. **Codec decoder on CPU**: ALiBi attention + causal Conv1D + ConvTranspose1D run on CPU
   (Phase 6 scope to move to TTNN).

3. **max_seq_len=4096**: Limits input length. Increase for longer texts (more DRAM needed).

4. **License**: CC BY-NC 4.0 — non-commercial only.

5. **No codec encoder**: Only decoder released. Preset voices work; voice cloning from
   arbitrary audio requires the encoder (not in the released weights).

---

## File Structure

```
models/demos/voxtral_tts/
├── ARCHITECTURE.md          # Block inventory, N150 layout, non-standard op strategies
├── BRINGUP_LOG.md           # Phase-by-phase results, bug fixes, debug history
├── README.md                # This file
├── demo/
│   └── demo.py              # CLI demo
├── reference/
│   ├── functional.py        # Pure PyTorch reference implementation
│   ├── test_functional.py   # 20 reference tests
│   ├── save_goldens.py      # Golden tensor generation
│   └── golden/              # 26 .pt golden tensor files
├── tt/
│   ├── model_config.py      # VoxtralTTSConfig (N150 settings)
│   ├── load_checkpoint.py   # Weight loading + weight_norm fusing
│   ├── attention.py         # TtVoxtralTextAttention (GQA, RoPE, causal SDPA)
│   ├── mlp.py               # TtVoxtralTextMLP (SwiGLU, BF8)
│   ├── acoustic_transformer.py  # TtVoxtralAcousticTransformer + ODE solver
│   ├── codec_decoder.py     # TtVoxtralCodecDecoder (CPU Phase 1)
│   └── model.py             # VoxtralTTSModel (full autoregressive pipeline)
└── tests/
    ├── test_tt_text_decoder.py          # 4 PCC tests
    ├── test_tt_acoustic_transformer.py  # 2 tests
    ├── test_tt_codec_decoder.py         # 3 tests
    ├── test_integration.py              # 2 E2E tests
    └── test_e2e_whisper_verification.py # Whisper WER tests
```
