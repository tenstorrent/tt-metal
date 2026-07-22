# CosyVoice2-0.5B — TTS on Wormhole N300

## Platforms
Wormhole (N300)

## Introduction

[CosyVoice2-0.5B](https://github.com/FunAudioLLM/CosyVoice) is a multilingual
text-to-speech model from Alibaba FunAudioLLM. It combines a Qwen2.5-0.5B LLM
(speech-token prediction), a flow-matching decoder (UNet1D estimator + Euler CFM),
and a HiFT vocoder (conv stack + iSTFT) to generate 24 kHz speech.

This bring-up on Tenstorrent Wormhole N300 using TTNN includes **Stage 1 (functional
correctness)** and **Stage 2 (performance optimization)**. The LLM and flow estimator
run on-device; the vocoder runs on host (CPU) due to bf16 precision constraints.

**4 generation modes:** SFT, zero-shot voice cloning, cross-lingual, instruct.
**5 languages:** Chinese, English, Japanese (katakana), Cantonese (yue), Korean.

## Prerequisites

- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal)
- Installed: [TT-Metalium / TT-NN](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
- Wormhole N300 device available

## Setup

```bash
source /root/tt-metal/python_env/bin/activate

# Install curated dependencies (NEVER use CosyVoice's upstream requirements.txt)
uv pip install --python /root/tt-metal/python_env/bin/python \
  -r models/demos/cosyvoice/requirements-cosyvoice.txt

# Clone reference repo + download checkpoint (if not already present)
cd models/demos/cosyvoice
python scripts/clone_reference.py
python scripts/download_model.py

# Generate golden fixtures (if not already present)
python scripts/gen_golden.py --modes zero_shot,cross_lingual,instruct2,sft
```

## How to Run

### Interactive Demo

```bash
cd /root/tt-metal
python models/demos/cosyvoice/demo/try_it.py
```

### Pytest Demo (20 WAVs: 4 modes × 5 languages)

```bash
cd /root/tt-metal
python -m pytest models/demos/cosyvoice/demo/demo.py -v -s
```

Output WAVs are written to `models/demos/cosyvoice/demo/output/`.


## Tests

```bash
cd /root/tt-metal

# PCC gates (40 tests, ~90s)
python -m pytest models/demos/cosyvoice/tests/pcc/ -q

# E2E mode tests + token accuracy (8 tests, ~100s)
python -m pytest models/demos/cosyvoice/tests/e2e/test_modes.py -v

# Performance (C6 throughput)
python -m pytest models/demos/cosyvoice/tests/perf/test_throughput.py -v -s

# Audio quality evaluation (C8: WER + speaker similarity)
python -m pytest models/demos/cosyvoice/demo/eval.py -v -s
```

## Results

| Metric | Target | Measured | Status |
|--------|--------|----------|--------|
| LLM decode throughput | ≥ 30 tok/s | 113.6 tok/s (traced) | ✓ PASS |
| E2E RTF | < 0.5 | 0.61 | ✗ (vocoder on host — see limitations) |
| Token accuracy (zero_shot) | > 95% | 96.0% | ✓ PASS |
| Token accuracy (cross_lingual) | > 95% | 100.0% | ✓ PASS |
| Token accuracy (instruct2) | > 95% | 100.0% | ✓ PASS |
| Token accuracy (sft) | > 95% | 98.0% | ✓ PASS |
| Speaker similarity (CAM++) | > 60 | 82.9 | ✓ PASS |
| LLM prefill PCC | ≥ 0.99 | 0.997 | ✓ PASS |
| LLM decode PCC | ≥ 0.99 | 0.996–0.998 | ✓ PASS |
| Flow estimator PCC (mel) | ≥ 0.99 | 0.995–0.999 | ✓ PASS |
| Vocoder waveform PCC | ≥ 0.99 | 1.0 | ✓ PASS |

## Known Limitations

1. **Non-streaming only**: No bi-directional streaming; full utterance generated at once.
2. **Host-side DSP**: SineGen2 + iSTFT (n_fft=16) run on host (no native `ttnn.istft`).

## Architecture

```
Text → Frontend (host: normalize, tokenize, speech_tokenizer_v2.onnx, campplus.onnx)
     → LLM (N300: Qwen2.5-0.5B + speech heads + traced decode, 113.6 tok/s)
     → Flow (N300: UNet1D estimator × 6 NFE + host: encoder + Euler CFM)
     → Vocoder (host: HiFTGenerator conv stack + SineGen2 + iSTFT)
     → 24 kHz waveform
```

## Performance Optimization

| Change | Result |
|--------|--------|
| `ttnn.begin_trace_capture` on decode loop | 34 → **113.6 tok/s** (3.3×) |
|Flow estimator → device  Native TTNN UNet1D (`estimator_ttnn.py`, 910 weights) | 19.2s → **4.4s** (4.3×) |
|Pad caching, conv weight caching, scale folding, trace scaffold | Eliminated 140+ redundant ops/inference |
|NFE reduction 10→6 `CausalConditionalCFM(n_timesteps=6)` | 4.4s → **2.7s**, RTF 0.78 → **0.61** |

**E2E RTF: 0.61** (was ~3.4 at Stage 1). 40 PCC tests + 20 demo WAVs pass.


## References

- [CosyVoice2 paper](https://arxiv.org/abs/2412.10422)
- [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) @ `074ca6d`
- Checkpoint: `FunAudioLLM/CosyVoice2-0.5B` (HuggingFace, rev `eec1ae6c`)
- Full bring-up plan: `BRINGUP_PLAN.md`
