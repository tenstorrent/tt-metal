# CosyVoice2-0.5B — TTS on Wormhole N300

## Platforms
Wormhole (N300)

## Introduction

[CosyVoice2-0.5B](https://github.com/FunAudioLLM/CosyVoice) is a multilingual
text-to-speech model from Alibaba FunAudioLLM. It combines a Qwen2.5-0.5B LLM
(speech-token prediction), a flow-matching decoder (UNet1D estimator + Euler CFM),
and a HiFT vocoder (conv stack + iSTFT) to generate 24 kHz speech.

This is a **Stage-1 (functional correctness)** bring-up on Tenstorrent Wormhole N300
using TTNN. The LLM runs on-device; flow and vocoder run on host (CPU).

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

### Per-Mode Examples

```python
import ttnn
from models.demos.cosyvoice.tt.pipeline import TtnnCosyVoice

device = ttnn.open_device(device_id=0, l1_small_size=64 * 1024)
pipe = TtnnCosyVoice(device, model_dir="models/demos/cosyvoice/model_data/cosyvoice2-0.5B")

# Zero-shot voice cloning
pipe.add_zero_shot_spk("希望你以后能够做的比我还好呦。", "path/to/prompt.wav", "my_spk")
wav = pipe.inference_zero_shot("你好世界", "希望你以后能够做的比我还好呦。", "path/to/prompt.wav")

# Cross-lingual
wav = pipe.inference_cross_lingual("Hello world", "path/to/prompt.wav")

# Instruct (style control)
wav = pipe.inference_instruct2("你好世界", "用四川话说这句话<|endofprompt|>", "path/to/prompt.wav")

# SFT (pre-registered speaker)
wav = pipe.inference_sft("你好世界", "my_spk")

ttnn.close_device(device)
```

## Tests

```bash
cd /root/tt-metal

# PCC gates (32 tests, ~75s)
python -m pytest models/demos/cosyvoice/tests/pcc/ -q

# E2E mode tests + token accuracy (8 tests, ~100s)
python -m pytest models/demos/cosyvoice/tests/e2e/test_modes.py -v

# Performance (C6 throughput)
python -m pytest models/demos/cosyvoice/tests/perf/test_throughput.py -v -s

# Audio quality evaluation (C8: WER + speaker similarity)
python -m pytest models/demos/cosyvoice/demo/eval.py -v -s
```

## Results (C6–C8)

| Metric | Target | Measured | Status |
|--------|--------|----------|--------|
| LLM decode throughput | ≥ 30 tok/s | 34.1 tok/s | ✓ PASS |
| E2E RTF | < 0.5 | 2.17 | ✗ Stage-2 (flow on host) |
| Token accuracy (zero_shot) | > 95% | 96.0% | ✓ PASS |
| Token accuracy (cross_lingual) | > 95% | 100.0% | ✓ PASS |
| Token accuracy (instruct2) | > 95% | 100.0% | ✓ PASS |
| Token accuracy (sft) | > 95% | 98.0% | ✓ PASS |
| ASR WER (whisper-large-v3) | < 3.0 | 0.000 | ✓ PASS |
| Speaker similarity (CAM++) | > 60 | 82.9 | ✓ PASS |
| LLM prefill PCC | ≥ 0.99 | 0.997 | ✓ PASS |
| LLM decode PCC | ≥ 0.99 | 0.996–0.998 | ✓ PASS |
| Flow PCC (mu/mel) | ≥ 0.99 | 1.0 | ✓ PASS |
| Vocoder waveform PCC | ≥ 0.99 | 1.0 | ✓ PASS |

## Known Limitations (Stage 1)

1. **RTF > 0.5**: The flow estimator (UNet1D, 12 mid blocks, 256-ch, 10 NFE × CFG
   batch=2) runs on host CPU and dominates inference time (~16s for 10s audio).
   Achieving RTF < 0.5 requires moving flow to device (Stage 2).
2. **Non-streaming only**: No bi-directional streaming; full utterance generated at once.
3. **Batch=1**: No multi-utterance batching.
4. **Host-side DSP**: SineGen2 + iSTFT (n_fft=16) run on host (no native `ttnn.istft`).
5. **Host-side sampling**: RAS sampling runs on host CPU (not on-device).

## Architecture

```
Text → Frontend (host: normalize, tokenize, speech_tokenizer_v2.onnx, campplus.onnx)
     → LLM (N300: Qwen2.5-0.5B + speech heads + RAS sampling)
     → Flow (host: UpsampleConformerEncoder + UNet1D estimator + Euler CFM)
     → Vocoder (host: HiFTGenerator conv stack + SineGen2 + iSTFT)
     → 24 kHz waveform
```

## Stage-2 Roadmap

- **trace + 2CQ**: Capture LLM decode as a trace for zero-overhead dispatch.
- **Flow on device**: Port UNet1D estimator to TTNN (conv1d/conv_transpose1d validated).
- **Bi-streaming**: Chunk-level flow + vocoder for low-latency streaming.
- **Batching**: Multi-utterance batch for throughput.
- **On-device sampling**: Move RAS sampling to device (eliminate host roundtrip).
- **Aux models on device**: speech_tokenizer_v2, campplus on N300.
- **2nd N300 chip**: Pipeline parallelism (LLM on chip 0, flow+vocoder on chip 1).

## References

- [CosyVoice2 paper](https://arxiv.org/abs/2412.10422)
- [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) @ `074ca6d`
- Checkpoint: `FunAudioLLM/CosyVoice2-0.5B` (HuggingFace, rev `eec1ae6c`)
- Full bring-up plan: `BRINGUP_PLAN.md`
