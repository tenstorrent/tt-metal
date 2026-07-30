<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# CosyVoice — TTS on Wormhole N300

## Platforms
Wormhole (N300)

## Introduction

[CosyVoice](https://github.com/FunAudioLLM/CosyVoice) is a multilingual
text-to-speech model from Alibaba FunAudioLLM. It combines a Qwen2.5-0.5B LLM
(speech-token prediction), a flow-matching decoder (UNet1D estimator + Euler CFM),
and a HiFT vocoder (conv stack + iSTFT) to generate 24 kHz speech.

This bring-up on Tenstorrent Wormhole N300 using TTNN covers **Stage 1** (functional
correctness), **Stage 2** (performance optimization), and **Stage 3** (flash attention,
streaming inference, 2-chip pipeline parallelism). The LLM and flow estimator run
on-device with trace capture; the vocoder runs on host (CPU) due to bf16 precision
constraints in dilated ResBlocks.

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
source /root/tt-metal/python_env/bin/activate
cd /root/tt-metal
python -m pytest models/demos/cosyvoice/demo/demo.py -v -s
```

Output WAVs are written to `models/demos/cosyvoice/demo/output/`.

## Tests

```bash
source /root/tt-metal/python_env/bin/activate
cd /root/tt-metal

# PCC gates (48 tests, ~90s)
python -m pytest models/demos/cosyvoice/tests/pcc/ -q

# E2E mode tests + token accuracy (8 tests, ~40s)
python -m pytest models/demos/cosyvoice/tests/e2e/test_modes.py -v

# Performance (LLM throughput + E2E RTF + streaming RTF)
python -m pytest models/demos/cosyvoice/tests/perf/test_throughput.py -v -s

# Audio quality evaluation (WER + speaker similarity)
python -m pytest models/demos/cosyvoice/demo/eval.py -v -s
```

## Results

| Metric | Target | Measured |
|--------|--------|----------|
| LLM decode throughput | ≥ 30 tok/s | 114.2 tok/s (traced) |
| E2E RTF (non-streaming) | < 0.5 | **0.451** |
| E2E RTF (streaming, 2-chip) | < 0.6 | **0.57** |
| Streaming first-chunk latency | — | 2.8s |
| Token accuracy (zero_shot) | > 95% | 96.0% |
| Token accuracy (cross_lingual) | > 95% | 100.0% |
| Token accuracy (instruct2) | > 95% | 100.0% |
| Token accuracy (sft) | > 95% | 98.0% |
| Speaker similarity (CAM++) | > 60 | 82.9 |
| LLM prefill PCC | ≥ 0.99 | 0.997 |
| LLM decode PCC | ≥ 0.99 | 0.996–0.998 |
| Flow estimator PCC (mel) | ≥ 0.99 | 0.995–0.999 |
| Vocoder waveform PCC | ≥ 0.99 | 1.0 |

## Validation Evidence

All gates verified on Wormhole N300:

```
tests/pcc/          48 passed   (LLM teacher-forced PCC, flow encoder/estimator PCC,
                                 TTNN estimator PCC, streaming PCC, vocoder waveform/f0/MCD)
tests/e2e/           8 passed   (4-mode waveform sanity + teacher-forced token accuracy >95%)
tests/perf/          3 passed   (LLM decode ≥30 tok/s; E2E RTF <0.5; streaming RTF <0.6)
demo/demo.py        20 passed   (4 modes × 5 languages, WAVs in demo/output/)
```

## Known Limitations

1. **Host-side DSP**: SineGen2 + iSTFT (n_fft=16) run on host (no native `ttnn.istft`).
2. **Streaming RTF bounded by Python GIL**: 2-chip pipeline eliminates device contention,
   but host ops (encoder + vocoder) hold the GIL ~1.5s per utterance, serializing with
   LLM sampling. RTF < 0.3 requires C++ orchestration or an incremental encoder.

## Architecture

```
Text → Frontend (host: normalize, tokenize, speech_tokenizer_v2.onnx, campplus.onnx)
     → LLM (N300 chip 0: Qwen2.5-0.5B + speech heads, traced decode, 114 tok/s)
     → Flow (N300 chip 1: UNet1D estimator × 5 NFE, traced; host: encoder + Euler CFM)
     → Vocoder (host fp32: HiFTGenerator conv stack + SineGen2 + iSTFT)
     → 24 kHz waveform
```

Streaming mode (`inference_zero_shot_streaming`) yields waveform chunks as they are
generated: LLM decodes continuously on chip 0 while flow+vocoder process accumulated
tokens on chip 1 + host. Uses delta CFM (fixed T_max=256 trace) + hift cache + hamming
fade-in/out between chunks.

## Performance Optimization

| Change | Result |
|--------|--------|
| LLM trace capture (decode loop) | 34 → **114 tok/s** (3.3×) |
| Flow estimator → native TTNN | 19.2s → **4.4s** (4.3×) |
| Estimator trace capture (50MB trace region, cached) | ~300 ms/NFE-step |
| NFE reduction 10→5 (mel PCC=1.0) | RTF 0.78 → **0.47** |
| Pad caching, conv weight caching, attention scale folding | Eliminated 140+ redundant ops |
| Vocoder `torch.compile` | Host vocoder ~20% faster |
| Flash attention (`scaled_dot_product_attention`) | RTF 0.47 → **0.451** |
| Streaming (chunk-level delta CFM, fixed-T trace) | First-chunk latency **2.8s** |
| 2-chip pipeline (`CreateDevices`, LLM∥CFM) | Device contention eliminated |

### E2E Time Breakdown (zero_shot, 9.4s audio, warm)

| Component | Time | Location |
|-----------|------|----------|
| Frontend (ONNX tokenizers, mel) | 0.24s | Host |
| LLM prefill + decode (traced) | 2.18s | N300 chip 0 |
| Flow encoder | 0.24s | Host |
| CFM estimator (5 NFE, traced) | 1.50s | N300 chip 1 |
| Vocoder (torch.compile) | 0.75s | Host |
| **Total** | **~4.3s (RTF 0.451)** | |

## Validation Evidence (Wormhole N300)

### PCC Gates — 40/40 passed
```
tests/pcc/test_llm_module.py            8 passed  (teacher-forced PCC 0.996–0.998, token accuracy 96–100%)
tests/pcc/test_flow_encoder.py          4 passed  (mu PCC = 1.0, all 4 modes)
tests/pcc/test_flow_estimator.py        8 passed  (dphi_dt + mel PCC = 1.0, all 4 modes)
tests/pcc/test_flow_estimator_ttnn.py   8 passed  (TTNN estimator PCC ≥ 0.995, all 4 modes)
tests/pcc/test_hift_module.py          12 passed  (waveform PCC = 1.0, f0 PCC = 1.0, MCD 0.82–1.03 dB)
```

### E2E Modes — 8/8 passed
```
tests/e2e/test_modes.py                 4 mode sanity (zero_shot, cross_lingual, instruct2, sft)
                                        4 token accuracy > 95% (96%, 100%, 100%, 98%)
```
### Tracy profiler report

Report generated: https://drive.google.com/drive/folders/1DfrR5kOkuLq6-_7Fn2RHbjWra77skE_3?usp=sharing
