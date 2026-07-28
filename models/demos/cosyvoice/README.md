<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# CosyVoice2-0.5B — TTS on Wormhole N300

## Platforms
Wormhole (N300)

## Introduction

[CosyVoice2-0.5B](https://github.com/FunAudioLLM/CosyVoice) is a multilingual
text-to-speech model from Alibaba FunAudioLLM. It combines a Qwen2.5-0.5B LLM
(speech-token prediction), a flow-matching decoder (UNet1D estimator + Euler CFM),
and a HiFT vocoder (conv stack + iSTFT) to generate 24 kHz speech.

This bring-up on Tenstorrent Wormhole N300 using TTNN covers **Stage 1 (functional
correctness)** and **Stage 2 (performance optimization)**. The LLM and flow estimator
run on-device with trace capture; the vocoder runs on host (CPU) due to bf16 precision
constraints.

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

# E2E mode tests + token accuracy (8 tests, ~40s)
python -m pytest models/demos/cosyvoice/tests/e2e/test_modes.py -v

# Performance (C6 throughput + RTF)
python -m pytest models/demos/cosyvoice/tests/perf/test_throughput.py -v -s

# Audio quality evaluation (C8: WER + speaker similarity)
python -m pytest models/demos/cosyvoice/demo/eval.py -v -s
```

## Results

| Metric | Target | Measured | Status |
|--------|--------|----------|--------|
| LLM decode throughput | ≥ 30 tok/s | 114.2 tok/s (traced) | ✓ PASS |
| E2E RTF | < 0.5 | **0.47** | ✓ PASS |
| Token accuracy (zero_shot) | > 95% | 96.0% | ✓ PASS |
| Token accuracy (cross_lingual) | > 95% | 100.0% | ✓ PASS |
| Token accuracy (instruct2) | > 95% | 100.0% | ✓ PASS |
| Token accuracy (sft) | > 95% | 98.0% | ✓ PASS |
| WER (whisper-large-v3) | < 3.0 | 0.000 | ✓ PASS |
| Speaker similarity (CAM++) | > 60 | 82.9 | ✓ PASS |
| LLM prefill PCC | ≥ 0.99 | 0.997 | ✓ PASS |
| LLM decode PCC | ≥ 0.99 | 0.996–0.998 | ✓ PASS |
| Flow estimator PCC (mel) | ≥ 0.99 | 0.995–0.999 | ✓ PASS |
| Vocoder waveform PCC | ≥ 0.99 | 1.0 | ✓ PASS |

## Validation Evidence

All gates verified on Wormhole N300:

```
tests/pcc/          40 passed   (LLM teacher-forced PCC, flow encoder/estimator PCC,
                                 TTNN estimator PCC, vocoder waveform/f0/MCD — 4 modes each)
tests/e2e/           8 passed   (4-mode waveform sanity + teacher-forced token accuracy >95%)
tests/perf/          2 passed   (LLM decode 114.2 tok/s ≥ 30; E2E RTF 0.47 < 0.5)
demo/demo.py        20 passed   (4 modes × 5 languages, WAVs in demo/output/)
demo/eval.py         2 passed   (WER 0.000, speaker similarity 82.9)
```

Device performance data: `demo/output/device_perf_report.csv` (1746 device ops,
per-op kernel durations, core counts — generated via `tools/tracy/profile_this.py`).

## Known Limitations

1. **Non-streaming only**: Non-streaming is the Stage-1 design constraint. Streaming
   requires causal flow attention + chunk-level LLM→flow→vocoder pipelining (Stage 3;
   see BRINGUP_PLAN.md §12.3.6).
2. **Host-side vocoder**: The HiFT vocoder's dilated ResBlocks (effective kernel up to
   51 taps) accumulate ~10% std scale error per upsample stage in bf16, catastrophically
   amplified by `exp()` in the magnitude computation. The vocoder must stay on host
   (fp32) until TTNN supports fp32 conv accumulation.
3. **Host-side DSP**: SineGen2 + iSTFT (n_fft=16) run on host (no native `ttnn.istft`).

## Architecture

```
Text → Frontend (host: normalize, tokenize, speech_tokenizer_v2.onnx, campplus.onnx)
     → LLM (N300: Qwen2.5-0.5B + speech heads, traced decode, 114 tok/s)
     → Flow (N300: UNet1D estimator × 5 NFE, traced; host: encoder + Euler CFM)
     → Vocoder (host fp32: HiFTGenerator conv stack + SineGen2 + iSTFT)
     → 24 kHz waveform
```

## Performance Optimization

| Change | Result |
|--------|--------|
| LLM trace capture (`ttnn.begin_trace_capture` on decode loop) | 34 → **114 tok/s** (3.3×) |
| Flow estimator → native TTNN (`estimator_ttnn.py`) | 19.2s → **4.4s** (4.3×) |
| Estimator trace capture (50MB trace region, cached across calls) | 444 → **~300 ms/NFE-step** |
| NFE reduction 10→5 (mel PCC=1.0 validated) | RTF 0.78 → **0.47** |
| Pad caching, conv weight caching, attention scale folding | Eliminated 140+ redundant ops/inference |
| Vocoder `torch.compile` (reduce-overhead, dynamic) | Host vocoder ~20% faster |

**E2E RTF: 0.47** (was ~3.4 at Stage 1 baseline). 40 PCC tests + 20 demo WAVs pass.

### E2E Time Breakdown (zero_shot, 9.4s audio, warm)

| Component | Time |
|-----------|------|
| Frontend (ONNX tokenizers, mel) | 0.24s |
| LLM prefill + decode (traced) | 2.18s |
| Flow encoder | 0.24s | Host |
| CFM estimator (5 NFE, traced) | 1.50s |
| Vocoder (torch.compile) | 0.75s |
| **Total** | **~4.4s (RTF 0.47)** |

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

Report on ttnn-visualiser:


https://drive.google.com/drive/folders/1DfrR5kOkuLq6-_7Fn2RHbjWra77skE_3?usp=drive_link
