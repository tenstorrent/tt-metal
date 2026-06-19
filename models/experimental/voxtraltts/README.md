# Voxtral TTS — Documentation

## 1. Introduction

Voxtral TTS (`mistralai/Voxtral-4B-TTS-2603`) is Mistral AI's open-weights text-to-speech model for production voice agents. It produces realistic, expressive speech in nine languages (English, French, Spanish, Portuguese, Italian, Dutch, German, Arabic, and Hindi) at 24 kHz.

This directory contains the Tenstorrent TTNN bring-up of Voxtral TTS. The main neural-network stages have TTNN device implementations:

- **Text backbone** — 26-layer, 3072-dim LLM that decodes interleaved text and acoustic tokens (GQA: 32 heads / 8 KV heads, vocab 131072)
- **Acoustic transformer (Flow Matching head)** — 3-layer flow-matching refinement, same hidden size, produces continuous latents at 12.5 frames/s
- **Audio tokenizer decoder** — 1024-dim transformer that maps latents to 24 kHz waveform samples (8 heads, 36-codebook residual VQ output)

The host handles tokenization, voice-embedding preparation, and lightweight generation control/staging. The autoregressive decode path uses TTNN execution, with trace and 2-CQ async dispatch available for the text/acoustic loop.

---

## 2. Installation

### Step 1 — Activate environment

Run these commands at the start of every session, from the repo root:

```bash
source python_env/bin/activate
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
export ARCH_NAME=blackhole_140_arch_eth_dispatch.yaml
```

> `TT_METAL_HOME` and `PYTHONPATH` must point to the repo root. Without them, `import ttnn` fails and all tests fail with confusing import errors.

### Step 2 — Install Python packages

Use `python_env/bin/python -m pip install` for all installs. The system `pip` does not install into the project virtualenv.

**Core inference packages** — tokenizer, weight loading, audio output (from the repo root):

```bash
python_env/bin/python -m pip install -r models/experimental/voxtraltts/requirements.txt
```

**Quality metric packages** — required for `tests/pcc/test_voxtral_e2e_quality_metrics.py`:

```bash
python_env/bin/python -m pip install utmosv2 speechbrain
```

> No specific version is required for `utmosv2` or `speechbrain`. The code loads them via `importlib` and guards each test with `@pytest.mark.skipif(not _has_module(...))`, so missing them skips the quality-metric tests rather than crashing.

### Step 3 — Set model weights path

**Option A — HuggingFace repo ID (auto-download):**

```bash
export VOXTRAL_TTS_MODEL=mistralai/Voxtral-4B-TTS-2603
```

Weights are downloaded on first use and cached in `~/.cache/huggingface/`.

**Option B — Local directory:**

```bash
export VOXTRAL_TTS_MODEL=/path/to/local/weights
```

The directory must contain the `.safetensors` files. No download occurs.

---

## 3. Supported Devices

| Device | Status | Notes |
|--------|--------|-------|
| **P150** | Fully supported | PCIe single-card; primary perf target |
| **Blackhole Quiet Box 2 (BH QB2)** | Fully supported | Chassis with 4 × P150 chips; same arch as single P150 |

---

## 4. File Structure

```
models/experimental/voxtraltts/
├── requirements.txt             #   Core Python deps (Step 2 install)
├── demo/
│   ├── decode_trace_2cq.py      #   Trace + 2-CQ decode helpers
│   └── demo.py                  #   Full TT demo: text → WAV (device-resident)
├── reference/                   # CPU-only PyTorch reference
│   ├── audio_tokenizer_ops.py   #   Audio tokenizer decode ops (CPU reference)
│   ├── cpu_flow_matching_acoustic.py  #   CPU reference for the acoustic FM transformer
│   ├── cpu_reference.py         #   Full CPU inference pipeline (VoxtralCPUReference)
│   ├── demo_reference.py        #   CPU reference demo CLI
│   ├── functional.py            #   Layer-level functions (RMSNorm, attention, MLP, …)
│   ├── generate_golden.py       #   Generates golden tensors for unit tests
│   ├── golden/                  #   Committed golden tensors for unit tests
│   │   ├── acoustic_attention_golden.pt
│   │   ├── acoustic_layer_golden.pt
│   │   ├── rms_norm_golden.pt
│   │   ├── swiglu_mlp_golden.pt
│   │   ├── text_attention_golden.pt
│   │   └── text_decoder_layer_golden.pt
│   ├── voxtral_config.py        #   Model config dataclasses + weight loading
│   └── voxtral_request.py       #   Tokenizer + SpeechRequest construction (mistral-common)
├── tests/                       # All tests (unit, PCC, perf)
│   ├── audio_tokenizer_workload.py       #   Audio tokenizer test workload helpers
│   ├── common.py                         #   Shared device/model setup utilities
│   ├── conftest.py                       #   Pytest fixtures
│   ├── generate_voxtral_golden_codes.py  #   Script to regenerate committed golden codes
│   ├── pcc/                     #   E2E waveform PCC + quality-metric tests
│   │   ├── test_ttnn_voxtral_unittest.py        #   TTNN-level unit tests
│   │   ├── test_voxtral_e2e_debug.py            #   Free-run diagnostic (informational)
│   │   ├── test_voxtral_e2e_pcc.py              #   Teacher-forced + free-run waveform PCC
│   │   └── test_voxtral_e2e_quality_metrics.py  #   UTMOS / WER / speaker similarity
│   ├── perf/                    #   Wall-clock and device-perf tests
│   │   ├── test_e2e_performant.py                       #   E2E wall-clock (trace-enabled)
│   │   ├── test_profile_single_layer_prefill_decode.py  #   Single-layer profiling
│   │   ├── test_single_layer_device_perf.py             #   Single-layer device perf
│   │   ├── test_voxtral_tts_device_perf.py              #   Full model device perf
│   │   ├── test_voxtral_tts_perf_inference.py           #   Inference throughput perf
│   │   ├── test_voxtral_tts_stage_device_perf.py        #   Per-stage device perf
│   │   └── test_voxtral_tts_stage_perf_run.py           #   Per-stage perf run
│   ├── reference_outputs/       #   Committed reference outputs (voxtral_golden_codes.refpt)
│   ├── test_acoustic_model.py                      #   Acoustic model PCC tests (Euler steps, FM layers)
│   ├── test_attention.py                           #   Attention unit tests
│   ├── test_audio_tokenizer_*.py                   #   Audio tokenizer component unit tests
│   ├── test_mlp.py                                 #   MLP unit tests
│   ├── test_rmsnorm.py                             #   RMSNorm unit tests
│   ├── test_text_decoder_layer.py                  #   Text decoder layer tests
│   ├── test_text_decoder_layer_pcc.py              #   Text decoder layer PCC tests
│   ├── test_text_model.py                          #   Text model unit + decode PCC tests
│   └── test_voxtral_tts_pipeline_component_pcc.py #   Pipeline component PCC
├── tt/                          # TTNN on-device implementations
│   ├── acoustic_model.py        #   Acoustic flow-matching head
│   ├── attention.py             #   Grouped-query attention (prefill + decode)
│   ├── audio_tokenizer/         #   Audio tokenizer decode (encoder + decoder stack)
│   ├── mlp.py                   #   SwiGLU MLP
│   ├── rmsnorm.py               #   RMSNorm
│   ├── text_decoder_layer.py    #   Single text decoder layer
│   ├── text_model.py            #   Text backbone (Transformer)
│   ├── voxtral_tt_args.py       #   Model args, program configs, optimisation presets
│   └── voxtral_tts.py           #   VoxtralTTSPipeline (top-level inference entry point)
└── utils/
    ├── audio_tokenizer_optimizations.py  #   Optimisation preset factories
    ├── config_helpers.py                 #   Compute kernel configs (acoustic, semantic, …)
    ├── debug_trace.py                    #   Debug and trace utilities
    └── rng.py                            #   RNG helpers for FM noise
```

---

## 5. Tests

All tests require the environment from [Section 2](#2-installation) and `VOXTRAL_TTS_MODEL` set.

### 5.1 Unit tests

Single-module correctness checks (shape, finite outputs, basic model config). Run quickly; do not require a full model download for most cases.

```bash
# Text model — shape + config
pytest models/experimental/voxtraltts/tests/test_text_model.py::test_text_model_inference -sv

# Text model — prefill forward pass (finite output)
pytest models/experimental/voxtraltts/tests/test_text_model.py::test_text_model_prefill_inference -sv

# Audio tokenizer components
pytest models/experimental/voxtraltts/tests/test_audio_tokenizer_decoder_transformer_block.py -sv
pytest models/experimental/voxtraltts/tests/test_audio_tokenizer_full_decode.py -sv

# Acoustic model components
pytest models/experimental/voxtraltts/tests/test_acoustic_model.py::test_acoustic_predict_velocity_pcc -sv
pytest models/experimental/voxtraltts/tests/test_acoustic_model.py::test_acoustic_semantic_logits_pcc -sv
pytest models/experimental/voxtraltts/tests/test_acoustic_model.py::test_acoustic_forward_matches_cpu_reference -sv

# MLP / RMSNorm / attention primitives
pytest models/experimental/voxtraltts/tests/test_mlp.py -sv
pytest models/experimental/voxtraltts/tests/test_rmsnorm.py -sv
pytest models/experimental/voxtraltts/tests/test_attention.py -sv

# Run all unit tests at once
pytest models/experimental/voxtraltts/tests/ \
    --ignore=models/experimental/voxtraltts/tests/pcc \
    --ignore=models/experimental/voxtraltts/tests/perf -sv
```

### 5.2 PCC / accuracy tests

These tests compare TT hardware output against a float32 CPU reference and assert that Pearson Correlation Coefficient (PCC) is above a threshold. PCC of 1.0 means perfect numerical match; ≥ 0.99 is the standard pass threshold.

#### Text model decode PCC

```bash
# Single decode step vs float32 CPU reference
pytest models/experimental/voxtraltts/tests/test_text_model.py::test_text_model_decode_reference_pcc -sv

# Multi-step decode (26 steps) vs float32 CPU reference
pytest models/experimental/voxtraltts/tests/test_text_model.py::test_text_model_decode_multistep_reference_pcc -sv

# Prefill logits PCC
pytest models/experimental/voxtraltts/tests/test_text_model.py::test_text_model_prefill_pcc -sv
```
#### Acoustic model PCC

```bash
# Euler step sampled state + final pre-round scaled values PCC vs CPU
pytest models/experimental/voxtraltts/tests/test_acoustic_model.py::test_acoustic_decode_euler_stepwise_pcc -sv

# Attention + MLP PCC for every FM layer
pytest models/experimental/voxtraltts/tests/test_acoustic_model.py::test_acoustic_all_layers_attention_mlp_pcc -sv
```

#### E2E waveform PCC (teacher-forced + free-run)

Run all four tests in one invocation:

```bash
pytest models/experimental/voxtraltts/tests/pcc/test_voxtral_e2e_pcc.py -sv --timeout=0
```

**Teacher-forced tests** , in `tests/pcc/test_voxtral_e2e_pcc.py`:
- `test_ttnn_voxtral_tts_golden_codes_pcc`
- `test_ttnn_voxtral_tts_acoustic_pcc`
- `test_ttnn_voxtral_tts_golden_acoustic_pcc`

**Free-run diagnostic** , in `tests/pcc/test_voxtral_e2e_pcc.py`:
- `test_ttnn_voxtral_tts_staged_pcc`

#### E2E quality metrics (UTMOS, WER, speaker similarity)

These metrics are adopted from the Voxtral paper's evaluation protocol:
- **UTMOS** (MOS predictor) — measures perceived naturalness of the generated speech without needing human listeners.
- **WER** (Word Error Rate via Whisper) — verifies that the synthesised speech is intelligible and the correct words were produced. Whisper is used instead of Voxtral's own transcription because Voxtral transcribe requires a Mistral API key, whereas Whisper is open-source and requires no API key.
- **Speaker similarity** (SpeechBrain ECAPA-TDNN cosine score) — checks that the output voice matches the requested speaker embedding, confirming voice identity is preserved end-to-end.

Requires `utmosv2`, `speechbrain`, and `transformers` with Whisper. If these packages are not installed, the individual sub-tests are skipped automatically.

```bash
pytest models/experimental/voxtraltts/tests/pcc/test_voxtral_e2e_quality_metrics.py -sv --timeout=0
```

### 5.3 Performance tests

#### Wall-clock E2E (frames/s, TTFA) — trace-enabled; 2CQ opt-in

```bash
pytest models/experimental/voxtraltts/tests/perf/test_e2e_performant.py -sv --timeout=0
```

Reports: per-frame decode time (ms/frame), throughput (frames/s), and time-to-first-audio (TTFA ms).

#### full model device perf

```bash
pytest models/experimental/voxtraltts/tests/perf/test_voxtral_tts_device_perf.py -sv --timeout=0
```


#### Modulewise perf test (per-stage device profiling)

```bash
# Text prefill
pytest models/experimental/voxtraltts/tests/perf/test_voxtral_tts_stage_device_perf.py -sv -k text_prefill

# Text decode
pytest models/experimental/voxtraltts/tests/perf/test_voxtral_tts_stage_device_perf.py -sv -k text_decode

# Acoustic forward (semantic head + flow-matching)
pytest models/experimental/voxtraltts/tests/perf/test_voxtral_tts_stage_device_perf.py -sv -k acoustic_forward

# Audio decode (tokenizer latent → mel → waveform)
pytest models/experimental/voxtraltts/tests/perf/test_voxtral_tts_stage_device_perf.py -sv -k audio_decode

# End to end
pytest models/experimental/voxtraltts/tests/perf/test_voxtral_tts_device_perf.py
```

### 5.4 Demo

```bash
# Full TT demo — text in, WAV file out
python models/experimental/voxtraltts/demo/demo.py \
    --text "Paris is a beautiful city in the heart of Europe." \
    --output-dir /tmp/voxtral_out

# Using a JSON prompts file
python models/experimental/voxtraltts/demo/demo.py \
    --prompts models/experimental/voxtraltts/demo/data/sample_prompts.json \
    --output-dir /tmp/voxtral_out

# CPU reference demo (no TT hardware required — runs on host CPU only)
python -m models.experimental.voxtraltts.reference.demo_reference \
    --model mistralai/Voxtral-4B-TTS-2603 \
    --text "Paris is a beautiful city!" \
    --voice casual_male --write-audio
```

---

## 6. Performnce and Accuracy

### 6.1 PCC targets (accuracy)

| Test | Target | Measured | Notes |
|:-----|:------:|:--------:|:------|
| Text prefill logits | ≥ 0.99 | | BF16 weights + HiFi4 |
| Text decode (1 step) | ≥ 0.99 | | prompt_len clamped to ≤ 384 (L1 limit) |
| Text decode (26 steps) | ≥ 0.98 | | ~1–2% BF16 drift accumulates at step 19+ |
| Audio tokenizer decoder | ≥ 0.99 | | Dense ALiBi SDPA path |
| Pipeline component (prefill hidden) | ≥ 0.99 | | |
| Pipeline component (decode step) | ≥ 0.99 | | |
| E2E waveform (teacher-forced) | ≥ 0.99 | | Shared codes; gated in CI |
| E2E waveform (free-run) | ~0.957 | | North-star metric; logged only, not gated |

### 6.2 Quality metrics

| Metric | Tool | Target | Notes |
|:-------|:----:|:------:|:------|
| MOS (naturalness) | UTMOS-v2 | ≥ 3.0 | Override: `VOXTRAL_TTS_UTMOS_V2_MIN_SCORE` |
| Word Error Rate | Whisper Small | < 30% | Override: `VOXTRAL_TTS_WER_TARGET`; `openai/whisper-small` is used (Voxtral transcribe requires Mistral API key) |
| Speaker similarity | SpeechBrain ECAPA-TDNN | ≥ 0.55 cosine | Verified against reference voice embedding |

> Audio is produced at 12.5 acoustic frames/s at 24 kHz. Real-time factor = `frames_per_s / 12.5`. A value > 1.0 means faster than real time.

### 6.3 Accuracy Verification

| ISL (text tokens / audio tokens) | PCC (teacher-forced) | PCC (free-run — single /multi layer decoder) | WER | UTMOS | Speaker Similarity |
|:----------------------------------|:--------------------:|:-------------------------------:|:---:|:-----:|:-----------------:|
| | | | | | |

### 6.4 Performance Verification

#### P150 (1×1)

| ISL (text tokens / audio tokens) | Latency (ms) | RTF | Throughput (char/s) |
|:----------------------------------|-------------:|----:|--------------------:|
| 513 chars / 377 audio tokens | 21407 | 0.7098 | 23.96 |

#### BH QB2 (1×4)

| ISL (text tokens / audio tokens) | Latency (ms) | RTF | Throughput (char/s) |
|:----------------------------------|-------------:|----:|--------------------:|
| | | | |

### 6.5 Demo Verification

#### Configurable parameters

| Parameter | Default | Description |
|---|---|---|
| `--text TEXT` | — | Inline text prompt; repeat for multiple prompts.|
| `--voice NAME` | — | Override voice for all prompts in this run. If unset, per-item voice from the JSON is used, falling back to `--default-voice`. |
| `--default-voice NAME` | `casual_male` | Fallback voice applied to any prompt that does not specify one. |
| `--output-dir DIR` | `/tmp/voxtral_out` | Directory where output `.wav` files are written. |


#### Default voice

`casual_male` — best general-purpose output for English text.

#### Supported built-in voices (HF `.pt` presets)

These files ship inside `voice_embedding/` in the model weights and are loaded automatically by name.

| Voice name | Language / style |
|---|---|
| `casual_male` | English — casual male *(default)*|
| `casual_female` | English — casual female |
| `neutral_male` | English — neutral male |
| `neutral_female` | English — neutral female |
| `cheerful_female` | English — cheerful female |
| `fr_male` / `fr_female` | French |
| `de_male` / `de_female` | German |
| `es_male` / `es_female` | Spanish |
| `it_male` / `it_female` | Italian |
| `pt_male` / `pt_female` | Portuguese |
| `nl_male` / `nl_female` | Dutch |
| `hi_male` / `hi_female` | Hindi |
| `ar_male` | Arabic |

> Using an English voice for non-English text (or vice-versa) works but may reduce naturalness.

---

## 7. Optimizations

> **Device scope:** All optimizations below are validated on **P150 (1×1) only**. Multi-card P150×4 (BH QB2) optimization is pending.

### Text backbone (P150)

| Optimization | Default | Description |
|---|---|---|
| **BFP8 weights** | On (production) | FF1/FF3/FF2/WQKV/WO/KV-cache stored as BFP8; reduces DRAM bandwidth vs BF16 |
| **HiFi2 decode matmuls** | On (production) | Lower-fidelity matmul for decode MLP/attention; BF16 activations preserved |
| **HiFi4 prefill matmuls** | Always | Full precision for prefill SDPA and matmuls |
| **1D mcast DRAM-sharded MLP** | On | W1/W3 with 1D multicast DRAM-sharded matmul; ~1.36× faster vs 2D on Blackhole |
| **1D mcast DRAM-sharded Wo** | On | Attention output projection; ~1.32× faster vs 2D on Blackhole |
| **Traced AR decode + 2CQ** | On | Autoregressive loop captured as a single TT trace; dispatched over two command queues |

Disable individual optimizations:

```bash
export VOXTRAL_MLP_1D=0        # disable 1D MLP matmul
export VOXTRAL_ATTN_WO_1D=0   # disable 1D Wo matmul
```

Accuracy preset (`voxtral_text_high_accuracy_optimizations`): BF16 weights + HiFi4 everywhere — used in all PCC tests.

### Acoustic model / FM transformer (P150)

| Optimization | Default | Description |
|---|---|---|
| **1D mcast program configs** | On | All FM matmuls (W1/W3, W2, WQKV, WO, projections, velocity head) use 1D multicast on 8×6 = 48-core grid |
| **BFP8 weights** | On | Attention + MLP weights stored as BFP8; override with `VOXTRAL_ACOUSTIC_WEIGHT_DTYPE=bf16` |
| **BFP8 compute kernel** | On | Lower-fidelity matmul for FM attention + MLP; BF16 activations preserved |
| **Width-sharded RMSNorm** | On | Layer norms sharded across 32 cores (8×4); baseline was 2-core L1-interleaved at ~47 µs/call |
| **Batched CFG** | Always | Cond + uncond packed as bsz=2 in a single FM forward pass per Euler step; avoids two separate transformer runs |

### Audio tokenizer decoder (P150)

| Optimization | Default | Description |
|---|---|---|
| **Dense ALiBi SDPA** | On | Full causal + sliding-window ALiBi attention mask; production accuracy |
| **Matmul Tier 1 program configs** | On | Explicit 2D multicast configs per decoder layer; auto-disabled for `T > 6400` to avoid L1 OOM |

```bas
export VOXTRAL_AUDIO_TOKENIZER_MATMUL_PROGCFG_OFF=1   # disable Tier 1 configs
```

---

## 8. Caveats

### TODO

- Reason for choosing teacher-forced vs free-run evaluation
- Why free-run PCC decreases with larger token generation

---



## 8. Implementation Update

- Added full support for the model on **P150**, including performance optimizations.
- Enabled **BH QB2** support, allowing the model to run in a **1×4** mesh configuration.

## 9. Work in progress

- Optimization for **BH QB2** is still in progress.
- During QB2 testing, audible artifacts/noise are observed near the end of each chunk due to the current chunking mechanism. Root-cause analysis is underway, and a fix will be implemented based on the findings.

## 10. References

- **Model card**: [mistralai/Voxtral-4B-TTS-2603](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603)
- **Voxtral tts paper**: [arxiv:2603.25551](https://arxiv.org/pdf/2603.25551)
- **Voxtral paper**: [arxiv:2507.13264](https://arxiv.org/pdf/2507.13264)
- **Flow Matching for Generative Modeling   paper**: [arxiv:2210.02747](https://arxiv.org/abs/2210.02747)
- **Classifier-Free Diffusion Guidance  paper**: [arxiv:2207.12598](https://arxiv.org/abs/2207.12598)

---
