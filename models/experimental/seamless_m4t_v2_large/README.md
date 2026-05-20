# Seamless M4T v2 Large on Tenstorrent Hardware

## Platforms

| Device | Status | Notes |
|--------|--------|-------|
| P100 (Blackhole) | Supported | Single chip, all five tasks, 2CQ overlapped + traced |
| P150 (Blackhole) | Supported | Single chip, all five tasks, 2CQ overlapped + traced |

All performance / E2E pipeline tests are gated with `@run_for_blackhole()`. PCC tests run on any device with sufficient L1 (`l1_small_size=65536`).

---

## Introduction

[SeamlessM4T v2](https://huggingface.co/facebook/seamless-m4t-v2-large) is Meta's unified multilingual and multimodal translation model, introduced in [Seamless: Multilingual Expressive and Streaming Speech Translation](https://ai.meta.com/research/publications/seamless-multilingual-expressive-and-streaming-speech-translation/). A single set of weights covers five inference tasks across ~100 input and ~36 output languages:

| # | Task | Abbrev. | Input → Output |
|---|------|---------|----------------|
| 1 | Text-to-Text Translation | **T2TT** | text → text |
| 2 | Speech-to-Text Translation | **S2TT** | speech → text |
| 3 | Text-to-Speech Translation | **T2ST** | text → 16 kHz speech |
| 4 | Speech-to-Speech Translation | **S2ST** | speech → 16 kHz speech |
| 5 | Automatic Speech Recognition | **ASR** | speech → text (same language) |

This port runs the full HuggingFace `SeamlessM4Tv2Model.generate` pipeline (greedy decode, EOS early-stop, host-side language code lookups) on Tenstorrent hardware via TTNN. Weights are loaded in bfloat16; every tensor on the inference path lives on device.

---

## Model Architecture

A single `SeamlessM4Tv2Model` instance houses five sub-modules. The active path depends on the requested task.

```
                 ┌────────────────────────────┐
text input ─────►│  Text Encoder              │──┐
                 │  (24 layers, 1024 hidden,  │  │
                 │   16 heads, FFN 8192)      │  │
                 └────────────────────────────┘  │
                                                 ▼
                 ┌────────────────────────────┐ ┌────────────────────────────┐
speech input ───►│  Speech Encoder (Conformer)│►│  Text Decoder + lm_head    │──► text tokens
(80-bin mel,     │  (24 layers, 1024 hidden,  │ │  (24 layers, 1024 hidden,  │   (T2TT, S2TT, ASR end here)
 16 kHz)         │   16 heads + adaptor       │ │   16 heads, FFN 8192,      │
                 │   conv stack)              │ │   cross-attn, KV cache)    │
                 └────────────────────────────┘ └────────────────────────────┘
                                                            │
                                          T2ST / S2ST only  │ intermediate text token ids
                                                            ▼
                                                 ┌────────────────────────────┐
                                                 │  T2U (Text-to-Unit)        │
                                                 │  • char encoder (6 layers) │
                                                 │  • duration predictor      │
                                                 │  • unit decoder (6 layers) │
                                                 │  • unit lm_head            │
                                                 └────────────────────────────┘
                                                            │ discrete unit ids
                                                            ▼
                                                 ┌────────────────────────────┐
                                                 │  HiFi-GAN Vocoder          │
                                                 │  (unit / speaker / language│
                                                 │   embeddings + Conv1d      │
                                                 │   resblock stack)          │
                                                 └────────────────────────────┘
                                                            │
                                                            ▼
                                                  16 kHz mono waveform
```

### Key Model Parameters (`facebook/seamless-m4t-v2-large`)

| Parameter | Value |
|-----------|-------|
| Text encoder layers | 24 |
| Text decoder layers | 24 |
| Speech encoder layers (Conformer) | 24 |
| Text encoder/decoder attention heads | 16 |
| Speech encoder attention heads | 16 |
| Hidden size | 1024 |
| FFN dimension (text) | 8192 |
| Speech encoder FFN dimension | 4096 |
| Speech feature input dim | 160 (80 mel × 2 stride) |
| Adaptor conv kernel / stride | 8 / 8 |
| T2U encoder/decoder layers | 6 / 6 |
| Vocabulary size (text) | 256102 |
| Vocoder sample rate | 16 kHz |
| Total parameters | ~2.3 B |
| Weight precision | bfloat16 |

### Per-Task Active Modules

| Task | Text Enc | Speech Enc | Text Dec + lm_head | T2U | Vocoder |
|------|:--------:|:----------:|:------------------:|:---:|:-------:|
| T2TT | yes | — | yes | — | — |
| S2TT | — | yes | yes | — | — |
| T2ST | yes | — | yes | yes | yes |
| S2ST | — | yes | yes | yes | yes |
| ASR | — | yes | yes | — | — |

---

## TTNN Optimizations

### Single-prefill `forward()` parity
Every sub-module (`tt_text_encoder.py`, `tt_speech_encoder.py`, `tt_text_decoder.py`, `tt_text_to_unit.py`, `tt_code_hifigan.py`) is verified at **PCC ≥ 0.99** against the HuggingFace reference at bf16. The end-to-end forward parity tests in `tests/pcc/test_seamless_m4t_v2_model.py` exercise every task at the lm_head logits level (plus T2U logits for the speech-output tasks).

### Speech encoder trace masks
The speech encoder builds large 4D additive masks (encoder self-attention + per-adapter-block self-attention + depth-wise conv left-pad masks). `materialize_speech_encoder_trace_tensors()` constructs every mask **before** trace capture so the traced path replays without host writes.

### `ttnn.conv1d` weight caching
The first speech-encoder and HiFi-GAN call materialises and caches preprocessed conv1d weights on device. Trace replay (which Metal forbids from issuing `write_shard_to_device`) then reuses the cached tensors directly.

### T2U trace pre-allocation
For T2ST / S2ST, T2U inputs, hard-upsample cumulative tensors, and HuggingFace reference discrete durations are materialised before compile via `make_t2u_trace_prealloc_tensors`. The `ttnn.conv1d` weights inside T2U are also warmed once before capture.

### 2-Command-Queue overlap
The E2E pipeline is built on `models.tt_cnn.tt.pipeline.create_pipeline_from_config` with `num_command_queues=2`. Host→device transfers overlap with device execution, and the same wiring drives both the non-traced (`MultiCQModelOverlappedInputExecutor`) and traced E2E paths.

### KV cache
The text decoder uses a pre-allocated KV cache during `generate(use_kv_cache=True)`. Cross-attention K/V are computed once at step 0; self-attention K/V grow per token.

### On-device greedy decode
Per-step `argmax`, position-id update, and additive 4D causal mask construction all happen on device. The only per-step host readback is a single scalar for the EOS early-stop check.

---

## Prerequisites

1. Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal)
2. Installed [TT-Metalium / TT-NN](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
3. Optional HuggingFace login (the `facebook/seamless-m4t-v2-large` repo is public, but a token avoids rate limits):

   ```bash
   huggingface-cli login
   # or
   export HF_TOKEN=<your_token>
   ```

   Token: https://huggingface.co/docs/hub/security-tokens

---

## Environment Setup

Run from the `tt-metal` root directory:

```bash
export ARCH_NAME=blackhole
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
source python_env/bin/activate
```

---

## Weights

### One-time download

The first invocation of any demo or test triggers `ensure_seamless_m4t_v2_large_weights()`, which downloads the HuggingFace snapshot into `models/experimental/seamless_m4t_v2_large/weights/seamless-m4t-v2-large/` (~10 GB). To download explicitly:

```bash
python models/experimental/seamless_m4t_v2_large/scripts/download_weights.py
```

### Custom location

Set the destination directly:

```bash
python models/experimental/seamless_m4t_v2_large/scripts/download_weights.py \
    --destination /path/to/seamless-m4t-v2-large
```

Or point the demo / tests at an existing snapshot via:

```bash
export SEAMLESS_M4T_V2_WEIGHTS=/path/to/seamless-m4t-v2-large
```

---

## How to Run

### Full TTNN demo (all five tasks)

```bash
python models/experimental/seamless_m4t_v2_large/demo/demo.py
```

The demo runs in this self-contained order, with no external audio file required:

1. **T2TT** — English text → Hindi text
2. **T2ST** — English text → Hindi speech (saved to `demo/outputs/t2st_hindi_speech.wav`)
3. **S2TT** — Hindi speech (from task 2) → English text
4. **S2ST** — Hindi speech (from task 2) → Spanish speech (saved to `demo/outputs/s2st_spanish_speech.wav`)
5. **ASR** — Hindi speech (from task 2) → Hindi text

---

## Performance (Blackhole, 2CQ)

### End-to-end pipeline throughput (batch size 1)

Per-task FPS from `tests/perf/test_e2e_perf_2cq.py`:

| Task | `forward()` non-traced | `forward()` traced | `generate()` non-traced |
|------|------------------------|--------------------|--------------------------|
| T2TT | 54.94 | ≥ 54.94 | 0.5 |
| S2TT | 22.92 | ≥ 22.92 | 0.5 |
| T2ST | 29.61 | ≥ 29.61 | 0.25 |
| S2ST | 16.73 | ≥ 16.73 | 0.25 |
| ASR | 22.25 | ≥ 22.25 | 0.5 |

The `forward()` numbers are the `_EXPECTED_E2E_THROUGHPUT_FPS` targets enforced by `test_seamless_m4t_v2_large_e2e_perf_2cq` (non-traced) and shared by `test_seamless_m4t_v2_large_e2e_perf_2cq_trace` (traced). The `generate()` numbers are the loose `_EXPECTED_GENERATE_E2E_THROUGHPUT_FPS` lower bounds from `test_seamless_m4t_v2_large_e2e_perf_2cq_generate` (`max_new_tokens=48`).

### Device-kernel throughput

Raw kernel-only execution time per task, measured by the device profiler in `tests/perf/test_seamless_device_perf.py`. Each parametrization wraps the matching PCC `forward()` under `models.perf.device_perf_utils.run_device_perf` and reports `AVG DEVICE KERNEL SAMPLES/S` plus average kernel duration. Measured on a P150 (Blackhole), single chip:

| Task | Device kernel samples/s | Avg kernel duration (ms) |
|------|------------------------:|-------------------------:|
| T2TT | 63.07 | 15.85 |
| S2TT | 33.31 | 30.02 |
| T2ST | 31.20 | 32.06 |
| S2ST | 21.65 | 46.20 |
| ASR  | 33.38 | 29.96 |

These are pure kernel execution (no host dispatch, no transfer overlap), so they are always higher than the E2E `forward()` numbers above. No FPS lower bound is hard-coded in the test (`expected_results={}` in `test_perf_device_bare_metal_seamless`); the test logs the measured number so regressions can be tracked across runs.

### Why `forward()` (not `generate()`) for both PCC and perf

- **PCC:** Autoregressive `generate()` cascades bf16 round-off through (text decoder × N) → T2U → vocoder. The final waveform PCC sits well below 0.99 even with the fp32 duration-predictor path in `tt_text_to_unit._duration_predictor`, so a strict PCC bar against HF is not meaningful end-to-end. A single deterministic `forward()` step produces the same logits as HF to within fp32-accumulator precision, so **PCC ≥ 0.99** is the right bar (`tests/pcc/test_seamless_m4t_v2_model.py:14-19`).
- **Perf:** `forward()` is the canonical per-prefill latency unit and the only apples-to-apples figure that the 2CQ overlapped pipeline can keep the device busy on. `generate()`'s greedy loop serialises around per-step host scalar readbacks for the EOS check, plus a host T2U → vocoder remap for `t2st` / `s2st`; the resulting FPS is full E2E generation FPS, not per-prefill FPS, and is many times lower than `forward()` even with 2CQ (`tests/perf/test_e2e_perf_2cq.py:794-808`).

### Why traced `generate()` is not provided

A Metal trace captures a fixed sequence of device commands and forbids host writes during `begin_trace_capture` (`write_shard_to_device` raises). The autoregressive `generate()` loop has host-dependent control flow that cannot be captured end-to-end:

1. **Per-step EOS readback.** Each step reads the predicted greedy token id back to host and breaks on `eos_token_id`. The next step's command sequence depends on that scalar, so the loop body cannot be one fixed trace (`tt/tt_seamless_m4t_v2_model.py:1413-1425`).
2. **Per-step KV-cache position update.** `cur_pos` and the cache-write index advance per token; a trace captures a single fixed position, not a variable one.
3. **Speech path host remap.** For `t2st` / `s2st`, between the text decoder and T2U the model does a host-side token-id → unit-vocabulary remap and pad-token substitution (string-table lookups in `generation_config`) before launching the vocoder. Neither table lookups nor variable-length T2U inputs are trace-safe.
4. **Variable output lengths.** Both the text decoder sequence length and the T2U / vocoder output length vary per call. Trace replay requires fixed shapes; nothing currently pads them to a single canonical size.

The text decoder does have a per-step KV-decode trace (`capture_text_decoder_decode_trace` at `tt/tt_seamless_m4t_v2_model.py:798`) that captures the **inner cell** of the loop, so individual decode steps run from trace even though the outer loop stays on the host. The full traced E2E paths (`forward_text_e2e_*_trace`, `forward_speech_e2e_*_trace`) all stop after the lm_head (or after T2U for `t2st` / `s2st`); they never try to capture the autoregressive loop itself.

---

## Running Tests

### PCC tests (functional correctness)

End-to-end, per-task:

```bash
pytest models/experimental/seamless_m4t_v2_large/tests/pcc/test_seamless_m4t_v2_model.py -v
```

Individual modules:

```bash
pytest models/experimental/seamless_m4t_v2_large/tests/pcc/test_text_encoder.py -v
pytest models/experimental/seamless_m4t_v2_large/tests/pcc/test_speech_encoder.py -v
pytest models/experimental/seamless_m4t_v2_large/tests/pcc/test_text_decoder.py -v
pytest models/experimental/seamless_m4t_v2_large/tests/pcc/test_text_to_unit.py -v
pytest models/experimental/seamless_m4t_v2_large/tests/pcc/test_code_hifigan.py -v
```

All PCC tests pass at `PCC_THRESHOLD = 0.99` (`tests/pcc/test_seamless_m4t_v2_model.py:61`).

### E2E performance tests (Blackhole)

Non-traced 2CQ pipeline:

```bash
pytest models/experimental/seamless_m4t_v2_large/tests/perf/test_e2e_perf_2cq.py::test_seamless_m4t_v2_large_e2e_perf_2cq -v
```

Traced 2CQ pipeline (full encoder + decoder + optional T2U inside `use_trace=True`):

```bash
pytest models/experimental/seamless_m4t_v2_large/tests/perf/test_e2e_perf_2cq.py::test_seamless_m4t_v2_large_e2e_perf_2cq_trace -v
```

Full autoregressive `generate()` pipeline driven through 2CQ:

```bash
pytest models/experimental/seamless_m4t_v2_large/tests/perf/test_e2e_perf_2cq.py::test_seamless_m4t_v2_large_e2e_perf_2cq_generate -v
```

### Device-level performance (kernel-only)

```bash
pytest models/experimental/seamless_m4t_v2_large/tests/perf/test_seamless_device_perf.py \
    -v -m models_device_performance_bare_metal
```

This wraps each PCC task in the device profiler and reports `AVG DEVICE KERNEL SAMPLES/S`.

---

## Repository Layout

```
models/experimental/seamless_m4t_v2_large/
├── demo/
│   └── demo.py                          # Full five-task TTNN demo (writes WAVs)
├── reference/                           # PyTorch wrappers used by PCC tests
│   ├── torch_seamless_m4t_v2_model.py
│   ├── torch_text_encoder.py
│   ├── torch_speech_encoder.py
│   ├── torch_text_decoder.py
│   ├── torch_text_to_unit.py
│   └── torch_code_hifigan.py
├── scripts/
│   └── download_weights.py              # HF snapshot downloader + CLI
├── tests/
│   ├── pcc/                             # PCC ≥ 0.99 per-module and per-task
│   │   ├── test_seamless_m4t_v2_model.py
│   │   ├── test_text_encoder.py
│   │   ├── test_speech_encoder.py
│   │   ├── test_text_decoder.py
│   │   ├── test_text_to_unit.py
│   │   └── test_code_hifigan.py
│   └── perf/
│       ├── test_e2e_perf_2cq.py         # 2CQ E2E: non-traced + traced + generate
│       └── test_seamless_device_perf.py # Device-kernel-only profiler wrapper
├── tt/                                  # TTNN implementation
│   ├── common.py
│   ├── model_preprocessing.py           # HF state-dict → TTNN params
│   ├── tt_seamless_m4t_v2_model.py
│   ├── tt_text_encoder.py
│   ├── tt_speech_encoder.py
│   ├── tt_text_decoder.py
│   ├── tt_text_to_unit.py
│   └── tt_code_hifigan.py
└── weights/                             # Downloaded HF snapshot lands here
    └── seamless-m4t-v2-large/           # config.json + safetensors shards (~10 GB)
```

---

## References

- [SeamlessM4T paper (Seamless Communication Team, 2023)](https://arxiv.org/abs/2312.05187)
- [facebook/seamless-m4t-v2-large on HuggingFace](https://huggingface.co/facebook/seamless-m4t-v2-large)
- [HuggingFace Transformers `modeling_seamless_m4t_v2.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/seamless_m4t_v2/modeling_seamless_m4t_v2.py)
- [Tenstorrent TT-Metalium](https://github.com/tenstorrent/tt-metal)
