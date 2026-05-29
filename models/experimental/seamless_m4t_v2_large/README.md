# Seamless M4T v2 Large on Tenstorrent Hardware

## Platforms

| Device | Status | Notes |
|--------|--------|-------|
| P150 (Blackhole) | Supported | `MeshShape(1, 1)` — single chip, replicated batch-1, all five tasks, 2CQ + traced + `generate()` |
| BH QB (Blackhole) | Supported | `MeshShape(1, 4)`, `FABRIC_1D` — four chips, replicated batch-1, same task coverage as P150 |

Mesh shape and `device_params` come from `tt/mesh_helpers.py` (pytest `1x1` / `1x4` parametrization; demo uses `open_seamless_mesh_device()`). On a 1-device host only `1x1` cases run; on a 4-device host only `1x4` cases run (the other parametrization is skipped).

All performance / E2E pipeline tests are gated with `@run_for_blackhole()`. PCC tests run on Blackhole with sufficient L1 (`l1_small_size=65536`) and the same mesh parametrization.

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

This port runs a **greedy** HuggingFace-style `SeamlessM4Tv2Model.generate` pipeline (EOS early-stop, host-side language/subword tables and a few speech-path remaps) on Tenstorrent hardware via TTNN. Weights are loaded in bfloat16; core encoder/decoder/T2U/vocoder math runs on device. See **Known Limitations** for gaps versus full HF `generate()`.

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

The demo opens a mesh via `open_seamless_mesh_device()` — `MeshShape(1, 1)` on P150, `MeshShape(1, 4)` on BH QB — and runs in this self-contained order, with no external audio file required:

1. **T2TT** — English text → Hindi text
2. **T2ST** — English text → Hindi speech (saved to `demo/outputs/t2st_hindi_speech.wav`)
3. **S2TT** — Hindi speech (from task 2) → English text
4. **S2ST** — Hindi speech (from task 2) → Spanish speech (saved to `demo/outputs/s2st_spanish_speech.wav`)
5. **ASR** — Hindi speech (from task 2) → Hindi text

---

## Performance (Blackhole BH QB, 2CQ)

End-to-end `generate()` throughput from the demo (`demo/demo.py`). The demo exercises the realistic long-audio path used in production — task 2 (T2ST) produces ~22 s of Hindi speech that tasks 3–5 consume, so the speech encoder runs at ~1100 mel frames (above `_LONG_AUDIO_RES_DRAM_THRESHOLD = 1024` in [`tt/tt_speech_encoder.py`](tt/tt_speech_encoder.py)). All five tasks run through `TTSeamlessM4Tv2Model.generate(...)` with 2CQ + per-step KV-decode trace.

### BH QB — `MeshShape(1, 4)`, replicated batch-1

Measured on a four-chip Blackhole QB host (`1x4` mesh, `FABRIC_1D`). Inputs are **replicated** on all four devices, not data-parallel batched — each device runs the same single-sample forward. Numbers are the median of two demo runs (2026-05-29).

| Task | Output unit | Throughput | Per-unit time |
|------|-------------|-----------:|--------------:|
| T2TT | text tokens | 60.6 tok/s | 16.5 ms/tok |
| T2ST | audio samples | 32.5 k smp/s | 30.7 μs/smp |
| S2TT | text tokens | 3.03 tok/s | 330 ms/tok |
| S2ST | audio samples | 13.3 k smp/s | 75.2 μs/smp |
| ASR  | text tokens | 9.99 tok/s | 100 ms/tok |

Tasks ranked by output type / bottleneck dominance:
- **T2TT** — text encoder + text decoder. Decoder-loop dominated.
- **T2ST** — text path + T2U + vocoder. Vocoder dominates the wall-clock; per-sample throughput is highest because vocoder generates ~500 k samples per request.
- **S2TT** — speech encoder + text decoder. **Speech-encoder dominated**; ms/tok is highest because the encoder prefill is amortized over only ~60–70 output tokens.
- **S2ST** — speech encoder + text decoder + T2U + vocoder. Vocoder samples/s metric dominates.
- **ASR** — speech encoder + text decoder. Like S2TT, but generally generates more tokens, so the prefill cost is amortized across more decode steps.

### Reproducing

```bash
python models/experimental/seamless_m4t_v2_large/demo/demo.py
```

The synthetic perf test (`tests/perf/test_e2e_perf_2cq.py::test_seamless_m4t_v2_generate_perf`) uses a 1-second audio fixture (`_make_speech_inputs` at [test_e2e_perf_2cq.py:160](tests/perf/test_e2e_perf_2cq.py#L160)) and therefore does **not** exercise the long-audio path. Use it for regression CI, but use the demo's chained long-audio numbers above as the canonical real-world figure.

### Recent optimizations (speech encoder)

- **BFP8 activations on the Conformer FFN expand projection** ([`tt/tt_speech_encoder.py::_conformer_ffn`](tt/tt_speech_encoder.py)): the `intermediate_dense` matmul previously ran bf8 weights × bf16 activations; casting the input to `ttnn.bfloat8_b` makes it bf8×bf8, halving activation L1 footprint on the K=1024 → N=local_ff_dim hot path. Post-LN input is well-conditioned for bf8 quantization.
- **Fused add + final_layer_norm per Conformer layer** via `residual_input_tensor` ([`tt/tt_speech_encoder.py::_conformer_encoder_layer`](tt/tt_speech_encoder.py)): saves one DRAM `add` dispatch per encoder layer in the long-audio path (24× per encoder forward). Short-audio path falls back to explicit add+LN so semantics stay identical.
- Net impact (S2TT, demo, long-audio): **~1.18×** (~389 → ~330 ms/tok). Other tasks flat within run-to-run variance (T2TT/T2ST don't use the speech encoder; ASR is text-decoder-loop dominated; S2ST is vocoder-dominated). PCC ≥ 0.99 holds at `mel_seq=3000` (measured 0.9957).

### Why PCC is measured on `forward()` (single prefill)

Autoregressive `generate()` cascades bf16 round-off through (text decoder × N) → T2U → vocoder. The final waveform PCC sits well below 0.99 even with the fp32 duration-predictor path in `tt_text_to_unit._duration_predictor`, so a strict PCC bar against HF is not meaningful end-to-end. A single deterministic `forward()` step produces the same logits as HF to within fp32-accumulator precision, so **PCC ≥ 0.99** is the right bar (`tests/pcc/test_seamless_m4t_v2_model.py`). The perf tables above use `generate()` because that is the user-visible throughput; `tests/perf/test_e2e_perf_2cq.py` also exposes per-prefill `forward()` benchmarks for regression CI on the kernel-bound floor.

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

End-to-end, per-task (parametrized `1x1` / `1x4`; use `-k 1x1` or `-k 1x4` on multi-device hosts):

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

All PCC tests pass at `PCC_THRESHOLD = 0.99` (`tests/pcc/test_seamless_m4t_v2_model.py`).

### E2E performance tests (Blackhole)

Tests are parametrized for **P150** (`1x1`) and **BH QB** (`1x4`) via `tt/mesh_helpers.py`; only the matching cases run on the current machine. Use `-k 1x1` or `-k 1x4` to select one platform.

Non-traced 2CQ pipeline:

```bash
pytest models/experimental/seamless_m4t_v2_large/tests/perf/test_e2e_perf_2cq.py::test_seamless_m4t_v2_large_e2e_perf_2cq -v
# P150 only:  ... -k 1x1
# BH QB only: ... -k 1x4
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

Runs five parametrized tasks (`t2tt` … `asr`); the Tracy subprocess picks `1x1` or `1x4` from the PCC mesh parametrization. Reports `AVG DEVICE KERNEL SAMPLES/S` (see **Device-kernel throughput** above).

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
│   ├── mesh_helpers.py                  # MeshShape (1,1)/(1,4), fabric, pytest params, demo open
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

## Known Limitations


### Hardware and deployment

- **Blackhole only.** PCC and perf tests run on Blackhole (`@run_for_blackhole()`). There is no Wormhole or Grayskull path. Supported meshes are **P150** `MeshShape(1, 1)` and **BH QB** `MeshShape(1, 4)` with `FABRIC_1D` (see `tt/mesh_helpers.py`).
- **Multi-device mode is replication, not batch parallelism.** On BH QB, batch-1 inputs and weights are replicated on all four chips; each device runs the same forward. Throughput scales via replication accounting in perf logs (`batch_size = 4`), not by sharding a larger user batch across devices.
- **L1 budget.** Speech-generation paths (T2U + vocoder, and chained S2ST) require `l1_small_size=65536` in device params. Smaller L1 (e.g. 32768) is insufficient for the full speech pipeline (see demo comments). Long mel inputs (>128 frames) use chunked 1D matmul in the speech encoder; the demo chains full-length T2ST audio into S2TT/S2ST/ASR without trimming.

### API scope versus Hugging Face

| HF capability | TTNN port |
|---------------|-----------|
| `output_attentions` / `output_hidden_states` | **Not supported** |
| `generate()` beam search (`num_beams > 1`) | **Not supported** — raises `NotImplementedError` |
| `generate()` sampling (`do_sample=True`, temperature, top-*p*, etc.) | **Not supported** — greedy `argmax` only |
| `generate()` `batch_size > 1` | **Not supported** — batch size 1 only |

---

## To Do


---

## References

- [SeamlessM4T paper (Seamless Communication Team, 2023)](https://arxiv.org/abs/2312.05187)
- [facebook/seamless-m4t-v2-large on HuggingFace](https://huggingface.co/facebook/seamless-m4t-v2-large)
- [HuggingFace Transformers `modeling_seamless_m4t_v2.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/seamless_m4t_v2/modeling_seamless_m4t_v2.py)
- [Tenstorrent TT-Metalium](https://github.com/tenstorrent/tt-metal)
