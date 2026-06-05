# Seamless M4T v2 Large on Tenstorrent Hardware

## Platforms

| Device | Status | Notes |
|--------|--------|-------|
| **BH QB (Blackhole)** | Supported | `MeshShape(1, 4)`, `FABRIC_1D` — four chips, replicated batch-1, all five tasks, 2CQ + per-step KV-decode trace + `generate()` |

This port targets **Blackhole QB** hosts with **four** Tenstorrent devices. Mesh shape and `device_params` come from [`tt/mesh_helpers.py`](tt/mesh_helpers.py) (`open_seamless_mesh_device()` for the demo; pytest fixtures use `MeshShape(1, 4)`). There is no Wormhole, Grayskull, or single-chip path in the supported/tested configuration.

All performance / E2E pipeline tests are gated with `@run_for_blackhole()`. PCC and generate perf tests use `l1_small_size=65536` (speech-generation paths) on the four-device mesh.

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

### Supported input lengths

Limits below are what the TT port exercises on **BH QB `MeshShape(1, 4)`**. HF config allows up to **4096** positions on the text side; per-module PCC tests pin the longest shapes that pass at **PCC ≥ 0.99**.

| | **Text input** (T2TT, T2ST) | **Speech input** (S2TT, S2ST, ASR) |
|---|---|---|
| **Input unit** | Tokenized source tokens (`processor(text=..., src_lang=...)`) | Log-mel frames from 16 kHz audio (`processor(audios=..., sampling_rate=16000)`) |
| **Design / HF maximum** | **4096** source tokens (`max_position_embeddings`) | **4096** mel frames |
| **Longest shape validated (PCC)** | Text encoder forward @ **4096** tokens ([`test_text_encoder.py`](tests/pcc/test_text_encoder.py)); text-decoder cross-attention prefill @ **512** encoder frames ([`test_text_decoder.py`](tests/pcc/test_text_decoder.py)) | Speech encoder forward @ **4096** mel frames ([`test_speech_encoder.py`](tests/pcc/test_speech_encoder.py)); text-decoder cross-attention prefill @ **512** subsampled encoder frames (same decoder test, S2TT path) |
| **Typical demo input** | Joyce-style English paragraph (~64 generated tokens in timed demo) | Preamble WAV resampled to 16 kHz: **~479** mel frames (~9.6 s) |

Notes:

- **Encoder timeline vs raw input.** For text tasks the encoder timeline equals the tokenized source length (1 token → 1 text-encoder frame). For speech tasks the Conformer stack plus length adaptor (kernel/stride **8**) subsamples mel into a shorter encoder timeline fed to the text decoder (~8× shorter than mel length at the upper bound).
- **512 decoder cross-attention limit.** Text-decoder prefill is PCC-validated only up to **512** encoder frames on BH 1×4. The next tile-aligned length (**1024**) overflows L1 on cross-attention prefill, so longer source inputs are not yet validated end-to-end through `generate()` even though the text encoder alone supports **4096** tokens.
- **Decoder KV budget.** `TTSeamlessM4Tv2Model` allocates text-decoder KV cache for **`max_text_seq_len=4096`** (seed + generated tokens). T2U is separately validated at **4096** encoder frames ([`test_text_to_unit.py`](tests/pcc/test_text_to_unit.py)).
- **Utterance-level behavior.** SeamlessM4T v2 is trained on short clips; very long text or audio can degrade quality on both HF and TT (see **Known Limitations**). Segment long-form inputs for production use.

---

## Prerequisites

1. Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal)
2. Installed [TT-Metalium / TT-NN](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
3. **Blackhole QB host** with four devices available to the runtime
4. Optional HuggingFace login (the `facebook/seamless-m4t-v2-large` repo is public, but a token avoids rate limits):

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

The demo runs on **`MeshShape(1, 4)`** via `open_seamless_mesh_device()` with **2CQ + per-step KV-decode trace**. Each task opens its **own** mesh device, runs untimed warmup `generate()` calls (two warmups for speech paths), then times one steady-state iteration — so reported runtimes are not affected by prior tasks' program-cache clears or L1 pressure. A throwaway T2TT preflight warms the global JIT cache before the timed tasks.

**Text input (tasks 1–2):** a long English paragraph hardcoded in [`demo/demo.py`](demo/demo.py) — a Joyce-style passage (`src_lang=eng`):

> going along slushy country roads and speaking to damp audiences in draughty schoolrooms day after day for a fortnight he'll have to put in an appearance at some place of worship on sunday morning and he can come to us immediately afterwards

**Speech input (tasks 3–5):** the US Constitution preamble read aloud, from [`preamble10.wav`](https://www.cs.kzoo.edu/cs107/MediaSources/preamble10.wav) (Kalamazoo College CS107 media). On first run `ensure_demo_audio()` downloads it to `demo/outputs/preamble10.wav` (raises if download fails). The WAV is mono, resampled from 22050 Hz to **16 kHz** before feature extraction (~9.6 s, ~479 mel frames).

Task order:

1. **T2TT** — text above (`eng`) → Hindi text
2. **T2ST** — same text (`eng`) → Hindi speech (saved to `demo/outputs/t2st_hindi_speech.wav`)
3. **S2TT** — preamble speech (`eng`) → Hindi text
4. **S2ST** — preamble speech (`eng`) → Spanish speech (saved to `demo/outputs/s2st_spanish_speech.wav`)
5. **ASR** — preamble speech (`eng`) → English text

---

## Performance (Blackhole BH QB, 2CQ + decode trace)

End-to-end `generate()` timings measured by running [`demo/demo.py`](demo/demo.py) on a four-chip Blackhole QB host, using the **text and speech inputs described above** (Joyce-style English paragraph for T2TT/T2ST; downloaded `preamble10.wav` for S2TT/S2ST/ASR). Inputs are **replicated** on all four devices (batch-1, TP=4), not data-parallel batched. Reported numbers exclude host pre/post-processing (token decode, WAV I/O). Each task opens its own mesh device with warmup before the timed iteration.

**Per-unit latency** (`ms/tok`, `μs/sample`) is more stable than throughput when output length varies run-to-run.

### BH QB — `MeshShape(1, 4)`, replicated batch-1

| Task | Runtime | Throughput | Workload | Per-unit |
|------|--------:|-----------:|----------|----------|
| T2TT | 669.5 ms | 95.59 tokens/s | 64 tokens | 10.5 ms/tok |
| T2ST | 4106.7 ms | 56492.72 samples/s | 232000 samples | 17.70 μs/smp |
| S2TT | 1201.9 ms | 21.63 tokens/s | 26 tokens | 46.2 ms/tok |
| S2ST | 4042.2 ms | 36336.83 samples/s | 146880 samples | 27.52 μs/smp |
| ASR | 1495.0 ms | 19.40 tokens/s | 29 tokens | 51.6 ms/tok |

Task notes:
- **T2TT** — text encoder + traced text-decoder loop.
- **T2ST** — text path + T2U + vocoder; vocoder dominates wall-clock but yields high samples/s.
- **S2TT / ASR** — speech encoder prefill + text decoder; dominated by encoder prefill amortized over short outputs on the ~9.6 s preamble clip (~479 mel frames).
- **S2ST** — speech encoder + decoder + T2U + vocoder.

**Cold start:** the first **S2ST** (and sometimes **T2ST**) call in a fresh process can be much slower (~20 s) while vocoder/T2U kernels JIT-compile. The demo opens a fresh device per task, so the first speech-synthesis task in a run may still pay one-time compile cost if the disk cache is cold; the table above reflects a warm JIT cache.

### Reproducing

```bash
python models/experimental/seamless_m4t_v2_large/demo/demo.py
```


---

## Running Tests

### PCC tests (functional correctness)

End-to-end per-task generate PCC (requires four devices):

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

All PCC tests pass at `PCC_THRESHOLD = 0.99` ([`tests/pcc/test_seamless_m4t_v2_model.py`](tests/pcc/test_seamless_m4t_v2_model.py)).

### E2E performance tests (Blackhole)

Full autoregressive `generate()` pipeline (TP=4 + 2CQ + decode trace), one parametrized test per task:

```bash
pytest models/experimental/seamless_m4t_v2_large/tests/perf/test_e2e_perf_2cq.py::test_seamless_m4t_v2_generate_perf -v
```

### Device-level performance (kernel-only)

```bash
pytest models/experimental/seamless_m4t_v2_large/tests/perf/test_seamless_device_perf.py \
    -v -m models_device_performance_bare_metal
```

The outer driver spawns eager forward-only inner tests in [`tests/perf/test_device_perf_forwards.py`](tests/perf/test_device_perf_forwards.py) under Tracy (`use_decode_trace=False`, `use_2cq=False`) and reports per-device kernel throughput. This is the device-bound floor that E2E perf approaches but cannot beat.

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
│   │   ├── test_code_hifigan.py
│   │   └── decoder_pcc_fixtures.py      # Shared decoder PCC inputs / helpers
│   └── perf/
│       ├── test_e2e_perf_2cq.py         # generate() perf (2CQ + decode trace)
│       ├── test_seamless_device_perf.py # Tracy outer driver (kernel-only)
│       └── test_device_perf_forwards.py # Inner eager forwards for device perf
├── tt/                                  # TTNN implementation
│   ├── common.py
│   ├── mesh_helpers.py                  # MeshShape (1,4), fabric, pytest params, demo open
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

- **Blackhole QB only.** PCC and perf tests run on Blackhole (`@run_for_blackhole()`). Supported mesh is **`MeshShape(1, 4)`** with `FABRIC_1D` (see [`tt/mesh_helpers.py`](tt/mesh_helpers.py)). Requires a host with four devices.
- **Multi-device mode is replication, not batch parallelism.** Batch-1 inputs and weights are replicated on all four chips; each device runs the same forward. Throughput accounting in perf logs uses `batch_size = 4` for replication, not sharded user batches.
- **L1 budget.** Speech-generation paths (T2U + vocoder) require `l1_small_size=65536` in device params. Long mel inputs use chunked 1D matmul in the speech encoder above `_LONG_AUDIO_RES_DRAM_THRESHOLD = 1024` ([`tt/tt_speech_encoder.py`](tt/tt_speech_encoder.py)).

### API scope versus Hugging Face

| HF capability | TTNN port |
|---------------|-----------|
| `output_attentions` / `output_hidden_states` | **Not supported** |
| `generate()` beam search (`num_beams > 1`) | **Not supported** — raises `NotImplementedError` |
| `generate()` sampling (`do_sample=True`, temperature, top-*p*, etc.) | **Not supported** — greedy `argmax` only |
| `generate()` `batch_size > 1` | **Not supported** — batch size 1 only |

### Utterance-level model: long inputs degenerate (text and speech)

SeamlessM4T v2 is trained on short utterances. Given a **long input** (text *or* audio) the model loses coherence — it drops content, repeats phrases, and emits EOS early — and on speech it may *translate* rather than transcribe. **This is the Hugging Face model's behavior, not a TTNN bug**: the bf16 reference degenerates the same way on the same input. The demo uses a long English paragraph for T2TT/T2ST to stress the text path; speech-input tasks use the shorter ~9.6 s preamble clip. Accuracy on long inputs should be judged by **TT-vs-HF faithfulness** (chrF / CER versus the bf16 reference on the same input), not by absolute correctness. Production pipelines typically segment long audio/text into utterance-sized chunks.

### Greedy decode is not bit-reproducible run-to-run

The TT `generate()` output **varies run-to-run even on a byte-identical input** on `MeshShape(1, 4)`. Tiny per-step floating-point differences in multi-device (TP) compute can flip the greedy `argmax` at near-tie decode steps and the T2U duration-predictor `round()`. Consequences:

- S2TT / ASR **text content and length** can vary between runs; T2ST / S2ST **audio sample counts** vary with T2U duration rounding.
- Demo **performance numbers vary** run-to-run because throughput divides by nondeterministic output length. Per-unit latency (`ms/tok`, `μs/smp`) is more stable.

The right correctness metric is the **TT-vs-HF faithfulness gate** (chrF / CER), not exact reproducibility.

### Speech-output throughput degrades on repeated runs (vocoder)

T2ST and S2ST can be much slower on **second and subsequent** `generate()` calls within a single process when the vocoder sees a **new `ttnn.conv1d` shape** each time (~15 s cold compile). The demo mitigates this by opening a fresh device per task with warmup iterations and a warm on-disk kernel cache; the **first** speech-synthesis task in a cold process may still be an outlier. Partially mitigated by length-bucketing short vocoder convs (`SEAMLESS_VOCODER_CONV1D_BUCKET`, default 256, in [`tt/tt_code_hifigan.py`](tt/tt_code_hifigan.py)). Chunked upsampled convs still vary and need sharded-aware bucketing.

---

## To Do

- **Full vocoder length-bucketing.** Extend short-conv bucketing to chunked (>4096, upsampled) convs to remove repeated-run slowdown on speech output. Blocked on a sharded-aware `ttnn.slice` for block-sharded resblock outputs.
- **Deterministic decode.** Investigate deterministic multi-device reductions (TTNN / CCL) so greedy `generate()` is bit-reproducible run-to-run.
- **Utterance segmentation in demo.** Optional VAD / sentence splitting for long-form inputs so chained tasks stay in the target language.

---

## References

- [SeamlessM4T paper (Seamless Communication Team, 2023)](https://arxiv.org/abs/2312.05187)
- [facebook/seamless-m4t-v2-large on HuggingFace](https://huggingface.co/facebook/seamless-m4t-v2-large)
- [HuggingFace Transformers `modeling_seamless_m4t_v2.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/seamless_m4t_v2/modeling_seamless_m4t_v2.py)
- [Tenstorrent TT-Metalium](https://github.com/tenstorrent/tt-metal)
