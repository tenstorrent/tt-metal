# Seamless M4T v2 Large on Tenstorrent Hardware

## Platforms


| Device                | Status    | Notes                                                                                                                          |
| --------------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **BH QB (Blackhole)** | Supported | `MeshShape(1, 4)`, `FABRIC_1D` — four chips, replicated batch-1, all five tasks, 2CQ + per-step KV-decode trace + `generate()` |
| **P150 (Blackhole)**  | Supported | `MeshShape(1, 1)` — single chip; same five tasks. Select with `MESH_DEVICE=P150` (demo / CI / pytest).                         |


This port targets **Blackhole** hosts. Mesh shape and `device_params` come from `[tt/mesh_helpers.py](tt/mesh_helpers.py)` (`open_seamless_mesh_device()` for the demo; pytest uses `legacy_mesh_device_param()` so `MESH_DEVICE=P150` → 1×1 and `MESH_DEVICE=BH-QB` → 1×4).

E2E / full-`generate()` paths use `l1_small_size=65536`. Module PCC and the Tracy L1 decoder microbench use `32768`.

---

## Introduction

[SeamlessM4T v2](https://huggingface.co/facebook/seamless-m4t-v2-large) is Meta's unified multilingual and multimodal translation model, introduced in [Seamless: Multilingual Expressive and Streaming Speech Translation](https://ai.meta.com/research/publications/seamless-multilingual-expressive-and-streaming-speech-translation/). A single set of weights covers five inference tasks across ~100 input and ~36 output languages:


| #   | Task                         | Abbrev.  | Input → Output                |
| --- | ---------------------------- | -------- | ----------------------------- |
| 1   | Text-to-Text Translation     | **T2TT** | text → text                   |
| 2   | Speech-to-Text Translation   | **S2TT** | speech → text                 |
| 3   | Text-to-Speech Translation   | **T2ST** | text → 16 kHz speech          |
| 4   | Speech-to-Speech Translation | **S2ST** | speech → 16 kHz speech        |
| 5   | Automatic Speech Recognition | **ASR**  | speech → text (same language) |


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


| Parameter                            | Value                   |
| ------------------------------------ | ----------------------- |
| Text encoder layers                  | 24                      |
| Text decoder layers                  | 24                      |
| Speech encoder layers (Conformer)    | 24                      |
| Text encoder/decoder attention heads | 16                      |
| Speech encoder attention heads       | 16                      |
| Hidden size                          | 1024                    |
| FFN dimension (text)                 | 8192                    |
| Speech encoder FFN dimension         | 4096                    |
| Speech feature input dim             | 160 (80 mel × 2 stride) |
| Adaptor conv kernel / stride         | 8 / 8                   |
| T2U encoder/decoder layers           | 6 / 6                   |
| Vocabulary size (text)               | 256102                  |
| Vocoder sample rate                  | 16 kHz                  |
| Total parameters                     | ~2.3 B                  |
| Weight precision                     | bfloat16                |


### Per-Task Active Modules


| Task | Text Enc | Speech Enc | Text Dec + lm_head | T2U | Vocoder |
| ---- | -------- | ---------- | ------------------ | --- | ------- |
| T2TT | yes      | —          | yes                | —   | —       |
| S2TT | —        | yes        | yes                | —   | —       |
| T2ST | yes      | —          | yes                | yes | yes     |
| S2ST | —        | yes        | yes                | yes | yes     |
| ASR  | —        | yes        | yes                | —   | —       |


### Supported input lengths

Limits below are what the TT port exercises on **Blackhole** (`MeshShape(1, 1)` P150 and `MeshShape(1, 4)` BH-QB). HF config allows up to **4096** positions on the text side; per-module PCC tests pin the longest shapes that pass at **PCC ≥ 0.99**.


|                                   | **Text input** (T2TT, T2ST)                                                                                                                                                                                                  | **Speech input** (S2TT, S2ST, ASR)                                                                                                                                                                                    |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Input unit**                    | Tokenized source tokens (`processor(text=..., src_lang=...)`)                                                                                                                                                                | Log-mel frames from 16 kHz audio (`processor(audios=..., sampling_rate=16000)`)                                                                                                                                       |
| **Design / HF maximum**           | **4096** source tokens (`max_position_embeddings`)                                                                                                                                                                           | **4096** mel frames                                                                                                                                                                                                   |

---

## Prerequisites

1. Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal)
2. Installed [TT-Metalium / TT-NN](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
3. **Blackhole** host — **P150** (one device) or **BH-QB** (four devices)
4. Optional HuggingFace login (the `facebook/seamless-m4t-v2-large` repo is public, but a token avoids rate limits):
  ```bash
   huggingface-cli login
   # or
   export HF_TOKEN=<your_token>
  ```
   Token: [https://huggingface.co/docs/hub/security-tokens](https://huggingface.co/docs/hub/security-tokens)

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

Unless `SEAMLESS_M4T_V2_WEIGHTS` is set, the first demo or test invocation calls `ensure_seamless_m4t_v2_large_weights()`, which downloads the HuggingFace snapshot into `models/experimental/seamless_m4t_v2_large/weights/seamless-m4t-v2-large/` (~10 GB). To download explicitly:

```bash
python models/experimental/seamless_m4t_v2_large/scripts/download_weights.py
```

### Custom location

Set the destination directly:

```bash
python models/experimental/seamless_m4t_v2_large/scripts/download_weights.py \
    --destination models/experimental/seamless_m4t_v2_large/weights/my-copy
```

Custom `--destination` must lie under `models/experimental/seamless_m4t_v2_large/weights/` (or the same path under `TT_METAL_HOME`).

Or point the **demo / perf sweep** at an existing snapshot via:

```bash
export SEAMLESS_M4T_V2_WEIGHTS=/path/to/seamless-m4t-v2-large
```

(`demo.py` and `scripts/demo_perf_sweep.py` honor this env var for any readable checkpoint path. PCC / Tracy tests always use `ensure_seamless_m4t_v2_large_weights()` under `weights/`.)

---

## How to Run

### Full TTNN demo (all five tasks)

```bash
python models/experimental/seamless_m4t_v2_large/demo/demo.py
```

The demo opens a mesh via `open_seamless_mesh_device()` (**1×4** on BH-QB / four devices, **1×1** on P150 / one device) with **2CQ + per-step KV-decode trace**. Each task opens its **own** mesh device.

Warmups / timed iters (`demo.py`):

| Path | Untimed warmups | Timed iters | Notes |
| ---- | --------------: | ----------: | ----- |
| **T2TT** | 1 | 1 | — |
| **T2ST / S2ST** | 2 | 2 | Vocoder conv prewarm after warmups; report min of timed iters |
| **S2TT / ASR** | 2 | 1 | Speech-input text tasks |

`demo_perf_sweep.py` reuses the same warmup counts but always uses **1 timed iter** per task and does **not** call the vocoder `post_warmup_fn` (unlike `demo.py`).

**Text input (T2TT / T2ST):** an English sentence is hardcoded in `[demo/demo.py](demo/demo.py)` — a Joyce-style passage (`src_lang=eng`):

> going along slushy country roads and speaking to damp audiences in draughty schoolrooms day after day for a fortnight he'll have to put in an appearance at some place of worship on sunday morning and he can come to us immediately afterwards

**Speech input (S2TT / S2ST / ASR):** the US Constitution preamble read aloud, from `[preamble10.wav](https://www.cs.kzoo.edu/cs107/MediaSources/preamble10.wav)` (Kalamazoo College CS107 media). On first run `ensure_demo_audio()` downloads it to `demo/outputs/preamble10.wav` (raises if download fails). The WAV is mono, resampled from 22050 Hz to **16 kHz** before feature extraction (~9.6 s, ~479 mel frames).

Task order:

1. **T2TT** — text above (`eng`) → Hindi text
2. **T2ST** — same text (`eng`) → Hindi speech (saved to `demo/outputs/t2st_hindi_speech.wav`)
3. **S2TT** — preamble speech (`eng`) → Hindi text
4. **S2ST** — preamble speech (`eng`) → Spanish speech (saved to `demo/outputs/s2st_spanish_speech.wav`)
5. **ASR** — preamble speech (`eng`) → English text

### Sequence-length performance sweep

`[scripts/demo_perf_sweep.py](scripts/demo_perf_sweep.py)` runs the same five tasks while doubling input length **32 → 4096** (text = source token count; speech = mel-frame count). Long inputs are prepared once:

- **Text** — *A Tale of Two Cities* from `models/tt_transformers/tests/tale-of-two-cities.txt.bz2` (cached under `scripts/outputs/` when expanded).
- **Audio** — by default, concatenated **LibriSpeech-dummy** utterances until ≥ 4096 mel frames (`scripts/outputs/long_speech_input_librispeech.wav`). Falls back to looping the demo preamble WAV if the dataset is unavailable.

At mel lengths **≥ 1024**, **S2TT / S2ST / ASR** warmups run on a **throwaway** mesh device so the timed session is fresh (avoids multi-`generate()` decode-trace collapse). The timed session still runs speech-encoder prewarm; `generate()` releases prior T2U/vocoder/decode-trace state but does **not** wipe the program cache before speech encode (that clear was collapsing free-running mel-1024 output into token loops). Dummy `_encode_speech` prewarm is skipped only for mel **1920–2560** (proxy JIT at 3072). T2ST does not use split warmups.

```bash
python models/experimental/seamless_m4t_v2_large/scripts/demo_perf_sweep.py
# optional: --min-len 32 --max-len 512 --output scripts/outputs/perf_sweep.txt
```

Log, per-length WAVs, and summary tables land under `scripts/outputs/` (default log: `perf_sweep.txt`).

---

## Performance

Demo wall timings (BH QB, 2CQ + decode trace) are documented in [PERF.md](PERF.md).

### Device kernel perf (Tracy, eager, no trace)

```bash
pytest models/experimental/seamless_m4t_v2_large/tests/perf/test_seamless_device_perf.py \
    -v -m models_device_performance_bare_metal
```

The outer driver spawns eager forwards in `[tests/perf/test_device_perf_forwards.py](tests/perf/test_device_perf_forwards.py)` under Tracy and reports **both** Tracy per-device kernel floors and the same TT-aligned wall metrics (via JSON side-channel).

### Single-layer decoder prefill+decode (Tracy micro-benchmark)

Isolated **layer-0** text-decoder kernel profiling: one **128-token prefill** (KV fill) plus **one decode step** at position 128, with a **128-token** cross-attention encoder timeline.

| Item | Value |
|------|--------|
| Decoder layers profiled | 1 (layer 0 weights only) |
| Decoder prefill ISL | 128 tokens (single forward, not chunked) |
| Decode | 1 token at `DECODE_POS=128` |
| Encoder cross-attn length | 128 |
| Mesh | `MeshShape(1, 4)` on BH-QB (auto when 4 devices); `MeshShape(1, 1)` on P150 / single-chip (auto when 1 device). Override: `MESH_DEVICE=BH-QB` or `MESH_DEVICE=P150` |
| Mode | Eager (no decode trace / 2CQ) — Tracy-friendly |

This is **not** the full five-task demo (no encoder, no 24-layer stack, no `generate()` loop). Same **128-token** decoder ISL as the `128` rung on `[scripts/demo_perf_sweep.py](scripts/demo_perf_sweep.py)`, but kernel numbers are **not** comparable to demo `prefill_ms` (all layers + trace). On **1×1**, decode matmul head dims differ from **1×4** (no TP sharding); compare like mesh shapes only.

The driver avoids importing `ttnn` (no parent `CHIP_IN_USE` lock). It spawns Tracy on `[tests/perf/test_profile_single_layer_prefill_decode.py](tests/perf/test_profile_single_layer_prefill_decode.py)` (warmup, then signpost `start` → decoder `forward` prefill + decode → `stop`; masks/embeddings/KV init outside the window). Artifacts: `generated/profiler/seamless_m4t_v2_L1_prefill_decode/reports/<timestamp>/ops_perf_results_*.csv` plus a partial benchmark JSON (`seamless_m4t_v2_L1_prefill128_decode1_<mesh>`).

```bash
cd tt-metal
python models/experimental/seamless_m4t_v2_large/tests/perf/test_device_perf_single_layer_prefill_decode.py
```

**Inspect the CSV with `tt-perf-report`** (per-op device kernel time between signposts):

```bash
# use the latest ops CSV from the run above
CSV=$(ls -t generated/profiler/seamless_m4t_v2_L1_prefill_decode/reports/*/ops_perf_results_*.csv | head -1)

tt-perf-report "$CSV" --start-signpost start --end-signpost stop
```

On **1×4**, look for prefill matmuls with batch **128** and decode ops (`SdpaDecodeDeviceOperation`, `PagedUpdateCacheDeviceOperation`); on **1×1** the same ops appear with a single device row. A large **op-to-op gap** between prefill and decode is usually host embedding/setup, not device kernel time.

---

## Running Tests

### Continuous integration

| Workflow | What runs |
|----------|-----------|
| `(Blackhole) Demo tests` → `seamless-m4t-v2-large` | Full `demo_perf_sweep.py` plus logit PCC, token matching, and teacher-forced WER sweeps (`-k sweep`) on P150 and BH-QB |

| Job | SKU | Mesh |
|-----|-----|------|
| `demo perf sweep (P150 1x1)` | P150 CIv2 | 1×1 |
| `e2e logit PCC sweep (P150 1x1)` | P150 CIv2 | 1×1 |
| `e2e token matching sweep (P150 1x1)` | P150 CIv2 | 1×1 |
| `e2e WER sweep (P150 1x1)` | P150 CIv2 | 1×1 |
| `demo perf sweep (BH-QB 1x4)` | QuietBox 2 | 1×4 |
| `e2e logit PCC sweep (BH-QB 1x4)` | QuietBox 2 | 1×4 |
| `e2e token matching sweep (BH-QB 1x4)` | QuietBox 2 | 1×4 |
| `e2e WER sweep (BH-QB 1x4)` | QuietBox 2 | 1×4 |

**Note:** Job timeouts are capped by the shared `models.demo` time budget (`bh_p150b_civ2`: 360 min, `bh_quietbox_2`: 480 min). Full ISL sweeps (32→4096) can exceed the per-job allotment for logit / token / WER on P150 especially, so some CI jobs may fail due to time-budget constraints even when the underlying tests are correct.

Weights: `facebook/seamless-m4t-v2-large` (~10 GB). The demo and pytest entry points call `ensure_seamless_m4t_v2_large_weights()` on first use (`HF_TOKEN` from CI secrets).

```bash
export MESH_DEVICE=BH-QB   # or P150
python models/experimental/seamless_m4t_v2_large/scripts/demo_perf_sweep.py

pytest models/experimental/seamless_m4t_v2_large/tests/pcc/test_seamless_e2e_logit_pcc_sweep.py -k sweep -v
pytest models/experimental/seamless_m4t_v2_large/tests/pcc/test_seamless_e2e_token_matching_sweep.py -k sweep -v
pytest models/experimental/seamless_m4t_v2_large/tests/pcc/test_seamless_e2e_wer_sweep.py -k "teacher_forced and sweep" -v
```

### PCC tests (functional correctness)

**Per-module PCC** (PCC ≥ 0.99 at max sequence length):

```bash
pytest models/experimental/seamless_m4t_v2_large/tests/pcc/test_text_encoder.py -v
pytest models/experimental/seamless_m4t_v2_large/tests/pcc/test_speech_encoder.py -v
pytest models/experimental/seamless_m4t_v2_large/tests/pcc/test_text_decoder.py -v
pytest models/experimental/seamless_m4t_v2_large/tests/pcc/test_text_to_unit.py -v
pytest models/experimental/seamless_m4t_v2_large/tests/pcc/test_code_hifigan.py -v
```

**Prefill and Decode PCC**:

```bash
pytest models/experimental/seamless_m4t_v2_large/tests/pcc/test_prefill.py -v
pytest models/experimental/seamless_m4t_v2_large/tests/pcc/test_decode.py -v
```

**E2E ISL sweeps** (input-length ladder 32→4096; speech-input tasks **skip mel ≤ 64** — see **Short speech inputs** under Known Limitations):

```bash
# Text-output tasks (T2TT, S2TT, ASR): teacher-forced token top-1/top-5 vs offline HF refs
pytest models/experimental/seamless_m4t_v2_large/tests/pcc/test_seamless_e2e_token_matching_sweep.py -k sweep -v

# Text-output tasks: full-vocabulary logits PCC (HF-greedy decode)
pytest models/experimental/seamless_m4t_v2_large/tests/pcc/test_seamless_e2e_logit_pcc_sweep.py -k sweep -v

# Speech-output / ASR WER (T2ST, S2ST, ASR): teacher-forced and whisper round-trip flavors
pytest models/experimental/seamless_m4t_v2_large/tests/pcc/test_seamless_e2e_wer_sweep.py -k "teacher_forced and sweep" -v
pytest models/experimental/seamless_m4t_v2_large/tests/pcc/test_seamless_e2e_wer_sweep.py -k "whisper and sweep" -v
```

Per-module tests use `PCC_THRESHOLD = 0.99`. E2E sweeps use task-specific gates (token matching top-1/top-5, logits PCC ≥ 0.90, WER thresholds) documented in each test file. Speech-input points at mel **32 and 64** are skipped on both P150 and BH-QB.

---

## Repository Layout

```
models/experimental/seamless_m4t_v2_large/
├── README.md
├── PERF.md                               # Demo wall timings (BH QB ISL sweep)
├── demo/
│   ├── demo.py                          # Full five-task TTNN demo (writes WAVs)
│   └── outputs/                         # Generated: preamble WAV, demo speech outputs
├── reference/                           # PyTorch wrappers used by PCC tests
│   ├── torch_seamless_m4t_v2_model.py
│   ├── torch_text_encoder.py
│   ├── torch_speech_encoder.py
│   ├── torch_text_decoder.py
│   ├── torch_text_to_unit.py
│   └── torch_code_hifigan.py
├── scripts/
│   ├── download_weights.py              # HF snapshot downloader + CLI
│   ├── demo_perf_sweep.py               # Sequence-length perf sweep (32→4096, all five tasks)
│   ├── generate_t2tt_token_accuracy_reference.py  # Offline HF refs for token-matching / logit-PCC sweeps
│   ├── generate_wer_sweep_reference.py  # Offline HF refs for WER sweeps
│   └── outputs/                         # Generated: perf_sweep.txt, sweep WAVs, cached long inputs
├── tests/
│   ├── conftest.py                      # pytest markers + E2E sweep summary tables
│   ├── reference_outputs/               # Generated: fixed-length token-matching .refpt files
│   ├── teacher_forced_sweep_outputs/  # Generated: ISL sweep artifacts
│   │   ├── references/                  #   token-matching / logit-PCC .refpt (per task × length)
│   │   ├── wer_references/              #   WER sweep .refpt (T2ST, S2ST, ASR)
│   │   └── debug_mel_inputs/            #   optional mel debug dumps (SEAMLESS_SWEEP_SAVE_MEL)
│   ├── pcc/
│   │   ├── test_text_encoder.py         # Per-module PCC @ max seq (4096 tokens)
│   │   ├── test_speech_encoder.py       # Per-module PCC @ max mel (4096 frames)
│   │   ├── test_text_decoder.py         # T2TT / S2TT decoder PCC @ MAX_ENC_SEQ=4096
│   │   ├── test_text_to_unit.py         # T2U PCC @ encoder seq 4096
│   │   ├── test_code_hifigan.py         # Vocoder PCC @ unit seq 1024
│   │   ├── test_prefill.py              # Layer-0 decoder prefill PCC (ISL sweep)
│   │   ├── test_decode.py               # Layer-0 decoder decode PCC (10 steps)
│   │   ├── test_seamless_e2e_token_matching_sweep.py   # E2E top-1/top-5 ISL sweep (T2TT, S2TT, ASR)
│   │   ├── test_seamless_e2e_logit_pcc_sweep.py       # E2E logits PCC ISL sweep (T2TT, S2TT, ASR)
│   │   ├── test_seamless_e2e_wer_sweep.py             # E2E WER ISL sweep (T2ST, S2ST, ASR)
│   │   ├── pcc_test_common.py           # Shared weights / mesh pytest helpers
│   │   ├── decoder_pcc_common.py        # Layer-0 decode/prefill shared logic
│   │   ├── decoder_pcc_fixtures.py      # Production-shaped decoder inputs
│   │   ├── e2e_task_config.py           # Shared E2E task lists, short-speech skip, ISL lengths
│   │   ├── e2e_tt_model_helpers.py      # TTSeamlessM4Tv2Model builder + torch→ttnn I/O
│   │   ├── e2e_logit_pcc_helpers.py     # Logits PCC loop + speech E2E input builders
│   │   ├── e2e_token_matching_helpers.py  # Token matching loop + ISL sweep utilities
│   │   ├── e2e_wer_helpers.py           # WER / whisper round-trip + sweep utilities
│   │   └── token_matching_result_store.py  # In-process sweep result rows (pytest summary)
│   └── perf/
│       ├── test_seamless_device_perf.py              # Tracy outer driver (full-model kernel-only)
│       ├── test_device_perf_forwards.py              # Inner eager forwards for device perf
│       ├── test_profile_single_layer_prefill_decode.py  # Inner L1 prefill+decode workload (Tracy)
│       └── test_device_perf_single_layer_prefill_decode.py  # L1 driver: python entry + CSV/JSON dump
├── tt/                                  # TTNN implementation
│   ├── common.py                        # TP reductions, hf_aligned_generation_kwargs
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

- **Blackhole only (P150 + BH-QB).** Mesh selection is via `MESH_DEVICE` / device count in `[tt/mesh_helpers.py](tt/mesh_helpers.py)`: **P150** → `MeshShape(1, 1)`; **BH-QB** → `MeshShape(1, 4)` + `FABRIC_1D`. There is no Wormhole / Grayskull path.
- **L1 budget.** E2E / full-`generate()` paths use `l1_small_size=65536`. Module PCC and the Tracy L1 decoder microbench use `32768`. Long mel inputs switch the speech encoder to chunked 1D matmul and DRAM residuals above `_LONG_AUDIO_RES_DRAM_THRESHOLD = 512` (`[tt/tt_speech_encoder.py](tt/tt_speech_encoder.py)`). Text-encoder TP block-sharded matmul row-chunks at `SEAMLESS_TP_BS_CHUNK_M` (default **2048**) (`[tt/tt_text_encoder.py](tt/tt_text_encoder.py)`).
- **TP reductions.** On `MeshShape(1, 4)` several layers are row-parallel: each chip computes a partial output and a **TP reduction** sums those partials into one full activation before the next layer runs. This port uses `ttnn.all_reduce` (linear topology) for the **text encoder** and **text decoder** (`[encoder_all_reduce_sum_replicate](tt/common.py)`, `[decoder_all_reduce_sum_replicate](tt/common.py)` — decoder path does not deallocate the input, for residual adds). The **speech encoder** and **T2U** use `all_gather` + `sum` instead (`[all_reduce_sum_replicate](tt/common.py)`), because switching them to `ttnn.all_reduce` previously broke speech-path stability on BH QB (S2TT/ASR token loops, L1 pressure). Do not unify all modules to `all_reduce` without re-running PCC, determinism, and long-sequence speech tests.

### API scope versus Hugging Face


| HF capability                                                           | TTNN port                                                                                                                                                    |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `output_attentions` / `output_hidden_states`                            | **Not supported**                                                                                                                                            |
| `generate()` beam search (`num_beams > 1`)                              | **Supported** with KV cache — beam scoring runs on **host PyTorch** (log-softmax / top-*k*); not compatible with decode trace or 2CQ                         |
| `generate()` sampling (`do_sample=True`, temperature, top-*p*, top-*k*) | **Supported** — sampling math runs on **host PyTorch** after a logits readback; not compatible with decode trace or 2CQ                                      |
| `generate()` `repetition_penalty`                                       | **Supported** for greedy decode and beam search (host PyTorch, HF rule). **Not applied** when `do_sample=True`. Must be `>= 1.0`; default **1.0** (disabled) |
| `generate()` `batch_size > 1`                                           | **Not supported** — batch size 1 only                                                                                                                        |


### Host / PyTorch work in `generate()`

The TT model runs encoders, decoder, T2U, and vocoder on device, but `**TTSeamlessM4Tv2Model.generate()` is not a fully device-resident autoregressive loop**. Several steps still execute on the host CPU using Python and **PyTorch** (transport, scoring, or HF-parity string logic). This is intentional for HF alignment and for decode modes that cannot be captured in a Metal trace.

**Autoregressive text decode (all tasks)**


| Step                                                                 | Where          | Notes                                                                                                                                                                                                                                                                                                                                                                                      |
| -------------------------------------------------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Decode loop control                                                  | Host (Python)  | One iteration per output token; EOS checked against a host-maintained `seq_host` list                                                                                                                                                                                                                                                                                                      |
| Per-step token / position upload                                     | Host → device  | Reused `torch.int32` staging buffers + `copy_host_to_device_tensor` (required while decode trace is active — no device writes during trace replay)                                                                                                                                                                                                                                         |
| Greedy token pick (TP=4, traced path, `repetition_penalty=1.0`)      | Device + host  | Device fused/chunked argmax in trace; **chunk max + local index read back** and combined on host with PyTorch (64 scalars D2H). Matches HF on demo inputs.                                                                                                                                                                                                                                 |
| Greedy token pick (TP=4, eager / no trace, `repetition_penalty=1.0`) | Device + host  | `_ondevice_global_argmax_token`: per-shard chunk argmax + `all_gather` + HF tie-break on host + **single scalar D2H**                                                                                                                                                                                                                                                                      |
| Repetition penalty (`repetition_penalty > 1.0`)                      | Host (PyTorch) | HF `**RepetitionPenaltyLogitsProcessor`**: for each already-emitted token id, logits `< 0` are multiplied by penalty, logits `>= 0` are divided. Applied on a gathered `**[V]**` logits row in torch, then `**argmax**`. **Never runs on device.** With trace+2CQ, chunk-argmax is tried first; if the unpenalized winner is a repeat token, falls back to full-row D2H + penalty + argmax |
| Greedy token pick (eager / no trace)                                 | Device or host | Device `ttnn.argmax` when `repetition_penalty=1.0`; full logits row readback + host penalty + argmax when `repetition_penalty > 1.0`                                                                                                                                                                                                                                                       |
| Beam search (`num_beams > 1`)                                        | Host (PyTorch) | Per-beam logits read back; `**log_softmax`**, repetition penalty on emitted ids, and `**topk**` beam scoring on host; KV caches reordered with `ttnn.copy`                                                                                                                                                                                                                                 |
| Sampling (`do_sample=True`)                                          | Host (PyTorch) | Logits row read back; temperature / top-*k* / top-*p* filtering and `**multinomial*`* on host — **not traced**; `**repetition_penalty` is ignored** (HF applies it via logits processors before sampling)                                                                                                                                                                                  |
| Sequence bookkeeping                                                 | Host (Python)  | Token ids accumulated in Python lists; `ttnn.from_torch` used to rebuild `sequences_tt` after the loop                                                                                                                                                                                                                                                                                     |


**Speech generation path (T2ST / S2ST only, after text decode)**


| Step                              | Where                           | Notes                                                                                                                                                                |
| --------------------------------- | ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Trailing pad / EOS trim           | Host (Python)                   | `_trim_seq_host_for_speech` before T2U                                                                                                                               |
| Subword → character tables        | Host (Python)                   | `generation_config.id_to_text`, `char_to_id`, and HF-style `**_char_count_per_subword`** string analysis                                                             |
| T2U char ids / duration counts    | Host (Python + torch transport) | Character id lists and `char_count_per_id` built on host, uploaded with `from_torch`                                                                                 |
| T2U unit id → vocoder vocab remap | Host (PyTorch)                  | T2U `**argmax**` on device, then unit ids + padding mask read back; EOS/pad masking and `**vocoder_offset**` applied with `torch.where`, re-uploaded for the vocoder |


**Other host touches**

- `**generation_config` lookups** — target language code ids, EOS id sets, subword/char tables (string dict ops).
- **Scalar readbacks** — subsampled speech-encoder length (for slice bounds), per-decode-step greedy token id (EOS), and optional profiler/signpost hooks in tests.
- **Pre/post outside `generate()`** — demo and tests still run Hugging Face `**AutoProcessor**` tokenization and feature extraction on host before uploading tensors.

**Production implication:** the demo and documented perf numbers use **greedy decode + per-step KV trace + 2CQ** with `**repetition_penalty=1.0`** (HF default, penalty disabled). That keeps decode on the fast device chunk-argmax path with only chunk-index readback and EOS checks on host. Setting `**repetition_penalty > 1.0**` adds per-step host logits gather + penalty work (and occasional full-row fallback on the traced path). Beam search and sampling disable trace/2CQ and move most decode scoring to host PyTorch. A full end-to-end Metal trace of `generate()` is not possible while these host-dependent control paths remain (see module docstring in `[tt/tt_seamless_m4t_v2_model.py](tt/tt_seamless_m4t_v2_model.py)`).

### Short speech inputs (mel ≤ 64)

E2E ISL sweeps (token matching, logits PCC, WER) **skip speech-input tasks** (S2TT, S2ST, ASR) when the mel length is **≤ 64**, on **both P150 (1×1) and BH-QB (1×4)**. Text-input tasks (T2TT, T2ST) still run at 32/64.

**Reason:** Seamless mel features are ~50 Hz, so 32 frames ≈ 0.64 s and 64 ≈ 1.3 s of audio. After the speech encoder’s strided convolutions the effective timeline is even shorter. On these clips Hugging Face greedy decode typically emits **EOS after only a few tokens** (often 3–9 steps). That makes E2E gates unstable or meaningless:

- **Token matching / logits PCC** — with *n* = 3 teacher steps the only achievable top-1 rates are 0 / 33 / 67 / 100%, so an 87% gate is unreachable except at a perfect match; measured S2TT/ASR at mel 64 often land at **66.7% top-1** (2/3) while top-5 stays 100%.
- **WER** — the HF reference text can be a **single word** (e.g. ASR mel=32), so WER is 0 or 1 by quantization and is not a useful fidelity signal.

This is a **reference / metric limitation on ultra-short audio**, not a mesh-specific TT bug. Sweeps start scoring speech-input points from **mel 128** upward. Implemented via `maybe_skip_short_speech_input` in `[tests/pcc/e2e_task_config.py](tests/pcc/e2e_task_config.py)` (`SHORT_SPEECH_E2E_MAX_MEL = 64`).

### Utterance-level model: long inputs degenerate (text and speech)

SeamlessM4T v2 is trained on short utterances. Given a **long input** (text *or* audio) the model loses coherence — it drops content, repeats phrases, and emits EOS early — and on speech it may *translate* rather than transcribe. **This is the Hugging Face model's behavior, not a TTNN bug**: the bf16 reference degenerates the same way on the same input. The demo uses a long English paragraph for T2TT/T2ST to stress the text path; speech-input tasks use the shorter ~9.6 s preamble clip. Accuracy on long inputs should be judged by **TT-vs-HF faithfulness** (chrF / CER versus the bf16 reference on the same input), not by absolute correctness. Production pipelines typically segment long audio/text into utterance-sized chunks.

### Cross-run stability and TT-vs-HF parity

On `**MeshShape(1, 4)`** with the default demo settings (greedy decode, trace + 2CQ, `repetition_penalty=1.0`), **identical inputs produce stable TT outputs across repeated demo runs** for the Joyce paragraph and preamble speech inputs (verified in-process on BH 1×4 alongside the PCC suite).

TT outputs are **not required to be bit-identical to Hugging Face** on every task (E2E sweeps gate token matching, logits PCC, and WER — not exact token or sample match). Residual gaps include:

- **T2ST / S2ST waveform length** can differ from HF by a modest sample count while still passing PCC voicing checks.
- **Strict bit-reproducibility** (every greedy step identical run-to-run under all TP tie cases) is not yet guaranteed.

Phrase **repetition on very long inputs** (text or audio) is expected **HF model behavior** on utterance-scale inputs, not a TT-only bug — see **Utterance-level model** above.

### Vocoder throughput (mitigated)

Vocoder conv timelines are **length-bucketed** (short single-shot and chunked/upsampled paths via `_vocoder_timeline_bucket` / `_slice_nlc_time` in `[tt/tt_code_hifigan.py](tt/tt_code_hifigan.py)`; override with `SEAMLESS_VOCODER_CONV1D_BUCKET`). Speech `generate()` **preserves vocoder prep/program cache** across decode/T2U program evictions (`_clear_decode_and_t2u_programs(preserve_vocoder=True)`). Together with demo warmups this removes the prior ~8–25 s vocoder recompile on every in-process speech iter; see [PERF.md](PERF.md) (**Cold start**) for the remaining one-time JIT outlier on a cold process.

---

## To Do

- **Utterance segmentation (speech).** Optional VAD / audio chunking for long-form speech inputs (text tasks: `SEAMLESS_DEMO_MAX_SENTENCES` in `[demo/demo.py](demo/demo.py)`).

---

## References

- [SeamlessM4T paper (Seamless Communication Team, 2023)](https://arxiv.org/abs/2312.05187)
- [facebook/seamless-m4t-v2-large on HuggingFace](https://huggingface.co/facebook/seamless-m4t-v2-large)
- [HuggingFace Transformers `modeling_seamless_m4t_v2.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/seamless_m4t_v2/modeling_seamless_m4t_v2.py)
- [Tenstorrent TT-Metalium](https://github.com/tenstorrent/tt-metal)
