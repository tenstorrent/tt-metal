# Seamless M4T v2 Large on Tenstorrent Hardware

## Platforms

| Device | Status | Notes |
|--------|--------|-------|
| **BH QB (Blackhole)** | Supported | `MeshShape(1, 4)`, `FABRIC_1D` — four chips, replicated batch-1, all five tasks, 2CQ + per-step KV-decode trace + `generate()` |

This port targets **Blackhole QB** hosts with **four** Tenstorrent devices. Mesh shape and `device_params` come from [`tt/mesh_helpers.py`](tt/mesh_helpers.py) (`open_seamless_mesh_device()` for the demo; pytest fixtures use `MeshShape(1, 4)`). There is no Wormhole, Grayskull, or single-chip path in the supported/tested configuration.

PCC tests and the Tracy device-perf driver use `l1_small_size=65536` on speech-generation paths where required.

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
| **Longest shape validated (PCC)** | Text encoder forward @ **4096** tokens ([`test_text_encoder.py`](tests/pcc/test_text_encoder.py)); text-decoder cross-attention prefill @ **1024** encoder frames ([`test_text_decoder.py`](tests/pcc/test_text_decoder.py)) | Speech encoder forward @ **4096** mel frames ([`test_speech_encoder.py`](tests/pcc/test_speech_encoder.py)); text-decoder cross-attention prefill @ **1024** subsampled encoder frames (same decoder test, S2TT path) |
| **Typical demo input** | Joyce-style English paragraph | Preamble WAV resampled to 16 kHz: **~479** mel frames (~9.6 s) |

Notes:

- **Encoder timeline vs raw input.** For text tasks the encoder timeline equals the tokenized source length (1 token → 1 text-encoder frame). For speech tasks the Conformer stack plus length adaptor (kernel/stride **8**) subsamples mel into a shorter encoder timeline fed to the text decoder (~8× shorter than mel length at the upper bound).
- **Decoder cross-attention prefill.** Text-decoder prefill is PCC-validated up to **1024** encoder frames on BH 1×4 ([`test_text_decoder.py`](tests/pcc/test_text_decoder.py)). End-to-end ``generate()`` at **4096**-token scale is not yet certified.
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

The demo runs on **`MeshShape(1, 4)`** via `open_seamless_mesh_device()` with **2CQ + per-step KV-decode trace**. Each task opens its **own** mesh device. Speech paths (T2ST, S2ST) run **two untimed warmups**, a vocoder conv prewarm from cached shapes, then **two timed** `generate()` calls (phase timings taken from the faster timed iter). Text paths use one warmup and one timed iter.

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

Phase-separated timings from [`demo/demo.py`](demo/demo.py) with ``generate(return_timings=True)`` on a four-chip Blackhole QB host (`MeshShape(1, 4)`, batch-1 replicated TP=4). Uses the **text and speech inputs described above** (Joyce-style English paragraph for T2TT/T2ST; downloaded `preamble10.wav` for S2TT/S2ST/ASR). Host pre/post-processing (token decode, WAV I/O) is excluded. Each task opens its own mesh device with warmup before the timed iteration.

Metrics follow the TT model catalog (Whisper / LLM / Qwen3-TTS style):

| Metric | Meaning |
|--------|---------|
| **TTFT** | Time from `generate()` start to first **new** decoder token (includes encoder + decoder prefill + first decode step) |
| **Encoder** | Speech or text encoder only |
| **Prefill** | Text-decoder KV prefill on the seed sequence (`[decoder_start, lang]`) |
| **decode t/s/u** | `1000 / steady_ms_per_tok` — steady text-decoder step rate (**decode steps 2+**, excludes first-step trace/compile outlier) |
| **E2E** | Full synced `generate()` wall time (includes T2U + vocoder on T2ST/S2ST) |
| **RTF** | Real-time factor on speech tasks: `e2e_s / audio_duration_s` (`<1` = faster than real time) |

**Compare decode t/s/u across tasks** — unlike legacy E2E `tokens/s`, it is not penalized by long input encoders (S2TT/ASR) or variable output length.

Measured **2026-06-06** on BH QB from **one warm demo run** (3rd consecutive `demo/demo.py` invocation that day, JIT cache warm — not a cold-start process). All five tasks are from the **same** printed summary table.

### BH QB — `MeshShape(1, 4)`, replicated batch-1

| Task | TTFT | Encoder | Prefill | decode t/s/u | ms/tok (steady) | E2E | Output |
|------|-----:|--------:|--------:|-------------:|----------------:|----:|--------|
| T2TT | 129.5 ms | 24.7 ms | 50.0 ms | 116.6 | 8.6 | 682.6 ms | 64 tok |
| T2ST | 205.4 ms | 57.3 ms | 67.5 ms | 107.6 | 9.3 | 4128.6 ms | 232000 smp (RTF **0.28×**) |
| S2TT | 873.4 ms | 718.0 ms | 97.7 ms | 81.2 | 12.3 | 1186.5 ms | 26 tok (479 mel) |
| S2ST | 1002.4 ms | 811.5 ms | 102.8 ms | 90.0 | 11.1 | 4040.1 ms | 146880 smp (RTF **0.44×**) |
| ASR | 1103.1 ms | 872.2 ms | 73.2 ms | 76.1 | 13.1 | 1476.4 ms | 29 tok (479 mel) |

Speech-synthesis breakdown (same run):

| Task | T2U | Vocoder | RTF |
|------|----:|--------:|----:|
| T2ST | 1736 ms | 1285 ms | 0.28× |
| S2ST | 1025 ms | 1560 ms | 0.44× |

Reproduce:

```bash
python models/experimental/seamless_m4t_v2_large/demo/demo.py
```

Task notes:
- **T2TT** — text encoder + traced text-decoder loop.
- **T2ST** — text path + T2U + vocoder; vocoder dominates E2E; use **RTF** for speech QoS.
- **S2TT / ASR** — speech encoder dominates TTFT; **decode t/s/u** isolates the text-decoder steady rate.
- **S2ST** — speech encoder + decoder + T2U + vocoder.

**Cold start:** the first timed speech-synthesis call in a **brand-new process** with a cold on-disk JIT cache can still pay a one-time vocoder compile outlier (~15–20 s). Subsequent timed iters in the same session and later demo invocations (warm disk cache) report steady vocoder times (~1.1–1.6 s on BH QB). Speech warmups + vocoder prewarm keep the **reported** timed iter near steady state.

### Device kernel perf (Tracy, eager, no trace)

```bash
pytest models/experimental/seamless_m4t_v2_large/tests/perf/test_seamless_device_perf.py \
    -v -m models_device_performance_bare_metal
```

The outer driver spawns eager forwards in [`tests/perf/test_device_perf_forwards.py`](tests/perf/test_device_perf_forwards.py) under Tracy and reports **both** Tracy per-device kernel floors and the same TT-aligned wall metrics (via JSON side-channel).

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

### Repeatability check (optional)

Same-process stability on the demo text input (T2TT + T2ST ×3):

```bash
python models/experimental/seamless_m4t_v2_large/scripts/check_determinism.py
```

Full regression bundle (PCC + demo ×4 + determinism):

```bash
models/experimental/seamless_m4t_v2_large/scripts/multi_run_report.sh
```

### Device-level performance (kernel-only)

```bash
pytest models/experimental/seamless_m4t_v2_large/tests/perf/test_seamless_device_perf.py \
    -v -m models_device_performance_bare_metal
```

The outer driver spawns eager forward-only inner tests in [`tests/perf/test_device_perf_forwards.py`](tests/perf/test_device_perf_forwards.py) under Tracy (`use_decode_trace=False`, `use_2cq=False`). It reports Tracy per-device kernel floors plus TT-aligned wall metrics from ``return_timings=True`` (see **Performance** above).

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
│   ├── download_weights.py              # HF snapshot downloader + CLI
│   ├── check_determinism.py             # Same-process T2TT/T2ST repeatability check
│   └── multi_run_report.sh              # PCC + demo ×N + determinism runner
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
- **L1 budget.** Speech-generation paths (T2U + vocoder) require `l1_small_size=65536` in device params. Long mel inputs use chunked 1D matmul in the speech encoder above `_LONG_AUDIO_RES_DRAM_THRESHOLD = 1024` ([`tt/tt_speech_encoder.py`](tt/tt_speech_encoder.py)).

### API scope versus Hugging Face

| HF capability | TTNN port |
|---------------|-----------|
| `output_attentions` / `output_hidden_states` | **Not supported** |
| `generate()` beam search (`num_beams > 1`) | **Supported** with KV cache — beam scoring runs on **host PyTorch** (log-softmax / top-*k*); not compatible with decode trace or 2CQ |
| `generate()` sampling (`do_sample=True`, temperature, top-*p*, top-*k*) | **Supported** — sampling math runs on **host PyTorch** after a logits readback; not compatible with decode trace or 2CQ |
| `generate()` `repetition_penalty` | **Supported** for greedy decode and beam search (host PyTorch, HF rule). **Not applied** when `do_sample=True`. Must be `>= 1.0`; default **1.0** (disabled) |
| `generate()` `batch_size > 1` | **Not supported** — batch size 1 only |

### Host / PyTorch work in `generate()`

The TT model runs encoders, decoder, T2U, and vocoder on device, but **`TTSeamlessM4Tv2Model.generate()` is not a fully device-resident autoregressive loop**. Several steps still execute on the host CPU using Python and **PyTorch** (transport, scoring, or HF-parity string logic). This is intentional for HF alignment and for decode modes that cannot be captured in a Metal trace.

**Autoregressive text decode (all tasks)**

| Step | Where | Notes |
|------|--------|-------|
| Decode loop control | Host (Python) | One iteration per output token; EOS checked against a host-maintained `seq_host` list |
| Per-step token / position upload | Host → device | Reused `torch.int32` staging buffers + `copy_host_to_device_tensor` (required while decode trace is active — no device writes during trace replay) |
| Greedy token pick (TP=4, traced path, `repetition_penalty=1.0`) | Device + host | Device fused/chunked argmax; **chunk max + local index read back** and combined on host with PyTorch to match `torch.argmax` tie-breaking — **no logits row readback** |
| Repetition penalty (`repetition_penalty > 1.0`) | Host (PyTorch) | HF **`RepetitionPenaltyLogitsProcessor`**: for each already-emitted token id, logits `< 0` are multiplied by penalty, logits `>= 0` are divided. Applied on a gathered **`[V]`** logits row in torch, then **`argmax`**. **Never runs on device.** With trace+2CQ, chunk-argmax is tried first; if the unpenalized winner is a repeat token, falls back to full-row D2H + penalty + argmax |
| Greedy token pick (eager / no trace) | Device or host | Device `ttnn.argmax` when `repetition_penalty=1.0`; full logits row readback + host penalty + argmax when `repetition_penalty > 1.0` |
| Beam search (`num_beams > 1`) | Host (PyTorch) | Per-beam logits read back; **`log_softmax`**, repetition penalty on emitted ids, and **`topk`** beam scoring on host; KV caches reordered with `ttnn.copy` |
| Sampling (`do_sample=True`) | Host (PyTorch) | Logits row read back; temperature / top-*k* / top-*p* filtering and **`multinomial`** on host — **not traced**; **`repetition_penalty` is ignored** (HF applies it via logits processors before sampling) |
| Sequence bookkeeping | Host (Python) | Token ids accumulated in Python lists; `ttnn.from_torch` used to rebuild `sequences_tt` after the loop |

**Speech generation path (T2ST / S2ST only, after text decode)**

| Step | Where | Notes |
|------|--------|-------|
| Trailing pad / EOS trim | Host (Python) | `_trim_seq_host_for_speech` before T2U |
| Subword → character tables | Host (Python) | `generation_config.id_to_text`, `char_to_id`, and HF-style **`_char_count_per_subword`** string analysis |
| T2U char ids / duration counts | Host (Python + torch transport) | Character id lists and `char_count_per_id` built on host, uploaded with `from_torch` |
| T2U unit id → vocoder vocab remap | Host (PyTorch) | T2U **`argmax`** on device, then unit ids + padding mask read back; EOS/pad masking and **`vocoder_offset`** applied with `torch.where`, re-uploaded for the vocoder |

**Other host touches**

- **`generation_config` lookups** — target language code ids, EOS id sets, subword/char tables (string dict ops).
- **Scalar readbacks** — subsampled speech-encoder length (for slice bounds), per-decode-step greedy token id (EOS), and optional profiler/signpost hooks in tests.
- **Pre/post outside `generate()`** — demo and tests still run Hugging Face **`AutoProcessor`** tokenization and feature extraction on host before uploading tensors.

**Production implication:** the demo and documented perf numbers use **greedy decode + per-step KV trace + 2CQ** with **`repetition_penalty=1.0`** (HF default, penalty disabled). That keeps decode on the fast device chunk-argmax path with only chunk-index readback and EOS checks on host. Setting **`repetition_penalty > 1.0`** adds per-step host logits gather + penalty work (and occasional full-row fallback on the traced path). Beam search and sampling disable trace/2CQ and move most decode scoring to host PyTorch. A full end-to-end Metal trace of `generate()` is not possible while these host-dependent control paths remain (see module docstring in [`tt/tt_seamless_m4t_v2_model.py`](tt/tt_seamless_m4t_v2_model.py)).

### Utterance-level model: long inputs degenerate (text and speech)

SeamlessM4T v2 is trained on short utterances. Given a **long input** (text *or* audio) the model loses coherence — it drops content, repeats phrases, and emits EOS early — and on speech it may *translate* rather than transcribe. **This is the Hugging Face model's behavior, not a TTNN bug**: the bf16 reference degenerates the same way on the same input. The demo uses a long English paragraph for T2TT/T2ST to stress the text path; speech-input tasks use the shorter ~9.6 s preamble clip. Accuracy on long inputs should be judged by **TT-vs-HF faithfulness** (chrF / CER versus the bf16 reference on the same input), not by absolute correctness. Production pipelines typically segment long audio/text into utterance-sized chunks.

### Cross-run stability and TT-vs-HF parity

On **`MeshShape(1, 4)`** with the default demo settings (greedy decode, trace + 2CQ, `repetition_penalty=1.0`), **identical inputs produce stable TT outputs across repeated runs** for the demo text and preamble speech inputs (verified with [`scripts/check_determinism.py`](scripts/check_determinism.py) and multi-run demo logs).

A prior regression that caused **S2TT (and other speech-path tasks) to emit repetitive token loops** was traced to using `ttnn.all_reduce` for decoder / speech / T2U TP reductions. That path is **fixed**: gather+sum is restored for those modules; `ttnn.all_reduce` is kept only on the text encoder. Do not blanket-switch all TP reductions to `all_reduce` without validating the speech-encoder L1 path and residual adds.

TT outputs are **not required to be bit-identical to Hugging Face** on every task (integrated PCC uses chrF / CER / plausible-voiced gates, not exact token or sample match). Residual gaps include:

- **T2ST / S2ST waveform length** can differ from HF by a modest sample count while still passing PCC voicing checks.
- **Strict bit-reproducibility** (every greedy step identical run-to-run under all TP tie cases) is not yet guaranteed — see **To Do**.

Phrase **repetition on very long inputs** (text or audio) is expected **HF model behavior** on utterance-scale inputs, not a TT-only bug — see **Utterance-level model** above.

### Vocoder throughput (mitigated)

Vocoder conv timelines are **length-bucketed** (short single-shot and chunked/upsampled paths via `_vocoder_timeline_bucket` / `_slice_nlc_time` in [`tt/tt_code_hifigan.py`](tt/tt_code_hifigan.py); override with `SEAMLESS_VOCODER_CONV1D_BUCKET`). Speech `generate()` **preserves vocoder prep/program cache** across decode/T2U program evictions (`_clear_decode_and_t2u_programs(preserve_vocoder=True)`). Together with demo warmups this removes the prior ~8–25 s vocoder recompile on every in-process speech iter; see **Performance → Cold start** for the remaining one-time JIT outlier on a cold process.

---

## To Do

- **Bit-exact deterministic decode.** Optional hardening: deterministic multi-device reductions (TTNN / CCL) on decoder / speech / T2U without breaking the speech-encoder L1 path — cross-run stability on demo inputs is already restored with gather+sum.
- **End-to-end validation at 4096-token scale.** Text-decoder cross-attention prefill PCC now reaches **1024** encoder frames; full ``generate()`` on max-length text/speech inputs still needs E2E certification.
- **T2ST / S2ST waveform length vs HF.** Investigate remaining sample-count gap versus the bf16 reference while keeping plausible-voiced PCC gates.
- **Utterance segmentation in demo.** Optional VAD / sentence splitting for long-form inputs so chained tasks stay in the target language.

---

## References

- [SeamlessM4T paper (Seamless Communication Team, 2023)](https://arxiv.org/abs/2312.05187)
- [facebook/seamless-m4t-v2-large on HuggingFace](https://huggingface.co/facebook/seamless-m4t-v2-large)
- [HuggingFace Transformers `modeling_seamless_m4t_v2.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/seamless_m4t_v2/modeling_seamless_m4t_v2.py)
- [Tenstorrent TT-Metalium](https://github.com/tenstorrent/tt-metal)
