# Seamless M4T v2 Large on Tenstorrent Hardware

## Platforms


| Device                | Status    | Notes                                                                                                                          |
| --------------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **BH QB (Blackhole)** | Supported | `MeshShape(1, 4)`, `FABRIC_1D` — four chips, replicated batch-1, all five tasks, 2CQ + per-step KV-decode trace + `generate()` |


This port targets **Blackhole QB** hosts with **four** Tenstorrent devices. Mesh shape and `device_params` come from `[tt/mesh_helpers.py](tt/mesh_helpers.py)` (`open_seamless_mesh_device()` for the demo; pytest fixtures use `MeshShape(1, 4)`). There is no Wormhole, Grayskull, or single-chip path in the supported/tested configuration.

PCC tests and the Tracy device-perf driver use `l1_small_size=65536` on speech-generation paths where required.

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

Limits below are what the TT port exercises on **BH QB `MeshShape(1, 4)`**. HF config allows up to **4096** positions on the text side; per-module PCC tests pin the longest shapes that pass at **PCC ≥ 0.99**.


|                                   | **Text input** (T2TT, T2ST)                                                                                                                                                                                                  | **Speech input** (S2TT, S2ST, ASR)                                                                                                                                                                                    |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Input unit**                    | Tokenized source tokens (`processor(text=..., src_lang=...)`)                                                                                                                                                                | Log-mel frames from 16 kHz audio (`processor(audios=..., sampling_rate=16000)`)                                                                                                                                       |
| **Design / HF maximum**           | **4096** source tokens (`max_position_embeddings`)                                                                                                                                                                           | **4096** mel frames                                                                                                                                                                                                   |
| **Longest shape validated (PCC)** | Text encoder forward @ **4096** tokens (`[test_text_encoder.py](tests/pcc/test_text_encoder.py)`); text-decoder cross-attention prefill @ **1024** encoder frames (`[test_text_decoder.py](tests/pcc/test_text_decoder.py)`) | Speech encoder forward @ **4096** mel frames (`[test_speech_encoder.py](tests/pcc/test_speech_encoder.py)`); text-decoder cross-attention prefill @ **1024** subsampled encoder frames (same decoder test, S2TT path) |
| **Typical demo input**            | Joyce-style English paragraph                                                                                                                                                                                                | Preamble WAV resampled to 16 kHz: **~479** mel frames (~9.6 s)                                                                                                                                                        |


Notes:

- **Encoder timeline vs raw input.** For text tasks the encoder timeline equals the tokenized source length (1 token → 1 text-encoder frame). For speech tasks the Conformer stack plus length adaptor (kernel/stride **8**) subsamples mel into a shorter encoder timeline fed to the text decoder (~8× shorter than mel length at the upper bound).
- **Decoder cross-attention prefill.** Text-decoder prefill is PCC-validated up to **1024** encoder frames on BH 1×4 (`[test_text_decoder.py](tests/pcc/test_text_decoder.py)`; `MAX_ENC_SEQ=1024`).
- **End-to-end `generate()` at max length.** `[scripts/demo_perf_sweep.py](scripts/demo_perf_sweep.py)` exercises all five tasks at **4096** tokens / mel frames (warm JIT, split speech warmups at mel ≥ 1792). E2E **PCC** against HF at that scale is not yet certified — the sweep is a runtime/throughput harness, not a correctness gate.
- **Decoder KV budget.** `TTSeamlessM4Tv2Model` allocates text-decoder KV cache for `**max_text_seq_len=4096`** (seed + generated tokens). T2U is separately validated at **4096** encoder frames (`[test_text_to_unit.py](tests/pcc/test_text_to_unit.py)`).
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

The demo runs on `**MeshShape(1, 4)**` via `open_seamless_mesh_device()` with **2CQ + per-step KV-decode trace**. Each task opens its **own** mesh device. Speech paths (T2ST, S2ST) run **two untimed warmups**, a vocoder conv prewarm from cached shapes, then **two timed** `generate()` calls (phase timings taken from the faster timed iter). Text paths use one warmup and one timed iter.

**Text input (tasks 1–2):** an English sentence is hardcoded in `[demo/demo.py](demo/demo.py)` — a Joyce-style passage (`src_lang=eng`):

> going along slushy country roads and speaking to damp audiences in draughty schoolrooms day after day for a fortnight he'll have to put in an appearance at some place of worship on sunday morning and he can come to us immediately afterwards

**Speech input (tasks 3–5):** the US Constitution preamble read aloud, from `[preamble10.wav](https://www.cs.kzoo.edu/cs107/MediaSources/preamble10.wav)` (Kalamazoo College CS107 media). On first run `ensure_demo_audio()` downloads it to `demo/outputs/preamble10.wav` (raises if download fails). The WAV is mono, resampled from 22050 Hz to **16 kHz** before feature extraction (~9.6 s, ~479 mel frames).

Task order:

1. **T2TT** — text above (`eng`) → Hindi text
2. **T2ST** — same text (`eng`) → Hindi speech (saved to `demo/outputs/t2st_hindi_speech.wav`)
3. **S2TT** — preamble speech (`eng`) → Hindi text
4. **S2ST** — preamble speech (`eng`) → Spanish speech (saved to `demo/outputs/s2st_spanish_speech.wav`)
5. **ASR** — preamble speech (`eng`) → English text

### Sequence-length performance sweep

`[scripts/demo_perf_sweep.py](scripts/demo_perf_sweep.py)` runs the same five tasks while doubling input length **32 → 4096** (text = source token count; speech = mel-frame count). Long inputs are prepared once:

- **Text** — *Alice in Wonderland* from Project Gutenberg (cached under `scripts/outputs/alice_in_wonderland.txt`).
- **Audio** — preamble WAV concatenated until ≥ 4096 mel frames (cached as `scripts/outputs/long_speech_input.wav`).

At mel lengths **≥ 1792**, speech warmups run on a **throwaway** mesh device so the timed session stays decode-trace clean (fixes S2TT/S2ST/ASR collapse that appeared when multiple speech `generate()` calls shared one session).

```bash
python models/experimental/seamless_m4t_v2_large/scripts/demo_perf_sweep.py
# optional: --min-len 32 --max-len 512 --output scripts/outputs/perf_sweep.txt
```

Log, per-length WAVs, and summary tables land under `scripts/outputs/` (default log: `perf_sweep.txt`).

---

## Performance (Blackhole BH QB, 2CQ + decode trace)

Phase-separated timings from `[scripts/demo_perf_sweep.py](scripts/demo_perf_sweep.py)` with `generate(return_timings=True)` on a four-chip Blackhole QB host (`MeshShape(1, 4)`, batch-1 replicated TP=4). Alice in Wonderland text + concatenated preamble audio; sequence lengths double **32 → 4096** (text = source tokens; speech = mel frames). Host pre/post-processing (token decode, WAV I/O) is excluded. Each task opens its own mesh device; speech paths use split warmups at mel ≥ 1792.

Metrics follow the TT model catalog (Whisper / LLM / Qwen3-TTS style):


| Metric           | Meaning                                                                                                                     |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **TTFT**         | Time from `generate()` start to first **new** decoder token (includes encoder + decoder prefill + first decode step)        |
| **Encoder**      | Speech or text encoder only                                                                                                 |
| **Prefill**      | Text-decoder KV prefill on the seed sequence (`[decoder_start, lang]`)                                                      |
| **decode t/s/u** | `1000 / steady_ms_per_tok` — steady text-decoder step rate (**decode steps 2+**, excludes first-step trace/compile outlier) |
| **E2E**          | Full synced `generate()` wall time (includes T2U + vocoder on T2ST/S2ST)                                                    |
| **RTF**          | Real-time factor on speech tasks: `e2e_s / audio_duration_s` (`<1` = faster than real time)                                 |


**Compare decode t/s/u across tasks** — unlike legacy E2E `tokens/s`, it is not penalized by long input encoders (S2TT/ASR) or variable output length. At very short lengths (e.g. 32 mel) decode t/s/u is noisy because only a handful of decoder steps run.

Tables below are the per-length summary blocks from that log.

Reproduce:

```bash
python models/experimental/seamless_m4t_v2_large/scripts/demo_perf_sweep.py
```

### Sequence length: 32


| Task | TTFT      | Encoder  | Prefill  | decode t/s/u | ms/tok (steady) | E2E       | Output                     |
| ---- | --------- | -------- | -------- | ------------ | --------------- | --------- | -------------------------- |
| T2TT | 151.8 ms  | 23.6 ms  | 48.2 ms  | 107.1        | 9.3             | 447.6 ms  | 32 tok                     |
| T2ST | 215.4 ms  | 35.1 ms  | 59.5 ms  | 100.3        | 10.0            | 2564.0 ms | 100480 smp (RTF **0.41×**) |
| S2TT | 869.1 ms  | 709.7 ms | 89.3 ms  | 7.9          | 127.1           | 997.2 ms  | 2 tok (32 mel)             |
| S2ST | 976.2 ms  | 778.6 ms | 100.5 ms | 36.9         | 27.1            | 3101.0 ms | 11840 smp (RTF **4.19×**)  |
| ASR  | 1080.4 ms | 864.5 ms | 69.8 ms  | 18.4         | 54.5            | 1245.3 ms | 4 tok (32 mel)             |



| Task | T2U    | Vocoder | RTF   |
| ---- | ------ | ------- | ----- |
| T2ST | 872 ms | 983 ms  | 0.41× |
| S2ST | 938 ms | 987 ms  | 4.19× |


### Sequence length: 64


| Task | TTFT      | Encoder  | Prefill | decode t/s/u | ms/tok (steady) | E2E       | Output                     |
| ---- | --------- | -------- | ------- | ------------ | --------------- | --------- | -------------------------- |
| T2TT | 140.3 ms  | 25.1 ms  | 48.3 ms | 130.8        | 7.6             | 1074.3 ms | 120 tok                    |
| T2ST | 232.6 ms  | 63.7 ms  | 66.4 ms | 125.5        | 8.0             | 5452.0 ms | 315520 smp (RTF **0.28×**) |
| S2TT | 972.6 ms  | 774.9 ms | 95.8 ms | 18.2         | 55.0            | 1083.9 ms | 3 tok (64 mel)             |
| S2ST | 1065.6 ms | 859.6 ms | 68.1 ms | 24.2         | 41.3            | 3355.4 ms | 24960 smp (RTF **2.15×**)  |
| ASR  | 1149.8 ms | 918.0 ms | 71.5 ms | 17.3         | 57.7            | 1324.5 ms | 4 tok (64 mel)             |



| Task | T2U     | Vocoder | RTF   |
| ---- | ------- | ------- | ----- |
| T2ST | 1636 ms | 2262 ms | 0.28× |
| S2ST | 971 ms  | 1081 ms | 2.15× |


### Sequence length: 128


| Task | TTFT      | Encoder  | Prefill  | decode t/s/u | ms/tok (steady) | E2E       | Output                     |
| ---- | --------- | -------- | -------- | ------------ | --------------- | --------- | -------------------------- |
| T2TT | 177.0 ms  | 25.9 ms  | 48.5 ms  | 124.3        | 8.0             | 1084.3 ms | 111 tok                    |
| T2ST | 244.1 ms  | 69.6 ms  | 98.0 ms  | 90.8         | 11.0            | 5331.9 ms | 348160 smp (RTF **0.25×**) |
| S2TT | 1007.0 ms | 796.0 ms | 105.5 ms | 42.1         | 23.8            | 1199.3 ms | 9 tok (128 mel)            |
| S2ST | 1086.7 ms | 866.3 ms | 68.7 ms  | 44.5         | 22.5            | 3623.3 ms | 37120 smp (RTF **1.56×**)  |
| ASR  | 1189.1 ms | 944.1 ms | 72.1 ms  | 41.3         | 24.2            | 1410.0 ms | 10 tok (128 mel)           |



| Task | T2U     | Vocoder | RTF   |
| ---- | ------- | ------- | ----- |
| T2ST | 1683 ms | 1761 ms | 0.25× |
| S2ST | 1024 ms | 1221 ms | 1.56× |


### Sequence length: 256


| Task | TTFT      | Encoder   | Prefill | decode t/s/u | ms/tok (steady) | E2E       | Output                     |
| ---- | --------- | --------- | ------- | ------------ | --------------- | --------- | -------------------------- |
| T2TT | 149.3 ms  | 27.6 ms   | 48.8 ms | 136.0        | 7.4             | 1823.4 ms | 223 tok                    |
| T2ST | 292.3 ms  | 75.8 ms   | 68.1 ms | 130.5        | 7.7             | 6953.4 ms | 563520 smp (RTF **0.20×**) |
| S2TT | 1109.9 ms | 888.5 ms  | 72.7 ms | 58.7         | 17.0            | 1351.7 ms | 15 tok (256 mel)           |
| S2ST | 1221.9 ms | 992.1 ms  | 70.2 ms | 64.5         | 15.5            | 4567.1 ms | 83520 smp (RTF **0.87×**)  |
| ASR  | 1357.2 ms | 1085.9 ms | 76.1 ms | 46.2         | 21.7            | 1707.9 ms | 17 tok (256 mel)           |



| Task | T2U     | Vocoder | RTF   |
| ---- | ------- | ------- | ----- |
| T2ST | 1762 ms | 2559 ms | 0.20× |
| S2ST | 1133 ms | 1777 ms | 0.87× |


### Sequence length: 512


| Task | TTFT      | Encoder  | Prefill  | decode t/s/u | ms/tok (steady) | E2E       | Output                     |
| ---- | --------- | -------- | -------- | ------------ | --------------- | --------- | -------------------------- |
| T2TT | 151.3 ms  | 28.1 ms  | 50.1 ms  | 132.3        | 7.6             | 2125.5 ms | 256 tok                    |
| T2ST | 303.1 ms  | 78.0 ms  | 107.4 ms | 128.8        | 7.8             | 7208.7 ms | 541120 smp (RTF **0.21×**) |
| S2TT | 1026.6 ms | 805.8 ms | 73.9 ms  | 81.7         | 12.2            | 1363.2 ms | 28 tok (512 mel)           |
| S2ST | 1164.3 ms | 927.8 ms | 70.8 ms  | 83.0         | 12.1            | 4667.1 ms | 156160 smp (RTF **0.48×**) |
| ASR  | 1288.5 ms | 933.9 ms | 77.0 ms  | 65.5         | 15.3            | 1737.4 ms | 30 tok (512 mel)           |



| Task | T2U     | Vocoder | RTF   |
| ---- | ------- | ------- | ----- |
| T2ST | 1759 ms | 2537 ms | 0.21× |
| S2ST | 1131 ms | 1819 ms | 0.48× |


### Sequence length: 1024


| Task | TTFT      | Encoder   | Prefill  | decode t/s/u | ms/tok (steady) | E2E       | Output                     |
| ---- | --------- | --------- | -------- | ------------ | --------------- | --------- | -------------------------- |
| T2TT | 153.7 ms  | 27.6 ms   | 50.1 ms  | 123.6        | 8.1             | 2265.3 ms | 256 tok                    |
| T2ST | 342.1 ms  | 96.3 ms   | 81.2 ms  | 121.2        | 8.2             | 9035.7 ms | 799360 smp (RTF **0.18×**) |
| S2TT | 1165.1 ms | 917.1 ms  | 81.5 ms  | 95.6         | 10.5            | 1719.8 ms | 53 tok (1024 mel)          |
| S2ST | 1519.5 ms | 1257.7 ms | 78.0 ms  | 102.2        | 9.8             | 8403.0 ms | 286400 smp (RTF **0.47×**) |
| ASR  | 1868.7 ms | 1359.5 ms | 171.8 ms | 79.1         | 12.6            | 2626.4 ms | 60 tok (1024 mel)          |



| Task | T2U     | Vocoder | RTF   |
| ---- | ------- | ------- | ----- |
| T2ST | 1811 ms | 3860 ms | 0.18× |
| S2ST | 2235 ms | 3674 ms | 0.47× |


### Sequence length: 2048


| Task | TTFT      | Encoder   | Prefill  | decode t/s/u | ms/tok (steady) | E2E        | Output                     |
| ---- | --------- | --------- | -------- | ------------ | --------------- | ---------- | -------------------------- |
| T2TT | 179.4 ms  | 38.3 ms   | 50.6 ms  | 111.1        | 9.0             | 2521.8 ms  | 256 tok                    |
| T2ST | 318.4 ms  | 90.8 ms   | 71.8 ms  | 108.8        | 9.2             | 8106.2 ms  | 644800 smp (RTF **0.20×**) |
| S2TT | 8244.7 ms | 8022.0 ms | 74.4 ms  | 106.9        | 9.4             | 8975.2 ms  | 77 tok (2048 mel)          |
| S2ST | 8649.5 ms | 8394.1 ms | 73.1 ms  | 106.9        | 9.4             | 14272.6 ms | 300480 smp (RTF **0.76×**) |
| ASR  | 8908.3 ms | 8540.1 ms | 151.8 ms | 95.5         | 10.5            | 9773.6 ms  | 82 tok (2048 mel)          |



| Task | T2U     | Vocoder | RTF   |
| ---- | ------- | ------- | ----- |
| T2ST | 1781 ms | 2941 ms | 0.20× |
| S2ST | 1992 ms | 2695 ms | 0.76× |


### Sequence length: 4096


| Task | TTFT      | Encoder   | Prefill  | decode t/s/u | ms/tok (steady) | E2E       | Output                     |
| ---- | --------- | --------- | -------- | ------------ | --------------- | --------- | -------------------------- |
| T2TT | 315.9 ms  | 149.4 ms  | 51.0 ms  | 91.1         | 11.0            | 3164.1 ms | 256 tok                    |
| T2ST | 463.4 ms  | 160.9 ms  | 111.7 ms | 90.4         | 11.1            | 8761.7 ms | 647040 smp (RTF **0.22×**) |
| S2TT | 2223.4 ms | 1995.0 ms | 73.7 ms  | 115.7        | 8.6             | 3271.3 ms | 119 tok (4096 mel)         |
| S2ST | 2199.5 ms | 1966.2 ms | 71.4 ms  | 95.1         | 10.5            | 7158.0 ms | 226240 smp (RTF **0.51×**) |
| ASR  | 2523.2 ms | 2241.1 ms | 75.9 ms  | 104.0        | 9.6             | 3514.6 ms | 102 tok (4096 mel)         |



| Task | T2U     | Vocoder | RTF   |
| ---- | ------- | ------- | ----- |
| T2ST | 1796 ms | 2959 ms | 0.22× |
| S2ST | 2025 ms | 2183 ms | 0.51× |


Task notes:

- **T2TT** — text encoder + traced text-decoder loop.
- **T2ST** — text path + T2U + vocoder; vocoder dominates E2E; use **RTF** for speech QoS.
- **S2TT / ASR** — speech encoder dominates TTFT; **decode t/s/u** isolates the text-decoder steady rate.
- **S2ST** — speech encoder + decoder + T2U + vocoder.

**Cold start:** the first timed speech-synthesis call in a **brand-new process** with a cold on-disk JIT cache can still pay a one-time vocoder compile outlier (~~15–20 s). Subsequent timed iters in the same session and later demo invocations (warm disk cache) report steady vocoder times (~~1.1–1.6 s on BH QB). Speech warmups + vocoder prewarm keep the **reported** timed iter near steady state.

### Device kernel perf (Tracy, eager, no trace)

```bash
pytest models/experimental/seamless_m4t_v2_large/tests/perf/test_seamless_device_perf.py \
    -v -m models_device_performance_bare_metal
```

The outer driver spawns eager forwards in `[tests/perf/test_device_perf_forwards.py](tests/perf/test_device_perf_forwards.py)` under Tracy and reports **both** Tracy per-device kernel floors and the same TT-aligned wall metrics (via JSON side-channel).

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

All PCC tests pass at `PCC_THRESHOLD = 0.99` (`[tests/pcc/test_seamless_m4t_v2_model.py](tests/pcc/test_seamless_m4t_v2_model.py)`). E2E generate PCC uses prefix/voicing gates (not bit-identical token match) on demo-length inputs; T2ST asserts HF/TT sample-count ratio within **8%**.

### Device-level performance (kernel-only)

```bash
pytest models/experimental/seamless_m4t_v2_large/tests/perf/test_seamless_device_perf.py \
    -v -m models_device_performance_bare_metal
```

The outer driver spawns eager forward-only inner tests in `[tests/perf/test_device_perf_forwards.py](tests/perf/test_device_perf_forwards.py)` under Tracy (`use_decode_trace=False`, `use_2cq=False`). It reports Tracy per-device kernel floors plus TT-aligned wall metrics from `return_timings=True` (see **Performance** above).

---

## Repository Layout

```
models/experimental/seamless_m4t_v2_large/
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
│   └── outputs/                         # Generated: perf_sweep.txt, sweep WAVs, cached inputs
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
│   ├── common.py                        # TP reductions, hf_aligned_generation_kwargs
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

- **Blackhole QB only.** PCC and perf tests run on Blackhole (`@run_for_blackhole()`). Supported mesh is `**MeshShape(1, 4)`** with `FABRIC_1D` (see `[tt/mesh_helpers.py](tt/mesh_helpers.py)`). Requires a host with four devices.
- **L1 budget.** Speech-generation paths (T2U + vocoder) require `l1_small_size=65536` in device params. Long mel inputs switch the speech encoder to chunked 1D matmul and DRAM residuals above `_LONG_AUDIO_RES_DRAM_THRESHOLD = 512` (`[tt/tt_speech_encoder.py](tt/tt_speech_encoder.py)`). Text-encoder block-sharded TP matmul falls back to interleaved DRAM above **2048** tokens (`[tt/tt_text_encoder.py](tt/tt_text_encoder.py)`).
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

### Utterance-level model: long inputs degenerate (text and speech)

SeamlessM4T v2 is trained on short utterances. Given a **long input** (text *or* audio) the model loses coherence — it drops content, repeats phrases, and emits EOS early — and on speech it may *translate* rather than transcribe. **This is the Hugging Face model's behavior, not a TTNN bug**: the bf16 reference degenerates the same way on the same input. The demo uses a long English paragraph for T2TT/T2ST to stress the text path; speech-input tasks use the shorter ~9.6 s preamble clip. Accuracy on long inputs should be judged by **TT-vs-HF faithfulness** (chrF / CER versus the bf16 reference on the same input), not by absolute correctness. Production pipelines typically segment long audio/text into utterance-sized chunks.

### Cross-run stability and TT-vs-HF parity

On `**MeshShape(1, 4)`** with the default demo settings (greedy decode, trace + 2CQ, `repetition_penalty=1.0`), **identical inputs produce stable TT outputs across repeated demo runs** for the Joyce paragraph and preamble speech inputs (verified in-process on BH 1×4 alongside the PCC suite).

A prior regression that caused **S2TT (and other speech-path tasks) to emit repetitive token loops** was traced to using `ttnn.all_reduce` for speech / T2U TP reductions. That path is **fixed**: speech encoder and T2U keep `all_gather` + `sum`; the text decoder now uses a separate linear `all_reduce` path that does not deallocate residuals (`[decoder_all_reduce_sum_replicate](tt/common.py)`).

TT outputs are **not required to be bit-identical to Hugging Face** on every task (integrated PCC uses chrF / CER / plausible-voiced gates, not exact token or sample match). Residual gaps include:

- **T2ST / S2ST waveform length** can differ from HF by a modest sample count while still passing PCC voicing checks.
- **Strict bit-reproducibility** (every greedy step identical run-to-run under all TP tie cases) is not yet guaranteed — see **To Do**.

Phrase **repetition on very long inputs** (text or audio) is expected **HF model behavior** on utterance-scale inputs, not a TT-only bug — see **Utterance-level model** above.

### Vocoder throughput (mitigated)

Vocoder conv timelines are **length-bucketed** (short single-shot and chunked/upsampled paths via `_vocoder_timeline_bucket` / `_slice_nlc_time` in `[tt/tt_code_hifigan.py](tt/tt_code_hifigan.py)`; override with `SEAMLESS_VOCODER_CONV1D_BUCKET`). Speech `generate()` **preserves vocoder prep/program cache** across decode/T2U program evictions (`_clear_decode_and_t2u_programs(preserve_vocoder=True)`). Together with demo warmups this removes the prior ~8–25 s vocoder recompile on every in-process speech iter; see **Performance → Cold start** for the remaining one-time JIT outlier on a cold process.

---

## To Do

- **Bit-exact deterministic decode.** Text encoder + text decoder use `ttnn.all_reduce`; speech encoder and T2U still use gather+sum for L1 stability. Traced decode still combines per-shard chunk argmax on host.
- **E2E PCC at max input length.** Per-module PCC reaches 4096 on encoders and 1024 on decoder cross-attn; `demo_perf_sweep.py` exercises `generate()` at 4096, but E2E HF-vs-TT certification at that scale is still open.
- **S2ST waveform length vs HF.** Speech-input S2ST can diverge in intermediate text length vs HF (sample ratio logged in E2E PCC, not gated); T2ST text-path sample ratio is gated within 8%.
- **Utterance segmentation (speech).** Optional VAD / audio chunking for long-form speech inputs (text tasks: `SEAMLESS_DEMO_MAX_SENTENCES` in `[demo/demo.py](demo/demo.py)`).

---

## References

- [SeamlessM4T paper (Seamless Communication Team, 2023)](https://arxiv.org/abs/2312.05187)
- [facebook/seamless-m4t-v2-large on HuggingFace](https://huggingface.co/facebook/seamless-m4t-v2-large)
- [HuggingFace Transformers `modeling_seamless_m4t_v2.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/seamless_m4t_v2/modeling_seamless_m4t_v2.py)`
- [Tenstorrent TT-Metalium](https://github.com/tenstorrent/tt-metal)
