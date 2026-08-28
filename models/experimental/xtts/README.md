# XTTS-v2

## 1. Introduction

XTTS-v2 [(`coqui/XTTS-v2`)](https://huggingface.co/coqui/XTTS-v2) is Coqui's open-weights zero-shot voice-cloning text-to-speech model. It
speaks input text in a voice cloned from a few seconds of reference audio, and outputs 24 kHz mono
audio. Upstream advertises 17 languages; this port defaults to English and does not validate the
rest.

This directory contains the Tenstorrent TTNN implementation. Device implementations:

- **Audio conditioning encoder** — turns the reference waveform into a fixed `[1, 32, 1024]` GPT
  prompt (6-block non-causal attention encoder + a 2-layer PerceiverResampler, 16/8 heads)
- **GPT-2-style decoder** — 30 layers, hidden 1024, 16 heads (head_dim 64), FFN 4096, `gelu_new`,
  causal self-attention; autoregressively emits discrete audio codes over a fixed KV cache
- **Speaker encoder** — SE-ResNet-34 on a 64-mel/16 kHz frontend, attentive statistics pooling →
  512-d embedding `g`
- **HiFi-GAN vocoder** — XTTS-specific generator (4 upsample stages, rates `[8,8,2,2]`), conditioned
  on `g`, turning GPT latents into the waveform

Both mel frontends (80-mel conditioning, 64-mel speaker) also run on device. The host handles only
BPE tokenization, reference-audio load/resample, and the fade on the finished waveform. The whole
pipeline runs inside three chained TTNN traces.

### 1.1 Architecture

XTTS-v2 is a three-stage pipeline. Reference audio produces *two* independent embeddings — 32
conditioning latents that prefix the GPT prompt, and a 512-d speaker vector that conditions the
vocoder — and they are computed from different mel frontends at different sample rates.

```
   text ──► BPE tokenizer ──► [START] ids [STOP] ─────────────┐
                                                              ▼
   ref wav ──┬─► 22.05 kHz ─► 80-mel ─► ConditioningEncoder ─► GPT (30 layers)
             │                          + PerceiverResampler      │  autoregressive
             │                          → [1, 32, 1024] ──────────┘  audio codes + latents
             │                                                       │
             └─► 16 kHz ─► 64-mel ─► SE-ResNet-34 ─► g [1,512,1]     ▼
                                                     └──► HiFi-GAN vocoder ──► 24 kHz wav
```

| Property | Value |
|----------|-------|
| Task | Zero-shot voice cloning TTS: text + reference audio → speech in that voice |
| Max text stream | **404** positions (`gpt.text_pos_embedding.emb.weight` is `[404, 1024]`) |
| Max mel stream | **608** positions = 605 `gpt_max_audio_tokens`; upstream generates at most **602** |
| Vocab | Text 6681 BPE tokens; audio 1026 ids (1024 codes + start/stop sentinels) |
| Voice conditioning | Up to 30 s (`gpt_cond_len`), split into 4 s windows (`gpt_cond_chunk_len`), style embeddings **averaged** |
| Speaker embedding | SE-ResNet-34 → 512-d, L2-normalised. This port uses a 16 s window (see [§7](#7-caveats)) |
| Audio output | 24 kHz mono, batch 1 |
| Frame rate | ≈21.5 codes/s — one code is **46.4 ms** of audio |
| Speech:text ratio | ≈2.2 audio codes per BPE text token |
| Sampling | Greedy argmax, or on-device repetition-penalty → temperature → top-k → top-p → Gumbel-max |
| Precision | GPT bf16 activations / bfloat8_b weights, matmuls at `HiFi2`; HiFi-GAN bf16 activations / bf16 weights, convs at `HiFi4` (see [§6.1](#61-pcc-results)) |

---

## 2. Installation

### Step 1 — Activate environment

Run these at the start of every session, from the repo root:

```bash
source python_env/bin/activate
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
export ARCH_NAME=blackhole
```

> `TT_METAL_HOME` and `PYTHONPATH` must point to the repo root. Without them `import ttnn` resolves
> to the `python_env` wheel rather than the source tree, and tests fail with confusing errors.

### Step 2 — Install Python packages

Use `uv pip install` with the venv active for all installs. `create_venv.sh` builds `python_env`
with `uv venv`, which does **not** put `pip` inside it — `python_env/bin/python -m pip` fails with
`No module named pip`, and the system `pip` installs outside the project virtualenv.

**Core and eval packages** — single install from the repo root:

```bash
source python_env/bin/activate
uv pip install -r models/experimental/xtts/requirements.txt
```

That file adds `soundfile`, `scipy`, `librosa`, `tokenizers`, and CPU `torchaudio` (PyTorch CPU
index). `torch`, `transformers`, `huggingface-hub`, and `jiwer` already come from the tt-metal
dev env (`tt_metal/python_env/requirements-dev.txt`).

> **`librosa==0.10.0` is load-bearing.** Its filterbank feeds every mel in the model, on both the
> TTNN and the reference side, so a version bump moves every PCC number.

`torchaudio==2.11.0+cpu` is required for UTMOS resampling — without it `test_tt_eval*` still runs
and still reports CER and SECS, but logs `UTMOS skipped (ModuleNotFoundError: No module named
'torchaudio')`. The default PyPI wheel links CUDA and fails to load against CPU torch; the
requirements file pulls the CPU wheel via `--extra-index-url https://download.pytorch.org/whl/cpu`.
Equivalent standalone install:

```bash
uv pip install \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  'torchaudio==2.11.0+cpu'
```

### Step 3 — Model weights

Weights are downloaded from HuggingFace on first use and cached under `~/.cache/huggingface/`.
Revisions are pinned in code:

| Artifact | Repo | Used for |
|----------|------|----------|
| `model.pth` (~1.9 GB) | `coqui/XTTS-v2` | all model weights + `mel_stats` |
| `vocab.json` | `coqui/XTTS-v2` | BPE tokenizer |
| sample WAVs | `coqui/XTTS-v2` | default demo reference voice |
| LJSpeech clips | `coqui-ai/TTS` (raw GitHub) | long-reference test/demo audio |

---

## 3. Supported Devices

**Blackhole P150 is the only supported device.**

| Device | Mesh |
|--------|------|
| **P150** | 1×1 |

---

## 4. File Structure

```
models/experimental/xtts/
├── README.md
├── requirements.txt                # Python deps (Step 2 install)
├── config.py                       # ALL tunables: budgets, sampling, chunking, demo defaults
├── demo/
│   ├── xtts_demo.py                # CLI: text + reference audio -> WAV (traced)
│   └── xtts_reference_demo.py      # same CLI, pure-PyTorch, CPU only (no ttnn)
├── eval/
│   └── xtts_eval.py                # objective metrics: CER (Whisper), UTMOS, SECS (ECAPA2)
├── reference/                      # authored, dependency-free PyTorch reimplementation (PCC ground truth)
│   ├── xtts_conditioning.py        # ConditioningEncoder + PerceiverResampler + mel/wav chunking
│   ├── xtts_gpt_block.py           # checkpoint loader + one GPT-2 block + config/causal mask
│   ├── xtts_gpt_generate.py        # greedy AR generation loop
│   ├── xtts_gpt_model.py           # embeddings + stack + text/mel heads
│   ├── xtts_gpt_stack.py           # 30 blocks + ln_f
│   ├── xtts_hifi_decoder.py        # latent upsample + full HifiDecoder
│   ├── xtts_hifigan.py             # XTTS HiFi-GAN generator
│   ├── xtts_inference.py           # full reference pipeline
│   ├── xtts_mel.py                 # speaker mel frontend (64-mel/16 kHz)
│   ├── xtts_speaker_encoder.py     # SE-ResNet-34 + attentive stats pooling
│   └── xtts_text_embedding.py      # BPE tokenizer + text/position embedding
├── tests/
│   ├── pcc/                        # correctness vs the reference — see §5.1
│   └── perf/                       # latency / device-perf / sweeps — see §5.2
└── tt/                             # TTNN implementation
    ├── xtts_conditioning.py
    ├── xtts_conv.py                # TtConv1d / TtConvTranspose1d / TtConv2d primitives
    ├── xtts_full_decoder.py        # mel frontend + speaker encoder + decoder
    ├── xtts_generator.py           # AR decode drivers (free-run / teacher-forced / traced)
    ├── xtts_gpt_block.py
    ├── xtts_gpt_model.py
    ├── xtts_gpt_stack.py
    ├── xtts_hifi_decoder.py        # TtLatentUpsampler + TtHifiDecoder
    ├── xtts_hifigan.py             # TtHifiganGenerator
    ├── xtts_inference.py           # TtXtts — the top-level entry point
    ├── xtts_mel.py                 # TtMelFrontend (speaker) + TtConditioningMel
    ├── xtts_sampler.py             # on-device rep-penalty/temp/top-k/top-p/Gumbel-max
    ├── xtts_speaker_encoder.py
    └── xtts_text_embedding.py
```

## 5. Tests

### 5.0 Test taxonomy

| Layer | Directory | What it proves |
|-------|-----------|----------------|
| **Module PCC** | `tests/pcc/` | each block matches the PyTorch reference on real weights |
| **Pipeline PCC** | `tests/pcc/test_tt_inference.py`, `test_tt_trace.py` | the whole model matches end to end, eager and traced |
| **Performance** | `tests/perf/` | latency, device time, and the length envelope |

The reference implementation under `reference/` is the ground truth for every PCC number. It is an
authored reimplementation with no `coqui-tts` dependency, loading the real upstream checkpoint.

### 5.1 PCC / correctness tests

```bash
# Everything (~13 min warm — the three eval tests are ~9 min of that; auto-downloads weights on first run)
pytest models/experimental/xtts/tests/pcc/ -v

# By block
pytest models/experimental/xtts/tests/pcc/test_conditioning.py       # audio conditioning
pytest models/experimental/xtts/tests/pcc/test_speaker_encoder.py    # SE-ResNet-34
pytest models/experimental/xtts/tests/pcc/test_hifi_decoder.py       # vocoder
pytest models/experimental/xtts/tests/pcc/test_gpt_block.py          # 1 GPT layer, prefill
pytest models/experimental/xtts/tests/pcc/test_gpt_stack.py          # 30 layers + ln_f
pytest models/experimental/xtts/tests/pcc/test_gpt_model.py          # + embeddings + heads
pytest models/experimental/xtts/tests/pcc/test_gpt_decode_layer.py   # decode, 1 layer
pytest models/experimental/xtts/tests/pcc/test_tt_gpt_prefill.py     # prefill KV cache
pytest models/experimental/xtts/tests/pcc/test_gpt_isl_sweep.py      # ISL + cache-depth sweeps
pytest models/experimental/xtts/tests/pcc/test_tt_gpt_generate.py    # AR decode loop
pytest models/experimental/xtts/tests/pcc/test_tt_inference.py       # full pipeline, eager
pytest models/experimental/xtts/tests/pcc/test_tt_trace.py           # full pipeline, traced
```

| File | Covers | Gate |
|------|--------|-----:|
| `test_conditioning.py` | ConditioningEncoder + PerceiverResampler, en + es samples | 0.99 |
| `test_speaker_encoder.py` | SE-ResNet-34; plus a shape-reuse regression (one instance, two mel lengths) | 0.99 |
| `test_hifi_decoder.py` | latent upsample + HiFi-GAN; real GPT latents at 32 and 320 | 0.99 |
| `test_gpt_block.py` | one GPT block, full causal prefill, at 404 and 608 | 0.99 |
| `test_gpt_stack.py` | 30 blocks + `ln_f`, at 404 and 608 | 0.99 |
| `test_gpt_model.py` | embeddings + stack + both heads | 0.99 (separate text / mel knobs) |
| `test_gpt_decode_layer.py` | single-layer decode over 10 steps with a growing cache | 0.99 |
| `test_tt_gpt_prefill.py` | prefill K/V cache contents | 0.99 |
| `test_gpt_isl_sweep.py` | 8 prefill ISLs (32→384); decode over 4 cache depths | 0.99 |
| `test_tt_gpt_generate.py` | free-run + teacher-forced decode vs the reference loop | 0.99 |
| `test_tt_inference.py` | full pipeline eager, end-to-end spectrogram; CER/UTMOS/SECS eval ([§6.2](#62-audio-quality-metrics)) | 0.99 |
| `test_tt_trace.py` | full pipeline traced; traced-vs-eager; session reuse; prompt-pad masking; CER/UTMOS/SECS eval, single utterance + paragraph ([§6.2](#62-audio-quality-metrics)) | 0.99 |

**Every PCC gate is 0.99**, including the end-to-end spectrogram in `test_tt_inference.py`.
Measured values are in [§6.1](#61-pcc-results).

### 5.2 Performance tests

```bash
# Gated e2e baseline
pytest models/experimental/xtts/tests/perf/test_e2e_perf.py -m models_performance_bare_metal

# ISL sweep (8 lengths, one case per ISL on its own device; prints the table in §6.5)
pytest models/experimental/xtts/tests/perf/test_e2e_isl_sweep_perf.py -v -s
pytest models/experimental/xtts/tests/perf/test_e2e_isl_sweep_perf.py -v -s -k isl_96   # one ISL
XTTS_ISL_SWEEP=32,96 pytest models/experimental/xtts/tests/perf/test_e2e_isl_sweep_perf.py -v -s

# Length-envelope sweep
pytest models/experimental/xtts/tests/perf/test_traced_pass_len_sweep.py -v -s

# Device profiling (Tracy) — outer drivers spawn the inner signposted workloads
python models/experimental/xtts/tests/perf/test_device_perf_single_step_prefill.py
python models/experimental/xtts/tests/perf/test_device_perf_single_step_decode.py
```

### 5.3 Demo

```bash
# Default text + default reference voice (downloaded on first use), fully traced
python models/experimental/xtts/demo/xtts_demo.py

# Your own text and voice
python models/experimental/xtts/demo/xtts_demo.py \
    --text "Hello from Tenstorrent." --ref-audio /path/to/voice.wav

# CPU reference, no device
python models/experimental/xtts/demo/xtts_reference_demo.py --temperature 0
```

Output lands at `generated/xtts_demo/xtts_demo.wav` (24 kHz mono); the absolute path is logged.

| Flag | Default | Meaning |
|------|---------|---------|
| `--text` | bundled 2-sentence sample | Text to synthesise |
| `--ref-audio` | 4 LJSpeech clips ≈30 s | Local WAV, `+`-joined coqui clip names, or an HF sample name |
| `--min-tokens` | 0 | STOP-suppression floor in codes. `-1` = auto (~2× padded text length) |
| `--output` | `generated/xtts_demo/xtts_demo.wav` | Where to write the WAV |
| `--write-torch-ref` | off | Also write the CPU-torch reference audio for the same codes, for A/B |

Everything else is fixed in [`config.py`](config.py) so the demo always runs fully traced.

**Text longer than one pass is split automatically** — at sentence boundaries, then at internal
punctuation — and stitched with a 120 ms gap; see [§7](#7-caveats).

---

## 6. Performance and Accuracy

All numbers measured on **Blackhole P150**. The test suites are the source of truth; tables below
are representative runs.

### 6.1 PCC results

| Test | Case | PCC | Gate |
|------|------|----:|-----:|
| `test_conditioning` | `en_sample.wav` | 0.998266 | 0.99 |
| | `es_sample.wav` | 0.998096 | 0.99 |
| `test_gpt_block` | seq_len 404 | 0.999961 | 0.99 |
| | seq_len 608 | 0.999961 | 0.99 |
| `test_gpt_stack` | seq_len 404 | 0.999667 | 0.99 |
| | seq_len 608 | 0.999689 | 0.99 |
| `test_gpt_model` | text_head (64) | 0.998540 | 0.99 |
| | text_head (96) | 0.998601 | 0.99 |
| | mel_head (96) | 0.997558 | 0.99 |
| | mel_head (128) | 0.993741 | 0.99 |
| `test_gpt_decode_layer` | 10 steps | min 0.999937 / mean 0.999966 | 0.99 |
| `test_tt_gpt_prefill` | cache K / V | 0.999987 / 0.999976 | 0.99 |
| `test_gpt_isl_sweep` | decode latent, 8 ISLs 32→384 | 0.997384 – 0.999645 | 0.99 |
| | cache K median (worst ISL) | 0.999771 | 0.99 |
| | decode over max_seq 160/384/608/992 | 0.999095, bit-identical across all four | 0.99 |
| `test_tt_gpt_generate` | free-run codes | 16/16 exact match | — |
| | decode latent | 0.999898 | 0.99 |
| `test_speaker_encoder` | mel_len 200 | 0.999330 | 0.99 |
| | reused instance, mel_len 200 / 512 | 0.998787 / 0.999392 | 0.99 |
| `test_hifi_decoder` | real GPT latents, len 32 | 0.998397 | 0.99 |
| | real GPT latents, len 320 | 0.998639 | 0.99 |
| `test_tt_inference` | end-to-end spectrogram | 0.997045 | 0.99 |
| `test_tt_trace` | traced vs eager spectrogram | 1.0 | 0.99 |

- **Math fidelity is load-bearing — do not drop it for speed.** Two settings carry the pipeline's
  accuracy: `MM_FIDELITY` (`HiFi2`) in [`tt/xtts_gpt_block.py`](tt/xtts_gpt_block.py) for every GPT
  matmul, and `_CONV_FIDELITY` (`HiFi4`) in [`tt/xtts_hifigan.py`](tt/xtts_hifigan.py) for every
  vocoder convolution. Lowering either fails the 0.99 end-to-end gate.
  - ttnn only auto-raises matmul fidelity when *no* `program_config` is passed (the `increase_fidelity`
    heuristic in `matmul_device_operation.cpp`). Every GPT matmul here passes one, so
    `compute_kernel_config` has to be set explicitly — leave it unset and the whole 30-layer stack
    runs at `LoFi`, 5 mantissa bits against `bfloat8_b` weights.
  - The vocoder's bf16 weights × bf16 activations need `HiFi4` to use the full mantissa. `HiFi2`
    there buys ~26% vocoder time, which is not worth taking: the vocoder is under 2% of replay.
  - They are close to free. A/B on one P150b, same binary and same seeded input, only the two
    fidelity constants changed (2 runs each): decode +3.1% (6.167 → 6.356 ms/code), vocoder +9.3%
    (0.1001 → 0.1094 ms/code), setup +1.9%, **RTF +2.9% (0.140 → 0.144)**. The decode and RTF deltas
    are the same size as the run-to-run spread ([§6.4](#64-end-to-end-performance)), and the vocoder
    is under 2% of replay, so its +9.3% barely reaches the total.
- Traced output is bit-identical to eager (PCC 1.0).
- Free-run decode matches the reference 16/16 codes, and teacher-forced top-1 agrees 16/16.
- Decode is bit-identical across cache depths 160/384/608/992 (the attention mask ignores the
  unfilled tail of a fixed cache).

### 6.2 Audio quality metrics

Objective metrics on real sampled on-device generations (`en_sample.wav` reference voice), through
the traced production paths. All three are logged; only the paragraph test asserts anything
(`LONG_CER_MAX = 0.05`, `LONG_SECS_MIN = 0.55`), and none of them run in CI — see [CI](#ci).

| | Single utterance | Paragraph |
|---|---|---|
| Test (`tests/pcc/test_tt_trace.py`) | `test_tt_eval_traced` | `test_tt_eval_traced_long` |
| Input | 1 sentence, 125 chars | 4 sentences, 463 chars |
| Path | one pass, `inference_fully_traced` | chunked, one `traced_session` capture |
| Generated | 155 codes → 7.19 s, self-terminated | 4 chunks (151/145/129/133), 558 codes → 26.26 s |

| Metric | Tool | Target | Single utterance | Paragraph |
|:-------|:----:|:------:|:----------------:|:---------:|
| MOS (naturalness) | UTMOS22-strong | ≥ 3.0 | 4.345 | 4.324 |
| CER (intelligibility) | Whisper-large-v3 | — | 1.60% | 0.22% |
| Speaker similarity | ECAPA2 cosine | ≥ 0.55 | 0.767 | 0.697 |

The paragraph path includes chunk seams ([§7](#7-caveats)). Traced eval is seeded (host Gumbel +
`reset_seeds`); eager `test_tt_eval` is not. The demo uses a lighter CER (`whisper-base.en`) only to
rank best-of-N takes, and only when `GenerationConfig.num_outputs` is raised above 1 in
[`config.py`](config.py) — it is not exposed as a CLI flag.

```bash
pytest models/experimental/xtts/tests/pcc/ -k "eval" -sv --timeout=0
```

### 6.3 Context length envelope

The GPT runs two streams with two learned position tables. The checkpoint shapes are the hard
ceiling, and they cannot be taken at once: 32 conditioning latents + 404 text + 605 codes needs
1041 positions. The reachable region is the triangle `text + codes ≤ 980`.

| Limit | Value | Source |
|-------|------:|--------|
| Text stream | **404** | `gpt.text_pos_embedding.emb.weight` is `[404, 1024]` (= `gpt_max_text_tokens` 402 + 2) |
| Mel stream | **608** | `gpt.mel_pos_embedding.emb.weight` is `[608, 1024]` (= 605 + 2 sentinels + 1 conditioning slot) |
| Generatable codes | **602** | upstream `max_mel_tokens - max_conditioning_inputs - 2` |
| Port KV / causal mask | **1012** | `MAX_GPT_SEQ_LEN` = 404 + 608 |

| Envelope | Text | Codes | Status |
|----------|-----:|------:|--------|
| Architectural (checkpoint) | 404 | 602 | Hard ceiling; `text + codes ≤ 980` |
| PCC-validated | 384 | 608 | `test_gpt_isl_sweep.py` (largest tile multiple under 404) |
| End-to-end / demo | 352 | **205 per pass** | `max_text_ids` / `max_single_pass_codes`; longer text is chunked |

The 205-code pass cap is an L1 circular-buffer collision on the full traced pipeline, not a
checkpoint limit. Isolated vocoder PCC covers 320 codes (`test_hifi_decoder.py`). Past 205, total
codes are unbounded by sentence chunking.

### 6.4 End-to-end performance

[`test_e2e_perf.py`](tests/perf/test_e2e_perf.py) on its own two-sentence prompt (89 wrapped
tokens padded to 96, single pass, temp 0.65 / top-k 50 / top-p 0.85 / rep 5.0, 30 s reference
voice, fully traced). It does not read `DEMO.text`, so the figure is stable if the demo default
changes:

| Quantity | Value |
|----------|------:|
| Generated | 188 codes (self-terminated at STOP) → 8.725 s audio |
| Setup replay (conditioning + speaker + prefill) | 0.044 s (8 conditioning windows) |
| Decode replay (188 codes) | 1.178 s (**6.267 ms/code**) |
| Vocoder replay | 0.021 s (0.109 ms/code) |
| **Total replay** | **1.243 s** |
| **RTF** (replay ÷ audio) | **0.142** — ≈7.0× faster than real time |
| Time-to-first-audio | 1.243 s (non-streaming: first audio = full clip) |
| Compile / capture (one-time, excluded from RTF) | ≈44 s for a code length not yet compiled / **≈4.5 s** once it is |
| End-to-end wall (weight load → WAV) | ≈63 s / **≈23 s** on the same split |

Repeat runs of the same seeded input spread ~3% (decode 6.267 / 6.445 ms/code, RTF 0.142 / 0.146);
the gated baselines in `test_e2e_perf.py` carry a 40% margin over the first of those.

Compile and wall-clock depend on the JIT kernel cache, and the axis is the **sampled code count**,
not a cold/warm boot: the vocoder trace is captured on the trimmed latents, so a length never
generated before compiles fresh conv shapes (≈44 s) while a repeat length replays cached ones
(≈4.5 s). Sampling picks a new length most runs. RTF and the replay legs are trace replay and do not
depend on any of this. Setup replay scales with conditioning windows (≈13 ms for 1 window, 43 ms for
8), not the prompt.

### 6.5 ISL sweep

[`test_e2e_isl_sweep_perf.py`](tests/perf/test_e2e_isl_sweep_perf.py) on the demo path: one pass
when the text fits, otherwise sentence-chunked with one `traced_session` capture. Budget
`chunk_max_tokens` (256). One pytest case per ISL on a fresh device.

| ISL | chunks | pad_to | prompt | max_seq | codes | audio (s) | TTFT (ms) | codes/s | ms/code | replay (s) | RTF |
|----:|-------:|-------:|-------:|--------:|------:|----------:|----------:|--------:|--------:|-----------:|----:|
| 32 | 1 | 32 | 64 | 352 | 89 | 4.13 | 47.9 | 162.7 | 6.146 | 0.599 | 0.145 |
| 64 | 1 | 64 | 96 | 384 | 143 | 6.63 | 49.3 | 160.7 | 6.222 | 0.948 | 0.143 |
| 96 | 2 | 96 | 128 | 416 | 355 | 16.47 | 50.0 | 160.4 | 6.236 | 2.357 | 0.143 |
| 128 | 2 | 96 | 128 | 416 | 355 | 16.47 | 50.0 | 161.4 | 6.197 | 2.344 | 0.142 |
| 192 | 3 | 96 | 128 | 416 | 531 | 24.64 | 50.1 | 159.7 | 6.261 | 3.540 | 0.144 |
| 256 | 4 | 96 | 128 | 416 | 706 | 32.76 | 50.0 | 160.4 | 6.233 | 4.688 | 0.143 |
| 320 | 5 | 96 | 128 | 416 | 887 | 41.16 | 50.0 | 160.7 | 6.222 | 5.878 | 0.143 |
| 352 | 5 | 96 | 128 | 416 | 887 | 41.16 | 50.0 | 159.5 | 6.268 | 5.919 | 0.144 |

XTTS is non-streaming. *TTFT* is time to first code (`setup + decode/n`); time to first audio is
the first chunk's replay. One code = 46.4 ms of audio. `pad_to` is the padded text length actually
prefilled. ISL 96/128 and 320/352 are the same sentence groups (the sweep grows a whole sentence at
a time).

From ISL 96 up, chunking pins `max_seq` at 416, so ms/code is flat (~6.2) and ISL is no longer a
cost axis — replay grows with chunk count at ≈1.18 s each (0.599 s at ISL 32 → 5.919 s at ISL 352).
TTFT stays ~50 ms. RTF stays ~0.143 across the whole sweep.

A wrapped prompt must end in `[STOP]`. Trimming to a tile-aligned length can drop it; without it
the sampler drones to the code cap.

### 6.6 Device profiles

Per-op breakdowns come from Tracy via the two drivers in [§5.2](#52-performance-tests).

| Signposted workload | Device FW | Device kernel |
|---------------------|----------:|--------------:|
| One decode step (`xtts_gpt_decode1_L30_maxseq384`) | **5.627 ms** | 5.140 ms |
| Prefill, 96 text tokens (`xtts_gpt_prefill_L30_text96`) | **6.975 ms** | 6.556 ms |

The remaining ~0.64 ms of the 6.267 ms/code in [§6.4](#64-end-to-end-performance) is the per-step
fence and token readback, not device work.

---

## 7. Caveats

### Long text is chunked

**Upstream splits on characters.** `Xtts.inference()` takes
`enable_text_splitting` (default `False`); when set, `split_sentence()` packs sentences greedily up
to a per-language character limit (`tokenizer.char_limits`, 250 for English), hard-wraps a single
over-long sentence with `textwrap`, and concatenates the pieces with no gap. That threshold stands
in for the *checkpoint's* budget — the token count must stay under `gpt_max_text_tokens` (402, a
hard `assert`) and only ~602 codes are generatable — so over-long input degrades softly into
truncated audio, which is why splitting can stay opt-in.

**This port splits on estimated codes, always.** Text estimated over `max_single_pass_codes` (205)
is split into groups of ≤`max_chunk_codes` (205), synthesised independently. Splitting is tried in escalating order: sentence boundaries first, then internal
punctuation (`,;:`) for a sentence that still does not fit — a seam where a speaker would pause
anyway — and word wrapping only for a clause that is still over budget, which is the same fallback
upstream's `split_sentence` uses. The binding limit here is L1, not the checkpoint: one pass caps at
205 codes ([§6.3](#63-context-length-envelope)), about a third of what the weights support. A full
250-char upstream chunk works out to ≈300 codes (1.20 codes/char, measured on the §6.2 paragraph),
so upstream's pieces would not fit a single pass here — hence codes rather than characters as the
unit, and no opt-out.

| | upstream `coqui-ai/TTS` | this port |
|---|---|---|
| Trigger | `enable_text_splitting=True` **and** ≥250 chars (en) | always on, estimated codes > 205 |
| Unit | characters, per language | estimated audio codes |
| Group size | ≤250 chars | ≤205 codes (`max_chunk_codes`) |
| Over-long sentence | hard-wrapped mid-sentence (`textwrap`) | split at `,;:` first, word-wrapped only if still over |
| Reason | checkpoint: 402 text tokens / ~602 codes | L1 co-residency of the traced pipeline |

Chunks of one take share a single trace capture, so they are all padded to the longest chunk's
tile-aligned length with `[STOP]`. Each chunk's *real* length rides along and masks that padding out
of decode attention — without it a short chunk reads the padding as "there is more text" and runs to
the code cap instead of stopping on time (`test_tt_traced_session_text_padding` is the regression
gate). A chunk that still reaches the cap without emitting STOP is redrawn up to
`chunk_retries` (2) times, since sampling is stochastic and a redraw normally lands a clean take.

Each chunk is an independent generation off the same speaker latents, so prosody resets at every
boundary.

### Several full generations compiled on one device can exhaust L1

The HiFi-GAN's convolutions allocate from `L1_SMALL`, and every distinct conv shape compiles
its own program. Compiling several independent generations on one device exhausts it. The demo
opens a fresh device per take (`_take_on_device`); chunks within a take share one device because
they replay a single capture. A caller that captures once and replays many times is fine; one that
captures repeatedly is not.

### Functional gaps vs. upstream coqui

- **Speaker-embedding window is 16 s, not the whole reference.** Upstream runs the speaker encoder
  on the reference clipped to `max_ref_length` (30 s). The encoder here reaches the full 30 s (above
  `SINGLE_BUFFER_FRAMES` its convs drop activation double buffering, ~4-7% slower and numerically
  identical), but co-resident with the rest of the traced model L1 caps it near 20 s, and that
  boundary drifts with the sampled generation length — so the window keeps a margin at 16 s. Longer
  helps but flattens out (see `DemoConfig.spk_seconds`). The GPT conditioning path uses the full
  30 s.
- **Fixed decode-step budget on the traced path.** A captured trace cannot branch, so
  `inference_fully_traced` replays a fixed number of steps and trims at STOP afterwards.
- **Sampling is not bit-exact across runs.** The demo does not seed the sampler. Seeded tests can
  be deterministic ([§6.2](#62-audio-quality-metrics)).
- **No HF-style `generate()` surface, no batching.** Fixed sampling recipe; batch-1 only.
- **No multi-speaker/dialogue format** — one narrated voice per call.

### Host operations

BPE tokenization, reference-audio load/resample, and fade-in/out on the finished waveform. Both
mels, the conditioning encoder, GPT, sampler, and vocoder run on device.

---

## 8. References

- [coqui/XTTS-v2 model card](https://huggingface.co/coqui/XTTS-v2)
- [Coqui Public Model License (CPML)](https://coqui.ai/cpml)
- [coqui-ai/TTS](https://github.com/coqui-ai/TTS) — upstream implementation

### CI

XTTS runs as a **tier-3** model on single Blackhole P150 (`bh_p150b_civ2`), under `team: shield`:

| Workflow | Entry | Command | Timeout |
|----------|-------|---------|--------:|
| `(Tier 3) Models Unit Tests` (nightly 06:00 UTC) → [`models_unit_tests.yaml`](../../../tests/pipeline_reorg/models_unit_tests.yaml) | `xtts-v2 unit tests` | `pytest models/experimental/xtts/tests/pcc/ -k "not eval" --timeout=900` | 25 min |
| `(Tier 3) Models End-To-End Tests` (nightly 05:00 UTC) → [`models_e2e_tests.yaml`](../../../tests/pipeline_reorg/models_e2e_tests.yaml) | `xtts-v2 e2e tests` | `test_e2e_perf.py --timeout=900`, then `demo/xtts_demo.py` | 25 min |

`--timeout=900` is the per-test pytest timeout; the 25 min is the job timeout, and it is what the
`shield` tier-3 lines in [`.github/time_budget.yaml`](../../../.github/time_budget.yaml) are sized
for (`e2e_tier3` 43 min, `unit_tier3` 45 min on this SKU — both exactly consumed). Two tests carry
their own `@pytest.mark.timeout(1800)`, which pytest-timeout honours over the CLI value.

`-k "not eval"` selects 21 of the 24 PCC tests, deselecting `test_tt_eval`, `test_tt_eval_traced`
and `test_tt_eval_traced_long` so CI does not pull whisper-large-v3, UTMOS or ECAPA2. Warm and
uncontended those 21 run in **3 min 42 s**, and the e2e entry in ~2 min; the 25-min timeouts are
sized for a first-run download of the ~1.9 GB checkpoint. `bh_p150b_civ2` is
`weights-cache-mode: lfc`, so that cache is per-runner and read-write and the download happens once
per runner, not once per run.

The quality metrics in [§6.2](#62-audio-quality-metrics) are therefore **not gated in CI**. The long
chunked eval still carries local gates (`LONG_CER_MAX = 0.05`, `LONG_SECS_MIN = 0.55` in
`tests/pcc/test_tt_trace.py`) for when it is run by hand. Device-OP perf and the ISL-latency sweep
are also local-only: tier 3 has no sweep pipeline, and device-OP perf exists only at tier 1.
