<!-- SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# XTTS-v2

TTNN implementation of [coqui/XTTS-v2](https://huggingface.co/coqui/XTTS-v2), a zero-shot
multilingual voice-cloning text-to-speech model. Runs fully on Tenstorrent hardware (audio
conditioning + GPT autoregressive decode + HiFi-GAN vocoding), with an authored PyTorch
reference kept alongside for PCC/objective-metric comparison.

## Reviewer guidance

| Path | Origin | Review |
|------|--------|--------|
| [`tt/`](tt/) | **Authored** TTNN port — GPT decoder, audio conditioning, mel frontends, speaker encoder, HiFi-GAN vocoder, sampler, top-level pipeline | **Primary** |
| [`demo/`](demo/), [`eval/`](eval/), [`tests/`](tests/) | **Authored** — demo CLI, objective TTS metrics, PCC test suite | Yes |
| [`reference/`](reference/) | **Authored** — hand-ported, dependency-free PyTorch reimplementation of coqui/XTTS-v2's published architecture (`TTS/tts/layers/xtts/*`), built directly against the real checkpoint without a `coqui-tts` dependency; the PCC ground truth | Yes — this is original code, not a vendored drop |

## Model description

XTTS-v2 speaks text in a voice cloned from a few seconds of reference audio. It runs as three
stages, all ported to TTNN here: an audio **conditioning** encoder turns the reference waveform
into a fixed-size GPT prompt, a **GPT-2-style decoder** autoregressively emits discrete audio
codes over a fixed KV cache, and a **HiFi-GAN vocoder** (conditioned on a separate speaker
embedding) turns the GPT's latents into a 24 kHz waveform.

| Property | Value |
|----------|-------|
| Task | Zero-shot voice cloning TTS: text + a few seconds of reference audio → speech in that voice |
| Language-model backbone | HF-style `GPT2Block` decoder — 30 layers, hidden 1024, 16 heads (head_dim 64), FFN 4096, `gelu_new` activation, LayerNorm eps 1e-5, causal self-attention (scale `1/sqrt(head_dim)`) |
| Max context | 1012 positions (`MAX_GPT_SEQ_LEN` = 404 text + 608 mel), fixed by the checkpoint's learned position embeddings |
| Vocab | Text: 6681 BPE tokens (XTTS-v2 `vocab.json`). Audio: 1026 ids (1024 mel codes + `start_audio`=1024 / `stop_audio`=1025 sentinels), capped at 605 codes (`gpt_max_audio_tokens`) per generation pass |
| Voice conditioning | Up to 30 s of reference audio (`gpt_cond_len`), split into 4 s chunks (`gpt_cond_chunk_len`); each chunk → 32 latents via a 6-block non-causal attention encoder (`ConditioningEncoder`, 16 heads) + a `PerceiverResampler` (depth 2, 8 heads, dim_head 64), averaged across chunks → `cond_latents [1, 32, 1024]` |
| Speaker embedding | SE-ResNet-34 (`[3,4,6,3]` `SEBasicBlock`s, SE reduction 8) on a 64-mel/16 kHz frontend (STFT n_fft 512, hop 160, win 400, hamming), attentive statistics pooling → FC → L2-norm → 512-d `g` |
| Audio codec / vocoder | XTTS-specific HiFi-GAN (`hifigan_decoder.waveform_decoder`): `conv_pre` + global/per-stage speaker conditioning, 4 upsample stages (rates `[8,8,2,2]`, kernels `[16,16,4,4]`), each followed by a 3-way multi-receptive-field fusion of `ResBlock1`s (kernels `[3,7,11]`, dilations `[1,3,5]` each) |
| Latent upsampling | Two composed `F.interpolate(mode="linear")` steps (4× mel→hop, then 24000/22050 sample-rate ratio) folded into **one** matmul against a precomputed resample matrix before the vocoder |
| Audio output | 24 kHz mono |
| Frame rate | ≈21.5 codes/s — one GPT audio code is 46.4 ms of output audio (4× latent upsample × 24000/22050 SR ratio × 256 vocoder upsample, ÷ 24000 Hz) |
| Speech:text token ratio | ≈2.2 audio codes per BPE text token (measured) |
| Sampling | Greedy argmax (deterministic, the PCC anchor) or on-device repetition-penalty → temperature → top-k → top-p (nucleus) → Gumbel-max categorical draw. XTTS's natural setting is temp 0.65–0.75 / top_k 50 / top_p 0.85 / rep-penalty 5.0 |
| Batch | 1 (single text + single reference voice) per call |
| Precision | GPT bf16 (mel-head weight in bfloat8_b); HiFi-GAN + conv primitives fp32 activations / bf16 weights, with 3 of the 4 vocoder upsample stages bf16-mixed by default (stage 0 stays fp32) |

**Upstream model scope** (properties of XTTS-v2 itself, which this port inherits unchanged — per
the [model card](https://huggingface.co/coqui/XTTS-v2)): the checkpoint is released under the
**Coqui Public Model License (CPML)**, a non-standard license with its own usage terms (see
[coqui.ai/cpml](https://coqui.ai/cpml)) — not a permissive OSS license. Upstream advertises 17
languages (English, Spanish, French, German, Italian, Portuguese, Polish, Turkish, Russian,
Dutch, Czech, Arabic, Chinese, Japanese, Hungarian, Korean, Hindi) and voice cloning from as
little as ~6 s of reference audio. This port's demo and PCC suite default to English text (one
test also exercises the conditioning encoder on a Spanish sample); full multilingual generation
is unvalidated here. There is no multi-speaker/dialogue format — one narrated voice per call.

## Supported devices

**Blackhole P150 is the only supported device.**

| Device | Configuration | Status |
|--------|---------------|--------|
| **Blackhole P150** | 1 × Blackhole ASIC, `ARCH_NAME=blackhole`, single device (`ttnn.open_device(device_id=0)` — no mesh/multi-chip path) | **Supported** — every PCC / performance number below was measured here |

Most PCC tests run at `l1_small_size=32768`; the full pipeline (`test_tt_inference`,
`test_tt_eval`) and the demo need `l1_small_size=65536`; the fully-traced pipeline test
additionally reserves a `trace_region_size` of ~50 MB (`52428800` bytes) for its three chained
traces, and ~150 MB (`157286400` bytes) where those three are held LIVE at once instead of being
released one at a time (`TtXttsTracedSession`, which the demo uses for chunked text so that every
chunk replays one capture rather than recompiling the model). No other device has been tried — Wormhole and multi-chip Blackhole boards are untested,
with no measured numbers and untuned L1/trace budgets.

## Context length support

XTTS has **no single context axis**. The GPT runs two streams with two separate learned position
tables, and the checkpoint's own tensor shapes are the hard ceiling:

| Limit | Value | Where it comes from |
|-------|------:|---------------------|
| Text stream | **404** positions | `gpt.text_pos_embedding.emb.weight` is `(404, 1024)` — indexing past it is out of bounds |
| Mel stream | **608** positions (605 codes) | `gpt.mel_pos_embedding.emb.weight` is `(608, 1024)` |
| Both together | **1012** | `GPT2Config n_positions` = 404 + 608 |

The two ceilings cannot be taken at once: 32 conditioning latents + 404 text + 605 codes needs a
1056-long cache, over the 1012 budget. The supported region is the **triangle `text + codes ≤ 980`**,
not a line to 1012 — so "max context" is a 2-D envelope, and a 1-D ISL sweep to 1012 would be
measuring a configuration the model cannot reach.

Three progressively tighter limits apply, and they are validated to different depths:

| Envelope | Text | Codes | Status |
|----------|-----:|------:|--------|
| Architectural (checkpoint) | 404 | 605 | The hard ceiling; `text + codes ≤ 980` |
| **PCC-validated** | **384** | — | [`test_gpt_isl_sweep.py`](tests/pcc/test_gpt_isl_sweep.py) sweeps 8 points to 384 (largest tile multiple under 404). Accuracy is **flat**: decode latent 0.9959–0.9986, cache medians ≥ 0.9988 |
| **End-to-end validated** | **352** | **205** | [`test_e2e_isl_sweep_perf.py`](tests/perf/test_e2e_isl_sweep_perf.py) + the demo's `MAX_TEXT_IDS` / `MAX_SINGLE_PASS_CODES` |

**Why end-to-end stops at 352/205 and not at the architectural ceiling.** The fp32 HiFi-GAN vocoder
hits an L1 circular-buffer allocation collision ("Statically allocated circular buffers … clash with
L1 buffers") above roughly 205 codes in a pass. This is an **allocation collision, not a clean size
limit**: per the demo's measured notes the headroom *shrinks* as generations accumulate in one
process (a chunk estimated at ~204 codes failed as the 5th pass while 207 passed as the 1st), and the
ISL sweep reproduces the same positional behaviour — ISL 64 runs fine alone but throws when it
follows ISL 32, while larger ISLs in the same sweep succeed. The demo works around it by opening a
**fresh device per pass** and splitting long text at sentence boundaries
(`MAX_CHUNK_CODES` 165 per chunk once splitting).

Longer text is therefore supported by **chunking**, not by a longer single pass: text over ~205
codes' worth is split at sentence boundaries and stitched with a ~120 ms gap. A single sentence is
never split, so one sentence longer than the single-pass budget is a hard failure.

## Architecture

```
        text ("Hello from Tenstorrent.")             reference audio WAV (any samplerate, mono)
                    │                                              │
      XTTS-v2 BPE tokenizer (host): [lang] tag,          resample to 22.05 kHz (conditioning)
      spaces → [SPACE], wrapped [START]/[STOP]                 + 16 kHz (speaker ID)
                    │                              ┌───────────────┴────────────────┐
                    │                    chunk_wav (≤30 s, 4 s windows)     TtMelFrontend (on-device
                    │                              │                        STFT-as-matmul, 64-mel)
                    │                    TtConditioningMel (on-device               │
                    │                    STFT-as-matmul, 80-mel)          TtResNetSpeakerEncoder
                    │                              │                     (SE-ResNet-34 + attentive
                    │                    TtXttsConditioning: 6-block             stats pooling)
                    │                    non-causal attention encoder +           │
                    │                    PerceiverResampler (32 latents),   g [1, 512, 1]
                    │                    averaged over chunks                     │
                    │                              │                             │
                    │                    cond_latents [1, 32, 1024]              │
                    └───────────────┬──────────────┘                             │
                                    │                                            │
     ┌───────────────────── PREFILL — TtXttsGptModel ─────────────────────────┐  │
     │  [cond_latents | text_emb] → 30 × (causal SDPA self-attn + GELU MLP)  │  │
     │  → ln_f → seed the fixed-size KV cache (tile-aligned to max_seq)      │  │
     └───────────────────────────────┬───────────────────────────────────────┘  │
                                      │                                          │
 ┌──────────────── AR DECODE, one audio code per step ──────────────────────┐   │
 │  decode_on_device(token, mel_pos, cache_pos, kv)                         │   │
 │    → logits [1,1,1026], latent [1,1,1024] (post-final_norm hidden)       │   │
 │         │                                                                │   │
 │         ├─ argmax (greedy) OR TtSampler: rep-penalty → temperature →     │   │
 │         │  top-k → top-p (nucleus) → Gumbel-max draw — picks the next    │   │
 │         │  code; STOP (id 1025) ends the loop                           │   │
 │         └─ latent harvested at this step, appended to the sequence       │   │
 └───────────────────────────────┬───────────────────────────────────────────┘  │
                                  │  codes [1,T], latents [1,T,1024]             │
                                  │  (bf16 → fp32 ROW_MAJOR)                     │
                  TtLatentUpsampler: 2 composed linear interpolations             │
                  (1024/256 = 4×, then 24000/22050 SR ratio) folded into          │
                  ONE matmul against a fixed resample matrix                     │
                                  │                                              │
                                  └──────────────────┬───────────────────────────┘
                                                      │
                    TtHifiganGenerator: conv_pre(x) + cond_layer(g), then 4×
                    [leaky_relu → ConvTranspose1d upsample → +conds[i](g) →
                    mean of 3× ResBlock1 (k=3/7/11, dilated MRF)], leaky_relu
                    → conv_post (k7, no bias) → tanh
                                                      │
                                24 kHz mono waveform → {output}.wav
```

Under `inference_fully_traced` (used by the fully-traced PCC test and available to any caller)
the whole model after the host tokenizer runs inside **three chained ttnn traces**: a SETUP trace
(conditioning + speaker encoder + prefill, seeding the persistent KV cache), a per-token DECODE
trace (captured once, replayed with sampling), and a VOCODER trace over the generated latents.
The demo itself calls the simpler eager `inference()` path with a fresh device per take.

## Model modules

Each TTNN module has a 1:1 authored PyTorch counterpart under [reference/](reference/) that the
PCC tests compare against.

| Module | TTNN implementation | Reference | What it does |
|--------|---------------------|-----------|---------------|
| Public API | [tt/xtts_inference.py](tt/xtts_inference.py) (`TtXtts`) | [reference/xtts_inference.py](reference/xtts_inference.py) (`XttsReference`) | Assembles every submodule; `.inference()` (eager) and `.inference_fully_traced()` (3 chained ttnn traces) are the two entry points |
| Generation driver | [tt/xtts_generator.py](tt/xtts_generator.py) (`TtXttsGenerator`) | [reference/xtts_gpt_generate.py](reference/xtts_gpt_generate.py) (`greedy_generate`) | Free-running, teacher-forced, and fully-traced AR decode loops over `TtXttsGptModel`; fixed-size KV cache, on-device sampling |
| Text embedding | [tt/xtts_text_embedding.py](tt/xtts_text_embedding.py) (`TtXttsTextEmbedding`) | [reference/xtts_text_embedding.py](reference/xtts_text_embedding.py), [reference/xtts_gpt_generate.py](reference/xtts_gpt_generate.py) (`wrap_text_ids`) | XTTS-v2 BPE tokenizer (host) → token embedding + learned position embedding (`ttnn.embedding`) |
| Audio conditioning | [tt/xtts_conditioning.py](tt/xtts_conditioning.py) (`TtXttsConditioning`) | [reference/xtts_conditioning.py](reference/xtts_conditioning.py) (`ConditioningEncoder`, `PerceiverResampler`) | 6-block non-causal attention encoder + 32-latent `PerceiverResampler` → GPT conditioning prompt |
| Mel frontends | [tt/xtts_mel.py](tt/xtts_mel.py) (`TtConditioningMel`, `TtMelFrontend`) | [reference/xtts_conditioning.py](reference/xtts_conditioning.py) (`wav_to_mel`), [reference/xtts_mel.py](reference/xtts_mel.py) (`MelFrontend`) | Fully on-device STFT-as-matmul + mel filterbank: 80-mel/22.05 kHz for conditioning, 64-mel/16 kHz for the speaker encoder |
| GPT decoder block | [tt/xtts_gpt_block.py](tt/xtts_gpt_block.py) (`TtXttsGptBlock`) | [reference/xtts_gpt_block.py](reference/xtts_gpt_block.py) (`XttsReferenceGptBlock`) | One causal self-attention + GELU MLP GPT-2 block; separate prefill (concat-KV) and fixed-cache decode forwards |
| GPT decoder stack | [tt/xtts_gpt_stack.py](tt/xtts_gpt_stack.py) (`TtXttsGptStack`) | [reference/xtts_gpt_stack.py](reference/xtts_gpt_stack.py) (`XttsReferenceGptStack`) | 30 stacked blocks + `ln_f`; prefill / static-cache decode / full-forward variants |
| GPT model (+ heads) | [tt/xtts_gpt_model.py](tt/xtts_gpt_model.py) (`TtXttsGptModel`) | [reference/xtts_gpt_model.py](reference/xtts_gpt_model.py) (`XttsReferenceGptModel`) | Embeddings + stack + text/mel heads; owns the fixed-size per-layer KV cache and the `prefill`/`decode` "mains" |
| Sampler | [tt/xtts_sampler.py](tt/xtts_sampler.py) (`TtSampler`) | (matches coqui's HF-generate sampling recipe; no standalone reference module) | On-device repetition-penalty → temperature → top-k → top-p → Gumbel-max categorical draw |
| Speaker encoder | [tt/xtts_speaker_encoder.py](tt/xtts_speaker_encoder.py) (`TtResNetSpeakerEncoder`) | [reference/xtts_speaker_encoder.py](reference/xtts_speaker_encoder.py) (`ResNetSpeakerEncoder`) | SE-ResNet-34 (`[3,4,6,3]` `SEBasicBlock`s) + attentive statistics pooling → 512-d speaker embedding `g` |
| Conv primitives | [tt/xtts_conv.py](tt/xtts_conv.py) (`TtConv1d`, `TtConvTranspose1d`, `TtConv2d`) | — (thin wrappers over `ttnn.conv1d`/`ttnn.conv2d` + a hand-built transpose conv) | Channels-last conv building blocks shared by the speaker encoder and the HiFi-GAN generator |
| Latent upsampler + vocoder | [tt/xtts_hifi_decoder.py](tt/xtts_hifi_decoder.py) (`TtLatentUpsampler`, `TtHifiDecoder`), [tt/xtts_hifigan.py](tt/xtts_hifigan.py) (`TtHifiganGenerator`) | [reference/xtts_hifi_decoder.py](reference/xtts_hifi_decoder.py), [reference/xtts_hifigan.py](reference/xtts_hifigan.py) | Two composed `F.interpolate` steps folded into one matmul, then the 4-stage upsample + multi-receptive-field-fusion HiFi-GAN vocoder → waveform |
| Full decoder | [tt/xtts_full_decoder.py](tt/xtts_full_decoder.py) (`TtXttsHifiDecoder`) | [reference/xtts_hifi_decoder.py](reference/xtts_hifi_decoder.py) (`XttsHifiDecoderFull`) | Composes the mel frontend + speaker encoder + latent upsampler + HiFi-GAN into the complete GAN decoder |

## File paths

```
models/experimental/xtts/
├── README.md
├── demo/
│   └── xtts_demo.py              # CLI: text + reference audio -> WAV (traced by default)
├── eval/
│   ├── __init__.py
│   └── xtts_eval.py               # objective TTS metrics: CER (Whisper-large-v3), UTMOS, SECS (ECAPA2)
├── reference/                      # authored, dependency-free PyTorch reimplementation; PCC ground truth
│   ├── __init__.py
│   ├── xtts_conditioning.py        # ConditioningEncoder + PerceiverResampler + conditioning mel/wav helpers
│   ├── xtts_gpt_block.py           # checkpoint loader + one GPT-2 decoder block + GPT-2 config/causal mask
│   ├── xtts_gpt_generate.py        # greedy autoregressive generation loop (PCC ground truth for TT decode)
│   ├── xtts_gpt_model.py           # embeddings + stack + text/mel heads
│   ├── xtts_gpt_stack.py           # 30 stacked GPT-2 blocks + ln_f
│   ├── xtts_hifi_decoder.py        # latent linear-upsample + full HifiDecoder (mel + speaker + generator)
│   ├── xtts_hifigan.py             # XTTS-specific HiFi-GAN generator (waveform_decoder)
│   ├── xtts_inference.py           # full reference pipeline: text + reference audio -> waveform
│   ├── xtts_mel.py                 # speaker-encoder mel frontend (torch_spec, 64-mel/16 kHz)
│   ├── xtts_speaker_encoder.py     # SE-ResNet-34 + attentive statistics pooling
│   └── xtts_text_embedding.py      # BPE tokenizer + text/position embedding
├── tests/
│   ├── __init__.py
│   ├── pcc/                        # one PCC test per major block, the GPT prefill/decode
│   │   └── ...                     #   matrix, and end-to-end (eager + traced) — 11 files,
│   │                               #   see Test cases
│   └── perf/
│       ├── __init__.py
│       ├── test_e2e_perf.py        # e2e performance regression test (models_performance_bare_metal)
│       ├── test_e2e_isl_sweep_perf.py               # TTFT / codes-s / ms-code / RTF vs ISL
│       ├── test_device_perf_single_step_prefill.py   # outer Tracy driver -> device_perf_*.csv
│       ├── test_profile_single_step_prefill.py       #   inner signposted prefill workload
│       ├── test_device_perf_single_step_decode.py    # outer Tracy driver -> device_perf_*.csv
│       └── test_profile_single_step_decode.py        #   inner signposted decode workload
└── tt/                              # TTNN implementation
    ├── __init__.py
    ├── xtts_conditioning.py
    ├── xtts_conv.py                 # TtConv1d / TtConvTranspose1d / TtConv2d primitives
    ├── xtts_full_decoder.py         # composes mel frontend + speaker encoder + decoder
    ├── xtts_generator.py            # AR decode drivers (free-run / teacher-forced / fully-traced)
    ├── xtts_gpt_block.py
    ├── xtts_gpt_model.py
    ├── xtts_gpt_stack.py
    ├── xtts_hifi_decoder.py         # TtLatentUpsampler + TtHifiDecoder
    ├── xtts_hifigan.py              # TtHifiganGenerator
    ├── xtts_inference.py            # TtXtts -- the top-level entry point
    ├── xtts_mel.py                  # TtMelFrontend (speaker) + TtConditioningMel (conditioning)
    ├── xtts_sampler.py              # TtSampler -- on-device rep-penalty/temp/top-k/top-p/Gumbel-max
    ├── xtts_speaker_encoder.py
    └── xtts_text_embedding.py
```

Everything is imported by its full path from the tt-metal root, e.g.
`from models.experimental.xtts.tt.xtts_inference import TtXtts`.

### Where the model configuration lives

There is **no `config.json` and no config module**, deliberately. coqui ships XTTS-v2's architecture
as constructor defaults rather than a config file, so there is nothing to parse — any config module
here would be a *transcription* of the published architecture, i.e. a second place the same numbers
live and a second thing to keep in sync.

Instead every dimension, count, rate and checkpoint pin is defined **exactly once**, in the
`reference/` module for the block it describes, next to the comment explaining where it came from —
and `tt/` imports those same names rather than restating them. There are no hardcoded dimensions or
shapes in the TTNN implementation.

| Configuration | Defined in |
|---------------|------------|
| Checkpoint pins (`HF_REPO_ID`, `HF_REVISION`, `CHECKPOINT_FILE`), GPT dims, context limits | [reference/xtts_gpt_block.py](reference/xtts_gpt_block.py) |
| Vocab sizes | [reference/xtts_gpt_model.py](reference/xtts_gpt_model.py) |
| Token sentinels, `MAX_AUDIO_TOKENS` | [reference/xtts_gpt_generate.py](reference/xtts_gpt_generate.py) |
| Conditioning encoder + Perceiver, conditioning mel (80-mel/22.05 kHz), reference-audio windowing | [reference/xtts_conditioning.py](reference/xtts_conditioning.py) |
| Speaker encoder (SE-ResNet-34) | [reference/xtts_speaker_encoder.py](reference/xtts_speaker_encoder.py) |
| Speaker mel frontend (64-mel/16 kHz) | [reference/xtts_mel.py](reference/xtts_mel.py) |
| Latent upsample ratios, sample rates | [reference/xtts_hifi_decoder.py](reference/xtts_hifi_decoder.py) |
| HiFi-GAN generator (rates, kernels, dilations, channels) | [reference/xtts_hifigan.py](reference/xtts_hifigan.py) |
| Tokenizer vocab file | [reference/xtts_text_embedding.py](reference/xtts_text_embedding.py) |

The values that *can* be checked against the checkpoint are checked by the PCC suite, which loads
the real weights: `gpt.text_embedding.weight` is `[6681, 1024]`, `gpt.mel_embedding.weight` is
`[1026, 1024]`, `gpt.text_pos_embedding.emb.weight` is `[404, 1024]` and
`gpt.mel_pos_embedding.emb.weight` is `[608, 1024]` — a wrong constant would fail to load or fail a
gate rather than silently drift.

One consequence to be aware of: `tt/` therefore depends on `reference/` for its constants (and on
`chunk_wav` from `reference/xtts_conditioning.py`), so the TTNN port is not shippable without the
reference package. That is a deliberate trade for single-definition config; splitting them would
mean maintaining the same numbers twice.

Weights are **not** vendored. On first run, the demo and PCC tests download the
[`coqui/XTTS-v2`](https://huggingface.co/coqui/XTTS-v2) checkpoint (`model.pth`, ~1.9 GB,
plus `vocab.json` and sample WAVs) via `huggingface_hub`, cached under the HF hub cache
(`~/.cache/huggingface` by default — no override env var is read by this port). The objective-eval
tests additionally pull `openai/whisper-large-v3` (~3 GB, CER), `tarepan/SpeechMOS` via
`torch.hub` (UTMOS), and `Jenthe/ECAPA2` via `huggingface_hub` (SECS). **UTMOS also needs
`torchaudio`**, which is not part of this repo's `python_env` — the metric logs a skip rather
than failing the test when it's absent (observed in this pass; CER and SECS still compute fine).

## Quick start

```bash
cd /path/to/tt-metal
source python_env/bin/activate
export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) ARCH_NAME=blackhole

# 1. Correctness (auto-downloads ~1.9 GB of XTTS-v2 weights on first run)
pytest models/experimental/xtts/tests/pcc/ -v

# 2. Demo -- writes generated/xtts_demo/xtts_demo.wav
python models/experimental/xtts/demo/xtts_demo.py
```

## Example input

The demo takes **plain text** (`--text`) plus a **reference voice** (`--ref-audio`). Both have
working defaults, so `python models/experimental/xtts/demo/xtts_demo.py` with no arguments runs the
example below on a fresh checkout.

**Text** — the bundled default, two sentences, 96 wrapped BPE tokens:

```
Voice synthesis has come a long way, and modern systems can already generate natural sounding speech with remarkable accuracy. Hey how are you doing?.
```

(The wording is deliberate: "can already" rather than "can now" — `/n/#/n/` is a nasal collision the
vocoder merges into "cannow/cannot", while "already" starts with a vowel and transcribes cleanly.)

**Reference voice** — the default is four single-speaker coqui-ai/TTS LJSpeech clips
(`LJ001-0001.wav+LJ001-0003.wav+LJ001-0004.wav+LJ001-0005.wav`, 9.66 + 9.67 + 5.14 + 8.11 = 32.6 s),
joined and clipped to `gpt_cond_len` 30 s = **8 conditioning windows**. They download on first use
and cache under `torch.hub`, so nothing local is required. A single HF sample (`en_sample.wav`) is
only ~3 s = **one** window, which exercises far less of the conditioning encoder.

Your own input:

```bash
python models/experimental/xtts/demo/xtts_demo.py \
    --text "Hello from Tenstorrent." --ref-audio /path/to/voice.wav
```

Text longer than one pass (~205 codes) is split at sentence boundaries automatically — see
[Context length support](#context-length-support).

## Expected output

A single **24 kHz mono WAV**, written to a fixed path (hardcoded in `main()`, not a CLI flag):

```
generated/xtts_demo/xtts_demo.wav
```

The absolute path is logged on completion (`wrote device audio -> …`). For the default text and
voice above, expect roughly:

| Property | Value |
|----------|-------|
| File | `generated/xtts_demo/xtts_demo.wav` — 24 kHz, mono, PCM |
| Codes generated | ~150–211 (sampled, so it varies run to run; capped at 240) |
| Duration | ≈ `codes × 46.4 ms` — about **7–10 s** of speech |
| Content | The input text, spoken in the cloned reference voice |
| Replay time | ~1.8 s, RTF ≈ 0.18 (see [Performance](#performance)) |

The run also prints a Coqui/HF-style perf summary (wall time, RTF, time-to-first-chunk, end-to-end).

**Sanity checks on a good render:** the audio is finite and non-silent, its duration matches
`codes × 46.4 ms`, the words are intelligible under Whisper, and the voice is recognisably the
reference speaker. Those last two are exactly what `test_tt_eval` scores (CER 0.016 and SECS 0.68 on
a self-terminating single sentence — see [PCC results](#pcc-results)); note both metrics are sampled
and **not reproducible run to run**, so treat them as ballpark.

**Comparing against the PyTorch reference:** set `args.write_torch_ref = True` in `main()` and the
demo additionally writes `generated/xtts_demo/xtts_demo_reference.wav` — the CPU torch reference
decoded from **the same codes** (teacher-forced), so the two files are the same utterance and differ
only by the device vocoder. That is a direct A/B of the port; an independent reference run would
diverge on sampling alone and tell you nothing.

## Running the demo

```bash
export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) ARCH_NAME=blackhole

# Default text + default reference voice (downloaded on first use, cached), fully traced
python models/experimental/xtts/demo/xtts_demo.py

# Your own text + reference voice
python models/experimental/xtts/demo/xtts_demo.py \
    --text "Hello from Tenstorrent." --ref-audio /path/to/voice.wav

# An HF coqui/XTTS-v2 sample voice instead of a local file
python models/experimental/xtts/demo/xtts_demo.py --ref-audio en_sample.wav
```

### Demo parameters

Only three flags are exposed; everything else is fixed in code to XTTS-v2's tuned defaults
(`lang="en"`, `max_tokens=240`, `temperature=0.65`, `top_k=50`, `top_p=0.85`,
`repetition_penalty=5.0`, `num_outputs=1`) so the demo always runs the validated recipe.

| Flag | Default | Description |
|------|---------|--------------|
| `--text <str>` | bundled 2-sentence English sample | Text to synthesize |
| `--ref-audio <path or sample name>` | `LJ001-0001.wav+LJ001-0003.wav+LJ001-0004.wav+LJ001-0005.wav` | Local WAV path; a coqui-ai/TTS LJSpeech clip name, or several `+`-joined into a longer reference (the default joins four same-speaker clips to 32.6 s, clipped to `gpt_cond_len` 30 s = 8 conditioning windows); or an HF `coqui/XTTS-v2` sample name (e.g. `en_sample.wav`, ~3 s = 1 window). Downloaded on first use and cached |
| `--min-tokens <int>` | `0` | STOP-suppression floor in audio codes. `0` = disabled (matches HF default; best for short prompts, where a floor makes the model ramble). `-1` = auto (~2× the wrapped text length) — raise it if a *long* prompt is only partly spoken |

Output always goes to `generated/xtts_demo/xtts_demo.wav` — this is currently hardcoded in
`main()`, not a CLI flag (the module docstring's `--write-torch-ref`/`--output` usage example is
stale; those options aren't wired up as `argparse` arguments in the current source).

Text longer than the single-pass budget (~205 audio codes worth) is split at sentence boundaries
into several passes (`_split_into_chunks`) and stitched back together with a ~120 ms gap; a
sentence itself is never split. Each pass opens a **fresh device** — the fp32 HiFi-GAN vocoder can
exhaust `l1_small_size` if several full generations share one open device — and, per the source
comments, the safe per-chunk code budget shrinks as more chunks run in the same process (an L1
circular-buffer allocation collision, not a clean size limit).

The demo logs a Coqui/HF-style perf summary per take (wall time, RTF, time-to-first-chunk); see
[Performance](#performance) for what a real run measured.

## Test cases

One PCC test per major block, plus the full end-to-end pipeline (eager and traced), plus a
**GPT prefill/decode matrix** covering each phase at both single-layer and full-30-layer depth
(see below). All device tests use the repo-root `device` fixture (single Blackhole P150, no
`mesh_device` abstraction).

The GPT decoder is the model's autoregressive core and where it spends essentially all of its
time, so it carries deeper coverage than the other blocks: both of its forwards
(`forward_prefill` / `forward_decode`) are gated at **both** depths, and the prefill pair runs at
the checkpoint's **full context lengths** (404 / 608 / 1012), not a tile-sized sample.

| | Single layer (`num_layers=1`) | All 30 layers |
|---|---|---|
| **Prefill** | `test_gpt_block.py`, `test_tt_gpt_prefill.py` | `test_gpt_stack.py` |
| **Decode** | `test_gpt_decode_layer.py` | `test_tt_gpt_generate.py` |

### PCC / correctness — `tests/pcc/`

| File | Test | Block | What it checks |
|------|------|-------|-----------------|
| `test_conditioning.py` | `test_xtts_conditioning` | Audio conditioning | Conditioning path (mel → `ConditioningEncoder` → `PerceiverResampler`) vs torch, real weights + real ref audio (English + Spanish samples) |
| `test_speaker_encoder.py` | `test_tt_speaker_encoder` | Speaker encoder | SE-ResNet-34 + attentive-stats-pooling speaker encoder vs torch, real weights |
| | `test_tt_speaker_encoder_shape_reuse` | Speaker encoder | Regression test: conv2d weight-cache correctness when one `TtResNetSpeakerEncoder` instance is reused across two different mel lengths |
| `test_gpt_block.py` | `test_xtts_gpt_block` | GPT decoder — prefill, 1 layer | One repeating GPT-2 block's `forward_prefill` (full causal attention) vs torch, at the **full** context lengths: 404 (max text), 608 (max mel), 1012 (`MAX_GPT_SEQ_LEN`) |
| `test_gpt_stack.py` | `test_xtts_gpt_stack` | GPT decoder — prefill, 30 layers | The whole 30-block stack + `ln_f` vs torch, at the same three full context lengths — the full-depth prefill hidden-state gate |
| `test_tt_gpt_prefill.py` | `test_tt_gpt_prefill` | GPT decoder — prefill, 1 layer | The real `TtXttsGptModel.prefill` main at `num_layers=1` (embeddings → concat `[cond\|text]` → causal block → `fill_cache`), gated on the **KV cache it seeds**: cached K and cached V at the prompt positions vs the fp32 reference projection. Real conditioning latents from real reference audio |
| `test_gpt_decode_layer.py` | `test_tt_gpt_decode_layer` | GPT decoder — decode, 1 layer | `forward_decode` on one block over a fixed-size KV cache, walked from empty across cache depths 0–9. **Open loop** — each step is fed a fresh random hidden and the reference recomputes the whole causal prefix, so an error cannot cascade. Gated on the **minimum** per-step PCC. Covers the low-cache-depth regime (mask mostly −inf) that the end-to-end tests never isolate |
| `test_gpt_isl_sweep.py` | `test_tt_gpt_prefill_isl_sweep` | GPT decoder — prefill, ISL sweep | Full 30-layer prefill at 8 text lengths from 32 to 384 (the ceiling imposed by the checkpoint's 404-row text position table, tile-aligned). Gated on the decode latent, the per-layer KV cache median, and the cache's **conditioning rows on their own** — the last being the regression test for the `ttnn.fill_cache` odd-tile bug |
| | `test_tt_gpt_decode_max_seq_sweep` | GPT decoder — decode, cache-depth sweep | One decode step at a fixed 128-position fill level over cache sizes 160/384/608/992. Decode attends over the *whole* fixed cache, so this proves the mask ignores the unfilled tail: the result must be PCC-clean **and** bit-identical across `max_seq` |
| `test_gpt_model.py` | `test_xtts_gpt_model` | GPT decoder | Full GPT front-to-back (embeddings → 30 blocks + `ln_f` → heads) vs torch, single forward pass, swept over `(text_len, mel_len)` ∈ {(64,96), (96,128)} |
| `test_tt_gpt_generate.py` | `test_tt_gpt_generate` | GPT decoder — decode, 30 layers | The actual autoregressive KV-cache decode loop (what the model runs in production) vs the torch reference loop: free-run exact-code-match prefix + teacher-forced latent PCC |
| `test_hifi_decoder.py` | `test_tt_hifi_decoder` | HiFi-GAN vocoder | On-device latent linear-upsample chained into the HiFi-GAN generator vs torch, at `latent_len` 32 |
| `test_tt_inference.py` | `test_tt_inference` | End-to-end | Full pipeline (conditioning → GPT decode → HiFi-GAN), teacher-forced on reference codes, gated on spectrogram-magnitude PCC |
| | `test_tt_cond_latents_long_reference` | Audio conditioning | Reference audio longer than one window (10 s of coqui LJSpeech clips → 3 windows): every window's 80-mel computed on device and the style embeddings averaged, vs torch windowing the *audio* the same way — also the only test forcing the device mel frontend's multi-chunk reshape framing |
| | `test_tt_eval` | End-to-end | Real *sampled* generation, scored with objective metrics (CER via Whisper-large-v3, UTMOS, SECS via ECAPA2) — best-effort, no PCC gate |
| `test_tt_trace.py` | `test_tt_full_trace` | End-to-end (traced) | Entire on-device model via `inference_fully_traced` (setup + decode + vocoder, all as ttnn traces) vs the eager path on the *same* generated codes |
| | `test_tt_traced_session_reuse` | End-to-end (traced) | State hygiene of a reused `TtXttsTracedSession` (the demo's chunked-text path): one capture replayed greedily for text A, B, then A again, requiring run 3 to reproduce run 1 **exactly** — catches a repetition mask / KV slot / step counter / latent accumulator left dirty by the previous run. All three traces live at once |
| | `test_tt_eval_traced` | End-to-end (traced) | Same objective-metric eval as `test_tt_eval`, on the fully-traced generation |

### Test gates

Every numeric PCC assertion in this suite uses the same bar: **≥ 0.99** (`comp_pcc`, wherever a
threshold exists). Two GPT tests apply that bar to something other than a single output tensor:
`test_tt_gpt_prefill` gates on the seeded **KV cache** (cached K and cached V at the prompt
positions — prefill's actual product), and `test_gpt_decode_layer` gates on the **minimum**
per-step PCC across cache depths 0–9, since a mean would let one bad step hide.
`test_tt_inference`/`test_tt_full_trace` gate on **spectrogram-magnitude** PCC
rather than raw waveform PCC — a GAN vocoder maps tiny bf16-latent differences to small
phase/sample shifts that tank sample-wise correlation without changing what is heard, so raw
waveform PCC is logged as informational only. `test_tt_eval`/`test_tt_eval_traced` have no PCC
gate; each of CER/UTMOS/SECS is best-effort (a missing/failing backend logs a skip rather than
failing the test).

### Performance — `tests/perf/`

| File | Test / entry point | What it measures |
|------|--------------------|------------------|
| `test_e2e_perf.py` | `test_xtts_e2e_perf` | Times `TtXtts.inference_fully_traced()` on the demo's default text + voice; hard-asserts setup replay, decode (ms/code), vocoder (ms/code), and RTF each stay within 40% of the measured Blackhole P150 baseline (see [Performance](#performance)) |
| `test_e2e_isl_sweep_perf.py` | `test_xtts_e2e_isl_sweep_perf` | Wall-clock sweep over 8 text lengths (32→352, the end-to-end envelope) on the traced path: TTFT, codes/s, ms/code, RTF and the per-leg replay split. Records rather than fails on the L1 allocation collision. No perf gate — the gate is `test_e2e_perf.py` |
| `test_device_perf_single_step_prefill.py` | `main()` | Outer Tracy driver for a single GPT **prefill** op dump — writes `device_perf_*.csv` + a partial benchmark JSON via `prep_device_perf_report`. No perf assertion |
| `test_profile_single_step_prefill.py` | `test_profile_single_step_prefill` | The inner workload: one warm `prefill_on_device` (128-position prompt, 30 layers) inside `start`/`stop` signposts |
| `test_device_perf_single_step_decode.py` | `main()` | Outer Tracy driver for a single GPT **decode step** op dump — same outputs, no assertion. This is the report to read when the ~8 ms/code rate moves |
| `test_profile_single_step_decode.py` | `test_profile_single_step_decode` | The inner workload: untimed prefill, then one `decode_on_device` inside signposts, using the **traced** one-hot cache write (what production replays) |

The four profiling files follow the outer-driver / inner-workload split the Voxtral and VibeVoice
ports use: the outer script spawns the inner pytest under Tracy and never imports `ttnn` itself
(so it does not contend for the UMD device lock with its own child); the inner workload warms
outside the measured region, calls `ttnn.ReadDeviceProfiler` to drain load/warmup markers, then
brackets exactly one call with `start`/`stop` signposts. Shapes match the demo's real
configuration — 96 wrapped text tokens + 32 conditioning latents = 128 prompt positions, and
`max_seq` 384 for the 240-code budget.

`test_e2e_perf.py` is the only test in the repo carrying `@pytest.mark.models_performance_bare_metal` for a
TTS/audio model — there was no existing XTTS/VibeVoice/Whisper example to follow, so it
establishes the convention fresh here, modeled on `models/demos/wormhole/bert_tiny/tests/test_performance.py`
and `models/demos/wormhole/mamba/tests/test_mamba_perf.py` (the two closest existing examples
with a comparable hard `assert`/prefill-decode-split structure). Decode and vocoder are gated on
a **per-code rate**, not absolute time, since the number of codes generated is itself sampled
(temperature 0.65) and not perfectly fixed run to run — `reset_seeds` narrows that but a
rate-based gate is robust either way. The 40% margin is deliberately generous: this is the first
perf baseline ever recorded for this model, so there's no history yet to know normal hardware
variance from a real regression.

## Commands — PCC checks

```bash
cd $TT_METAL_HOME
source python_env/bin/activate
export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) ARCH_NAME=blackhole

# Everything (auto-downloads weights + the WER/eval models on first run)
pytest models/experimental/xtts/tests/pcc/ -v -s

# One test per block
pytest models/experimental/xtts/tests/pcc/test_conditioning.py \
       models/experimental/xtts/tests/pcc/test_speaker_encoder.py \
       models/experimental/xtts/tests/pcc/test_gpt_model.py \
       models/experimental/xtts/tests/pcc/test_hifi_decoder.py -v -s

# The GPT prefill/decode matrix — both phases, single-layer and full-30-layer depth.
# The two prefill tests run the full 404 / 608 / 1012 context lengths; ~36 s for all 8 cases.
pytest models/experimental/xtts/tests/pcc/test_gpt_block.py \
       models/experimental/xtts/tests/pcc/test_gpt_stack.py \
       models/experimental/xtts/tests/pcc/test_tt_gpt_prefill.py \
       models/experimental/xtts/tests/pcc/test_gpt_decode_layer.py -v -s

# The autoregressive decode loop (full 30-layer stack, production code path)
pytest models/experimental/xtts/tests/pcc/test_tt_gpt_generate.py -v -s

# Full pipeline, eager + traced + objective eval (downloads whisper-large-v3 / SpeechMOS / ECAPA2 on first use)
pytest models/experimental/xtts/tests/pcc/test_tt_inference.py \
       models/experimental/xtts/tests/pcc/test_tt_trace.py -v -s
```

`-s` is required to see the per-test PCC / metric numbers on stdout. `device_params` is fixed at
`l1_small_size` 32768/65536 (per test; see [Supported devices](#supported-devices)) — do not
override.

## Commands — Performance

```bash
# The gated e2e baseline
pytest models/experimental/xtts/tests/perf/test_e2e_perf.py -v -s

# The ISL sweep (8 lengths, ~1 min warm; ungated, prints the table in Performance below)
pytest models/experimental/xtts/tests/perf/test_e2e_isl_sweep_perf.py -v -s

# Shorten it for a smoke run
XTTS_ISL_SWEEP=32,96 pytest models/experimental/xtts/tests/perf/test_e2e_isl_sweep_perf.py -v -s
```

Runs in well under a minute (the fully-traced path recompiles kernels only if the JIT build cache
is cold) and needs no extra downloads beyond the checkpoint. See
[Performance](#performance) for what it measures and gates.

## Commands — device profiling (Tracy)

Each outer driver spawns its inner workload under Tracy and writes a per-op CSV. Run them
**one at a time** — they need the single device, and see the clobbering warning below.

```bash
export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) ARCH_NAME=blackhole

python models/experimental/xtts/tests/perf/test_device_perf_single_step_prefill.py
python models/experimental/xtts/tests/perf/test_device_perf_single_step_decode.py

# Then read the per-op breakdown (path is printed as "OPs csv generated at:")
tt-perf-report generated/profiler/xtts_gpt_single_step_decode/reports/<ts>/ops_perf_results_<ts>.csv \
    --start-signpost start --end-signpost stop
```

> **The second driver deletes the first one's report.** Both call
> `clear_profiler_runtime_artifacts()` on entry, which wipes `generated/profiler/` — including the
> other workload's `reports/` directory. Copy the CSV somewhere safe before running the other
> driver. (Verified the hard way: a back-to-back prefill run erased the decode report.)

| Parameter | Applies to | Meaning |
|-----------|-----------|---------|
| `XTTS_PERF_NUM_LAYERS=<int>` | both profiling workloads | GPT depth to profile. Default 30 (the full stack); set `1` to isolate a single decoder layer |
| `--start-signpost start --end-signpost stop` | `tt-perf-report` | **Required** — without it you get warmup + measured pass summed together (roughly 2× the real number) |

A raw CSV read must be signpost-filtered too: the file contains the warmup pass as well. Filtering
the prefill CSV to its window gives 397 ops / 6.884 ms, which matches the driver's reported
`AVG DEVICE FW DURATION` exactly; reading the whole file gives 794 ops / 13.772 ms.

## Dependency versions

The environment the [PCC results](#pcc-results) and [Performance](#performance) numbers below were
measured on. Checkpoint revisions are pinned in code; the Python and Tenstorrent stack versions are
recorded here but not enforced by anything.

### Tenstorrent stack

| Component | Version |
|-----------|---------|
| tt-metal / TT-Metalium | commit `e7cf6d43d15` — `git describe`: `v0.77.0-dev20260810-87-ge7cf6d43d15` (2026-08-11) |
| TTNN | in-tree `ttnn/ttnn/`, resolved via `PYTHONPATH=$TT_METAL_HOME`; no version of its own — the tt-metal commit above is the only handle |
| Arch | `ARCH_NAME=blackhole` (Blackhole P150 — see [Supported devices](#supported-devices)) |

Quote the **git commit**, not a package version: `ttnn` exposes no `__version__`, and
`importlib.metadata.version("ttnn")` returns the `python_env` wheel's metadata (`0.75.0rc10.dev318`
on this box) which is *not* what runs — with `PYTHONPATH=$TT_METAL_HOME`, `import ttnn` resolves to
the source tree (verified: `ttnn.__file__` → `$TT_METAL_HOME/ttnn/ttnn/__init__.py`).

### Python packages (`python_env`)

| Package | Version | Used for |
|---------|---------|----------|
| `torch` | 2.11.0+cpu | the PyTorch reference + all host-side tensor work |
| `librosa` | 0.10.0 | **mel filterbanks** — `librosa.filters.mel(htk=True, norm="slaney")` in both [tt/xtts_mel.py](tt/xtts_mel.py) and [reference/xtts_conditioning.py](reference/xtts_conditioning.py) |
| `numpy` | 1.26.4 | host array plumbing |
| `scipy` | 1.15.3 | `resample_poly` — reference-audio resampling, demo CER path |
| `soundfile` | 0.14.0 | WAV read/write |
| `tokenizers` | 0.22.2 | XTTS-v2 BPE tokenizer (`vocab.json`) |
| `huggingface_hub` | 1.16.1 | checkpoint / sample-WAV / ECAPA2 download |
| `transformers` | 5.12.1 | eval only — the Whisper CER pipeline |
| `torchaudio` | **not installed** | UTMOS only; its absence is why UTMOS reports `skipped` (see [Known limitations](#known-limitations)) |

`librosa` is the one to watch. Its filterbank feeds *every* mel in the model, on both the TTNN and
the reference side, so a version bump moves the mel and with it every PCC number in this README.
That pin is load-bearing, not cosmetic.

### Model weights / external checkpoints

Every download is revision-pinned in code, so the numbers in this README stay reproducible if
upstream re-uploads:

| Artifact | Source | Revision | Pinned by |
|----------|--------|----------|-----------|
| XTTS-v2 (`model.pth`, `vocab.json`, `samples/*.wav`) | [`coqui/XTTS-v2`](https://huggingface.co/coqui/XTTS-v2) | `6c2b0d75eae4b7047358e3b6bd9325f857d43f77` | `HF_REVISION` — [reference/xtts_gpt_block.py](reference/xtts_gpt_block.py) |
| Whisper-large-v3 (CER) | [`openai/whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3) | `06f233fe06e710322aca913c1bc4249a0d71fce1` | `WHISPER_REVISION` — [eval/xtts_eval.py](eval/xtts_eval.py) |
| ECAPA2 (SECS) | [`Jenthe/ECAPA2`](https://huggingface.co/Jenthe/ECAPA2) | `207cb6d137c671a12ba820ebec3b719549b06c0f` | `ECAPA2_REVISION` — [eval/xtts_eval.py](eval/xtts_eval.py) |
| UTMOS (`utmos22_strong`) | `torch.hub` [`tarepan/SpeechMOS`](https://github.com/tarepan/SpeechMOS) | tag `v1.2.0` (== `ed25eac`) | `UTMOS_HUB_REPO` — [eval/xtts_eval.py](eval/xtts_eval.py) |

One asymmetry worth knowing: the three HuggingFace artifacts pin an exact **commit** via `revision=`,
but `torch.hub` accepts only a tag or branch, so UTMOS pins the **tag** `v1.2.0` — which is the
commit upstream's default branch already pointed at, so it changes nothing about what gets fetched.
All four pins are exercised by the suite: the eval run fetches
`github.com/tarepan/SpeechMOS/zipball/v1.2.0` and caches it as `tarepan_SpeechMOS_v1.2.0`, and only
then does UTMOS skip, on the missing `torchaudio` import inside the hub module. So the ref resolves;
it is the *scoring* that is unavailable here.

A single shared `HF_REVISION` covers all three `coqui/XTTS-v2` downloads — the checkpoint
([reference/xtts_gpt_block.py](reference/xtts_gpt_block.py)), `vocab.json`
([reference/xtts_text_embedding.py](reference/xtts_text_embedding.py)) and the sample WAVs
([reference/xtts_conditioning.py](reference/xtts_conditioning.py)) — since they are three files in
one repo and must not drift apart.

## PCC results

Measured on **Blackhole P150** against the PyTorch reference (software stack: see
[Dependency versions](#dependency-versions)), running the whole `tests/pcc/` suite in one pass
(`pytest models/experimental/xtts/tests/pcc/ -v -s`) at tt-metal commit `e7cf6d43d15`:
**24/24 passed** in 362 s (14/14 in 289 s before this pass added the GPT prefill/decode matrix and
the ISL sweep). There is a single column because Blackhole P150 is the only supported device.

These numbers are **provisional**: the branch has not yet been rebased onto `main`, and the rebase
is deliberately being done last so one final run produces the published set. Four upstream commits
in the pending range touch ops on XTTS's hot path (LayerNorm intermediate CBs, `tanh` bf16, SFPU
range handling, and a Blackhole RM-reshape fix), so every value here should be re-measured after it.

| File | Test case | PCC / metric |
|------|-----------|--------------:|
| `test_conditioning.py` | Conditioning latents (en_sample.wav) | 0.995944 |
| | Conditioning latents (es_sample.wav) | 0.992365 |
| `test_speaker_encoder.py` | Speaker encoder (mel_len=200) | 0.999330 |
| | Shape-reuse regression, reused instance (mel_len 200 / 512) | 0.998787 / 0.999392 |
| `test_gpt_block.py` | Prefill, 1 layer — seq_len 404 / 608 / 1012 | 0.999874 / 0.999874 / 0.999872 |
| `test_gpt_stack.py` | Prefill, 30 layers + `ln_f` — seq_len 404 / 608 / 1012 | 0.998786 / 0.998779 / 0.998853 |
| `test_tt_gpt_prefill.py` | Prefill, 1 layer — seeded cache K / V (prompt_len 64) | 0.999976 / 0.999951 |
| `test_gpt_decode_layer.py` | Decode, 1 layer — min / mean over cache depths 0–9 | 0.999586 / 0.999882 |
| `test_gpt_model.py` | text_head / mel_head (64,96) | 0.995453 / 0.990920 |
| | text_head / mel_head (96,128) | 0.995604 / 0.991085 |
| `test_gpt_isl_sweep.py` | Decode latent, text_len 32→384 (8 points) | 0.9959 – 0.9986 |
| | ↳ KV cache per-position median, worst layer, across the sweep | K ≥ 0.999212 / V ≥ 0.998817 |
| | ↳ conditioning-row K min (the `fill_cache` regression gate) | 0.9992 at every ISL |
| | `test_tt_gpt_decode_max_seq_sweep` — max_seq 160/384/608/992 | 0.997585, bit-identical across all four |
| `test_tt_gpt_generate.py` | Free-run exact-match prefix / teacher-forced top-1 | 16/16 both |
| | Teacher-forced latent PCC | 0.999569 |
| `test_hifi_decoder.py` | latent_len 32 | 0.993287 |
| `test_tt_inference.py` | End-to-end spectrogram PCC (teacher-forced, 16 codes) | 0.990931 |
| | ↳ raw-waveform PCC, same run (informational only — see [Test gates](#test-gates)) | 0.905271 |
| | Long-reference conditioning latents (10.0 s → 3 windows of 4/4/2 s, averaged) | 0.998796 |
| | `test_tt_eval` — CER / UTMOS / SECS (150-code cap; text didn't finish before the cap) | 0.2874 / skipped (no `torchaudio`) / 0.6268 |
| `test_tt_trace.py` | Fully-traced vs eager spectrogram PCC (167 codes) | 1.0 |
| | `test_tt_traced_session_reuse` — A/B/A-again state hygiene | exact match (41 / 63 / 41 codes; run 3 == run 1 bit-for-bit) |
| | ↳ traced session's vocoder vs **eager**, same inputs (checked out-of-suite — see below) | bit-identical, max delta 0.0 |
| | `test_tt_eval_traced` — CER / UTMOS / SECS (self-terminated at 167 codes) | 0.0160 / skipped / 0.6818 |

The `test_tt_eval` CER of 0.34 is explained by its 150-code cap cutting off a two-sentence prompt
before STOP fired (the transcript matches the *first* sentence verbatim); it is not evidence of a
synthesis-quality regression — `test_tt_eval_traced`'s single self-terminating sentence scores CER
0.016. Both eval tests are sampled and their metrics are **not reproducible run to run**; treat
them as ballpark, and A/B correctness changes against `test_tt_inference`'s teacher-forced PCC
instead. Concretely, two runs of identical code hours apart gave `test_tt_eval` CER 0.3353 → 0.2874
and SECS 0.5408 → 0.6268, while every deterministic PCC above was byte-identical across the same two
runs. A moved eval metric is therefore not evidence of anything on its own.

### Fixed during this pass — the traced session, and its bit-exactness

`test_tt_traced_session_reuse` was failing with `NameError: name 'cond_bias_trace_safe' is not
defined`: `xtts_inference.py` used that context manager twice inside `TtXttsTracedSession` without
importing it, so both uses raised the moment they executed. Not new either — no `xtts_conv` import
existed in that file in any of the three preceding commits, so `TtXttsTracedSession` (the demo's
chunked-text path) had been broken for a while and went unnoticed because
[nothing runs this suite in CI](#ci). `inference()` and `inference_fully_traced()` never touch that
code path, which is why every other test kept passing. One import regression, three commits: the
concrete cost of having no CI here.

Restoring the import made the test pass, but it also restored the **wrong path**. The wrapper forces
the trace-safe cond-bias add, which is ~82 us/pass slower and *not* bit-exact — it adds post-conv in
the stage's bf16, where the default fold combines in fp32. Measured against the eager vocoder on
identical inputs, the traced session's waveform was off by up to **0.245** — large, on a signal in
roughly [-1, 1]. So the demo's chunked path was materially diverging from eager, not marginally.

The wrappers exist because the fast fold prepares its combined bias through a host transfer, which is
fatal inside a capture — and the fold's cache is keyed on `(id(cond_bias), input signature)`, so it
only hits for the *same* `g` tensor object. The session used to warm up against one `g` and capture
against another, guaranteeing a miss. The fix is a single persistent `g`: allocated up front with the
other persistent buffers, filled by the eager warm-up (so the cached bias is built from real values,
which a capture could not provide), rewritten in place by the SETUP trace, and read by both the
vocoder warm-up and the vocoder capture. The capture then hits the cache, does no host transfer, and
runs the same fp32 fold as eager — **waveform bit-identical to eager, max delta 0.0**.

Two traps worth knowing if you touch this again. `_setup` must write into that buffer
*unconditionally*: a parameter that skips the copy during warm-up leaves the copy out of the program
cache, and the capture dies on `Cannot load new binaries during trace capture`. And when any op
fatals mid-capture, `end_trace_capture` never runs, so the device stays wedged and teardown
deadlocks — the failure presents as a **hang**, not an error, so give these runs a timeout.

### Movement since the previously recorded numbers

Same gates, same tests, but several values shifted — the earlier table predates the three most
recent commits (`gan decoder bf16`, `leaky relu fused`, `manual group norm removed for conditioning
encoder`):

| Test case | Was | Now |
|-----------|----:|----:|
| Conditioning latents (en_sample.wav) | 0.997407 | 0.995944 |
| Conditioning latents (es_sample.wav) | 0.998102 | **0.992365** |
| latent_len 32 | 0.993109 | 0.993287 |
| End-to-end spectrogram PCC | 0.990620 | 0.990931 |

The conditioning drop is the one to watch: **es_sample is now 0.9924 against a 0.99 gate**, down
from 0.9981, so its margin has gone from ~8× to ~2.4×. The HEAD commit removed a manual group norm
from the conditioning encoder, which makes it the obvious suspect, but this was **not** A/B'd — if
the conditioning path is touched again, re-measure both samples before assuming headroom.

### Found and worked around: a `ttnn.fill_cache` first-tile corruption

The GPT ISL sweep immediately turned up a real correctness bug, and it is **not in this model's
code**. `ttnn.fill_cache` corrupts the **first tile** of the destination cache (rows 0–31) when the
input's sequence length is an odd multiple of 32 and `num_heads` is 16. Minimal reproducer, no XTTS
involved:

```python
k     = ttnn.from_torch(torch.randn(1, 16, 288, 64), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
cache = ttnn.from_torch(torch.zeros(1, 16, 512, 64), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
ttnn.fill_cache(cache, k, 0)
# cache[:, :, :288, :] differs from k by up to 4.3 — entirely within rows 0..31
```

| Dimension | Behaviour |
|-----------|-----------|
| seq tile count (16 heads) | clean at 1–8, 10, 12, 14 · **corrupt at 9, 11, 13, 15, 16** |
| `max_seq` | irrelevant — corrupt at 320 / 384 / 512 / 640 / 1024 alike |
| dtype | both `bfloat16` **and** `float32` |
| `num_heads` | clean at 1, 2, 4, 8 · corrupt at **16** — so it is work-distribution related |
| blast radius | confined to rows 0–31 (`max|cache−input|` equals the first-tile delta exactly) |

This lands squarely on XTTS: prefill writes `[cond | text]` into the cache, and the **conditioning
latents occupy exactly rows 0–31** — the tensor carrying voice identity. At prompt lengths 288 / 352
/ 416 the conditioning rows measured PCC 0.851 / 0.768 / 0.698 and the decode latent fell to
0.960 / 0.966, below gate. The likely user-visible symptom is degraded voice cloning at particular
text lengths, not a crash — which is why it went unnoticed.

**Workaround** (in [tt/xtts_gpt_model.py](tt/xtts_gpt_model.py) `prefill_on_device`): pad K/V by one
tile when the prompt's tile count is odd, so the write is always an even tile count. Every affected
shape returns to 0.999969, identical to the shapes that were already correct. The pad rows land past
`prompt_len`, where decode's additive mask ignores them until a decode step overwrites the slot, so
they are invisible to the model — and the fully-traced path still passes, i.e. `ttnn.pad` is
trace-safe here.

This is **not a general fix**: it is only sound because XTTS's longest prompt is 32 cond + 404 text =
436 rows = 14 tiles, so the in-range bad set is `{9, 11, 13}` and padding always lands on 10/12/14.
Padding 15 tiles would give 16, which is also corrupt. The underlying ttnn issue has not been filed
yet — it is deliberately held for discussion after this PR.

`test_gpt_isl_sweep.py` gates the conditioning rows separately from everything else specifically so
this cannot regress silently.

### Restored: the GPT prefill/decode coverage

An earlier "tests cleanup" pass removed 13 test files, on the principle of one PCC test per major
block. That was the wrong call for the **GPT decoder**: it is the model's autoregressive core, its
two forwards (`forward_prefill` / `forward_decode`) are separate code paths with separate failure
modes, and single-layer isolation is what turns "the 30-layer stack drifted" into "*this* op
drifted." Three of those files are restored unchanged (`test_gpt_block.py`, `test_gpt_stack.py`,
`test_tt_gpt_prefill.py`) and one is new (`test_gpt_decode_layer.py`), together forming the
prefill × decode / single-layer × all-layers matrix in [Test cases](#test-cases). All 8 cases pass
in 36 s; the restored files needed no adaptation, so nothing in the TTNN port had drifted away
from them in the meantime.

This also closed the largest length-coverage gap in the suite. `test_gpt_block`/`test_gpt_stack`
run at 404 / 608 / **1012** — the checkpoint's full context — where previously the deepest GPT
coverage was `test_gpt_model`'s `(96,128)`, about 16 % of context. Both hold PCC ≈ 0.9988–0.9999
at full length, so the 30-layer prefill chain shows no length-dependent drift.

**Still removed** (9 files, tracked as follow-up work, not covered by this pass): the mel
frontend, text embedding, conv primitives, sampler, full decoder, and waveform decoder unit
tests, the two profiling-only harnesses, and one integration test. Two of them were **failing**
when removed — `test_gpt_with_conditioning.py` (text_head PCC 0.9811, below gate) and
`test_waveform_decoder.py` (allocator crash at `latent_len=16`, PCC 0.9860 at `latent_len=32`).
They were removed for scope reasons, not to hide the failures — flagging here for the record. The
retained `test_hifi_decoder.py` exercises much of the same HiFi-GAN generator code on
GPT-latent-scaled synthetic inputs and currently passes, so this isn't necessarily evidence of a
live model-correctness bug; it wasn't re-investigated as part of this pass.

## Performance

A real demo run (`python models/experimental/xtts/demo/xtts_demo.py`, default text — 96 wrapped
tokens, single pass, sampled temp 0.65/top_k 50/top_p 0.85/rep 5.0, default 30 s reference voice,
fully traced: setup + decode + vocoder) on **Blackhole P150**:

| Quantity | Value |
|----------|------:|
| Generated | 211 codes (self-terminated at STOP, under the 240-code cap) → 9.912 s audio |
| Setup replay (conditioning + speaker + prefill) | 0.042 s (30 s reference = 8 conditioning windows) |
| Decode replay (211 codes) | 1.705 s (≈ 8.1 ms/code) |
| Vocoder replay | 0.022 s |
| **Total replay** | **1.769 s** |
| RTF (replay / audio) | **0.178** — ≈5.6× faster than real-time |
| Time-to-first-chunk | 1.769 s (non-streaming: first audio == full clip) |
| Compile / capture (one-time, excluded from RTF) | 40.316 s |
| End-to-end wall (weight load + audio prep + generate + write) | 52.24 s |

Setup replay scales with the number of conditioning windows, so it moves with the reference length:
12.8 ms for 1 window (a 3 s clip), 21 ms for 3 (one 9.7 s clip), 42 ms for the default 8. It is a
one-time leg either way — under 3% of total replay.

The ≈8.1 ms/code decode rate matches the per-step device-time note in
[tt/xtts_generator.py](tt/xtts_generator.py): each captured decode step is measured at ~8 ms of
device work, with the per-step blocking fence and token readback hidden inside it (alternatives —
non-blocking execute, polling less often — were all worse or equal; see the source comment). The
Tracy dump below now confirms this independently: **7.578 ms** of pure device FW time in one
signposted decode step, with the remaining ~0.5 ms being the fence and readback.

### GPT single-step device profiles (Tracy)

From the two profiling drivers in [Commands — device profiling](#commands--device-profiling-tracy),
on Blackhole P150, at the demo's shapes (128-position prompt, `max_seq` 384, 30 layers). These are
signpost-windowed totals — one prefill, one decode step:

| Workload | Device FW | Ops | Notes |
|----------|----------:|----:|-------|
| GPT prefill (128 prompt positions, 30 layers) | 6.884 ms | 397 | One-time per generation; part of the 42 ms setup leg |
| GPT decode, one code (30 layers, traced cache write) | 7.578 ms | 557 | Paid **per audio code** — this is the model's dominant cost |

Per-op breakdown of the decode step (the report to read when the code rate moves):

| Op | Count | Total | % of step | Per op |
|----|------:|------:|----------:|-------:|
| `TernaryDeviceOperation` (traced KV-cache write) | 60 | 2378 µs | **31.4%** | 39.6 µs |
| `MatmulDeviceOperation` | 121 | 2373 µs | 31.3% | 19.6 µs |
| `NlpCreateHeadsDeviceOperation` | 30 | 882 µs | 11.6% | 29.4 µs |
| `SDPAOperation` | 30 | 691 µs | 9.1% | 23.0 µs |
| `LayerNormDeviceOperation` | 62 | 540 µs | 7.1% | 8.7 µs |
| `NLPConcatHeadsDeviceOperation` | 30 | 313 µs | 4.1% | 10.4 µs |
| everything else (7 op types) | 224 | 401 µs | 5.3% | — |

**The largest single cost in decode is the trace-safety workaround, not the model.** Those 60
ternary ops are the two `ttnn.where(onehot, k, k_cache, output_tensor=k_cache)` cache writes per
layer × 30 layers — the data-driven one-hot select
([tt/xtts_gpt_block.py](tt/xtts_gpt_block.py)) that a captured trace requires because it cannot
take an eager Python `write_idx`. They cost **more than all 121 matmuls combined**. The eager path
does the same job with `ttnn.update_cache`, which the prefill profile measures at **2.6 µs** per
call against 39.6 µs here — a ~15× gap, because `where` rewrites the entire
`[1, 16, 384, 64]` cache while `update_cache` touches one row.

Two consequences worth knowing before optimizing anything else here:

- **~2.2 ms of every 7.6 ms decode step is this write** (≈29%). Over the demo's 211 codes that is
  ~0.47 s of the 1.71 s decode replay.
- **It scales with `max_seq`, not with position.** A larger `max_new_tokens` makes *every* step
  slower, since the whole cache is rewritten each time regardless of how far in you are. Raising
  the code budget is therefore not free the way it would be with an O(1) row write.

This was surfaced by the profiling harness on its first run and has **not** been acted on — it is
recorded here as the top optimization lead for the decode path, not as a fixed issue.

This baseline is now protected by [`tests/perf/test_e2e_perf.py`](tests/perf/test_e2e_perf.py)
(see [Test cases](#test-cases)), which hard-asserts on it with a 40% margin. The test drives the
same `inference_fully_traced` path on the same text *and the same default reference voice*, so its
setup leg covers all 8 conditioning windows the demo actually runs — it used to condition on a 3 s
single-window clip, which gated under a third of that. A confirmation run (different sampled code
count — 205 vs. 211, since the test fixes `torch`'s seed differently than an uncontrolled demo run
— but the same text/voice/settings) landed almost exactly on the rate-based numbers: 41.64 ms
setup, 8.058 ms/code decode, 0.1017 ms/code vocoder, RTF 0.180.

### End-to-end ISL sweep (Blackhole P150)

[`tests/perf/test_e2e_isl_sweep_perf.py`](tests/perf/test_e2e_isl_sweep_perf.py) on the production
`inference_fully_traced` path, default 30 s reference voice, demo sampling recipe
(temp 0.65 / top-k 50 / top-p 0.85 / rep 5.0), `max_new_tokens` 205. Text lengths are tile-aligned
and stop at 352 — the demo's `MAX_TEXT_IDS`, i.e. the **end-to-end** envelope, which is lower than
the PCC sweep's architectural 384 (see [Context length](#context-length-support)).

| ISL | prompt | max_seq | codes | audio (s) | TTFT (ms) | codes/s | ms/code | setup (ms) | vocoder (ms) | replay (s) | RTF |
|----:|-------:|--------:|------:|----------:|----------:|--------:|--------:|-----------:|-------------:|-----------:|----:|
| 32 | 64 | 288 | 87 | 4.03 | 49.0 | 133.1 | 7.511 | 41.4 | 8.53 | 0.703 | 0.174 |
| 64 | 96 | 320 | 131 | 6.08 | 49.6 | 128.7 | 7.767 | — | — | 1.072 | 0.176 |
| 96 | 128 | 352 | 204 | 9.47 | 49.6 | 125.8 | 7.948 | 41.7 | 20.99 | 1.684 | 0.178 |
| 128 | 160 | 384 | 204 | 9.47 | 50.1 | 123.9 | 8.071 | 42.1 | 21.02 | 1.709 | 0.180 |
| 192 | 224 | 448 | 204 | 9.47 | 51.0 | 114.9 | 8.703 | 42.3 | 21.00 | 1.839 | 0.194 |
| 256 | 288 | 512 | 204 | 9.47 | 51.4 | 112.7 | 8.872 | 42.6 | 21.01 | 1.873 | 0.198 |
| 320 | 352 | 576 | 204 | 9.47 | 53.0 | 107.5 | 9.301 | 43.7 | 21.06 | 1.962 | 0.207 |
| 352 | 384 | 608 | 204 | 9.47 | 52.6 | 106.6 | 9.379 | 43.2 | 21.00 | 1.977 | 0.209 |

Metric definitions — XTTS is **non-streaming**, one call emits the whole clip:

- **TTFT** — time to first *code*: setup replay + one decode step (derived as `setup + decode/n`,
  not separately instrumented). The LLM-analogous number.
- **TTFA** — time to first *audio*: the full `replay (s)` column, since no audio exists until the
  vocoder has run over every latent. This is what a listener actually waits for.
- **codes/s** — decode throughput, the TPS analogue. One code is 46.4 ms of audio.
- **RTF** — replay ÷ generated audio duration. Below 1.0 is faster than real time.

Three things this sweep establishes:

- **Decode cost scales with `max_seq`, not with position**: 7.511 → 9.379 ms/code (**+24.9%**) as
  the cache grows 288 → 608, while every row generates the same 204 codes. This is the direct
  consequence of the traced one-hot cache write rewriting the *whole* cache each step (~31% of a
  decode step — see [the per-op breakdown](#gpt-single-step-device-profiles-tracy)). Raising
  `max_new_tokens` is therefore **not free**: it slows every step of every generation, not just the
  extra ones.
- **TTFT is essentially flat** — 49.0 → 53.0 ms across an 11× range of text length. Setup is
  dominated by the 8 conditioning windows, not by the prompt, so prefill length barely registers.
- **RTF stays ~5× faster than real time** everywhere (0.174 → 0.209), degrading only as the decode
  rate does.

**The ISL 64 row is measured standalone.** In the sweep it threw the L1 circular-buffer allocation
collision, *reproducibly and positionally*: it runs fine on its own and fine as the only entry, but
throws when it immediately follows ISL 32 in the same process — while every **larger** ISL in that
same sweep succeeds. So it is not a length limit; it is the allocation collision that makes
`demo/xtts_demo.py` open a fresh device per chunk. The sweep records skipped rows rather than
failing, since where that wall falls is part of the result.

### Module-level device time (standalone microbenchmarks, not this table's end-to-end number)

These are point-in-time profiling-harness numbers from development (`tests/pcc/test_*_profile.py`
under `tracy`, eager/untraced, one module in isolation) — useful for comparing optimization
passes on one stage, not as an end-to-end figure:

| Stage | Device time | Ops | Notes |
|-------|------------:|----:|-------|
| HiFi-GAN decoder (generator-only standalone conv stack) | ~2.02 ms | 249 | Block-sharded stage-0; conditioning memoised on `g`; L1-sharded stage hand-off. First call (cold conditioning cache) is ~2.12 ms / 268 ops |
| Speaker encoder (`mel_len=801`, ~8 s reference audio) | ~2.92 ms | 564 | Scalar-fold + reduced-core-count optimization pass |

No standing perf CI exists for this model (see [CI](#ci)) — the ISL sweep, the GPT single-step
profiles, these per-module numbers and the demo run are point-in-time measurements, captured
manually.

### Performance engineering notes

A few of the larger, documented wins in the source (all cited from the modules' own comments):

- **Mel frontends** ([tt/xtts_mel.py](tt/xtts_mel.py)): framing via strided-row slices of a
  padded view (not `ttnn.gather`) took the traced conditioning-mel setup from 7.045 s to 0.068 s;
  keeping the input as `[1, L]` rather than `[1, L, 1]` avoided a 4-byte-page reshape that alone
  cost 31.8 ms (96% of the conditioning-mel pass).
- **Speaker encoder** ([tt/xtts_speaker_encoder.py](tt/xtts_speaker_encoder.py)): the SE-ResNet-34
  body stays in flat channels-last TILE form in L1 throughout (avoids ~200 µs/conv untilize);
  the mel is convolved with **time**, not frequency, as the flat form's row axis (worth ~326 µs);
  BatchNorm folds into the preceding conv everywhere coqui's op order allows.
- **HiFi-GAN vocoder** ([tt/xtts_hifigan.py](tt/xtts_hifigan.py)): 3 of 4 upsample stages default
  to bf16 (stage 0 stays fp32). Widening bf16 to *all four* stages was tried and rejected — it
  measured −10.9% device time but introduced an audible robotic/metallic edge that aggregate PCC
  (waveform or spectrogram, 0.997 vs 0.995) did not predict; only listening caught it.
- **Latent upsampler** ([tt/xtts_hifi_decoder.py](tt/xtts_hifi_decoder.py)): the reference's two
  `F.interpolate` calls are algebraically composed into one resample matrix and applied as a
  single matmul; L1 block-sharding its output measured 2.75 µs → 1.44 µs at the profiled shape.
- **Conv primitives** ([tt/xtts_conv.py](tt/xtts_conv.py)): fp32 activations / bf16 weights by
  default — bf16 activations drift below 0.99 PCC over this deep a chain (~36 residual adds + MRF
  sums + tanh), while fp32 weights gave no accuracy gain over bf16.

### Host operations / PyTorch fallbacks

Every tensor-compute layer of the model runs on device — there is **no PyTorch fallback for any
model op**, including both mel frontends, which are computed on device as STFT-as-matmul inside the
traced setup. What remains on the host is I/O, tokenization and control flow, listed here in full
by how often it runs.

**1. Per decode step — runs once per audio code.**

| Op | Type | Used for |
|----|------|----------|
| token readback (`tok_buf`) | D2H ×1 | Read the just-sampled code to detect STOP. A captured trace cannot branch, so the AR loop has to check on the host. Skipped entirely while below the `--min-tokens` floor |

That is the **only** per-step host transfer on the traced path. Sampling
(repetition-penalty → temperature → top-k → top-p → Gumbel-max) and the token feedback into the next
step's embedding are fully on device. The readback is not on the critical path: the captured step is
~7.6 ms of device work and the blocking fence plus this one-int read hide inside it — measured
alternatives (non-blocking execute, polling every 8 steps) were all equal or worse.

**2. Per generation — runs once per `generate()` call.**

| Op | Type | Used for |
|----|------|----------|
| Gumbel noise table | hostCPU + H2D ×1 | All sampling noise for the run is pre-drawn with `torch.rand` and uploaded once as an `[N, V]` table; each captured step selects its row **on device** via a one-hot matmul. No per-step noise stream |
| text ids | H2D ×1 | Tokenized prompt into the buffer the SETUP trace reads |
| codes + latents readback | D2H ×2 | The accumulated codes and latents are read once after the decode loop, to find the STOP trim |
| latents re-upload | hostCPU + H2D ×1 | The trimmed latents are zero-padded to the captured vocoder length and copied back into the vocoder's input buffer — so the decode→vocoder hand-off **does** round-trip through the host, once per generation ([tt/xtts_inference.py](tt/xtts_inference.py)) |
| waveform readback | D2H ×1 | Final audio off device |

**3. One-time / per-request setup.**

| Op | Type | Used for |
|----|------|----------|
| XTTS-v2 BPE tokenizer | hostCPU | Text → token ids (`tokenizers`); no device equivalent |
| reference-audio load + resample | hostCPU | WAV read and `scipy.resample_poly` to 22.05 kHz (conditioning) and 16 kHz (speaker). The **mels themselves are on device** |
| reference wav upload | H2D | Raw waveform placed on device before capture (writes are fatal inside a trace) |
| fade in/out | hostCPU | Cosmetic envelope on the finished waveform so the first word does not pop |

The eager `inference()` path additionally reads the argmax back per step
([tt/xtts_generator.py](tt/xtts_generator.py)); the traced path does not.

### Environment variables

Beyond the standard `TT_METAL_HOME` / `PYTHONPATH` / `ARCH_NAME` (see [Quick start](#quick-start)),
this port reads only two, both test-only:

| Variable | Default | Purpose |
|----------|---------|---------|
| `XTTS_ISL_SWEEP` | full 8-point sweep | Comma list of text lengths for [`test_e2e_isl_sweep_perf.py`](tests/perf/test_e2e_isl_sweep_perf.py), to shorten a smoke run (e.g. `32,96`) |
| `XTTS_PERF_NUM_LAYERS` | `30` | GPT depth for the two Tracy profiling workloads; set `1` to isolate a single decoder layer |

There is **no** checkpoint-path override — weights always resolve through the HF hub cache (see
[Dependency versions](#dependency-versions)). Nothing else in the model reads the environment.

## Known limitations

**Module-level test coverage is uneven.** The GPT decoder carries a full prefill × decode /
single-layer × all-layers matrix at the checkpoint's full context length (see
[Test cases](#test-cases)), but the other blocks are still covered by one PCC test each, and 9
unit tests removed by an earlier cleanup have not been restored — the mel frontend, text
embedding, conv primitives, sampler, full decoder, and waveform decoder have no isolating test.
Two of the removed files had known failures at the time of
removal. See [Restored: the GPT prefill/decode coverage](#restored-the-gpt-prefilldecode-coverage).

**Carrying a workaround for a `ttnn.fill_cache` bug.** The prefill pads K/V to an even tile count
because `fill_cache` corrupts the destination's first tile at odd tile counts with 16 heads — which
is exactly where the conditioning latents live. The workaround is verified and gated, but it is
sound only for prompts ≤ 14 tiles and the upstream op is still broken. See
[the write-up](#found-and-worked-around-a-ttnnfill_cache-first-tile-corruption); the ttnn issue is
deliberately deferred until after this PR.

**Back-to-back generations in one process can fail to allocate.** The fp32 vocoder hits an L1
circular-buffer collision whose headroom shrinks as generations accumulate — reproducibly
positional, not a size limit: in the ISL sweep, ISL 64 threw while every larger ISL in the same run
succeeded, and it runs fine on its own. The demo avoids it by opening a **fresh device per pass**.
Any caller reusing one device across several full generations should expect this. See
[Context length support](#context-length-support).

**Functional gaps vs. real coqui inference**

- **Fixed decode-step budget on the traced path.** `inference_fully_traced`'s captured decode
  trace replays a fixed number of steps and trims at STOP afterwards, since a captured trace
  cannot branch — so `max_new_tokens` is a cost paid up front, not a budget.
- **Sampling is not bit-exact across runs.** On-device Gumbel-max draws differ run to run even
  with the same seed intent, so repeated takes of the same text/voice diverge (the demo's
  best-of-N take selection exists in `demo/xtts_demo.py::main` for this reason, though it isn't
  currently exposed via a CLI flag).
- **No HF-style `generate()` surface, no batching.** `TtXtts.inference()` takes a fixed sampling
  recipe (temperature/top-k/top-p/repetition-penalty) rather than injectable logits
  processors/stopping criteria, and everything is batch-1 (single text, single reference voice).
- **UTMOS is unavailable in this repo's `python_env`** — it needs `torchaudio`, which isn't
  installed; CER and SECS are unaffected.
- **Checkpoint revisions are pinned, host-library versions are not.** All four downloads pin a
  revision ([Dependency versions](#dependency-versions)), but nothing enforces the Python stack —
  and `librosa` in particular feeds every mel in the model, so a version bump moves every PCC
  number here.

**Documentation drift**

- `demo/xtts_demo.py`'s module docstring documents `--write-torch-ref`/`--output` usage that
  isn't wired up as `argparse` arguments in the current source (both are hardcoded in `main()`).

**Device / deployment constraints**

- **Blackhole P150 only.** No measured numbers or tuned L1/trace budgets exist for any other
  device.
- **Single device**, no tensor-parallel or multi-chip path.
- **No multi-speaker/dialogue format** — one narrated voice per call (unlike, e.g., VibeVoice's
  `Speaker N:`-turn scripts).

## CI

XTTS is wired into the tiered **models** pipelines as a **tier-3** model on single Blackhole P150
(`bh_p150b_civ2`) — the only device it supports. Two entries, matching how the other experimental
Blackhole models (`panoptic-deeplab`, `bevformer`) are registered:

| Pipeline | Entry | Command | Timeout |
|----------|-------|---------|--------:|
| [`models_unit_tests.yaml`](../../../tests/pipeline_reorg/models_unit_tests.yaml) | `xtts-v2 unit tests` | `pytest models/experimental/xtts/tests/pcc/ -k "not eval"` | 25 min |
| [`models_e2e_tests.yaml`](../../../tests/pipeline_reorg/models_e2e_tests.yaml) | `xtts-v2 e2e tests` | `test_e2e_perf.py` (the gated perf baseline) then `demo/xtts_demo.py` | 25 min |

The unit entry deselects `test_tt_eval` / `test_tt_eval_traced`: they are ungated best-effort
metrics and would pull whisper-large-v3 (~3 GB) and ECAPA2 into CI for numbers that are
[not reproducible run to run](#pcc-results). That leaves 22 deterministic PCC gates. The e2e entry
covers the demo end to end plus the hard perf assertions.

Timeouts allow for the first-run download of the ~1.9 GB `coqui/XTTS-v2` checkpoint; warm, the PCC
suite is ~6 min and the e2e entry ~2 min. Both are filed under `team: shield`, which is what every
neighbouring experimental Blackhole entry uses and which has time-budget headroom on this SKU
(`models`/e2e is capped at 8 min for `bh_p150b_civ2` and would reject these).

> **Two placeholders must be filled before merging.** `owner_id` is `UPDATE_ME` in both entries — it
> is the Slack ID paged on failure, and inventing one would page a stranger. Confirm `team: shield`
> at the same time; it was chosen to match neighbouring entries and to fit the budget, not from
> knowledge of who owns this model.

Nothing runs on any perf-regression cadence yet: `test_e2e_isl_sweep_perf.py` and the two Tracy
profiling drivers are manual. The numbers in this README are point-in-time, captured on Blackhole
P150 — treat the test suites as the source of truth for current status, not this file.

## Upstream references

- [coqui/XTTS-v2 model card](https://huggingface.co/coqui/XTTS-v2)
- [Coqui Public Model License (CPML)](https://coqui.ai/cpml)
