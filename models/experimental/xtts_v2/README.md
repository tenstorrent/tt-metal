<!-- SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# XTTS-v2

Zero-shot voice-cloning text-to-speech ([coqui/XTTS-v2](https://huggingface.co/coqui/XTTS-v2))
on Tenstorrent hardware. XTTS-v2 clones a voice from a few seconds of reference audio and
speaks arbitrary text in it; the upstream model supports 17 languages, but this TTNN
implementation currently ships with an **English-only** text front-end (see
[Known limitations](#known-limitations)). Output is 24 kHz audio.

## Hardware

- **Board:** Wormhole N150 (single chip)

## Architecture

All four neural blocks run on device; the text/audio front-end and token sampling run on host.

| Block | Component | Where | dtype | PCC gate |
|---|---|---|---|---|
| 0 | BPE tokenizer + mel/STFT front-ends + prompt assembly (`frontend.py`) | host (pure torch) | fp32 | bit-exact vs coqui |
| 1 | Conditioning encoder + Perceiver resampler → `gpt_cond_latent` `[1,32,1024]` | device | fp32 | > 0.999 |
| 2 | ResNet speaker encoder → d-vector `[1,512,1]` | device | bf16 | > 0.99 |
| 3 | GPT decoder (30 layers): one-shot parallel **prefill** + Metal-Traced KV-cached **decode** | device | bf16 | > 0.999 |
| 4 | HiFi-GAN vocoder → waveform | device | fp32 | > 0.99 |

The GPT decode step is captured as a single Metal Trace at warmup and replayed for every
token of every request (no host dispatch overhead). Sampling (mel head, repetition penalty,
temperature/top-k/top-p) runs on host between traced steps with coqui's default parameters.

## Dependencies

Install model-specific dependencies:

```bash
pip install -r models/experimental/xtts_v2/requirements.txt
```

(`num2words` is used by the text normalizer to expand digits; without it, texts containing
digits raise.)

### Checkpoint

The checkpoint is fetched automatically from the HF hub on first use, or point `$XTTS_CKPT`
at a local `model.pth` (with `config.json` and `vocab.json` alongside):

```bash
huggingface-cli download coqui/XTTS-v2 --local-dir xtts_ref
export XTTS_CKPT=$(pwd)/xtts_ref/model.pth
```

> **License note:** the XTTS-v2 *weights* are released under the
> [Coqui Public Model License (CPML)](https://coqui.ai/cpml) — **non-commercial use only**.
> Coqui's own tooling gates the download behind an explicit terms-of-service acceptance
> (`COQUI_TOS_AGREED=1`); by downloading the checkpoint you accept the CPML. The code in
> this directory is Apache-2.0, but anything you synthesize with these weights is bound by
> the CPML.

## Quick Start

```bash
# Speak a sentence in the voice of a reference clip
python -m models.experimental.xtts_v2.demo.demo "Hello from Tenstorrent." \
    --ref my_voice.wav --out hello.wav --seed 0

# Interactive server (REPL): load + warm once, then one wav per typed line
python -m models.experimental.xtts_v2.demo.demo_server --ref my_voice.wav
```

`--ref` accepts a mono `.wav` (any sample rate) or a torch-saved `.pt` (raw waveform
tensor, `(tensor, sr)` tuple, or HF audio-sample dict). The REPL supports `\seed N`,
`\ref PATH` (recompute the voice), and `\quit`.

### Integration API

`XttsV2` (`tt/ttnn_xtts_model.py`) is the serving surface — one persistent object,
many requests:

```python
from models.experimental.xtts_v2.tt.ttnn_xtts_model import XttsV2

tts = XttsV2()                        # opens the device, loads + preprocesses weights
tts.warmup()                          # once: compile all programs + capture the decode trace
voice = tts.compute_voice(wav, sr)    # once per speaker: reference clip -> Voice
audio = tts.generate("Hello!", voice, seed=0)  # torch [1,1,N] float @ 24 kHz; repeatable
tts.close()
```

## Tests

The suite is self-contained: references are computed live in-process from the checkpoint
(no golden files needed) — it needs only the checkpoint and a device.

```bash
# Run all tests (11 tests)
pytest models/experimental/xtts_v2/tests/

# Individual blocks
pytest models/experimental/xtts_v2/tests/test_cond_pcc.py       # Block 1
pytest models/experimental/xtts_v2/tests/test_speaker_pcc.py    # Block 2
pytest models/experimental/xtts_v2/tests/test_gpt_trace_pcc.py  # Block 3 (traced decode)
pytest models/experimental/xtts_v2/tests/test_hifigan_pcc.py    # Block 4
```

## Performance

Measured on Wormhole N150 (warm, program cache + trace in place):

| Stage | Time | Notes |
|---|---|---|
| `compute_voice` | ~2–6 s | once per speaker (Blocks 1+2); the first clip of a new length pays one-time conv compiles |
| GPT prefill | ~18–30 ms | one-shot (18 ms @ 61-token prompt; ~30 ms @ ≈425) |
| GPT decode | ~10 ms/token | traced; each token = 46.4 ms of audio → **~4.7x real-time** decode |
| HiFi-GAN vocoder | ~1.2 s | fixed-length (compiled once at the 605-code model cap) |

A short sentence (~3.7 s of audio) generates end-to-end in ~2.0 s (~1.9x real-time — the
fixed-length vocoder dominates short utterances; longer utterances amortize it). One-time
warmup (program compiles + trace capture) takes ~20 s with a hot kernel JIT cache, longer
on a first-ever run when kernels build from scratch.

## Known limitations

- **English-only text normalization.** The BPE vocab is multilingual, but the cleaner
  tables (abbreviations, symbols, ordinals, number expansion) are transcribed only for
  `en` — extend `frontend.py` for other languages.
- **Model hard caps:** text ≤ 402 GPT tokens per utterance; audio ≤ 605 codes ≈ ~28 s of
  speech per utterance.
- **Single chip.** This class targets one N150. Data-parallel multi-chip decode exists as
  bringup scripts (`pipeline/phase_tt_mesh.py`) but is not part of the serving class.
- Sampling is stochastic (coqui defaults: temperature 0.75, top-k 50, top-p 0.85,
  repetition penalty 10); fix `seed` for reproducible output.
- **Reference-clip quality drives output quality.** Use a clean, mono clip of **~6 s**
  (a single ≤6 s conditioning chunk). Short clips (~3 s) audibly degrade output on some
  seeds — background noise "under" the speech and rough word onsets — identically on
  coqui's own CPU implementation (verified by matched-sample A/B), i.e. it is a property
  of the model's conditioning, not of this port. Text beyond ~250 characters per
  utterance also degrades (long silences, rushed pace — a training-distribution limit):
  split long text at sentence boundaries and generate per chunk.

## Directory layout

| Path | Role |
|---|---|
| `tt/` | TTNN blocks + the `XttsV2` serving class |
| `frontend.py` | coqui-free host front-end (tokenizer, mel/STFT, prompt assembly) |
| `reference/` | CPU reference implementations (PCC oracles) |
| `tests/` | per-block PCC tests (self-contained) |
| `demo/` | one-shot CLI + interactive REPL server |
| `pipeline/` | bringup/validation harness (coqui-in-the-loop, benches, mesh DP scripts) |
