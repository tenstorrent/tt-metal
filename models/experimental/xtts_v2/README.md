<!-- SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# XTTS-v2

Zero-shot voice-cloning text-to-speech ([coqui/XTTS-v2](https://huggingface.co/coqui/XTTS-v2))
on Tenstorrent hardware. XTTS-v2 clones a voice from a few seconds of reference audio and
speaks arbitrary text in it; the upstream model supports 17 languages, of which this TTNN
implementation ships **13** (see [Known limitations](#known-limitations)). Output is 24 kHz
audio.

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
| 4 | HiFi-GAN vocoder → waveform, at one of 7 bucketed lengths | device | fp32 | > 0.99 |

The GPT decode step and the vocoder are captured as Metal Traces at warmup and replayed (no
host dispatch overhead) — the decode step for every token of every request, the vocoder once
per request at whichever bucket fits the utterance. Sampling (mel head, repetition penalty,
temperature/top-k/top-p) runs on host between traced steps with coqui's default parameters.

## Dependencies

Install model-specific dependencies into tt-metal's venv (which is uv-managed and ships
without pip — a bare `pip install` falls through to the system Python and fails with
`externally-managed-environment`):

```bash
uv pip install -r models/experimental/xtts_v2/requirements.txt
```

### Checkpoint

The checkpoint is fetched automatically from the HF hub on first use, or point `$XTTS_CKPT`
at a local `model.pth` (with `config.json` and `vocab.json` alongside):

```bash
hf download coqui/XTTS-v2 --local-dir xtts_ref   # (`huggingface-cli download` on older huggingface_hub)
export XTTS_CKPT=$(pwd)/xtts_ref/model.pth
```

> **License note:** the XTTS-v2 *weights* are released under the
> Coqui Public Model License (CPML) — **non-commercial use only**.
> Coqui's own tooling gates the download behind an explicit terms-of-service acceptance
> (`COQUI_TOS_AGREED=1`); by downloading the checkpoint you accept the CPML.

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
tts.warmup()                          # once: compile all programs + capture the traces
voice = tts.compute_voice(wav, sr)    # once per speaker: reference clip -> Voice
audio = tts.generate("Hello!", voice, seed=0)  # torch [1,1,N] float @ 24 kHz; repeatable
tts.close()
```

## Tests

The suite is self-contained: references are computed live in-process from the checkpoint
(no golden files needed) — it needs only the checkpoint and a device.

```bash
# Run all tests
pytest models/experimental/xtts_v2/tests/

# Individual blocks
pytest models/experimental/xtts_v2/tests/pcc/test_cond_pcc.py       # Block 1
pytest models/experimental/xtts_v2/tests/pcc/test_speaker_pcc.py    # Block 2
pytest models/experimental/xtts_v2/tests/pcc/test_gpt_decode_pcc.py # Block 3 (traced decode)
pytest models/experimental/xtts_v2/tests/pcc/test_hifigan_pcc.py    # Block 4

# Per-stage timings and RTF, gated against per-stage ceilings
pytest models/experimental/xtts_v2/tests/perf/test_perf.py
```

## Performance

Measured on Wormhole N150 (warm, program cache + trace in place):

| Stage | Time | Notes |
|---|---|---|
| `compute_voice` | ~2–6 s | once per speaker (Blocks 1+2); the first clip of a new length pays one-time conv compiles |
| GPT prefill | ~21–22 ms | one-shot, incl. tokenize + prompt assembly; near-flat across prompt length (the prompt is padded to one of 7 length buckets) |
| GPT decode | ~9 ms/token | traced, tuned matmul configs; each token = 46.4 ms of audio → **~5x real-time** decode |
| HiFi-GAN vocoder | 0.09–0.71 s | traced, per length bucket: 0.09 s ≤ 3.5 s of audio … 0.71 s ≤ 28 s |

End to end at seed 0, with a 5.4 s reference clip: 2.31 s of audio in 0.56 s (**4.1x
real-time**), 10.77 s in 2.53 s (**4.3x**), 25.63 s in 6.08 s (**4.2x**). One-time warmup
(program compiles + trace captures) takes ~40 s with a hot kernel JIT cache, longer on a
first-ever run when kernels build from scratch. It compiles every prefill length and every
vocoder bucket, so no request pays a compile at request time.

## Known limitations

- **All 17 languages**, with two caveats. `ja` is romanized without number expansion (upstream
  runs neither cleaner for it), so digits reach the model as digits. `cs` ordinals read as
  cardinals: `num2words` has no Czech ordinals, and the pattern cannot tell `3.` (third) from a
  sentence-ending `50.` in any case, since Czech writes them identically — so `3. test` is spoken
  "tři test" rather than "třetí test". `zh`, `ja` and `ko` are romanized before the BPE because the
  vocab holds no CJK, each needing its package: `pypinyin`, `cutlet` (~45 MB of dictionary) and
  `hangul_romanize`.
- **`cs` and `hi` are noticeably weaker than the rest** and are gated to catch regression, not to
  assert the quality is acceptable. Czech appends audible words past the end of an utterance.
- **Model hard caps:** text ≤ 402 GPT tokens per utterance; audio ≤ 605 codes ≈ ~28 s of
  speech per utterance. Text needing more than that is cut off **mid-sentence** — the model
  stops wherever it has got to, and the rest is never spoken. Chunk long input by sentence.
- **Repeated input degenerates.** The same sentence several times over makes the model lose
  its place: measured near-silence and then nonsense partway through a 27 s utterance, while
  varied prose of the same length is clean. Lowering the repetition penalty only moves it.
- **Very short input over-runs.** A three-word prompt makes the model invent ~1.5 s of babble
  after the text before it stops — measured on every seed tried, and reproduced by the CPU
  reference generating on its own, so it is the model rather than this port. A six-word
  sentence does not. Give it a sentence, not a fragment.
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
