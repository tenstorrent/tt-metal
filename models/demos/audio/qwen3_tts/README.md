<!-- SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen3-TTS

Text-to-speech with voice cloning ([Qwen/Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base))
on Tenstorrent hardware. The model generates discrete audio tokens with a 1.7B decoder at a
12.5 Hz frame rate, then decodes them to a 24 kHz waveform with a 0.2B neural codec. Apache 2.0.

## Status

Bring-up in progress. This directory holds what is finished, and nothing that is not.

| Block | Component | Where | State |
|---|---|---|---|
| 0 | Checkpoint access (`weights.py`) | host | **done** |
| 1 | BPE tokenizer, prompt assembly, mel front-end | host | not started |
| 2 | Speaker encoder (ECAPA-TDNN) → `[1, 2048]` | device | not started |
| 3 | Talker (28 layers, hidden 2048, MRoPE) | device | not started |
| 4 | Code Predictor (5 layers, 15 steps per frame) | device | not started |
| 5 | Codec decoder → waveform | device | not started |

## Hardware

Bring-up runs on Blackhole. CI covers both single-chip SKUs, Wormhole N150 and Blackhole
P150, so an architecture-specific regression shows up on whichever side it breaks. P150
stands in for the dev machine's P300s: the model is single-device at batch 1, so a two-card
SKU buys no coverage, and P150 uses the standard shared weight cache while P300 runs in LFC
mode and would need to pull its own weights.

## Dependencies

Nothing beyond the tt-metal environment. The reader uses `safetensors` and `huggingface_hub`,
and the mel front-end will use `librosa` and `soundfile`, all of which ship in `python_env`.
This directory carries no `requirements.txt` on purpose: adding one that installs nothing
would put the file under a codeowner for no gain. Add it when a real dependency appears.

## Checkpoint

Fetched from the HF hub on first use and cached (3.6 GB), or point `$QWEN3_TTS_CKPT` at a
local directory holding `config.json` and `model.safetensors`:

```bash
hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir qwen3_tts_ref
export QWEN3_TTS_CKPT=$(pwd)/qwen3_tts_ref
```

`weights.py` resolves `$QWEN3_TTS_CKPT`, then `$HF_MODEL` (a hub id or a path, matching the
tiered-CI convention), then the default repo at a pinned revision. Override the revision with
`$QWEN3_TTS_REVISION`.

The checkpoint holds two top-level prefixes, `speaker_encoder.` (76 tensors, 12.0M parameters)
and `talker.` (the rest). Readers open the file lazily and name their keys, so speaker-encoder
work never materialises the talker.

## Tests

```bash
pytest models/demos/audio/qwen3_tts/tests/test_checkpoint_loading.py
```

Host only, no device. It derives every speaker-encoder tensor name and shape from `config.json`
and checks them against the file, so a checkpoint that stops matching its own config fails here
rather than surfacing later as a PCC miss. Nothing is skipped when the checkpoint is missing: a
skip would turn an unreachable checkpoint into a green run.

## CI

Registered in the Tier 3 unit pipeline on WH N150 and BH P150
(`tests/pipeline_reorg/models_unit_tests.yaml`, model identifier `qwen3-tts-1.7b-base`). The
identifier drops the frame rate that the HF name carries; `HF_MODEL` keeps the canonical
`Qwen/Qwen3-TTS-12Hz-1.7B-Base`, and the target resolver matches on that through its aliases.
Dispatch a single run from
[`all-model-tests`](https://github.com/tenstorrent/tt-metal/actions/workflows/all-model-tests.yaml)
with tier 3, type unit, and that identifier.

The end-to-end leg is deliberately absent. It lands with the first change that produces a
waveform, together with its own `e2e_tier3` budget; registering one before then would either
duplicate these tests or claim coverage that does not exist.

## Directory layout

| Path | Role |
|---|---|
| `weights.py` | checkpoint resolution and the speaker-encoder weight reader |
| `tests/` | host and PCC tests |
