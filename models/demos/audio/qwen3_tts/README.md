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
| 1 | Mel front-end, 128 bins at 24 kHz | host | **done** |
| 2 | Speaker encoder (ECAPA-TDNN) → `[1, 2048]` | device | **done**, PCC 0.999996 |
| 3 | BPE tokenizer and prompt assembly (`frontend.py`) | host | **done** |
| 4 | Talker (28 layers, hidden 2048, MRoPE) | device | **prefill done**, PCC 0.995 |
| 5 | Code Predictor (5 layers, 15 steps per frame) | device | **done**, logits PCC 0.995 |
| 6 | Codec decoder → waveform | device | not started |

The speaker encoder reads a reference clip and emits one 2048-wide vector, which occupies a
single position of the talker's prompt. Its width matches the talker's hidden size, so
nothing projects between them.

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

The suite is self-contained: references are computed live in-process from the checkpoint, so
it needs only the checkpoint and, for the device tests, a card. 58 tests, 75 s warm.

```bash
pytest models/demos/audio/qwen3_tts/tests/                             # everything
pytest models/demos/audio/qwen3_tts/tests/test_checkpoint_loading.py   # host only
pytest models/demos/audio/qwen3_tts/tests/test_tokenizer.py            # host only
pytest models/demos/audio/qwen3_tts/tests/pcc/test_speaker_pcc.py      # speaker encoder
pytest models/demos/audio/qwen3_tts/tests/pcc/test_talker_pcc.py       # talker
pytest models/demos/audio/qwen3_tts/tests/pcc/test_code_predictor_pcc.py  # code predictor
```

`test_checkpoint_loading.py` derives every speaker-encoder tensor name and shape from
`config.json` and checks them against the file, so a checkpoint that stops matching its own
config fails there rather than surfacing later as a PCC miss. Nothing is skipped when the
checkpoint is missing: a skip would turn an unreachable checkpoint into a green run.

`test_tokenizer.py` pins the ids for a phrase in each of the ten languages and checks the
prompt scaffolding has the shape the model was trained on: the text prompt leaves a turn
open for the model to continue, a reference transcript closes its turn, and a VoiceDesign
instruction speaks as the user. It also pins the seam that is easiest to get wrong later:
language never enters the text stream, and every language id falls inside the talker's
3072-entry codec vocabulary rather than the 151k text one.

`pcc/test_talker_pcc.py` runs a real prompt, built from real token ids through the model's
own embedding and projection path, and reports three things:

| measurement | value |
|---|---|
| per layer, each fed the reference's fp32 input | 0.9998 to 0.99999 |
| end to end, 28 layers of bf16 | **0.9949** |
| codec top-1 token agreement | 24/26 |

The per-layer number measures the implementation, since feeding each layer the reference's
own input removes accumulated drift: a wiring error shows as one bad layer, rounding shows
as nothing. The end-to-end number carries 28 layers of bf16 rounding on top. The token check
is the one that says whether any of it matters, and it allows a disagreement only where the
reference is nearly indifferent; both misses here are near-ties where the device took the
reference's second choice, with logit gaps of 0.11 and 0.07.

The input choice is load-bearing. Random embeddings sit far outside the activation
distribution the weights were trained on, and the same graph scores 0.936 with 71% token
agreement on them. Raising device tensors to fp32 recovers almost nothing (0.9396), and fp32
weights change nothing at all, because the compute is bf16-class whatever the tensors say.

`pcc/test_code_predictor_pcc.py` runs a frame the model produced itself: the talker on a real
prompt, its last hidden state, codebook 0 from `codec_head`, then codebooks 1 to 15 decoded
greedily by the reference. Teacher-forced blocks hold 0.9965 to 0.99999 and the 15 output
heads reach 0.9946.

Greedy decode is scored per step, not as a sequence. One flipped token changes the input to
every later step, so comparing whole sequences measures the cascade rather than the port: the
device matches 12 of 15 steps but only 8 of 15 codes. Each disagreement is judged by how much
the reference prefers its own pick over the device's, which is the question worth asking.
Every one measured is a near-tie, widest gap 0.10 against logits spanning several units, and
at the single step where the device took the reference's third choice its top three sat within
0.042 of each other.

`pcc/test_speaker_pcc.py` gates every block and the embedding at **0.999**, not the usual
0.99. Upstream pads each convolution in reflect mode, which `ttnn.conv1d` cannot do, so this
port builds the mirrored columns by hand; substituting plain zero padding still scores 0.9961,
which a 0.99 gate would wave through. Two further tests keep the first one honest: the padding
is compared against `torch.nn.functional.pad` for an exact match, and the angle between a low
voice and a high one on device is checked against the same angle on the reference (0.9496 vs
0.9497), which a graph that ignored its input could not reproduce.

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
| `frontend.py` | host text path: tokenizer, prompt wrappers, language resolution |
| `tt/` | TTNN blocks |
| `reference/` | CPU references (PCC oracles); `reference/qwen/` is vendored upstream, Apache-2.0 |
| `tests/` | host tests, `tests/pcc/` for device correctness |

The vendored reference exists because the `qwen-tts` package pins transformers 4.57.3, which
conflicts with the version this repository runs. `reference/qwen/speaker_encoder.py` is a
byte-for-byte copy of the upstream encoder apart from two mechanical deviations recorded in
its header. Treat it as an oracle: any edit that is not a faithful copy makes it useless.
