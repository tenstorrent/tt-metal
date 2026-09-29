<!-- SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Voxtral-TTS

Text-to-speech ([mistralai/Voxtral-4B-TTS-2603](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603))
on Tenstorrent hardware. Text plus a named voice preset in, 24 kHz audio out. One frame is
**80 ms of audio** (12.5 frames/s), so real time is 80 ms/frame.

## Hardware

- **Board:** Blackhole p150b (single chip). Runs on both the full 13x10 compute grid and cards with
  two Tensix columns fused off (11x10): the decode matmul grid is 11x7, bit-identical to the 12x6
  it replaced. Check `tt-smi -s` `ENABLED_TENSIX_COL` if a device run fails on a grid assertion.

Measured on this board: DRAM ceiling **367 GB/s**, per-op launch floor **~68 µs**. Those two
together invert the Wormhole N150's economics — bytes are cheap and launches are expensive, so
*deleting ops* wins here where the N150 wanted fewer, bigger kernels. Seven N150-tuned constants
did not survive the port.

## Architecture

All three neural blocks run on device; tokenization, prompt assembly and frame sampling run on host.

| Stage | Component | Where | dtype | PCC gate |
|---|---|---|---|---|
| tokenizer | Tekken BPE tokenizer + voice-preset prompt assembly | host (pure torch) | fp32 | bit-exact vs `mistral_common` |
| backbone | Autoregressive backbone (3.4B, 26 layers, DIM 3072): one-shot **prefill**, then Metal-Traced KV-cached **decode**, one hidden state per frame | device | bf16 acts; bfp8 wqkv/wo/FF1/FF3, **w2 bf16 for accuracy**; fp32 accumulation | prefill > 0.999, decode > 0.999 |
| flow model | Flow-matching acoustic transformer (390M, 3 layers): hidden state → 37 acoustic codes, ODE in 7 Euler steps | device | bf16 acts; bfp8 weights; **semantic head fp32** | velocity > 0.999 |
| codec | Codec decoder: codes → waveform, once per utterance | device | fp32, bf16 inside attention only | > 0.999 |

The frame loop is captured as a Metal Trace and replayed, so no per-frame host dispatch cost.
`tt/ttnn_voxtral_pipeline.py` wires the blocks together. **`reference/` is a pure-fp32 PyTorch
implementation and it is the ground truth, not the device.**

## Dependencies

`ttnn.graph` imports `graphviz` unconditionally:

```bash
uv pip install -r models/experimental/voxtral_tts/requirements.txt
```

> **Do NOT install `torchaudio`.** Its wheel ABI is broken against this torch, and merely having it
> importable breaks `transformers`, which takes the WER scorer down with it. `scipy.signal.resample_poly`
> covers the one thing it was needed for (24 kHz → 16 kHz).

`PYTHONPATH` needs **all three** entries — `$TT_METAL_HOME` alone resolves `ttnn` to an empty
namespace package, and `tools` holds the `tracy` module that `import ttnn` requires:

```bash
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$TT_METAL_HOME/ttnn:$TT_METAL_HOME/tools:$TT_METAL_HOME
```

### Checkpoint

> **License note:** the Voxtral-4B-TTS *weights*, including the 20 voice presets, are released under
> **CC BY-NC 4.0 — non-commercial use only**. By downloading the checkpoint you accept that license.
> The code in this directory is Apache-2.0.

The model directory holds `consolidated.safetensors`, `params.json`, `tekken.json` and
`voice_embedding/*.pt`. It is resolved as: an explicit `ckpt_path` / `--ckpt`, then
`$VOXTRAL_CKPT` (the directory, or its `consolidated.safetensors`), then a download of
`mistralai/Voxtral-4B-TTS-2603` into the local Hugging Face cache. To fetch it yourself:

```bash
hf download mistralai/Voxtral-4B-TTS-2603 consolidated.safetensors params.json tekken.json \
    "voice_embedding/*" --local-dir voxtral_model
export VOXTRAL_CKPT=$(pwd)/voxtral_model
```

The structural and reference tests run **without** the 8 GB download — they build random weights at
the real checkpoint shapes. The tests that need the real checkpoint find it through
`$VOXTRAL_CKPT` or an existing Hugging Face download, and skip otherwise; they never download.

## Quick Start

```bash
# Speak a sentence in one of the 20 shipped voices (--ckpt DIR to point at a model directory)
python -m models.experimental.voxtral_tts.demo.demo "Hello from Tenstorrent." \
    --voice neutral_male --out hello.wav --seed 0
python -m models.experimental.voxtral_tts.demo.demo --list-voices

# Interactive server (REPL): load + warm once, then one wav per typed line
python -m models.experimental.voxtral_tts.demo.demo_server --voice neutral_male
```

The REPL supports `\voice NAME`, `\voices`, `\seed N`, `\out PATH` and `\quit`.

The 15-prompt quality set, its WER scoring and the two-tag quality report are bring-up tooling (see
[Bring-up tooling](#bring-up-tooling)).

### Integration API

`TtVoxtralPipeline` (`tt/ttnn_voxtral_pipeline.py`) is the serving surface — one persistent object,
many requests, in the shape tt-inference-server drives (the same as xtts_v2's):

```python
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import TtVoxtralPipeline

tts = TtVoxtralPipeline()      # opens its own one-chip mesh device; or (mesh_device=..., ckpt_path=...)
tts.warmup()                   # once: compiles every prefill shape and codec bucket, captures the trace
wav = tts.synthesize("Hello from Tenstorrent.", "neutral_male", seed=0)   # [1, 1, N] @ 24 kHz
tts.close()                    # releases the trace, and the device if the pipeline opened it
```

`synthesize` chains the text front end, `generate` (prompt embeddings -> codes) and `decode`
(codes -> waveform); those two halves stay public for callers that need the codes.
`tts.last_timings` holds the last request's per-stage times. Progress goes through loguru.
A device you pass in must be opened with the L1 scratch and trace region the model needs: use
`open_device()` from the same module, or `l1_small_size=L1_SMALL_SIZE, trace_region_size=TRACE_REGION_SIZE`.

## Tests

The suite is self-contained: references are computed live in-process (no golden files), and the
structural half needs neither a device nor the checkpoint. Tests that need the checkpoint find it via
`$VOXTRAL_CKPT` (or an existing Hugging Face download) and skip without it.

```bash
# Run all tests
pytest models/experimental/voxtral_tts/tests/

# Per-block reference/architecture invariants (host only, no device, no checkpoint)
pytest models/experimental/voxtral_tts/tests/test_backbone_ref.py    # the backbone reference
pytest models/experimental/voxtral_tts/tests/test_flow_ref.py        # the flow model reference
pytest models/experimental/voxtral_tts/tests/test_codec_ref.py       # the codec reference

# On-device PCC against the fp32 reference (needs a device + the checkpoint)
# Naming: test_<block>_ref.py is the fp32 reference; test_<block>_pcc.py is the device.
pytest models/experimental/voxtral_tts/tests/pcc/test_backbone_prefill_pcc.py
pytest models/experimental/voxtral_tts/tests/pcc/test_backbone_decode_pcc.py
pytest models/experimental/voxtral_tts/tests/pcc/test_flow_pcc.py
pytest models/experimental/voxtral_tts/tests/pcc/test_codec_pcc.py
pytest models/experimental/voxtral_tts/tests/test_codec_request_path.py
pytest models/experimental/voxtral_tts/tests/pcc/test_model_teacher_forced_pcc.py
# Those gates skip the few positions/frames where the fp32 reference itself is decided by rounding
# (tests/conditioning_fixture.json, from make_conditioning_fixture.py, see Bring-up tooling);
# this host test keeps that list small.
pytest models/experimental/voxtral_tts/tests/test_conditioning_fixture.py

# The serving contract: TtVoxtralPipeline() owning its device, synthesize(), close()
pytest models/experimental/voxtral_tts/tests/test_serving_contract.py

# The traced frame loop -- the path that actually ships. Traced vs eager over FULL utterances:
# all 15 prompts x 3 seeds to their natural [END_AUDIO], asserting the sweep crossed sdpa's
# 512-position chunk boundary. ~9 min.
pytest models/experimental/voxtral_tts/tests/test_traced_frame_loop.py

# Request paths: a sequence of requests, not one in isolation
pytest models/experimental/voxtral_tts/tests/test_backbone_request_path.py
pytest models/experimental/voxtral_tts/tests/test_request_path_repeatability.py

# All 20 voice presets run
pytest models/experimental/voxtral_tts/tests/test_all_voices_smoke.py

# Shipped TTNN configuration is what it is documented to be
pytest models/experimental/voxtral_tts/tests/test_tt_defaults.py

# Host-side contracts: sampling/seed/CFG, request independence
pytest models/experimental/voxtral_tts/tests/test_sampling.py
pytest models/experimental/voxtral_tts/tests/test_request_path_repeatability.py

# Per-stage timings and RTF, gated against per-stage ceilings
pytest models/experimental/voxtral_tts/tests/perf/test_perf.py
pytest models/experimental/voxtral_tts/tests/perf/test_warmup.py   # its own module: opens a fresh device

# The recogniser itself, on known audio, before it gates anything: fp32-reference speech in
# English, Hindi and Arabic must score within a word of their measured WER; silence, noise and
# cut tails must score badly; a 39 s clip must transcribe to its last word (and must NOT with
# long form off). CPU only, ~12 min.
pytest models/experimental/voxtral_tts/tests/test_asr_calibration.py

# End-to-end intelligibility, one gate per language: the full request path, transcribed
# back with whisper-large-v3 and scored against the prompt. ~75 min, 180+ runs.
pytest models/experimental/voxtral_tts/tests/test_wer_languages.py
pytest models/experimental/voxtral_tts/tests/test_wer_languages.py -k hindi   # one language

# Naturalness per language (DistillMOS) against fixed per-language floors, set from a three-seed
# spread. Needs the isolated MOS venv once -- tests/mos_setup.sh -- and FAILS without it
# rather than skipping. ~15 min.
pytest models/experimental/voxtral_tts/tests/test_mos.py

# All on-device tests are marked `slow`, at module level. `-m "not slow"` is the
# host-only subset: ~140 tests, 1-2 min, no device and no checkpoint needed.
```

**Gate on real prompts, never random activations.** Random embeddings are off-manifold and read
PCC 0.892 where real prompts give 0.9994 on the same weights — the most expensive measurement
mistake in this port. `tests/reference_helpers.py` builds the real thing.

## Performance

Measured on Blackhole p150b (warm — program cache and trace in place), 15-prompt set, 3 seeds,
case 0 excluded because it pays one-time program-cache compilation:

| Stage | Time | Notes |
|---|---|---|
| Backbone prefill | 0.07–0.68 s | one-shot, scales with prompt length |
| Backbone decode | ~15.9 ms/frame | traced |
| Flow model | ~14.2 ms/frame | traced, 7 Euler steps |
| Codec decoder | ~3.5 ms/utterance | once per utterance, not per frame |
| **whole frame** | **25–27 ms/frame** | vs 80 ms real time → **~3x faster than real time** |

Per request, as `tests/perf/test_perf.py` measures it (best of 2, warm; decode includes the one-time
trace capture; 11x10 p150b, tt-metal main aa97958452, 2026-09-29):

| utterance | frames | audio | prefill | decode | ms/frame | codec | total | vs real time |
|---|---|---|---|---|---|---|---|---|
| short | 31 | 2.5 s | 0.05 s | 0.89 s | 28.55 | 0.02 s | 0.95 s | 2.61x |
| long | 459 | 36.7 s | 0.07 s | 11.69 s | 25.48 | 0.04 s | 11.80 s | 3.11x |

Quality: long-form **WER 0 wrong of 894 words**, MOS long-form **4.61**.

One-time `warmup()` takes **~13 s** with the kernel cache on local disk — 16 prefill shapes (4.5 s), the
flow model (0.3 s), 16 codec buckets (7.6 s) and one trace capture (0.2 s) — and much longer on a
first-ever run, when kernels build from scratch. It compiles **every** prefill shape and **every** codec bucket, so
no request pays a compile at request time. `TtVoxtralPipeline.warmed` records what was compiled;
`tests/perf/test_warmup.py` asserts it.

> **Quote ms/frame, not RTF, when comparing builds.** ms/frame is repeatable to 0.390 ms; RTF also
> carries prefill, the codec and trace capture, which amortise differently as frame counts change —
> two runs of *identical* code have read 0.4559 and 0.4415. And never compare against a number
> recorded in another session: an identical-code re-run has measured +0.75 ms/frame at 4.6σ purely
> from box state. Run the tier on the base commit, change something, run it again, compare the two.

## Known limitations

- **Single stream only.** This workload uses 0.37% of the chip's compute and ~49% of its DRAM, so
  single-stream latency is nearly exhausted and batching is the only order-of-magnitude lever left.
- **One voice-preset family**, the named presets shipped in the checkpoint; no zero-shot cloning
  from a reference clip.
- **Frame counts are not request-independent.** The pipeline object and its KV cache are reused
  across requests, and an utterance's frame count can depend on what ran before it in the same
  process — run a case alone before believing a changed frame count is a changed model.
- **MOS scoring needs a second venv** (`tests/mos_setup.sh` → `/tmp/mosvenv`), because
  DistillMOS pulls `torchaudio`, which must not enter the main venv.

## Directory layout

| Path | Role |
|---|---|
| `tt/` | TTNN blocks + the `TtVoxtralPipeline` serving class |
| `frontend.py` | host front end: text + voice name -> prompt embeddings |
| `demo/` | one-shot CLI + interactive REPL server |
| `reference/` | pure-fp32 PyTorch implementation — the ground truth / PCC oracle |
| `tests/` | reference invariants, on-device PCC (`pcc/`), perf (`perf/`), traced loop, WER, MOS |
| `generated/` | run artifacts (gitignored) |

## Bring-up tooling

The measurement tooling lives outside tt-metal, in the
[bring-up repo](https://github.com/acicovicTT/model-bringup) under `voxtral_tts/tools/`, and runs
against this checkout through `TT_METAL_HOME` (see its README): the quality report, the audio-set
generators and scorers, the fixture generators (`make_conditioning_fixture.py`,
`make_asr_calibration_fixture.py`, `dump_prompt_ids.py`), the upstream comparison and the probes.
