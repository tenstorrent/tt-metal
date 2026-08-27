<!-- SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Voxtral-TTS

Text-to-speech ([mistralai/Voxtral-4B-TTS-2603](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603))
on Tenstorrent hardware. Text plus a named voice preset in, 24 kHz audio out. One frame is
**80 ms of audio** (12.5 frames/s), so real time is 80 ms/frame.

## Hardware

- **Board:** Blackhole p150b (single chip)

Measured on this board: DRAM ceiling **367 GB/s**, per-op launch floor **~68 µs**. Those two
together invert the Wormhole N150's economics — bytes are cheap and launches are expensive, so
*deleting ops* wins here where the N150 wanted fewer, bigger kernels. Seven N150-tuned constants
did not survive the port; see `VOXTRAL_TTS_BRINGUP.md` in the bringup repo.

## Architecture

All three neural blocks run on device; tokenization, prompt assembly and frame sampling run on host.

| Block | Component | Where | dtype | PCC gate |
|---|---|---|---|---|
| 0 | Tekken BPE tokenizer + voice-preset prompt assembly | host (pure torch) | fp32 | bit-exact vs `mistral_common` |
| 1 | Autoregressive backbone (3.4B, 26 layers, DIM 3072): one-shot **prefill**, then Metal-Traced KV-cached **decode**, one hidden state per frame | device | bf16 acts; bfp8 wqkv/wo/FF1/FF3, **w2 bf16 for accuracy**; fp32 accumulation | prefill > 0.999, decode > 0.999 |
| 2 | Flow-matching acoustic transformer (390M, 3 layers): hidden state → 37 acoustic codes, ODE in 7 Euler steps | device | bf16 acts; bfp8 weights; **semantic head fp32** | velocity > 0.999 |
| 3 | Codec decoder: codes → waveform, once per utterance | device | fp32, bf16 inside attention only | > 0.999 |

The frame loop is captured as a Metal Trace and replayed, so no per-frame host dispatch cost.
`tt/ttnn_voxtral_pipeline.py` wires the blocks together. **`reference/` is a pure-fp32 PyTorch
implementation and it is the ground truth, not the device.**

## Dependencies

`ttnn.graph` imports `graphviz` unconditionally:

```bash
uv pip install graphviz
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

Point `$VOXTRAL_CKPT` at a local `consolidated.safetensors`, or let the default path resolve under
the model's weights dir:

```bash
hf download mistralai/Voxtral-4B-TTS-2603 \
    consolidated.safetensors params.json tekken.json --local-dir voxtral_ref
export VOXTRAL_CKPT=$(pwd)/voxtral_ref/consolidated.safetensors
```

The structural and reference tests run **without** the 8 GB download — they build random weights at
the real checkpoint shapes. Only the device PCC, WER and perf tests need the real checkpoint.

## Quick Start

```bash
# Generate the 15-prompt quality set (audio + per-case timings) and score it
python models/experimental/voxtral_tts/scripts/generate_quality_set.py --tag mychange
python models/experimental/voxtral_tts/scripts/score_quality_set_scipy.py \
    models/experimental/voxtral_tts/generated/resultsmychange.json
```

> A one-shot `demo/demo.py` CLI and an interactive `demo/demo_server.py` REPL are **not yet
> written** — the quality-set scripts above are currently the only entry points. See the
> `VOXTRAL_TTS_NEXT_STEPS.md` in the bringup repo.

### Integration API

`TtVoxtralPipeline` (`tt/ttnn_voxtral_pipeline.py`) is the serving surface — one persistent object,
many requests:

```python
import ttnn
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import TtVoxtralPipeline, open_device

device = open_device()
pipe = TtVoxtralPipeline(device)
frames, t_prefill, t_decode = pipe.generate(embeds, max_frames=150, seed=0)
wav = pipe.decode(frames)          # [1, 1, T*1920] @ 24 kHz
```

## Tests

The suite is self-contained: references are computed live in-process (no golden files), and the
structural half needs neither a device nor the checkpoint.

```bash
# Run all tests
pytest models/experimental/voxtral_tts/tests/

# Individual blocks — reference/architecture invariants (host only, no checkpoint)
pytest models/experimental/voxtral_tts/tests/test_backbone_pcc.py    # Block 1 reference
pytest models/experimental/voxtral_tts/tests/test_flow_pcc.py        # Block 2 reference
pytest models/experimental/voxtral_tts/tests/test_codec_pcc.py       # Block 3 reference

# On-device PCC against the fp32 reference (needs a device + the checkpoint)
pytest models/experimental/voxtral_tts/tests/test_backbone_ttnn_pcc.py
pytest models/experimental/voxtral_tts/tests/test_flow_ttnn_pcc.py
pytest models/experimental/voxtral_tts/tests/test_codec_ttnn_pcc.py
pytest models/experimental/voxtral_tts/tests/test_model_teacher_forced_pcc.py

# Shipped TTNN configuration is what it is documented to be
pytest models/experimental/voxtral_tts/tests/test_tt_defaults.py

# Per-stage timings and RTF, gated against per-stage ceilings
pytest models/experimental/voxtral_tts/tests/test_perf.py
```

**Gate on real prompts, never random activations.** Random embeddings are off-manifold and read
PCC 0.892 where real prompts give 0.9994 on the same weights — the most expensive measurement
mistake in this port. `tests/reference_helpers.py` builds the real thing.

## Performance

Measured on Blackhole p150b (warm — program cache and trace in place), 15-prompt set, 3 seeds,
case 0 excluded because it pays one-time program-cache compilation:

| Stage | Time | Notes |
|---|---|---|
| Block 1 prefill | 0.07–0.68 s | one-shot, scales with prompt length |
| Block 1 decode | ~15.9 ms/frame | traced |
| Block 2 | ~14.2 ms/frame | traced, 7 Euler steps |
| Codec decoder | ~3.5 ms/utterance | once per utterance, not per frame |
| **whole frame** | **27.7 ms/frame** | vs 80 ms real time → **RTF 0.375** |

| utterance | codes | prefill | decode | decode/frame | codec + overhead | total | vs real time |
|---|---|---|---|---|---|---|---|
| short (3.3 s) | 42 | 0.07 s | 1.25 s | 30.93 ms | 0.02 s | 1.34 s | 2.50x |
| medium (13.0 s) | 163 | 0.14 s | 4.52 s | 27.88 ms | 0.24 s | 4.91 s | 2.65x |
| long (37.3 s) | 466 | 0.68 s | 12.77 s | 27.42 ms | 0.74 s | 14.18 s | 2.63x |

Quality at the same build: long-form **WER 0 wrong of 894 words**, MOS long-form **4.61**,
132/132 tests passing.

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
- **No demo CLI or server yet** (see Quick Start).
- **Frame counts are not request-independent.** The pipeline object and its KV cache are reused
  across requests, and an utterance's frame count can depend on what ran before it in the same
  process — run a case alone before believing a changed frame count is a changed model.
- **MOS scoring needs a second venv** (`tests/probes/mos_setup.sh` → `/tmp/mosvenv`), because
  DistillMOS pulls `torchaudio`, which must not enter the main venv.

## Directory layout

| Path | Role |
|---|---|
| `tt/` | TTNN blocks + the `TtVoxtralPipeline` serving class |
| `reference/` | pure-fp32 PyTorch implementation — the ground truth / PCC oracle |
| `tests/` | per-block reference + on-device PCC tests, config gates, perf, WER (self-contained) |
| `scripts/` | quality-set generation, WER scoring, the two-tag quality report |
| `generated/` | run artifacts (gitignored) |

Bringup history, per-block notes, known bugs and next steps live in the separate bringup repo as
`voxtral_tts/VOXTRAL_TTS_*.md`.
