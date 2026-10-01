# VoxCPM2 (experimental port in progress)

This directory starts a TTNN port of [OpenBMB VoxCPM2](https://huggingface.co/openbmb/VoxCPM2).
The reference implementation is pinned to OpenBMB/VoxCPM source revision
`f0c787f0937dc1c9a8f4f64d9a332d9c5da2e629`. Checkpoint dimensions are read from
its own `config.json`; an immutable checkpoint revision must be recorded with
validation captures.

## Implementation status

This is an initial component implementation, not a qualified end-to-end TTS
model. On 2026-10-01, 16 real-device replay cases covering 14 component types
passed PCC >= 0.99 against native CUDA captures. See the [accuracy report](validation/RESULTS.md).
Integrated TT synthesis and steady-state inference performance remain unqualified.

The model consists of a base MiniCPM language model, residual language model,
local audio-patch encoder, finite scalar quantizer, local diffusion transformer,
and AudioVAE V2. Text tokenization, checkpoint loading, and reproducible noise
creation are host boundaries. Learned network operations belong on TT.

Accuracy development starts from native CUDA generation, capturing each
component's actual inputs and outputs, then replaying those inputs into its TT
implementation. Captures are validation data, not the input to an integrated
TT inference claim. PCC is accompanied by relative RMS and maximum absolute
error because correlation alone cannot detect a scale or offset error.

## Development environment

Use the fork's built TTNN environment for device tests. Reference capture also
requires the pinned official VoxCPM package and its CUDA dependencies. Checkpoint
weights and tensor captures must live outside Git, preferably on the data disk.

Implementation and commands are expanded as component milestones land. Audio
quality, voice cloning, streaming, multilingual behavior, and integrated TT
speech synthesis remain unqualified beyond the specific component cases described in the accuracy report.

## Landed components and remaining work

| Component | Current code | Real-device CUDA PCC |
| --- | --- | --- |
| MiniCPM attention, RMSNorm, SwiGLU, residual blocks and final norm | Full-sequence TTNN forward | Covered by full-model prefill checks |
| Base and residual LM | Embedding-input prefill; raw KV cache / AR decode pending | Base 0.9999093; residual 0.9999660 |
| Local patch encoder | Learned special token and noncausal transformer | 0.9999862 |
| Finite scalar quantization | TTNN inference rounding and projections | 0.9999977 |
| Local DiT V2 | mu / timestep / condition / target sequence and velocity output | 0.9999425–0.9999570 across three timesteps |
| AudioVAE V2 | Deterministic nonstreaming encode/decode graph | Encoder 0.9991306; decoder 0.9997781 |
| Integrated synthesis | AR cache, diffusion solver, tokenizer, stop loop and WAV wiring pending | Pending |

The initial numerical configuration uses BF16 storage with HiFi4 and FP32
destination accumulation where supported. FP32 storage is also selectable for
component testing, particularly the AudioVAE, whose native CUDA weights use
FP32. BF16 storage was tested in the listed component cases; FP32 storage remains
unqualified. Explicit QK/softmax attention can differ numerically from CUDA fused
SDPA. Native BF16 timestep rounding is preserved before trigonometry. The codec
uses 32-row activation blocks and a 256 KiB L1-small device reservation. Its BF16
waveform has 7.50% relative RMS error despite PCC 0.9997781; passing correlation
does not establish a 1% amplitude/error bound.

Unsupported codec options fail explicitly: streaming, noise blocks,
concatenated sample-rate conditioning, and stride-one transpose blocks. The
initial replay path accepts a single sample rate per batch.

## CUDA reference and TT component replay

The independent dependency declaration is `pyproject.toml`. Install the native
CUDA reference with `uv sync --extra reference` from this model directory when
network access is available. Build/install TTNN from this fork into the model
venv before device replay, using the repository's build instructions. TTNN is an
external built dependency, not an unrelated model's shared Python environment.
Dependencies are now locked in `uv.lock`, with Torch/Torchaudio 2.8.0 and the
pinned official source. The `tt-runtime` extra declares Python dependencies
needed by the tested TTNN wheel. The tested remote environment used its own venv
with Torch 2.8.0+cpu and the installed TTNN 0.65.1rc17.dev6333+h3.3 wheel; it did
not use CUDA or another model's shared Python environment. This checks the model
code against that installed runtime, not a fresh build of this fork revision.

Run commands from the fork root with the model environment. The checkpoint must
already be a local snapshot of `openbmb/VoxCPM2`; `--checkpoint-revision` must be
its exact Hugging Face commit, not `main`. Capture records file hashes and replay
checks those bytes before uploading weights. Keep checkpoints and captures on
the data disk; they are excluded from Git.

```bash
uv run --locked --project models/experimental/voxcpm2 --extra reference \
  python -m models.experimental.voxcpm2.validation.capture_reference \
  --checkpoint "$HOME/data/voxcpm2/checkpoint" \
  --checkpoint-revision 32279effe8c19989596f05d353d1447f51d9e915 \
  --device cuda:0 --seed 42 --max-len 32 --inference-timesteps 10 \
  --text "The quick brown fox jumps over the lazy dog." \
  --output "$HOME/data/voxcpm2/cuda-reference"

# Use a device ID only after confirming/reserving its ownership.
timeout 600 uv run --project models/experimental/voxcpm2 \
  python -m models.experimental.voxcpm2.validation.replay_component \
  --reference "$HOME/data/voxcpm2/cuda-reference" \
  --checkpoint "$HOME/data/voxcpm2/checkpoint" \
  --component feat_encoder.forward --event-index 0 --device-id 0 \
  --dtype bfloat16 --l1-small-size 262144 --min-pcc 0.99 \
  --output "$HOME/data/voxcpm2/encoder-pcc.json"
```

Replay supports base/residual prefill hidden outputs, local encoder, quantizer,
local DiT estimator, LM/DiT/fusion/stop projections, stop activation, and codec
encode/decode. It reports PCC, relative RMS, maximum absolute error, and cold
launch time. Cold launch time may include compilation and is **not** a published
steady-state benchmark. MiniCPM cache output and `forward_step` are captured by
the CUDA harness but are not yet replayed or qualified on TT.

The capture also writes `reference.wav`, source/device/version metadata, seeds,
RNG states, valid KV prefixes, latent outputs, and checksummed tensor files.
Missing/incomplete captures, non-CUDA oracles, changed checkpoint bytes,
nonfinite tensors, shape mismatches, and missing outputs fail validation.

## Checks performed

```bash
python -m unittest discover \
  -s models/experimental/voxcpm2/tests -p 'test_*.py' -v
python -m compileall -q models/experimental/voxcpm2
git diff --check
```

These are host utility/contract checks; they do not substitute for CUDA or TT
model inference. The device test is skipped unless `VOXCPM2_CUDA_CAPTURE`,
`VOXCPM2_CHECKPOINT`, and `VOXCPM2_DEVICE_ID` are set. Select a replay component
with `VOXCPM2_COMPONENT`. A configured device test fails on a runtime error or
insufficient PCC rather than falling back to a CPU implementation.

The configured test suite also passed all 30 tests on the remote model venv,
including an actual TT local-encoder replay; no test was skipped in that run.
When hardware variables are absent, the device test is explicitly skipped.
