# VoxCPM2 (experimental port in progress)

This directory starts a TTNN port of [OpenBMB VoxCPM2](https://huggingface.co/openbmb/VoxCPM2).
The reference implementation is pinned to OpenBMB/VoxCPM source revision
`f0c787f0937dc1c9a8f4f64d9a332d9c5da2e629`. Checkpoint dimensions are read from
its own `config.json`; an immutable checkpoint revision must be recorded with
validation captures.

## Implementation status

This is an initial component implementation, not a qualified end-to-end TTS
model. CUDA-reference PCC and Tenstorrent execution are pending. No inference
latency or supported hardware configuration has been measured for this port.

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
speech synthesis remain unqualified until actual device tests pass.
