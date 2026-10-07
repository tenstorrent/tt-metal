# VoxCPM2 component port: design and test plan

## Contribution boundary

This experimental contribution implements learned VoxCPM2 components in TTNN
and a reproducible CUDA-to-TT component validation path. It does not advertise
end-to-end TT text-to-speech support. No core TTNN/Metalium APIs or existing
model implementations are changed.

The reference is OpenBMB/VoxCPM source revision
`f0c787f0937dc1c9a8f4f64d9a332d9c5da2e629`, with `openbmb/VoxCPM2` checkpoint
revision `32279effe8c19989596f05d353d1447f51d9e915`. Configuration dimensions
come from checkpoint metadata rather than substituting a similar model.

## Architecture

| Path | Implementation | Qualified boundary |
| --- | --- | --- |
| Base/residual LM | MiniCPM full-sequence GQA attention, LongRoPE where configured, RMSNorm, SwiGLU, residual scaling | Embedding-input prefill hidden output; KV outputs and AR cache decoding excluded |
| Local encoder and FSQ | Noncausal patch transformer, learned special token, projections and finite scalar quantization | Captured native generated patch |
| Local DiT | Native mu/timestep/condition/target sequence, timestep embedding and velocity prediction | Three estimator calls within one patch's denoising schedule |
| Auxiliary heads | Encoder/fusion/LM-to-DiT/residual-to-DiT/stop projections and stop activation | Actual native component inputs |
| AudioVAE2 | Causal convolution, transpose convolution, residual blocks, Snake activation and sample-rate conditioning | Deterministic nonstream encoder and decoder |

Checkpoint preparation, structural masks/RoPE constants, tokenization, and
validation tensor transfer are host work. Learned activation computation in
the implemented components uses TTNN. Codec device tensors use
`[batch, 1, time, channels]`; the replay harness converts native channel-first
inputs at that explicit boundary.

The qualified numerical configuration is BF16 storage, HiFi4 math, and FP32
destination accumulation where supported. Attention materializes QK/softmax
rather than using native CUDA fused SDPA. Masks cover tile padding so padded
keys do not contribute to short local sequences. Timestep construction retains
the reference's BF16 rounding boundaries. Codec convolutions use 32-row
activation blocks and require a 256 KiB L1-small reservation in the tested
runtime. These are correctness/allocation choices, not measured speedups.

## Validation plan and evidence

1. Generate speech with the pinned official implementation on CUDA, with
   compilation, TF32, retries and VAD trimming disabled. Snapshot component
   inputs/outputs before mutable KV buffers can change; record RNG state,
   source/checkpoint revisions, checkpoint hashes and tensor checksums.
2. Replay the exact native input into each TT component with real checkpoint
   weights. Require PCC >= 0.99; also report relative RMS and maximum absolute
   error. Shape mismatches, empty/nonfinite tensors, changed checkpoint bytes,
   corrupt captures and missing outputs fail validation.
3. Run host tests for configuration, capture integrity and numerical comparison.
   The device test requires an explicit oracle, checkpoint and reserved device;
   absent hardware is a visible skip, never a substitute CPU result.
4. Exercise the complete decoder waveform and save it for listening comparison.
   A TT waveform decoded from CUDA latents establishes codec replay only.

The [results report](validation/RESULTS.md) and its machine-readable summaries
contain the real-device evidence. Hardware coverage is one Blackhole p150b,
batch 1, the documented input shapes and an installed TTNN runtime. A fresh
build of current upstream main, Wormhole, additional lengths/voices/languages,
streaming, and steady-state performance are not qualified by these tests.
PCC alone does not imply a 1% elementwise or amplitude bound.

## Follow-up milestones

The integrated model needs device KV-cache decoding, the diffusion solver,
reproducible noise handling, text/voice conditioning, stop/retry behavior and
WAV output wiring. Those require native CUDA comparisons at autoregressive and
denoising boundaries, followed by complete TT-generated speech and appropriate
audio-quality/performance coverage. FP32 storage and additional codec options
need separate qualification.

The upstream feature issue should agree this initial component scope and link
the design/test plan. Maintainer approval of the plan, current-main build/CI
validation and any codeowner-specific acceptance criteria remain submission
requirements; this document does not claim those approvals have occurred.
