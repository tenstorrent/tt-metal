# ltx-rt Historical Fix Integration Specification

## Goal

Integrate only the production-relevant work recovered from `1defef6` and
`land/ltx-rt-fixes` into current `origin/ltx-rt`, prove it on the local
Blackhole server, and push the tested commit to `origin/ltx-rt`.

## Baseline

- Integration base: `origin/ltx-rt` at `3a41478432f`.
- Worktree: `/home/smarton/.claude/worktrees/ltx-rt-fixes-2026-09-03`.
- Serving checkout: `/home/smarton/tt-metal`.
- Server endpoint: `http://127.0.0.1:8081`.
- Device work is serialized through `user-tt-device-mcp`.
- The completion authority is the frozen
  `/home/smarton/ltx-rt-integration/goal_check.sh`.

## Required changes

### 1. Restore the complete kernel-prewarm interface

Keep the current prewarm engine and later sibling-worktree safety checks. Restore
only the merge-resolution losses:

- Restore the three `ttnn._ttnn.device` bindings for capture-only control,
  cold-start detection, and offline compilation.
- Export `TT_METAL_KERNEL_CAPTURE_ONLY=1` before the wrapper's stage-one command
  so a compound command such as `cd ... && python ...` passes the variable to
  Python.
- Restore and adapt the two stale-code regressions:
  `OfflinePrewarmReflectsEditedKernelBody` and
  `EditedKernelBodyForcesRecompileNotStaleCacheHit`.

Do not replace current prewarm C++ implementation files with historical blobs.

### 2. Complete AGMM program-cache identity

In `AllGatherMinimalMatmulAsyncParams`, add these compile-affecting fields to
`attribute_names` and `attribute_values()` in matching order:

- `fused_activation`
- `output_dtype`
- `compute_kernel_config`

Keep `chunks` and `dim`, which current `ltx-rt` already keys. Do not key the
value of `fused_ternary_scalar`; it is a runtime common argument and cache-hit
code rewrites it.

Add:

- A device-free C++ identity regression that proves otherwise-identical
  attributes produce distinct cache identity for each of the three fields.
- A same-process device regression that runs equal tensor/config shapes first
  without activation and then with `gelu_tanh`, with `chunks=1` for both, and
  checks each result against its own Torch reference.

### 3. Make the existing Ring-SDPA override effective

When `LTX_SDPA_RING_CHUNK=q,k` is set, use `(q,k)` while constructing every
matching `_ring_pc_by_n` entry. When it is absent, preserve the current per-N
defaults exactly.

Add a no-device selection test that proves:

- Unset 1080p entries remain `(96,256)` and `(192,512)`.
- `LTX_SDPA_RING_CHUNK=128,256` changes both mapped entries to `(128,256)`.
- A map miss uses the same override through the fallback program config.

Do not forward this variable through `ltx-server`.

### 4. Preserve the shared checkpoint path repair

Commit the already-authorized `audio_compile_bench.sh` default-path repair as a
separate change:

- LTX checkpoint:
  `/home/models/ltx-2.3/ltx-2.3-22b-distilled-1.1.safetensors`
- Gemma:
  `/home/models/gemma-3-12b-it-qat-q4_0-unquantized/`

The LTX file is the served file and has canonical SHA-256
`b33b7fe4bbfe084f484be4aaf90b0f1d95dca20d403ac4c0e037eb8c4f0af7cc`.

## Explicit exclusions

- Do not forward-port the historical equal-width gate-merge stack. Current
  default-on shared-gather dedup removes the duplicate collective without the
  historical BF8 precision conflict or full-width gate chunk.
- Do not port `LTX_RING_8K`; the explicit configuration already exists and the
  measured result was neutral.
- Do not port `TT_DIT_FUSED_MMRS` or `TT_DIT_AGMM_NUM_BUFFERS`; they are
  unvalidated global sweep parsers, not production configurations.
- Do not port latent fingerprints, per-step fingerprints, global operation
  counters, reblocking controls, or unscored seed loops.
- Do not duplicate the CLIP dependency failure; current `ltx-rt` already has it.

## Verification

### Host gates

- Build C++ and the required test targets through
  `/home/smarton/tt-workflows/scripts/build.sh cpp`, with ccache enabled by the
  configured `build_metal.sh -c` command.
- Run the new device-free AGMM identity regression.
- Check the three restored Python bindings in the built worktree.
- Run the Ring-SDPA selection test.
- Run shell syntax and wrapper command-propagation checks.
- Run `git diff --check` and relevant Python lint/compile checks.

### Device gates

- Query the broker queue first and never set a custom timeout.
- Run each restored stale-code prewarm regression separately.
- Run the same-process AGMM activation-alias regression.
- Run one prewarmed current LTX transformer/pipeline regression on
  `bh_2x4sp1tp0`.

### Server gates

- Fast-forward the serving `ltx-rt` checkout to the reviewed candidate without
  discarding the pre-existing path repair.
- Restart the local LTX server through its deployment lock.
- Require healthy/ready status and an empty failed-job result.
- Generate, download, and validate one 6-second 720p and one 6-second 1080p
  `ltx-fast` video. Each output must contain video and audio streams, have the
  requested dimensions and duration, and pass non-flat luma validation.

### Review and push

- Run a TT-aware whole-branch code review after all host, device, and server
  gates pass.
- Resolve all correctness findings and re-run affected gates.
- Fast-forward local `ltx-rt`, push `origin/ltx-rt`, and verify the remote SHA.
- Write evidence tied to the pushed commit, then require the frozen goal check
  to exit zero.

## Safety constraints

- No force push, destructive reset, raw device command, direct device `pytest`,
  foreign-job kill, or force reset.
- Keep compilation outside the device reservation.
- Use the three-stage prewarm wrapper for a cold full-pipeline build key.
- Do not weaken or edit the frozen goal check after implementation starts.
- Keep each independent production change in a reviewable commit.
