# Embedding + Decoder Fusion

## 2026-03-19

### Goal
Implement embedding-decoder kernel fusion by extending decoder fused op to contain an optional host i/o and embedding step.

### Details
- Currently we have a fused op 'decoder' and a micro op host_io/op.py
- Goal is to fuse H2D socket functionality as well as embedding (stage 0) into the decoder
- H2D socker read and embedding should be optional
- Alternatively, the input for the rms norm comes from a D2D socket (current behaviour)

### Review Summary
- The `deepseek_v3_b1` stack uses a descriptor-driven fused-kernel model:
  Python assembles per-device/per-core compile-time args, runtime args, CBs, semaphores, and sockets,
  then launches unified kernels specialized across NCRISC, BRISC, and TRISC roles.
- The current stage-0 embedding path lives in `micro_ops/host_io/op.py` and
  `micro_ops/host_io/kernels/fused_h2d_receiver_embedding.cpp`.
- The current decoder ingress path already supports socket-fed input through the decoder input/broadcast path
  in `fused_ops/attention_block/op.py` and `unified_kernels/broadcast.hpp`.
- Because of that, the right fusion point is the decoder input-reader boundary:
  both input modes should produce the same activation payload before RMSNorm.

### Scope Correction
- The decoder should support exactly two ingress modes:
  1. `D2D_SOCKET -> RMSNorm`
  2. `H2D -> embedding -> RMSNorm`
- There is no product need for a third `tensor/local` ingress mode.
- `H2D + embedding` is only required for the first decoder stage, which is dense.
- `MoE + H2D + embedding` is not required.

### Design Direction
- Keep the existing `D2D_SOCKET` decoder path unchanged for later layers.
- Add an `H2D_EMBED` decoder ingress mode on the same decoder input core that currently consumes socket input.
- Reuse the current stage-0 token packet contract:
  64-byte token page, token id in word 0, embedding row read from DRAM interleaved row-major tensor.
- Ensure that both ingress modes write into the same decoder input CB that feeds RMSNorm,
  so all logic after the ingress boundary remains unchanged.

### Implementation Plan
1. Define the decoder ingress contract.
   Add an explicit two-mode decoder ingress selection in the fused attention/decoder builder:
   `D2D_SOCKET` or `H2D_EMBED`.

2. Preserve the current D2D decoder path.
   Keep the existing `upstream_socket` flow and current later-layer behavior unchanged.
   This is the regression baseline.

3. Add the H2D+embedding ingress branch to the decoder input reader.
   Extend the decoder input-side kernel path so the decoder input core can:
   wait on an H2D socket,
   read the 64-byte token packet,
   extract the token id,
   perform the embedding DRAM read,
   place the embedding row into the same input CB used by the current decoder path.

4. Reuse the existing host-io embedding contract.
   Match the current assumptions in `host_io`:
   token page size,
   embedding tensor layout,
   embedding row page size,
   tensor accessor usage.
   This keeps stage-0 behavior consistent and gives us a reference implementation for validation.

5. Thread new args through the Python fused-op builders.
   Extend `fused_ops/attention_block/op.py` and `fused_ops/decoder_block/op.py` to emit the correct
   compile-time args, runtime args, and defines for the two ingress modes.
   Required data includes:
   socket config,
   token page size,
   H2D mode if needed,
   embedding buffer address,
   embedding accessor args,
   and any mode-selection compile-time flags.

6. Add strict validation in Python.
   Hard-fail on invalid configurations:
   both ingress modes enabled,
   missing socket for selected mode,
   missing embedding tensor in `H2D_EMBED`,
   wrong token page size,
   wrong embedding layout or memory config,
   activation width mismatch between embedding row and decoder input shape.

7. Validate the fused op before touching pipeline stage wiring.
   First prove the kernel-level fusion in unit and integration-style tests using the decoder op directly.
   Only after that, change stage-0 pipeline construction.

8. Integrate stage 0 with the fused decoder ingress.
   Update stage/pipeline wiring so stage 0 uses the decoder in `H2D_EMBED` mode instead of a separate
   embedding-only stage.
   Later decoder stages should continue using `D2D_SOCKET`.

9. Clean up obsolete stage-0 plumbing after validation.
   Once the fused path is stable, decide whether the standalone `host_io + embedding` stage remains
   as a reference/test utility or can be removed from the decode pipeline path.

### Intermediate Milestones
- M1: Decoder ingress builder supports exactly two modes: `D2D_SOCKET` and `H2D_EMBED`.
- M2: Dense decoder passes in `H2D_EMBED` mode on a single device.
- M3: Existing dense and MoE decoder flows still pass in `D2D_SOCKET` mode.
- M4: Stage-0 pipeline runs end-to-end with fused `H2D -> embedding -> decoder`.
- M5: Persistent-mode validation passes for the real stage-0 dense case using the fused ingress path.

### Testing Strategy
- Regression test: current `D2D_SOCKET -> decoder` behavior must remain unchanged.
- Equivalence test: the new `H2D_EMBED` decoder ingress should match the old `host_io` embedding path
  for the same token stream.
- Dense decoder correctness test: `H2D_EMBED` path should match the existing decoder reference/golden path.
- Dense decoder persistent-mode test: repeated decode iterations through `H2D_EMBED`.
- Dense/MoE regression tests: existing `D2D_SOCKET` decoder usage must continue to pass for later layers.
- Pipeline test: stage 0 uses `H2D_EMBED`, later stages use `D2D_SOCKET`, and final output still returns correctly.
- Negative tests:
  invalid ingress mode combinations,
  invalid embedding tensor layout,
  wrong token page size,
  missing required socket or embedding config.

### Acceptance Criteria
- One fused decoder op supports both required ingress modes.
- The first decoder stage can run `H2D -> embedding -> RMSNorm` without a separate embedding stage.
- Later decoder stages continue to run `D2D_SOCKET -> RMSNorm` unchanged.
- No requirement to support `H2D + embedding` on MoE stages.
- Kernel-level fusion is validated before broader stage/pipeline cleanup.

### Reminder
- Step-1 implementation keeps a temporary legacy no-socket fallback in the Python ingress resolver so
  existing local tensor-fed decoder tests do not break during the refactor.
- Before closing this task, explicitly check whether that fallback is still needed.
- If all decoder bring-up, regression, and pipeline tests have been migrated to the two real ingress modes,
  remove the fallback and keep only:
  `D2D_SOCKET -> RMSNorm`
  `H2D -> embedding -> RMSNorm`

### Implementation Update: Step 1
- Added explicit decoder ingress API plumbing in the Python fused-op builders.
- `AttentionBlock.op(...)` and `DecoderBlock.op(...)` now take an explicit ingress mode instead of
  inferring behavior from `None` or mixed socket arguments.
- Current enum values are:
  `D2D_SOCKET`
  `H2D_EMBED`
  `LEGACY_LOCAL_TENSOR`
- `LEGACY_LOCAL_TENSOR` is temporary and exists only to preserve current local tensor-fed decoder tests
  during the refactor.
- Added Python-side validation for all three modes:
  `D2D_SOCKET` requires `upstream_socket`
  `H2D_EMBED` requires `h2d_socket` and `embedding_tensor`
  `LEGACY_LOCAL_TENSOR` rejects socket and embedding resources
- Added `H2D_EMBED` validation for:
  64-byte token page size,
  row-major layout,
  interleaved memory layout,
  DRAM residency,
  embedding row size matching decoder input activation size,
  embedding dtype matching decoder input dtype.
- No kernel-side changes are included in step 1.
- No compile-time/runtime arg threading for `H2D_EMBED` has been added yet.
- Existing behavior should remain unchanged because the default temporary mode is `LEGACY_LOCAL_TENSOR`.

### Implementation Update: Step 2
- Added real `H2D_EMBED` threading through the fused attention/decoder builders.
- `AttentionBlock.get_program_context(...)` now threads ingress-mode-dependent metadata into the kernel build:
  selected external-input mode per device,
  H2D socket config when applicable,
  H2D token page size,
  embedding tensor address,
  embedding row size,
  H2D mode (`HOST_PUSH` vs `DEVICE_PULL`),
  BRISC positional compile-time arg for embedding tensor accessor.
- The external-input mode is only enabled on the actual input-source device.
  Other devices remain on the normal local/broadcast path.
- Updated both unified decoder kernels so the broadcast reader can run when:
  `skip_ccl == false`
  or `skip_ccl == true` with an external input mode active.
- Updated both unified decoder kernels so NCRISC does not blindly push the RMSNorm input CB in
  single-device external-input mode.
  This prevents double-signaling when BRISC owns the CB push for socket/H2D input.
- Extended `unified_kernels/broadcast.hpp` reader path with an `H2D_EMBED` branch:
  wait for one H2D token page,
  optionally perform PCIe pull when H2D mode is `DEVICE_PULL`,
  extract token id from word 0,
  perform DRAM embedding lookup,
  write embedding row into the decoder input CB,
  pop/notify/update the H2D socket.
- Kept the current D2D socket reader path intact.
- Added a focused single-device dense decoder unit test that exercises `H2D_EMBED` for both:
  `H2DMode.HOST_PUSH`
  `H2DMode.DEVICE_PULL`
- The new test uses `skip_ccl=True` and validates the decoder output against the existing dense decoder golden path.
- Current status:
  Python syntax verification is complete.
  No tests have been run yet.
