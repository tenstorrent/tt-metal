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
