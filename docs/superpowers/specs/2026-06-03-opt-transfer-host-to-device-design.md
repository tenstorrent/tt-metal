# opt_transfer — Host→Device Residency Audit & Migration — Design

**Date:** 2026-06-03
**Status:** Design approved; ready for implementation planning
**Builds on:** the implemented `models/experimental/opt_transfer/` package (op-fusion transfer +
L1↔DRAM placement). This adds a **third residency axis: host↔device.**

## Problem

The framework has **no host-residency concept.** Verified against the code: `MemoryPlacement.buffer`
is `"L1"|"DRAM"` (both on-device); `decide_placement` never considers host; `codegen` always
`from_torch(..., device=device)`. It *structurally assumes every op is on the device* and only
chooses where in device memory. So it cannot see — let alone migrate — a **host-resident compute op.**

Motivating case: dots.ocr's `patch_embed` is a `Conv2d(3,1536,k=14,s=14)` + RMSNorm run on the **host
(torch/CPU)** — a deliberate "host-resident boundary" **copied from the qwen25_vl demo convention**.
At production resolution (19,520 patches) that host op is **~240 ms (~12% of e2e)** — measured. A
device matmul of the same shape is ~10–15 ms.

**Key insight — the donor signal is wrong here.** For L1↔DRAM, donors encode the right rule and the
matcher mimics them. For host↔device, donors often copy the *wrong* convention (dots.ocr inherited
host patch-embed from qwen25_vl), so mining donor placement would *reinforce* host. And a perf gate
on small inputs is misleading: small inputs aren't the production regime, and host can look fine
there. **At production sizes, device always wins for real compute.**

## Goal

Add a **host→device residency audit + migration** to `opt_transfer`, governed by a simple policy:

> **Every model compute op belongs on the device.** A host-resident compute op is a defect to
> migrate, regardless of donor convention. Migration is **unconditional for compute ops**;
> correctness (PCC at the *production* input shape) is the only gate. Perf is measured/reported, never
> a veto.

Validation target: migrate dots.ocr's `patch_embed` on-device — PCC of the vision tower stays
≥ 0.99 (currently 0.99998), `PatchEmbed` zone drops ~240 ms → ~10–15 ms (~12% e2e win).

## Non-goals

- **Not** migrating host *glue*: tokenizer, image load, rope/mask **metadata** precompute, control
  flow. "Model ops" = **tensor compute in the forward** (matmul/conv/norm/activation/attention) only.
- **Not** a perf-gated host↔device decision. Device-residency of compute ops is invariant; the perf
  number is reported, not used to keep an op on host.
- Does not change the L1↔DRAM placement axis (that stays size+budget+perf-gated, within-device).

## Architecture (a third residency axis above L1↔DRAM)

### 1. Detection — find host-resident compute ops

Two contexts:
- **Generating from the torch reference** (the framework's normal path): codegen already emits device
  ttnn ops, so the fix is a **policy guard** — the matcher/placement must **never adopt a `host`
  residency** for a compute op from `placement_observations` or donor convention. Device is invariant.
- **Auditing an existing TT impl** (dots.ocr): flag tensor-compute executed in `torch` inside the
  model forward that feeds a `from_torch(..., device=...)` upload — the "host-resident boundary"
  pattern. dots.ocr's `host_patch_embed` (`vision_tower.py`) is exactly this; it's even documented as
  the only host step. The audit reports each such op + its torch signature + the input size at
  production shape.

### 2. Decision — migrate, unconditionally (compute ops)

For each detected host compute op that is **device-supported** (the KB op-inventory already knows
`matmul`/`conv`/`rms_norm` exist as ttnn ops): **migrate to device.** No size threshold, no perf veto.
The op's torch computation maps to device ttnn ops via the existing matcher/codegen path (e.g. the
Conv2d patchify → im2col + `ttnn.matmul`; the RMSNorm → `ttnn.rms_norm`).

### 3. Codegen — emit the device op + rewire inputs

Codegen emits the device version and **rewires the boundary**: instead of `host_op(pixels) → tokens →
from_torch(tokens, device)`, the model now uploads the **raw input** (pixels) once and runs the op on
device. For patch-embed: upload pixel patches `[num_patches, 3*14*14]`, `ttnn.matmul` with the proj
weight (already on device), `ttnn.rms_norm`. The downstream device path is unchanged.

### 4. Verification — correctness only, at production shape

- **PCC ≥ 0.99 vs HF at the production input shape** (full-doc 19,520 patches — *not* a toy size; ties
  to the "validate at model shapes" rule). This is the only gate.
- **Perf measured + reported** at production shape (expected ~240 ms → ~10–15 ms) — informational.
- If PCC fails, the migration is rejected and reported (a real ttnn-vs-torch numerics issue to debug),
  never silently kept on host.

## Integration with existing package

| Piece | Change |
|-------|--------|
| `schema.py` | `MemoryPlacement` gains an optional residency notion, or a small `HostOp` audit record (op, torch_sig, produces, input_shape, source). Compute-op residency is **device-invariant** — no `"host"` value is ever *returned* by placement. |
| `residency.py` (new) | `audit_host_ops(model_or_trace)` → list of host-resident compute ops; `is_compute_op(op)` (vs glue); policy `must_be_on_device(op)`. |
| `codegen.py` | emit the device equivalent of a migrated host op (reuse the matcher/emitter path; patch-embed = im2col→matmul + rms_norm) and rewire the input boundary. |
| `graph.py` | the audit runs as part of the bring-up; migrations feed codegen; verify checks PCC at production shape. |
| matcher/KB | **policy guard**: ignore any `host` residency in donor/observations for compute ops. |

## Scope decomposition

1. **Residency audit + policy** (`residency.py`) — detect host compute ops, classify compute-vs-glue,
   assert device-invariance. Pure/offline, unit-tested.
2. **Patch-embed device emitter + boundary rewire** (`codegen.py`) — im2col→matmul + rms_norm on
   device; on-device PCC test vs the host reference at a representative shape.
3. **dots.ocr application** — migrate `host_patch_embed` in the real model, PCC the vision tower
   (≥ 0.99 vs HF at 19,520 patches), re-profile to confirm the ~240 ms → ~10–15 ms drop.

Thin vertical slice first: the patch-embed device emitter + its PCC check (sub-projects 1–2), then the
real dots.ocr migration (3).

## Open questions / risks

- **im2col on device.** The patchify needs the pixels arranged as `[num_patches, 3*temporal*14*14]`.
  If that packing is itself non-trivial on device, do the (cheap, metadata-like) reshape/unfold on
  host and run only the matmul+norm on device — the matmul is the 240 ms cost, not the unfold.
  Confirm where the real cost is before deciding how much to migrate.
- **PCC at bf16.** Host reference is fp32; the device matmul is bf16/HiFi. The tower PCC gate
  (0.99998 today) is tight — confirm the migrated patch-embed holds it (use fp32_dest_acc / HiFi4 if
  needed). This is the real risk, not the migration mechanics.
- **Compute-vs-glue boundary.** Mis-classifying a metadata op as "compute" would try to migrate
  something that shouldn't be. Keep `is_compute_op` conservative (matmul/conv/norm/activation/
  attention/elementwise on activations); everything else stays host.
- **Generality vs one-op.** This spec generalizes the *policy/audit* but only ships the *patch-embed*
  emitter. Other host ops (if any surface) get emitters incrementally, each PCC-gated — same model as
  the fusion-emitter registry.
