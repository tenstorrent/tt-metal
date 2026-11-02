# generic_op on Quasar: handoff from session `tt-metal-fd` (2026-10-30)

> Kept verbatim (2026-11-02) from the untracked `generic_op_quasar.md` in the shared tree, so it isn't lost. It is an input hand-off from another session; see `QUASAR_MAP.md` for the current state.

From: a Claude session on `aus-wh-08` (WH machine, repo `/localdev/gchoudhary/tt-metal`, branch `main` @ `962a1f71e69`).
To: the sessions on `tensix-l-04` working on tt-blaze Quasar enablement.
Why a file: those sessions can't be messaged directly (no Remote Control link between machines).

This file is self-contained. The full audit lives only on aus-wh-08 at
`ttnn/cpp/ttnn/operations/generic/METAL2_PREPORT_AUDIT.md` (uncommitted). The parts that matter are copied below.

---

## TL;DR

1. **`ttnn.generic_op` cannot run on Quasar today, and porting it is the wrong approach.** It is a pass-through: it returns the caller's `ProgramDescriptor` verbatim. Quasar only runs on the Metal 2.0 host API (`ProgramSpec`). The fix is a **new `generic_op` overload that takes caller-built `ProgramSpec` + `ProgramRunArgs` per mesh range**, next to the descriptor overload. The descriptor overload stays as is.
2. **That overload is owned by Kostas (kfragkiadakis, TTNN), not by us.** The Python bindings for Metal 2.0 are his too. Audrey Kertesz confirmed this in a Slack DM to the user on 2026-10-30. Each item is about 1 week, tracked in Runtime's *Blaze-Required Features* table. **Do not implement the overload or the Metal 2.0 pybindings on this side.**
3. **Audrey's view: a tt-blaze-on-Quasar experiment is premature** until the Blaze-required runtime features land. Joseph (Blaze side) assigns a Blaze person once it is green-lit. The user told Audrey they would stop on their end.
4. **What's still useful from here:** the coverage gaps and Quasar requirements below, which may not be on Kostas's list yet. Treat them as input for him, not as a plan for us.

---

## What happened in the session (chronological)

1. The user asked for a Metal 2.0 pre-port audit of `ttnn/cpp/ttnn/operations/generic/`, following the `metal2_audit.md` recipe. The recipe docs are **not on main**; read them from `origin/akertesz/op-porting-recipe` (docs commit `c41f7f4ef0a`).
2. The audit came back **RED: not a port candidate, no portable subset.** Reasons:
   - **Structural.** The op builds nothing. Its contract is "give me a ProgramDescriptor." Translating arbitrary caller descriptors into specs at runtime isn't mechanical: positional RTAs need names, CB indices need DFB producer/consumer roles that can only be read off the kernels, and the kernels themselves need rewriting.
   - **TTNN readiness gate.** `Is able to port?` = no, because `TensorParameter relaxation` = `(legality - pending analysis)`.
   - **Feature compatibility.** All three Appendix A features reach the op through callers:
     - GlobalCircularBuffer
     - GlobalSemaphore
     - CBDescriptor `address_offset`, the CRITICAL one: `deepseek_v3_b1` uses it, and it needs a redesign agreed with the runtime team.
   - **Device 2.0.** Caller kernels are mostly Device 1.0:
     - `deepseek_v3_b1`: 47/62 files
     - `gpt_oss/tt`: 38/42
     - `gemma4/custom_ops`: 14/33
3. The user proposed a `ProgramSpec` overload and I agreed. The user's framing:
   - Callers rewriting their kernels is expected and not a concern.
   - What matters is that the new overload has **comprehensive coverage**, so callers *can* rewrite onto it.
   - This is a **new task, "Quasar support for generic_op"**, not a port.
4. The user briefly took ownership, and I drafted a design-session prompt. **Both are void:** Audrey's DM showed that Kostas owns it.

---

## Facts verified in the code (main @ 962a1f71e69)

- **Mixed factory concepts are allowed.** A `program_factory_t` variant may mix concepts; each alternative must satisfy exactly one (`ttnn/api/ttnn/operation_concepts.hpp:194-204`). A `MeshWorkloadSpecFactoryConcept` factory (`create_mesh_workload_artifacts`) can therefore sit next to the existing descriptor factory.
- **Artifacts are range-keyed.** `MeshWorkloadArtifacts` holds one `{MeshCoordinateRange, ProgramSpec, ProgramRunArgs}` per range (`ttnn/api/ttnn/metal_v2_artifacts.hpp:41-52`). That mirrors `MeshProgramDescriptor::mesh_programs`, so a spec factory can return caller specs verbatim.
- **No hash on `ProgramSpec`.** `program_spec.hpp` has no hash and no `operator==`. A structural hash has to be written, the analog of `compute_program_descriptor_hash` in `generic_op_device_operation.cpp:52`.
- **No Metal 2.0 Python bindings.** Nothing for `ProgramSpec`/`ProgramRunArgs` exists in `ttnn/cpp/ttnn-nanobind/`. Most generic_op callers are Python.
- **Tensor binding is UNVERIFIED.** `ProgramRunArgs::tensor_args` holds `reference_wrapper<const MeshTensor>`. Nobody has confirmed that caller-built run args map onto the op's io tensors correctly on both cache miss and cache hit.

## Coverage: what callers use through generic_op today vs Metal 2.0

| Capability | Users (evidence) | Metal 2.0 |
|---|---|---|
| A different program per mesh range, empty program for uncovered coordinates | `generic_op_program_factory.cpp:20-28` | ✓ `MeshWorkloadArtifacts::programs` |
| Kernels, CTAs, RTAs/CRTAs, variable-count blocks | all | ✓ `KernelSpec` + varargs advanced options |
| Semaphores | many | ✓ `SemaphoreSpec` |
| CBs backed by a tensor or buffer | ~26 caller files | ✓ `DataflowBufferSpec::borrowed_from` |
| GlobalCircularBuffer | fusion framework (`models/experimental/ops/descriptors/fusion/fusion.py:2150`) | **Gap**: waits on GlobalDataflowBuffer |
| GlobalSemaphore | gpt_oss fused_decode/dflash, deepseek_v3_b1 (sampling, pre/post_sdpa, persistent_loop) | **Gap, in flight**: Manish, ~1-2 weeks |
| CB `address_offset` | `deepseek_v3_b1/circular_buffer_utils.py:128`, `attention_block/op.py:2117-2413` | **Gap, redesign**: consult runtime team |
| Per-core allocation (`supports_per_core_allocation`) | deepseek_v3_b1 attention_block, moe, matmul_expert | **In flight**: Oleksandr, PR in review |
| Caller-supplied program hash (`custom_program_hash`) | gemma4 qkv_norm_rope_prefill, tt_dit, k2_horizon binding, `test_generic_op.cpp` | **Not found** |
| Runtime binary reload (`reload_table`) | `test_reload_host_support.py` | **Not found** |
| Blaze named args (`blaze_named_args`, #48704) | no direct callers; hashed at `generic_op_device_operation.cpp:69` | **Not found**: confirm whether Blaze still needs it |
| Compile without dispatch (`prepare_generic_op`) | `test_generic_op_preparation.py` | needs a mirror on the overload |

"Not found" means a grep of `tt_metal/api/tt-metalium/experimental/metal2_host_api/` found nothing. It is a lead to confirm, not a verdict.

## Quasar-specific requirements for the overload (likely missing from Kostas's scope)

Kostas's binding scope is "won't bind all AdvancedOptions, only as needed." Quasar callers need more than the Gen1 subset:

- **Gen2 hardware configs.** Callers need `DataMovementGen2Config` and `ComputeGen2Config` bound. A kernel with only a Gen1 config cannot run on Quasar.
- **Gen2 validator errors must reach Python readably and point at the offending spec.** Validation runs in `MakeMeshWorkloadFromSpecs` on cache miss. The errors to cover:
  - DM self-loop DFBs are rejected.
  - Multi-binding is Gen1-only.
  - SPSC is enforced per node.
  - DFB-count limits.
- **Quasar-legal constructs.** Expose `ScratchpadSpec`, which replaces self-loop and sync-free CBs. Also expose `LocalTensorAccessor` and the remote/semaphore options.
- **Non-zero-initialized semaphores don't run on Quasar.** Gen1 accepts them, so the overload should reject them, or at least warn, on Gen2.
- **Tests need a Quasar target** (craq-sim and/or the emulator). A wrong Gen2 config is invisible on WH/BH.

## Other generic_op anomalies found (non-gating)

- `tensor_args_t` holds references (`device/generic_op_device_operation_types.hpp:18-19`), which breaks the repo rule.
- `validate_inputs` is declared (`generic_op_device_operation.hpp:32`) but never defined.
- The hash may under-key CBs: `CBDescriptor::tensor` and `address_offset` are not hashed (`generic_op_device_operation.cpp:83-99`). A tensor-backed CB can hash the same as a local CB. Unverified whether this causes a wrong cache hit.
- `SemaphoreDescriptor::id` is not hashed.

---

## More from the Audrey DM (context for tt-blaze-on-Quasar)

- Blaze → Metal 2.0 is motivated mainly by **runtime features**, not Quasar.
- On Quasar + Blaze, Audrey said: "wishful thinking it's a port; reality more like a rearchitecture."
- **All Blaze microkernels have to migrate at the same time** to stay composable with each other. The proposed proof of concept is 2-3 micro-ops plus fictional mega-kernels composed from them.
- The user's spike was self-initiated; nobody assigned it.
- Audrey offered to keep the user in the loop. She said she and Joseph would welcome the user **taking point on the Blaze migration**. As of this writing the user has **not answered** that offer, so don't assume either way.

## Guidance for the tensix-l-04 sessions

- Don't build the generic_op `ProgramSpec` overload or the Metal 2.0 pybindings; that's Kostas's work. If your tt-blaze-on-Quasar work hits generic_op, it is blocked on his overload, plus GlobalSemaphore (Manish) and per-node allocation (Oleksandr).
- Ask the user before continuing any tt-blaze-on-Quasar implementation. Audrey called it premature, and the user said they'd stop.
- Don't use `ttnn/cpp/ttnn/operations/experimental/quasar/` as precedent. Those are hacky copies.
- Nothing has been sent to Kostas. A note listing the gaps above was proposed to the user and is not drafted yet.
