# LLK Matmul Helper API Redesign — Multi-Instance Orchestration Plan

## HOW TO USE THIS FILE

You are a Claude Code instance working on a multi-instance orchestration. The user will tell you:
"You are instance X in phase Y" (e.g., "You are instance 2 in phase 1").

**Instance numbering resets each phase.** Phase 1 has instances 1-4. Phase 2 has instance 1. Phase 3 has instances 1-3.

1. Read the GOAL, CURRENT STATE (including the "REFERENCE ONLY" warning), and CONSTRAINTS sections for context.
2. Find your specific assignment under PHASE Y > INSTANCE X.
3. Follow the instructions exactly. Write output to the specified file.
4. When done, end your response with: **"Completed phase Y for instance X."**
5. Do NOT do work assigned to other instances.
6. Do NOT propose design changes in phase 1 (analysis only).
7. Do NOT modify existing helper code in phases 1 or 2 (analysis/design only).

---

## GOAL

### Why this matters

Tenstorrent kernels are currently written by hand against a low-level API that requires manual register control, CB synchronization, and hardware-specific intrinsics. The goal of the helper library is to raise the abstraction level so that **AI agents (Claude Code) can write correct matmul kernels** — not just human developers. Today the low-level API is too complex for Claude to use reliably. The helpers need to be simple enough, well-documented enough, and hard-to-misuse enough that an AI agent can produce working kernels by composing them.

This isn't just about reducing boilerplate. It's about making matmul kernel development accessible to AI — which means the API design should optimize for **clarity and safety** over maximum flexibility.

### The problem

The current approach has a single `matmul_block` helper that works for the simplest case but can't handle the production kernel's `#ifdef` feature combinations. There are 60+ matmul kernel variants combining orthogonal features (L1 accumulation, bias fusion, activation fusion, transpose, mixed precision, etc.) behind `#ifdef` trees.

### What success looks like

A set of matmul compute helpers that:
1. **Cover the production kernel** (`bmm_large_block_zm_fused_bias_activation.cpp`, 827 lines, ~10 `#ifdef` paths) — this is the primary target
2. **Cover 3+ additional "not migratable" kernels** from the status doc
3. **Are usable by Claude Code** — an AI agent given the helper headers and docs should be able to write a correct matmul kernel for a new use case without needing to understand the underlying LLK intrinsics
4. **Scale without combinatorial explosion** — adding support for a new feature shouldn't require a new helper for every existing feature combination

The design approach (composable building blocks, a single well-parameterized helper, policy-based templates, or something else) is open — phase 2 should determine the right architecture based on the phase 1 analysis.

---

## CURRENT STATE

### Branch
- Working branch: `wransom/llk5` (based on `wransom/llk4`, which has the latest PR feedback addressed)
- Prior branches `wransom/llk3` and `wransom/llk4` diverged; llk4 is canonical (see LESSONS below).

### IMPORTANT: Existing matmul helper is REFERENCE ONLY

The `matmul_block` helper on this branch represents a prior attempt that works for the simplest case (basic sub-blocked matmul) but can't handle the production kernel's `#ifdef` feature combinations. **This work is expendable.** Use it to understand what was tried and what the limitations are, but do not feel constrained to preserve or extend it. If the right design requires throwing away `matmul_block` and starting fresh, that is acceptable. The non-matmul helpers (tilize, untilize, reduce, binary_op) are colleague-owned and stable — those patterns should be followed, not the matmul helper specifically.

Similarly, the "successfully migrated" kernels listed below may need to be re-migrated if the API changes. That's fine.

### LESSONS FROM PRIOR ATTEMPTS (read this carefully)

Two prior branches attempted matmul helpers. Their failures are instructive:

1. **`matmul_tile` helper was removed** (PR feedback). It only had one call site (`bmm.cpp`). The reviewer said it wasn't worth a helper — the inline code was clearer and preserved tile-by-tile CB pipelining without increasing L1 usage. **Lesson**: don't create a helper for a pattern with only one call site.

2. **Param structs were removed** (PR feedback). The original `matmul_block` used `In0BlockParams`, `In1BlockParams`, `OutSubblockParams` structs. The reviewer said: take only independent params (`block_w`, `in0_num_subblocks`, `in1_num_subblocks`, `out_subblock_h`, `out_subblock_w`) and derive computed values internally. **Lesson**: don't make the caller compute derived quantities.

3. **Unused enums were removed** (PR feedback). `InitUninitMode` and `ReconfigureRegisterDatatypeMode` had no call sites using non-default values. **Lesson**: don't add configurability nobody uses. Add it when a real call site needs it.

4. **Programming examples can't use kernel_lib** (JIT compile failure). Programming example kernels are JIT-compiled at runtime and installed separately from ttnn headers — `kernel_lib/` headers aren't in the installed include paths. **Lesson**: don't migrate programming examples to use helpers.

5. **`matmul_block_fused_bias` caused device hangs** (on llk3, never made it to the PR). An attempt to migrate the production kernel's fused bias path led to hangs, revealing the monolithic helper design was incompatible with the production kernel's complex state management. **Lesson**: the production kernel's `#ifdef` paths have subtle interactions that a monolithic helper can't capture.

### Existing helpers in `ttnn/cpp/ttnn/kernel_lib/`

| File pair | Purpose | Status |
|-----------|---------|--------|
| `matmul_block_helpers.hpp/inl` | Sub-blocked matmul with spill/reload. Wraps `mm_block_init` + `matmul_block`. Simplified API with flat params + PostComputeFn. | **Expendable** — redesign may replace |
| `tilize_helpers.hpp/inl` | Tilize (row-major to tiled). Colleague-owned. | Stable — follow patterns |
| `untilize_helpers.hpp/inl` | Untilize (tiled to row-major). Colleague-owned. | Stable — follow patterns |
| `reduce_helpers_compute.hpp/inl` | Reduce ops with f32 scaler. Colleague-owned. | Stable — follow patterns |
| `binary_op_helpers.hpp/inl` | FPU eltwise binary with CB policy. Colleague-owned. | Stable — follow patterns |
| `copy_tile_helpers.hpp/inl` | Copy tile utilities. | Stable |
| `cb_helpers.hpp/inl` | CB wait/pop/reserve helpers. | Stable |
| `common_types.hpp` | NoAccumulation, NoOp shared types. | Stable |
| `dest_helpers.hpp` | Dest register helpers. | Stable |
| `l1_helpers.hpp` | L1 memory helpers. | Stable |

### Shared API patterns across all helpers

All helpers in kernel_lib use these conventions:
- **Namespace**: `compute_kernel_lib` (types in sub-namespace like `matmul_block_config`)
- **Template params**: CB indices are compile-time; dimensions are runtime
- **Flat runtime params preferred**: pass independent values, derive computed quantities internally (per PR feedback)
- **PostComputeFn**: Functor template param for fused operations (default: NoPostCompute/NoOp)
- **File structure**: `.hpp` has types + function declaration + `#include "*.inl"` at bottom; `.inl` has implementation
- **Runtime asserts**: validate dimensions > 0, dest register limits, CB capacity

### Successfully migrated kernels (on this branch)

| Kernel | Helper used |
|--------|-------------|
| `matmul/.../bmm_large_block_zm.cpp` | matmul_block |
| `programming_examples/matmul/.../bmm_large_block_zm.cpp` | matmul_block |

Note: `bmm.cpp` was reverted to inline code (matmul_tile removed per PR feedback). Programming examples other than bmm_large_block_zm were reverted (JIT include path issue).

### NOT migratable with current helpers

| Kernel | Blocking features |
|--------|-------------------|
| `bmm_large_block_zm_fused_bias_activation.cpp` | PACKER_L1_ACC, FP32_DEST_ACC_EN, SKIP_COMPUTE, in0_transpose paths (~70% of the file) |
| `bmm_large_block_zm_fused_bias_activation_gathered.cpp` | Same + multi-device CCL gather |
| `conv2d/conv_bmm_tilize.cpp` | Fused tilize + matmul + bias with custom CB switching |
| `transformer/sdpa/compute_streaming.hpp` | Complex SDPA pipeline; matmul is one stage |
| `experimental/minimal_matmul/compute.cpp` | Out-of-order packing (`pack_tile<true>`) + L1_ACC + ternary |
| `experimental/matmul/group_attn_matmul/...` | Uses matmul_tiles + pack_untilize_dest |
| `experimental/deepseek/mla/matmul_wo/compute.cpp` | Ring-distributed matmul, custom tile indexing |
| `experimental/deepseek/moe/moe_gate_mm/compute.cpp` | MoE dispatch with multiple matmul_block configs |
| `experimental/ccl/moe_compute/compute.cpp` | SILU + eltwise multiply interleaved with matmul |
| `experimental/topk_router_gpt/compute.cpp` | matmul + binary_dest_reuse + bias + topk + softmax |
| `experimental/ccl/llama_all_gather_matmul_async/...` | Fused all-gather + matmul + bias + CCL |

### Production kernel `#ifdef` feature dimensions

The 827-line `bmm_large_block_zm_fused_bias_activation.cpp` uses these compile-time switches:
- `PACKER_L1_ACC` — L1 accumulation (avoids spill/reload to intermediate CB)
- `FP32_DEST_ACC_EN` — 32-bit destination accumulation
- `FUSE_BIAS` — Row-broadcast bias addition after matmul
- `SFPU_OP_INIT_ACTIVATION` / `SFPU_OP_FUNC_ACTIVATION` — Fused SFPU activation
- `PACK_RELU` — Fused relu via pack_relu config
- `IN1_TRANSPOSE_TILE` — Transpose B tiles before matmul
- `SKIP_COMPUTE` — Skip matmul (used for bias-only path)
- `MATMUL_DRAM_SHARDED` — DRAM sharded weight loading

These combine to create ~10 distinct code paths through the kernel. The current helpers only cover 1 path (no L1_ACC, no FP32_DEST_ACC, no transpose).

---

## CONSTRAINTS

1. **Respect existing non-matmul helpers.** tilize, untilize, reduce, binary_op helpers are colleague-owned. Follow their patterns. Do not modify them.
2. **Backward compatibility.** `bmm_large_block_zm.cpp` uses `matmul_block` today — ensure it still works (or re-migrate it) with any new API.
3. **Compute-only scope.** Helpers are for compute kernels (TRISC threads). Dataflow/reader/writer helpers are a separate effort.
4. **No TTNN name collisions.** Helper names should not collide with higher-level TTNN concepts.
5. **Architecture support.** Helpers must work on Wormhole B0 and Blackhole. Quasar support is a plus but not required.
6. **Testing — two kinds required.** (a) **Isolated helper tests**: write C++ integration tests that exercise the helpers directly, independent of any TTNN op. These test the helpers themselves. (b) **Regression tests**: run existing matmul Python tests (588 tests) and C++ integration tests to ensure nothing is broken by kernel changes.
7. **No environment variable tricks.** Helper behavior must NOT depend on environment variables. Feature selection must be handled through template parameters, runtime parameters, or compile-time defines set by the host-side op code (the normal kernel JIT compilation path). Claude sometimes tries to use env vars as a shortcut — do not accept this.
8. **No programming example migrations.** Programming example kernels are JIT-compiled and can't include `kernel_lib/` headers (they're not in installed include paths).
9. **Flat params over structs.** PR reviewer preferred flat independent params over param structs. Derive computed values internally.
10. **Only add configurability with real call sites.** Don't add enums/modes unless a concrete kernel migration needs them.

---

## KEY FILE PATHS

### Helpers (the code being redesigned)
```
ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp
ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.inl
ttnn/cpp/ttnn/kernel_lib/common_types.hpp
ttnn/cpp/ttnn/kernel_lib/cb_helpers.hpp
ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp
```

### Prior attempts (on wransom/llk3, NOT on this branch — use `git show origin/wransom/llk3:<path>` to read)
```
ttnn/cpp/ttnn/kernel_lib/matmul_tile_helpers.hpp/inl    — REMOVED per PR feedback
ttnn/cpp/ttnn/kernel_lib/matmul_block_fused_bias_helpers.hpp/inl — caused device hangs, never merged
```

### Production matmul kernels (primary migration targets)
```
ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp
ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp
```

### Other matmul kernels (secondary migration targets)
```
ttnn/cpp/ttnn/operations/experimental/conv3d/device/kernels/compute/compute.cpp
ttnn/cpp/ttnn/operations/conv2d/device/kernels/compute/conv_bmm_tilize.cpp
ttnn/cpp/ttnn/operations/moreh/moreh_matmul/device/kernels/moreh_matmul.cpp
ttnn/cpp/ttnn/operations/experimental/matmul/group_attn_matmul/device/kernels/compute/
ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp
```

### Reference matmul kernels (test/example implementations)
```
tests/tt_metal/tt_metal/test_kernels/compute/matmul.cpp
tests/tt_metal/tt_metal/test_kernels/compute/matmul_block.cpp
tests/tt_metal/tt_metal/test_kernels/compute/matmul_large_block.cpp
tests/tt_metal/tt_metal/test_kernels/compute/matmul_large_block_zm.cpp
tests/tt_metal/tt_metal/test_kernels/compute/matmul_large_block_generalized.cpp
tests/tt_metal/tt_metal/test_kernels/compute/matmul_with_bias.cpp
tests/tt_metal/tt_metal/test_kernels/compute/bmm.cpp
tests/tt_metal/tt_metal/test_kernels/compute/bmm_large_block_zm.cpp
tests/tt_metal/tt_metal/test_kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp
tests/tt_metal/tt_metal/test_kernels/compute/bmm_large_block_zm_mixed_precision.cpp
tests/tt_metal/tt_metal/test_kernels/compute/bmm_tilize_untilize.cpp
tests/tt_metal/tt_metal/test_kernels/compute/transformer_attn_matmul.cpp
```

### DeepSeek V3 unified kernels (modern composability reference)
```
models/demos/deepseek_v3_b1/unified_kernels/matmul.hpp
models/demos/deepseek_v3_b1/unified_kernels/dram_streaming_matmul.hpp
models/demos/deepseek_v3_b1/unified_kernels/dram_streaming_matmul_compressed.hpp
models/demos/deepseek_v3_b1/unified_kernels/dram_streaming_experts_matmul.hpp
models/demos/deepseek_v3_b1/unified_kernels/kn_sliced_matmul.hpp
models/demos/deepseek_v3_b1/unified_kernels/matmul_compressed.hpp
```

### Low-level LLK APIs (what helpers wrap)
```
tt_metal/hw/inc/api/compute/matmul.h
tt_metal/hw/inc/api/compute/experimental/matmul_custom.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_math_matmul_api.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_unpack_AB_matmul_api.h
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_math_matmul_api.h
```

### Non-matmul helpers (patterns to follow)
```
ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp
ttnn/cpp/ttnn/kernel_lib/tilize_helpers.inl
ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp
ttnn/cpp/ttnn/kernel_lib/untilize_helpers.inl
ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp
ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl
ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp
ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.inl
```

### Status doc
```
docs/matmul_helper_status.md
```

---

## PHASE 1: PARALLEL ANALYSIS

All 4 instances run simultaneously. Each produces FACTS, not design proposals. Each writes output to the specified file.

### INSTANCE 1 — Feature Dimension Matrix

**Output file**: `docs/analysis_feature_matrix.md`

**Task**: Read every matmul compute kernel in the codebase and build a comprehensive feature taxonomy.

**Steps**:
1. Read `docs/llk_matmul_orchestration.md` (this file) for context.
2. Read every kernel listed under KEY FILE PATHS > "Reference matmul kernels", "Production matmul kernels", and "Other matmul kernels". Also search for any matmul compute kernels not listed (use `Grep` for `matmul_tiles\|matmul_block\|mm_init\|mm_block_init` in `.cpp` and `.hpp` files under `ttnn/` and `tests/`).
3. For each kernel, document these feature dimensions:

| Column | Values |
|--------|--------|
| File path | full path |
| Blocking strategy | tile-by-tile / sub-blocked / large-block-generalized |
| K-dim spill/reload | yes/no (uses intermediate CB for multi-block K reduction) |
| Fused tilize input | yes/no |
| Fused untilize output | yes/no |
| Fused bias | none / ROW broadcast / COL broadcast |
| Fused SFPU activation | none / relu / gelu / sigmoid / silu / other (specify) |
| Fused eltwise multiply | yes/no |
| PACK_RELU | yes/no |
| PACKER_L1_ACC | yes/no |
| FP32_DEST_ACC_EN | yes/no |
| Mixed precision / reconfig | yes/no (describe) |
| B matrix transpose | yes/no |
| Out-of-order packing | yes/no |
| CB IDs used | list (e.g., in0=0, in1=1, out=16, interm=24, bias=3) |
| `#ifdef` count | number of conditional compilation branches |
| LOC | line count |
| Already migrated? | yes (which helper) / no / partial |

4. After the table, write a "Feature Frequency" section: for each feature, count how many kernels use it. Rank features by frequency.

**Rules**:
- Include ALL matmul compute kernels you can find, not just the ones listed.
- If a kernel has multiple `#ifdef` paths, document the UNION of all features used across all paths.
- Do NOT propose any design or API changes. Pure documentation of facts.

---

### INSTANCE 2 — Production Kernel Control Flow

**Output file**: `docs/analysis_production_kernel.md`

**Task**: Create a complete control-flow map of the production kernel `bmm_large_block_zm_fused_bias_activation.cpp`.

**Steps**:
1. Read `docs/llk_matmul_orchestration.md` (this file) for context.
2. Read the full 827-line production kernel at:
   `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp`
3. Also read the gathered variant:
   `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp`
4. Document the following:

**Section A — Define combinations**: List every valid combination of `#ifdef` defines and what code path it selects. Identify which combinations are actually used in production (check how the host code sets these defines — look in `ttnn/cpp/ttnn/operations/matmul/device/matmul_op_multi_core_reuse_mcast_program_factory.cpp` or similar factory files).

**Section B — Code path tree**: Draw a decision tree showing how the code branches. At each branch point, note:
- The condition (`#ifdef PACKER_L1_ACC`, etc.)
- The LLK calls made in that branch
- The CB IDs accessed
- How dest registers are managed (acquire/release/commit)

**Section C — LLK call sequences**: For each distinct code path, list the exact sequence of LLK API calls (e.g., `mm_block_init` -> `cb_wait_front(in0)` -> `matmul_block` -> `pack_tile` -> ...). Use pseudocode, not full C++.

**Section D — What the current helper covers**: Identify exactly which lines/paths of the production kernel are currently handled by the `matmul_block` helper (the `#ifndef PACKER_L1_ACC` section on this branch). Also check what the failed `matmul_block_fused_bias` helper on llk3 attempted to cover (`git show origin/wransom/llk3:ttnn/cpp/ttnn/kernel_lib/matmul_block_fused_bias_helpers.inl`). Then identify what each uncovered path does differently.

**Section E — Gathered variant delta**: Document what `_gathered.cpp` adds on top of the base kernel.

**Rules**:
- Be exhaustive. Every `#ifdef` path must be documented.
- Do NOT propose any design or API changes.

---

### INSTANCE 3 — Existing Helper & Composition Pattern Analysis

**Output file**: `docs/analysis_composition_patterns.md`

**Task**: Study how existing helpers and modern kernels handle composability, to inform the redesign.

**Steps**:
1. Read `docs/llk_matmul_orchestration.md` (this file) for context.

**Part A — Existing kernel_lib helpers**:
2. Read every `.hpp` and `.inl` file in `ttnn/cpp/ttnn/kernel_lib/` (the full implementations, not just headers). For each helper pair, document:
   - Public API: function name, template params, runtime params
   - How it handles feature variants (what's a template param vs runtime param vs separate function?)
   - Init/uninit lifecycle
   - CB synchronization strategy
   - How it composes with other helpers (e.g., does matmul_block_fused_bias call matmul_block internally, or duplicate the logic?)
   - What the PostComputeFn / callback pattern enables and what it can't express

3. Also read the prior attempt's fused bias helper from llk3 (`git show origin/wransom/llk3:ttnn/cpp/ttnn/kernel_lib/matmul_block_fused_bias_helpers.hpp` and `.inl`). Analyze: does it duplicate code from `matmul_block_helpers.inl`, or does it call/reuse it? This is critical for understanding whether composition works today — and why this approach caused device hangs.

**Part B — DeepSeek V3 unified kernels**:
4. Read the 6 DeepSeek V3 unified kernel files listed in KEY FILE PATHS. Document:
   - How they handle multiple matmul variants in a single templated framework
   - How fused operations (SiLU, eltwise multiply, compression) are plugged in
   - The NCRISC/BRISC/TRISC thread model and how it differs from the helper approach
   - What composition patterns they use that the kernel_lib helpers don't

**Part C — Cross-cutting patterns**:
5. Identify patterns that appear in 3+ helpers or kernels:
   - Dest register acquire/release/commit sequences
   - CB wait/pop patterns
   - Pack loops (sequential vs out-of-order)
   - Data format reconfiguration sequences
   - Spill/reload sequences

**Rules**:
- Read actual implementation files (.inl), not just headers.
- Document concrete code patterns with line references, not abstract principles.
- Do NOT propose any design or API changes.

---

### INSTANCE 4 — Migration Gap Classifier

**Output file**: `docs/analysis_migration_gaps.md`

**Task**: For each "not migratable" kernel, identify the exact features blocking migration and rank them by impact.

**Steps**:
1. Read `docs/llk_matmul_orchestration.md` (this file) for context.
2. Read the current matmul helper (`matmul_block_helpers.hpp` + `.inl` in `ttnn/cpp/ttnn/kernel_lib/`). Also read the prior attempts from llk3 for additional context: `git show origin/wransom/llk3:ttnn/cpp/ttnn/kernel_lib/matmul_tile_helpers.hpp` (removed per PR feedback) and `git show origin/wransom/llk3:ttnn/cpp/ttnn/kernel_lib/matmul_block_fused_bias_helpers.hpp` (caused device hangs).
3. For each kernel in the "NOT migratable" table above, read the full source and document:

**Per-kernel analysis**:
- **File**: full path
- **Total LOC**: line count
- **Standard matmul_block LOC**: lines that follow the standard sub-blocked matmul pattern (init, K-loop, sub-block loops, pack)
- **Custom LOC**: lines that are NOT standard matmul_block (fusions, special CB handling, pipeline stages, etc.)
- **Missing features**: the minimal set of features the helpers lack to cover this kernel. Be specific (e.g., "PACKER_L1_ACC support in pack loop" not just "L1_ACC").
- **Feature interactions**: do the missing features interact with each other? (e.g., does L1_ACC change how bias is added?)
- **Composability assessment**: could this kernel be expressed as a composition of existing helpers + a new feature, or does it need a fundamentally different approach?

**Summary table**: rank missing features by migration coverage:

| Missing feature | Kernels it would unlock | Estimated complexity |
|----------------|------------------------|---------------------|
| ... | ... | low/medium/high |

**Rules**:
- Read the full source of each kernel. Do not guess from file names.
- "Missing feature" means specifically what the helper API lacks, not what the kernel does.
- Do NOT propose any design or API changes.

---

## PHASE 2: DESIGN SYNTHESIS

Instance numbering resets each phase. Phase 2 has 1 instance.
Run AFTER all phase 1 outputs are written and reviewed.

### INSTANCE 1 — API Architect

**Output file**: `docs/matmul_api_design.md`

**Task**: Synthesize all phase 1 analysis into a concrete API design.

**Steps**:
1. Read `docs/llk_matmul_orchestration.md` (this file) for context.
2. Read ALL phase 1 outputs:
   - `docs/analysis_feature_matrix.md`
   - `docs/analysis_production_kernel.md`
   - `docs/analysis_composition_patterns.md`
   - `docs/analysis_migration_gaps.md`
3. Read the current matmul helper on this branch (`matmul_block_helpers.hpp/inl`). Also read prior attempts from llk3 via `git show origin/wransom/llk3:<path>` for the removed matmul_tile and failed fused_bias helpers.
4. Read the non-matmul helpers (tilize, untilize, reduce, binary_op — both `.hpp` and `.inl`) to understand established patterns you must follow.

**Deliverables** (all in the output file):

**Section A — Design principles**: 3-5 concrete design principles derived from the analysis. Not abstract platitudes — specific to this codebase. E.g., "Feature X should be a template param because [analysis shows it affects compile-time code generation]" or "Feature Y should be a runtime param because [analysis shows it varies per call site]".

**Section B — Type system**: The proposed structs, enums, and policy types. Include full C++ declarations (not just names). Explain what changed from the current types and why.

**Section C — Function signatures**: For each helper function, provide the full template signature and a brief description. Show how the functions compose (which ones call which).

**Section D — Coverage matrix**:

| Kernel | Helper expression | Notes |
|--------|------------------|-------|
| bmm.cpp | (inline — no helper, only 1 call site) | per PR feedback |
| bmm_large_block_zm.cpp | `matmul_block<...>(...)` | currently migrated |
| bmm_fused_bias_activation.cpp (all paths) | `???` | PRIMARY TARGET |
| ... | ... | ... |

Every kernel from the feature matrix should appear. Mark kernels that remain uncovered and explain why.

**Section E — Migration sequence**: Ordered list of kernels to migrate, with rationale:
1. First: [kernel] — because [reason]
2. Second: ...

Prioritize: production kernels > test kernels > experimental kernels.

**Section F — Code sketches**: For the 3 hardest migration targets, show what the kernel would look like after migration. Not full implementations — enough to validate the design works. Include the `#ifdef` handling approach.

**Section G — Backward compatibility**: Explain exactly how `bmm_large_block_zm.cpp` (the one currently-migrated kernel) will continue to work. If the `matmul_block` function signature changes, show the migration path.

**Section H — Open questions**: List any design decisions you couldn't resolve and what information would help.

**Rules**:
- The design must be grounded in the phase 1 analysis. Reference specific findings.
- Follow the established kernel_lib patterns (namespace, file structure, naming conventions).
- Prefer fewer, more composable helpers over many specialized ones.
- The PostComputeFn pattern is proven — extend it, don't replace it.
- Do not write implementation code (.inl files). Design only.

---

## PHASE 3: PARALLEL IMPLEMENTATION

Instance numbering resets each phase. Phase 3 has up to 3 instances.
The phase 2 design (`docs/matmul_api_design.md`) has been reviewed and approved with these decisions:
- **Q6 (mm_block_init ownership)**: APPROVED — remove `mm_block_init` from the helper. Caller calls it once at kernel start. Update `bmm_large_block_zm.cpp` to add the init call.
- **Q1 (in0_transpose)**: Keep inline for now. Will revisit during phase 3 if needed.

### INSTANCE 1 — Core Helper Implementation

**Output**: Modified/new files in `ttnn/cpp/ttnn/kernel_lib/`

**Task**: Implement the two helpers from the phase 2 design.

**Steps**:
1. Read `docs/llk_matmul_orchestration.md` (this file) for context — especially LESSONS FROM PRIOR ATTEMPTS and CONSTRAINTS.
2. Read `docs/matmul_api_design.md` — this is your primary specification. Pay close attention to Sections B (type system), C (function signatures), the Appendix (pseudocode), and Section H (open questions with approved decisions above).
3. Implement:
   - **Modify** `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp/inl` — add the new template params (`packer_l1_acc`, `pack_last_to_interm`, `pack_relu`), switch to 4-phase DST, remove `mm_block_init` from the helper body. All new template params have defaults so existing call sites are unchanged.
   - **Create** `ttnn/cpp/ttnn/kernel_lib/bias_add_helpers.hpp/inl` — the `add_bias_bcast_rows` helper per the design.
   - **Update** `bmm_large_block_zm.cpp` (both production and programming_examples) to add `mm_block_init` call before the helper (since the helper no longer calls it internally).
4. Build with `./build_metal.sh` and fix any compile errors.
5. Run existing matmul C++ integration tests to verify backward compatibility:
   ```
   ./build/test/tt_metal/integration/matmul/test_matmul_block_helper
   ```
   Report pass/fail counts and architecture (run `tt-smi`).
6. Run existing matmul Python tests:
   ```
   pytest tests/ttnn/unit_tests/operations/test_matmul.py -x --no-header -q
   ```
   Report pass/fail counts.

**Rules**:
- Follow the pseudocode in the design doc Appendix closely. It was derived from the production kernel's actual code paths.
- Use `pack_tile_block` (not a per-tile loop) for packing, per Open Question 4.
- Do NOT use environment variables for any feature selection.
- Do NOT create param structs — flat independent params per constraint 9.
- Write output summary to `docs/phase3_instance1_results.md`.

---

### INSTANCE 2 — Isolated Helper Tests

**Output**: New test files in `tests/tt_metal/tt_metal/integration/matmul/`

**Task**: Write C++ integration tests that exercise the new helpers in isolation.

**Steps**:
1. Read `docs/llk_matmul_orchestration.md` (this file) for context.
2. Read `docs/matmul_api_design.md` — Sections C (function signatures) and D (coverage matrix) to understand what feature combinations exist.
3. Read `docs/analysis_feature_matrix.md` — to understand which features are most common across kernels.
4. Read the existing test file for patterns: `tests/tt_metal/tt_metal/integration/matmul/test_matmul_block_helper.cpp`
5. Write new test cases that cover these feature combinations (prioritized by coverage):

   **matmul_block tests:**
   - Basic (no L1_ACC, no pack_last_to_interm) — should match existing test behavior
   - `packer_l1_acc=true`, `pack_last_to_interm=false` — L1 accumulation, pack to output
   - `packer_l1_acc=true`, `pack_last_to_interm=true` — L1 accumulation, pack to interm (for bias pipeline)
   - `packer_l1_acc=false`, `pack_last_to_interm=true` — software spill, pack to interm
   - `pack_relu=true` — RELU on last K-block
   - `PostComputeFn` with a simple SFPU activation
   - `num_k_blocks=1` (no spill/reload) and `num_k_blocks>1` (with spill/reload) for each L1_ACC mode
   - `transpose=true` variants

   **add_bias_bcast_rows tests:**
   - Basic bias addition (no PostBiasFn)
   - Bias addition with PostBiasFn (SFPU activation)
   - Varying subblock dimensions

   **Composition tests (matmul_block → add_bias_bcast_rows):**
   - `matmul_block<..., pack_last_to_interm=true>` followed by `add_bias_bcast_rows`
   - Same with `packer_l1_acc=true`

6. Each test should:
   - Set up input tensors with known values (or random with seed)
   - Run the helper kernel
   - Compare output against a CPU/PyTorch reference
   - Assert PCC >= 0.999
7. Build and run tests. Report pass/fail counts and architecture (run `tt-smi`).

**Rules**:
- Tests must exercise helpers directly in test kernels, NOT through TTNN ops.
- Follow the existing test file's patterns for device setup, CB configuration, and kernel launch.
- Write output summary to `docs/phase3_instance2_results.md`.

---

### INSTANCE 3 — Migration + Regression Tests

**Output**: Migrated kernel files + test results

**Task**: Migrate the production kernel to use the new helpers, following the phase 2 migration sequence.

**Prerequisites**: Instances 1 and 2 must be complete. Their changes must be on the branch before this instance starts. The user will coordinate this.

**Steps**:
1. Read `docs/llk_matmul_orchestration.md` (this file) for context.
2. Read `docs/matmul_api_design.md` — Sections E (migration sequence) and F (code sketches). The code sketch in F1 is your primary reference for the production kernel migration.
3. Read the instance 1 and 2 results: `docs/phase3_instance1_results.md` and `docs/phase3_instance2_results.md` to understand the current state of helpers and tests.
4. Read the current production kernel:
   `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp`
5. Migrate following the design doc's migration sequence:

   **Step 1 — Verify backward compat** (quick):
   Run the existing matmul tests to establish a baseline. Record pass/fail counts.

   **Step 2 — Migrate production kernel** (primary target):
   Rewrite `bmm_large_block_zm_fused_bias_activation.cpp` to use `matmul_block` and `add_bias_bcast_rows` helpers for all paths EXCEPT `in0_transpose` (stays inline) and `SKIP_COMPUTE` (caller-managed). Follow the code sketch in design doc Section F1.

   **Step 3 — Run full regression**:
   ```
   pytest tests/ttnn/unit_tests/operations/test_matmul.py -x --no-header -q
   ./build/test/tt_metal/integration/matmul/test_matmul_block_helper
   ```
   Compare pass/fail counts against Step 1 baseline. Any regressions must be investigated and fixed before proceeding.

   **Step 4 (if time permits) — Migrate test kernel**:
   Migrate `tests/tt_metal/tt_metal/test_kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp` per design doc Section E step 3.

6. Report all results with architecture (run `tt-smi`).

**Rules**:
- The production kernel migration is the critical deliverable. Do NOT skip the regression tests.
- If a regression is found, fix it before migrating more kernels.
- Keep `in0_transpose` paths as inline code per the approved decision.
- Do NOT use environment variables.
- Write output summary to `docs/phase3_instance3_results.md`.

---

## REMINDERS FOR ALL INSTANCES

- **Instance numbering resets each phase.** Phase 1: instances 1-4. Phase 2: instance 1. Phase 3: instances 1-3.
- **Existing matmul helper is expendable.** The `matmul_block` helper on this branch is a prior attempt that didn't scale. Use it as reference to understand what was tried, but the redesign may replace it entirely. The non-matmul helpers (tilize, untilize, reduce, binary_op) are stable and their patterns should be followed.
- **Read the LESSONS FROM PRIOR ATTEMPTS section.** The prior branches tried matmul_tile (removed — only 1 call site), param structs (removed — pass flat params instead), unused enums (removed — no call site used them), programming example migrations (reverted — JIT include paths), and a fused_bias helper (device hangs). Don't repeat these mistakes.
- **Submodules and building — DO THIS FIRST, BEFORE ANY OTHER WORK.** After ANY branch checkout, pull, or reset, you MUST immediately run `git submodule update --init --recursive` before doing anything else. Then build with `./build_metal.sh`. Do not skip this step. Do not defer it. Stale submodules cause cryptic build/link failures that waste significant time. This has been a recurring problem — treat it as step 0.
- **Wormhole vs Blackhole systems.** Instances may be running on either Wormhole or Blackhole hardware. These have different core grid sizes and slightly different LLK implementations, so test results (pass/fail counts, PCC values) may differ between machines. Run `tt-smi` to identify which system you are on. **Always note the architecture in your output** (e.g., "Tests run on Wormhole" or "Tests run on Blackhole") so results from different instances can be compared correctly. Do not assume a test failure is a bug if you only see it on one architecture — note it as architecture-specific and move on.
- Write your output to the specified file using the Write tool. Keep it structured and dense.
- Read actual source files. Do not guess contents from names.
- Use `git show origin/wransom/llk3:<path>` to read files from the prior attempt branch (matmul_tile, fused_bias helpers).
- The production kernel path is: `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp`
- The kernel_lib path is: `ttnn/cpp/ttnn/kernel_lib/`
- When done, state: **"Completed phase Y for instance X."**
