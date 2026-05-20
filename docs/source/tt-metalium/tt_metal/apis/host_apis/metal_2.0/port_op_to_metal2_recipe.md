# Porting an Op to Metal 2.0 — Phases A–D: Port Recipe

> This is the second of two documents covering the Metal 2.0 op port workflow. **This document covers Phases A–D (legacy inventory, spec planning, mechanical construction, verification).** Phase 0 — the feasibility audit — lives in [`port_op_to_metal2_audit.md`](port_op_to_metal2_audit.md) and is a hard prerequisite to anything in this document.

## Read this first

**Audience**: AI agents asked to perform the actual Metal 2.0 port of a TTNN op, *after* the Phase 0 feasibility audit has cleared with GREEN status and the user has explicitly approved proceeding.

**Precondition — non-negotiable**: You may only invoke this document if:

1. The audit in [`port_op_to_metal2_audit.md`](port_op_to_metal2_audit.md) was performed for this op and produced an **overall GREEN** result (or a YELLOW with all questions resolved by the user in favor of proceeding).
2. The user has **explicitly asked you to proceed** with the port. A green audit alone is not sufficient — the user must have read the audit and given an unambiguous go-ahead.

If either condition is unmet, stop. Return to the audit document. Do not improvise.

**Operating principle**: Refusing to write code is still a successful outcome.

The audit cleared the *features* and *prereqs* known at audit time. During the port you may discover something the audit missed — a feature gate that didn't fire, a kernel pattern that doesn't translate cleanly, an interaction the audit didn't anticipate. When that happens, the correct response is the same as in Phase 0: **stop and report**. Do not improvise around it.

In particular: if you find yourself constructing a clever workaround during the port — packing data into varargs to simulate a missing field, threading a buffer address through an RTA because the binding mechanism doesn't fit, hand-rolling a synchronization primitive — **stop**. Whatever you are about to write is almost certainly wrong. Surface the problem; do not paper over it.

**The port plan as central artifact.** This recipe centers on a port plan (`PORT_PLAN.md`) that you write to the op's directory and commit alongside the port. The plan externalizes structural decisions before mechanical translation begins. It is read by you (during Phases C and D) and by human reviewers (during PR review and later debugging). See [Appendix A](#appendix-a--port_planmd-template) for the template.

**Workflow at a glance**:

1. [**Phase A**](#phase-a--legacy-inventory) — Legacy inventory. Consume the audit; record the legacy structure to `PORT_PLAN.md`.
2. [**Phase B**](#phase-b--plan-the-spec) — Plan the spec. Apply host-side specialization principles; identify legacy plumbing that should evaporate. Externalize all structural decisions to the plan.
3. [**Phase C**](#phase-c--construct-paired-spec--run-params) — Construct the spec and run-params, paired by resource. Mechanical translation per the plan.
4. [**Phase D**](#phase-d--verification) — Verify. Build, run tests, run anti-pattern self-audit against the [patterns catalog](metal2_port_patterns.md).

**Reference material** the recipe relies on, loaded on demand:

- [Migration guide — Design Principles](metal2_migration_guide.md#design-principles): why certain legacy plumbing should evaporate during port.
- [Migration guide — TTNN Framework Integration](metal2_migration_guide.md#ttnn-framework-integration): cache lifecycle and `ProgramSpecFactoryConcept` shape.
- [Patterns catalog](metal2_port_patterns.md): recognition signals + decisions for structural patterns and anti-patterns.

---

## Phase A — Legacy inventory

*This is an observation phase. No decisions yet.*

**Inputs**:
- The audit report for this op (produced in Phase 0). The audit's "kernels referenced" and "factory shape" sections are the starting point for the inventory.
- The op's program-factory `.cpp` / `.hpp` files.
- The kernel sources referenced by the factories.

**Output**: write the **Legacy Inventory** section of `PORT_PLAN.md` to the op's directory. Record:

- **Factory shape**: which `ttnn::device_operation` concept the factory currently satisfies (`ProgramFactoryConcept` / `ProgramDescriptorFactoryConcept`). For each variant (if the device-operation is multi-variant), record separately.
- **Kernels**: every `KernelDescriptor` (one row per descriptor):
  - `kernel_source` (file path; flag any path outside the op's own directory — cross-op kernels are a Caution case).
  - `core_ranges` (verbatim).
  - `compile_time_args` (positional values).
  - `named_compile_time_args` (name → value pairs).
  - `runtime_args` and `common_runtime_args` (names if known, dimensions, and shapes).
  - `defines` (key → value pairs).
  - `config` (the descriptor type: `ReaderConfigDescriptor` / `WriterConfigDescriptor` / `ComputeConfigDescriptor` and its content).
- **CBs**: every `CBDescriptor` (one row per descriptor):
  - `total_size`, `core_ranges`, format descriptors (`buffer_index`, `data_format`, `page_size`).
- **Semaphores**: every `SemaphoreDescriptor`:
  - `id`, `core_type`, `core_ranges`, `initial_value`.
- **Tensor accessors**: every `TensorAccessor` use site (host and device):
  - Originating `Tensor` (input / output / which input/output).
  - Where its address surfaces in the host RTA list.
- **Work split**: which `split_work_to_cores` call (or similar) drives the per-core counts. Record `(num_cores, all_cores, core_group_1, core_group_2, count_per_group_1, count_per_group_2)`.
- **Cross-op kernels**: explicitly list any kernel source path outside the op's directory; flag for [Caution: Modifying a shared dataflow kernel](metal2_port_patterns.md#caution-modifying-a-shared-dataflow-kernel).
- **Factory variants**: if the device-operation has multiple variants (e.g., reduce W vs H vs HW; Welford W/H/HW), record each variant's complete inventory.

**Stop signals**:
- An unreferenced kernel file in the op's directory: not a stop, but note in the inventory's "Flags" subsection (so the report makes clear what was *not* audited).
- A descriptor type not in the audit's scan: stop and report. (Most commonly: a `BufferDescriptor` for a borrowed-memory CB, which the audit should have flagged as RED but may have missed.)

---

## Phase B — Plan the spec

*This is the load-bearing planning phase. Externalize all structural decisions before writing code.*

**Inputs**:
- The Legacy Inventory from Phase A.
- The [migration guide — Design Principles](metal2_migration_guide.md#design-principles), especially Principle 2 (named bindings) which drives the "Dropped Plumbing" section below.
- The [migration guide — TTNN Framework Integration](metal2_migration_guide.md#ttnn-framework-integration) for the factory concept the ported op will satisfy.
- The [patterns catalog](metal2_port_patterns.md).
- Any yellow-tier items from the audit (apply per the catalog's override guidance for each).

**Output**: extend `PORT_PLAN.md` with the following sections (see [Appendix A](#appendix-a--port_planmd-template) for the template). Each section may say "none" with a one-line justification when no items apply.

### Planned Spec Shape

The Metal 2.0 spec shape. Default: 1:1 with legacy.

- **KernelSpecs**: one per legacy `KernelDescriptor` (default). When the legacy factory has multiple `KernelDescriptor`s of the same source for work-split, **preserve the multiplicity**: one `KernelSpec` per legacy `KernelDescriptor`, with the per-group CTAs reproduced. See [Anti-pattern: Demoting per-group CTA to RTA](metal2_port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta) for why this is non-negotiable.
- **DataflowBufferSpecs**: one per legacy `CBDescriptor`. Note any borrowed-memory cases (which should have failed the audit; their presence here is a regression).
- **SemaphoreSpecs**: one per legacy `SemaphoreDescriptor`.
- **TensorParameters**: one per distinct legacy `TensorAccessor` originating tensor. Note that multiple kernel-side accesses to the same tensor collapse to one `TensorParameter` with multiple `TensorBinding`s.
- **WorkUnitSpecs**: one per distinct (set of kernels, target nodes) pairing. Most ops have one or two.

### Preserved Multiplicity

For each work-split case, explicitly list the multi-`KernelSpec` mapping:

```
Legacy KernelDescriptors [list] of source <path>
  → KernelSpecs [list] of same source
  → in WorkUnitSpecs [list]
  → sharing upstream/downstream DFBs as multi-bindings: [list]
```

If the legacy code has no multi-`KernelDescriptor` work split, this section reads "none — no work-split multiplicity in legacy."

### Dropped Plumbing

For each legacy RTA or CTA that should *not* survive the port, list it with the replacement primitive:

- **Buffer-address RTAs** (`tensor.buffer()` / `tensor.buffer()->address()` in legacy RTA values; `get_arg_val<uint32_t>(N)` in kernel passed to `TensorAccessor`): replaced by `TensorBinding`. List each kernel's affected RTA slot.
- **Magic CB indices in CTAs**: replaced by `DFBBinding`. List each kernel's affected CTA slot.
- **`TensorAccessorArgs` plumbing**: replaced by the binding mechanism end-to-end. List each `TensorAccessorArgs(buffer).append_to(cta)` site and its kernel-side `TensorAccessorArgs<N>()` / `next_compile_time_args_offset()` chain.
- **Semaphore-ID RTAs**: replaced by `SemaphoreBinding`. List each kernel's affected RTA slot.
- **Positional CTAs**: replaced by named CTAs. List each kernel's positional CTA list with the names you'll assign.

This section's enumeration is the gate against builder-pattern carry-over. If a legacy RTA / CTA is not listed here, it will be translated by reflex in Phase C — which is exactly the failure mode this gate exists to prevent. See [migration guide — Principle 2](metal2_migration_guide.md#principle-2-first-class-named-resource-bindings) for the rationale.

### Applied Patterns

For each non-trivial pattern from the [catalog](metal2_port_patterns.md) invoked by this port, name the pattern and the context:

- "[Self-loop DFB binding](metal2_port_patterns.md#pattern-self-loop-dfb-binding): ACC_DFB on compute KernelSpec (both PRODUCER and CONSUMER)."
- "[Conditional optional binding](metal2_port_patterns.md#pattern-conditional--optional-dfb-bindings): SCALED_DFB on compute KernelSpec, gated by `do_scale` CTA."
- "[Multi-variant factory](metal2_port_patterns.md#pattern-multi-variant-factories): `reduce_dim` variant selection inside `create_program_spec`."

### Deferred / Flagged

Any items the audit flagged as YELLOW that affect this port, plus any new findings Phase B uncovered:

- Yellow audit items, with the relevant [catalog](metal2_port_patterns.md) override guidance entry referenced.
- New findings: anything the audit missed that surfaced during Phase B's structural planning.

**Stop signal**: if Phase B uncovers a structural issue the audit didn't catch — e.g., a kernel that genuinely cannot be expressed without one of the legacy workarounds, or a feature gate that the audit's Appendix A doesn't cover — **stop and report**. Don't paper it over by demoting CTAs, packing varargs, or hand-rolling primitives. The audit's gate set improves with what later phases discover.

---

## Phase C — Construct paired spec + run-params

*Mechanical translation from the plan. Build each resource's spec entry and its run-params entry together.*

**Operating principle**: prefer designated initializers. Metal 2.0 was designed to support them and the spec reads as data, not as procedure.

For each resource type, construct the spec entry and its run-params entry as a pair. The order within Phase C emerges naturally from the op's existing structure (reader / writer / compute order, tensor → DFB → semaphore precedence); the recipe does not prescribe a fixed sequence.

- **`KernelSpec` ↔ `KernelRunParams`.** For each planned `KernelSpec`, build the schema (`compile_time_arg_bindings`, `runtime_arguments_schema`, `dfb_bindings`, `tensor_bindings`, `semaphore_bindings`, `config_spec`); alongside, build the corresponding `KernelRunParams` entry (per-node `named_runtime_args` and `named_common_runtime_args`). If the kernel has no RTAs, the run-params entry may be omitted entirely.
- **`DataflowBufferSpec`.** Build with `entry_size`, `num_entries`, `data_format_metadata`. No placement field — placement is derived from the kernel bindings. Borrowed-memory DFBs (not yet supported as of this guide's date) would have failed the audit; their presence here indicates the audit missed something.
- **`SemaphoreSpec`.** Build with `target_nodes`. (Semaphores have no per-execution counterpart on `ProgramRunParams`.)
- **`TensorParameter` ↔ `TensorArg`.** Declare each tensor as a `TensorParameter` (using `<tensor>.tensor_spec()`); alongside, add the corresponding `TensorArg` to `ProgramRunParams::tensor_args`.
- **`WorkUnitSpec`.** Build with `kernels` (by `unique_id`) and `target_nodes`. No per-execution counterpart.

After all resources are built, assemble the `ProgramSpec` (collecting `kernels`, `dataflow_buffers`, `semaphores`, `tensor_parameters`, `work_units`) and the `ProgramRunParams` (collecting `kernel_run_params`, `tensor_args`). Return `ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_params)}`.

**Stop signals**: any urge to —

- Demote a CTA to a runtime arg to make a single `KernelSpec` work where the legacy had multiple. ([Anti-pattern](metal2_port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta).)
- Add an `#ifdef` to the kernel source to gate optional binding usage. ([Pattern: Conditional / optional DFB bindings](metal2_port_patterns.md#pattern-conditional--optional-dfb-bindings).)
- Bind an optional DFB unconditionally on the host and gate the *uses* with `if constexpr`. ([Anti-pattern](metal2_port_patterns.md#anti-pattern-always-bind-optional-dfb--gate-uses-only).)
- Extract `.id` from a `dfb::name`, or construct a temporary `DataflowBuffer` to retrieve its underlying id. ([Anti-pattern](metal2_port_patterns.md#anti-pattern-id-extraction-or-temp-dfb-wrappers-at-llk-call-sites).)
- Pack data into varargs that should be named arguments. ([Caution](metal2_port_patterns.md#caution-avoid-varargs-unless-absolutely-necessary).)
- Thread a buffer address through an RTA because the binding mechanism doesn't fit.
- Hand-roll a synchronization primitive.

If any of these appear in your draft, **stop and report**. The likely cause is a structural decision in Phase B that should be revisited.

---

## Phase D — Verification

*Build, test, anti-pattern self-audit.*

### Build

Build the ported op's TTNN target:

```bash
cmake --build build_Release --target ttnncpp -j 8
```

Common build failures and their likely causes:

- `AllFactoriesValid` `static_assert` fires → a factory satisfies two concepts (likely a stale `cached_program_t` declaration alongside the new `create_program_spec`). Audit for missed deletions in the header.
- Unresolved symbol for `override_runtime_arguments` → some code path still calls it. Should only happen for the framework adapter, which doesn't for `ProgramSpecFactoryConcept` factories. Re-audit.
- Error referencing `metal2_artifacts.hpp` (or other framework header) not found → the framework dependency is not on this branch. Stop and report; the framework PR was a precondition for the audit (which should have failed pre-port).
- `kernel_args_generated.h` mentions a name that doesn't exist → host added a named CTA / RTA without the kernel referencing it (or vice versa). Reconcile.

### Run tests

Find and run the op's correctness tests. The audit report names them; if not, the tests typically live under `tests/ttnn/unit_tests/operations/<op_family>/`. Run with `pytest`:

```bash
pytest tests/ttnn/unit_tests/operations/<op_family>/ -x -v
```

All tests passing pre-conversion should continue to pass post-conversion. If a previously-passing test now fails, **stop and report** — likely cause is a structural error in the spec that compiled but failed at `MakeProgramFromSpec` validation, or an incorrect tensor-arg / runtime-arg layout.

If compilation passes but the test fails with a `TT_FATAL` from `program_spec.cpp` or `program_run_params.cpp`, the [patterns catalog](metal2_port_patterns.md) has entries for the most common failure modes (DFB binding multiplicity mismatches, missing kernel run-params entries, etc.). Cross-reference the error message against the catalog.

### Anti-pattern self-audit

Scan the ported code against this checklist. Each item is a Metal 2.0 design-intent failure to look for; the [patterns catalog](metal2_port_patterns.md) has the full discussion of each.

- [ ] **No `tensor.buffer()->address()` survived.** Search the factory `.cpp` for this string; if present, the corresponding tensor needs a `TensorBinding` instead.
- [ ] **No magic-number CB indices in CTAs.** Search `compile_time_arg_bindings` for values that are CB indices (typically small integers or `CBIndex::c_*`); if found, the value should come from a `DFBBinding` instead.
- [ ] **No `TensorAccessorArgs<N>()` survived in any ported kernel.** Search for this; if present, the kernel needs `TensorAccessor(ta::name)` instead.
- [ ] **No `#ifdef` newly added to gate optional bindings in kernel.** Search the ported kernels for `#ifdef` blocks gating DFB wrapper declarations; if present, convert to `if constexpr` on a named CTA. (Pre-existing `#ifdef`s that pre-date the port are out of scope; this audit catches *newly introduced* ones.)
- [ ] **No `.id` extraction at LLK call sites.** Search for `.id` on `dfb::` handles; if present, pass `dfb::name` directly.
- [ ] **No CTA→RTA demotion in compute kernels.** If a per-group dimension was moved from CTA to RTA in the port, the structural decision is wrong; revisit Phase B.
- [ ] **All CTAs are named.** Search the factory for positional `compile_time_args = {...}`; should be `compile_time_arg_bindings = {{name, value}, ...}` only.
- [ ] **No new varargs unless the kernel reads them in a loop.** Check `num_runtime_varargs` use; if the kernel reads `get_vararg(0)`, `get_vararg(1)`, ..., the named form is the right answer.

If any checklist item fails, return to Phase B / C to fix. Do not paper over with kernel-side modifications.

### Commit

The `PORT_PLAN.md` file is committed alongside the port. It serves as the documented evidence of the port's structural decisions for human review and later debugging.

---

## Appendix A — `PORT_PLAN.md` template

Copy this template to the op's directory at the start of the port:

````markdown
# Port Plan — <op name>

Port plan for `<op>`, ported from `<legacy api>` to Metal 2.0.
Written during Phases A–B; committed alongside the port for review.

## Legacy Inventory

*Filled in during Phase A.*

### Factory shape
- Concept: <ProgramFactoryConcept | ProgramDescriptorFactoryConcept>
- Variants: <list, or "single">

### Kernels
| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | config |
|---|---|---|---|---|---|---|---|---|
| reader | ... | ... | ... | ... | ... | ... | ... | ... |
| writer | ... | ... | ... | ... | ... | ... | ... | ... |
| compute | ... | ... | ... | ... | ... | ... | ... | ... |

### CBs
| index | total_size | core_ranges | data_format | page_size |
|---|---|---|---|---|
| 0 | ... | ... | ... | ... |

### Semaphores
| id | core_type | core_ranges | initial_value |
|---|---|---|---|
| 0 | ... | ... | ... |

(or "none")

### Tensor accessors
| host site (file:line) | originating Tensor | RTA slot (host) | CTA offset (kernel) |
|---|---|---|---|
| ... |

### Work split
- Driver: `split_work_to_cores(<args>)`
- num_cores: ...
- core_group_1: ..., count_per_core: ...
- core_group_2: ..., count_per_core: ...

(or "n/a — single core")

### Cross-op kernels
List any kernel `source` path outside the op's directory. Each one is a Caution case (see [catalog](metal2_port_patterns.md#caution-modifying-a-shared-dataflow-kernel)).

(or "none")

### Flags
Anything Phase A noticed but didn't classify — unreferenced kernel files, unusual descriptors, etc.

(or "none")

## Planned Spec Shape

*Filled in during Phase B.*

- KernelSpecs: ...
- DataflowBufferSpecs: ...
- SemaphoreSpecs: ...
- TensorParameters: ...
- WorkUnitSpecs: ...

## Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (multi-binding) |
|---|---|---|---|
| ... |

(or "none — no work-split multiplicity in legacy")

## Dropped Plumbing

For each legacy RTA / CTA that disappears in the port:

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader.cpp RTA slot 0 | `tensor.buffer()->address()` | `TensorBinding(input)` |
| reader.cpp CTA slot 0 | `cb_idx = 0` | `DFBBinding(input, "in0", CONSUMER)` |
| ... |

## Applied Patterns

List any patterns from the catalog invoked by this port:

- ... ([catalog entry link])
- ...

(or "none")

## Deferred / Flagged

- Yellow items from audit: ... ([catalog override guidance link])
- New findings during Phase B: ...

(or "none")
````

---

## Appendix B — Cross-references

- [Phase 0 feasibility audit](port_op_to_metal2_audit.md)
- [Migration guide (concept map, design principles, TTNN integration)](metal2_migration_guide.md)
- [Patterns catalog](metal2_port_patterns.md)
