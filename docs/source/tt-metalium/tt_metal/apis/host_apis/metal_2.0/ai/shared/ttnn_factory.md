# Porting an Op to Metal 2.0 — TTNN Integration

> The TTNN device-operation glue a Metal 2.0 port needs: which factory concept the op lands on, the factory entry point that returns the spec, and the two device-op-class edits the port forces (custom-hash deletion, pybind cleanup). Lives in its own document because the TTNN factory layer churns on a different cadence than the Metal 2.0 host API — the [port recipe](../port/metal2.md) covers building the `ProgramSpec` + `ProgramRunArgs` (stable); this doc covers wiring that into TTNN's framework (in flux).

## Read this first

**Primary audience**: AI agents performing the [audit](../audit/metal2.md) on a TTNN op. The audit's final step is to confirm the op fits the Metal 2.0 factory concept and record the choice in the audit report.

**Secondary audience**: AI agents performing the [port](../port/metal2.md). The port inherits the audit's decision and implements the factory entry point against it. The "Port plan" and "Port report" deliverable sections at the bottom of this document carry the decision forward through the port artifacts.

**The division of labor with the recipe.** The recipe owns the *contents* of the artifact — how you build a `ProgramSpec` (kernels, DFBs, semaphores, tensor parameters, work units) and its paired `ProgramRunArgs`. This document owns the *wrapper* — the factory method that returns those two objects to the framework, how the framework caches and dispatches it, and the handful of device-operation-class edits the port forces. When the recipe says "return the artifact," the shape of that return lives here.

---

## The Metal 2.0 factory concept

A Metal 2.0 op factory satisfies **`MetalV2FactoryConcept`**: it implements a single method, `create_program_artifacts`, that returns a `ProgramArtifacts` (a `ProgramSpec`, its `ProgramRunArgs`, and any op-owned tensors the factory allocates). The framework adapter stamps a `Program` from the spec onto each mesh coordinate range on cache miss, and refreshes tensor bindings on cache hit.

This is the Metal 2.0 factory concept ops port to today. It supports:

- **Single-program** — the factory produces one `ProgramSpec`, stamped identically across the mesh.
- **Op-owned device tensors (optional)** — the factory *may* carry op-owned `MeshTensor`s (config / index-table / workspace tensors it builds beyond the op's io) in `op_owned_tensors`; the framework parks them in the cache entry at a stable address for the cached `Program`'s lifetime. It may **not** allocate its own `GlobalSemaphore`s (the artifact carries only `MeshTensor`s). Every tensor a `TensorArgument` references must be reachable from `tensor_args` / `tensor_return_value` *or* be one of the `op_owned_tensors`.
- **Strict tensor-arg matching** — every `TensorParameter` enforces an exact `TensorSpec` match when the framework binds a tensor to it.

```cpp
struct MyProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const operation_attributes_t& attributes,
        const tensor_args_t&          tensor_args,
        tensor_return_value_t&        tensor_return_value);
};
```

`ProgramArtifacts` has three fields — `spec`, `run_params`, and `op_owned_tensors` (default-empty):

```cpp
ttnn::device_operation::ProgramArtifacts MyProgramFactory::create_program_artifacts(
    const operation_attributes_t& attributes,
    const tensor_args_t&          tensor_args,
    tensor_return_value_t&        tensor_return_value) {

    // ... build the spec and run-args (see the recipe's Construct step) ...
    tt::tt_metal::experimental::ProgramSpec    spec{ /* ... */ };
    tt::tt_metal::experimental::ProgramRunArgs run_args{ /* ... */ };

    return ttnn::device_operation::ProgramArtifacts{
        .spec       = std::move(spec),
        .run_params = std::move(run_args),
        // .op_owned_tensors = std::move(op_owned),  // only if the factory carries op-owned tensors
    };
}
```

### How the framework caches and dispatches it

The op author writes only `create_program_artifacts`. The framework adapter does the rest:

- **Cache miss**: the adapter calls `create_program_artifacts`, builds one `Program` per mesh coordinate range from the spec, applies the initial `ProgramRunArgs`, and resolves each `TensorArgument` against the tensors enumerated from `tensor_args` / `tensor_return_value` followed by the factory's `op_owned_tensors` (matched by `MeshTensor` identity within the call). The op-owned tensors are parked in the cache entry so their allocation outlives the miss and stays at a stable address across dispatches.
- **Cache hit**: the adapter enumerates fresh tensors (io tensors plus the parked op-owned ones) and patches their `TensorArgument`s into the cached `Program` in place — **only the tensor arguments are refreshed**, nothing else in the run-args — no `Program` rebuild, no factory re-run.

The cache key is the op itself — its type, attributes, and tensor args (the framework's automatic hash), combined with the target mesh coordinates; a custom `compute_program_hash`, if present, replaces that default. Two dispatches with matching op-args share a cache entry, and only the tensor bindings are refreshed between them. The porter doesn't write any of this — it falls out of returning a correct `ProgramArtifacts`.

### Extracting the tensor

The factory receives device-resident `ttnn::Tensor`s through `tensor_args` and `tensor_return_value`. **Extract the underlying `MeshTensor` from each at the top of the factory and work with it throughout.** A ProgramFactory builds against Metalium APIs, so its body should hold a Metalium memory object — the `MeshTensor` — rather than the TTNN wrapper. `ttnn::Tensor::mesh_tensor()` returns a `const MeshTensor&` (the rvalue overload is deleted, so call it on the named arguments, never a temporary); extract once at entry and pass `const MeshTensor&` to helpers instead of reaching back through `.mesh_tensor()` at each site. See [migration guide — Factory skeleton](migration_guide.md#factory-skeleton) for the worked example and the full tensor-type story.

Declare each `TensorParameter` from the tensor's `tensor_spec()`, and reference the same tensor from the paired `TensorArgument` in `ProgramRunArgs::tensor_args`. The adapter matches a `TensorArgument` back to its input by `MeshTensor` identity — so a `TensorArgument` must reference a tensor reachable from the factory's parameters (an io tensor, or one of the `op_owned_tensors`), never a copy. (Constructing or copying a tensor and referencing the copy fails at runtime.)

---

## Feasibility gate

The audit's job here is one question: **does the op fit the single concept above?**

- **Single-program** (the common case — op-owned tensors are fine) → `MetalV2FactoryConcept`. Proceed.
- **Anything else** → the port is **blocked on framework work**, not porter-resolvable. Record RED and stop.

The "anything else" cases — and the reason they're blocked:

- **Op-owned `GlobalSemaphore`s.** The factory allocates its own `GlobalSemaphore`s in its body. Op-owned *tensors* are supported (the artifact carries them; see above), but the artifact has no slot for op-owned semaphores yet. An op needing only op-owned *tensors* is **not** blocked — it ports cleanly.
- **Multi-program / per-coord variation.** The op's programs genuinely differ across mesh coordinates (CCL-style). The single-program adapter stamps one spec everywhere.

Op-owned *tensors* are supported on this concept; the remaining gaps — op-owned `GlobalSemaphore`s and genuine multi-program workloads — are not porter-resolvable and are tracked separately. If you hit one, the op is parked until that framework work lands — record it in the audit so it feeds prioritization.

> **Heads-up — a legacy `MeshWorkload` is not automatically a multi-program op.** Some legacy ops construct a `MeshWorkload` only because the legacy framework couldn't carry op-owned tensors on the single-program path. If every per-coord program is structurally identical (same kernels, same DFB shape, same bindings — only the tensor data differs) and the only thing pushing it multi-program was a resource workaround, the op is *morally* single-program and **ports cleanly** — carry its **op-owned** tensors in `op_owned_tensors` on the single-program concept and drop the `MeshWorkload`. Record it as a resource-workaround unwind rather than genuine per-coord variation.

### Tensor-arg matching — keep strict

Every `TensorParameter` enforces an exact `TensorSpec` match by default. **Don't deviate during a port.** The relaxation infrastructure exists (`TensorParameter::advanced_options`, holding `dynamic_tensor_shape` / `match_padded_shape_only`), and the per-dispatch legality check respects it, but relaxations are a deliberate correctness-sensitive opt-in: the kernel must *actually* tolerate the relaxation, and declaring one the kernel doesn't tolerate is a silent wrong-answer bug. The bias of mistakes favors strict — forgetting to relax is merely slower (narrower cache equivalence, still correct); relaxing incorrectly is wrong output. A port is not the context to make that call. If you notice a kernel that *would* tolerate a relaxation (e.g. padding-only dimension differences), capture it in the port report under "Open items for downstream" — don't bake it into the port.

**The exception is a *known-required* relaxation the docs already call out.** Where a kernel is known to need one, it is flagged for you — the [pre-migration `ArgConfig::Runtime*` check](migration_guide.md#tensorparameter) and its op-family heads-ups (e.g. `eltwise` → `dynamic_tensor_shape = true`). Those are faithful mirrors of a relaxation the legacy op *already* declared, not a judgment call you're making. So the rule is two-sided: don't *self-decide* a relaxation, but *do* apply the ones the docs flag as required — follow the hint rather than DIY-ing it (or, conversely, ignoring it).

---

## Device-operation-class edits the port forces

The port's writeable surface is the program factory body — the device-operation class (`validate`, `invoke`, `compute_output_specs`, attribute parsing) is otherwise off-limits (see the recipe's [Scope discipline](../port/metal2.md#scope-discipline)). There are **three** sanctioned exceptions, each forced by the port, each recorded prominently in the port report.

### 1. Delete a custom `compute_program_hash`

If the device-operation defines a custom `compute_program_hash` (overriding the default reflection-based hash), **the port deletes it**, reverting to the default. This is sanctioned — not a freelance device-op edit — because:

- No Metal 2.0 factory concept reads a custom hash; the framework's automatic hash of the op (type + attributes + tensor args) is the cache key.
- ProgramDescriptor-ported custom hashes are frequently incorrect now: they silently omit `TensorSpec`, which trips `UpdateTensorArgs` legality failures on the *second and later* dispatches (program cache hot), not the first.
- The default is correct-by-construction.

Delete it as part of the port. Do **not** patch it to add `TensorSpec` (that path leads to subtle bugs), and do **not** defer it to "see if it bites at verification" — it's proactive port work. Record the deletion (file:line of what was removed) in the port report. If a custom hash is ever missed, its signature is `UpdateTensorArgs` `TensorSpec` legality failures on the second-and-later test invocations; the fix is the same — find and delete it.

### 2. Remove pybound legacy factory entry points

When the port causes a legacy factory entry point to vanish (`create_program_descriptor` is the canonical case), any pybind line referencing it must be deleted — leaving it would break the post-port build. This is a *user-visible* API surface change: downstream Python consumers (tests, notebooks, internal tooling) may reference the removed entry point. The exception is narrow — it applies *only* to the disappearing factory entry point, not to other pybind lines on the same op. See [Pattern: Removing pybound legacy factory entry points](port_patterns.md#pattern-removing-pybound-legacy-factory-entry-points) for the procedure, and record the removal in the port report under Handoff points (cite the pybind file, the function name, and what it was for).

### 3. Drop a factory parameter that exists only for a pybind hook

Some legacy factories carry a non-standard parameter that production code never sets — it exists only so a pybind test/introspection hook can drive the factory (layernorm's `create_descriptor` took an extra `const std::optional<CoreRangeSet>& core_range_set` used only by its pybind hook). The fixed `create_program_artifacts` signature (`attributes`, `tensor_args`, `tensor_return_value`) cannot carry it. Drop the parameter, inline its production default in the factory body, and delete the pybind hook that passed it (same procedure and report-handling as exception 2). This is mechanically the pybind-removal case with an extra parameter to unwind; flag it the same way. Don't try to preserve the hook — its `ProgramDescriptor` return is exactly what the port eliminates.

---

## Audit report deliverable

The auditor adds the following to `METAL2_PREPORT_AUDIT.md`. The decision is recorded here; the port inherits it.

```markdown
## TTNN ProgramFactory

### Concept
MetalV2FactoryConcept — or — BLOCKED (op-owned GlobalSemaphores / genuine multi-program; see below)

### Fit
- Single vs multi-program: [single — one ProgramSpec stamped across the mesh / multi — BLOCKED]
- Op-owned device resources: [none / op-owned tensors (supported) / op-owned GlobalSemaphores — BLOCKED, list them]
- Tensor-arg matching: strict [default; deviation requires a paragraph and is not a port-time call]
- Legacy-to-Metal-2.0 shape: [1:1 with legacy — or — legacy MeshWorkload was a resource workaround, see heads-up]

### Custom compute_program_hash
[present at file:line → port deletes it / none — already default reflection-based hash]

### Stop signals
[If BLOCKED: which framework capability is missing (op-owned GlobalSemaphores / genuine multi-program), and confirm the overall audit result is RED. Otherwise: "None."]
```

## Port plan deliverable (porter-facing)

The porter inherits the audit's decision; the port plan's TTNN section is a brief carry-forward, not a re-derivation:

```markdown
## TTNN ProgramFactory
- Concept (inherited from audit): MetalV2FactoryConcept
- Custom compute_program_hash: [delete (was at file:line) / none]
- Implementation notes: [optional — anything specific about how this op realizes the concept; most ports won't need this]
```

If you find yourself disagreeing with the audit's decision, **stop and surface it** — don't unilaterally override. An in-port revision is a signal the audit was incomplete, and the invoker needs to know.

## Port report deliverable (porter-facing)

The porter adds the following to `METAL2_PORT_REPORT.md` at the end of the port. The audit decided; the report confirms what was realized and surfaces friction.

```markdown
## TTNN ProgramFactory

### Concept realized
[Confirm MetalV2FactoryConcept, or — if something changed — explain why and confirm it was surfaced with the invoker before re-deciding.]

### Device-op-class edits
- Custom compute_program_hash deleted: [file:line, or "none"]
- Pybind entry points removed: [file + function, or "none"]

### Open items
[Anything noticed about the factory layer during the port:
- Relaxation candidates (kernels that would tolerate relaxed tensor matching — not applied during port).
- Reasons the op would benefit from a capability not yet on this concept (op-owned GlobalSemaphores, multi-program, caching-strategy control).
- Friction with the concept fit or the entry-point wiring.]
```

If the port stayed on the default concept with no device-op edits, these sections are short — that's the success case.

---

## Cross-references

- [Audit doc](../audit/metal2.md) — the feasibility audit that invokes this document as its final step.
- [Port recipe](../port/metal2.md) — builds the `ProgramSpec` + `ProgramRunArgs` this document's factory entry point returns.
- [Migration guide — Design Principles](migration_guide.md#design-principles) — the named-binding model the spec is built on.
- [`ttnn/api/ttnn/operation_concepts.hpp`](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/api/ttnn/operation_concepts.hpp) — `MetalV2FactoryConcept` definition in code.
- [`ttnn/api/ttnn/metal_v2_artifacts.hpp`](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/api/ttnn/metal_v2_artifacts.hpp) — `ProgramArtifacts` field layout.
