# Porting an Op to Metal 2.0 — TTNN Integration

> The TTNN device-operation glue a Metal 2.0 port needs: which factory concept the op lands on, the factory entry point that returns the spec, and the two device-op-class edits the port forces (pybind cleanup, and dropping a pybind-hook-only factory parameter). Lives in its own document because the TTNN factory layer churns on a different cadence than the Metal 2.0 host API — the [port recipe](../port/metal2_port.md) covers building the `ProgramSpec` + `ProgramRunArgs` (stable); this doc covers wiring that into TTNN's framework (in flux).

## Read this first

**Primary audience**: AI agents performing the [audit](../audit/metal2_audit.md) on a TTNN op. The audit's final step is to confirm the op fits one of the two Metal 2.0 factory concepts, decide which, and record the choice in the audit report.

**Secondary audience**: AI agents performing the [port](../port/metal2_port.md). The port inherits the audit's decision and implements the factory entry point against it. The "Port plan" and "Port report" deliverable sections at the bottom of this document carry the decision forward through the port artifacts.

**A note on "legacy."** The audit and the readiness sheet use **`legacy device-op`** narrowly, for a *pre-`ProgramDescriptor`* op on `ProgramFactoryConcept`. To avoid colliding with that, this document says **"the ported-from factory"** for whatever shape a given port starts from — which, for the ops portable today, is always a `ProgramDescriptor` factory.

**The division of labor with the recipe.** The recipe owns the *contents* of the artifact — how you build a `ProgramSpec` (kernels, DFBs, semaphores, tensor parameters, work units) and its paired `ProgramRunArgs`. This document owns the *wrapper* — the factory method that returns those two objects to the framework, how the framework caches and dispatches it, and the handful of device-operation-class edits the port forces. When the recipe says "return the artifact," the shape of that return lives here.

---

## The two Metal 2.0 factory concepts

Both are built on the same method, `create_program_artifacts`, and both build the cached `Program` the same way. **They differ only in what happens on a cache hit:**

| Concept | On a cache hit, what gets refreshed | Written by |
|---|---|---|
| **`ProgramSpecFactoryConcept`** (base) | the tensor bindings, and nothing else | the framework |
| **`CustomProgramSpecFactoryConcept`** | exactly what the op's `override_runtime_arguments` returns | the op author |

**One question decides which one this op targets: does the ported-from factory have an `override_runtime_arguments`?** No → the base concept. Yes → the custom concept, and the port's job is to *translate* that method rather than delete it. The audit records the target concept; the porter inherits it and does not re-decide.

**What does *not* decide it: a custom `compute_program_hash`.** The two axes are independent and the corpus populates all four combinations. A pointer-patching op can legitimately carry a custom hash — it may be shrinking the *cost* of hashing derived attributes rather than widening the cache-equivalence class — and an op with an `override_runtime_arguments` may have no custom hash at all. Don't read either as a signal about the other. What the port does about a custom hash is the same on both paths: [leave it alone](#the-cache-key-leave-the-custom-hash-alone).

## The base concept: `ProgramSpecFactoryConcept`

A factory satisfying it implements a single method, `create_program_artifacts`, that returns a `ProgramArtifacts` (a `ProgramSpec`, its `ProgramRunArgs`, and any op-owned tensors the factory allocates). The framework adapter stamps a `Program` from the spec onto each mesh coordinate range on cache miss, and refreshes tensor bindings on cache hit.

It supports:

- **Single-program** — the factory produces one `ProgramSpec`, stamped identically across the mesh.
- **Op-owned device tensors (optional)** — the factory *may* carry op-owned `MeshTensor`s (config / index-table / workspace tensors it builds beyond the op's io) in `op_owned_tensors`; the framework parks them in the cache entry at a stable address for the cached `Program`'s lifetime. It may **not** allocate its own `GlobalSemaphore`s (the artifact carries only `MeshTensor`s). Every tensor a `TensorArgument` references must be reachable from `tensor_args` / `tensor_return_value` *or* be one of the `op_owned_tensors`.
- **Strict tensor-arg matching** — every `TensorParameter` enforces an exact `TensorSpec` match when the framework binds a tensor to it. The struct also carries a `relaxations` field that loosens those rules; it is real, and it is not a port-time decision — see [Tensor-arg matching — keep strict](#tensor-arg-matching--keep-strict).

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

## The custom concept: `CustomProgramSpecFactoryConcept`

Everything above still applies. This concept **is** the base concept plus one method — same `create_program_artifacts`, same stamping onto each mesh coordinate range, and the cache-miss path is inherited unchanged, **op-owned tensor support included**. Only the cache hit differs.

```cpp
struct MyProgramFactory {
    // create_program_artifacts exactly as on the base concept ...

    static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
        const operation_attributes_t& attributes,
        const tensor_args_t&          tensor_args,
        tensor_return_value_t&        tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& coord = std::nullopt);
};
```

Only the **return type** is concept-enforced. A ported-from `override_runtime_arguments` returning `void` does not satisfy the concept — it leaves the factory on the base concept, silently. So the port has to change the method's *shape*, not only its body.

### The cache-hit contract

On every cache hit the adapter calls `override_runtime_arguments` **once per mesh coordinate range**, passing that range's start coordinate, and applies the result via `UpdateProgramRunArgs`. The factory is not re-run; the `Program` is not rebuilt.

- **Per-device values are expressible** — that is what `coord` is for. Use it where the ported-from override varied values by device.
- **`UpdateProgramRunArgs` is an arbitrary *partial* update.** Anything you leave out keeps whatever value it was last given, from the cache-miss `SetProgramRunArgs` or from an earlier hit. That makes **the set of things the override refreshes part of the op's behavior** — and that set is **inherited, not decided.** Refresh exactly what the ported-from override refreshed: no more, no less. Adding a refresh it didn't do is as much a deviation as dropping one it did. You are not being asked to work out what does or doesn't vary across the enqueue loop; the ported-from override already encodes that answer.
- **Borrowed-memory DFBs couple the two.** If the override sets a size override for a borrowed-memory DFB, it must *also* supply that DFB's backing tensor argument.

### The override owns the tensor bindings too

**This is where a port most easily stops being faithful** — not by translating the ported-from override badly, but by assuming the framework still helps.

On the base concept the framework refreshes tensor bindings for you. On the custom concept **it does not** — the custom adapter *replaces* that step rather than adding to it, so the only bindings refreshed on a cache hit are the ones your returned `ProgramRunArgs::tensor_args` carries. Omit a tensor and its binding stays frozen at the address the cache-miss dispatch wrote, for the life of the cache entry.

**This responsibility is not new — it transfers.** A `ProgramDescriptor` factory that declares an `override_runtime_arguments` *already* owns its entire cache-hit re-derivation: for such an op the descriptor adapter performs **no** address inference of its own and re-applies "every runtime arg AND every tensor-backed CB address" through the override alone. The two shapes are aligned by design and the port preserves the contract exactly. What changes is the *form* and the *channel*:

| | Ported-from (`ProgramDescriptor`) | Metal 2.0 |
|---|---|---|
| Form | **imperative** — mutates the cached `Program` | **declarative** — returns a `ProgramRunArgs` |
| Addresses travel as | raw buffer addresses written into runtime args / CB fields | `TensorArgument`s in `tensor_args` |

So the address-handling statements in the ported-from override do not disappear on the way across — **they become `tensor_args` entries.** Re-expressing them as runtime-arg *values* instead is the smuggling anti-pattern the binding model exists to prevent: it would mean declaring a runtime argument that should not exist, for something Metal 2.0 represents as a binding.

**The failure that is silent is the opposite mistake:** assuming the framework still patches bindings for you. It does on the base concept — which is what every other example in these docs shows — and it does not here. An override that simply omits `tensor_args` compiles, runs, and returns wrong numbers only on cache hits, and only once the incoming tensors stop landing at the first call's addresses.

**In practice this means rebuilding `tensor_args` for every `TensorParameter` bound to an io tensor, on every dispatch** — because that is what the ported-from override was already doing. **Op-owned tensors are the exception, and they are excluded automatically:** the override's parameters do not expose them, and they do not need refreshing — the framework parks them at a stable address for the cached `Program`'s lifetime, so the binding written at cache-miss stays valid. The API header takes the same position on the general API: a partial set is marked *advanced users only*, with a caution that a stale binding to a destroyed `MeshTensor` is undefined behavior.

```cpp
ProgramRunArgs MyProgramFactory::override_runtime_arguments(
    const operation_attributes_t& attributes,
    const tensor_args_t&          tensor_args,
    tensor_return_value_t&        tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& coord) {

    const auto& input  = tensor_args.input.mesh_tensor();
    const auto& output = tensor_return_value.mesh_tensor();

    ProgramRunArgs params;

    // Every TensorParameter, every dispatch. Not optional on this path.
    params.tensor_args = {
        {INPUT,  TensorArgument{input}},
        {OUTPUT, TensorArgument{output}},
    };

    // Then whatever the ported-from override recomputed per dispatch, keyed by name —
    // same shape as the run-args built in create_program_artifacts.
    params.kernel_run_args = { /* ... */ };

    return params;
}
```

**If the ported-from override skipped a binding, skip it too.** An op author may deliberately omit one they know cannot change, to save the patching cost — and from the outside that is indistinguishable from an oversight. Either way it is not yours to adjudicate: reproduce the set you found. If the omission looks wrong to you, that is a line in the port report, not an edit to the port. The port reproduces the ported-from op's behavior *including its bugs* — see [the porting invariant](../port/metal2_port.md#the-principle) for why that is a rule rather than a resignation.

---

## Feasibility gate

The audit's job here is one question: **does the op fit one of the two concepts above?** Their limits are identical — the fork between them is about the cache-hit path, not about what the op is allowed to be.

- **Single-program** (the common case — op-owned tensors are fine) → proceed, on whichever concept the [selector](#the-two-metal-20-factory-concepts) picks.
- **Op-owned `GlobalSemaphore`s.** The factory allocates its own in its body, and the artifact carries only `MeshTensor`s — no slot for them. Not porter-resolvable: record RED and stop. Op-owned *tensors* are supported and **not** blocked.
- **Multi-program / per-coord variation.** The op's programs genuinely differ across mesh coordinates (CCL-style), and both concepts above stamp one spec everywhere. **Don't RED this on your own judgment** — see below.

### Multi-program: the sheet gates, you name the target

A Metal 2.0 mesh-workload concept was expected to follow the two above, so this document holds no verdict on whether one exists — that would go stale the day it lands. The sheet's `Is able to port?` is the gate and already accounts for framework support: [read that cell, don't vet it](../audit/metal2_audit.md#ttnn-factory-concept-prerequisite). Attribute and route a `no` like any other; on a `yes`, name the target concept from the code:

```bash
grep -n "concept \|create_" ttnn/api/ttnn/operation_concepts.hpp
```

Look for a **Metal 2.0** concept — one built on a `ProgramSpec` / `ProgramArtifacts` entry point rather than a `ProgramDescriptor` — that admits per-coordinate programs, and record its name as the target. **As of 2026-08-31 there was none:** the only two Metal 2.0 concepts were `ProgramSpecFactoryConcept` and `CustomProgramSpecFactoryConcept`, both single-program, both excluding `MeshWorkloadFactoryConcept`. If that is still the picture, record **target concept: none yet** and RED, citing this lookup as the evidence.

Either way the **port procedure covers only the two single-program concepts**: a mesh-workload target means the porter stops at [its coverage boundary](../port/metal2_port.md#what-this-procedure-covers). Note that in the brief so the stop reads as expected, not as a failed port.

> **Heads-up — a legacy `MeshWorkload` is not automatically a multi-program op.** Some legacy ops construct a `MeshWorkload` only because the legacy framework couldn't carry op-owned tensors on the single-program path. If every per-coord program is structurally identical (same kernels, same DFB shape, same bindings — only the tensor data differs) and the only thing pushing it multi-program was a resource workaround, the op is *morally* single-program and **ports cleanly** — carry its **op-owned** tensors in `op_owned_tensors` on the single-program concept and drop the `MeshWorkload`. Record it as a resource-workaround unwind rather than genuine per-coord variation.

### Tensor-arg matching — keep strict

Every `TensorParameter` enforces an exact `TensorSpec` match by default. **Don't deviate during a port.** The relaxation infrastructure exists (`TensorParameter::relaxations`, a `TensorSpecRelaxations` holding `dynamic_tensor_shape` / `match_padded_shape_only`), and the per-dispatch legality check respects it, but relaxations are a deliberate correctness-sensitive opt-in: the kernel must *actually* tolerate the relaxation, and declaring one the kernel doesn't tolerate is a silent wrong-answer bug. The bias of mistakes favors strict — forgetting to relax is merely slower (narrower cache equivalence, still correct); relaxing incorrectly is wrong output. A port is not the context to make that call. If you notice a kernel that *would* tolerate a relaxation (e.g. padding-only dimension differences), capture it in the port report under "Open items for downstream" — don't bake it into the port.

**The exception is a *known-required* relaxation the docs already call out.** Where a kernel is known to need one, it is flagged for you — the [pre-migration `ArgConfig::Runtime*` check](migration_guide.md#tensorparameter) and its op-family heads-ups (e.g. `eltwise` → `dynamic_tensor_shape = true`). Those are faithful mirrors of a relaxation the legacy op *already* declared, not a judgment call you're making. So the rule is two-sided: don't *self-decide* a relaxation, but *do* apply the ones the docs flag as required — follow the hint rather than DIY-ing it (or, conversely, ignoring it).

---

## Device-operation-class edits the port forces

The port's writeable surface is the program factory body — the device-operation class (`validate`, `invoke`, `compute_output_specs`, attribute parsing) is otherwise off-limits (see the recipe's [Scope discipline](../port/metal2_port.md#scope-discipline)). There are **three** sanctioned exceptions, each forced by the port, each recorded prominently in the port report. The op's cache key is *not* among them — see [The cache key: leave the custom hash alone](#the-cache-key-leave-the-custom-hash-alone).

### 1. Remove pybound legacy factory entry points

When the port causes a legacy factory entry point to vanish (`create_descriptor` is the canonical case), any pybind line referencing it must be deleted — leaving it would break the post-port build. This is a *user-visible* API surface change: downstream Python consumers (tests, notebooks, internal tooling) may reference the removed entry point. The exception is narrow — it applies *only* to the disappearing factory entry point, not to other pybind lines on the same op. See [Pattern: Removing pybound legacy factory entry points](port_patterns.md#pattern-removing-pybound-legacy-factory-entry-points) for the procedure, and record the removal in the port report under Handoff points (cite the pybind file, the function name, and what it was for).

### 2. Drop a factory parameter that exists only for a pybind hook

Some legacy factories carry a non-standard parameter that production code never sets — it exists only so a pybind test/introspection hook can drive the factory (layernorm's `create_descriptor` took an extra `const std::optional<CoreRangeSet>& core_range_set` used only by its pybind hook). The fixed `create_program_artifacts` signature (`attributes`, `tensor_args`, `tensor_return_value`) cannot carry it. Drop the parameter, inline its production default in the factory body, and delete the pybind hook that passed it (same procedure and report-handling as exception 1). This is mechanically the pybind-removal case with an extra parameter to unwind; flag it the same way. Don't try to preserve the hook — its `ProgramDescriptor` return is exactly what the port eliminates.

### 3. Give a direct-descriptor op a conventional program factory

**Recognition signal.** The device-operation declares `create_descriptor` (and any `override_runtime_arguments`) as its *own* static member, with no `program_factory_t` — there is no factory struct at all. The framework accepts that shape through a shim, `MeshDeviceOperationAdapter::DirectDescriptorFactory`, selected by the `HasDirectDescriptor` predicate in [`operation_concepts.hpp`](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/api/ttnn/operation_concepts.hpp).

**Why the port is forced.** That shim is keyed on the literal name `create_descriptor`, and **there is no equivalent for `create_program_artifacts`.** So the method the port replaces is the same one that made the op satisfy `DeviceOperationConcept`: swap it and `HasDirectDescriptor` goes false while `HasProgramFactoryType` was never true, leaving the op not a valid device operation. The failure is a concept mismatch at the call site, not a message naming the cause. **A `DirectSpecFactory` counterpart is a deliberate non-goal** — TTNN keeps one factory shape rather than two, so converting the op is the sanctioned resolution, not a workaround.

**Procedure.** Move the factory methods into a nested struct and declare the variant — the op keeps everything else, and the factory body still lives in the same `.cpp`:

```cpp
struct MyOpDeviceOperation {
    struct MyOpProgramFactory {
        static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
            const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
        // plus the translated override, on CustomProgramSpecFactoryConcept only
    };
    using program_factory_t = std::variant<MyOpProgramFactory>;
    // ... the rest of the device-operation class is untouched ...
};
```

**Name the struct `<OpName>ProgramFactory`** — `MorehAdamWProgramFactory`, `UniformProgramFactory` — matching the convention TTNN uses for these ops. Nothing cross-references the name, so a divergent one is merely a divergent one; pick this so a reviewer reading two ports sees the same shape.

Record it under Handoff points, noting that the op arrived in the direct-descriptor shape. Some ops reach the port already converted — TTNN moves them out of this shape when it has other reasons to touch them — so check for an existing `program_factory_t` before assuming the edit is yours to make; if one is there, this exception does not apply and the port is a method swap inside the existing struct.

---

## The cache key: leave the custom hash alone

If the device-operation defines a custom `compute_program_hash` — or reaches one through the backdoor route (a hand-written `attribute_values` / `to_hash` that narrows what the default reflection hash sees) — **the port leaves it exactly as it is.** Not deleted, not patched, not "reverted to the default." Touching it is a scope violation like any other device-op-class edit.

This holds on **every** port path; it is not specific to one factory concept.

### Why: the call was already made, upstream, by a human

The ops team analyses each op's custom hash *before* the port and records the verdict; that verdict reaches the audit, and the audit reaches you. **An op that arrives with a cleared audit is one whose hash a domain expert has already vetted against this port and green-lit.** You are not being asked to make that architectural call on the fly — you are being asked to respect one that has been made.

That is also why deleting the hash is not a neutral simplification. The hash *is* the op's cache-equivalence class: removing it trades away the op's cache hits, which is a performance decision the port has no standing to make, and overrules the person who cleared it.

Do not reason your way back to deleting it from "Metal 2.0 doesn't read a custom hash" — it does. The spec-path adapter uses the op's `compute_program_hash` whenever one is present ([`mesh_device_operation_adapter.hpp:982-983`](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/api/ttnn/mesh_device_operation_adapter.hpp)), and the exact collision-resolution key beside it is deliberately built to accommodate one.

### If the hash contradicts the audit, stop — you have found an upstream error

The audit tells you which tensor relaxations apply to this op. Today that answer is always **none**, which means the hash has to pin the whole `TensorSpec`. A hash that tolerates *any* `TensorSpec` deviation contradicts that, and the contradiction means the pre-port vetting was wrong.

It surfaces two ways, and they are the same defect:

- **At verification** — a `TensorSpec` legality failure on the *second and later* dispatches (program cache hot), never the first. It is reported by whichever call refreshes the cache hit on your concept: `UpdateTensorArgs` on the base concept, `UpdateProgramRunArgs` on the custom one.
- **By reading** — you notice, while working from the inventory, that the hash omits part of the `TensorSpec`; equivalently, that its tolerance could be expressed as a relaxation. A failing test is not required to have found this. Don't go hunting for it — but don't sit on it either.

Either way the response is the same, and it is **not** a fix:

- **Stop, and flag it prominently.** Record the hash's file:line, exactly which deviation it tolerates (or what the legality check rejected, and on which dispatch), and that the audit declared no relaxations for this op.
- Do **not** delete the hash to make the symptom go away, and do **not** patch it to fold in `TensorSpec`. Both bury a mistaken expert verdict inside a port, where the next person to hit it has no trail back to the decision that was actually wrong.

---

## Audit report deliverable

The auditor adds the following to `METAL2_PREPORT_AUDIT.md`. The decision is recorded here; the port inherits it.

```markdown
## TTNN ProgramFactory

### Concept
[ProgramSpecFactoryConcept (ported-from factory has no override_runtime_arguments) / CustomProgramSpecFactoryConcept (it does — name the method's file:line) / <mesh-workload concept, named from the header lookup> / BLOCKED (op-owned GlobalSemaphores, or multi-program with no concept yet; see below)]

### Fit
- Single vs multi-program: [single — one ProgramSpec stamped across the mesh / multi — quote the sheet's `Is able to port?` cell and the header-lookup result]
- Op-owned device resources: [none / op-owned tensors (supported) / op-owned GlobalSemaphores — BLOCKED, list them]
- Tensor-arg matching: strict [default; deviation requires a paragraph and is not a port-time call]
- Legacy-to-Metal-2.0 shape: [1:1 with legacy — or — legacy MeshWorkload was a resource workaround, see heads-up]

### Custom compute_program_hash
[present at file:line / backdoor (attribute_values | to_hash) at file:line / none — default reflection-based hash]
[Recorded so the porter knows it is there and leaves it alone; the port never edits it.]

### Stop signals
[If BLOCKED: which framework capability is missing (op-owned GlobalSemaphores, or multi-program with no concept yet — for that one quote the sheet cell and the lookup), and confirm the overall audit result is RED. If the target is a mesh-workload concept: note that the port procedure does not cover it yet, so the porter will stop at its coverage boundary. Otherwise: "None."]
```

## Port plan deliverable (porter-facing)

The porter inherits the audit's decision; the port plan's TTNN section is a brief carry-forward, not a re-derivation:

```markdown
## TTNN ProgramFactory
- Concept (inherited from audit): [ProgramSpecFactoryConcept / CustomProgramSpecFactoryConcept — anything else is outside this procedure; stop and report]
- Custom compute_program_hash: [present at file:line — leave intact / none]
- Implementation notes: [optional — anything specific about how this op realizes the concept; most ports won't need this]
```

If you find yourself disagreeing with the audit's decision, **stop and surface it** — don't unilaterally override. An in-port revision is a signal the audit was incomplete, and the invoker needs to know.

## Port report deliverable (porter-facing)

The porter adds the following to `METAL2_PORT_REPORT.md` at the end of the port. The audit decided; the report confirms what was realized and surfaces friction.

```markdown
## TTNN ProgramFactory

### Concept realized
[Confirm the concept the audit chose, or — if something changed — explain why and confirm it was surfaced with the invoker before re-deciding.
On CustomProgramSpecFactoryConcept, also confirm the override returns a TensorArgument for every io-tensor TensorParameter (op-owned ones are excluded by construction), or name the ones deliberately skipped and the ported-from statement that justifies each.]

### Device-op-class edits
- Pybind entry points removed: [file + function, or "none"]
- Custom compute_program_hash: [left intact at file:line — confirm untouched / none]

### Open items
[Anything noticed about the factory layer during the port:
- Relaxation candidates (kernels that would tolerate relaxed tensor matching — not applied during port).
- Reasons the op would benefit from a capability not yet on this concept (op-owned GlobalSemaphores, multi-program, caching-strategy control).
- Friction with the concept fit or the entry-point wiring.]
```

If the port stayed on the default concept with no device-op edits, these sections are short — that's the success case.

---

## Cross-references

- [Audit doc](../audit/metal2_audit.md) — the feasibility audit that invokes this document as its final step.
- [Port recipe](../port/metal2_port.md) — builds the `ProgramSpec` + `ProgramRunArgs` this document's factory entry point returns.
- [Migration guide — Design Principles](migration_guide.md#design-principles) — the named-binding model the spec is built on.
- [`ttnn/api/ttnn/operation_concepts.hpp`](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/api/ttnn/operation_concepts.hpp) — `ProgramSpecFactoryConcept` definition in code.
- [`ttnn/api/ttnn/metal_v2_artifacts.hpp`](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/api/ttnn/metal_v2_artifacts.hpp) — `ProgramArtifacts` field layout.
