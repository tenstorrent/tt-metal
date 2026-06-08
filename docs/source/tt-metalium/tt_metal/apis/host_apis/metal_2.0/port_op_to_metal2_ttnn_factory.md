# Porting an Op to Metal 2.0 — TTNN Factory Selection

> Final step of the [audit](port_op_to_metal2_audit.md). Covers the TTNN ProgramFactory concept choice: which concept the port should land on, when an escalation is needed, when the port is blocked on framework work. Lives in its own document because the TTNN factory layer is on a feature branch and still evolving while the Metal 2.0 host API stabilizes — the two have separate update cadences and intentionally separate docs.

## Read this first

**Primary audience**: AI agents performing the [audit](port_op_to_metal2_audit.md) on a TTNN op. The audit's final step is to walk the decision tree below, choose the Metal 2.0 factory concept the port will use, and record the choice in the audit report.

**Secondary audience**: AI agents performing the [port](port_op_to_metal2_recipe.md). The port inherits the audit's decision — porters do not re-walk the decision tree, they implement against the chosen concept. The "Port plan" and "Port report" template sections at the bottom of this document carry the audit's decision forward through the port artifacts; consult them when filling out `METAL2_PORT_PLAN.md` and `METAL2_PORT_REPORT.md`.

**Inputs (auditor)**:
- The legacy op's existing TTNN factory class (one of `ProgramFactoryConcept` / `ProgramDescriptorFactoryConcept` / `MeshWorkloadFactoryConcept`).
- The rest of the audit findings (the factory selection is the *final* step of the audit, after the feasibility gates have produced a tentative overall result).
- This document.

**Outputs (auditor)**:
- A "TTNN ProgramFactory" section in `METAL2_PREPORT_AUDIT.md` documenting the chosen concept, the path through the decision tree, and any stop signals encountered (see [Audit report deliverable](#audit-report-deliverable) below).
- If the decision tree directs to a stubbed concept (Advanced), the overall audit result becomes **RED — port blocked on framework**; the section records the case for downstream triage.

**Workflow (auditor)**:
1. Read this whole document.
2. Read the legacy factory and walk the [decision tree](#decision-tree).
3. Record the result in the audit report's TTNN ProgramFactory section.
4. If a stop signal fires, downgrade the audit's overall result accordingly.

**Stop signal**: if the decision tree directs to one of the **Advanced** concepts, **stop walking the tree and record the case as a RED audit outcome.** Those concepts are stubbed-only today (see [Today's implementation status](#todays-implementation-status)); the port is blocked on framework implementation, not a porter-resolvable situation.

---

## Branch context

The TTNN framework code that supplies the new factory concepts lives on the branch **`akertesz/ttnn-metal2concept-improvements`** and is not yet on `main`.

**The audit can be performed against `main`** — the decision tree walks the legacy code, which the chosen-concept decision doesn't depend on the framework code being locally available. The auditor records the chosen concept name; checkout-of-the-right-branch is a porter-side concern, not an audit-side concern.

**The port must work on the factory branch**:
- Branched off **`akertesz/ttnn-metal2concept-improvements`**, not `main`.
- Port PRs target the same branch until the framework merges to main.

If, during the port, you find that the new concept names (`Metal2FactoryConcept`, `ProgramSpecFactoryConcept`, `WorkloadSpecFactoryConcept`) aren't available in `ttnn/api/ttnn/operation_concepts.hpp`, you're branched off the wrong base. Stop and ask the invoker.

---

## The default — `ProgramSpecFactoryConcept`

**Reach for this first.** Almost every op should land here. The framework's four-axis design has a *safe-by-construction* default, and `ProgramSpecFactoryConcept` with no further declarations is it:

- **Basic shape** — one combined `create_program_artifacts` method.
- **Single-program** — the factory produces one `ProgramSpec` per dispatch.
- **No op-owned device resources** — the factory references only resources reachable from `tensor_args` and `tensor_return_value`.
- **Default caching strategy `MaximizeCacheReuse`** — implicit; the framework re-runs the factory on every dispatch and re-applies the full `ProgramRunArgs`, eliminating stale-state bugs by construction.
- **Strict tensor-arg matching** — every `TensorParameter` enforces exact `TensorSpec` match on `set_program_run_args`.

Code skeleton:

```cpp
struct MyProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const operation_attributes_t& attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

ttnn::device_operation::ProgramArtifacts MyProgramFactory::create_program_artifacts(
    const operation_attributes_t& attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {

    // ... build the spec and run-args ...
    tt::tt_metal::experimental::ProgramSpec    spec{ /* ... */ };
    tt::tt_metal::experimental::ProgramRunArgs run_args{ /* ... */ };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_args = std::move(run_args),
        // .op_owned_tensors = {},  // default-empty; see escalation 2 below
    };
}
```

The `ProgramArtifacts` struct's fields are `op_owned_tensors` (default-empty), `spec`, `run_args`. Most ports leave `op_owned_tensors` defaulted and only populate `spec` and `run_args`.

If the decision tree directs you anywhere other than this default, **flag the deviation prominently** in your port report. Non-default factory shapes are unusual and reviewers will want to confirm the escalation was warranted.

---

## Decision tree

Walk these in order. Each level reflects observable evidence in the legacy code — read the legacy factory before deciding. The "common case" answer at every level keeps you on the default.

### 1. Single-program or multi-program?

**Observe**: how many `ProgramDescriptor`s does the legacy factory produce per dispatch? Does it construct a `MeshWorkload`?

- **One program** (the common case) → single-program. Use `ProgramSpecFactoryConcept`.
- **Multiple programs with genuinely different shapes across mesh coords** → multi-program. Use `WorkloadSpecFactoryConcept`. (Caveat below — be sure the variation is genuine.)

**Heads-up — the legacy framework forced single-program-with-resources ops into a multi-program shape.** Some legacy ops construct a `MeshWorkload` only because the legacy framework couldn't carry op-owned tensors on the single-program path. The new Metal 2.0 framework removes this constraint: `ProgramSpecFactoryConcept` carries `op_owned_tensors` natively (escalation 2 below). If your legacy op uses a `MeshWorkload` but every per-coord program is structurally identical — same kernels, same DFB shape, same bindings, only the tensor data differs — **drop back to `ProgramSpecFactoryConcept`** with `op_owned_tensors`. Do *not* transliterate the workload shape.

Signs the legacy multi-program shape exists only for the resource workaround:
- Every per-coord program in the workload is a stamping of one underlying spec.
- The factory creates `MeshTensor`s or `GlobalSemaphore`s inside the body that the spec then references.
- No per-coord logic varies the program shape itself — only the data identities.

If you find this pattern, port to `ProgramSpecFactoryConcept` with `op_owned_tensors`. Record the unwinding in the port report under "Legacy-to-Metal-2.0 shape change."

### 2. Does the factory need to allocate op-owned device resources?

**Observe**: does the legacy factory body construct any device-side resources whose lifetime should track the cached entry, not the caller's tensors?

Patterns to look for:
- `MeshTensor` allocations inside the factory body (scratch tensors, lookup tables, intermediate buffers).
- `GlobalSemaphore` allocations (cross-program coordination).
- Any device-side resource the factory creates rather than receives via `tensor_args`.

- **No** (the common case) → no escalation. Leave `op_owned_tensors` empty (the default).
- **Yes, MeshTensor(s) only, single-program** → populate `ProgramArtifacts.op_owned_tensors` with the move-only tensors. The framework moves them into the cache entry; the spec and run-args can reference their stable addresses.
- **Yes, multi-program (any combination of MeshTensors and GlobalSemaphores)** → use `WorkloadSpecFactoryConcept`; populate `MeshWorkloadArtifacts.{op_owned_tensors, global_semaphores}` at the workload level. Resources are workload-scoped, shared across the per-coord programs.
- **Yes, GlobalSemaphore(s), single-program** → structurally impossible. A `GlobalSemaphore` is a cross-program coordination resource; on a single Program it has nothing to coordinate. If you see this, the legacy code is multi-program (revisit Decision 1).

#### Op-owned resources require explicit caching-strategy opt-in

If `op_owned_tensors` (or `global_semaphores`, in the workload case) is non-empty, you **must** declare `MinimizeCacheHitCost` on the factory:

```cpp
struct MyFactoryWithResources {
    static constexpr auto caching_strategy =
        ttnn::device_operation::ProgramCachingStrategy::MinimizeCacheHitCost;

    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(/* ... */);
};
```

**Why.** The default `MaximizeCacheReuse` re-runs the factory on every dispatch and re-allocates any owned resources — but the cache also already holds the *previous* allocation, which the cached Program is bound to. The result is freed-then-reused device memory. The framework rejects this combination at runtime with a clear `TT_FATAL`: omit the opt-in on a factory with non-empty `op_owned_tensors` and the first dispatch crashes.

`MinimizeCacheHitCost` runs the factory once on cache miss, allocates resources once, and skips the factory entirely on cache hit (refreshing only tensor bindings via `UpdateTensorArgs`). This is the only correct path for op-owned-resource factories on the Basic concepts.

#### MinimizeCacheHitCost adds restrictions on `ProgramRunArgs`

Opting into `MinimizeCacheHitCost` imposes restrictions on what the factory's `ProgramRunArgs` may contain:

- **Per-node RTAs are allowed** — they're the standard mechanism for per-node work distribution. They must be deterministic from the cache-miss-time inputs (this is the structural invariant the strategy relies on, not something you separately verify).
- **No common runtime args (CRTAs)** — the framework rejects these at runtime.
- **No DFB size overrides** — same; rejected at runtime.

If your factory legitimately needs CRTAs or DFB size overrides, the Basic + resources path is closed to you: you need the Advanced concept ([escalation 4](#4-heavy-immutable-extraction-work)), which is not yet implemented. Stop and report.

### 3. Do any `TensorParameter`s need to admit dimension variability?

**Default**: strict. Every `TensorParameter` enforces exact `TensorSpec` match. **Don't deviate during port.**

Tensor-arg relaxations are a deliberate opt-in: the kernel must *actually* tolerate the relaxation, and lying about kernel tolerance is a silent correctness bug. The bias-of-mistakes flips against the porter here: forgetting to opt-in is safe (narrower cache equivalence, slower but correct); declaring incorrectly is a wrong-answer.

A port is not the context to make this call. Keep all `TensorParameter::relaxations` default-constructed. If you notice that the kernel *would* tolerate a relaxation (e.g., padding-only dimension differences), capture the observation in the port report under "Open items for downstream" — don't bake it into the port itself.

The relaxation infrastructure exists in `TensorParameter::relaxations` and is wired through the per-dispatch legality check, but the TTNN auto-hasher integration that makes relaxations effective for cache breadth is still being implemented. Strict is the right choice during the port window.

### 4. Heavy immutable-extraction work?

Almost certainly no — and if yes, the answer is still **stop the audit**.

The Advanced concepts (`AdvancedProgramSpecFactoryConcept`, `AdvancedWorkloadSpecFactoryConcept`) split factory work into three methods — `extract_immutable_info` / `create_program_spec` / `create_program_run_args` — so that immutable extraction runs cheaply on every dispatch while spec construction runs miss-only. **The Advanced concepts are method-presence stubs in the framework today; the adapter `static_assert`s "not yet supported."** An op that genuinely needs Advanced is blocked on framework implementation.

If the legacy op does substantial immutable-extraction work that the basic shape would re-run every dispatch, **downgrade the audit to RED** and record the case in the audit report's TTNN ProgramFactory section: which extraction work, why the basic-shape per-dispatch re-run is unacceptable. This feeds prioritization of the Advanced implementation; it is not a porter-resolvable situation, so no port should be initiated.

---

## Concept summary

The four leaves of `Metal2FactoryConcept`. Exactly one factory class satisfies exactly one of these.

| Concept | Method | Returns | Programs | Op-owned resources | Status |
|---|---|---|---|---|---|
| `ProgramSpecFactoryConcept` | `create_program_artifacts` | `ProgramArtifacts` | One | MeshTensors | **Implemented** |
| `WorkloadSpecFactoryConcept` | `create_workload_artifacts` | `MeshWorkloadArtifacts` | Per-coord | MeshTensors + GlobalSemaphores | **Implemented** |
| `AdvancedProgramSpecFactoryConcept` | `extract_immutable_info` + `create_program_spec` + `create_program_run_args` | (multi) | One | MeshTensors | Stubbed — not yet usable |
| `AdvancedWorkloadSpecFactoryConcept` | (workload variants) | (multi) | Per-coord | MeshTensors + GlobalSemaphores | Stubbed — not yet usable |

Reaching the right leaf is mechanical from the decision tree above. If your port lands on a stubbed concept, see Decision 4.

### Method signatures for the implemented leaves

`ProgramSpecFactoryConcept`:

```cpp
static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
    const operation_attributes_t&  attributes,
    const tensor_args_t&            tensor_args,
    tensor_return_value_t&          tensor_return_value);
```

`WorkloadSpecFactoryConcept`:

```cpp
static ttnn::device_operation::MeshWorkloadArtifacts create_workload_artifacts(
    const operation_attributes_t&        attributes,
    const tensor_args_t&                  tensor_args,
    tensor_return_value_t&                tensor_return_value,
    const ttnn::MeshCoordinateRangeSet&   tensor_coords);
```

For the field layouts of `ProgramArtifacts` and `MeshWorkloadArtifacts`, see [`ttnn/api/ttnn/metal2_artifacts.hpp`](https://github.com/tenstorrent/tt-metal/blob/akertesz/ttnn-metal2concept-improvements/ttnn/api/ttnn/metal2_artifacts.hpp). For the conceptual shape — what `ProgramSpec` and `ProgramRunArgs` contain, how `MeshTensor` flows through `TensorArgument`, the cache miss/hit lifecycle — see [migration guide — TTNN Framework Integration](metal2_migration_guide.md#ttnn-framework-integration).

---

## Today's implementation status

Snapshot date: 2026-06-08. Branch: `akertesz/ttnn-metal2concept-improvements`.

**Implemented and usable for porting:**
- `ProgramSpecFactoryConcept` — basic single-program, with optional `op_owned_tensors` (MeshTensors only).
- `WorkloadSpecFactoryConcept` — basic multi-program, with optional `op_owned_tensors` and `global_semaphores`.
- Default caching strategy `MaximizeCacheReuse`; opt-in `MinimizeCacheHitCost` via `static constexpr auto caching_strategy = ProgramCachingStrategy::MinimizeCacheHitCost`.
- Forbidden cell enforcement: non-empty op-owned resources without the `MinimizeCacheHitCost` opt-in produces a runtime `TT_FATAL` at dispatch.
- Strict tensor-arg matching (default; relaxations infrastructure partial).

**Stubbed — not yet usable:**
- `AdvancedProgramSpecFactoryConcept` and `AdvancedWorkloadSpecFactoryConcept`. Concepts compile (method-presence shape check), but the framework adapter `static_assert`s "not yet supported." Method names on the workload variant are deliberately placeholder-distinct from the single-program variant pending a final-naming pass.

**Partial — keep strict during port:**
- Tensor-arg relaxations: `TensorParameter::relaxations` field exists and the per-dispatch legality check respects it; the TTNN auto-hasher integration that makes relaxations broaden cache equivalence is still in progress. Use defaults during port; capture relaxation candidates in the report.

If you find yourself blocked by something that should be in the "implemented" column but isn't working, surface it — the framework is actively maintained.

---

## Audit report deliverable

The auditor adds the following section to `METAL2_PREPORT_AUDIT.md`. This is the primary artifact of the factory selection step — the decision is recorded here, and the port inherits it.

```markdown
## TTNN ProgramFactory

### Concept chosen
[One of: ProgramSpecFactoryConcept / WorkloadSpecFactoryConcept / AdvancedProgramSpecFactoryConcept / AdvancedWorkloadSpecFactoryConcept]

### Path through the decision tree
- Decision 1 (single vs multi-program): [answer; what observable in legacy made this clear]
- Decision 2 (op-owned resources): [answer; what resources, if any]
- Decision 3 (tensor-arg relaxations): strict [default; deviation from this requires a paragraph explaining why]
- Decision 4 (advanced shape): not needed [or: STOP — Advanced required, audit RED]

### Caching strategy
[MaximizeCacheReuse (default) — or — MinimizeCacheHitCost (required because op-owned resources are present)]

### Legacy-to-Metal-2.0 shape change
[Does the chosen concept match the legacy factory shape, or does it unwind a legacy workaround?
Common case: "1:1 with legacy."
Special case: "legacy was MeshWorkload only to carry op-owned tensors; port should target single-program ProgramSpecFactoryConcept with op_owned_tensors per the TTNN factory doc's single-program-with-resources heads-up."]

### Stop signals
[If Decision 4 fired (Advanced required): explain which extraction work, why the basic-shape per-dispatch re-run is unacceptable, and confirm the overall audit result has been downgraded to RED.
Otherwise: "None."]
```

If a stop signal fires, the audit's overall result downgrades — the port can't proceed against a stubbed concept.

## Port plan deliverable (porter-facing)

The porter inherits the audit's decision. The port plan's TTNN ProgramFactory section is a brief carry-forward, not a re-derivation:

```markdown
## TTNN ProgramFactory

### Concept (inherited from audit)
[Copy from METAL2_PREPORT_AUDIT.md — concept name + caching strategy.]

### Implementation notes
[Optional: anything specific to how this op will realize the chosen concept that's worth surfacing before construction. Most ports won't need this section.]
```

Do not re-walk the decision tree during planning. If you find yourself disagreeing with the audit's choice, **stop and surface the disagreement** — don't unilaterally override; the audit is the source of truth for the chosen concept, and an in-port revision is a signal the audit was incomplete.

## Port report deliverable (porter-facing)

The porter adds the following to `METAL2_PORT_REPORT.md` at the end of the port. The audit chose; the report confirms the realized choice and surfaces any friction.

```markdown
## TTNN ProgramFactory

### Concept realized
[Confirm: matches audit's choice, or — if it changed — explain why and confirm the discrepancy was surfaced with the invoker before re-deciding.]

### Escalations applied
[For each non-default axis the realized port reached for:
- "Op-owned MeshTensors: <which tensors, why>"
- "Op-owned GlobalSemaphores (workload only): <why>"
- "Multi-program: <why per-coord program shapes differ>"
- "MinimizeCacheHitCost caching strategy: <required because op-owned resources>"
If the port stayed entirely on the default, write "None — default ProgramSpecFactoryConcept, no op-owned resources, strict tensor matching, default MaximizeCacheReuse caching."]

### Open items
[Anything you noticed about factory concepts during the port:
- Relaxation candidates (kernels that would tolerate relaxed tensor matching — not applied during port).
- Reasons the port would benefit from an Advanced concept if available.
- Anything the decision tree didn't anticipate.
- Friction with the concept selection process itself.]
```

If the port stayed entirely on the default, this section is small — that's the success case, not a failure to fill the template.

---

## Cross-references

- [Audit doc](port_op_to_metal2_audit.md) — the feasibility audit that invokes this document as its final step.
- [Port recipe](port_op_to_metal2_recipe.md) — the port workflow that inherits this document's decision.
- [Migration guide — TTNN Framework Integration](metal2_migration_guide.md#ttnn-framework-integration) — factory skeleton, `MeshTensor` extraction style, cache miss/hit lifecycle reference.
- [`ttnn/api/ttnn/operation_concepts.hpp`](https://github.com/tenstorrent/tt-metal/blob/akertesz/ttnn-metal2concept-improvements/ttnn/api/ttnn/operation_concepts.hpp) — concept definitions in code.
- [`ttnn/api/ttnn/metal2_artifacts.hpp`](https://github.com/tenstorrent/tt-metal/blob/akertesz/ttnn-metal2concept-improvements/ttnn/api/ttnn/metal2_artifacts.hpp) — `ProgramArtifacts` and `MeshWorkloadArtifacts` field layouts.
