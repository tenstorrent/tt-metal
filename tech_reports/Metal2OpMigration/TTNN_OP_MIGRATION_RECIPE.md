# Porting a TTNN op to Metal 2.0 — the factory recipe

> Companion to the Metal 2.0 documentation set (`metal2_migration_guide.md`, `port_op_to_metal2_recipe.md`,
> `port_op_to_metal2_ttnn_factory.md`). Those docs own *how to build* a `ProgramSpec` + `ProgramRunArgs`
> (kernels, DFBs, named args, tensor parameters). **This document owns the TTNN factory shape decision**:
> which factory concept the op lands on, and the rule for deciding — *measure dispatch time and compare,
> never guess*.

## Read this first

A Metal 2.0 op factory produces a **`ProgramArtifacts`**: the immutable `ProgramSpec` (the blueprint) plus
its `ProgramRunArgs` (the per-execution values). There are exactly **TWO concepts**, distinguished by ONE
thing — the cache key:

| concept | extra method | cache key | buys you |
|---|---|---|---|
| **`ProgramSpecFactoryConcept`** | — | the generated `ProgramSpec` | the default |
| **`AdvancedProgramSpecFactoryConcept`** | `extract_immutable_info` | a small hashable `ImmutableInfo` | skips the spec rebuild on a hit; structurally excludes mutable values (e.g. a seed) from the key |

Within **either** concept, the enqueue-invariant fast path is an **optional, incremental** refinement —
add a `create_per_enqueue_args` method (Audrey's "Option 2 → 2++" / "Option 3 → 3++"):

| | `create_program_artifacts` returns | `create_per_enqueue_args`? | cache-hit work |
|---|---|---|---|
| **degenerate** | spec + **all** run-args | absent | refresh tensor bindings (`UpdateTensorArgs`) |
| **++** | spec + **enqueue-invariant** run-args | present (the per-dispatch rest) | re-apply only the per-enqueue set (`UpdateProgramRunArgs`) |

You only move **enough** args into the invariant bundle to clear your perf target — moving one more arg
from `create_per_enqueue_args` into `create_program_artifacts`'s run-args (and declaring it invariant in
the spec) is a one-line, behavior-preserving optimization. **Leaving some on the table is fine.**

Detected by method surface: `extract_immutable_info` ⇒ Advanced (immutable-keyed); `create_per_enqueue_args`
⇒ ++. **No `select_program_factory`** for single-factory ops (auto-selected). **No `compute_program_hash`**
for the Advanced concept — `extract_immutable_info` is the key.

> ### 🚪👹 Never write a custom hash. The boogeyman will eat you.
>
> A Metal 2.0 op **must not** define a custom `compute_program_hash`, and must not hand-roll the key by any
> other means — e.g. an `attribute_values()` that omits a field to "hide" it from the hash. Both are
> custom hashes wearing a disguise, and both let a mutable value silently corrupt the cache.
>
> **The legitimate need** — "I want value X (a seed, a runtime scalar) OUT of the cache key" — has exactly
> one sanctioned answer: **climb to `AdvancedProgramSpecFactoryConcept` and define an `immutable_info_t`
> struct that simply doesn't contain X.** Option 3 *bounds* you to do it the safe way: the struct you
> declare *is* the key, the builder receives *only* that struct, so X cannot leak into either the key or
> the spec — not by oversight, not ever. You get correctness by construction instead of by discipline.
>
> If you catch yourself reaching for a custom hash, stop and write the `immutable_info_t` instead.

---

## Step 0 — Port to the degenerate `ProgramSpecFactoryConcept` (everyone starts here)

Do this without thinking about caching at all.

1. Build the `ProgramSpec` and the full `ProgramRunArgs` exactly as the legacy factory built its
   descriptor (per `port_op_to_metal2_recipe.md`: CBs → DFBs, positional args → named args, tensor
   addresses → `TensorParameter` + binding). Keep kernels faithful — change *only* binding mechanisms
   (`dfb::`, `ta::`, `args::`), never logic.
2. Return both, bundled — one method, no split:

```cpp
struct MyOpProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args, tensor_return_value_t& out);
};
using program_factory_t = std::variant<MyOpProgramFactory>;
```

3. Delete a custom `compute_program_hash` if the op has one (sanctioned device-op edit; the framework
   hashes the spec). Delete pybind hooks for vanished legacy entry points.
4. Validate correctness (op's test suite + cache regression if it has one). **Done. Most ops stop here.**

## Step 1 — Measure before climbing (the gate, not a formality)

Climbing is justified by **one number**: warm cache-hit host dispatch time, before vs after, same shape,
same machine. The discipline (skipping it produces fiction; we have produced fiction):

- **Verify which binary you measure** (`print(ttnn.__file__)`; worktree `.pth` shadowing is real).
- **Wipe the kernel cache once**, warm ≥ 50 calls, then time ≥ 1000 calls; report median + p10/p90 (the
  dispatch distribution is bimodal; a mean alone lies).
- **Run a no-op control** to know your timer floor; measure host dispatch, not device time (migration
  shifts host cost only — kernels are unchanged).
- **Baseline = the legacy/descriptor path for the same shape**, measured the same way the same day.
- **Measure the realistic workload.** If a per-dispatch value (an RNG seed) is in the legacy cache key,
  the legacy op *recompiles every call* — measure with that value VARYING, not pinned. (Real example:
  legacy `sampling` with a pinned seed hit ≈ 33µs, but with a varying seed it recompiled every call at
  ≈ 1 *second* — the pinned number hid a 30000× cliff.)

Climbing pays only if the workload **cache-hits frequently** (enqueue loops). A cold-path or
shape-churning op never repays factory complexity. Indicative magnitudes from the rand/matmul case
studies (WH B0, ~64-core shapes): degenerate hit ≈ 130–160µs (full factory re-run + full re-apply); ++
hit ≈ 25–40µs. Your op will differ — **measure**.

## Step 2 — "++": split off the per-enqueue args (if the hit cost hurts)

Add `create_per_enqueue_args` and move the args that change per dispatch into it; leave everything else
in `create_program_artifacts`'s run-args and declare those enqueue-invariant in the spec. Split by one
question: *can the value differ between two dispatches that share a cache entry?*

- **No** (work splits, shape scalars) → keep in `create_program_artifacts`'s run-args; declare invariant
  (`KernelAdvancedOptions::enqueue_invariant_runtime_args` / `_common_runtime_args`,
  `TensorParameterAdvancedOptions::enqueue_invariant`). Set once on miss, retained.
- **Yes** (tensor addresses, RNG seed, fill value) → `create_per_enqueue_args`. Rebuilt every dispatch;
  takes the mesh coordinate (per-device values, e.g. a sharded-mesh seed offset).

Miss merges both (`MergeProgramRunArgs`) + `SetProgramRunArgs`; hit re-applies only the per-enqueue set
(`UpdateProgramRunArgs`). The runtime validates that every omitted arg was declared invariant — a
forgotten per-enqueue value is a hard error, not stale "random" data. **You need only move ENOUGH args
into the invariant bundle to clear your target — leaving some on the table is fine.** Measure again; stop
when you hit budget.

## Step 3 — `AdvancedProgramSpecFactoryConcept`: switch the key to ImmutableInfo

Use this when (a) the remaining hit cost is the `create_program_artifacts` rebuild itself, or (b) a
per-dispatch value is polluting the cache key (the sampling/​rand seed cliff above). Add
`extract_immutable_info` and make `create_program_artifacts` take it:

```cpp
struct immutable_info_t {
    TensorSpec output_spec;
    CoreCoord  grid;   // everything the spec depends on — and ONLY that (no seed/from/to)
    static constexpr auto attribute_names = std::forward_as_tuple("output_spec", "grid");
    auto attribute_values() const { return std::forward_as_tuple(output_spec, grid); }
};
static immutable_info_t      extract_immutable_info(const operation_attributes_t&, const tensor_args_t&);
static ProgramArtifacts      create_program_artifacts(const immutable_info_t&);     // miss-only; spec + invariant args
static ProgramRunArgs        create_per_enqueue_args(attrs, tensor_args, out, coord);  // optional "++"; every dispatch
```

ImmutableInfo is the cache key **and** the sole input to `create_program_artifacts` — a mutable value
structurally cannot leak into the key or the spec. Worked example: `rand`
(`ttnn/cpp/ttnn/operations/rand/device/`), validated by `test_rand_different_seed_values` (one cache
entry across seeds; different seed ⇒ different output). **Measure once more** and record before/after in
the port report. If it still misses budget, the bottleneck is elsewhere — profile, don't tune.

## Blocked cases

Multi-program / per-coord-varying ops (CCL-style); op-owned device resources beyond what enqueue-invariant
tensor args cover. Park (the workload-factory family is pending) and record in the audit.

## Port report (append to METAL2_PORT_REPORT.md)

```markdown
## TTNN Factory
- Concept: ProgramSpecFactoryConcept | AdvancedProgramSpecFactoryConcept; per-enqueue split: yes/no
- Measurements (median/p10/p90, ≥1000 iters, realistic varying inputs): legacy = __µs; degenerate = __µs; final = __µs
- Climb justification: [why this rung — "no climb needed" is a fine answer]
- Device-op edits: [custom hash deleted; select_program_factory removed; pybind hooks removed]
- Open items: [blockers, relaxation candidates]
```
