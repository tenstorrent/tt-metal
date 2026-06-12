# Porting a TTNN op to MetalV2 — the factory recipe

> Companion to the MetalV2 documentation set (`metal2_migration_guide.md`, `port_op_to_metal2_recipe.md`,
> `port_op_to_metal2_ttnn_factory.md`). Those docs own *how to build* a `ProgramSpec` + `ProgramRunArgs`
> (kernels, DFBs, named args, tensor parameters). **This document owns the TTNN factory shape decision**:
> which factory concept the op lands on, and the rule for deciding — *measure dispatch time and compare,
> never guess*.

## Read this first

A MetalV2 op factory produces a **`ProgramArtifacts`**: the immutable `ProgramSpec` (the blueprint) plus
its `ProgramRunArgs` (the per-execution values). There are exactly **TWO concepts**, distinguished by ONE
thing — the cache key:

| concept | extra method | cache key | buys you |
|---|---|---|---|
| **`ProgramSpecFactoryConcept`** | — | default reflection hash of (op type + attributes + tensor args) | the default |
| **`AdvancedProgramSpecFactoryConcept`** | `extract_immutable_info` | a small hashable `ImmutableInfo` | skips the spec rebuild on a hit; structurally excludes mutable values (e.g. a seed) from the key |

Every MetalV2 factory writes **two** methods (in either concept):

- **`create_program_spec`** → `ProgramArtifacts` = the `ProgramSpec` plus its run-args split by cadence:
  `invariant_run_args` (enqueue-invariant — set once on a cache miss, retained across hits) and `run_args`
  (the per-enqueue set for the miss dispatch). Cache-miss only.
- **`create_per_enqueue_args`** → `std::optional<ProgramRunArgs>` — the per-enqueue set, rebuilt each
  dispatch (takes the mesh coordinate, so per-device values like a sharded-mesh seed offset can vary).
  **Mandatory**, but return `std::nullopt` to opt out.

The split is the default, not an opt-in: requiring `create_per_enqueue_args` forces you to decide, per op,
what is enqueue-invariant vs per-enqueue. What it buys on a **cache hit**:

| `create_per_enqueue_args` returns | when | cache-hit work |
|---|---|---|
| **a `ProgramRunArgs`** | a value changes per dispatch (seed, fill value, tensor addresses) | re-apply only that set (`UpdateProgramRunArgs`) — invariant set stays baked, no factory re-run |
| **`std::nullopt`** | nothing per-enqueue beyond which tensors are bound | refresh tensor bindings only (`UpdateTensorArgs`) |

The metal runtime validates that every arg **omitted** from the per-enqueue set was declared
`enqueue_invariant` in the spec — a forgotten per-enqueue value is a hard error, not silently-stale data.
Move only **enough** args into `invariant_run_args` to clear your perf target; leaving some in
`create_per_enqueue_args` is fine.

Detected by method surface: `extract_immutable_info` ⇒ Advanced (immutable-keyed). **No
`select_program_factory`** for single-factory ops (auto-selected). **No `compute_program_hash` on any
MetalV2 op** — the `ProgramSpec` (spec-keyed) or `ImmutableInfo` (Advanced) is the key; the framework
`static_assert`s against a custom hash.

> ### 🚪👹 Never write a custom hash. The boogeyman will eat you.
>
> A MetalV2 op **must not** define a custom `compute_program_hash`, and must not hand-roll the key by any
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

> ### All variants or none — no mixing, no try
>
> If an op's `program_factory_t` has several variants, you migrate **all** of them to MetalV2 or **none**.
> The framework rejects (`static_assert`) a `program_factory_t` that mixes MetalV2 spec factories with
> prior-framework factories. So a hash-carrying multi-variant op (e.g. matmul, whose one
> `compute_program_hash` serves every variant) migrates as a unit: port every factory **and delete the
> shared hash** — the default key (or an `immutable_info_t` per variant) replaces it. Half-migrating to
> dodge the hash is exactly the "try" this forbids.

---

## Step 0 — Port to the base `ProgramSpecFactoryConcept` (everyone starts here)

Do this without thinking about caching at all.

1. Build the `ProgramSpec` and the full `ProgramRunArgs` exactly as the legacy factory built its
   descriptor (per `port_op_to_metal2_recipe.md`: CBs → DFBs, positional args → named args, tensor
   addresses → `TensorParameter` + binding). Keep kernels faithful — change *only* binding mechanisms
   (`dfb::`, `ta::`, `args::`), never logic.
2. Write the two methods. Put everything the spec needs into `create_program_spec`; if you can't yet tell
   what's per-enqueue, return `std::nullopt` from `create_per_enqueue_args` — the simplest valid op (cache
   hit just refreshes tensor bindings). Per-call tensors go in `run_args`, not `invariant_run_args`:

```cpp
struct MyOpProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_spec(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args, tensor_return_value_t& out);
    static std::optional<tt::tt_metal::experimental::ProgramRunArgs> create_per_enqueue_args(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args, tensor_return_value_t& out,
        const std::optional<ttnn::MeshCoordinate>& coord);  // std::nullopt to opt out
};
using program_factory_t = std::variant<MyOpProgramFactory>;
```

3. Delete any custom `compute_program_hash` — mandatory, the framework forbids it (the key is the spec).
   Migrate **every** variant of a multi-variant op here (no mixing). Delete pybind hooks for vanished
   legacy entry points.
4. Validate correctness (op's test suite + cache regression if it has one). **Done. Most ops stop here**
   (`create_per_enqueue_args` returning `std::nullopt`).

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

The mandatory split already makes every **hit** cheap — the hit path re-applies the per-enqueue set and
never re-runs `create_program_spec`. So the remaining question is whether you *get* hits: a spec-keyed op
whose cache key contains a per-dispatch value (a seed) **misses every call** and pays the full miss cost
forever. That — not hit latency — is what climbing to Advanced fixes. Indicative magnitudes (WH B0,
~64-core shapes): hit ≈ 25–44µs; miss (factory + spec realize) ≈ 130–160µs, or up to ~1s if it triggers a
kernel JIT recompile. A cold-path or shape-churning op that never cache-hits won't repay an
`immutable_info` key. Your op will differ — **measure**.

## Step 2 — Decide the split (which args are per-enqueue)

`create_per_enqueue_args` is mandatory, so this is the one design call every op makes — not an optional
climb. Split by one question: *can the value differ between two dispatches that share a cache entry?*

- **No** (work splits, shape scalars) → put it in `create_program_spec`'s `invariant_run_args` and declare
  it invariant in the spec (`KernelAdvancedOptions::enqueue_invariant_runtime_args` / `_common_runtime_args`,
  `TensorParameterAdvancedOptions::enqueue_invariant`). Set once on miss, retained across hits.
- **Yes** (tensor addresses, RNG seed, fill value) → return it from `create_per_enqueue_args`. Rebuilt
  every dispatch; takes the mesh coordinate (per-device values, e.g. a sharded-mesh seed offset). If
  *nothing* qualifies, return `std::nullopt` — the hit just refreshes tensor bindings.

Miss merges both (`MergeProgramRunArgs`) + `SetProgramRunArgs`; hit re-applies only the per-enqueue set
(`UpdateProgramRunArgs`), or refreshes tensor bindings when you returned `std::nullopt`. The runtime
validates that every omitted arg was declared invariant — a forgotten per-enqueue value is a hard error,
not stale "random" data. Move only ENOUGH into `invariant_run_args` to clear your perf target — leaving
some in `create_per_enqueue_args` is fine. Measure; stop when you hit budget.

## Step 3 — `AdvancedProgramSpecFactoryConcept`: switch the key to ImmutableInfo

Use this for **one** reason: a per-dispatch value is polluting the default cache key, so the op misses
every call (the sampling/​rand seed cliff above). It does **not** reduce hit latency — the hit never
rebuilds the spec regardless — it makes the op *cache-hit at all*. Add `extract_immutable_info` and make
`create_program_spec` take it:

```cpp
struct immutable_info_t {
    TensorSpec output_spec;
    CoreCoord  grid;   // everything the spec depends on — and ONLY that (no seed/from/to)
    static constexpr auto attribute_names = std::forward_as_tuple("output_spec", "grid");
    auto attribute_values() const { return std::forward_as_tuple(output_spec, grid); }
};
static immutable_info_t              extract_immutable_info(const operation_attributes_t&, const tensor_args_t&);
static ProgramArtifacts              create_program_spec(const immutable_info_t&);  // miss-only; spec + invariant_run_args
static std::optional<ProgramRunArgs> create_per_enqueue_args(attrs, tensor_args, out, coord);  // mandatory; seed lives here
```

ImmutableInfo is the cache key **and** the sole input to `create_program_spec` — a mutable value
structurally cannot leak into the key or the spec. A consequence worth expecting: the builder *cannot see*
the per-dispatch value (e.g. the seed), so that value rides `create_per_enqueue_args`, and the
`ProgramArtifacts::run_args` returned by `create_program_spec` is typically **empty** for such an op (its
per-enqueue set comes entirely from `create_per_enqueue_args`). Worked example: `rand`
(`ttnn/cpp/ttnn/operations/rand/device/`), validated by `test_rand_different_seed_values` (one cache
entry across seeds; different seed ⇒ different output). **Measure once more** and record before/after in
the port report. If it still misses budget, the bottleneck is elsewhere — profile, don't tune.

> **Note — which path each reference exercises.** `rand` is the **Advanced+++** reference: it fills
> `spec` + `invariant_run_args` and supplies the dynamic set via `create_per_enqueue_args`, so its struct
> `run_args` is empty and the cache-hit goes through `UpdateProgramRunArgs`. The **base / opt-out** path —
> a spec-keyed op that fills the struct's `run_args` (tensors) and returns `std::nullopt`, so the hit
> refreshes bindings via `UpdateTensorArgs` — is the common case and is runtime-exercised as the spec-keyed
> ops migrate. (In the framework PR it is covered by the compile-time concept contracts in
> `test_launch_operation.cpp`.)

## Required test coverage

Correctness-vs-reference alone is **not** enough. The cache contract has its own failure modes — a stale
per-enqueue value, or a per-call value leaking into the key — and both are **silent**: they only surface
under a warm program cache, never on the first call. So every migrated op needs explicit cache regression
guards, with `device.enable_program_cache()` on. Model them on `test_rand.py`
(`test_rand_different_seed_values`, `test_rand_different_from_to_values`, `test_rand_program_cache_with_mesh_mapper`):

1. **Key stability** — vary a per-dispatch value (seed, fill value, from/to) across calls; assert
   `device.num_program_cache_entries()` stays **1**. Catches a per-call value leaking into the cache key
   (the recompile-every-call cliff).
2. **Hit re-application** — a *different* per-enqueue value must yield *different* output on a cache hit
   (proves the value is re-applied, not the first call's baked-in stale value); the *same* value must
   reproduce *identical* output (proves deterministic re-application). This is the freeze-bug guard.
3. **No-op stability** — repeating the identical call must not grow the cache.
4. **Mesh program cache** — run replicated and sharded across a mesh; assert the entry count is stable
   across repeated dispatches (guards the per-coordinate hit path).
5. **Correctness vs reference** across representative dtypes / layouts / shapes (the op's existing suite).

Tests 1–2 are the ones migrations keep regressing (PR #45350 hacked exactly this) — **mandatory, not
optional.** An op with a `create_per_enqueue_args` that returns a value MUST carry tests 1–2; an op that
opts out (`std::nullopt`) still needs 3–4 for its tensor-binding hit path.

## Gotchas (learned the hard way)

- **Unity build:** file-scope constants in sibling factory `.cpp`s collide even inside anonymous namespaces
  (one anon namespace per TU) — give them unique names.
- **Local DFB producer + consumer must share the same `WorkUnitSpec`.** A `KernelSpec` may belong to
  several; put an all-cores reader/writer into each per-core-group compute work unit.
- **A borrowed-output DFB needs exactly one producer + one consumer** — split reader = Producer,
  writer = Consumer (both still write the same `dfb::` id; the role only shapes the dep graph).
- **A kernel with any `TensorBinding` needs a `KernelRunArgs` entry** (even with zero scalar args), or
  `ValidateProgramRunArgs` aborts.
- **`UnpackToDestFp32` is rejected unless the input DFB is Float32** — guard it (legacy applied it as a
  tolerated no-op).
- Actually assign `spec.kernels` / `.dataflow_buffers` / `.tensor_parameters` / `.work_units` — easy to
  build locals and drop them.

## Blocked cases

- **Multi-program / per-coord-varying ops** (CCL-style): the workload-factory family is pending. Park.
- **Metal-side gaps (owned by Audrey, not ttnn)** — surface, don't work around: optional-feature kernels
  that reference a `dfb::`/`ta::` token under `if constexpr` (needs first-class kernel args); runtime /
  variadic / indexed tensor accessors; per-vararg `enqueue_invariant` (tree-reduction / mcast ops). An op
  blocked only by one of these is parked until the metal fix lands.

Record any parked op in the audit.

## Port report (append to METAL2_PORT_REPORT.md)

```markdown
## TTNN Factory
- Concept: ProgramSpecFactoryConcept | AdvancedProgramSpecFactoryConcept; create_per_enqueue_args: returns args | std::nullopt
- Measurements (median/p10/p90, ≥1000 iters, realistic varying inputs): legacy = __µs; final = __µs
- Climb justification: [why ImmutableInfo, or "default key is fine"]
- Cache tests: [key-stability + hit-re-application + mesh — see "Required test coverage"]
- Device-op edits: [custom hash deleted; select_program_factory removed; pybind hooks removed]
- Open items: [blockers, relaxation candidates]
```
