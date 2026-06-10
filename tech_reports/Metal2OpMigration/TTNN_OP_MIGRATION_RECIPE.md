# Porting a TTNN op to Metal 2.0 — the factory recipe

> Companion to the Metal 2.0 documentation set (`metal2_migration_guide.md`, `port_op_to_metal2_recipe.md`,
> `port_op_to_metal2_ttnn_factory.md` on the documentation branch). Those docs own *how to build* a
> `ProgramSpec` + `ProgramRunArgs` (kernels, DFBs, named args, tensor parameters). **This document owns
> the TTNN factory shape decision**: which of the four factory concepts the op lands on, and the rule for
> deciding — *measure dispatch time and compare, never guess*.

## Read this first

A Metal 2.0 op factory produces exactly two things: the immutable **`ProgramSpec`** (the blueprint) and
the **`ProgramRunArgs`** (the per-execution values). The four factory concepts below are points on a
**progressive ladder**. You climb only as far as **measured** performance forces you — each rung adds
authoring work in exchange for a cheaper cache-hit dispatch, and on most ops the cheaper dispatch does
not matter.

| rung | concept | factory methods | cache key | cache-hit work |
|---|---|---|---|---|
| **1 (default)** | `ProgramSpecFactoryConcept` | `create_program_spec` → `{spec, run_args}` | spec | factory re-runs; tensors refreshed (`UpdateTensorArgs`) |
| **2** | `AdvancedProgramSpecFactoryConcept` | + `create_enqueue_invariant_args`, `create_per_enqueue_args` | spec | only per-enqueue args rebuilt + re-applied (`UpdateProgramRunArgs`) |
| **3** | `ImmutableProgramSpecFactoryConcept` | + `extract_immutable_info` | ImmutableInfo | no spec rebuild; tensors refreshed |
| **4** | `AdvancedImmutableProgramSpecFactoryConcept` | both | ImmutableInfo | no spec rebuild; only per-enqueue args re-applied |

The concepts are detected by the factory's method surface (`extract_immutable_info` ⇒ immutable-keyed;
`create_enqueue_invariant_args` + `create_per_enqueue_args` ⇒ split). The method names are the
documentation: a reader of the factory knows what is fixed per cache entry and what changes per dispatch.

**No `select_program_factory`** for single-factory ops — the framework auto-selects a single-alternative
`program_factory_t`. **No `compute_program_hash`** for immutable-keyed ops — `extract_immutable_info` is
the key.

---

## Step 0 — Port to rung 1 (everyone starts here)

Any dev should be able to do this without thinking about caching at all.

1. Build the `ProgramSpec` and the full `ProgramRunArgs` exactly as the legacy factory built its
   descriptor (per `port_op_to_metal2_recipe.md`: CBs → DFBs, positional args → named args, tensor
   addresses → `TensorParameter` + binding). Keep kernels faithful — change *only* binding mechanisms
   (`dfb::`, `ta::`, `args::`), never logic.
2. Return both, bundled:

```cpp
struct MyOpProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_spec(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args, tensor_return_value_t& out);
};
using program_factory_t = std::variant<MyOpProgramFactory>;
```

3. Delete a custom `compute_program_hash` if the op has one (sanctioned device-op edit; the framework
   hash is the cache key at rungs 1–2). Delete pybind hooks for vanished legacy entry points.
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

Climbing pays only if the workload **cache-hits frequently** (enqueue loops). A cold-path or
shape-churning op never repays factory complexity. Indicative magnitudes from the rand/matmul case
studies (WH B0, ~64-core shapes): rung 1 hit ≈ 130–160µs (full factory re-run + full re-apply); rung
2/4 hit ≈ 25–40µs (≈ descriptor parity at best); rung 3 only skips the spec rebuild. Your op will
differ — **measure**.

## Step 2 — Rung 2 if the hit cost hurts: split the run-args

Split run-args by one question: *can the value differ between two dispatches that share a cache entry?*

- **No** → `create_enqueue_invariant_args` (work splits, shape scalars). Declare each invariant in the
  spec: `KernelAdvancedOptions::enqueue_invariant_runtime_args` / `_common_runtime_args`,
  `TensorParameterAdvancedOptions::enqueue_invariant`. Applied once on miss, retained.
- **Yes** → `create_per_enqueue_args` (tensor addresses, RNG seed, fill value). Rebuilt every dispatch.
  Takes the mesh coordinate (per-device values, e.g. a sharded-mesh seed offset).

Miss merges both (`MergeProgramRunArgs`) + `SetProgramRunArgs`; hit re-applies only the per-enqueue set
(`UpdateProgramRunArgs`). The runtime validates that every omitted arg was declared invariant — a
forgotten per-enqueue value is a hard error, not stale "random" data. **Measure again.** If hit time
hit your budget, stop.

## Step 3 — Rungs 3–4 if the spec rebuild still dominates: ImmutableInfo

If profiling shows the remaining hit cost is `create_program_spec` itself, add the immutable-keyed shape:

```cpp
struct immutable_info_t {
    TensorSpec output_spec;
    CoreCoord  grid;   // everything the spec depends on — and ONLY that (no seed/from/to)
    static constexpr auto attribute_names = std::forward_as_tuple("output_spec", "grid");
    auto attribute_values() const { return std::forward_as_tuple(output_spec, grid); }
};
static immutable_info_t extract_immutable_info(const operation_attributes_t&, const tensor_args_t&);
static ProgramSpec      create_program_spec(const immutable_info_t&);              // miss-only
static ProgramRunArgs   create_enqueue_invariant_args(const immutable_info_t&);    // miss-only
static ProgramRunArgs   create_per_enqueue_args(attrs, tensor_args, out, coord);   // every dispatch
```

ImmutableInfo is the cache key **and** the sole input to the spec/invariant builders — a mutable value
structurally cannot leak into the key or the spec. Worked example: `rand`
(`ttnn/cpp/ttnn/operations/rand/device/`), validated by `test_rand_different_seed_values` (one cache
entry across seeds; different seed ⇒ different output). **Measure once more** and record before/after
in the port report. If rung 4 still misses budget, the bottleneck is elsewhere — profile, don't tune.

## Blocked cases

Multi-program / per-coord-varying ops (CCL-style), op-owned device resources (scratch `MeshTensor`s,
`GlobalSemaphore`s), per-device seed offsets needing per-coord programs: park (the workload-factory
family is pending) and record in the audit.

## Port report (append to METAL2_PORT_REPORT.md)

```markdown
## TTNN Factory
- Rung & concept: [1–4] [name]
- Measurements (median/p10/p90, ≥1000 iters): legacy = __µs; rung 1 = __µs; final rung = __µs
- Climb justification: [link table — "no climb needed" is a fine answer]
- Device-op edits: [custom hash deleted; select_program_factory removed; pybind hooks removed]
- Open items: [blockers, relaxation candidates]
```
