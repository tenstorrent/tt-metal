# Correct-by-Construction Per-Enqueue Run Args in Metal 2.0

**Status:** RFC + landed, hardware-verified reference migration (`rand` on Metal 2.0, `get_dynamic_runtime_args` deleted)
**Scope:** the Metal 2.0 device-op adapter (`MetalV2MeshWorkloadFactoryAdapter`) + Metal 2.0 op factories
**Author:** Diego Gomez

---

## TL;DR

A value that varies per dispatch but is deliberately kept out of the program-cache key (an RNG
`seed`, a `[from,to)` range) must be re-applied to the cached program on every cache hit, or it
stays frozen at whatever the first cache miss baked in — a silent wrong result that only appears on
the second dispatch. Metal 2.0 already has the right *shape* for this (immutable `ProgramSpec` = the
cache key; mutable `ProgramRunArgs` = re-specified per enqueue; `SetProgramRunArgs` validates every
named runtime arg is set) — **but the adapter's cache-hit path did not use it**: it re-applied only
tensor args (`UpdateTensorArgs`) and left the scalar kernel run args frozen. So the frozen-runtime-arg
bug was unsolved in Metal 2.0, not just in the descriptor shim.

This change adds one small, additive hook: if a Metal 2.0 factory declares `create_program_run_args`,
the adapter re-derives the `ProgramRunArgs` from the live `operation_attributes` and re-applies them
via `SetProgramRunArgs` on **every cache hit**. Completeness is enforced by construction — Metal 2.0's
own validation rejects a run-args set that doesn't specify every named runtime arg in the
`ProgramSpec`. No new binding type, no positional `(kernel_idx, arg_idx)`, nothing hand-mirrored.

**Verified on hardware:** `rand` is fully migrated to Metal 2.0 with this hook and its
`get_dynamic_runtime_args` deleted; the rand test suite passes, including
`test_rand_different_seed_values` and `test_rand_different_from_to_values` — the **changing-value**
cache-hit case (same shape, different seed/range → single cache entry → correct new output, not
frozen). See §5 for a measured perf comparison (currently a host-side regression — the naive re-apply
is a known optimization target).

## 1. The gap (unsolved in Metal 2.0)

The program cache keys a Metal 2.0 op on its `operation_attributes` (`compute_mesh_workload_hash`),
not on the `ProgramSpec`. So an op excludes a value from the key by omitting it from
`attribute_values()`; that obligates re-application every dispatch. On a cache miss the adapter builds
the program from the spec and applies the initial run args via `SetProgramRunArgs`. On a cache **hit**
the adapter previously did only `UpdateTensorArgs(program, fresh_tensor_args)` — the scalar kernel run
args (where a `seed`/range lives) were never re-applied. The mechanism to fix it existed; it just
wasn't wired into the hit path.

## 2. The fix

```cpp
// MetalV2MeshWorkloadFactoryAdapter::apply_descriptor (cache hit)
if constexpr (requires { MetalV2Factory::create_program_run_args(attrs, tensor_args, ret); }) {
    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        auto run_args = MetalV2Factory::create_program_run_args(attrs, tensor_args, ret);
        SetProgramRunArgs(program, run_args);   // re-applies scalars + tensors, validates completeness
    }
    return;
}
// else: existing tensor-only UpdateTensorArgs fast path (unchanged for ops that don't opt in)
```

`create_program_run_args` re-derives the run args from the **live** attributes, so a hit applies the
current value, not the frozen one. `SetProgramRunArgs` validates every named runtime arg in the
`ProgramSpec`'s `RuntimeArgSchema` is present — a missing one is a hard error, not a silent freeze.
That is the correct-by-construction property, and it is Metal 2.0's own, not a bolt-on.

A second, smaller adapter fix was needed for input-less ops: the adapter sourced the `MeshDevice`
only from `tensor_args`. `rand` has no input tensor, so it now falls back to the output tensor
(`tensor_return_value`) and then to a `MeshDevice*` in `operation_attributes`.

## 3. Reference migration: `rand`

`rand` is the natural proof — it excludes `seed`/`from`/`to` from the key and must re-apply them per
dispatch. Migration (following Audrey's op-porting recipe):

- **Kernels forked to Metal 2.0** (`rand/device/kernels/compute_rand.cpp`, `writer_rand.cpp`):
  named runtime args (`get_arg(args::seed)`), DFB accessor (`DataflowBuffer(dfb::cb_intermed)`),
  tensor accessor (`TensorAccessor(tensor::output)`). Forked (not editing the shared `uniform`
  kernels) so the legacy `uniform` op is untouched. The writer drops the legacy bf16 conversion
  staging CB (rand is always FLOAT32), which also removes the single-ended-FIFO shape the M2 lowering
  rejects on a DM kernel.
- **`RandProgramFactory`** with `create_program_artifacts` (ProgramSpec: one Float32 intermed DFB,
  compute PRODUCER + writer CONSUMER, output `TensorParameter`) and `create_program_run_args`
  (seed/from/to/start_id/num_tiles as named per-node run args).
- **`RandDeviceOperation`** restructured to the `program_factory_t` variant + `select_program_factory`
  form; `create_descriptor` and `get_dynamic_runtime_args` **deleted**. `attribute_values()` still
  omits seed/from/to (key excludes them).

**Tests (hardware):** the existing `test_rand_different_seed_values` / `test_rand_different_from_to_values`
pass — different seed / range, same shape → single cache entry (hit) → correct, changed output. This
is the changing-value case, on a real op.

**Known limitation:** the M2 adapter stamps one program's run args across the mesh, so per-**device**
unique seeding (sharded mesh) is not yet supported on this path (needs per-coord run args). Single-
device / replicated seeding is correct.

## 4. What landed

- `mesh_device_operation_adapter.hpp` — the `create_program_run_args` opt-in hook + input-less device
  sourcing.
- `rand/device/*` — the full Metal 2.0 migration (kernels, factory, device op), `get_dynamic_runtime_args`
  deleted.

## 5. Performance (measured, honest)

Same `benchmark_rand_perf.py` command on both revisions (single device, float32, tile, DRAM;
5 warmup, 100 hit iters, 3 repeats, median):

| shape | miss before → after | hit before → after |
|---|---|---|
| 1024×1024 | 469 → 881 µs (**+88%**) | 80 → 146 µs (**+83%**) |
| 4096×4096 | 907 → 1217 µs (**+34%**) | 588 → 600 µs (**+2%**) |

**This migration is currently a host-side regression.** On every cache hit, `create_program_run_args`
rebuilds the full run args (work split + per-core string-keyed `Table<std::string,uint32_t>` for all
cores) and `SetProgramRunArgs` re-applies *everything* with validation — versus the descriptor path's
targeted patch of just the changed scalar slots. It is host-bound, so it hurts most on small shapes
(1024², host-dominated) and washes out at 4096² hit (+2%, device-dominated).

The regression is a property of the *naive* re-apply, not of the correctness model. Closing it:
- **Re-apply only the changing args** via `UpdateProgramRunArgs` + `enqueue_invariant` advanced
  options (mark start_id/num_tiles/tensor invariant; re-apply only seed/from/to) instead of a full
  `SetProgramRunArgs`.
- **Avoid rebuilding the full run args** on hit — factor out just the changing scalars; avoid the
  per-node string-keyed tables on the hot path.

## 6. Follow-ups

- The perf optimization above (the priority before this pattern scales to more ops).
- Per-coord run args on the M2 adapter → per-device unique seeding for sharded mesh.
- Op-owned tensors on the per-enqueue path (the reference handles io tensors only).
- Migrate the remaining excluded-scalar ops (uniform, bernoulli, dropout, …) once the perf path is in.
