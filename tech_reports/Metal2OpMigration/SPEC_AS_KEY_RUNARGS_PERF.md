# Spec-as-key run-args perf: builders, profiling, and the output-allocation lever

Status: WIP exploration branch `dgomez/metal2-spec-runargs-builders`. Captures the spec-as-key
host-dispatch perf work — what was built, how it was measured, what was found, and what's next.

## Background: which path we're optimizing

Quasar (Metal 2.0) ops can sit on three host-side concepts:

| concept | factory entry | cache key | cache-hit cost |
|---|---|---|---|
| `ProgramDescriptorFactoryConcept` | `create_descriptor` | attribute hash | cheap (legacy-like) |
| `MetalV2FactoryConcept` | `create_program_artifacts` | attribute hash | cheap — `UpdateTensorArgs` on hit, **no spec rebuild** |
| **`ProgramSpecFactoryConcept` / `…WithOwnedTensorsConcept`** | `create_program_spec` (+ owned-tensor span) | **the ProgramSpec content** | **rebuild + re-hash the whole spec every dispatch** ("build-to-hash") |

This work targets the **spec-as-key** path (`ProgramSpecFactoryWithOwnedTensorsConcept`), defined on the
`#47473` base. It is intentionally more expensive than MetalV2/legacy: the spec is rebuilt every dispatch
so its content can be hashed into the cache key. The question is how cheap we can make that rebuild while
keeping the spec-as-key contract (the cache key stays the spec; nothing else).

## Measurement methodology (locked)

- Op vs its legacy twin, **same shape**, on hardware.
- **Fixed condition: DIM 1024, `TT_METAL_INSPECTOR=0`.** (The Inspector debug recorder does a synchronous
  per-dispatch file `write()`/`flush()` ~5 µs that inflates both paths; off = production-like.)
- Warm the program cache, then time N iters in batches; report **min/median µs/op** (min ≈ least-contended
  host work). Beware bimodal noise — single-rep numbers lie.
- **Verify the binary before every measurement:** `_ttnncpp.so` timestamp is fresh AND
  `nm -C _ttnncpp.so | grep <new symbol>` > 0. Do **not** trust the build exit code — a `build_metal.sh`
  on a tests-configured tree can exit while ninja stopped on a broken test *before relinking the libs*,
  leaving a stale `.so`. (This cost real time here: several "no change" measurements were a stale binary.)

## The helpers (`ttnn/api/ttnn/spec_run_args.hpp`)

Goal: give op authors spec-building tools that are easier to write **and** faster than hand-rolling, so
not using them is the wrong choice. Each helper bakes in an invariant the general API can't assume.

- `Table::append_unchecked` (metal) — append without the `find()` dup-scan; O(N) fill instead of O(N²).
  Safe because schema names are unique by construction.
- `ttnn::spec::KernelRunArgsBuilder` — one kernel's per-node run-args: declare names once, `emit(node,
  v0, v1, …)` positional + arity-checked, fills each Table via `append_unchecked`.
- `ttnn::spec::ProgramRunArgsBuilder` — whole program: `kernel()` per kernel, `emit()` per node,
  `take()` **moves** every kernel into the `ProgramRunArgs`. This makes the
  `kernel_run_args = {std::move(a), b, c}` bug impossible — `std::initializer_list` elements are `const`,
  so that braced-init silently **copies** every per-node Table. (`take()` + `push_back(std::move)` move.)

## Results (DIM 1024, inspector off, PCC 1.0)

| op | spec-as-key baseline | + run-args builder | legacy |
|---|---|---|---|
| **transpose** (WH) | ~41 µs (1.56×) | **~35 µs (1.34×)** | ~26 µs |
| **interleaved_to_sharded** | **~48 µs (~2.0×)** (builder not yet wired) | — | ~24 µs |

The builder is a real **−6 µs / −15%** on transpose. i2s was migrated (concept generalizes cleanly, PCC
1.0) but its branchy per-core loop (DRAM writer 7 args vs L1 writer 1 arg) hasn't had the builder wired,
so it shows the spec-as-key cost, not yet the builder win.

## Profiling — where the remaining gap is

**transpose, ~9 µs over legacy** (frames present in quasar, absent in legacy):
- per-dispatch spec **rebuild** (`create_program_spec`: KernelSpec ctors, DFB/CB specs, tensor params) ~4–5 µs
- spec **hash** for the key (`program_spec_cache_key` → reflection `hash_objects`) ~1.1 µs
- run-args build (now ~parity with legacy's `GetRuntimeArgs`)
- `std::filesystem::path` (kernel `source`) construct + hash ~0.8 µs
- `~ProgramSpec`/`~KernelSpec` teardown ~0.7 µs

**i2s, ~48 µs:**
- output tensor **alloc + free** (`MeshBuffer`/`Buffer` create+dtor, `MeshTensorHolder` dealloc) ~45% — **shared with legacy** (the floor)
- **`CoreRangeSet::merge` + `_Rb_tree<CoreRange>` ~3 µs — i2s-specific**: rebuilds+sorts the shard core ranges every dispatch
- spec hash ~3 µs; run-args/KernelSpec build ~2.4 µs (builder not wired); `compute_output_specs` ~1.4 µs

## Findings / dead ends (don't redo)

- **skip-validate-on-hit is not an optimization** — it's a parameter (`SetProgramRunArgs(..., skip_validation=true)`),
  already wired in the owned adapter's `apply_run_args`. Confirmed on the hot path.
- **Flat `Table<RtaName,size_t>` slot map regresses at high RTA count.** Sweep: O(N²) linear scan beats the
  hash map only below ~4–8 RTAs (N=2→1.4 µs, N=16→16 µs, N=32→61 µs). Reverted. The by-slot run-args
  *eliminates* the lookup instead — but the run-args copy turned out to be wall-clock-cheap anyway.
- **Run-args memoization is a frozen-args footgun.** Caching per-core run-args keyed on an author-defined
  invariance key is exactly the staleness bug class that's caused regressions (work-core-set-change). Not
  shipped as a general tool. The *safe* version would key on the spec — which needs a framework change.

## The principle, and the next lever (output allocation)

**Spec-derived data can be safely memoized on the spec hash; not-spec-derived data (run-args, tensor
addresses) cannot.** The spec *is* the cache key, so a different spec ⇒ different hash ⇒ miss ⇒ recompute —
staleness is impossible by construction. Run-args aren't part of the key, which is why memoizing them is a
footgun; spec-derived layout is not.

Output allocation (~45% of dispatch, shared with legacy) splits into:
- **Irreducible:** the physical address allocation. The caller owns the returned output tensor, so its
  memory can't be pooled/reused while held.
- **Memoizable, spec-exclusive, footgun-free:** the spec-*derived* buffer **layout** —
  `generate_buffer_page_mapping`, the buffer distribution spec, and (sharded) the shard `CoreRangeSet`.
  These are pure functions of the output `TensorSpec`, which is encoded in the ProgramSpec = the cache key.

**Proposed prototype:** park the computed buffer layout (page mapping + distribution spec + shard
`CoreRangeSet`) on the spec cache entry, keyed on the spec hash already computed for the program cache. On
a hit: reuse the layout, do only the physical allocation. Spec-exclusive (legacy has no spec hash; it's a
derived-data sidecar, not a change to the program cache). Expected to recover the "redundant `Buffer::create`"
recompute and i2s's `CoreRangeSet::merge` (~3 µs). Measured on i2s (where the CoreRangeSet cost is biggest).
