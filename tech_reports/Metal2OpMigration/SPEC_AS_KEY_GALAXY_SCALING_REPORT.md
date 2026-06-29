# Spec-as-key host dispatch: cost, optimization, and Galaxy (32-device) scaling

**For:** Audrey, Artem  **From:** Diego  **Date:** 2026-06-26

## TL;DR

- On a single device, the spec-as-key path (`ProgramSpecFactory…Concept`, build-the-spec-every-dispatch)
  costs **~2.1–2.4× the legacy host dispatch** for the ops measured. Measured, not estimated.
- The **defining spec-as-key cost — building the ProgramSpec and hashing it — is incurred exactly ONCE
  per dispatch, independent of mesh device count** (verified by code reading, not yet by a 32-device run).
- The part that *can* scale with device count is the **run-args apply** (`SetProgramRunArgs`) +
  program instantiation: **once for uniform-storage ops, ×N for per-device-sharded ops**. Command
  writing to each device CQ is ×N always — but that is **shared with legacy**.
- **Net for Galaxy:** for uniform/data-parallel ops the spec tax does **not** amplify with mesh size
  (it's amortized once across the mesh); for non-uniform/sharded ops the run-args apply scales ×N,
  largely in line with legacy's own per-program scaling.
- One **shipped, behavior-preserving win** already: a `CoreRangeSet` fast path, **−14%** on the i2s
  spec-as-key dispatch, which also speeds up legacy and every sharded op.

## Measurement methodology (so the numbers are reproducible)

- Op vs. its legacy twin, **same shape**, on hardware (1× Wormhole). **DIM 1024, `TT_METAL_INSPECTOR=0`,**
  warm program cache, N=4000 iters, min/median µs/op, PCC checked = 1.0.
- Demo ops on the spec-as-key path: quasar `interleaved_to_sharded` (i2s) and quasar `transpose`.
- Binary discipline: measured against the worktree's freshly-relinked `_ttnncpp.so` (the editable ttnn
  install otherwise silently loads the main-repo build).

## Single-device baseline (measured)

| op (1024², INSPECTOR=0, PCC 1.0) | legacy | spec-as-key | ratio |
|---|---|---|---|
| interleaved_to_sharded (32-core height-shard) | 15.4 µs | 38.4 µs → **33.1 µs** (after fast path) | 2.50× → **2.11×** |
| transpose | 14.0 µs | 33.8 µs | 2.41× |

> Note: prior write-ups quoted ~1.34× for transpose; that used a legacy baseline ~2× higher than we
> now measure. The quasar absolute (~34 µs) matched; the **ratio was optimistic**. Corrected here.

## Where the spec-as-key host time goes (py-spy --native, i2s, post-fast-path; inclusive %)

| area | ~% dispatch | notes |
|---|---|---|
| `create_program_spec` (spec rebuild) | ~19% | the build-to-hash tax; frozen by design |
| output allocation (allocate / create_output / Buffer::create) | ~22% | **shared floor with legacy** |
| enqueue / command write | ~20% | per-device; shared with legacy |
| `SetProgramRunArgs` (run-args apply) | ~12% | per-arg `slot_of.find` name→slot hashing |
| alloc churn (malloc / SmallVector) | ~9% | |
| spec hash (`program_spec_cache_key`) | ~3–8% | reflection hash of the spec content |
| `filesystem::path` (kernel source) | ~4.5% | path construct + hash, per dispatch |

## The Galaxy question: does the spec-as-key tax multiply by 32?

**Verified from code** (`ttnn/api/ttnn/device_operation.hpp`, `ttnn/api/ttnn/mesh_device_operation_adapter.hpp`):

| component | how many calls per dispatch | scales with #devices? |
|---|---|---|
| `create_program_spec` (the spec **build**) | **exactly once** (`create_spec`, adapter L828/L892 — single call; invoked at device_operation.hpp:395, outside the hit/miss branch and all loops; `run_spec_flow` itself called once per op at L499) | **NO** |
| spec **hash** (`program_spec_cache_key`) | **once** (device_operation.hpp:400) (+ a trivial O(N) loop hashing coord structs into the key, L401–403) | **NO** (negligible) |
| `SetProgramRunArgs` (run-args **apply**) | **per coordinate range**: 1 for uniform storage, N for non-uniform (apply_run_args L855/L918; build_mesh_workload L842/L907) | **NO if uniform; ×N if non-uniform** |
| `MakeProgramFromSpec` (miss path only) | per coordinate range | NO if uniform; ×N if non-uniform |
| command write (`enqueue_mesh_workload` → `write_program_cmds_to_subgrid`, fd_mesh_command_queue.cpp L412/L1071) | **per device** (`for_each_local`) | **always ×N — shared with legacy** |

`tensor_coords` is one full-mesh range when `all_tensors_have_uniform_storage(tensor_args)` is true
(device_operation.hpp:386-387), else one range per coordinate (L389-391). That single boolean decides
the "once" vs "×N" bucket for the run-args apply.

### Interpretation
- **Uniform / data-parallel ops:** one program for the whole mesh ⇒ spec build once, run-args apply
  once. The only ×N host work is the per-device command write, which **legacy pays identically**. The
  spec-as-key tax stays ~constant (absolute) and **shrinks as a fraction** of dispatch as the mesh
  grows (per-device command issue dominates). **No amplification.**
- **Non-uniform / per-device-sharded ops:** the workload holds one program per device ⇒ `SetProgramRunArgs`
  runs ×N. The spec **build** is still once. Legacy also has per-program run-args here, so most of the
  ×N is shared; the spec-as-key-specific part is the per-program run-args delta.

### Honest caveats
- This box has **1 device** — the scaling conclusion is **static code analysis, not a 32-device
  measurement**. It rests on: `create_program_spec` called once (verified), and the per-range loops
  (verified). A real multi-device run would harden it.
- Not yet traced: exactly which production Galaxy ops are uniform vs. non-uniform storage. That
  determines how many real ops sit in the "amortized once" bucket vs. the "×N run-args" bucket.

## Optimizations

### Shipped / validated — `CoreRangeSet` solid-rectangle fast path (metal primitive)
`CoreRangeSet(Span<const CoreCoord>)` ran `merge_ranges` (2D grid alloc + `std::set` churn) every
dispatch. Added an O(N) fast path: bounding box + flat-bitset duplicate check; a solid rectangle (the
common full-worker-grid case) emits a single `CoreRange`, else falls back to `merge_ranges`.
- **i2s 38.4 → 33.1 µs (−5.3 µs / −14%), PCC 1.0.** Transpose unchanged (not built from a coord list).
- Behavior-preserving: 13/13 `CoreRangeSet` unit tests pass, incl. new cases (solid rect, single,
  L-shape-with-hole, and a duplicate-coords case that must still throw via `validate_no_overlap`).
- **Helps legacy and every sharded op too** — it's a shared metal primitive. Staged for a standalone PR.

### Finding (not pursued) — run-args apply `SetProgramRunArgs`
The apply fast path does one `slot_of.find(name)` per arg per node (~670 RtaName hashes/dispatch for
i2s 32-core). A throwaway experiment replacing it with positional application (the `ttnn::spec` builder
emits named args in schema order) measured **−2.5 µs / −7.5%** (i2s 33.1 → 30.6 µs, PCC 1.0), then was
reverted. This is in the run-args apply path; a production version would want to preserve the name→slot
map as the default/fallback (e.g. an opt-in positional path only for builder-emitted, in-order args).
Flagged for discussion rather than implemented.

### Other non-spec levers, quantified and de-prioritized
- `filesystem::path` kernel-source construct+hash: ~0.6–4.5%, too small to pursue.
- SHM allocation telemetry (`ShmTrackingProcessor`, `getpid` per alloc): ~1–2 µs, runtime toggle
  (`TT_METAL_SHM_TRACKING_DISABLED=1`), not a code fix; disabling loses tt-smi memory stats.
- Output allocation (~22%): largely irreducible (caller owns the output buffer) and shared with legacy.

## Asks / open questions
1. Sanity-check the scaling reading of `run_spec_flow` / the adapters above — is "spec build once,
   run-args apply per-range" the intended contract?
2. For the run-args apply ×N on non-uniform ops: is that acceptable, or worth an in-order fast path?
3. Which Galaxy model ops are uniform vs. non-uniform storage (determines real-world impact)?
4. OK to land the `CoreRangeSet` fast path as a standalone metal PR (helps legacy too)?
