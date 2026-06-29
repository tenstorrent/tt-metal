# DRAFT PR — Metal 2.0 spec-as-key: migrate matmul + transpose

Branch: `dgomez/metal2-3ops-pr` (local, NOT pushed) · 2 commits (`cab753a9555` code + `51e9d881a45` perf metrics) · 63 files

## What this PR does
Migrates **matmul** and **transpose** from the ProgramDescriptor path to the Metal 2.0
**spec-as-key** path, and adds the ttnn-side spec-as-key infrastructure they need. The
`metal2_host_api` framework and the upstream `SetProgramRunArgs` fast path (**#47344**) are already
upstream — this PR does not re-ship them.

### Infrastructure (op-agnostic)
- **`ProgramSpecMeshWorkloadFactoryAdapter`** (`mesh_device_operation_adapter.hpp`): on cache miss,
  builds one `ProgramSpec` and stamps a Program onto each mesh coordinate-range; on cache hit,
  re-applies run-args to the cached programs (no rebuild of kernels). `create_program_spec` runs
  **once per dispatch**, not per device — confirmed empirically (N dispatches = N calls) and
  structurally (the call sits outside the per-coordinate loop).
- **`metal2_program_spec_hash.hpp`** — ProgramSpec content hash. **Currently measurement-only; NOT
  the live cache key** (live key remains `attrs+tensor_args`). See follow-ups.
- **`metal2_artifacts.hpp`** — `ProgramArtifacts` (spec + run-args bundle).
- **Perf (data structures):** per-node `RuntimeArgValues` Table backed by `ttsl::SmallVector<,8>`
  (no per-node heap alloc for ≤8 args) and a heap-free inline **`RtaName`** key (no `std::string`
  alloc for long arg names; ops still write `{{"name", value}}` — API unchanged).

### Ops
- **matmul**: MultiCore, reuse-optimized, and batched-HS-DRAM factories emit `create_program_spec`.
- **transpose**: all variants (WH, WH-sharded, WH-sharded-RM, CN, HC-RM, HC-sharded, HC-tiled,
  HC-tiled-interleaved). Per-node run-args built via `reserve()`+`push_back(std::move(...))`.

## Verification & metrics
Cache-hit **host dispatch** µs/call, 10-run device median-of-medians, coherent `build_metal.sh`,
PCC-verified. Chart + repro committed at
`tech_reports/Metal2OpMigration/perf/matmul_transpose_metrics.md`.

![matmul + transpose: descriptor vs spec-as-key](perf/matmul_transpose_spec_vs_descriptor.png)

| op | ProgramDescriptor (origin/main, being replaced) | Metal 2.0 spec-as-key (this PR) | Δ |
|----|---:|---:|---:|
| matmul    | 24.3 µs | 23.4 µs | **−0.9 µs (on par)** |
| transpose | 17.5 µs | 35.7 µs | +18.2 µs (inherent spec rebuild) |

- PCC: matmul **0.99998**, transpose **1.00000**, int32 sharded-RM transpose **exact match**.
- **matmul** spec-as-key is on par with the descriptor path it replaces.
- **transpose** carries the per-dispatch `create_program_spec` rebuild (inherent to spec-as-key, not
  a migration regression). The data-structure opts in this PR (small-vector `Table` + `RtaName`)
  already cut that path **≈25%** (transpose ≈47.5 → 35.7µs) and **≈20%** on matmul-MultiCore; they
  apply to every spec op (shared per-node `Table`).

## Reviewer findings addressed in this PR
- Removed `FORCE_MM_MULTICORE` (a measurement hack that altered production matmul routing).
- Removed `SPECKEY_LOG` fprintf probe + 4 unused probe includes; dropped 4 `METAL2_PORT_REPORT.md` notes.
- Fixed transpose **int32 sharded-RM `DST_ACCUM_MODE`** parity (the spec port over-forced it to 1,
  halving `max_bct` 8→4 vs the original; restored genfiles-derived value). Verified int32 exact match.

## Known follow-ups (NOT in this PR)
1. **rand — descoped.** On a *sharded multi-device* mesh, rand needs a per-device seed; the spec
   path builds run-args once per dispatch (no per-coordinate hook), so all devices would get
   identical seeds. Needs a per-coordinate run-args mechanism (a Metal 2.0 framework gap). matmul
   and transpose are unaffected (their per-device data is tensor addresses, handled by the mesh buffer).
2. **Move-opt consistency.** The reserve+move conversion is applied to the matmul-MultiCore and
   transpose-WH paths; the matmul reuse-mcast variants and several transpose variants still use
   init-list assignment (correct, but deep-copy). Mechanical follow-up; perf-validate per path.
3. **Spec-hash completeness.** `metal2_program_spec_hash` omits semaphores, vararg counts,
   `opt_level`, and tile-format metadata. **Latent only** — it is not the live cache key yet; must be
   completed before promoting it to the live key.
4. **Rebase onto latest `origin/main`.** Branch is based on `574d3bff` (2-day-old main) for a
   conflict-free draft; rebasing to current main has ~15 shared-file conflicts (main's evolution of
   the adapter + program_run_args) to resolve before merge.
5. Cosmetic nits: stale comments (`get_dynamic_runtime_args`, `METAL2_PORT_REPORT.md` references),
   a missing `#include <filesystem>` in `transpose_cn`, unused `RtaName::str()`.
