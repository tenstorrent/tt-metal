# Multimesh pipeline-stage rank bindings — SC16 capacity limit + the correct run recipe

Two multimesh "pipeline-stage" MGDs, each a set of **separate meshes joined in a RING**
(a pipeline of stages) — **not** one big mesh:

| file | stages (meshes) | per-stage device topology | chips / stage | total chips | galaxies needed |
|------|-----------------|---------------------------|---------------|-------------|-----------------|
| [`mgd_tests/01_homogeneous_36stage_multimesh.textproto`](mgd_tests/01_homogeneous_36stage_multimesh.textproto) | 36 | `[8,4]` = full galaxy | 32 | **1152** | **36** |
| [`mgd_tests/03_64stage_multimesh.textproto`](mgd_tests/03_64stage_multimesh.textproto) | 64 | `[4,4]` = half galaxy | 16 | **1024** | **32** |

Each Blackhole galaxy host = 32 chips. "Galaxies needed" = ⌈total chips / 32⌉.

---

## Result on SC16: both files FAIL Mapping (hard capacity limit)

**SC16 = 16 galaxies = 512 chips.** Both files ask for more logical chips than SC16 has,
so no placement exists. Ran on **two** SC16 FSDs (`SC16_32x4_revAB_aisleD`,
`SC16_32x4_revC_aisleC`), **both routes** (PSD live-discovery and FSD offline, `np=16`):

| file | SC16 chips available | chips required | solver verdict |
|------|----------------------|----------------|----------------|
| 01 (36-stage) | 512 | 1152 | `SAT joint placement: no placement found … candidate lists COMPLETE — the UNSAT verdict is trustworthy` |
| 03 (64-stage) | 512 | 1024 | `no placement found` (UNSAT; larger search truncated at the solver cap, still no placement) |

- **PSD vs FSD make no difference here** — this is capacity, not route. The FSD offline path
  (#52859) and the PSD live-discovery path hit the identical physical ceiling.
- **"Splitting" does not help.** Splitting spreads one mesh *across* hosts; it never lets two
  logical chips share one physical chip. 1152 (or 1024) logical chips cannot occupy 512 physical
  chips. The solver proves this (UNSAT), it does not merely fail to find a mapping.

Trimmed solver logs for all runs: [`results/sc16_unsat/`](results/sc16_unsat/). The definitive,
*complete-candidate* UNSAT is
[`SC16_32x4_revAB_aisleD__file01_PSD.log`](results/sc16_unsat/SC16_32x4_revAB_aisleD__file01_PSD.log).

---

## How to actually run them (a system with enough galaxies)

File 01 needs ≥36 galaxies, File 03 needs ≥32. **`SC36_32x4_revC_subtorus_aisleD` (36 galaxies)
fits both.** On SC36 both pass Mapping and emit one rank binding per stage:

| file | system | np | result | bindings |
|------|--------|----|--------|----------|
| 01 (36-stage) | SC36_32x4_revC_subtorus_aisleD | 36 | **Mapping OK** | [`rank_bindings/01_homogeneous_36stage_sc36_rank_bindings.yaml`](rank_bindings/01_homogeneous_36stage_sc36_rank_bindings.yaml) (36) |
| 03 (64-stage) | SC36_32x4_revC_subtorus_aisleD | 36 | capacity fits (1024 ≤ 1152); **placement solve heavy** — see note | _(pending)_ |

> **File 03 solver note.** File 03's stages are half-galaxy `[4,4]`, which the physical-grouping
> matcher resolves to **624 candidate groupings per mesh** (`4x4_Mesh_flat`, `4x4_SplitHost_flat`,
> TORUSX/Y/XY variants…). Across 64 meshes that is an enormous joint-placement SAT. File 01's
> full-galaxy `[8,4]` stages match a galaxy cleanly and solve in ~2 s; File 03 did **not** converge
> within a 420 s budget and is being re-run with a longer budget. This is a **solver-scaling** limit
> for many ambiguous sub-galaxy stages, not a capacity limit — see the kissat SAT-engine work
> ([#56920](https://github.com/tenstorrent/tt-metal/pull/56920)) for the convergence angle. Bindings
> will be added here once the solve lands.

### Run recipe (CPU-only, mock discovery = PSD route)

One MPI rank per host; each rank reads one per-host mock cluster descriptor via
`TT_METAL_MOCK_CLUSTER_DESC_PATH`. `rank_paths.txt` lists those per-host descriptors in rank
order (built from the supercluster's `*_cluster_desc/*_rank_N.yaml` files).

```bash
cd /workspace/group/tt-metal
export TT_METAL_HOME=/workspace/group/tt-metal

# rank_wrapper.sh exports TT_METAL_MOCK_CLUSTER_DESC_PATH=$(sed -n "$((RANK+1))p" rank_paths.txt)
mpirun --allow-run-as-root --oversubscribe -x TT_METAL_HOME -np 36 \
  ./rank_wrapper.sh <out_dir> \
  build_Release/tools/scaleout/generate_rank_bindings \
  -m tests/scale_out/fsd_psd_parity/mgd_tests/01_homogeneous_36stage_multimesh.textproto \
  -o <out_dir>
```

### FSD route (offline-from-FSD, #52859)

Same command, additionally `-x TT_METAL_FACTORY_SYSTEM_DESCRIPTOR_PATH=<fsd.textproto>`.
On this tree the FSD offline builder falls back to the PSD mapper (#52859), so for a system that
*fits* the placement is identical to the PSD route; for a system that does **not** fit (SC16) both
routes return the same UNSAT. See [`run_fsd_route.sh`](run_fsd_route.sh) and
[`experiment_C_52859.patch`](experiment_C_52859.patch) for the offline wiring.

---

## TL;DR

- **On SC16, files 01 and 03 cannot pass Mapping** — they need 1152 / 1024 chips, SC16 has 512.
  Proven UNSAT, identical on PSD and FSD routes.
- **Run them on SC36** (`SC36_32x4_revC_subtorus_aisleD`, 36 galaxies). File 01 passes Mapping
  (36 bindings in [`rank_bindings/`](rank_bindings/)). File 03 is capacity-feasible there but its
  half-galaxy stages make the placement SAT expensive — solve in progress.
