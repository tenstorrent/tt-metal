# How to use `generate_rank_bindings`

`generate_rank_bindings` must be run under an **MPI** launcher. It performs physical-system discovery and topology mapping, then writes **`rank_bindings.yaml`**, a **`rankfile`**, and optionally **`phase2_mock_mapping.yaml`** under an output directory (rank 0 only).

It can also enumerate **every** valid topology solution (not just the first) and write one artifact set per solution — see [Enumerating all solutions](#enumerating-all-solutions).

**`tt-run` (auto allocation mode):** For application launches, you normally do **not** call this binary directly. Use **`tt-run --mesh-graph-descriptor <mgd> --hosts …`** (real cluster) or **`--mock-cluster-rank-binding …`** (mock); `tt-run` runs `generate_rank_bindings` as **Phase 1** (or reuses a cache under `generated/ttrun/<cache_id>/`). See [ttnn/ttnn/distributed/README_ttrun.md](../../ttnn/ttnn/distributed/README_ttrun.md).

**Sweeping solutions:** to run a workload across every enumerated solution, see [README_sweep_rank_binding_solutions.md](README_sweep_rank_binding_solutions.md).

---

## Prerequisites

- A working **tt-metal** build that includes the `generate_rank_bindings` target (see [Build](#build)).
- **`TT_METAL_HOME`** (or equivalent) set so Metal and descriptor search paths resolve the same way as your other jobs.
- Launch with **`mpirun`**, **`srun`**, or another launcher compatible with how `DistributedContext::create` expects `argc`/`argv`.

---

## Build

From your CMake build tree (example):

```bash
cmake --build build --target generate_rank_bindings
```

The binary is typically at:

```text
<build>/tools/scaleout/generate_rank_bindings
```

Ensure **`LD_LIBRARY_PATH`** includes `<build>/lib` (or your install `lib`) when running, matching other tt-metal binaries.

---

## Command-line options

| Option | Short | Required | Default | Description |
|--------|-------|----------|---------|-------------|
| `--mesh-graph-descriptor` | `-m` | **Yes** | — | Path to the Mesh Graph Descriptor (`.textproto`). |
| `--physical-grouping-descriptor` | `-p` | No | auto | Path to the Physical Grouping Descriptor (`.textproto`). If omitted, a default file is chosen automatically (see [Default Physical Grouping Descriptor](#default-physical-grouping-descriptor-omitting--p)). |
| `--output-dir` | `-o` | No | `generated/ttrun` | Directory for outputs (created if needed). |
| `--all-solutions` | `-a` | No | off | Enumerate multiple solutions and write one artifact set per solution into per-solution subdirectories (see [Output layout](#output-layout)). |
| `--max-solutions <N>` | `-n` | No | `0` | Cap on how many solutions to enumerate. `0` = enumerate all up to the solver's safety cap. Implies `--all-solutions`. |
| `--distinct-host-sets` | `-d` | No | off | Count solutions by the **set of hosts used** — collapse solutions that occupy the same hosts but differ only in connectivity/mapping. Maps to the solver's `unique_shapes=true`. |
| `--help` | `-h` | No | — | Print usage and exit (does not require MPI work beyond parsing). |

Example (single solution — default):

```bash
mpirun -np <N> <mpi-args> \
  /path/to/build/tools/scaleout/generate_rank_bindings \
  --mesh-graph-descriptor /absolute/or/repo/path/to/mesh_graph_descriptor.textproto \
  --output-dir /path/to/out
```

Use paths that every rank can read. Replace `<N>` and `<mpi-args>` with what your cluster and Tenstorrent workflows require (hosts, binding, etc.).

**Default usage (no Physical Grouping Descriptor on the command line):** you only need `-m` (and optionally `-o`). Omit `--physical-grouping-descriptor` and set up [environment](#default-physical-grouping-descriptor-omitting--p) so the tool can find a PGD file.

---

## Help

```bash
/path/to/generate_rank_bindings --help
```

---

## Outputs (written on rank 0)

Under `--output-dir` (default `generated/ttrun`):

| File | When |
|------|------|
| `rank_bindings.yaml` | Always (after successful mapping). |
| `rankfile` | Always. OpenMPI-style lines: `rank N=hostname slot=S`. |
| `phase2_mock_mapping.yaml` | Only if mock cluster descriptor paths were collected from ranks (see [Environment](#environment)). |

`rank_bindings.yaml` includes `mesh_graph_desc_path` and a `rank_bindings` list: each entry has `rank`, `mesh_id`, `mesh_host_rank`, and optional `env_overrides` (e.g. `TT_VISIBLE_DEVICES`).

> With `--all-solutions` these files are written **per solution subdirectory** instead of flat — see [Output layout](#output-layout).

### Example `rank_bindings.yaml`

Illustrative shape (values depend on your cluster, MGD, and mapping):

```yaml
rank_bindings:
  - rank: 0
    mesh_id: 0
    mesh_host_rank: 0
    env_overrides:
      TT_VISIBLE_DEVICES: "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
  - rank: 1
    mesh_id: 1
    mesh_host_rank: 0
    env_overrides:
      TT_VISIBLE_DEVICES: "16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"
mesh_graph_desc_path: "/path/you/passed/to/--mesh-graph-descriptor"
```

Ranks are sequential starting at `0`. `TT_VISIBLE_DEVICES` lists UMD/MMIO device indices visible to that process when applicable.

---

## Enumerating all solutions

By default `generate_rank_bindings` runs topology mapping and emits exactly **one** `rank_bindings.yaml` / `rankfile` pair — the first solution the solver returns. For a given host list and MGD there are usually **many** valid physical placements: different sets of hosts can host the topology, and a fixed set of hosts can be wired/mapped in more than one way.

`--all-solutions` surfaces **every** distinct solution as its own set of rank-binding artifacts, which the DC-team bring-up wrapper uses to sweep cluster shapes (SC4, SC16, SC20, SC24, SC36) and validate each candidate mapping.

Tracking: epic [#49514](https://github.com/tenstorrent/tt-metal/issues/49514) · ticket [#49515](https://github.com/tenstorrent/tt-metal/issues/49515) — *Build topology solver sweep infrastructure*.

### What counts as a distinct solution

`--distinct-host-sets` selects the granularity:

- **default (flag off)** — every distinct mapping is its own solution, **including** several that reuse the **same set of hosts** but wire connectivity / assign fabric nodes differently. (solver `unique_shapes=false`)
- **`--distinct-host-sets` (flag on)** — one solution per **unique host set**; permutations/automorphisms on the same hosts collapse to a single entry. Use this to answer *"which groups of hosts can satisfy this topology?"* (solver `unique_shapes=true`)

### Examples

```bash
# Current behavior (single solution, flat output) — unchanged.
mpirun -np <N> <mpi-args> generate_rank_bindings \
  -m mesh.textproto -o generated/ttrun/<cache_id>

# All distinct solutions (incl. same-hosts / different wiring).
mpirun -np <N> <mpi-args> generate_rank_bindings \
  -m mesh.textproto -o generated/ttrun/<cache_id> --all-solutions

# At most 8 solutions.
mpirun -np <N> <mpi-args> generate_rank_bindings \
  -m mesh.textproto -o generated/ttrun/<cache_id> --max-solutions 8

# One solution per unique host set (distinct footprints only).
mpirun -np <N> <mpi-args> generate_rank_bindings \
  -m mesh.textproto -o generated/ttrun/<cache_id> --all-solutions --distinct-host-sets
```

### Output layout

Single-solution mode (default, no `--all-solutions`) is **byte-for-byte unchanged**: `rank_bindings.yaml` and `rankfile` are written directly into `--output-dir`.

With `--all-solutions`, each solution gets its **own subdirectory**, named by a short **content hash** of the solution, plus a top-level index:

```
generated/ttrun/<cache_id>/
  solutions_index.yaml          # summary of every enumerated solution
  <solution_hash>/              # e.g. 3f9c1a20
    rank_bindings.yaml
    rankfile
    solution_meta.yaml          # hosts used, mapping summary, solver stats
    .solution_key               # full canonical signature string (collision disambiguation)
  <solution_hash>/              # e.g. 8e77d0b4
    rank_bindings.yaml
    rankfile
    solution_meta.yaml
    .solution_key
  ...
```

- `<cache_id>` is the existing tt-run Phase-1 cache id (16-hex SHA-256 prefix over MGD bytes + sorted host list); see the tt-run README. This feature adds a **second level** below it, one directory per solution.
- In mock mode, `phase2_mock_mapping.yaml` is written per solution subdirectory (same as it is written flat today).

### Directory naming — content hash of host-set + mapping

Each `<solution_hash>` is a **16-hex-character FNV-1a** hash of a **canonical signature** of the solution:

1. the **sorted set of hostnames** the solution occupies, followed by
2. the **sorted per-`(mesh_id, mesh_host_rank)` assignment** — which host and which `TT_VISIBLE_DEVICES` each logical mesh host maps to.

(FNV-1a is a fast, deterministic, non-cryptographic hash — chosen over SHA-256 to avoid a crypto dependency; the full signature string in `.solution_key` remains the source of truth for exact comparison.)

Properties this gives us:

- **Stable / reproducible** — the same solution produces the same directory name across runs and machines, so caches and downstream references stay valid.
- **Content-addressed dedupe** — two runs that yield the same solution write to the same directory instead of piling up `sol_00`, `sol_01`, … that drift with solver ordering.
- **Distinguishes wiring** — because the mapping is part of the signature, two solutions on the **same hosts** but different connectivity get **different** directories (they are genuinely different solutions). With `--distinct-host-sets` those variants are never generated, so the host set effectively dominates the hash.
- **Collision-detectable** — the full canonical signature string is written to `.solution_key` in each solution directory, so a short-hash collision (two distinct signatures hashing to the same 16-hex id) is detectable by comparing `.solution_key`. With a 64-bit hash over a handful of solutions this is vanishingly unlikely in practice.

> Why not `sol_00` / `sol_01`? Sequential names are readable but **unstable** — they depend on solver enumeration order, so the "same" solution can land in a different directory between runs, breaking any cached reference to it. The content hash was chosen for stability and dedupe. The human-readable ordering and labels still live in `solutions_index.yaml`.

### `solutions_index.yaml`

Machine-readable summary the bring-up wrapper consumes:

```yaml
mesh_graph_desc_path: /path/to/mesh.textproto
enumeration:
  mode: all              # or: distinct-host-sets
  max_solutions: 0       # 0 = all up to solver cap
  found: 3
  truncated: false       # true if max-solutions or the solver safety cap was hit
solutions:
  - id: 3f9c1a20
    dir: 3f9c1a20
    num_hosts: 4
    num_ranks: 8
    host_set: [host-a, host-b, host-c, host-d]
    rank_bindings: 3f9c1a20/rank_bindings.yaml
    rankfile: 3f9c1a20/rankfile
  - id: 8e77d0b4
    dir: 8e77d0b4
    num_hosts: 4
    num_ranks: 8
    host_set: [host-a, host-b, host-c, host-e]
    rank_bindings: 8e77d0b4/rank_bindings.yaml
    rankfile: 8e77d0b4/rankfile
```

`solutions` is ordered by the solver's preference ranking (best first); `id` and `dir` are the stable content hash so consumers can address a solution without depending on order.

---

## Environment

These matter for discovery, PGD lookup, and optional phase-2 mapping:

| Variable | Role |
|----------|------|
| `TT_METAL_PHYSICAL_GROUPING_DESCRIPTOR_PATH` | If set to an existing file, that PGD is used when you omit `-p` (checked before cluster/arch search). |
| `TT_CLUSTER_NAME` | If set, enables cluster-specific PGD paths under `/data/scaleout_configs/...` and `TT_METAL_HOME/tests/tt_metal/tt_fabric/physical_groupings/<cluster>_physical_grouping_descriptor.textproto` (see [below](#default-physical-grouping-descriptor-omitting--p)). |
| `TT_METAL_HOME` | Base for repo-relative PGD search paths; defaults to `.` if unset. |
| `TT_METAL_MOCK_CLUSTER_DESC_PATH` | If set per rank, rank 0 can gather paths and emit **`phase2_mock_mapping.yaml`**; rankfile behavior may use a single local hostname for placement in mock scenarios. |

Implementation: `find_and_load_physical_grouping_descriptor` in `tt_metal/fabric/fabric_host_utils.cpp` (shared with ControlPlane / TopologyMapper).

---

## Default Physical Grouping Descriptor (omitting `-p`)

If you do **not** pass `--physical-grouping-descriptor`, the tool picks a PGD file in this order; the **first path that exists** as a regular file is used.

1. **`TT_METAL_PHYSICAL_GROUPING_DESCRIPTOR_PATH`** — if the variable is set and points to an existing file, that file is the PGD (explicit default you control).

2. **Cluster name paths** (only if **`TT_CLUSTER_NAME`** is non-empty), under `TT_METAL_HOME` (default `.` if unset):
   - `/data/scaleout_configs/<TT_CLUSTER_NAME>/<TT_CLUSTER_NAME>_physical_grouping_descriptor.textproto`
   - `<TT_METAL_HOME>/tests/tt_metal/tt_fabric/physical_groupings/<TT_CLUSTER_NAME>_physical_grouping_descriptor.textproto`

3. **Architecture / cluster-type candidate** under the repo tree — **MetalContext** (discovered cluster type and architecture) picks a **first** filename to try at
   `<TT_METAL_HOME>/tests/tt_metal/tt_fabric/physical_groupings/<filename>`:
   - Galaxy + Wormhole B0 → `wh_bh_rev_c_galaxy_physical_grouping_descriptor.textproto`
   - Blackhole Galaxy + Blackhole Rev C → `wh_bh_rev_c_galaxy_physical_grouping_descriptor.textproto`
   - Blackhole Galaxy + Blackhole Rev A/B → `bh_galaxy_rev_ab_physical_grouping_descriptor.textproto`
   - T3K + Wormhole B0 → `wh_t3k_physical_grouping_descriptor.textproto`
   - Any other combination → `default_physical_grouping_descriptor.textproto` (only this path is attempted for step 3; there is no separate “specialized then default” for this case).

4. **Generic default fallback** — if step 3 chose one of the **specialized** filenames above (not already `default_physical_grouping_descriptor.textproto`), the tool **also** tries
   `<TT_METAL_HOME>/tests/tt_metal/tt_fabric/physical_groupings/default_physical_grouping_descriptor.textproto`
   so a missing specialized file does not fail until the default has been checked.

The **first path in this combined search order that exists** as a regular file is used. If every attempted path is missing, the tool fails with an error listing what it searched (including the default when step 4 applied).

**Typical local default:** with `TT_METAL_HOME` pointing at your checkout, discovery often selects a specialized PGD when it exists; otherwise step 4 supplies `default_physical_grouping_descriptor.textproto`. If MetalContext maps to “any other combination,” only the default filename from step 3 is tried for repo-relative PGDs.

---

## Implementation (all-solutions path)

The solver-level enumeration already existed; `--all-solutions` **surfaces it through the multi-mesh layer and the CLI**.

1. **`topology_mapper_utils` — multi-solution entry point.**
   `std::vector<TopologyMappingResult> map_multi_mesh_to_physical_n(..., size_t max_solutions, bool unique_shapes)` sits alongside the untouched single-result `map_multi_mesh_to_physical` (zero regression risk for existing callers). It **enumerates distinct inter-mesh placements** — which physical meshes / hosts host each logical mesh — via `solve_topology_mapping_n` (the proven blocking-clause search), passing `max_solutions` and `unique_shapes` straight through. Each placement is then completed with the same per-mesh intra-mesh (fabric-node → ASIC) solve the single-result path uses; placements whose intra-mesh mapping is infeasible are skipped, and results are deduplicated by their full fabric-node → ASIC assignment.

   > **Enumeration granularity:** distinctness is currently driven by the **inter-mesh placement** (which physical meshes / hosts). This fully covers `--distinct-host-sets` and `--max-solutions`. Enumerating additional intra-mesh permutations *within a fixed placement* (finer-grained "all") is a bounded follow-up; the plumbing (`unique_shapes`, per-mesh `solve_topology_mapping_n`) is already in place to extend it.

2. **`generate_rank_bindings.cpp` — CLI + write loop.**
   `--all-solutions` / `--max-solutions` / `--distinct-host-sets` are parsed into `ProgramArgs`; the shared setup is factored into `build_topology_mapping_inputs` so `run_topology_mapping` (single) and `run_topology_mapping_n` (all) share it. In the rank-0 path, the default writes one solution flat (unchanged); with `--all-solutions` it loops the results, runs `extract_rank_bindings` per solution, computes the signature hash, writes `<output-dir>/<solution_hash>/` (`rank_bindings.yaml` + `rankfile` + `phase2_mock_mapping.yaml` in mock mode + `solution_meta.yaml` + `.solution_key`), then writes `solutions_index.yaml`. The existing rank-0-only + `fsync` + MPI-barrier discipline is preserved (every file and subdir is fsync'd before the barrier).

3. **`generate_rank_bindings_helpers.hpp` — writers + hashing.**
   `compute_solution_signature_hash` / `compute_solution_signature_string`, `write_solution_meta_yaml`, and `write_solutions_index_yaml` (plus the `SolutionIndexEntry` struct) sit next to the existing `write_rank_bindings_yaml` / `write_rankfile`.

4. **Tests** (`tools/tests/scaleout/test_generate_rank_bindings.cpp`, CPU-only):
   hash stability / order-independence; same-hosts-different-mapping → different hash; `solutions_index.yaml` field correctness (`mode`, `max_solutions`, `found`, `truncated`, per-solution `host_set`); and `solution_meta.yaml` contents. (End-to-end enumeration on real/mock clusters runs under MPI and is exercised via the tt-run flows.)

### Minimal host count

When a multi-mesh MGD is placed on a larger physical system, the mapper minimizes the number of hosts used (e.g. a 64-mesh blitz superpod packs onto **16** hosts, not 17–23). This is expressed as a **same-rank-group occupancy** constraint on `MappingConstraints` — `set_max_same_rank_groups_used(k_min)` (hard "at most `k_min` host groups occupied", solver chooses which) with `set_minimize_same_rank_groups_used(true)` as a best-effort fallback — so both the single solve and `--all-solutions` reach the minimum host count. See `add_inter_mesh_minimal_host_cover_from_hostname_map` in `tt_metal/fabric/topology_mapper_utils.cpp` and the occupancy encoding in `tt_metal/fabric/topology_solver_sat.cpp`.

### Edge cases / decisions

- **Backward compatibility:** without `--all-solutions`, output is exactly as today (flat `rank_bindings.yaml` in `--output-dir`; no index, no subdirs).
- **Exactly one solution found with `--all-solutions`:** still uses the subdirectory layout + index, so consumers see one consistent structure.
- **Zero solutions:** write a `solutions_index.yaml` with `found: 0` and exit non-zero (mirrors today's mapping-failure exit code).
- **Truncation is explicit:** `truncated: true` in the index whenever `--max-solutions` or the solver's internal safety cap bounded the result — so a capped sweep is never mistaken for an exhaustive one.

### Out of scope (later tickets)

- Teaching `tt-run` to consume `solutions_index.yaml` (pick/validate a solution).
- PipelineBuilder passthrough validation per solution (epic ticket [#49517](https://github.com/tenstorrent/tt-metal/issues/49517)).
- The DC-team bring-up shell wrapper ([#49519](https://github.com/tenstorrent/tt-metal/issues/49519)).

---

## Troubleshooting

- **Topology mapping fails:** Check MGD/PGD consistency with the real or mock cluster and PSD; logs use `LogFabric`.
- **No `phase2_mock_mapping.yaml`:** Normal if `TT_METAL_MOCK_CLUSTER_DESC_PATH` was not set (or paths were not gathered).
- **Wrong device visibility:** Bindings derive `TT_VISIBLE_DEVICES` from the cluster and mapping; verify MGD and PSD match your allocation.

For a higher-level description of what the tool does in the codebase and what landed in a given change, see the associated pull request description.
