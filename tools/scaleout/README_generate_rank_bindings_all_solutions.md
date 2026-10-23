# `generate_rank_bindings` — enumerate rank bindings for *all* solutions

Status: **implemented**. The `generate_rank_bindings` CLI flags, the
`map_multi_mesh_to_physical_n` mapper, and the per-solution writers described
below are in the tree; CPU-only unit tests cover the hashing and index writers.

Tracking: epic [#49514](https://github.com/tenstorrent/tt-metal/issues/49514) ·
ticket [#49515](https://github.com/tenstorrent/tt-metal/issues/49515) — *Build
topology solver sweep infrastructure*.

Related docs:
[`README_generate_rank_bindings.md`](README_generate_rank_bindings.md) ·
[`../../ttnn/ttnn/distributed/README_ttrun.md`](../../ttnn/ttnn/distributed/README_ttrun.md).

---

## Motivation

Today `generate_rank_bindings` runs topology mapping and emits exactly **one**
`rank_bindings.yaml` / `rankfile` pair — the first solution the solver returns.
For a given host list and mesh-graph descriptor (MGD) there are usually **many**
valid physical placements: different sets of hosts can host the topology, and a
fixed set of hosts can be wired/mapped in more than one way.

The topology solver already knows how to enumerate these
(`solve_topology_mapping_n` / `solve_topology_mapping_all` /
`TopologyMappingEnumerationSession` in
`tt_metal/api/tt-metalium/experimental/fabric/topology_solver.hpp`), but
`generate_rank_bindings` only consumes the single `TopologyMappingResult`
returned by `map_multi_mesh_to_physical`.

This feature surfaces **every** distinct solution as its own set of rank-binding
artifacts, which the DC-team bring-up wrapper (the parent epic) uses to sweep
cluster shapes (SC4, SC16, SC20, SC24, SC36) and validate each candidate mapping.

---

## New command-line options

Added to `generate_rank_bindings` (all optional; **default behavior is
unchanged** — a single solution written flat into `--output-dir`):

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--all-solutions` | `-a` | off | Enumerate multiple solutions and write one artifact set per solution into per-solution subdirectories (see [Output layout](#output-layout)). |
| `--max-solutions <N>` | `-n` | `0` | Cap on how many solutions to enumerate. `0` = enumerate all up to the solver's safety cap. Implies `--all-solutions`. |
| `--distinct-host-sets` | `-d` | off | Count solutions by the **set of hosts used** — collapse solutions that occupy the same hosts but differ only in connectivity/mapping. Maps to the solver's `unique_shapes=true`. |

`--distinct-host-sets` selects **what counts as a distinct solution**:

- **default (flag off)** — every distinct mapping is its own solution,
  **including** several that reuse the **same set of hosts** but wire
  connectivity / assign fabric nodes differently. (solver `unique_shapes=false`)
- **`--distinct-host-sets` (flag on)** — one solution per **unique host set**;
  permutations/automorphisms on the same hosts collapse to a single entry.
  Use this to answer *"which groups of hosts can satisfy this topology?"*
  (solver `unique_shapes=true`)

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

---

## Output layout

Single-solution mode (default, no `--all-solutions`) is **byte-for-byte
unchanged**: `rank_bindings.yaml` and `rankfile` are written directly into
`--output-dir`.

With `--all-solutions`, each solution gets its **own subdirectory**, named by a
short **content hash** of the solution, plus a top-level index:

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

- `<cache_id>` is the existing tt-run Phase-1 cache id (16-hex SHA-256 prefix
  over MGD bytes + sorted host list); see the tt-run README. This feature adds
  a **second level** below it, one directory per solution.
- In mock mode, `phase2_mock_mapping.yaml` is written per solution subdirectory
  (same as it is written flat today).

### Directory naming — content hash of host-set + mapping

Each `<solution_hash>` is a **16-hex-character FNV-1a** hash of a **canonical
signature** of the solution:

1. the **sorted set of hostnames** the solution occupies, followed by
2. the **sorted per-`(mesh_id, mesh_host_rank)` assignment** — which host and
   which `TT_VISIBLE_DEVICES` each logical mesh host maps to.

(FNV-1a is a fast, deterministic, non-cryptographic hash — chosen over SHA-256
to avoid a crypto dependency; the full signature string in `.solution_key`
remains the source of truth for exact comparison.)

Properties this gives us:

- **Stable / reproducible** — the same solution produces the same directory
  name across runs and machines, so caches and downstream references stay valid.
- **Content-addressed dedupe** — two runs that yield the same solution write to
  the same directory instead of piling up `sol_00`, `sol_01`, … that drift with
  solver ordering.
- **Distinguishes wiring** — because the mapping is part of the signature, two
  solutions on the **same hosts** but different connectivity get **different**
  directories (they are genuinely different solutions). With
  `--distinct-host-sets` those variants are never generated, so the host set
  effectively dominates the hash.
- **Collision-detectable** — the full canonical signature string is written to
  `.solution_key` in each solution directory, so a short-hash collision (two
  distinct signatures hashing to the same 16-hex id) is detectable by comparing
  `.solution_key`. With a 64-bit hash over a handful of solutions this is
  vanishingly unlikely in practice.

> Why not `sol_00` / `sol_01`? Sequential names are readable but **unstable** —
> they depend on solver enumeration order, so the "same" solution can land in a
> different directory between runs, breaking any cached reference to it. The
> content hash was chosen for stability and dedupe. The human-readable ordering
> and labels still live in `solutions_index.yaml`.

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

`solutions` is ordered by the solver's preference ranking (best first); `id`
and `dir` are the stable content hash so consumers can address a solution
without depending on order.

---

## Implementation

The solver-level enumeration already existed; this feature **surfaces it through
the multi-mesh layer and the CLI**.

1. **`topology_mapper_utils` — multi-solution entry point.**
   `std::vector<TopologyMappingResult> map_multi_mesh_to_physical_n(...,
   size_t max_solutions, bool unique_shapes)` sits alongside the untouched
   single-result `map_multi_mesh_to_physical` (zero regression risk for existing
   callers). It **enumerates distinct inter-mesh placements** — which physical
   meshes / hosts host each logical mesh — via `solve_topology_mapping_n`
   (the proven blocking-clause search), passing `max_solutions` and
   `unique_shapes` straight through. Each placement is then completed with the
   same per-mesh intra-mesh (fabric-node → ASIC) solve the single-result path
   uses; placements whose intra-mesh mapping is infeasible are skipped, and
   results are deduplicated by their full fabric-node → ASIC assignment.

   > **Enumeration granularity:** distinctness is currently driven by the
   > **inter-mesh placement** (which physical meshes / hosts). This fully covers
   > `--distinct-host-sets` and `--max-solutions`. Enumerating additional
   > intra-mesh permutations *within a fixed placement* (finer-grained "all") is
   > a bounded follow-up; the plumbing (`unique_shapes`, per-mesh
   > `solve_topology_mapping_n`) is already in place to extend it.

2. **`generate_rank_bindings.cpp` — CLI + write loop.**
   `--all-solutions` / `--max-solutions` / `--distinct-host-sets` are parsed into
   `ProgramArgs`; the shared setup is factored into `build_topology_mapping_inputs`
   so `run_topology_mapping` (single) and `run_topology_mapping_n` (all) share it.
   In the rank-0 path, the default writes one solution flat (unchanged); with
   `--all-solutions` it loops the results, runs `extract_rank_bindings` per
   solution, computes the signature hash, writes `<output-dir>/<solution_hash>/`
   (`rank_bindings.yaml` + `rankfile` + `phase2_mock_mapping.yaml` in mock mode
   + `solution_meta.yaml` + `.solution_key`), then writes `solutions_index.yaml`.
   The existing rank-0-only + `fsync` + MPI-barrier discipline is preserved (every
   file and subdir is fsync'd before the barrier).

3. **`generate_rank_bindings_helpers.hpp` — writers + hashing.**
   `compute_solution_signature_hash` / `compute_solution_signature_string`,
   `write_solution_meta_yaml`, and `write_solutions_index_yaml` (plus the
   `SolutionIndexEntry` struct) sit next to the existing
   `write_rank_bindings_yaml` / `write_rankfile`.

4. **Tests** (`tools/tests/scaleout/test_generate_rank_bindings.cpp`, CPU-only):
   hash stability / order-independence; same-hosts-different-mapping → different
   hash; `solutions_index.yaml` field correctness (`mode`, `max_solutions`,
   `found`, `truncated`, per-solution `host_set`); and `solution_meta.yaml`
   contents. (End-to-end enumeration on real/mock clusters runs under MPI and is
   exercised via the tt-run flows.)

### Edge cases / decisions

- **Backward compatibility:** without `--all-solutions`, output is exactly as
  today (flat `rank_bindings.yaml` in `--output-dir`; no index, no subdirs).
- **Exactly one solution found with `--all-solutions`:** still uses the
  subdirectory layout + index, so consumers see one consistent structure.
- **Zero solutions:** write a `solutions_index.yaml` with `found: 0` and exit
  non-zero (mirrors today's mapping-failure exit code).
- **Truncation is explicit:** `truncated: true` in the index whenever
  `--max-solutions` or the solver's internal safety cap bounded the result — so a
  capped sweep is never mistaken for an exhaustive one.

### Out of scope (later tickets)

- Teaching `tt-run` to consume `solutions_index.yaml` (pick/validate a solution).
- PipelineBuilder passthrough validation per solution (epic ticket
  [#49517](https://github.com/tenstorrent/tt-metal/issues/49517)).
- The DC-team bring-up shell wrapper
  ([#49519](https://github.com/tenstorrent/tt-metal/issues/49519)).
