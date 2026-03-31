# How to use `generate_rank_bindings`

`generate_rank_bindings` must be run under an **MPI** launcher. It performs physical-system discovery and topology mapping, then writes **`rank_bindings.yaml`**, a **`rankfile`**, and optionally **`phase2_mock_mapping.yaml`** under an output directory (rank 0 only).

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

| Option | Short | Required | Description |
|--------|-------|----------|-------------|
| `--mesh-graph-descriptor` | `-m` | **Yes** | Path to the Mesh Graph Descriptor (`.textproto`). |
| `--physical-grouping-descriptor` | `-p` | No | Path to the Physical Grouping Descriptor (`.textproto`). If omitted, a default file is chosen automatically (see [Default Physical Grouping Descriptor](#default-physical-grouping-descriptor-omitting--p)). |
| `--output-dir` | `-o` | No | Directory for outputs. Default: `generated/ttrun` (created if needed). |
| `--help` | `-h` | No | Print usage and exit (does not require MPI work beyond parsing). |

Example:

```bash
mpirun -np <N> <mpi-args> \
  /path/to/build/tools/scaleout/generate_rank_bindings \
  --mesh-graph-descriptor /absolute/or/repo/path/to/mesh_graph_descriptor.textproto \
  --output-dir /path/to/out
```

Use paths that every rank can read. Replace `<N>` and `<mpi-args>` with what your cluster and Tenstorrent workflows require (hosts, binding, etc.).

**Default usage (no Physical Grouping Descriptor on the command line):** you only need `-m` (and optionally `-o`). Omit `--physical-grouping-descriptor` and set up [environment](#default-physical-grouping-descriptor-omitting--p) so the tool can find a PGD file.

```bash
mpirun -np <N> <mpi-args> \
  /path/to/build/tools/scaleout/generate_rank_bindings \
  --mesh-graph-descriptor /absolute/or/repo/path/to/mesh_graph_descriptor.textproto \
  --output-dir /path/to/out
```

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

## Environment

These matter for discovery, PGD lookup, and optional phase-2 mapping:

| Variable | Role |
|----------|------|
| `TT_METAL_PHYSICAL_GROUPING_DESCRIPTOR_PATH` | If set to an existing file, that PGD is used when you omit `-p` (checked before cluster/arch search). |
| `TT_CLUSTER_NAME` | If set, enables cluster-specific PGD paths under `/data/scaleout_configs/...` and `TT_METAL_HOME/tests/tt_metal/tt_fabric/physical_groupings/<cluster>_physical_grouping_descriptor.textproto` (see [below](#default-physical-grouping-descriptor-omitting--p)). |
| `TT_METAL_HOME` | Base for repo-relative PGD search paths; defaults to `.` if unset. |
| `TT_METAL_MOCK_CLUSTER_DESC_PATH` | If set per rank, rank 0 can gather paths and emit **`phase2_mock_mapping.yaml`**; rankfile behavior may use a single local hostname for placement in mock scenarios. |

Implementation: `find_and_load_pgd` in `tools/scaleout/src/generate_rank_bindings.cpp`.

---

## Default Physical Grouping Descriptor (omitting `-p`)

If you do **not** pass `--physical-grouping-descriptor`, the tool picks a PGD file in this order; the **first path that exists** as a regular file is used.

1. **`TT_METAL_PHYSICAL_GROUPING_DESCRIPTOR_PATH`** — if the variable is set and points to an existing file, that file is the PGD (explicit default you control).

2. **Cluster name paths** (only if **`TT_CLUSTER_NAME`** is non-empty), under `TT_METAL_HOME` (default `.` if unset):
   - `/data/scaleout_configs/<TT_CLUSTER_NAME>/<TT_CLUSTER_NAME>_physical_grouping_descriptor.textproto`
   - `<TT_METAL_HOME>/tests/tt_metal/tt_fabric/physical_groupings/<TT_CLUSTER_NAME>_physical_grouping_descriptor.textproto`

3. **Architecture / cluster-type candidate** under the repo tree — **MetalContext** (discovered cluster type and architecture) picks a **first** filename to try at
   `<TT_METAL_HOME>/tests/tt_metal/tt_fabric/physical_groupings/<filename>`:
   - Galaxy + Wormhole B0 → `wh_galaxy_physical_grouping_descriptor.textproto`
   - Blackhole Galaxy + Blackhole → `bh_galaxy_physical_grouping_descriptor.textproto`
   - T3K + Wormhole B0 → `wh_t3k_physical_grouping_descriptor.textproto`
   - Any other combination → `default_physical_grouping_descriptor.textproto` (only this path is attempted for step 3; there is no separate “specialized then default” for this case).

4. **Generic default fallback** — if step 3 chose one of the **specialized** filenames above (not already `default_physical_grouping_descriptor.textproto`), the tool **also** tries
   `<TT_METAL_HOME>/tests/tt_metal/tt_fabric/physical_groupings/default_physical_grouping_descriptor.textproto`
   so a missing specialized file does not fail until the default has been checked.

The **first path in this combined search order that exists** as a regular file is used. If every attempted path is missing, the tool fails with an error listing what it searched (including the default when step 4 applied).

**Typical local default:** with `TT_METAL_HOME` pointing at your checkout, discovery often selects a specialized PGD when it exists; otherwise step 4 supplies `default_physical_grouping_descriptor.textproto`. If MetalContext maps to “any other combination,” only the default filename from step 3 is tried for repo-relative PGDs.

---

## Troubleshooting

- **Topology mapping fails:** Check MGD/PGD consistency with the real or mock cluster and PSD; logs use `LogFabric`.
- **No `phase2_mock_mapping.yaml`:** Normal if `TT_METAL_MOCK_CLUSTER_DESC_PATH` was not set (or paths were not gathered).
- **Wrong device visibility:** Bindings derive `TT_VISIBLE_DEVICES` from the cluster and mapping; verify MGD and PSD match your allocation.

For a higher-level description of what the tool does in the codebase and what landed in a given change, see the associated pull request description.
