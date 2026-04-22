# Pipeline Parallel Training

This example demonstrates pipeline parallel training of large transformer models (Llama 8B, 70B, 405B) across multiple Tenstorrent Blackhole Galaxy units using TTNN fabric for inter-host communication.

In pipeline parallelism, the model is split into sequential stages across ranks. Rank 0 processes input embeddings and early layers, intermediate ranks process middle layers, and the final rank computes the output and loss. Activations flow forward through the pipeline; gradients flow backward. Inter-stage communication uses either MPI or TTNN fabric sockets depending on the configuration.

## Files

| File | Description |
|------|-------------|
| `runner.sh` | SLURM batch script that configures the environment, selects the workload config, generates an MPI rankfile, and launches training via `tt-run`. |
| `training.py` | Main entry point. Loads the YAML config, initializes the device mesh and distributed context, runs a round-robin inter-rank connectivity warmup, constructs the model and optimizer, then calls the training loop. |
| `trainer.py` | Core training loop. Implements the pipeline parallel forward and backward passes with gradient accumulation. Rank 0 sends target labels directly to the final rank via socket; only the final rank computes and logs loss. |
| `make_rankfile.py` | Generates an MPI rankfile from the SLURM hostnames provided on stdin. Sorts hosts into canonical snake/zigzag order (alternating ascending/descending unit numbers per host group) to match physical Galaxy interconnect topology. |
| `ttnn_fabric_verification.py` | Standalone diagnostic tool that opens the device mesh, initializes the distributed context, and verifies point-to-point tensor send/recv over the TTNN fabric in a ring topology. Not invoked during normal training; run manually to validate hardware connectivity. |
| `requirements.txt` | Python dependencies: `datasets`, `transformers`, `tqdm`, `numpy`, `loguru`, `ml_dtypes`. |

## Running with sbatch

The `runner.sh` script contains embedded SBATCH directives and is submitted directly with `sbatch`. Before submitting, open `runner.sh` and set the `WORKLOAD` variable to your desired workload (see [Workloads](#workloads) below).

```bash
# Single-Galaxy workloads (llama8b): 1 node
sbatch --nodes=1 runner.sh

# Multi-Galaxy workloads (llama70b_4stage, llama70b_16stage, llama405b): 4 nodes
sbatch --nodes=4 runner.sh
```

### What the script does

1. **Environment setup** - Sources the `tt-metal` Python virtualenv, sets `PYTHONPATH`, `LD_LIBRARY_PATH` (OpenMPI), and `TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS`.
2. **Galaxy reset** - Runs `srun tt-smi -glx_reset` across all allocated nodes to put all Galaxy units into a clean state before training.
3. **Rankfile generation** - Pipes `scontrol show hostnames` (the SLURM-allocated host list) into `make_rankfile.py`, which sorts hosts into canonical order and writes an MPI rankfile to `/tmp/rankfile.txt`. The file is written to `/tmp` to avoid path-parsing issues with hostnames containing dashes.
4. **Launch** - Calls `tt-run` with `--mpi-args` pointing to the rankfile and `--rank-binding` pointing to the appropriate `rank_bindings.yaml` for the selected configuration.

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TT_METAL_HOME` | `/data/$USER/tt-metal` | Root of the tt-metal installation. Override before submitting if your install is elsewhere. |
| `WORKLOAD` | `llama70b_16stage` | Set inside `runner.sh` to select the model and hardware configuration. |

## Workloads

| `WORKLOAD` value | Model | Configuration | Nodes | Ranks | YAML config |
|------------------|-------|---------------|-------|-------|-------------|
| `llama8b` | Llama 8B | `1galaxy_pp4` | 1 | 4 | `training_shakespeare_llama8b_intrahost_pp4.yaml` |
| `llama70b_4stage` | Llama 70B | `4galaxy_pp4` | 4 | 4 | `training_shakespeare_llama70b_pp4_tp32_fabric_galaxy.yaml` |
| `llama70b_16stage` | Llama 70B | `4galaxy_pp16` | 4 | 16 | `training_shakespeare_llama70b_pp16_fabric_galaxy.yaml` |
| `llama405b` | Llama 405B | `4galaxy_pp16` | 4 | 16 | `training_shakespeare_llama405b_pp_fabric.yaml` |

## Configurations

Each configuration directory under `configurations/` contains two files:

- `mgd.textproto` - Mesh Graph Descriptor: defines the hardware mesh topology (device layout, number of ethernet channels, ring connections between mesh instances).
- `rank_bindings.yaml` - Maps each MPI rank to a mesh instance and sets per-rank environment overrides (primarily `TT_VISIBLE_DEVICES` to control which physical devices a rank owns).

---

### `1galaxy_pp4`

**Use case:** Llama 8B, 4-stage pipeline parallelism on a single Galaxy unit.

**Hardware:** 1 host, 1 Galaxy (32 Blackhole devices arranged as 4 rows x 8 columns).

**Mesh topology (`mgd.textproto`):**
- Mesh descriptor `M0`: `1x8` device topology (one row of a Galaxy), single host.
- Graph `G0`: 4 instances of `M0` (mesh IDs 0-3), connected in a bidirectional ring with **8 ethernet channels** per link.

**Rank bindings (`rank_bindings.yaml`):**
- 4 ranks, all on the same host (`mesh_host_rank: 0`).
- Each rank owns one Galaxy row via `TT_VISIBLE_DEVICES`, selecting 8 devices. The device indices interleave rows to match the physical Galaxy wiring:

  | Rank | Devices |
  |------|---------|
  | 0 | `0,1,2,3,11,10,9,8` |
  | 1 | `4,5,6,7,15,14,13,12` |
  | 2 | `28,29,30,31,23,22,21,20` |
  | 3 | `24,25,26,27,19,18,17,16` |

---

### `4galaxy_pp4`

**Use case:** Llama 70B, 4-stage pipeline parallelism with tensor parallelism across 4 Galaxy units.

**Hardware:** 4 hosts, 4 Galaxies (32 Blackhole devices each).

**Mesh topology (`mgd.textproto`):**
- Mesh descriptor `Galaxy_4x8`: `1x32` device topology (full Galaxy), single host.
- Graph `G0`: 4 instances (mesh IDs 0-3), connected in a bidirectional ring with **4 ethernet channels** per inter-Galaxy link.

**Rank bindings (`rank_bindings.yaml`):**
- 4 ranks, one per host. Each rank owns an entire Galaxy (all 32 devices); no `TT_VISIBLE_DEVICES` override is needed since each rank runs on a dedicated host.
- Each rank sets a unique `TT_METAL_PROFILER_DIR` for profiler output isolation.

  | Rank | Galaxy | Host |
  |------|--------|------|
  | 0 | `mesh_id: 0` | Host 0 |
  | 1 | `mesh_id: 1` | Host 1 |
  | 2 | `mesh_id: 2` | Host 2 |
  | 3 | `mesh_id: 3` | Host 3 |

---

### `4galaxy_pp16`

**Use case:** Llama 70B (16-stage) and Llama 405B, 16-stage pipeline parallelism across 4 Galaxy units.

**Hardware:** 4 hosts, 4 Galaxies. Each Galaxy contributes 4 pipeline stages (4 ranks x 8 devices per rank).

**Mesh topology (`mgd.textproto`):**
- Mesh descriptor `M0`: `1x8` device topology (one row of a Galaxy), single host.
- Graph `G0`: 16 instances of `M0` (mesh IDs 0-15), connected sequentially in a bidirectional ring with **4 ethernet channels** per link (0-1, 1-2, ..., 14-15, 15-0).

**Rank bindings (`rank_bindings.yaml`):**
- 16 ranks across 4 hosts (4 ranks per host, `RANKS_PER_HOST=4` in `runner.sh`).
- Within each Galaxy, ranks select non-overlapping rows of 8 devices via `TT_VISIBLE_DEVICES`. The device index pattern repeats across hosts since each rank runs on its own host:

  | Ranks | Devices per rank |
  |-------|-----------------|
  | 0, 4, 8, 12 | `0-7` |
  | 1, 5, 9, 13 | `24-31` |
  | 2, 6, 10, 14 | `16-23` |
  | 3, 7, 11, 15 | `8-15` |

  The non-sequential device ordering (0-7, 24-31, 16-23, 8-15) follows the physical Galaxy row topology to ensure that adjacent pipeline stages use physically connected rows for optimal inter-stage bandwidth.
