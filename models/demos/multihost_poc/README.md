# Multi-host proof-of-concept (TTNN report feature)

The smallest possible multi-host run that exercises the TTNN report feature across
**2 Blackhole hosts**. It is the lightweight stand-in for "run DeepSeek in multi-host
and look at the report", without any of the DeepSeek weight-loading, size, or runtime.

It does just enough to be representative of a real multi-host model run:

1. **MPI handshake** – the two hosts barrier and all-gather their ranks over MPI, the
   same coordination layer the real multi-host models use to stay in lockstep.
2. **A few TTNN ops per host** – `mul`, `add`, `gelu` on a tensor sharded across all
   16 devices (8 per host), so each host produces report data.
3. **One cross-host collective** – `all_gather` over the host boundary, so the two
   hosts actually move data to each other over TT-fabric.
4. **The report** – the autouse `ttnn_graph_report` fixture (in the repo-root
   `conftest.py`) captures each host's graph to its own JSON, then rank 0 merges them
   into a single multi-host report.

Tracks tenstorrent/ttnn-visualizer#1335.

## Files

| File | Purpose |
|------|---------|
| `test_multihost_poc.py` | The pytest test (`test_multihost_poc`). |
| `run_multihost_poc.sh` | Slurm batch script that allocates 2 BH nodes and launches the test with `tt-run`. |

## What "multi-host" means here (the short version)

- It is **SPMD**: the *same* program runs as one MPI process per host. Each process
  only drives its own local chips, but together they form one logical mesh.
- The hosts are launched and kept in sync by **MPI** (via `tt-run`, which wraps
  `mpirun`). Device-to-device data (the `all_gather`) travels over **TT-fabric**.
- You cannot run this with plain `pytest`; it has to be launched with `tt-run` so
  that two processes start, one per host. On a single host the test **skips**.

We use the **BH "loudbox" (`bh-lb-*`) nodes** because they are the smallest 2-host
config (2 individual machines, 8 chips each). The unified mesh is `1x16` across the
two hosts, defined by
`tests/tt_metal/distributed/config/bh_lbx2_1x16_rank_bindings.yaml`.

## How to run

### Option A: submit as a slurm batch job (recommended)

From the repo root (`/data/ctr-smountenay/tt-metal`):

```bash
sbatch models/demos/multihost_poc/run_multihost_poc.sh
```

Watch it:

```bash
squeue --me
tail -f multihost_poc_<jobid>.out
```

The script defaults to partition `bh_lb_B67` (had idle nodes in your `sinfo`). If it
is busy, edit the `#SBATCH --partition=` line (e.g. `bh_lb_B45`, `bh_lb_B67`).

### Option B: interactive (good for debugging)

```bash
# 1. grab 2 nodes
salloc -p bh_lb_B67 -N 2 --ntasks-per-node=1 --time=00:30:00

# 2. on the allocation, from the repo root:
export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) ARCH_NAME=blackhole
source python_env/bin/activate
export TTNN_CONFIG_OVERRIDES='{
    "enable_fast_runtime_mode": false,
    "enable_logging": true,
    "report_name": "multihost_poc",
    "enable_graph_report": true,
    "enable_detailed_buffer_report": true,
    "enable_detailed_tensor_report": false,
    "enable_comparison_mode": false
}'

HOSTFILE=$(mktemp)
scontrol show hostnames "$SLURM_JOB_NODELIST" | awk '{print $1" slots=1"}' > "$HOSTFILE"

tt-run \
  --rank-binding tests/tt_metal/distributed/config/bh_lbx2_1x16_rank_bindings.yaml \
  --mpi-args "--hostfile $HOSTFILE" \
  bash -c "source $TT_METAL_HOME/python_env/bin/activate && cd $TT_METAL_HOME && \
           pytest --disable-warnings -svv \
           models/demos/multihost_poc/test_multihost_poc.py::test_multihost_poc"
```

## Where the report lands

The merged multi-host report is written by **rank 0** to:

```
generated/ttnn/reports/multihost_poc_<month><dd>_<HHMM>/
├── db.sqlite                 # the merged report (has a `rank` column per op)
├── config_1_of_2.json        # rank 0 config snapshot
├── config_2_of_2.json        # rank 1 config snapshot
└── ...                       # cluster descriptor / mesh mapping sidecars
```

Point `ttnn-visualizer` at that directory. Because each host captured its own ops,
the report contains the operations from **both** hosts, tagged by rank — that is the
multi-host reporting behavior we want to validate.

> Note: `generated/` must be on a filesystem shared by both nodes (e.g. the NFS
> `/data/...` checkout you launch from) so rank 0 can read rank 1's capture before
> merging. The repo you launch from already satisfies this.

## Prerequisites

- `tt-metal` built (`./build_metal.sh`) and the venv created (`./create_venv.sh`),
  on the shared checkout both nodes can see.
- Passwordless SSH between the allocated nodes (required by `mpirun`).
- Only one `tt-run` job on a given set of nodes at a time.

## Troubleshooting

- **Test skipped / "only makes sense in a distributed environment"** – you ran it on
  a single host. It must be launched via `tt-run` across 2 nodes.
- **MPI cannot reach the other host** – check passwordless SSH and that the hostnames
  in the generated hostfile are reachable.
- **Stale chips / hangs** – reset the chips on the allocated nodes (`tt-smi`) and
  retry; make sure no other `tt-run` is using them.
- **Different partition / hardware** – this test assumes the `bh_lbx2_1x16` topology
  (2 loudboxes, 8 chips each). On other 2-host BH systems you'd swap the rank-binding
  YAML and mesh size accordingly.
