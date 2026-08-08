# All-Reduce Experiments

Correctness and performance tests for intra-galaxy and inter-galaxy all-reduce on Blackhole Galaxies.

## Scripts

| Script | Description |
|--------|-------------|
| `test_single_host_all_reduce.py` | Intra-galaxy all-reduce (32 devices, single host) |
| `test_multi_galaxy_all_reduce.py` | Local all-reduce + inter-galaxy ring all-reduce via sockets (4 galaxies) |
| `run_4galaxy.sh` | SLURM runner that launches either script across 4 galaxies via `tt-run` |

## Single-Galaxy (one host)

Run directly on a galaxy node:

```bash
# Allocate a single galaxy
srun -p <partition> --nodelist=bh-glx-c01u08 --pty /bin/bash

# Basic correctness check
python tt-train/all_reduce_experiments/test_single_host_all_reduce.py

# With timing (iteration 1 = warmup, iterations 2-10 = measured)
python tt-train/all_reduce_experiments/test_single_host_all_reduce.py --iterations 10

# Custom shape and topology
python tt-train/all_reduce_experiments/test_single_host_all_reduce.py \
    --per-chip-shape 1 1 128 4096 --topology ring --iterations 20
```

## Multi-Galaxy (4 hosts)

Requires a SLURM allocation with 4 galaxy nodes.

```bash
# Allocate 4 galaxies
salloc -p <partition> -N4 \
    --nodelist=bh-glx-c01u08,bh-glx-c01u02,bh-glx-c02u02,bh-glx-c02u08

# Correctness check (local + inter-galaxy all-reduce)
bash tt-train/all_reduce_experiments/run_4galaxy.sh

# With timing
bash tt-train/all_reduce_experiments/run_4galaxy.sh --iterations 10

# Custom tensor shape
bash tt-train/all_reduce_experiments/run_4galaxy.sh --per-chip-shape 1 1 64 2048 --iterations 10

# Run the single-host test on all 4 galaxies independently (no inter-galaxy comms)
TEST_SCRIPT=test_single_host_all_reduce.py bash tt-train/all_reduce_experiments/run_4galaxy.sh \
    --fabric-config 2D --iterations 10
```

Alternatively, submit as a batch job:

```bash
sbatch -p <partition> -N4 \
    --nodelist=bh-glx-c01u08,bh-glx-c01u02,bh-glx-c02u02,bh-glx-c02u08 \
    tt-train/all_reduce_experiments/run_4galaxy.sh --iterations 10
```

## Configuration

Files in `configs/`:

| File | Purpose |
|------|---------|
| `mgd.textproto` | Mesh Graph Descriptor: 4 galaxies (32x1 each) in a ring topology |
| `rank_bindings.yaml` | Maps MPI ranks 0-3 to mesh IDs 0-3 |
| `rankfile.txt` | Static MPI rankfile mapping ranks to physical hosts |

Edit `HOSTS` in `run_4galaxy.sh` and `configs/rankfile.txt` if your nodes differ.
