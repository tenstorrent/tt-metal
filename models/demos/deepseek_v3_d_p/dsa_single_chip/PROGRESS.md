# DSA single-chip generic ops: progress and resume guide

**Goal (set 2026-11-10 by the user):** for each single-chip op of the DSA indexer chain, write a new
`ttnn.generic_op` (Python `ProgramDescriptor` + custom kernels) that is as fast as possible on **one fixed
shape** (see `SHAPES.md`: Kimi K3.1, Galaxy 8x4 per-chip view, S=5120). Nothing else matters.
Contract: same input/output shapes, dtypes and layouts as the existing op (internals are free).
Judged by the end-to-end DSA test, which the user runs themselves at the end.

## Resume checklist (read this first after any disconnect)
1. `cd /Users/malimpic/Projects/tt-metal-1011-1 && git switch dsa_single_chip_ops && git log --oneline -5`
2. Read the status table and the experiment log below; the newest log entry says what was in flight.
3. Sync to bh-47 and run (details in "Workflow"): `scripts/sync_bh47.sh` then `scripts/run_bh47.sh <op> [args]`.
4. Long jobs run in background hc terminals (`hc ls` shows them; `hc read bh-47/tN --lines 50`).
5. Commit after every meaningful result and update this file (status table + log) in the same commit.
6. herdr task id for this work: 3 (`herdr-task show 3`).

## Environment
- Dev machine: Mac (this worktree, branch `dsa_single_chip_ops`, from `llk_helper_library`).
- Hardware: bh-47, one Blackhole p100a. Repo there: `/localdev/malimpic/tt-metal` (same branch via ssh push).
- bh-47 has no web access: submodules (tracy, umd, tt-cluster-descriptors) were fetched on the Mac via HTTPS
  and copied over; CPM deps come from `/localdev/malimpic/tt-metal-cutoffs/.cpmcache` (another tree of the
  user's, on branch cutoff-sweep-p2, **do not modify it**; its python_env is reusable read-only).
- Git: use HTTPS for any fetch from GitHub (never change the origin remote). Do not push to GitHub.
- Single-chip emulation of mesh-aware ops: the test runs ONE chip's view (e.g. `mesh_partition` = fixed slice,
  `update_padded_kv_cache` = write at the chosen chip's block-cyclic window).

## Ops and status
| # | Op (existing) | New op file | Status | Best vs existing | Notes |
|---|---|---|---|---|---|
| K1 | layer_norm (k-norm) | ops/k_layer_norm.py | todo | | |
| K3 | k x Hadamard matmul | ops/k_hadamard.py | todo | | |
| K4 | update_padded_kv_cache | ops/kv_cache_write.py | todo | | |
| Q1 | mesh_partition (TP slice) | ops/qr_partition.py | todo | | |
| Q2 | wq_b linear | ops/wq_b_linear.py | todo | | |
| Q3 | nlp_create_qkv_heads | ops/qkv_heads.py | todo | | |
| Q5 | q x Hadamard matmul | ops/q_hadamard.py | todo | | |
| T1 | topk_large_indices | ops/topk_indices.py | todo | | |

Order of work: K1, K3, K4, Q1, Q2, Q3, Q5, T1 (cheap ones first; T1 and Q2 are likely the big wins).
Stop on an op only when further ideas are exhausted; record why in the log.

## Workflow
- Shapes/configs of the existing ops: `SHAPES.md`. Each op has `tests/test_<op>.py` that builds the inputs
  at the exact shape, runs the existing op and the new op, checks equality/PCC, and times both (device
  profiler or traced replay, median of N).
- Build on bh-47: see log. Scripts live in `scripts/`.

## Experiment log (newest last)
- 2026-11-10: branch created, shapes derived (`SHAPES.md`), submodules populated on Mac. Build on bh-47 not started yet.
