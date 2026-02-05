# Expand Input Gather to 64+64 Source Cores (A/B Grid Layout)

## Problem

The fused GatedLocalReduce+DownProj operation previously sourced input gather tiles from only the 112 matmul cores. This limited the maximum number of source cores per group to 56 (half of 112), which was insufficient for the production configuration of 8 expert collections × 8 K-tiles = 64 sources per group.

## Solution

We expanded source core assignment from matmul-cores-only to a full A/B grid layout that recruits **all 128 non-sender cores** in the 13×10 grid — including DRAM workers and phantom cores — as input gather senders. This required **no changes to the op or kernel**; only the test file was modified.

The kernel already handled this via compile-time flags: DRAM and phantom cores with `is_input_gather_sender_g1/g2=1` execute `setup_sharded_buffer` + input gather send on NCRISC, then wait on mcast semaphores. They skip matmul, add, and output gather since those flags remain 0.

## A/B Grid Layout

```
                    Input Gather: 64+64 A/B Source Core Layout
                    ==========================================

        Col:  0    1    2    3    4    5    6    7    8    9   10   11   12
            ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
  Row 0:    │ A* │ A  │ A  │ A* │ B  │ B  │ B  │ A  │ A  │ A  │ B  │ B  │ B° │
            ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
  Row 1:    │ A  │ A  │ A  │ A  │ B  │ B  │ B  │ A  │ A  │ A  │ B  │ B  │ B° │
            ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
  Row 2:    │ A  │ A  │ A  │ A  │ B  │ B  │ B  │ A  │ A  │ A  │ B  │ B  │ B° │
            ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
  Row 3:    │ A  │ A  │ A  │ A  │ B  │ B  │ B  │ A  │ A  │ A  │ B  │ B  │ B° │
            ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
  Row 4:    │ A  │ A  │ A  │ B  │ B  │ B  │ B  │ A  │ A  │ A  │ B  │ B  │ B° │
            ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
  Row 5:    │ A  │ A  │ A  │ B  │ B  │ B  │ B  │ A  │ A  │ A  │ B  │ B  │ B° │
            ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
  Row 6:    │ A  │ A  │ A  │ B  │ B  │ B  │ B  │ A  │ A  │ A  │ B  │ B  │ B° │
            ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
  Row 7:    │ A  │ A  │ A  │ B  │ B* │ B  │ B* │ A  │ A  │ A* │ B  │ B  │ B° │
            ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
  Row 8:    │ A  │ A  │ A  │ B  │ B  │ B  │ B  │ A  │ A  │ A  │ B  │ B  │    │
            ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
  Row 9:    │ A  │ A  │ A  │ B  │ B  │ B  │ B  │ A  │ A  │ A* │ B  │ B  │ M  │
            └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘

  Legend:
    A  = Group 1 source (matmul core)       ──┐
    A* = Group 1 source (DRAM worker)         ├── 64 cores total
    ───────────────────────────────────────────┘
    B  = Group 2 source (matmul core)       ──┐
    B* = Group 2 source (DRAM worker)         ├── 64 cores total
    B° = Group 2 source (phantom, col 12)     │
    ───────────────────────────────────────────┘
    M  = Sender/receiver core (12,9)
         = Excluded blank phantom (12,8)

  Core Breakdown:
  ┌──────────────────┬───────────┬───────────┐
  │ Role             │ A (grp 1) │ B (grp 2) │
  ├──────────────────┼───────────┼───────────┤
  │ DRAM workers     │     4     │     4     │
  │ Phantom (col 12) │     0     │     8     │
  │ Matmul cores     │    60     │    52     │
  ├──────────────────┼───────────┼───────────┤
  │ Total            │    64     │    64     │
  └──────────────────┴───────────┴───────────┘
```

## What Changed

**One file modified:** `test_gated_local_reduce_down_proj.py`

1. **`build_ab_grids()`** — New helper that partitions the 13×10 grid into two balanced groups of 64 cores each, following the A/B column assignment pattern above. Group A gets the left/middle columns, Group B gets the right columns plus all 8 phantoms in column 12.

2. **New test parameter** — `(tiles_per_k=8, K=256, N_per_core=64, bfloat8_b)` exercises the full 64+64 configuration with face-view enabled (8 collections × 8 K-tiles, both even).

3. **Conditional source selection** — When `num_sources_per_group == 64`, the test uses `build_ab_grids()` to shard source tensors across all core types. Smaller test cases continue using matmul-cores-only.

## What Did NOT Change

- **`op.py`** — The op derives source grids from tensor shard specs, so it works with any source core set automatically.
- **Kernel `.cpp`** — The kernel uses compile-time flags (`is_input_gather_sender_g1/g2`). DRAM/phantom cores with these flags execute the input gather send on NCRISC and skip matmul/add/output gather.

## Test Results

```
PASSED  test_gated_local_reduce_down_proj[8-256-64-DataType.BFLOAT8_B]
  PCC: 0.9998 (threshold: 0.97)
  Output: [1, 7168]  (112 matmul cores × 64 N_per_core)
  Source: 64 A + 64 B cores confirmed
  Face-view: enabled
```
