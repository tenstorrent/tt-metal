# r02-b04-a01: spread the 20 AG-path worker cores over all grid rows (2 full columns) instead of packing them row-major into 2 rows

## Motivation
Every node so far keeps the factory's row-major worker placement: `all_cores_vec[0..num_workers)`. On this
11x10 grid, the 20 workers sit in only two NoC rows. Per-core profile of the round root (r01-b04-a04, h7168,
device 1, call 58, µs from the first marker; virtual coords) shows: 11 workers at y=2 (x=1-7,10,11,13,14),
9 at y=3, forwarder at (13,3).

| x | 1 | 3 | 5 | 7 | 10 | 11 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|
| R_INPUT end (y=2) | 14.20 | 14.07 | 14.54 | 15.13 | 15.00 | 15.09 | 13.90 | 13.79 |
| W_DRAIN end (y=2) | 26.88 | 27.36 | 28.34 | 28.81 | 28.74 | 29.37 | 27.38 | 27.43 |

- Both the input read (NCRISC, NoC1) and the output drain (BRISC, NoC0) get slower with x, and y=3 is a
  little later than y=2. That is a per-row link-congestion gradient. All of a row's DRAM traffic (~1.5 MB/call
  on y=2 at h7168) runs along that row's x-links, and every core on a row shares them.
- The slow-read cores gate the AG start: F_COLLECT waits for the slowest W_PUSH (17.15, core (7,3)). Fast cores
  push at ~15.2. So ~1.3 µs of the AG start is read-congestion skew.
- The drain tail after TRISC end (25.5-25.9) is 1.0-3.7 µs and follows the same x gradient.
- Nodes r01-b02-a04, r01-b03-a04 and r01-b03-a03 attacked this with NoC splitting and bank de-phasing. Dual-NoC
  just mirrored the gradient, and bank rotation didn't fix the drain. Their reflections say the limit is the
  shared links, not the DRAM banks. r01-b02-a04 suggested "placing workers ... across" the grid as the better
  lever. Nobody has tried that.

## Mechanism
Host-only change in `dit_fused_distributed_rmsnorm_program_factory.cpp`. On the AG path (`use_mux`), place the
workers column-major over ncols = ceil(num_workers / grid_y) full logical columns, spread across the grid width
(col_k = grid_x-1 - k*floor(grid_x/ncols)). For 20 workers on 11x10 that is logical columns 10 and 5. So every
NoC row carries 2 workers instead of 9-11. Worker w keeps tile row w (same partition, slot and forwarder
grouping). The forwarder(s) take the first row-major cores not used by a worker. Full columns merge into one
CoreRange each, so dispatch stays cheap: 2 ranges, one kernel group, the same as now. The is_tp_1 path and the
fallback (if the columns can't hold the workers) keep the old row-major placement. Kernels are unchanged.

## Why this is not a repeat
- r01-b02-a04 and r01-b03-a04 split the drain across NoC0/NoC1 with the same co-located workers. That moves the
  congestion to the other direction. This change cuts the per-row load itself, for reads and writes alike.
- r01-b03-a03 rotated the DRAM bank order (bank-side de-phasing). This is link-side.
- The column-split nodes (r01-b02-a01, r01-b03-*) add cores. This keeps the 20+1 cores and only moves them.

## Expected effect and risk
- Expected: the input-read gradient flattens, so the slowest W_PUSH and the AG start come ~0.5-1.3 µs earlier on
  wide shapes. The drain tail after TRISC end shrinks by ~1-2 µs. Overall about -1.5 to -2.5 µs at h6144/h7168
  and -0.5 to -1 µs at h3584/h4096, score ~1.27-1.30.
- Risks: longer worker->forwarder hops for the stick push and go-sem (tens of ns). The vertical DRAM-column links
  might now be the limit, so the gain could be smaller. Accuracy is unaffected (same math and order). A hang is
  unlikely (the forwarder addresses workers by per-core coords). Judge it by the per-core R_INPUT end / W_DRAIN end
  spread with the same per-core table as above.
