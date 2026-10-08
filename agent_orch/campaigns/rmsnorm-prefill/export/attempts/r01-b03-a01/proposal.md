# r01-b03-a01: column-split each tile-row across k worker cores (leader combines partial sum-of-squares on the FPU)

## Motivation
Baseline profile (reports/baseline_1, device 0, per-zone timeline, µs from kernel start):

| shape | R_INPUT end | W_PUSH end (PRE done) | F_FABRIC | W_AGWAIT end | TRISC end | W_DRAIN end |
|---|---|---|---|---|---|---|
| h3584 (28 tiles/row) | 3.8-4.9 | 5.1-6.1 | 6.1-8.5 | 8.6-9.0 | 14.3-14.7 | 15.1-16.9 |
| h7168 (56 tiles/row) | 7.3-9.0 | 8.4-10.1 | 10.1-12.3 | 12.4-12.8 | 21.6-22.0 | 23.0-25.9 |

The test has only 20 tile-rows per chip (seq 640), so the op uses 20 worker cores + 1
forwarder = 21 of the 120 Tensix cores. Each worker owns one whole tile-row: it reads
28-56 tiles, computes PRE (x^2 + row reduce), and after the gather runs POST
(x*1/rms, *gamma) at ~200 ns/tile. The per-row cost scales linearly with
width (h3584 -> h7168 adds ~9 µs). That is per-core serial work, and ~100 cores
sit idle.

## Mechanism
Split each row's columns across k worker cores (k = floor(64 / row_workers), so k=3
for these shapes, 60 workers). Each worker handles a contiguous column slice of
ceil/floor(num_tile_cols/k) tiles. That means two kernel groups when the widths differ, since
num_tile_cols is a compile-time arg.

The fabric packet holds only 4352/128 = 34 sticks, so the k slices can't each push
their own stick. Instead:
- **Followers** (slice j>0) NoC-write their transposed row-0 partial stick into the
  row **leader**'s new `peer_partials_cb` (c_22, grid-uniform address) slot j-1, then
  inc the leader's new `peer_sem`.
- **Leader writer** waits peer_sem, pushes the peer tiles to compute. The **leader compute**
  sums them into its own transposed stat in DST (transpose_tile, then
  binary_dest_reuse ELWADD per peer) before packing the stick. The leader pushes
  one stick to the forwarder as before (forwarder kernel unchanged, group = leaders).
- After the go-sem, the leader re-raises the go-sem on its followers. All k
  workers read the row's ring_size gathered sticks (leader's slot) and run POST on their
  own slice.
- Reader/writer take `col_start` + full row stride as RT args. Compute gets the full
  width as a new CT arg so 1/H_full stays right.

Only enabled for: AG path, RMS, no rope, 1 head, broadcast (non per-token/per-batch)
affine, resident (non-streaming) layout, and row_workers*2 <= 64. Everything else gets k=1
and behaves exactly as before. compute_sizing / stats-buffer geometry is unchanged
(max_rounds and the leader count are the same).

Files: device/dit_fused_distributed_rmsnorm_program_factory.{cpp,hpp},
kernels/dataflow/dit_rmsnorm_fused_reader.cpp, dit_rmsnorm_fused_worker_writer.cpp,
kernels/compute/dit_rmsnorm_fused_compute.cpp.

## Why this is not a repeat
No prior nodes exist (round 1, empty history). The baseline's own worker-count
heuristic is "one worker per tile-row", and the comments say that balancing
row counts was tried. Splitting along columns, with an on-chip partial-sum
combine, is new.

## Expected effect and risk
Read+PRE and POST per core drop ~3x. Fixed costs remain: fabric ~2-3 µs, launch ~1 µs, the
extra leader combine hop ~0.5-1 µs, and aggregate DRAM bandwidth (2.2 MB read + 2.2 MB write
per chip on the widest shape). Expected roughly 26 -> 14-16 µs on h7168, 17 -> 11-12 µs on
h3584, so a score of maybe 1.4-1.6.
Risks: hang if the semaphore protocol is wrong (peer_sem / go relay), PCC drop if the
peer sum is wrong (shows as accuracy_fail), and DRAM contention from 60 cores eating the gain.
