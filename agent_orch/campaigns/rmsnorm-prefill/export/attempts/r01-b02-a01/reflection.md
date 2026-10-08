# r01-b02-a01 result: 1.0025 (ok) — mechanism was INERT (never engaged)

## What happened vs expected
Expected 40-60 workers per chip (column split of each tile-row). The profiled run still shows
CORE COUNT = 21 (20 workers + 1 forwarder) on every shape, i.e. pick_col_split returned 1 and the
op ran the baseline decomposition. Per-shape speedups (1.017 / 0.997 / 0.999 / 0.997) are noise
(only extra RT-arg reads and an index add changed on the executed path). PCC/max_abs identical
to baseline, which at least shows the reader/writer RT-arg + offset plumbing is harmless at
col_split=1.

## Why (best explanation, with profiler evidence)
The split is gated by derive_worker_cap(), which clamps to sticks_per_packet * num_forwarders.
On this BH 1D-fabric config get_tt_fabric_max_payload_size_bytes() = 4352 B
(FabricEriscDatamoverBuilder::default_packet_payload_size_bytes, Bfp8 tile*4), so
sticks_per_packet = 4352/128 = 34 and, with num_links=1 (one forwarder), cap = 34 — NOT the 64 the
factory comments suggest. cap / num_tile_rows = 34/20 = 1 -> no split. With one forwarder and a
single fabric packet per round, at most 34 sticks can be gathered, so "one stick per (row, column
group)" can never fit 20 rows x 2 groups = 40 sticks. The idea as designed is infeasible here
without changing how sticks are produced (the forwarder is outside allowed_paths).

## Classification
repairable failure (bug: eligibility/cap — the one-stick-per-worker design exceeds the 34-stick
fabric packet; the mechanism never ran). Not evidence against column-splitting.

## What a child of this node should try next
Keep this node's plumbing (reader col_offset/row_stride RT args, writer col_offset, factory
num_workers = rows*col_split, worker->row mapping) but PRE-REDUCE siblings on-chip so the
forwarder still sees one stick per tile-row (20 sticks, fits 34):
- Choose col_split from grid budget only (e.g. 2-3; cores = rows*col_split + 1 <= ~108), not from
  derive_worker_cap's packet clamp; forwarder group = the row leaders only (group_size = rows,
  forwarder worker list = leaders, slot = row).
- Non-leader writer: NoC-write its 128 B transposed stick into a grid-uniform scratch CB on its
  row leader at offset g*128, then sem-inc the leader (new grid-wide semaphore).
- Leader writer: wait for col_split-1 sibling arrivals, add the col_split sticks (32 fp32 each)
  on the RISC in software into its own stick, push to the forwarder as today.
- Go: leader waits go-sem, then incs each sibling's go-sem (or forwarder rt list includes all
  workers — forwarder only incs group workers, so use leader fan-out). All workers then read the
  same row slot (slot = row) from all ring_size devices: stats_tiles_cols stays ring_size, and
  1/H must use full width: compute's recip_h_full uses num_tile_cols*32*stats_tiles_cols, so
  either pass full_tile_cols to that formula or add a CT multiplier (compute kernel change).
- Alternative cheaper first step if the above is too big: fix the reader's per-4-tile
  read barrier (R_INPUT ~8 us for 56 tiles on kimi-k2-7, latency bound) — issue the whole row
  under one barrier — and/or reduce POST cost (HiFi4 eltwise muls, two passes via fp32
  intermediate: ~9 us on kimi-k2-7).
Baseline zone timeline (kimi-k2-7, ns): R_INPUT 0-8000, W_PUSH end 8400-10100, F_FABRIC
10100-12300, AG wait end ~12800, TRISC end ~22000, W_DRAIN end 23000-25700.
