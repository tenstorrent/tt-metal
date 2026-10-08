# r01-b02-a01: column-split each tile-row across multiple worker cores (more cores, same single AG round)

## Motivation
Baseline (reports/baseline_1, device 0, kimi-k2-7, 20 tile-rows x 56 local tile-cols) zone timeline
(ns from kernel start): 20 workers (one tile-row each) + 1 forwarder = 21 of 120 cores.
- R_INPUT 0 -> ~8000 (56 tiles/worker; reader barriers every 4-tile block -> latency bound, ~15 GB/s/core)
- W_PUSH ends 8400-10100 (PRE tracks the read)
- F_COLLECT ends ~10000, F_FABRIC 10100 -> 12300, W_AGWAIT ends ~12800
- TRISC ends ~22000 (POST = x*rms -> fp32 intermediate, *weight -> output: ~9 us for 56 tiles)
- W_DRAIN ends 23000-25700
Everything except the ~4 us AG section scales with tiles-per-worker, and only 20 of ~119 cores
work. All four shapes have only 20 tile-rows, so the op is badly under-parallelised.

## Mechanism
When the AG path has a single round (num_tile_rows <= worker cap) and the configuration is plain
(RMS, whole-row norm, 1 head per device, no RoPE, broadcast or no weight/bias), split each tile-row
into `col_split` equal column groups, one worker each (col_split = largest of 2..cap/rows that
divides num_tile_cols and keeps >= 8 tiles per worker; 28->2, 32->2, 48->3, 56->2).
- Each worker computes a partial sum-of-squares over its column slice and pushes its own stick
  (slot = row*col_split + g) to the forwarder, exactly like today (forwarder unchanged: group_size
  just grows to rows*col_split sticks; one packet, one fabric round).
- After the go-sem, each worker reads the col_split sibling sticks from every device
  (ring_size*col_split sticks) into stats_transposed_gathered_cb. The compute kernel is unchanged:
  its CT stats_tiles_cols becomes ring_size*col_split (still even) and its local num_tile_cols the
  slice width, so 1/H = 1/(local_cols*32*ring*col_split) is still the full hidden size.
- Reader: input tile index = row*full_cols + col_offset; broadcast weight/bias page += col_offset
  (new reader RT args col_offset, row_stride_tiles).
- Worker writer: output col += col_offset; gather loop over col_split slots (new RT args).
- Stats DRAM buffer geometry is unchanged (max_rounds stays 1, pages = num_forwarders), so
  compute_sizing / create_stats_buffer need no change; the decision lives in create_at only.
Files: device/dit_fused_distributed_rmsnorm_program_factory.cpp,
kernels/dataflow/dit_rmsnorm_fused_reader.cpp, kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp.

## Why this is not a repeat
No previous nodes exist (round 1, first attempt). It is structurally different from the obvious
local tweaks (math fidelity, reader barrier depth, CB depth): it changes the work decomposition
and is orthogonal to / composable with them.

## Expected effect and risk
Per-worker tiles halve (third for glm), so read+PRE, POST and drain should all shrink roughly
proportionally; AG (~4 us) stays. Expect kimi-k2-7 ~26 -> ~17 us, glm ~23.5 -> ~14 us,
deepseek/kimi-k3 ~17-18 -> ~12-13 us; aggregate DRAM bandwidth may cap the gain. Fabric packet
grows from 2.5 KB to 5-7.5 KB (small extra fabric time).
Risks: wrong slot/sibling mapping -> PCC failure (wrong rms per column group); wrong col offset ->
garbage output; forwarder present_count mismatch -> hang (tool reports `hang`). Accuracy should
be identical in kind to baseline (same fp32 partial sums, just added in a different order).
