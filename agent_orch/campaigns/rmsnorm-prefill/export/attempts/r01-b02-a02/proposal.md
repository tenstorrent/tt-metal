# r01-b02-a02: keep DRAM I/O deep: per-block transaction-ID input read and one flush per row in the output drain

## Motivation
Parent r01-b02-a01 runs the baseline decomposition (its column split never engaged: 21 cores). Its profile
(reports/r01-b02-a01, device 1, µs from kernel start):

| shape | R_INPUT end | W_PUSH end (PRE done) | AG wait end | TRISC end | W_DRAIN end |
|---|---|---|---|---|---|
| h3584 (28 tiles/core) | 3.7-4.7 | 5.0-6.1 | 8.8-9.1 | 14.2-14.9 | 15.2-16.7 |
| h7168 (56 tiles/core) | 7.5-8.5 | 8.7-9.8 | 12.7-13.1 | 21.7-22.5 | 22.5-26.2 |

1. **Input read is per-core latency bound.** `read_input_pass` barriers after every 4-tile (8 KB) block, so at
   most 4 reads are in flight per core: ~0.5 µs per block, ~15 GB/s per core, ~300 GB/s aggregate on 20 cores on
   every shape (same rate at 28 and 56 tiles). Its comment even says "deep read ... ONE barrier", but the code
   barriers per block. r01-b03-a01 (60 cores, 3x the in-flight reads) read the same 2.24 MB in ~5.5 µs
   (~400 GB/s), so there is headroom of ~2 µs on h7168 and ~1 µs on h3584. Everything downstream (PRE, stick push,
   AG, POST, drain) is serialized behind this read.
2. **Output drain flushes per block.** `W_DRAIN` waits for each 4-tile block, issues 4 writes, then
   `async_writes_flushed()` + `pop_front` before looking at the next block. output_cb already holds 2 padded rows
   (factory comment: "so the writer can deep-drain a whole row under ONE flush"), so the per-block flush only
   serializes the writer. The drain ends 0.1-4 µs after compute (h7168) and is the kernel-end critical path.

## Mechanism
- `kernels/dataflow/dit_rmsnorm_fused_reader.cpp`: for the resident INPUT_FIRST path (not streaming_low_l1), read
  the row with one `reserve_back(num_tile_cols)` and tag each block's reads with its own NoC read transaction id
  (`async_read<NocOptions::TXN_ID>`, trids 1..15, sliding window). Then, in block order,
  `async_read_barrier<TXN_ID>(trid)` + `push_back(block)`. All blocks are in flight at once (14 blocks <= 15 trids on
  the widest shape), and compute still gets block-granular pushes, so PRE starts on block 0 as early as before.
  The sticky packet tag is reset to 0 afterwards. Streaming / SPLIT / DEFER_ALL schedules keep the old per-block pass.
- `kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp`: new trailing CT arg `deep_drain` (= !block_major_post).
  When set, W_DRAIN waits cumulatively per block (`wait_front(col_tile + block_size)`), issues that block's writes
  with no flush, and does ONE `async_writes_flushed()` + `pop_front(padded row)` per row. Block-major keeps the old
  per-block loop (its output_cb is only 2 blocks).
- `device/dit_fused_distributed_rmsnorm_program_factory.cpp`: append `deep_drain` to the worker writer CT args.
No change to compute, CB sizes, the AG protocol, or the parent's column-split plumbing (still col_split=1).

## Why this is not a repeat
- r01-b01-a01 / r01-b04-a01 (best nodes) move the x*gamma multiply under the AG wait (compute). Neither touched the
  input read; b01 only batched the *weight* read. b01's and b04's reflections both name the input-read barrier and
  the per-block drain flush as the next levers. This attempt takes those, and is orthogonal to x*gamma, so they can
  stack in a later combination.
- r01-b02-a01 / r01-b03-a01 add cores (column split). This keeps 21 cores and the same dispatch cost (b03's skew
  problem came from more cores/kernel groups), and attacks per-core NoC depth instead.

## Expected effect and risk
R_INPUT: h7168 ~8 -> ~5.5-6 µs (then DRAM-bandwidth or PRE-compute bound, ~90 ns/tile), h3584 ~4 -> ~3 µs. The
whole pipeline shifts earlier by that amount. The drain change should shrink the post-compute tail (mostly wide
shapes). Expect ~1-2.5 µs per shape, score ~1.05-1.10.
Risks: wrong trid accounting -> compute reads tiles before they land (PCC fail) or hangs on a trid barrier; reserve
of a whole row must not wrap the input CB (it is 2 whole rows, so it can't). A missed cumulative wait in the drain
would write stale output (PCC fail). If DRAM bandwidth is the real limit, R_INPUT won't move and the score will be
in the noise; the R_INPUT / W_DRAIN zone ends in the report will tell which.
