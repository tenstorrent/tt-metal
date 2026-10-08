# r01-b02-a04: split the output drain across both NoCs: the idle reader (NCRISC/NoC1) writes the odd output blocks while the writer (BRISC/NoC0) writes the even ones

## Motivation
On this lineage (parent r01-b02-a03, 1.1721) the kernel end is set by the output drain, not by compute. Parent profile,
h7168, µs from kernel start: AG wait end ~11.1, TRISC end 16.8-17.3, W_DRAIN end 18.5-20.6. h3584: TRISC end ~11.0,
W_DRAIN end 12.2-12.9. The drain writes 20 cores x 112 KB = 2.24 MB/chip from AG end to drain end (~9.5 µs), which
is ~235 GB/s, about half of the ~400 GB/s the deep input read gets on NoC1 (r01-b02-a02, r01-b03-a01).
Every node shows the same write ceiling regardless of core count:
- r01-b04-a01: ~210 GB/s with 20 cores, latency-bound per-tile writes.
- r01-b02-a02: one flush per row instead of per block did nothing, so flush serialization is not the cause.
- r01-b03-a02: ~200 GB/s with 80 cores, so more writers don't help.
- r01-b03-a03: bank de-phasing cut the drain tail only 0.4-0.6 µs (not bank-bound). The drain end grew with core
  x/y, which points to NoC0 write-path congestion: all writers are BRISC on NoC0.
Meanwhile NCRISC (reader, NoC1) is idle from its gamma read end (~4.4 µs h3584, ~8.5 µs h7168) to kernel end.
Every reflection lists "split the drain across both NoCs" as a next step (b02-a02 #3, b03-a03 #1, b04-a01 #1,
b04-a03 #2). No node has tried it.

## Mechanism
- `kernels/dataflow/dit_rmsnorm_fused_reader.cpp`: new trailing CT args after the recip accessor: `reader_drain`,
  output_cb, total_num_tile_rows, drain_sem_id, plus the output TensorAccessorArgs. Reader common arg 6 = output addr.
  When `reader_drain`, after the row loop (all input and side-input reads done) the reader waits cumulatively on
  output_cb for each ODD block and NoC-writes its valid tiles to the output (same out_idx math as the writer),
  without popping. Then it does one write barrier and increments a local `drain_sem` (self-targeted NoC atomic).
- `kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp`: new trailing CT args `reader_drain`, drain_sem_id. In the
  deep drain, BRISC still waits cumulatively on every block but writes only the EVEN blocks. Before `pop_front` it
  waits until drain_sem reaches the row count. The reader must have finished all its wait_fronts before the shared
  acked counter moves, otherwise its cumulative wait could underflow. drain_sem is reset to 0 at the end, for
  trace replay.
- `device/dit_fused_distributed_rmsnorm_program_factory.cpp`: `reader_drain = use_mux && !block_major_post &&
  num_tile_rows_per_worker == 1`. With one row per worker the reader's drain cannot delay a later row's input read.
  Create a worker-core semaphore and append the CT args to both kernels. Set reader common arg 6 = output_addr and
  refresh it in override_runtime_arguments.
Compute, CB sizes and the AG protocol are unchanged.

## Why this is not a repeat
- r01-b02-a02 changed only the flush granularity on BRISC (neutral). r01-b03-a03 changed the bank order (small).
  Neither changed which NoC carries the writes. This node moves half the write traffic to NoC1, whose routes go
  -x/-y, so the NoC0 links toward the DRAM columns carry half the bytes.
- b01-a03 and b04-a03 moved gamma onto BRISC, which is a different lever (gamma timing). This lineage keeps gamma on
  NCRISC after the input, which ends well before the AG, so the reader is free for the drain.

## Expected effect and risk
If the ~235 GB/s ceiling is the NoC0 path: drain throughput up to ~1.5-2x, so the drain tail after TRISC end
(1.2-3.5 µs on h7168, 1.2-1.9 µs on h3584) mostly disappears and the kernel end approaches TRISC end. That is about
-1.5 to -3 µs per shape, score ~1.22-1.28. If the ceiling is DRAM write bandwidth, it will be neutral. Check
W_DRAIN/R_DRAIN end vs TRISC end in the report.
Risks: hang if the drain_sem / CB-wait protocol is wrong (fail_class=hang). Wrong output if block ownership or
out_idx is wrong (PCC fail). NCRISC has no other work after the gamma read, so it has nothing to delay.
Accuracy should be bit-identical (same data, different NoC).
