# r01-b01-a03: read gamma on the idle writer RISC (BRISC/NoC0) at kernel start, rotating each worker's start tile across DRAM banks, so the NCRISC input read carries no gamma traffic

## Motivation
Parent r01-b01-a02 (0.8994) did two things on NCRISC: a trid-pipelined deep input read, plus the broadcast-gamma
face-row reads interleaved between input blocks. Its profile showed:
- Early gamma pays off: x*gamma hid fully under the AG and post-AG dropped 8.1 -> 5.7 us at h7168 (-2.4 us).
- The input row finished 3-7 us LATER (R_INPUT h7168 7.4 -> 15.7 us), because the 2 x num_tile_cols gamma face-row
  reads sat on the input push path.
r01-b04-a02 saw the same thing (0.8397) and measured ~70 ns per 64 B gamma read whether it was issued deep or not.
Every one of the 20 workers reads the SAME gamma pages in the SAME order, so all 20 hit one DRAM bank at a time.
Meanwhile BRISC (the worker writer) does nothing from the scalar setup (~0.5 us) until PRE pushes the stat tile
(~5-9 us). r01-b02-a02 showed the deep trid input read alone is worth ~+3.4% (R_INPUT -2.3 us at h7168).

## Mechanism
- `kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp`: new trailing CT args (weight_cb, writer_gamma flag,
  weight_bcast_tiles, weight TensorAccessorArgs) and common RT arg 3 = weight addr. When `writer_gamma`, right after
  the scalar setup the writer reserves the whole weight_cb row and issues ALL face-row reads (face_00 row 0 +
  face_01 row 0 per tile) with no barrier. Worker `my_slot` starts at tile `my_slot % weight_bcast_tiles` and wraps,
  so at any moment the 20 workers read 20 different gamma pages (spread across banks) instead of one. After the
  first row's stick push (so the AG start is never delayed), it does one read barrier and pushes the whole row.
- `kernels/dataflow/dit_rmsnorm_fused_reader.cpp`: new trailing CT flag `writer_gamma` (after the recip accessor).
  When set, the reader treats the weight as already pushed: the pipelined input read runs with `with_weight=false`
  and the deferred weight read is skipped. The parent's trid-pipelined input read is kept unchanged.
- `device/dit_fused_distributed_rmsnorm_program_factory.cpp`: `writer_gamma = use_mux && broadcast weight &&
  !streaming_low_l1`; pass the flag to both kernels, the weight accessor to the worker writer, and weight_addr as
  writer common arg 3 (also refreshed in override_runtime_arguments).
Compute is unchanged (it already waits on weight_cb cumulatively per block, so it doesn't matter which RISC produces it).

## Why this is not a repeat
- This repairs r01-b01-a02 (parent). The bug is that gamma issue and DRAM traffic sat on NCRISC between input blocks.
  This node moves them to a different RISC and NoC, and de-phases them across workers.
- r01-b04-a02 interleaved gamma the same way (flawed). r01-b01-a01 read gamma after the input on NCRISC (late gamma).
  No node has read gamma on BRISC or rotated the gamma read order per worker. This is the parent reflection's
  suggestion #1, plus the bank rotation.

## Expected effect and risk
If gamma no longer slows the input: R_INPUT goes back to the deep-read ~5.5 us at h7168 (b02-a02), and gamma is
resident at PRE end, so x*gamma hides under the AG (post-AG ~5.7 us as in the parent). Expect h7168 ~21-22 us,
h3584 ~14-14.5 us, score ~1.15-1.2.
Risks:
- The gamma reads still share DRAM with the input read (now on the other NoC). If DRAM is the real limit, R_INPUT
  will grow. Check R_INPUT end vs b02-a02.
- If gamma isn't done by PRE end, x*gamma waits. Check W_PUSH end vs TRISC timing.
- Hang if the weight_cb producer/consumer accounting is wrong. Accuracy should be bit-identical.
