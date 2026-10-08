# r01-b04-a02: trid-pipelined input read with gamma face-row reads interleaved per block

## Motivation
Parent r01-b04-a01 profile (`reports/r01-b04-a01`, chip 1, h7168 call 61441, µs from kernel start):
- R_INPUT 0.07 -> 7.56-8.85 (56 tiles/core). `read_input_pass` barriers after every 4-tile block, so only
  one 8 KB block is ever in flight: ~0.5 µs per block, latency-bound. 20 cores x 112 KB in ~7.5 µs is
  ~315 GB/s per chip. r01-b03-a01's 60-core run reached ~460 GB/s, so DRAM can deliver more.
- The broadcast gamma is read only AFTER the input row, with a per-block barrier (14 more round trips):
  NCRISC ends 15.4-16.8 µs, AFTER the AG wait ends (13.9-14.3). This node's x*gamma pre-pass (a01's
  mechanism) waits on cb_weight, so it does not actually hide under the AG on the wide shapes. Compute ends
  20.5-22.2, and W_DRAIN ends 21.6-25.2.
- h3584 (call 15361): R_INPUT ends 3.8-4.9 and NCRISC ends 7.7-8.8, vs. AG end 9.5-9.8. Here gamma is
  just in time.
Everything downstream (PRE, stick push, AG, POST, drain) is serialized behind the input read.

## Mechanism
Reader only (`kernels/dataflow/dit_rmsnorm_fused_reader.cpp`), for the resident INPUT_FIRST path
(non-streaming, which is the campaign config):
- Reserve the whole row in input_cb once. Issue each 4-tile block's reads tagged with its own NoC read
  transaction id (trid 1..14, rotating), keeping a lookahead of several blocks in flight. Then wait for
  just that block's trid (`async_read_barrier<TXN_ID>`) before pushing it, so PRE still consumes block by
  block, but DRAM latency overlaps across blocks.
- For the first row of a broadcast-weight worker, issue that block's gamma face-row reads (2 x 64 B per
  tile) right after each input block's reads, under a dedicated trid (15). After the last input block, wait
  on trid 15 once and push the whole gamma row. The existing deferred per-block-barriered weight read is
  skipped (weight_pushed=true). Compute already waits on weight cumulatively, and weight_cb holds one
  full row.
- Restore read trid 0 at the end.
Every other schedule (streaming, SPLIT, DEFER_ALL, per-token/per-batch affine) keeps the old code.

## Why this is not a repeat
- r01-b01-a01 batched the gamma read into one deep barrier but issued it after the still per-block-barriered
  input. Its reflection names the input pipelining as the biggest untried lever. Nobody has touched the
  input read.
- r01-b04-a01 (parent) left the reader untouched, which is why its x*gamma pass stalls on gamma at h6144/h7168.
- r01-b02/b03 changed work decomposition (more cores). This keeps 20 workers and fixes per-core read latency.

## Expected effect and risk
R_INPUT approaching the per-chip DRAM limit: h7168 ~7.5 -> ~5 µs, h3584 ~3.7 -> ~2.7 µs. Gamma is resident
right after the input, so the x*gamma pass fully hides under the AG on all shapes (recovers the ~1.5-2 µs
r01-b01-a01 got from its gamma batching at h6144/h7168). Expect roughly -2..-4 µs per shape. Score ~1.15-1.2.
Risks: NCRISC issue rate (~40 cycles per read; the gamma face-row reads double the read count) could cap
the gain. A trid bookkeeping mistake would show as an accuracy failure (block pushed before landing) or a hang.
Accuracy should be bit-identical (same data, same compute).
