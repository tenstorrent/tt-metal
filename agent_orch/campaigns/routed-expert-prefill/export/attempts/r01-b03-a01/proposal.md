# r01-b03-a01: one-block DRAM read-ahead of weight blocks on the weight-sender cores

## Motivation
All benchmark shapes run with tiny per-expert loads (68-272 tokens -> 3-9 tile-rows), so the
runtime picker gives per_core_M = 1 for nearly every expert (GPT-OSS: 1 or 2). At per_core_M = 1
the op is not compute-bound:

- The op's own single-expert perf table (test_single_routed_expert_perf.py, kimi_k2_7 ND-sharded):
  128 tok (M=1) 104 us, 512 tok (M=2) 156 us, 768 (M=3) 224 us, 1024 (M=4) 294 us. The marginal
  cost of one more tile-row per core is ~70 us (= compute, ~23 cycles/tile-matmul), yet M=1 costs
  104 us. So ~30% of the M=1 time is weight delivery the compute is waiting on.
- Baseline here: DSv3/Kimi-K2.7 ~100 us per expert, Mistral/DSv4-flash ~65 us, i.e. ~19 us fixed
  + ~27 cycles per weight tile per column. Per column the sender delivers 2.35 MB/expert at
  ~23 GB/s (aggregate ~250 GB/s against the ~370 GB/s the ND-shard docstring quotes).

Reading the dataflow kernels, the weight pipeline on the sender core of each N-column is serial
per K-block, so DRAM latency is paid every block instead of being overlapped with the multicast:

- gate/up phase (IN1_WRITER_MCAST): the writer (BRISC) does `wait up_go(k) -> read up(k) ->
  barrier -> mcast gate+up(k) 110 KB -> flush -> valid -> wait up_go(k+1) -> read up(k+1) ...`.
  The `up` read of block k+1 never overlaps the 110 KB multicast of block k.
- down phase: the reader (NCRISC) does `reserve(k) -> down_go(k) -> wait receivers ready ->
  read its half(k) -> barrier -> wait writer half -> mcast 72 KB -> act mcast -> push`, and the
  writer's half read of k+1 is gated by down_go(k+1), which the reader only grants at the top of
  its next iteration. Both halves of block k+1 start only after the multicast of block k.

Per block that is roughly (DRAM latency + read) + (multicast) + semaphore latency, which with
plausible BH numbers (~1 us latency, ~40 GB/s per-RISC read, ~80 GB/s column multicast) is
~4.3 us per gate/up block and ~3.4 us per down block against ~3.3 us / ~2.15 us of M=1 compute --
an excess of ~28 us/expert, matching the measured ~30 us.

## Mechanism
Opportunistic one-block read-ahead, only when the ring slot of block k+1 is ALREADY free (the
sender core's compute has popped block k-1). No CB is resized, no protocol for receivers changes.

1. gate/up, reader (sender core): while waiting for the writer's `up_done(k)`, poll
   `cb_in1_up.pages_reservable_at_back(2 blocks)` (slot k reserved-but-unpushed + slot k+1 free).
   When true, grant `up_go(k+1)` early (the reader's next-iteration grant sets the same value again,
   so it is idempotent).
2. gate/up, writer: after `up_done(k)`, mcast_go and the receivers' ready, and BEFORE issuing the
   multicast of block k, check `up_go >= seq(k+1)` non-blockingly; if granted, issue the `up(k+1)`
   DRAM reads into the other slot. The next iteration skips the wait/issue and only barriers.
3. down, reader: after the read barrier of block k (before its multicast), if
   `cb_in1_down.pages_reservable_at_back(2 blocks)`, grant `down_go(k+1)` (the writer starts its
   half immediately) and issue the reader's own half of block k+1 into the other slot (address from
   the CB's fifo_limit/fifo_size; the ring is exactly 2 blocks). Next iteration skips grant/issue.
   The receivers'-ready wait moves from before the read to just before the multicast (it only
   gates the multicast, not the sender's own-slot read).

Files: `unified_routed_expert_ffn/device/kernels/dataflow/unified_routed_expert_ffn_reader.cpp`,
`.../unified_routed_expert_ffn_writer.cpp`. No host change (kernel-only, JIT).

## Why this is not a repeat
First round, no prior nodes. It is not the existing UP_SPLIT/DOWN_SPLIT (those split one block's
read across two RISCs, but every block is still read after the previous one is multicast), nor the
IN1_WRITER_MCAST overlap (which overlaps the reader's gate read with the writer's multicast, but not
the writer's own up read). It adds cross-block read-ahead, which nothing in the op does today.

## Expected effect and risk
- Shapes at per_core_M = 1 (all but some GPT-OSS experts): delivery cycle drops toward
  max(read, mcast) per block; expected 1.1-1.3x if the delivery-bound model holds. GPT-OSS M=2
  experts and any compute-bound regime: neutral (the slot is not free early, read-ahead does not
  trigger, behaviour identical to baseline).
- Accuracy: bit-identical results expected (same data, same math); any PCC change means a race.
- Hang risk: the read-ahead is non-blocking and only re-orders reads into slots that the consumer
  has already released, so no new wait cycles. A wrong slot address would show as accuracy_fail.
