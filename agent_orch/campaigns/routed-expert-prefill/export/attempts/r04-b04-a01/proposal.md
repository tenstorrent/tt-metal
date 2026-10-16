# r04-b04-a01: decouple the gate/up x and weight handshakes on the receivers: each stream's next-block ack (and an x owner's next-block multicast) fires as soon as that stream's own block lands, not after both streams' blocks land

## Motivation
- Every gate/up receiver runs one lock-step loop per K-block kb (reader, `unified_routed_expert_ffn_reader.cpp`):
  1. reserve the x, gate and up slots;
  2. ack the x owner (`in0_ready`) and the weight sender (`in1_ready`);
  3. wait for x(kb) valid, push it, then wait for w(kb) valid, push it.
  The ack for block kb+1 of EITHER stream is sent only after BOTH streams' block kb landed.
- The x owner of kb+1 multicasts only from its own iteration kb+1, i.e. also only after its own w(kb)
  landed.
- So the two streams are coupled across the whole group:
  - the column weight sender's multicast of kb+1 waits for every column receiver's x(kb);
  - each row's x multicast of kb+1 waits for every row core's w(kb), plus the owner's own w(kb).
  Each block then starts at roughly the max over ~all cores of the group of both streams' arrival
  times, plus one ack round trip.
- Evidence this coupling is on the path:
  - r03-b03-a02's DM counters: receivers spend 202 µs (x) + 100 µs (weights) of 348 µs GU waiting
    on dsv3. The x wait is twice the weight wait, so the weight multicast is often held by x lateness.
  - r03-b04-a02: per gate/up block, 3-8 µs is spent neither in the sender's DRAM read nor in matmul
    (mistral median block period 10.5 µs vs ~6.8 µs LoFi, 2 µs gate read).
  - r01-b04-a01's analysis: "the limit is most likely the lock-step coupling across the grid. Each K-block
    needs x from the row sender and weights from the column sender, with only 2 slots of slack, so the
    slowest core sets the pace." Nobody has tried removing it.
  - r03-b02-a03 / r03-b03-a04: the compute-bound shapes' FPU idles ~50% with MATH rarely stalled and
    the unpackers idle too. That is consistent with compute waiting on data delivery, not with a
    per-tile engine limit (operand swaps changed nothing).

## Mechanism
Reader kernel only (`kernels/dataflow/unified_routed_expert_ffn_reader.cpp`, gate/up loop). No host,
writer or compute change. Step 3 becomes a polling loop over both streams of block kb:
- Push whichever of x(kb) / w(kb) lands first.
- While the other stream is still in flight, try to start that stream's block kb+1 early (kb+1 in the
  same chunk only):
  - **w lands first, this core is a weight receiver:** if the gate and up slots of kb+1 are already free
    (`pages_reservable_at_back`), reset `in1_valid` and ack the sender for kb+1 now.
  - **x lands first, this core is an x receiver of kb+1:** if the x slot of kb+1 is free, reset
    `in0_valid` and ack owner(kb+1) now.
  - **x lands first, this core owns x(kb+1) (row-major):** if its x slot is free, its compute has
    tilized x(kb+1) (`cb_x_tl` front available) and all GRID_X-1 row acks are in, multicast x(kb+1)
    now, with exactly the owner code's linked data+valid multicast and loopback barrier.
- Iteration kb+1 skips the ack / multicast / reserve that already happened. Anything not done early
  falls back to today's path.

Why the semaphore protocol stays valid unchanged:
- Each receiver still has at most one outstanding ack per stream. It acks kb+1 only after kb's valid
  of that stream, and resets the valid flag before acking, exactly as today.
- So every sender (writer `in1_ready` wait + `set(0)`, x owner, TILE-path x sender) sees only the
  acks for the block it is serving. The next ack can only follow its own multicast.
- A slot is acked only when it is free, so no multicast lands on unread data.
- The TILE x path gets the early ack too (fixed sender, same protocol); the early owner multicast is
  row-major only.

Profiler-only: three counters (early w acks, early x acks, early x multicasts) emitted once at kernel
end as `RE-R-EARLY` TS_DATA. They show whether the mechanism fired.

## Why this is not a repeat
- The read-ahead nodes moved the sender's DRAM reads earlier, against the multicast or ack:
  r01-b03-a01 and r01-b04-a01 (gate/up and down), r03-b04-a04 (down only).
- r03-b03-a02 moved gate/up reads onto one NoC.
- r03-b02-a02 and r03-b04-a02 changed x owner placement and x read timing.
- r02-b02-a02 changed request VCs.
- None of them touched WHEN receivers ack, or the coupling between the x and weight streams.
- r03-b04-a04 next #2 suggests decoupling the x ack. This node does that, and the weight-ack
  decoupling, without changing the sems.

## Expected effect and risk
- **Expected:** shorter gate/up block period wherever the group is paced by handshake chains rather
  than by aggregate DRAM:
  - gpt-oss, minimax-m3, mistral, dsv4-flash GU: -2..-5% chip time;
  - the lone-group tail and the start-up blocks on every shape.
  - The DRAM-saturated GU phases (kimi-k2.7, dsv4-pro, dsv3 phase A) should be flat.
  - Overall about +1-2% geomean if the coupling matters, ~0 if it doesn't.
- **Bit-identical:** same blocks in the same slots, same order of pushes. PCC / mean_abs must match the
  root exactly.
- **Risks:**
  - a hang if an early ack, multicast or skip is mis-sequenced, e.g. an ack sent twice, a skipped
    reserve that was not done, or owner bookkeeping out of step;
  - stale-data accuracy failure if a slot were acked before it is free;
  - a small slowdown from the polling (sem reads) on cores where the other stream lands late anyway.
- The `RE-R-EARLY` counters tell "never fired" apart from "fired but nothing moved".

## Post-eval note (implementation as evaluated)
- The first eval overflowed the kernel config buffer on kimi-k2.7 (70896 > 70656 B). That run had the
  owner early multicast and the `RE-R-EARLY` counters.
- The scored code keeps only the two early acks (x receiver, weight receiver). The owner early multicast,
  which fired on ~1.5% of blocks, and the counters were dropped to fit. See reflection.md.
