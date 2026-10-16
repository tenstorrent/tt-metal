# r04-b02-a01: weight senders start block kb+1's DRAM reads as soon as their own ring slot is free, in both phases

## Motivation
Every weight K-block a sender reads today is gated by more than its own ring slot:

- **Gate/up (new here).** On a column's weight sender the reader issues gate(kb+1) and releases the
  writer's `up(kb+1)` read (UP_SPLIT `up_go`) only at the top of iteration kb+1. It gets there only
  after the row's x block kb has reached this core (`in0_valid` wait at the end of iteration kb).
  So the whole column's weight stream for kb+1 waits on the sender ROW's x stream for kb. That x
  multicast in turn waits on all 11 row cores' acks, which come only after each received its own
  column's w(kb-1). This is a grid-wide lock-step: one late x owner or column reaches every column
  within ~2 blocks.
  - r03-b03-a02's DM counters (different reader split, same coupling) measured the sender waiting
    for x for 3-4 µs per block on dsv3 (106 µs per call), 24-47 µs on mistral / mm3 / gpt-oss.
  - r03-b04-a02 found 3-8 µs per gate/up block that is neither the sender's DRAM read nor matmul.
- **Down (port + repair).** r03-b04-a04 (sibling lineage, 1.6078) issued down block kb+1's reads right
  after multicasting kb when the slot was already free.
  - Measured on the many-expert shapes: RE-DN -9..-31 µs per core, mean core end -7..-18 µs.
  - It did nothing on the 4-expert shapes. Its own next-step #1 guessed the one-shot
    `pages_reservable` check usually failed there: the sender's compute was still on kb-1.
  - The root (r03-b02-a03) does not have it. Its dataflow files are byte-identical to that node's
    parent, so the patch applies cleanly.

## Mechanism
One principle in both phases: a weight sender starts block kb+1 once (a) its own compute has freed
the slot and (b) nothing still reads that slot. It no longer waits for the row's x / activated
arrival or the receivers' acks; only the multicast needs those.

- **Gate/up** (reader only; IN1_WRITER_MCAST + UP_SPLIT, the configuration every shape runs):
  - Right after the sender pushes block kb, `try_gu_lookahead(kb+1)` checks three things. The gate
    and up rings have a free block (`pages_reservable_at_back`). The writer has finished
    multicasting the slot's previous block (`mcast_done >= mc_seq-1`, non-blocking). And kb+1 is in
    range.
  - If all hold, it bumps `up_seq` / `up_go` (the writer reads `up(kb+1)` right after it finishes
    multicasting kb), bumps `mc_seq`, and issues the NoC-0 gate reads into the next slot
    (`get_write_ptr()` after the push).
  - The sender's `in0_valid` wait becomes a spin that re-tries the look-ahead, because the compute
    frees the slot meanwhile.
  - Iteration kb+1 then skips the `up_go`, the `mcast_done` wait and the read. Its existing read
    barrier covers the early reads.
- **Down** (reader):
  - r03-b04-a04's look-ahead, verbatim.
  - Plus the same spin re-check while the in1_down sender waits for `act_valid` (the repair that
    node proposed for the 4-expert shapes).
- **Unchanged:**
  - slots, per-RISC read/multicast order, bytes and the NoC split: gate on NoC 0, up on NoC 1, down
    half and half;
  - every other mode (LEGACY up, no writer multicast) keeps today's code via `if constexpr`.

So the results stay bit-identical. File: `kernels/dataflow/unified_routed_expert_ffn_reader.cpp` only.

## Why this is not a repeat
- **r01-b03-a01 / r01-b04-a01 (read-ahead, -6..-7%)** issued the next block's reads BEFORE
  multicasting the current block, on the root's single-RISC sender. That delayed the critical
  multicast. Here the multicast is on the writer (NoC 1), and the reader's early NoC-0 gate read
  only starts after block kb's multicast has been handed off. The writer's early `up` read starts
  after its own multicast of kb has finished, so no multicast is delayed. r01-b03-a01's own advice
  was "issue the next block's reads AFTER the multicast has been issued".
- **r02-b02-a03 (down look-ahead across the activation gap, deeper ring)**: no ring change here, and
  down block 0 keeps its order.
- **r03-b04-a02 (x reads on their own trid)** moved the x DRAM reads. This node moves the WEIGHT
  stream off the x arrival.
- **r03-b03-a02 (all gate/up reads on NoC 0, -5%)**: the NoC split is unchanged here.
- **r03-b04-a04** is ported for its down half and extended with the spin re-check. The gate/up half
  is new.

## Expected effect and risk
- **Expected:** mean core end -1..-3% on the many-expert, DRAM-bound shapes (dsv3, kimi-k2.7,
  kimi-k3, glm, dsv4-pro), from the down port. Plus some tail reduction wherever the gate/up
  lock-step paced a column. Geomean +0.5..+1.5%.
  - The gate/up look-ahead can only help where the sender row's x arrives after its compute frees
    the slot, i.e. x-limited rows.
  - On purely DRAM-share-bound shapes it only moves bandwidth earlier. That is neutral, or slightly
    widens the race as in r02-b02-a03, which I'd see as a larger max-mean.
- **Risks:**
  - Hang if the slot / sequence bookkeeping is wrong (`up_seq` / `mc_seq` vs the writer's
    counters). Wrong data if a slot is overwritten while it is still being multicast.
    `mcast_done >= mc_seq-1` mirrors the existing `wait_min(mc_seq-2)` for the next sequence
    number.
  - Kernel config buffer: the reader grows ~1.3 KB (local -Os profiler compile, 13.0 -> 14.3 KB).
    The h7168 programs were near the 70656 B limit. A `Program size too large` runtime_error on
    dsv3 / kimi-k2.7 / dsv4-pro would mean trimming code before any timing.
- **How to read the eval:** RE-GU / RE-DN MATH zones (zones.py) split the effect between the two
  phases. Compare against the root's report (r03-b02-a03).
