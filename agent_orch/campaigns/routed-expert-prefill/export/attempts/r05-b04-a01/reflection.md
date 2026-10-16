# r05-b04-a01 result: 0 (hang)

## What happened vs expected
- Host build OK (19 s). The kernels JIT-compiled.
- The first shape (deepseek-v3) hung, and the eval timed out at 900 s and reset the device.
- No profile was dumped, so I have no timing and no per-core state.
- Expected: bit-identical outputs and +1..+3% (fewer wrapped, path-reserved column multicasts).
- The mechanism was never measured.

## The finding this node was built on (still valid, worth reusing)
On r04-b03-a01's per-core UNPACK wait tallies (`analysis/wrap.py`), receivers whose column weight multicast has to
wrap around the NoC torus wait much longer for their weights. "Wrap" means the sender is not at the rectangle's NoC
start corner:
- gate/up: the writer's NoC-1 -y multicast from a non-bottom row;
- down: the reader's NoC-0 +y multicast from a non-top row.

The effect, on every shape with a known layout:

| shape | gate/up weight wait, wrap / no wrap (µs) | down weight wait, wrap / no wrap (µs) |
|---|---|---|
| kimi-k3 | 156.5 / 108.7 | 130.7 / 46.1 |
| dsv4-flash | 44.2 / 21.6 | 18.3 / 9.0 |
| minimax-m3 | 31.1 / 23.7 | 15.6 / 4.1 |
| mistral | 14.9 / 9.3 | 6.7 / 0.6 |
| gpt-oss | 10.4 / 2.4 | 4.1 / 1.1 |

Also measured on r04-b02-a01 (`analysis/strag.py`, `grp2.py`, `slowuw.py`):
- The op's biggest remaining waste is the per-call tail. Max minus median core end is 20-90 µs on 7 of 9 shapes,
  and it varies a lot from call to call (glm: 5 5 5 86 85 ...).
- It comes from whole groups whose row stream (x in gate/up, activated in down) runs 1.5-2x slow for a whole
  phase, plus single straggler cores that wait on weights.

## Why it hung (static analysis; not proven)
This node changes which core plays each role:
- gate/up sender = the group's bottom row in every column;
- down sender = the top row;
- x owner = column j % 11 in every row.

I re-traced the reader / writer / compute handshakes and found no ack or semaphore cycle:
- gate/up acks now go only to the bottom core and down acks only to the top core;
- the writer-lag drain is kept on the gate/up sender;
- the up_seq / mc_seq / down_seq predicates match between reader and writer.

So the bug is in something this run could not show. The combinations that are NEW in this node, i.e. never
exercised by any earlier valid node, are the candidates:
1. **x owner == gate/up sender, together with r04-b02-a01's gate/up look-ahead.**
   - Before r03-b02-a02, owners could be senders, but the look-ahead did not exist yet.
   - Since r04-b02-a01, owners were never senders.
   - Now 100% of the bottom row's owner blocks sit on sender cores, which run `try_gu_lookahead` and the
     owner's `cb_x_tl` / `in0_ready` / INCL_SRC multicast / `async_write_barrier` in one iteration.
   - Most likely suspect. Shared NoC-0 read barrier: the look-ahead's gate reads, the owner's x_rm reads and the
     single-block cb_x_rm/cb_x_tl rings all meet there.
2. **The gate/up sender and the down sender of a column are different cores.** Every earlier node used one core
   for both. Check:
   - the top core's in1 semaphores, now DN-only as sender and GU-only as receiver;
   - the writer's per-chunk order: UP loop, then DOWN loop, then cb_out drain. The bottom writer now goes UP ->
     drain and the top writer DOWN -> drain.
3. **Every top-row core is a down sender and also, for block kb == gx, the activated sender.**
   - Each one blocks on its column's in1 acks (step 2) before the row multicast (step 4).
   - Before, only about half the top row did.
   - All 11 top cores' linked NoC-0 column multicasts then run alongside the linked INCL_SRC row multicasts.
   - A NoC-level linked-path interaction can't be ruled out.

## Classification
repairable failure (bug: hang from one of the new role combinations above, most likely (1)). The NoC finding
behind it is measured and stands. The mechanism's value is unmeasured.

## What a child of this node should try next
1. **Split the change to find the hang.** Do it in this order, and treat each step as the repair candidate.
   - (a) Keep the x owner OFF gate/up senders: restore r03-b02-a02's skip. The skip predicate is now "this row is
     its group's bottom row". The bottom row has no eligible column, so either:
     - give that row's blocks to the row ABOVE's owners: impossible, x is per row; or
     - use **one fixed non-sender core** for the bottom row: there is none, every bottom core is a sender.

     So instead use the simpler variant (b).
   - (b) **Only fix the down multicast:** down sender = top row in every column. Keep the gate/up sender at
     `group_base + gx % rows` and keep the x owners off it, exactly as today.
     - This removes the gate/up-sender-is-owner combination (1) entirely.
     - It keeps the larger half of the measured effect: down-wait wrap ratio 2-11x vs gate/up 1.4-4x.
     - It touches only the reader's phase-4 role, the writer's DOWN_SPLIT predicate (`my_mt == 0`), and the down
       ack target (rect start).
     - Hazard to keep: the column's gate/up sender (writer multicast) and its down sender are now different
       cores for half the columns. That is candidate (2). If it hangs too, (2) is the bug.
   - (c) For gate/up, the no-wrap option that keeps owners off senders is to switch the multicasting RISC, not
     the sender row:
     - top-row gate/up senders multicast from the reader on NoC 0 (+y, the existing `!writer_mcasts_in1` path);
     - bottom-row senders keep the writer's NoC-1 multicast.

     That is a per-core runtime choice between two existing paths.
2. **Before any device run, check the new code for deadlock with a tiny host-side simulation.** Neither the
   static review nor the local compile catches a hang. Model:
   - the reader/writer/compute per-block waits and pushes;
   - the semaphores;
   - the ring depths (2 slots; x_rm/x_tl 1 block);

   for one 2-row group on the dsv3 plan.
3. Keep `analysis/wrap.py`: after a valid run, the wrap/no-wrap gap in weight waits should close. Judge with
   `strag.py` (per-call max - median) and `rows.py`.
4. Note: the comment "the only core that's both senders" and other gy=0 comments in the reader/writer headers
   are stale on this lineage. That is harmless.
