# r01-b02-a02 result: 1.4460 (ok)

## What happened vs expected
All 9 shapes are valid and bit-identical to the baseline (PCC, max_abs and mean_abs match to every
printed digit). The score went from 1.4017 (parent) to 1.4460, the best so far (b01 is 1.4376).
- gpt-oss: 477 → 383 µs (1.25x over the parent). Predicted ~385.
- dsv4-pro: 1211 → 1148 µs (1.055x). Predicted ~1140.
- kimi-k3: 1487 → 1472 (-1%, at the edge of noise). It now runs 2-row groups instead of 1-row.
- Everything else is within ±1% of the parent: dsv3 -0.9%, kimi-k2.7 -0.6%, glm +0.9%,
  mistral +0.9%, dsv4-flash +0.7%, minimax -0.7%. Mistral (196 µs) and dsv4-flash (343 µs) keep
  the parent's edge over b01 (200 / 353). That edge came from the parent's group-size rule or
  round-robin deal, not from x-sender rotation. Rotation was neutral on the 2-row shapes and
  harmful on gpt-oss.

## Why (best explanation, with profiler evidence)
gpt-oss per-core kernel end times (last measured call, profile_log_device.csv): the two 4-row
groups now end at ~379 and ~375 µs. In the parent they ended at ~425 and ~460 µs, so the same
2+2-expert work runs ~17% faster per group. The parent's rotation put the row's x read (up to 96
row-major sticks per K-block) on a weight-sender core in ~1/4 of the K-blocks. That core's NCRISC
then did the x read, the x multicast and its gate read back to back, which stalled both the row
and the column. With the fixed sender at column gy+1, the x reads never share a RISC with weight
reads. The unread biases of other groups' experts (~9.5 MB per call) also no longer sit on the
reader's critical path at expert boundaries. I didn't measure the two gpt-oss fixes separately.
dsv4-pro: the 2-row groups with 3 experts each (per_core_M 2) replace 1-row groups. In those, half
the rows ran 2 experts and half ran 1, and every core read its own weights and collided with x. The
result now matches b01's 2-row schedule (1143).
Weight bandwidth is unchanged in kind. The large shapes are at ~330-390 GB/s and gpt-oss at
141 GB/s, so gpt-oss is still latency/compute-bound (2 serial experts per group, per_core_M 3,
1x3 subblocks).

## Classification
win (repairable failure in the parent, bug: x-sender rotation collides with group weight senders
+ bias reads for other groups' experts). Confirms the b01/b02 gap was placement, not schedule.

## What a child of this node should try next
1. The 4-expert shapes (mistral, minimax-m3, gpt-oss) have no cross-expert overlap inside a group
   (1-2 experts per group). Try splitting one expert's N (or the down K) across two groups when
   experts < groups. For example, mistral/minimax could run 4-row groups (2 groups x 2 experts)
   versus 2-row groups with N halves, to shorten each expert's serial pipeline.
2. gpt-oss specifically: per_core_N 9 forces 1x3 subblocks (DST 8, 9 % 8 != 0). With per_core_M 3,
   try a ragged 1x8 + 1x1 tiling (like the down path's ragged tail) to cut in0 re-unpacks. Its 141 GB/s says compute/latency is the limit there.
3. The large shapes sit at ~340-390 GB/s. The next lever is per-K-block compute overhead on the
   row-major path: every core in a row re-tilizes the same x block (11x redundant), so tilizing
   once on the x sender and multicasting bf8 tiles would also halve the x multicast bytes. Rotation
   is NOT the way to spread that load (this node), so keep x reads off the weight-sender cores.
4. Don't reintroduce 1-row groups unless the x read moves off the weight readers (e.g. the column's
   x arrives by multicast from a dedicated non-weight core).
