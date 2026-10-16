# r01-b01-a02 result: 1.4716 (ok)

## What happened vs expected
The node is valid on all 9 shapes. PCC, max_abs and mean_abs are bit-identical to the baseline and
the parent, which confirms that the owner-core tilize + bf8 multicast delivers exactly the tiles
each core used to make for itself. Score is 1.4376 -> 1.4716 (+2.4%). Per shape vs the parent
(noise ±1%):

| shape | parent µs | this µs | gain |
|---|---|---|---|
| mistral | 200.1 | 185.9 | +7.1% |
| gpt-oss | 382.3 | 364.5 | +4.9% |
| dsv4-flash | 352.7 | 340.7 | +3.5% |
| glm | 497.5 | 486.9 | +2.2% |
| minimax-m3 | 384.8 | 378.2 | +1.8% |
| dsv3 | 568.0 | 563.1 | +0.9% (noise) |
| kimi-k2.7 | 777.9 | 771.5 | +0.8% (noise) |
| kimi-k3 | 1484.7 | 1478.8 | +0.4% (noise) |
| dsv4-pro | 1142.8 | 1148.9 | -0.5% (noise) |

This is in the predicted direction but below the 4-10% I expected on the compute-bound shapes. The
DRAM-bound shapes (kimi-k2.7, kimi-k3, dsv4-pro, all at ~350-390 GB/s) are flat, as predicted.

## Why (best explanation, with profiler evidence)
I compared per-core math-end times (TRISC1 kernel end, averaged over the 10 measured calls) from
`reports/<node>/.logs/profile_log_device.csv`, as min / median / max in µs:
- gpt-oss: 373/377/381 -> 355/359/363. Its groups are balanced and compute-bound (147 GB/s), so
  the whole distribution shifted ~5%. That is the clean measure of what removing the per-block
  tilize + reconfig is worth: ~5% of a compute-bound pipeline.
- dsv3: 498/537/567 -> 489/526/562. glm: 432/471/496 -> 424/459/486. The median moved ~2%.
- mistral: 152/165/198 -> 153/166/184. The median did not move; the gain is all in the TAIL. The
  slowest group got faster, and the run-to-run total max dropped from 226 to 193 µs.
- The random group-to-group imbalance found in the parent's profile is still the biggest
  inefficiency. With equal work per group, the slowest group finishes 6-14% after the median core
  (dsv3 +7%, mistral +11%, dsv4-flash +6%, minimax-m3 +7%). Which group is slow changes from run to
  run, so this is contention, not a schedule bug. Finishing at the median would be worth ~5-10%
  more on the few-expert shapes.
- Why less than expected: the fastest groups still run at ~28-30 cycles per tile-mult, so the
  tilize was not the main per-block cost. What remains per gate/up K-block: the packer L1_ACC
  drain (STALLWAIT), 2 x M x 16 matmul_block calls with ct=6 (or ct=3 on the N_gu=9 shapes), and
  the lock-step ready/valid handshakes on x (row) and weights (group column).

## Classification
win (small): +2.4% overall, +3.5-7% on mistral / gpt-oss / dsv4-flash, neutral on the DRAM-bound
shapes. The mechanism is cheap in L1 (+128 B per M*w tile) and leaves the TILE path untouched.

## What a child of this node should try next
1. **Group imbalance / tail (largest remaining prize, ~5-10% on few-expert shapes).** Groups doing
   identical work finish up to 15% apart at random. Ideas: (a) find the contention source with a
   few DeviceZoneScopedN zones per expert phase on one group (weight-sender wait vs compute). (b)
   Make the expert -> group assignment dynamic, with a leader core claiming the next expert via a
   NoC atomic; this only helps shapes with >= 3 experts per group (kimi-k3, kimi-k2.7, dsv4-pro).
2. **kimi-k3: add an R=1 group size** (8 one-row groups, no weight multicast, per_core_M = tiles)
   with tile-weighted LPT balancing. r01-b02-a01's G=1 matched R=2 on kimi-k3 even with a 4-vs-3
   expert imbalance, and R=2 wastes 25% of rows on its 3-tile experts. The cost model's +1 per
   chunk overhead is too high for this (it would also wrongly pick R=1 for dsv4-pro, where
   r01-b02-a01 measured G=1 6% slower), so keep R=2 for dsv4-pro.
3. **Expert boundary bubble.** On weight-sender cores the writer drains cb_out for expert e
   before it starts expert e+1's `up` reads / gate+up multicast. Issuing the first 1-2 up blocks of
   e+1 before the drain would hide ~3-5 µs per expert transition (the writer is idle while compute
   finishes the last down blocks).
4. Remaining compute per-block cost: try amortising the L1_ACC packer drain, e.g. by accumulating
   two K-blocks in dst before packing when DST capacity allows at per_core_M <= 2.
- Note for children: the row-shared tilize relies on every core of an M-row pushing the identical
  cb_in0_x block sequence (same slot address on all row cores). Keep x_blk counting in sync in the
  reader and compute if you change chunking or expert assignment.
