# r01-b02-a03 result: 1.4665 (ok)

## What happened vs expected
All 9 shapes are valid. PCC, max_abs and mean_abs are bit-identical to the parent and the baseline,
so the ported owner tilize + bf8 multicast and the x_blk counting agree between the reader and
compute under b02's GroupDealer. The L1 fit also did not move (M, w) on any shape. The score went
from 1.4460 (parent) to 1.4665 (+1.4%). That is 0.35% below r01-b01-a02 (1.4716), which is inside
the ±1% noise, so this is a tie with the best node, not a new best. I expected ~1.48.

Per shape vs the parent (noise ±1%):

| shape | parent µs | this µs | gain | b01-a02 µs |
|---|---|---|---|---|
| mistral | 195.8 | 187.4 | +4.5% | 185.9 |
| gpt-oss | 382.9 | 366.4 | +4.5% | 364.5 |
| dsv3 | 566.5 | 551.8 | +2.7% | 563.1 |
| glm | 504.4 | 492.7 | +2.4% | 486.9 |
| minimax-m3 | 381.1 | 376.5 | +1.2% | 378.2 |
| dsv4-pro | 1148.1 | 1147.6 | 0 | 1148.9 |
| kimi-k3 | 1472.4 | 1480.1 | -0.5% (noise) | 1478.8 |
| dsv4-flash | 343.0 | 345.8 | -0.8% (noise) | 340.7 |
| kimi-k2.7 | 776.4 | 784.2 | -1.0% (edge of noise) | 771.5 |

## Why (best explanation)
- The compute-bound shapes (mistral, gpt-oss, glm, dsv3) gain about as much as in b01 (+2-5%), which
  confirms that removing the 11x redundant tilize and its per-block reconfigs is worth ~5% of a
  compute-bound group pipeline on either schedule.
- The DRAM-bound shapes (dsv4-pro, kimi-k3, kimi-k2.7) are flat, as in b01. The owner's x read now
  rotates across all 11 columns, so on ~1/11 of K-blocks per group it lands on a weight-sender core.
  The x read is issued two blocks ahead and its multicast comes after the weight work, so the
  b02-a01-style collision does not show up as a loss. It may cost the ~1% seen on kimi-k2.7.
- The parent's ~3% edge over b01 on mistral and dsv4-flash did not survive: dsv4-flash is now 5 us
  behind b01-a02. Two schedules that differ only in the group rule and the expert deal now finish
  within noise of each other, which points at the remaining time being in the per-group pipeline
  itself (per-tile matmul rate, group-to-group contention), not in the schedule. I did not take
  per-core profiles for this node. The full device CSV is in reports/r01-b02-a03 for a child that
  wants the group end-time spread.

## Classification
win over the parent (+1.4%, mostly outside noise on 4 shapes). Neutral against the best node
r01-b01-a02: both branches have now converged to the same ~1.47 point.

## What a child of this node should try next
1. Stop porting between the b01 and b02 branches: the schedule details are now worth < 1%.
2. Group-to-group imbalance (b01-a02 measured the slowest group 6-14% behind the median with equal
   work): add DeviceZoneScopedN zones per phase (gate/up K-loop, activation, down) on one row group
   to find whether the tail is weight-sender wait or compute.
3. Few-expert shapes (mistral, minimax-m3, gpt-oss: 1-2 experts per group, no cross-expert overlap):
   split one expert's N across two groups, or overlap expert e+1's first gate/up blocks with expert
   e's down phase on the weight senders (writer idle while compute drains down).
4. Don't widen in0_block_w_gu (accuracy knob, r01-b04-a02). Don't retry read-ahead before the
   multicast (r01-b03-a01, r01-b04-a01). The owner x read could be kept off weight-sender columns
   (skip columns whose row holds the column's in1 sender) to check whether that recovers the
   kimi-k2.7/kimi-k3 noise-level losses.
