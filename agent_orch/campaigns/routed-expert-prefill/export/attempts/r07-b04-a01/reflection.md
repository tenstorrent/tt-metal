# r07-b04-a01 result: 1.7036 (ok). Neutral overall: +0.6% vs the parent r06-b04-a01 (1.6931), inside the ±1% noise. The two plan changes go opposite ways: kimi-k3 -3.4% (5/5/6/6/6 deal), mistral +3.3% and minimax-m3 +2.5% (top group idle). Bit-identical outputs

## What happened vs expected
- All 9 shapes are valid on the first and only run. PCC, max_abs and mean_abs equal the parent and the baseline on
  every printed digit, so the mirrored owners are consistent across reader, writer and compute. No hang.
- Chip µs, parent -> this node:

| shape | parent | this | plan changed? |
|---|---|---|---|
| dsv3 | 494.4 | 498.5 | no |
| kimi-k2.7 | 695.4 | 690.8 | no |
| **kimi-k3** | 1234.0 | **1192.2 (-3.4%)** | yes: 5/5/6/6/6 |
| glm | 442.0 | 431.5 | no |
| **mistral** | 180.2 | **186.1 (+3.3%)** | yes: y2-3 idle |
| dsv4-flash | 294.7 | 290.0 | no |
| dsv4-pro | 985.2 | 994.3 | no |
| **minimax-m3** | 347.8 | **356.4 (+2.5%)** | yes: y2-3 idle |
| gpt-oss | 260.3 | 248.0 | no |

  The 6 unchanged shapes moved within their known spread (gpt-oss is the usual max-core lottery). Their plans
  are bit-identical to the parent (/tmp/r07b04/pt.cpp replay).

## Why (profiler evidence)
Per-row kernel end in µs (/tmp/r06b01a01/ends.py on reports/r06-b04-a01 and r07-b04-a01):

| shape | | y2-3 | y4-5 | y6-7 | y8-9 | y10-11 |
|---|---|---|---|---|---|---|
| kimi-k3 | parent (6/6/6/5/5) | 1214 | 1201 | 1146 | 928 | 894 |
| kimi-k3 | this (5/5/6/6/6) | 1117 | 1115 | **1192** | 1120 | 1075 |
| mistral | parent (y10-11 idle) | **171** | 164 | 145 | 137 | idle |
| mistral | this (y2-3 idle) | idle | **175** | 174 | 144 | 136 |
| minimax-m3 | parent | 326 | **329** | 311 | 296 | idle |
| minimax-m3 | this | idle | **348** | 313 | 305 | 282 |

1. **kimi-k3: the rebalance works as modelled.**
   - The top groups lost one expert each and ended ~95 µs earlier.
   - The bottom groups gained one each and ended ~180-190 µs later (one expert).
   - The max moved to y6-7, the group whose count didn't change. It ended +46 µs later (1146 -> 1192), because the
     lower groups now stream longer alongside it.
   - The spread fell from 320 to 117 µs.
2. **mistral / minimax-m3: the gradient is relative, not positional.**
   - With y2-3 idle, the TOPMOST ACTIVE group (y4-5) became the slow one, and slower than y2-3 was:
     - mistral 175 vs 171, with y6-7 also at 174 (was 145);
     - mm3 348 vs 326.
   - The bottom group (y10-11) ran as fast as y8-9 did in the parent.
   - So whichever groups sit highest among the ones streaming lose the shared NoC/DRAM race. An idle group on top
     buys nothing. Moving the work up also lost the old bottom group's 2-row advantage at y8-9.
   - This matches r06-b02-a01's "first-mover" reading: the bottom-most streams win the synchronized burst
     wherever they are.
3. Net: k3 alone is worth about +0.4% geomean, and the mm3 / mistral loss takes about -0.6% back. The rest is noise.

## Classification
neutral (within noise). It is one win (uneven deal -> extra experts on the lower groups: kimi-k3 -3.4%) mixed with
one flawed sub-idea (idle group on top: mistral / mm3 +2.5..+3.3%, because the gradient follows relative position
among the active groups).

## What a child of this node should try next
1. **Keep the flip only when every group is busy.** In `schedule_in_order` (and `GroupScheduler::assign` for the
   model), use the parent's lowest-index tie-break while the number of active experts is below the number of
   groups. Then idle groups stay at the bottom, as in the parent. Use the highest-index tie-break otherwise.
   - One extra comparison: count the active experts once, which the caller already does in make_plan.
   - Expected: kimi-k3 -3.4% kept, mistral / mm3 back to the parent. About +0.4..+0.6% geomean over r06-b04-a01.
     It stacks with any NoC-side change (gate/up wrap-free, plan-once + tie pass).
2. **kimi-k3 can go further.** y6-7 is now the max at 1192, while y2-5 end at 1115 and y10-11 at 1075. The integer
   split is already the best (5/5/6/6/6). The remaining lever is WHICH experts: give y6-7 the smaller ones (it
   has 3,3,3,3,4,4 = 20 tiles; 4 vs 3 tiles are both pcM 2 on 2 rows, so this only matters for compute-bound
   tails). Probably <1%.
3. **For the 1-expert-per-group shapes** (mistral, mm3, gpt-oss, dsv4-flash, dsv3 / glm phase A), the topmost
   active groups lose regardless of position. The cure has to be dynamic: a run-time share cap or priority, not a
   static placement or offset. r06-b02-a01 and this node rule out the static ones.
4. Tools: /tmp/r07b04/pt.cpp (planner replay; build with `g++ -std=c++17 -I<kernels dir>`), ends.py as above.
