# r01-b04-a04 result: 1.5011 (ok), a new best (previous best r01-b01-a02 1.4716, parent 1.4622)

## What happened vs expected
All 9 shapes are valid and bit-identical to the baseline (PCC, max_abs and mean_abs equal to every
printed digit). The 10-row topology (groups of 10/5/2 rows, 109-destination counts multicast) is
correct on the first device run, with no hang. Per shape vs the parent r01-b04-a03 (noise ±1%):

| shape | parent µs | this µs | Δ | predicted |
|---|---|---|---|---|
| gpt-oss | 368.5 | 338.8 | **-8.1%** | -15-20% |
| kimi-k3 | 1477.7 | 1378.8 | **-6.7%** | up to -14% |
| dsv3 | 561.9 | 547.5 | **-2.6%** | neutral |
| dsv4-pro | 1153.2 | 1128.0 | **-2.2%** | neutral |
| dsv4-flash | 337.6 | 331.2 | **-1.9%** | neutral |
| minimax-m3 | 385.6 | 382.0 | -0.9% (noise) | neutral |
| kimi-k2.7 | 767.6 | 764.3 | -0.4% (noise) | -2-5% |
| glm | 488.2 | 487.6 | 0 | neutral |
| mistral | 193.2 | 193.0 | 0 | neutral |

The direction is as predicted, the gains on the two target shapes are about half of the per-core
unit model, and three "neutral" shapes gained 2-3%.

## Why (best explanation, with profiler evidence)
TRISC1 kernel-end per physical row (y2..y11 = logical rows 0..9), mean of the 10 measured calls,
µs from the earliest core start (`/tmp/r01b04a04/rows.py` on reports/r01-b04-a04/.logs):

- gpt-oss: all ten rows end at 324-329. Both 5-row groups (2 experts each, per_core_M 2) are
  balanced. Units fell 5 -> 4 (-20%) but time fell only 8%, so per-expert fixed costs (bias add,
  SwiGLU-OAI, activated multicast, down phase over 11 blocks) are a large part of gpt-oss. They
  don't scale with per_core_M.
- kimi-k3: the 6-expert groups end at 1336-1346 and the 5-expert groups at 1136-1177. The makespan
  is set by 6 experts (12 units) instead of 7 (14), so -14% was the ceiling. -6.7% shows that
  shared DRAM (5 concurrent streams now) absorbs half of it.
- dsv3 / glm: groups 0-2 run 2 experts (dsv3 end 508-535, close to the parent's median of 516), and
  groups 3-4 run 1 expert and end at ~250-313. The gain on dsv3 comes from the two light groups
  finishing early: the heavy groups' second expert then has DRAM to itself. glm is flat.
- kimi-k2.7: groups 0-1 end at ~741-753 and groups 2-4 at 541-583. The list scheduler's makespan
  is the same as on 8 rows for the heavy groups, so there is no gain. dsv4-pro is similar
  (1087-1122 vs 818-885).
- mistral / minimax-m3: 4 experts on 5 groups, so rows y10/y11 sit idle (end at 2 µs) and the
  result is neutral, as predicted.

## Classification
win (+2.7% score over the best node, outside noise on 5 shapes, bit-identical numerics, no
specialisation: devices with < 10 rows fall back to the old 8-row grid).

## What a child of this node should try next
1. **Group-tail imbalance is now the largest, measured waste.** With 5 groups the expert counts
   rarely divide: kimi-k2.7 (12 experts, 3/3/2/2/2) has groups 2-4 idle for the last ~25% of the
   call, dsv4-pro (12) ~22%, kimi-k3 (28, 6/6/6/5/5) ~14%, dsv3/glm (8, 2/2/2/1/1) ~45% of two
   groups. Options:
   (a) Split the last-wave experts' work across the idle groups, e.g. give an expert's N range
       (gate/up N and down N) in halves to two groups. The down phase needs all of the
       activated K, so the split must be by output N on gate/up (= down K), with a cross-group
       reduction. Or split by M (tile-rows): a 5-6-tile expert on 2 groups is pcM 2+1 instead
       of 3, at the price of reading its weights twice.
   (b) Cheaper: let the cost model consider mixed sizes, or choose the group count that divides
       the work best (e.g. kimi-k2.7 with 4 groups of 2 rows + 1 group of 2 rows idle is no
       better). An M-split of only the leftover experts is probably the simplest real fix.
2. **gpt-oss per-expert fixed cost.** Rows are balanced but the time fell only 8% for 20% fewer
   units. Profile the bias/activation/down phases (DeviceZoneScopedN) on one core. Suspects: the
   single-buffered bias CBs serialising expert boundaries, the 1x3 gate/up subblocks
   (per_core_N_gu 9), and the phantom 11th down K-block.
3. mistral/minimax leave 2 rows idle (4 experts on 5 groups). Splitting one expert's tile-rows
   across the two idle rows (as in 1a) would help them too.
4. Note: the exported UNIFIED_ROUTED_EXPERT_CORE_GRID is still 11x8 (kCoreGridY). It is only used
   for moe_fused_swiglu's grid, so the hybrid crossover calibration in the model may want
   re-measuring against this op's 10-row timing. That is outside this op's allowed paths.
