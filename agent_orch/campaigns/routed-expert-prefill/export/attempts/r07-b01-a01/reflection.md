# r07-b01-a01 result: 1.7196 (ok). Win: +1.6% over the parent r06-b04-a01 (1.6931), outside the ±1% noise band; new campaign best. Bit-identical outputs

## What happened vs expected
- r05-b01-a01's 5-file diff applied onto r06-b04-a01 with a single conflict: the planner pragma block in
  adaptive_chunk.hpp (r06-b04-a01's writer -Os branch vs plan-once). It was resolved toward plan-once: the planner
  now lives only in the reader, at its default -O2, and the TRISC -O2 / BRISC -Os pragmas are gone.
- Host build OK. No JIT error, no hang, no kernel-config overflow on the first and only run.
- All 9 shapes are valid. PCC, max_abs and mean_abs equal the parent and the baseline on every printed digit.
- Expected: mm3 -3..-6%, mistral -2..-5%, others flat, about +0.7..+1.2% geomean. Got +1.6%. mistral gained much
  more than expected and mm3 much less.

| shape | parent µs | this µs | Δ |
|---|---|---|---|
| dsv3 | 494.4 | 504.3 | +2.0% |
| kimi-k2.7 | 695.4 | 693.3 | -0.3% |
| kimi-k3 | 1234.0 | 1225.7 | -0.7% |
| glm | 442.0 | 435.4 | -1.5% |
| mistral | 180.2 | **162.6** | **-9.8%** |
| dsv4-flash | 294.7 | 287.9 | -2.3% |
| dsv4-pro | 985.2 | 987.2 | +0.2% |
| minimax-m3 | 347.8 | 345.7 | -0.6% |
| gpt-oss | 260.3 | 259.2 | -0.4% |

## Why (profiler evidence)
Per-row kernel end, mean of the 10 measured calls (/tmp/r06b02/ends.py on reports/{r06-b04-a01,r07-b01-a01}):

| shape | parent (5x2, y10-11 idle) | this node (3+3+2+2) |
|---|---|---|
| mistral | y2-3 172, y4-5 164, y6-7 146, y8-9 137, y10-11 idle | y2-4 145, y5-7 135, **y8-9 157**, y10-11 133 |
| minimax-m3 | y2-3 326, y4-5 329, y6-7 311, y8-9 296, y10-11 idle | y2-4 285, y5-7 264, **y8-9 311, y10-11 315** |
| gpt-oss (already 3+3+2+2) | y2-4 213, y5-7 199, y8-9 247, y10-11 244 | 209 / 198 / 246 / 245 |

1. **The tie pass is the mechanism, and it fires on mistral and mm3 as designed.** Rows y10-11 now work on both.
   The two 3-row groups (6-tile experts at per_core_M 2) end 30-60 µs earlier than the parent's 2-row groups did.
2. **mm3 barely gained because the 2-row groups became the tail.** Its 5-tile experts run on 2 rows at per_core_M 3
   and end at 311-315 µs. The 3-row groups end at 264-285. That is a 30-50 µs imbalance in the opposite direction.
   The planner's cost model (`group_cost` ~ per_core_M) ties the two kinds of group, but a 2-row group streams the
   same expert bytes as a 3-row group and does 1.5x the per-core compute. So in practice the 2-row group is slower.
   gpt-oss's 3+3+2+2 shows the same signature (y8-11 245 vs y2-7 198-210), and it is unchanged from the parent.
3. **mistral gained more than r05-b01-a01 measured (-4.1% there).** Its 2-row groups are less behind (157 / 133 vs
   135-145), and the parent's critical group (y2-3 at 172) is gone. Part of the -9.8% is mistral's usual ±5%
   max-core lottery, but the mean row end also fell (~155 -> ~142).
4. **The 7 unchanged plans moved within their known spreads.** glm -1.5% and dsv4-flash -2.3% are on the gain side.
   dsv3 +2.0% sits on its phase-B 4-row group y2-5 (490 -> 497), which r05-b02-a01 already named as dsv3's critical
   group. Plan-once makes compute wait for the reader's plan page (r05-b01-a01: first-GU-start +0.2..+0.5 µs). That
   is too small to explain +7 µs, so it is most likely the lottery.
5. Kernel binary size was not measured this time. The jit cache is shared with concurrent evals, so I could not
   attribute ELFs to this run. By construction this node removes the writer's (-Os) and UNPACK's (-O2) planner
   copies, ~4-5 KB each per r05-b01-a01 / r03-b01-a02. So the h7168 programs should have several KB of headroom
   again.

## Classification
win (+1.6% geomean, bit-identical, host+kernel). It combines two measured wins from different branches. mistral is
most of the gain, and mm3 shows a new tail: the tie pass's 2-row per_core_M-3 groups.

## What a child of this node should try next
1. **Gate/up wrap-free multicasts (r06-b04-a01 next #1).** That item was blocked by kernel-config space, and this
   node frees it. Top-row senders' gate/up multicast still wraps on the writer's NoC 1. Make the multicasting RISC a
   runtime choice on `my_mt == 0` (the reader's NoC-0 path, compiled out today). Check sizes first with a local
   compile.
2. **Fix the 3+3+2+2 imbalance on mm3 / gpt-oss in the planner.** The 2-row groups end 30-50 µs after the 3-row
   groups on both shapes. Options, in planner order:
   - (a) Swap which experts go to the 3-row groups (give the 3 rows to the experts whose 2-row pcM would be
     largest). mm3 is 6/6/5/5, so this already holds; then (b).
   - (b) Make `group_cost` reflect that a group's time is roughly max(bytes-bound term, pcM-bound term), so 2-row
     pcM-3 is costed above 3-row pcM-2.
   - (c) For 4-expert shapes try 4+2+2+2 vs 3+3+2+2, or put the 2-row groups on the faster bottom rows. The bottom
     groups win the start-up race (r06-b02-a01), and here the 2-row groups are already at the bottom (y8-11), so
     instead try the 2-row groups at the TOP (y2-5).
   Measure with /tmp/r06b02/ends.py.
3. dsv3 / glm: the phase-B 4-row group on y2-5 is still the critical group (dsv3 497 vs 461 / 424). A {3,3,4}
   phase-B order with the 4-row group at the bottom is still untried (r05-b02-a01 next #1).
4. Don't retry: -Os planners on any RISC (moot now), start-up stagger, DM handshake decoupling.
