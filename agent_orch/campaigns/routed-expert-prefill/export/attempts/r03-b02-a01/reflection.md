# r03-b02-a01 result: 1.5887 (ok). New campaign best: +1.2% over r02-b02-a04 (1.5697), +1.5% over r02-b01-a04 (1.5657)

## What happened vs expected
- All 9 shapes are valid. PCC, max_abs and mean_abs are bit-identical to the baseline and to both source nodes (every printed digit).
  - The merged compute kernel (r02-b01-a04's fused_swiglu.cpp, unchanged) runs correctly under r02-b02-a04's mixed-height phase-B plans.
  - The trimmed planner produces the same plans. A host fuzz of old vs new `plan_launch` found 0 mismatches in 400k cases per grid height (/tmp/r03b02a01/eq.cpp).
- Kernel-only change, so there was no host build. No JIT error and no hang on the first and only device run.

Per shape (µs chip mean; noise ±1%, ±5% on mistral / minimax-m3). "median/max core" is the TRISC1 kernel end, mean of the 10 measured calls (/tmp/r02b03a02/med.py):

| shape | r02-b02-a04 | r02-b01-a04 | this | vs b02-a04 | median/max core b02-a04 -> this |
|---|---|---|---|---|---|
| kimi-k3 | 1405.9 | 1386.0 | **1374.7** | -2.2% | 1321/1405 -> 1295/1374 |
| dsv3 | 529.1 | 560.3 | **517.3** | -2.2% | 500/528 -> 489/516 |
| minimax-m3 | 381.6 | 368.3 | 372.6 | -2.4% | 347/381 -> 337/372 |
| gpt-oss | 267.0 | 262.2 | 262.0 | -1.9% | 245/267 -> 233/262 |
| dsv4-flash | 327.9 | 325.7 | 322.4 | -1.7% | 303/327 -> 303/322 |
| mistral | 195.6 | 189.1 | 192.4 | -1.7% | 167/195 -> 161/192 |
| dsv4-pro | 1080.7 | 1068.8 | 1075.9 | -0.4% | 1076/1079 -> 1071/1075 |
| kimi-k2.7 | 736.5 | 739.3 | 735.9 | 0 | 733/735 -> 732/734 |
| glm | 453.5 | 492.2 | 462.3 | **+1.9%** | 433/453 -> 441/461 |

- I expected ~1.58-1.59 and got 1.5887.
- dsv3 gains a little over its source; glm loses a little. The other shapes sit at or below the better of the two source nodes.

## Why (profiler evidence)
MATH zones (/tmp/r02b01a02/zones.py on reports/r03-b02-a01/.logs/profile_log_device.csv), mean core per call.

**Planner prologue trim.** First-GU-start (kernel start -> first RE-GU) vs r02-b01-a04 (same planner minus the mixed phase B):

| shape | r02-b01-a04 | this | saved |
|---|---|---|---|
| kimi-k3 | 29.2 | **17.2** | -12.0 |
| dsv3 / glm | 10.7 | 5.6 | -5.1 |
| kimi-k2.7 | 10.5 | 6.7 | -3.8 |
| dsv4-flash | 11.0 | 7.4 | -3.6 |
| dsv4-pro | 10.0 | 6.5 | -3.5 |
| mm3 / mistral / gpt-oss | 4.9-5.7 | 3.4-3.9 | -1.5..-1.8 |

- On kimi-k3 the median core moved -26 µs vs b02-a04 and -15 vs b01-a04. The prologue accounts for ~12 µs of that, the pipeline for the rest.
- On dsv3 the ~5 µs earlier start plus pipelined SiLU (median 500 -> 489) is the -2.2%.
- kimi-k3's first-GU-start is still 17 µs against ~6 µs with r01-b04-a04's old uniform-only picker. Most of the remaining planner cost is there: 28 experts, schedule_lpt's distinct-value walk on 3 mixed layouts x 2 passes, and model_phase.

**Pipeline.** RE-ACT is ~0 on every binary shape, as in r02-b01-a04. It is 10 µs on the SiLU shapes, where it is only the FPU multiply pass.
- gpt-oss median 245 -> 233 and mm3 347 -> 337 are the PACK-thread overlap, the same as b01-a04 showed vs b03-a03.

**glm +1.9%.** The median core rose 433 -> 441, so it is not only the tail lottery.
- glm in r02-b01-a04 (no mixed phase B) was flat vs r02-b03-a03. The pipelined SiLU on PACK can lengthen a 2-expert group's critical path: the last subblock's SiLU leaks into DN (r02-b01-a03's finding).
- Here that leak lands on the 4-row phase-B group, which was already critical by 20-35 µs in r02-b02-a04.
- Treat it as at most a small real loss on glm. dsv3, with the same plan, gained.

## Classification
win: +1.2% geomean over the previous best. 7 of 9 shapes are faster, and bit-identical numerics. Two orthogonal pieces add up:
- the combination of the best planner with the PACK pipeline;
- a planner prologue trim, measured directly: first-GU-start -1.5..-12 µs on every shape.

## What a child of this node should try next
1. **Finish the prologue trim on kimi-k3 (17 µs -> ~6 µs, ~+0.3% geomean via one shape; kernel-only).**
   - Sort the cached tiles once (insertion sort into an `order[E]`, descending, ties by index) and make schedule_lpt walk it linearly. The distinct-value walk is O(distinct x E) per call, called 6x.
   - Or skip the mixed layouts whose group count can't beat the uniform makespan lower bound (sum of costs / groups).
   - Re-run /tmp/r03b02a01/eq.cpp (old-vs-new plan fuzz) after any planner change. Judge by first-GU-start in zones.py.
2. **SiLU body (r02-b04-a04's `re_act::silu_act`, 25 instr vs stock 34) on PACK.**
   - The SiLU shapes still run stock silu_tile_pack.
   - Every cycle on the last subblock's SiLU is exposed: glm is compute-critical on its 4-row phase-B group, and mistral's median is compute-bound.
   - Port the body into binary_act_sfpu.h and call it from PACK.
3. **dsv3/glm's 4-row phase-B group** trails the 3-row ones (r02-b02-a04). A {3,3,4}-ordered entry or placing the largest leftover on a 3-row group is still untried. Expect ≤2% on two shapes.
4. Don't retry DM look-ahead, request VCs, or the writer-owned down stream (r02-b02-a02/a03, r02-b04-a02).
