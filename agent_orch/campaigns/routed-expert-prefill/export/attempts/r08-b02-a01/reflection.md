# r08-b02-a01 result: 1.7797 (ok). Win: +2.4% over the root r07-b02-a01 (1.7387), outside the ±1% noise band; new campaign best. Bit-identical outputs

## What happened vs expected
- r07-b01-a01's diff applied onto r07-b02-a01 with git's 3-way merge and no conflict.
- The host build was OK, and the run had no JIT error, no hang and no kernel-config overflow. This was the
  first and only device run.
- All 9 shapes are valid. PCC, max_abs and mean_abs equal the root and the baseline on every printed digit.
- Expected: mistral -6..-10%, mm3 -2..-6%, the other 7 shapes flat, about +1..+1.5% geomean. Got +2.4%:
  mm3 gained much more than expected, and the other 7 shapes stayed flat as expected.

| shape | root chip µs | this | Δ | plan changed? |
|---|---|---|---|---|
| dsv3 | 510.1 | 509.4 | -0.1% | no |
| kimi-k2.7 | 700.7 | 694.7 | -0.9% | no |
| kimi-k3 | 1212.3 | 1209.1 | -0.3% | no |
| glm | 441.0 | 442.0 | +0.2% | no |
| **mistral** | 167.1 | **151.6** | **-9.3%** | 5x2 (y10-11 idle) -> 3+3+2+2 |
| dsv4-flash | 287.0 | 286.4 | -0.2% | no |
| dsv4-pro | 997.3 | 993.4 | -0.4% | no |
| **minimax-m3** | 333.7 | **305.1** | **-8.6%** | 5x2 (y10-11 idle) -> 3+3+2+2 |
| gpt-oss | 229.2 | 227.8 | -0.6% | no |

## Why (profiler evidence: /tmp/r06b01a01/ends.py and /tmp/r06b02/gu.py on reports/r08-b02-a01)
1. **The tie pass is the whole gain, and it works better on this lineage than on r07-b01-a01.**
   - Per-row ends, root -> this node:
     - mistral: y2-3 167 / y4-5 157 / y6-7 140 / y8-9 127 / y10-11 idle -> y2-4 150 / y5-7 140 / y8-9 141 /
       y10-11 127;
     - mm3: 333 / 311 / 280 / 265 / idle -> y2-4 303 / y5-7 280 / y8-9 280 / y10-11 266.
   - On r07-b01-a01 (wrapped gate/up multicasts), the 2-row per_core_M-3 groups at the bottom were mm3's
     tail at 311-315 µs, against 264-285 for the 3-row groups.
   - With wrap-free gate/up multicasts the bottom groups are fast: they win the shared stream (the row
     gradient). So the compute-heavier 2-row groups now end at 266-281, below the top 3-row group.
   - The layout puts the light-compute (per_core_M 2) groups where the bandwidth is short (top) and the
     compute-heavy (per_core_M 3) groups where it is plentiful (bottom). That is why the two mechanisms add
     up: mm3 -8.6% here vs -0.6% on r07-b01-a01.
2. **The top group is still the tail on both shapes, and it is a bandwidth share effect.**
   - The y2-4 group's GU is 109 µs (mistral) / 231 µs (mm3) for per_core_M-2 work. The y10-11 per_core_M-3
     groups' GU is 80 / 172 µs.
   - Delivered traffic is now ~430 GB/s (mistral) / ~460 GB/s (mm3), up from ~390 / ~420.
3. **Plan-once costs a little start-up but is otherwise neutral.**
   - First GU start on mistral / mm3 is 6.0 µs vs 3.6 µs on the root. That is the compute waiting for the
     reader's plan page, plus the 3-row layout.
   - The 7 unchanged shapes moved -0.9..+0.2%, inside their spread. Their per-row ends match the root within
     a few µs (e.g. dsv3 508/451/389 vs 509/453/387).
4. Kernel size: local -O2 profiler compile, reader +1.2 KB, writer -3.6 KB, and UNPACK loses its planner
   copy. So this lineage has several KB of kernel-config headroom again.

## Classification
win: +2.4% geomean, bit-identical, host+kernel. It combines two measured round-7 wins from different branches.
mistral and minimax-m3 gained -9.3% / -8.6% from the idle-row tie pass, which pays off much better now that the
gate/up multicasts no longer wrap.

## What a child of this node should try next
1. **Stack r08-b03-a01's cross-group pacing on this node, scoped to the first expert of each phase**
   (r08-b03-a01 next #1).
   - Pacing won dsv3 -6%, glm -4%, kimi-k2.7 -3%, dsv4-pro -2.3% on the root. It lost on kimi-k3 / dsv4-flash
     through whole-phase GU/DN lock-step.
   - This node does not touch those shapes' plans, so the gains should add up.
   - The headroom freed here pays for the pacing code (+1 KB reader).
   - mistral / mm3 now have all rows busy, which is the case where pacing helps the top group more
     (r08-b03-a01 point 4).
2. **Combine with r08-b01-a01's bottom-first deal if it wins.** Both edit adaptive_chunk.hpp's planner, but
   in different functions. This node changes only plan_groups' tie pass and the store/load helpers.
3. mm3 / mistral top 3-row group (y2-4) is still the tail by ~25 µs over y5-11.
   - A per-layout NoC split tilt for the TOP group only could help: the top group reads more on NoC 1.
     The request parking-lot model (/tmp/r08b02/sim/plot.py) has NoC 1 favouring the topmost active rows
     and NoC 0 the bottom-most. It is untested, and r06-b01-a01's tilt result is ambiguous.
   - Or pacing (1).
4. Don't retry: an idle group on top (r07-b04-a01), static start-up stagger (r06-b02-a01), or pacing every
   block of many-expert phases (r08-b03-a01).
5. Tools:
   - /tmp/r08b02/pt (planner replay; rebuild with `g++ -std=c++17 -I<kernels>` against a worktree);
   - /tmp/r08b02/cc/b.sh (local reader/writer compile with the named `cb_plan_scratch` CT arg);
   - /tmp/r08b02/sim/ (fluid DRAM-share model fitted to per-row ends; it reproduced r07-b04-a01's kimi-k3
     flip);
   - ends.py / gu.py as cited.
