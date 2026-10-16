## Files read
- agent_orch/WORKER.md, campaign.yaml — rules, allowed paths, accuracy gate.
- kernels/adaptive_chunk.hpp — the whole launch planner: layout table (kLayouts10: 10 / 5+5 / 2x5 / 3+3+2+2 / 4+3+3 /
  4+2+2+2), plan_groups (mixed layouts only when STRICTLY better, cap = best - 1), make_plan (two-phase), plan_launch.
  The change is confined to plan_groups.
- tests/.../test_routed_expert_prefill.py — routing (uniform top-k, seed 0); mistral and minimax-m3 both get 4 experts
  of 5/6/6/5 tile-rows.
- reader / writer / fused_swiglu.cpp call sites of plan_launch (every RISC plans identically; no host-side plan).

## Nodes consulted
- All 56 nodes' reflections ("what a child should try next" sections, /tmp/r04b03a03/nexts.md), plus the full
  reflections of r04-b03-a01 / a02 (parent chain), r04-b02-a01 (best), r04-b02-a02, r04-b04-a02, r04-b01-a02
  (three activated-stream decoupling attempts on the best reader: neutral / negative; I did not take the parent's
  suggested act-only decoupling for that reason), r02-b02-a01 / r02-b04-a01 (mixed layouts; gpt-oss 3-row vs 2-row
  group end times, U ≈ 29 µs per tile-row), r02-b02-a04 (idle rows on mistral / mm3: "only a sub-expert split"),
  r03-b04-a01 (T = F + U·pcM fit on mistral, U ≈ 25 µs).
- r03-b03-a03 reflection: mistral uses only 8 of 10 rows.

## Measurements used (reports/r04-b03-a02, the parent)
- /tmp/r04b01a02/rows.py: per-row end / GU / DN. mistral and mm3 rows y10-11 end at 4-5 µs (idle).
- /tmp/w_r04b03a03c/grp.py: per-call group end times; mean per-call max over 4 groups vs over the two 5-tile groups
  (mistral 183 vs 173 µs, mm3 368 vs 356 µs).
- /tmp/w_r04b03a03c/outl.py, uwout.py: the per-call outliers are single-group stalls of +45 µs, seen as x (GU) or
  activated (DN) waits on the UNPACK thread.

## Tools written
- /tmp/w_r04b03a03c/sim/pt.cpp (copied from /tmp/routed_r02b02a04): plan replay on the test routing, old vs new header.
- /tmp/w_r04b03a03c/sim/fz.cpp + fzmain.cpp: 400k random routings per grid (10 and 8 rows), old vs new plan_launch:
  the plan changes only when the old one was a single-phase uniform layout with idle groups, the new one is a mixed
  layout with one expert per group, and every active expert is dealt exactly once (0 bad).
- /tmp/w_r04b03a03c/sim/sz.cpp: riscv-tt-elf-g++ size of plan_launch<12> at -O2: 4832 -> 5428 B (+0.6 KB per planning
  RISC: reader, writer, UNPACK).

## After evaluation
- reports/r04-b03-a03 run 1: TT_FATAL program size 70736..71648 > 70656 (run.log); eval of run 1 saved in /tmp/w_r04b03a03c/run1.
- reports/r04-b03-a03 run 2: ends.py (first GU start +3 µs everywhere), rows.py / grp2.py (mm3 groups balanced at ~330 µs).
