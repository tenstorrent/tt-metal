## Files read
- agent_orch/WORKER.md, campaigns/routed-expert-prefill/campaign.yaml — job definition, allowed paths, gate.
- $DREAM_HOME/routed-expert-prefill/history.md — index of all 69 nodes.
- unified_routed_expert_ffn/device/kernels/adaptive_chunk.hpp (whole file) — the launch planner: uniform
  list scheduling (`schedule_in_order`), mixed LPT, two-phase `make_plan` (phase-B candidate loop, model),
  `plan_launch` (phase-B rank -> group mapping). The only file changed.
- kernels/dataflow/unified_routed_expert_ffn_reader.cpp (phase loop, lines ~380-600) — confirmed every RISC
  consumes `expert_group[phase][e] == my_group` generically, so any owner map (idle groups, bottom-first) is
  handled; phase A has its own weight sem pair; per-phase topology by layout index.
- grep of writer / fused_swiglu for plan_launch use — same plan on every RISC.

## Nodes consulted
- r07-b02-a01 (root) reflection: deterministic row gradient; next #1 "rebalance the plan toward the bottom
  groups"; profile used for per-row ends and per-job timelines.
- r07-b04-a01: unconditional bottom tie-break: k3 -3.4%, mistral/mm3 +2.5..3.3% (idle top group makes the
  next group slow); next #1 = flip only when every group busy (part 1 of this node).
- r07-b01-a01, r05-b01-a01: plan-once / tie pass (not ported; disjoint shapes); size notes.
- r07-b03-a01: kernel-config overflow by 128 B on k2.7 from +180 B local writer growth (size caution).
- r06-b01-a01, r06-b02-a01: gradient is relative / first-mover; static stagger and NoC tilt don't fix it.
- r05-b02-a01, r06-b04-a01: row speeds unequal after the NoC split; {3,3,4} proposals never tried.
- r02-b02-a04, r02-b03-a01/a02: why two-phase and one-round phase B exist (lone big group ~250 GB/s cap).
- r01-b01-a03: pacing failure analysis (b04 in this round is doing pacing; this node stays planner-side).
- r03-b04-a02, r04-b03-a01: x reads not late; compute waits on row streams.
- All other reflections skimmed via history.md "next" columns.

## Tools / data
- /tmp/r08b01/jobs.py — per-row job intervals (GU start | GU end | DN end) from profile_log_device.csv.
- /tmp/r08b01/pt.cpp (copy of /tmp/r07b04/pt.cpp) — native planner replay on the test counts; base.txt vs
  new.txt show the plan changes (dsv3, glm, k2.7, dsv4-pro, k3 only).
- /tmp/r07b02/sz/b.sh (reader/writer local -O2 size), /tmp/r08b01/tr/kc.sh (TRISC0 size, base header via
  -I/tmp/r08b01/basek).
- /tmp/r06b01a01/ends.py, zones.py — per-row ends and zone timelines.

## Docs / external references
- none
