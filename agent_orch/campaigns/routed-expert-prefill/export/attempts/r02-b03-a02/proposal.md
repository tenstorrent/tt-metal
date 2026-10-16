# r02-b03-a02: combine r02-b04-a01's mixed-height layouts with this branch's two-phase schedule, restricted to leftovers that fit in one phase-B round

## Motivation
Two round-2 schedules each win on different shapes, and they don't overlap:
- r02-b04-a01 (best, 1.5314): mixed-height single layouts (3,3,2,2) give gpt-oss -18% and dsv4-flash -2.7%
  against r01-b04-a04. dsv4-pro also switched to (3,3,2,2) there, but that was neutral (1131.8 µs).
- r02-b03-a01 (parent, 1.4973): the two-phase plan (full waves on 2-row groups, then the leftover on 5-row
  groups) removed the tail on kimi-k2.7 (764 -> 745 µs, -2.6%) and dsv4-pro (1128 -> 1084 µs, -3.9%). It lost on
  dsv3 (+3.2%) and glm (+1%), where 3 leftovers on two 5-row groups need TWO phase-B rounds. The parent's
  reflection (item 1) measured why: a lone 5-row group streams at only ~250 GB/s (sender-capped), so a second
  phase-B round is far slower than the model's DRAM-share estimate.

## Mechanism
`kernels/adaptive_chunk.hpp`:
- Take r02-b04-a01's layout table (kLayouts10/kLayouts8), `layout_group`, `schedule_in_order`, `schedule_lpt`
  and `plan_groups` unchanged.
- Keep the parent's two-phase `make_plan` / `PhaseDealer` / model, with one new rule: a phase-B size is only a
  candidate when the leftover experts fit in one round (`active - n_a <= kGridY / rows_b`).
- New `plan_launch`: if `make_plan` returns a two-phase plan, use it (phase A and phase B are the uniform layouts
  of rows_a / rows_b, owners from PhaseDealer). Otherwise phase A is empty and phase B is `plan_groups`' layout
  (uniform or mixed) with its owner map. The result is a per-phase layout index plus an `owner[2][E]` map.
- Reader, writer, compute: the parent's phase loop, but the group, its base row and height come from
  `layout_group(layout[phase], my_gy)`, the topology from set `layout[phase]` (b04's per-layout sets), and
  ownership from `owner[phase][e]`. The phase-A semaphore pair moves to after the `num_layouts` topology sets.
- Program factory: b04's per-layout topology emission plus the parent's phase-A semaphore pair, merged.

A host replay of the test's seed-0 routing through the new header (/tmp/r02b03a02/pt) gives:
kimi-k2.7 and dsv4-pro -> the parent's two-phase plan, exactly as before. gpt-oss and dsv4-flash -> (3,3,2,2),
exactly as b04. dsv3, glm, kimi-k3, mistral and minimax-m3 -> the uniform 2x5 deal of r01-b04-a04. Zero counts,
a single expert, and big multi-chunk experts all fall to single-phase plans.

## Why this is not a repeat
- It combines two measured wins that touch disjoint shapes. Nobody has run both schedules together.
- The single-round restriction is the repair the parent's reflection proposed (item 1). The bug it fixes is that
  `make_plan` accepted two-round phase-B plans because `model_expert` doesn't cap per-group stream rate.
- Phase-A and phase-B mechanics (ring alignment, semaphore pair, gu_blk/down_blk slot counters) and the
  mixed-layout mechanics are reused unchanged from the parent and r02-b04-a01, both bit-identical on device.

## Expected effect and risk
Per shape, roughly the better of the two nodes: gpt-oss ~277 µs, dsv4-flash ~322, kimi-k2.7 ~745,
dsv4-pro ~1084, the rest at the r01-b04-a04/b04 level. Estimated score ~1.54 (vs 1.5314 best), +0.8%,
which is at the edge of noise. Numerics should be bit-identical: whole experts per group, no change to math.
Risks: argument-offset mistakes (PHASE_A_SEM_RT now depends on num_layouts) would hang or deadlock the two-phase
shapes. A planner-cost increase (make_plan and plan_groups both run on every RISC) would show up as a uniform few-µs shift.
