# r02-b04-a01: mixed-height row-group layouts (e.g. 3+3+2+2 rows) with a height-aware LPT expert assignment, picked at runtime only when the cost model says they strictly beat the uniform 10/5/2 layouts

## Motivation
The two largest wins in round 1 were pure work-partition changes with bit-identical numerics: row
groups (r01-b01-a01, 1.44x) and spanning all 10 rows (r01-b03-a04 / r01-b04-a04, +2-3%). Every
layout so far is UNIFORM (groups of 10, 5 or 2 rows), so an expert's per_core_M is
ceil(tiles / rows) for one of three fixed heights. Many shapes waste a whole tile-row per core there:

- gpt-oss (tiles [8,9,9,8]): two 5-row groups, 2 experts each at per_core_M 2 (10 slots for 8-9
  tiles) -> 4 units + 2 serial expert pipelines per row. It is the least efficient shape (~200 GB/s
  of total DRAM traffic, groups balanced to <1%, b04-a04). r01-b03-a04 / r01-b04-a04 showed that
  its per-expert fixed cost is large (time fell 8-10% for 20% fewer units).
- dsv4-flash (tiles [5,4,5,5,4,4,4,4]): on 2-row groups the 5-tile experts need per_core_M 3, so the
  critical groups do 3+3 or 3+2 units (model makespan 7).
- dsv4-pro (12 experts x 3 tiles): 2-row groups need per_core_M 2 for 3 tiles (25% phantom), 3/3/2/2/2
  experts per group (model makespan 9).

A 3-row group runs a 9-tile expert at per_core_M 3 and a 3-5-tile expert at per_core_M 1-2; a 2-row
group runs an 8-tile expert at per_core_M 4 (the CB max). Mixing heights fixes the rounding:
replaying the test's seed-0 routing through the cost model (units + chunks, /tmp/r02b04a01/plan.py):

| shape | today (uniform) | with (3,3,2,2) | assignment |
|---|---|---|---|
| gpt-oss | 6 (5,5) | 5 | one expert per group: 9-tile experts on the 3-row groups (pcM 3), 8-tile on the 2-row groups (pcM 4) |
| dsv4-flash | 7 (2x5) | 6 | 5-tile experts on 3-row groups, every group runs 2 experts at pcM 2 |
| dsv4-pro | 9 (2x5) | 8 | 3-row groups: 4 experts at pcM 1; 2-row groups: 2 at pcM 2 |
| the other 6 shapes | unchanged | tie -> keep uniform | identical schedule to today |

## Mechanism
`kernels/adaptive_chunk.hpp`:
- A layout table per grid height: the three uniform layouts first, in today's order (10, 5x2, 2x5 on
  10 rows; 8, 4x2, 2x4 on 8 rows), then mixed ones: (3,3,2,2), (4,3,3), (4,2,2,2) on 10 rows and
  (3,3,2), (4,2,2) on 8 rows. Every group is >= 2 rows (kMinGroupRows) and <= kMaxGroups groups.
- `plan_groups()` replaces `pick_group_rows` + the inline `GroupScheduler::assign` calls. It writes
  `owner[e]` for every local expert and returns the layout index:
  1. Uniform layouts: today's in-order list scheduling, unchanged (same costs, same tie rules), so
     every shape that keeps a uniform layout gets the identical expert -> group map as today.
  2. Mixed layouts: only accepted if their modelled makespan is STRICTLY below the best uniform one.
     Experts are taken largest-first (LPT, stable by index); each goes to the group with the cheapest
     cost for it whose load stays within the cap (best makespan - 1); ties -> lower load -> lower index.
     Infeasible -> layout rejected.
  - Costs use a division-free path (ceil(t/h) by a <= per_core_M_max-step loop) for single-chunk
    experts; the multi-chunk case falls back to `expert_cost`. RISC-Vs here have no hardware divide,
    and the plan runs on all five RISCs of every core before any work starts.
- `layout_group()` gives a core its group index, base row and height for a layout.

Kernels (reader, writer, compute UNPACK): call `plan_groups` once, then `mine = owner[e] == my_group`,
`my_mt = my_gy - group_base`, `chunk_M = per_core_M_max * group_rows` with the core's OWN group height,
and the weight-multicast topology from the set for the chosen layout. UNPACK mailboxes its group's
height to MATH/PACK exactly as before. x / activated multicasts are per M-row and unchanged.

Host (`unified_routed_expert_ffn_program_factory.cpp`): emit one topology set (8 args: is_sender,
num_receivers, rectangle, sender) per layout of the table instead of per uniform size. Sender row stays
group_base + gx % rows; the existing collision TT_FATAL runs for every layout (none collide on 11x10 or
11x8). Pass nothing else new.

TILE / ROW_MAJOR x, biases, every activation, 0-token experts (owner = none) and experts_per_chip > 1
all go through the same code; only the expert -> group map and group heights change. Numerics are
bit-identical (per_core_M only changes which core computes which tile-rows; K accumulation order and
formats are untouched), as in r01-b03-a04 / r01-b04-a04.

## Why this is not a repeat
- r01-b03-a04 and r01-b04-a04 changed the set of UNIFORM layouts (adding 10-row groupings). This node
  adds non-uniform heights and a height-aware assignment; both nodes' reflections list "mixed group
  heights / cost-aware layouts" as the untried next step (r01-b03-a04 next-1, r01-b04-a04 next-1b).
- r01-b01-a03 (pacing) and the M-/N-split ideas are about the tail between equal groups; this is about
  per_core_M rounding, and it never re-reads weights (one chunk per expert, one group per expert).

## Expected effect and risk
- gpt-oss: largest effect, maybe -10..-25% (one expert per group instead of two; per_core_M 3-4).
- dsv4-flash: -5..-10% (it has ~13% DRAM headroom: ~390 of ~440-460 GB/s).
- dsv4-pro: DRAM-bound (~420 GB/s), so at most -3..-7%.
- Other shapes: same layout and same expert map as today; within noise. Kernel prologue gets slightly
  cheaper (division-free costs, one plan instead of plan + per-expert re-assign).
- Risks: a hang or wrong rows if the reader / writer / compute disagree on the layout or a group's
  height (all three call the same function on the same counts); a multicast rectangle error for 3-row
  groups (host-built the same way as today's). The eval would show it as hang / accuracy_fail. A
  shape could get slower if the model's unit count misjudges a mixed layout (fewer concurrent streams:
  4 groups instead of 5); that would show as a per-shape regression on exactly gpt-oss / dsv4-flash /
  dsv4-pro.
