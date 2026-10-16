# r02-b02-a01: mixed-height expert row groups: runtime layouts 3+3+2+2 and 3+3+3 (beside 10 / 5+5 / 2x5) with a height-aware largest-first expert deal

## Motivation
Every row group today has the same height (10, 5 or 2 rows on the 11x10 grid, r01-b04-a04), so a
group of R rows runs an expert of t tile-rows at per_core_M = ceil(t / R), and the only knob is
the uniform R. Two shapes lose a lot to that granularity:

- **gpt-oss** (4 experts of 8, 9, 9, 8 tile-rows) runs as 2 groups of 5 rows, 2 experts each at
  per_core_M 2: 4 M-units plus 2 per-expert fixed costs (bias, SwiGLU-OAI, activated multicast,
  down drain, 11 down K-blocks) per group. The fixed cost is large: r01-b03-a04/r01-b04-a04
  measured -20% M-units for only -8..-10% time, i.e. ~1 expert's fixed cost is worth ~2 M-units.
  With groups of 3+3+2+2 rows every group runs ONE expert: the 9s at per_core_M 3 on 3 rows, the
  8s at per_core_M 4 on 2 rows. Same 4 M-units on the critical path, one fixed cost instead of two.
- **dsv4-pro** (12 experts of 3 tile-rows) runs 5 groups of 2 rows at per_core_M 2, i.e. 25% of
  every expert's M-work is a phantom row, and the expert count (12 on 5 groups) leaves 3 rounds
  on the critical groups. On 3-row groups each expert is exactly per_core_M 1 (no phantom row),
  4 experts per group. The modelled makespan drops 9 -> 8 cost units; the shape is close to its
  DRAM floor (~420 GB/s now), so the real gain is capped at a few %.

Replaying the test's seed-0 routing through the cost model (/tmp/r02b02/picker.py) shows that no
other shape changes its pick: dsv3/glm/kimi-k2.7/kimi-k3/mistral/dsv4-flash/minimax keep 2x5
with today's exact expert deal.

## Mechanism
- `kernels/adaptive_chunk.hpp`: replace the uniform group size with a small table of layouts per
  grid height (`layout_for_grid`): 10 rows -> {10}, {5,5}, {2,2,2,2,2}, {3,3,2,2}, {3,3,3}+1 idle row;
  8 rows -> {8}, {4,4}, {2,2,2,2}, {3,3,2}. A layout's groups are consecutive rows from row 0;
  rows past its last group idle (no experts). `ExpertPlan<N>` caches every expert's tile count
  once, builds a largest-first order, and picks the layout with the smallest modelled makespan
  (same `expert_cost` units as today; ties keep the earlier layout, so the 3 existing layouts win
  every tie). Each expert goes to the group where it FINISHES first (load + its cost on that
  group's height, lowest index on a tie). For the uniform layouts this is exactly today's
  least-loaded deal in expert order; the mixed layouts deal largest-first.
- Reader / writer / compute(UNPACK): build the plan, derive (my group, its height, my row in it)
  from the chosen layout, and walk experts in the plan's order. Everything downstream (chunk_M,
  per_core_M, row mapping, x-blk owner count) is unchanged, it just uses the group's own height.
- Program factory: emit one weight-multicast topology set per LAYOUT (was per group size), from
  the same `layout_for_grid` table (the factory includes the kernel header, single source of
  truth). Idle rows get an inert set.

Numerics are untouched: every tile is computed by the same K-blocking, formats and fidelity; only
which core group computes which expert, and with how many rows, changes.

## Why this is not a repeat
- r01-b03-a04 / r01-b04-a04 added a whole-grid row count (10 rows, groups 10/5/2). This node adds
  NON-uniform group heights within one launch, which neither had, plus a height-aware deal.
- r01-b01-a01/b02-a01 introduced uniform groups; r01-b01-a03 (pacing) tried to fix imbalance by
  throttling and lost. This changes the partition, not the timing.
- Splitting one expert across groups (suggested by r01-b04-a04) is a different mechanism (re-reads
  weights, needs cross-group work) and is not attempted here.

## Expected effect and risk
- gpt-oss: -12..-25% (one fewer per-expert fixed cost per group). dsv4-pro: 0..-6% (DRAM-capped).
  All other shapes: identical schedule, so within noise. Score: roughly +2..+3%.
- Risks: a deadlock if any kernel derives a different layout/order (all use the same integer code
  on the same cached counts); a sender/receiver rectangle mistake on 3-row groups (would hang or
  give wrong rows on gpt-oss/dsv4-pro only); per_core_M 4 on gpt-oss's 2-row groups needs
  per_core_M_max 4 from the L1 fit (else the model sees 2 chunks and does not pick the layout).
  Accuracy must stay bit-identical (same per-tile math); any PCC change means a row-mapping bug.
