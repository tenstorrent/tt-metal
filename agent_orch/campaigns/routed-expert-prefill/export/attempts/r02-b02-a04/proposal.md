# r02-b02-a04: on the best node's code (r02-b03-a03), let the two-phase schedule's phase B use a mixed-height layout (e.g. 4+3+3 rows) so an odd leftover of experts runs one per group at lower per_core_M

## Motivation
- Best node r02-b03-a03 (1.5435) still has a whole-expert tail on dsv3 and glm. Each has 8 experts of 5-6
  tile-rows on five 2-row groups. Three groups run 2 experts at per_core_M 3 and two groups run 1, so the
  light groups idle for the last ~45% of the call. Rows from r01-b04-a04 / r02-b04-a01: heavy rows end
  508-555 µs, light rows 250-313 µs. r02-b02-a03 measured max - mean of 117 µs (26%) on dsv3 and 109 µs
  (28%) on glm, mostly from this imbalance.
- r02-b03-a01 tried two-phase plans for these shapes with a uniform phase B (two 5-row groups). Its 3
  leftovers needed TWO phase-B rounds on group 0 (140 + 100 µs), and dsv3 lost 3%. r02-b03-a02 then
  limited phase B to one round. Since no uniform layout has exactly 3 groups, dsv3 and glm fell back to
  the plain 2x5 deal.
- r02-b03-a02's reflection (next #2) proposed this fix: a mixed phase B such as 4+3+3 rows takes 3
  leftovers in ONE round with 3 streams. The kernels already accept any layout index per phase. Mixed
  layouts {4,3,3} and {4,2,2,2} are already in the table, with host topology sets, from r02-b04-a01.

## Mechanism
Only `kernels/adaptive_chunk.hpp`, which is the planner every RISC runs. No host or kernel-loop change,
so there is no host rebuild.
- `Plan` gains `layout_b`, the layout index for phase B.
- `make_plan`: collect the leftover (phase-B) experts, largest first. There are fewer than phase A's group
  count, so at most 4. Then score every layout as a phase-B candidate, except phase A's own layout and
  uniform layouts smaller than phase A, when it has at least one group per leftover expert (one round).
  The k-th largest leftover goes on group k (layouts list taller groups first). The phase-B model is the
  max of `model_expert(t_k, rows_k, pcM_max, streams = n_left)`. Uniform layouts are scored first and a
  mixed one must be strictly better, so every plan that was two-phase before keeps its uniform phase B.
  The 5% margin over the single-phase model is unchanged.
- `plan_launch`: phase A is dealt as before by `PhaseDealer`. Phase B gives leftover expert e the group
  rank(e) in (tiles desc, index asc) order. For the uniform one-round phase B this is the same owner
  map the old in-order least-loaded deal produced.

Host replay of the test's seed-0 routing (/tmp/routed_r02b02a04/pt.cpp, compiled against the header):
- dsv3 and glm become phase A = 2x5 with one expert per group (the five largest: 6,6,6,5,5 at
  per_core_M 3), then phase B = 4+3+3 with three 5-tile experts at per_core_M 2.
- Every other shape keeps exactly r02-b03-a03's plan (kimi-k2.7 and dsv4-pro two-phase 2x5 -> 5+5;
  kimi-k3, mistral, minimax-m3 2x5; dsv4-flash and gpt-oss 3+3+2+2). The 8-row-grid plans are
  identical to the old header's.
- A 300k-case random fuzz (10- and 8-row grids) checks that every active expert is dealt exactly once,
  owners are inside their layout, each phase-B group gets at most one expert, and every phase-A group
  has equal chunks (the ring-alignment rule).

## Why this is not a repeat
- r02-b03-a01: uniform phase B needing two rounds. That lost on dsv3/glm.
- r02-b03-a02: one-round rule, uniform only, so dsv3/glm stayed single-phase.
- This node is the untried mixed phase B: 3 streams at per_core_M 2, one round.
- It is not a stream-count reduction like r02-b01-a01 / r02-b03-a01. The tail already runs 3 streams
  (the three heavy groups' second expert). It moves those 3 experts from per_core_M 3 on 2 rows to
  per_core_M 2 on 3-4 rows, using the 4 rows that idle today.
- The rest of the code is r02-b03-a03 verbatim. This also reverts my parent's look-ahead (r02-b02-a03),
  which its own reflection recommends.

## Expected effect and risk
- dsv3 / glm: phase A ends where today's light groups end (~300 µs, lottery included). Phase B then
  costs ~180-200 µs (pcM 2 compute ≈ 2 x 90 µs per tile-row at ~31 cycles per tile-mult; 3 streams of
  ~25 MB at ~400 GB/s ≈ 185 µs), so the end is ~490-510 µs vs ~555 / ~492. That is about -8..-10% on
  both. The model predicts -14%.
- Other shapes: identical plans, so noise only.
- Expected score: +1.5..2.5% if it works.
- Risks:
  - Phase-B groups straddle phase-A groups, so a phase-B group starts at the max of its phase-A groups'
    finish (lottery). The 4-row group spans phase-A groups 0-1, and the 3-row groups span 2-3 (rows
    4-6) and 3-4 (rows 7-9).
  - The 4-row group runs a 5-tile expert at pcM 2 with 3 phantom tile-rows, wasting compute on those
    cores. It is still pcM 2, the same as the 3-row groups.
  - A planner prologue cost on the phase-B scoring is only a few model_expert calls (≤ 6 layouts x ≤ 4
    experts).
  - Numerics are bit-identical, because every expert still runs whole on one group.
