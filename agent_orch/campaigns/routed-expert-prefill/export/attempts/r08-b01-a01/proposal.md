# r08-b01-a01: bottom-first launch plan: extra experts and phase-B leftovers go to the lower row groups, never the top one

## Motivation
Since r07-b02-a01 removed the wrapped multicasts, per-row ends are deterministic (±4-5 µs per call), and
the top row group is the critical path on every DRAM-bound shape. Per-row kernel end on r07-b02-a01
(`/tmp/r06b01a01/ends.py`, mean of the 10 measured calls):

| shape | top group | middle | bottom |
|---|---|---|---|
| dsv3 | y2-5 509 | y6-8 453 | y9-11 387 |
| glm | y2-5 440 | y6-8 390 | y9-11 340 |
| kimi-k2.7 | y2-6 699 | | y7-11 646 |
| dsv4-pro | y2-6 996 | | y7-11 907 |
| kimi-k3 | y2-3 1210 | y6-7 1131 | y10-11 883 |

Per-job timelines (`/tmp/r08b01/jobs.py`, GU start | GU end | DN end per row) show where the tail comes from
on the two-phase shapes. Phase A (2x5) ends top-last by ~1.8x for the same work:

| shape | phase-A end per 2-row group, y2-3 .. y10-11 (µs) | phase B |
|---|---|---|
| dsv3 | 374 / 346 / 293 / 221 / 207 | 4+3+3: y2-5 starts 369 -> 508; y6-8 283 -> 452; y9-11 196 -> 384 |
| kimi-k2.7 | 588 / 558 / 525 / 401 / 352 | 5+5: y2-6 587 -> 698; y7-11 514 -> 645 (y8-11 idle 352..514) |
| dsv4-pro | 838 / 807 / 725 / 596 / 533 | 5+5: y2-6 835 -> 995; y7-11 720 -> 906 (y8-11 idle 533..720) |

So the phase-B expert dealt to the group that holds the top rows only starts after the slowest phase-A
group and then runs alone, at the end, at a single group's stream rate (dsv3 452 -> 508 has one stream,
~180 GB/s vs ~450 with 5 streams). And k2.7 / dsv4-pro's lower rows idle 150-190 µs because their 5-row
phase-B group includes y7, whose phase-A group is the middle one.

kimi-k3 (single-phase 2x5, 28 experts) deals 6/6/6/5/5 top-first; r07-b04-a01 measured the bottom-first
5/5/6/6/6 deal at -3.4% but lost on mistral / minimax-m3 by also leaving the top group idle on short deals.

## Mechanism
Planner only (`kernels/adaptive_chunk.hpp`), kernel-only, bit-identical math. Every RISC still runs the same
plan, so reader / writer / compute agree.
1. `schedule_in_order` (uniform layouts): ties go to the highest group index (lowest grid rows) when every
   group gets an expert (`active >= groups`, `active` counted once in `plan_launch`), else to the lowest
   index as today. So uneven deals put their extra experts on the bottom groups, and short deals (mistral,
   minimax-m3: 4 experts on 5 groups) keep the idle group at the bottom (r07-b04-a01 next #1).
2. Two-phase plans (`make_plan` / `plan_launch`): phase-B leftovers run on the layout's BOTTOM groups, the
   k-th largest on the k-th group from the bottom, and a candidate layout is only taken when its top group
   stays idle and covers the top phase-A group (`groups > n_left && rows[0] >= rows_a`). The model's t_b
   uses the actually used groups' heights.

Resulting plans on the test counts (`/tmp/r08b01/pt.cpp` replay):
- dsv3, glm: two-phase {2x5 + 4+3+3} is no longer >5% better than single-phase (the only top-free fits put
  the 5-tile leftovers at per_core_M 3 on 2-row groups), so they run single-phase 2x5 dealt 1/1/2/2/2 top to
  bottom (the bottom three groups take the 2nd expert).
- kimi-k2.7, dsv4-pro: phase A unchanged; the 2 leftovers run per_core_M 1 on 4+3+3's y6-8 and y9-11 groups
  (the 4-row top group y2-5 only finishes phase A).
- kimi-k3: 5/5/6/6/6 top to bottom.
- mistral, minimax-m3, dsv4-flash, gpt-oss: plan unchanged.

## Why this is not a repeat
- r07-b04-a01 flipped the tie-break unconditionally (k3 -3.4%, but mistral / mm3 +2.5..+3.3% from an idle
  top group). Here the flip is conditional, as its reflection's next #1 proposes; mistral / mm3 plans are
  identical to the root's.
- Phase-B placement has never been changed since r02-b02-a04 introduced the mixed-height phase B ("the k-th
  largest on group k, taller groups first"). r05-b02-a01 / r06-b01-a01 / r07-b01-a01 next items proposed a
  {3,3,4} reorder; this goes further and keeps phase B off the top phase-A group entirely, and lets the
  model drop two-phase on dsv3 / glm in favour of a bottom-heavy single-phase deal.
- r06-b02-a01 (start-up stagger) and b04's pacing change WHEN groups stream; this changes WHO gets the work.

## Expected effect and risk
- dsv3, glm: the top 4 rows do one expert each and stop (~370 / ~320 µs); y6-7 does 2 experts and should end
  ~470-490 (dsv3) / ~420-430 (glm): -3..-6%.
- kimi-k2.7: y9-11 starts its leftover at ~400 instead of y8-11 idling to 514; y6-8 at ~525; the top rows end
  phase A at ~590 plus extra contention. Bounded by the aggregate DRAM rate (~450 GB/s -> ~665 µs): -3..-5%.
- dsv4-pro: already at ~448 GB/s aggregate, so ~0 (risk: a different tail).
- kimi-k3: -3.4% (r07-b04-a01).
- Others: plans unchanged, noise only.
- Geomean +1..+2% if the estimates hold.
- Risks: (a) kernel-config size: local -O2 deltas reader +84 B, writer +92 B, TRISC0 -12 B; r07-b02-a01 shrank
  the reader/writer by 320 B, so it should fit. (b) A phase B with an idle top group is new in two-phase
  plans (idle groups exist in single-phase plans, e.g. mistral); phase-B groups still span whole phase-A
  groups' rows with equal block counts, so ring alignment is unchanged. (c) dsv3 / glm's single-phase 2-row
  per_core_M-3 second expert on y6-7 may be slower than estimated (it is the middle group); per-row ends
  will show it.
