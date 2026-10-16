# r02-b01-a01: DRAM-aware group-size picker (fluid DRAM-sharing model) so few-tile expert sets run 2 groups of 5 rows instead of 5 groups of 2 rows

Parent: worktree HEAD = r01-b04-a04 (1.5011, best node; the orchestrator calls it `root` for round 2).

## Motivation
The runtime picker (`adaptive_chunk::pick_group_rows`) chooses the group size by the makespan of
`units + chunks` per group (units = per_core_M). That model treats every group as independent
compute. The round-1 measurements say the big shapes are DRAM-bound while 5 groups run, and
compute-bound in the tail, when only the heavy groups are left:
- kimi-k2.7 (12 experts of 3-5 tiles, 5 groups of 2 rows, 3/3/2/2/2): groups 2-4 end at ~560 µs,
  groups 0-1 at ~745 µs (r01-b04-a04 reflection). The 5-group phase pulls ~440 GB/s of weights; the
  2-group tail (pcM 2, ~180 µs) pulls ~270 GB/s and is compute-bound.
- dsv4-pro (12 x 3 tiles): same 3/3/2/2/2 deal, tail 1122 vs 818-885 µs.
- dsv4-flash (8 x 4-5 tiles): 2/2/2/1/1 at pcM 2-3.
The baseline shows what a group does at per_core_M 1: one 8-row group runs one expert per
~99 µs (dsv3/kimi-k2.7), ~133 µs (dsv4-pro), ~65 µs (dsv4-flash), pulling ~250 GB/s. So TWO groups
of 5 rows at per_core_M 1 (every expert of <= 5 tiles fits) already saturate DRAM, have no idle
rows beyond the ceil padding, and have no tail (6/6, 6/6 and 4/4 experts per group).

A fluid model (each group = solo compute time, demand = weight bytes / solo time, shared DRAM ~400-480
GB/s, fitted on b04-a04, b01-a02 and baseline per-shape times; /tmp/r02b01a01/fit3.py) predicts the
(5 rows x 2 groups) layout vs today's (2 x 5) at: kimi-k2.7 0.93-0.96x, dsv4-pro 0.93-0.955x,
dsv4-flash 0.83-0.90x time, robust over the parameter range. It predicts it WORSE for dsv3/glm/
mistral/minimax (6-tile experts need pcM 2 on 5 rows) and ambiguous for kimi-k3 (0.97-1.10).

## Mechanism
Only `unified_routed_expert_ffn/device/kernels/adaptive_chunk.hpp` (shared by reader, writer and
compute, JIT; no host change):
- `pick_group_rows` keeps the same candidate sizes (kGridY, /2, /4), the same expert -> group list
  schedule (expert_cost) and the same strict-< tie rule (larger groups win ties). It now also records
  per group the units (sum per_core_M) and chunks (weight streams), and scores the layout with
  `modelled_makespan` instead of max(units + chunks).
- `modelled_makespan`: solo(g) = 16*units + 6*chunks; demand(g) = chunks/solo(g); DRAM = 1.5 expert
  weight reads per 16 time units. Groups progress at a common slow-down factor max(1, demand/DRAM),
  so they finish in solo order; the makespan is the sum of the phases. Integer only (32-bit with a
  64-bit accumulate), at most kMaxGroups phases. Weight bytes per expert = GRID_X x per-core
  tile-mults per tile-row, so the model needs no shape constants.
- Resulting picks (host-compiled check of the exact header, /tmp/r02b01a01/t.cpp): 10-row grid:
  kimi-k2.7, dsv4-flash, dsv4-pro -> 5 rows (was 2); gpt-oss stays 5; dsv3, glm, mistral, minimax,
  kimi-k3 stay 2; zero tokens / a single expert / large counts -> 10. 8-row grid: every benchmark
  shape keeps its old pick.

All kernels call the same function on the same counts, so they agree. Layout (5,2) is already
exercised by gpt-oss, so no new topology/protocol path. Numerics: each output tile is still one
core's full K accumulation in the same K-block order, so bit-identical results are expected.

## Why this is not a repeat
- r01-b01-a01 / r01-b02-a01 introduced the runtime picker and its units+chunks model; r01-b03-a04 /
  r01-b04-a04 added the 10-row layouts but kept the model. Nobody changed the scoring. Children
  suggested splitting tail experts across groups (r01-b04-a04 next #1); an M-split re-reads the
  weights per part, which the DRAM-bound shapes can't afford. Larger groups shrink the tail without
  extra weight bytes.
- Not pacing (r01-b01-a03), not ring depth, not DM timing.

## Expected effect and risk
- kimi-k2.7 -4..-7%, dsv4-pro -5..-7%, dsv4-flash -10..-17%; others unchanged (identical layout and
  code path). Geomean +2..+4%.
- Risk: two groups may not pull DRAM as hard as five (fewer outstanding weight streams); then the
  switched shapes could lose. Per-row TRISC1 ends in the profile will show it. No hang/accuracy risk
  expected (the 5-row layout already runs on gpt-oss); any accuracy change means a bug.
