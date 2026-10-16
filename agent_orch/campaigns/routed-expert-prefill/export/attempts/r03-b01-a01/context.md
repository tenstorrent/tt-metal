## Files read
- agent_orch/WORKER.md, campaigns/routed-expert-prefill/campaign.yaml: rules, allowed paths, accuracy gate.
- history.md: leaderboard and per-node "next" lines.
- kernels/compute/fused_swiglu.cpp (adaptive_chunk:: uses): confirmed the compute kernel only consumes
  plan_launch / layout_at / layout_group / num_chunks / per_core_M_for_chunk, which r02-b02-a04 keeps unchanged in
  signature.
- `git diff --stat` r02-b03-a03..{r02-b02-a04, r02-b01-a04, r02-b03-a04}: the planner and compute lines touch
  disjoint files.
## Nodes consulted
- r02-b02-a04 (parent, best): mixed phase B; reflection next #2 asks for this combination.
- r02-b03-a04, r02-b01-a04: PACK-thread pipeline + short bodies; the source of the ported diff.
- r02-b03-a03, r02-b04-a04: short SFPU bodies (SiLU treatment in b04-a04 not in this lineage).
- r02-b01-a02, r02-b01-a03: phase zones, activation cost measurements, DN head-start effect.
- r02-b03-a02, r02-b02-a01, r02-b04-a01: two-phase, mixed layouts, per-expert fixed cost.
- r02-b02-a02, r02-b02-a03, r02-b04-a02: DM look-ahead / VCs / writer-owned down stream; don't repeat.
- Round 1 (all 16 via history.md next lines): row groups, x tilize multicast, 10-row grid; read-ahead failures.
## Docs / external references
- none
