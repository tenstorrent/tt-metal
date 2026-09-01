# Heterogeneous placement in the auto-mapper — plan index

Tracking issue: [#54623 — \[Auto-mapper\] Verify inter-mesh connectivity in heterogeneous placements via
SAT-based joint planning](https://github.com/tenstorrent/tt-metal/issues/54623)

Related: #40640 (SAT engine), #50510 (epic: auto-mapper blockers for blaze scale-out),
#52016 (pipeline-stage adjacency in MGD).

Three independent optimizations to the topology mapper, one document each. Each plan is self-contained:
context, design, ownership and function passing, validation, and its own open questions.

| # | Plan | Priority | One-line summary |
| --- | --- | --- | --- |
| 1 | [PGD-shape-aware inter-mesh constraints](TOPOLOGY_MAPPER_PLAN_1_PGD_SHAPE_INTERMESH_CONSTRAINTS.md) | 1 | Prune the inter-mesh SAT domain so shape-mismatched mesh pairs are unreachable, removing the dominant source of intra-mesh retry churn |
| 2 | [Incremental inter-mesh solving + stronger rejection](TOPOLOGY_MAPPER_PLAN_2_INCREMENTAL_INTERMESH_SOLVE.md) | 3 | Reuse the SAT encoding across retries instead of re-solving from scratch, and generalize the rejection constraints |
| 3 | [Connectivity-aware PGD grouping placement](TOPOLOGY_MAPPER_PLAN_3_CONNECTIVITY_AWARE_PGD_PLACEMENT.md) | 2 | Replace the per-shape maximum-coverage tiling with one adjacency-guided DFS that grows a mixed-shape placement along the MGD's own mesh graph |

Priorities are stated per plan and do not follow the numbering: plan 2 is deliberately last.

## Sequencing

1. **Plan 1** — self-contained, no solver changes, immediate reduction in retry churn. Ship first.
2. **Plan 3, §4(h) measurement** — check whether the per-grouping enumeration cap is being hit on the
   validation MGDs. The answer decides whether anchored enumeration is a prerequisite or a follow-up.
3. **Plan 3** — the adjacency-guided search, behind a fallback to the existing path.
4. **Plan 2** — needs the session-tightening fix in the solver bridge; its payoff shrinks once plans 1
   and 3 have removed most retries. Do it for the encode-once win, not for correctness.

## How the plans relate

Plan 1 fixes *which physical region a logical mesh may use*, given a set of regions. Plan 3 fixes *which
set of regions gets chosen* — and does so by keeping the candidate pool that `find_all_in_psd` currently
collapses to a maximum-coverage tiling. Plan 2 makes the retry loop that plans 1 and 3 mostly empty
cheaper still. Plans 1 and 3 are both required to close #54623; plan 2 is a performance change.

Plan 3 supersedes the earlier "seam check at the leaf of the packing DFS" approach. That version treated
the symptom: it filtered bad combinations out of a search whose tile boundaries were already frozen. The
8-chip example in Plan 3 §2 shows an MGD that is unsatisfiable over those frozen tilings no matter what
is checked downstream, which is why the fix moved upstream into placement itself.

## Validation MGDs

Both live in `tests/tt_metal/tt_fabric/custom_mesh_descriptors/` and run from the `bh-heterogeneous`
group in `tests/scripts/multihost/run_fabric_cpu_only_unit_tests.sh`:

- `bh_glx_2branch_mesh_per_stage_router_pipeline.textproto` — 69 single-rank meshes (60× 4×1, 8× 4×2,
  1× 4×4) forming a two-branch FABRIC-return fork off a degree-4 router mesh, 352 chips on the SC36 mock.
  The mesh-per-stage case from #54623.
- `llama_8b_4galaxy_unpinned_mesh_graph_descriptor.textproto` — the llama + audio 7-mesh ring with the
  tray-4 audio pinnings removed, on the four SC4 single-pod mocks. Maximum inter-mesh freedom, so the
  sharpest regression signal.
