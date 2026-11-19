# Remaining stage-5 work after guard11

2026-11-13. This updates the older independent inventory in
`pending_requirement_matrix_integrated5_snapshot.md`; it is not acceptance.

The original stop rule controls: measured roofline >=70% and the last two kept
gains <3%, or8h occupied device time. Latest ledger `device_time_resume1810.json`
records1.4955h conservative occupancy; roofline remains below70%. TP4 time counts
once. Neither stop condition is met.

| Requirement | Current evidence | Remaining action |
|---|---|---|
| Topology/coherent families | Initial audit, residual/CCL/persistent families, flat/tiled prefill, packed projections,32 passing `ag_adapted9` whole-finish comparisons | Final decisions/artifacts; continue optimization under original stop rule |
| Native CCL | MMRS NoC repair; Linear AGMM receiver/admission repair, fresh-cache and queued/rebound/skewed traces, guard11 and independent rereview | Final source/runtime provenance and local commit; AGMM rejected on latency after adaptations |
| Precision × geometry |30 `precision_geometry5` rows:28 pass,2 BF4-attention KV-state failures; actual weight shapes/dtypes | Completed: four lowered-op profiles and64 phase/role/rank observations verified; BF4 remains stage8-only |
| MLP/activation | Earlier comparisons, prepared current-policy `mlp9` | Completed10 current-policy whole-layer rows; retain packed BF16 activation default |
| Recurrence | Cumulative/column exact probes; native packing differs bitwise but passes short PCC |Completed512-step oracle/control/native-pack,96 column and48 cumulative watcher cases;20 whole-layer recurrence12 comparisons finishing |
| Geometry | Broad readers/cores/blocks and376 prefill configurations | Four unrun full-attention down rows in `geometry3_full_tail_unrun.json`; adapt new material failures |
| Q/K norm | Four watcher batches and three `qknorm9` decoder rows pass | Completed all B1..32 watcher; final cumulative default gates remain |
| Cache ownership/nonalignment | B2/B32 ownership22 checks, skipped-page probes; B32 suffix L1 collision repaired with watcher proof | Final default contract/stack/watcher tests |
| Defaults/performance | GDN `integrated8`, attention `qknorm9`+L1 are candidates | Select after material comparisons; six final same-harness rows without overrides; B1 prefill nonregression |
| Profile/advice | Baseline/interim tables/CSV;10 advice adaptations | Final default profiles/advice/accounting and provenance |
| Context/capacity |262144 inherited, unchanged KV/state dtype/layout, current source-derived inventory | Final32K smoke,64-owner allocator/resource checks, context-contract stage5 entry |
| Final quality | Numerous focused and candidate proofs |68 serialized final-default validation jobs with fresh source pins |
| Runtime fallback |113 reachable functions,0 violations and negative controls | Final source audit plus runtime sentinel attachment |
| Review/commit | No stage verdict/checkpoint yet | Evidence manifest/gate; fresh stage-review, repair/rereview; local stage-owned commits and SHAs, no push |

`remaining_candidates11.sh` serializes quality, MLP, recurrence and precision
profiles. Parent owns all devices. Failed outputs are preserved; retries use
fresh names. Missing duplicate manifests are not distinct untried families.

The inherited KV cache already uses BFP8. Further BFP4 KV exploration requires
explicit allocation changes and actual dtype/context evidence; a policy-only
flag does not change the hardcoded allocator. The model is dense. Full-model,
generation and serving remain outside scope.
