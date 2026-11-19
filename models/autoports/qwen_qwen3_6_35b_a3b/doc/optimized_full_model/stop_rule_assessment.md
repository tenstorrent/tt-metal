# Stop Rule assessment — final gates complete

The stage started2026-11-16 01:33:26 UTC; default four-hour wall cap is05:33:26.
The two kept all40 transitions are25.081180984348673→22.27276945683746→
21.424391834701478ms queued token-out:11.197285842575509% and3.809035170862148%
improvement. Thus the last-two-kept<3% branch is **not met**. The frozen final
repeat21.41686824405214ms is confirmation of candidate2, not a third kept change.
**Stop Rule3 applies:** the user issued the stage time/token-cap wrap-up on
2026-11-16; recorded at 2026-11-16T05:34:07.267199+00:00. Keep the best fully validated
version and run only remaining required gates. No further optimization
experiments will run. Native profile/advice and long-context validation are
complete; the exact router L1/subblock controls found no kept gain. Broader
listed topology/geometry candidates are handoff notes. Branch1 is not claimed;
branch2 remains false. The cap never waives a correctness or review finding.

[candidate_search.csv](candidate_search.csv) retains87 exact extracted search
rows and [provenance](candidate_search.provenance.json). Failed geometry,
incorrect sampler setup and the failed candidate1 floor remain explicitly
classified. Different scopes are not added into a fictitious decomposition.

The final queued layer budget is
`30*0.5867155900341459 + 10*0.3066842349653598 = 20.668310050677974 ms`.
It uses selected residual carriers and no per-replay fence. Token-out is3.62%
above it before terminal allowance. This is an engineering lower-bound estimate
from real-weight fixed-input/router/position controls, not an exact natural
full-model decomposition. The older blocking estimate21.29762784989ms is
retained as a different measurement regime. See [accounting](perf_summary.json)
and [layer budget](final/layer_stack_budget.json).

## Completed final acceptance evidence

- `final/correctness_acceptance.json`: all40 AIME prefill and teacher top5=100%,
  top100=100%; exact161-ID AIME generation; six shared qualitative prompts;
  B32 fixed slots; all40 Watcher mixed/resumed/inactive replay; nine serving
  checkpoints; tracker positive control; final correct greedy sampler fixtures.
  See README and runtime_contract_audit for the bounded scope of each result.
- `final/performance.json`: clean all40 queued21.416868ms/token,46.6922t/s/u;
  useful-byte floor10.1001% passes.127 model/sampling pairs have no resident
  refresh/readback/recapture. Teacher logits20.933991ms and caller-observed
  delivery21.461293ms remain separate.
- `final/prefill_boundaries.json`: context4096 logical64/255/256/257/1025/2049
  pass warm prefill and two resident feedback steps with stable cache/traces.
- `final/queued_layer0.json` and `final/queued_layer3.json`:96four-rank output/cache
  exact checks per kind plus five200-replay queue samples with stable cache.

The32769/context65536 smoke passes with stable allocation/cache and two resident
replay pairs (zero host-boundary counters). Independent review returned clean-pass (stage_review_round1.md); local
checkpoint is complete (SHA in work_log.md). A performance Stop Rule never waives
a failed correctness, context, FUNCTIONAL or review gate.

## Every optimize checkbox, in installed order

Numbers below correspond one-for-one to the29 checkbox lines of installed
`tt-model-bringup/0.1.24/skills/optimize/references/evidence-checklist.md`.
`Inherited` refers to exact selected stage5 measurements/policy, not a fresh
stage7 gate. `Pending` is required work, not an optional handoff.

| # / requirement | Evidence or binding disposition | Status |
|---|---|---|
|1 Traced decoder/no host fallback | Final performance.json reports127 model/sampling pairs, zero refresh/readback/recapture and runtime guard pass; explicit host compatibility branch excluded | Measured; tracked all40 replay also passes |
|2 Decode L1 width shards | Selected B1 norm/projection shards plus structural sharded-head extraction; replicated BF16 inter-layer stream is the accepted contract | Inherited policy, retained; native scan/sparse interfaces are named exceptions |
|3 Prefill DRAM/2D | Existing bounded executor/selected prefill configs retained; semantic config key deduplicates equal policies without changing kernel fields | Retained; final TTFT boundaries pass; native prefill table/audit complete; broader2D geometry untried |
|4 Topology audit | `topology_audit.md`, `runtime_contract_audit.md`, structural/scan/slab audits cover full embedding→sampling path | Source/native reconciliation complete in final_profile_audit.md; cached embedding height limitation disclosed |
|5 Coherent mesh families | Stage5 `optimization_decisions.md`: directAR,RS→AG,true sharded residual,fused-output,BFP8CCL candidates including next consumers | Inherited measured decisions; user prohibits restoring rejected policy |
|6 Sharded residual without old-contract restore | Stage5 `batch_sharded_l0/l3`, `sharded_bfp8_l0`, `residual_contract.md` retain shards through next norm/add and lose | Inherited measured rejection, not API-error rejection |
|7 Strongest correct candidate comparison | Search CSV includes original, candidate1,candidate2, exact components and matched controls; candidate1 floor fail retained | Measured; final cumulative correctness passes with explicitly bounded output scopes |
|8 Final default reproduces best | Final default21.416868ms reproduces candidate2 within0.04%; fixed-token control kept separate from actual HF forcing | Measured final default; later production changes require affected refresh |
|9 Runtime dtype/fidelity rows | Selected source/config and native operand/fidelity/kernel rows in final_profile_audit.md | Measured native selected policy; no contradictory policy found |
|10 Optimized SDPA/composite ops | Selected paged SDPA retained; sensitive ordered recurrence cannot be replaced by rejected TF32/FDR arithmetic | Retained exact policy |
|11 Packed repeated projections | Stage5 packed linear/split-QKV controls and active-expert packed GU; structural splits removed where exact | Measured inherited groups; shared GU separate-family expansion remains untried, not declared rejected |
|12 Explicit memory/program/compute configs | Head r3/K4/core16/N32, sparse geometry and native kernel descriptors explicit; CSV+source hashes | Source and native program/operand/kernel verification complete |
|13 Dominant-role geometry sweep | Head readers/blocks/split/ballast; scan reader/cache/store families; native direct/product/slab kernels; inherited local/sparse geometry | Measured families; broader selected-fidelity cross-products and wider FP32-router core/subblock cross-products remain untried; exact L1/subblock advice measured without gain |
|14 Decode fidelity as knob | Stage5 LoFi/HiFi2 real B1/B32 screens; `lofi_repeat_l3` prefill regression; selected HiFi2 and active-expert LoFi fixed | Binding inherited selection; no new policy frontier authorized |
|15 Separate attention precision | Stage5 `attention_bfp4_l3` real PCC below0.995; LoFi and KV+LoFi screens recorded | Binding inherited rejection/selection |
|16 Expert/MLP BFP4 candidates | User explicitly preserves selected policy/rejection ledger and forbids broad full-model datatype frontier here | Assigned to datatype-sweep; not falsely claimed as a fresh stage7 experiment |
|17 Tile-dividing shards/grids | Head padded storage/logical vocab separation,32 scan workers, exact K512 roles, poisoned-row structural tests | Measured exact geometry; arbitrary larger grids not universally optimal |
|18 Blackhole DRAM readers1/2/3 | Head r1 first exceeds L1, adapted split4 fits/exact but loses; r2/r3 swept. Other roles inherit recorded geometry | Measured head family; role-specific inherited evidence only, no universal sweep claim |
|19 Collective topology minimized | Batched directAR, removed singleton padding/permutes, no inter-layer CCL; stage5 residual families measured | Measured/retained; final_profile_attribution.csv names native CCL work |
|20 Fused matmul-CCL adapted | Stage5 repaired fused-output candidates lose after layout adaptation; exact KV constraints retained | Inherited measured rejection, not first API error |
|21 Persistent CCL buffers | Decoder workspace/scratch enabled; sampler explicit-output barrier/ownership candidate investigated in `sampler_persistent_audit.md` | Rejected with raw poisoned-gather repro and native tail-packet diagnosis in sampler_persistent_broadcast_rejection.md; production unchanged; captured addresses alone do not satisfy explicit-buffer claim |
|22 Active-expert MoE | Indexed TP4 B1/grouped EP4 B>1, sparse GU/down tuned, nnz/score combine retained; no dense all-expert substitution | Inherited measured policy; final B32 fixed-slot/chip0 inactive smoke passes; all40 mixed serving and Watcher replay pass |
|23 Terminal/sampling/token feedback | Head exact real-terminal sweep; fresh split/power2/argmax correct-greedy table; localtop32 candidate gather, persistent tt_out_tok/device positions | Measured; native split sampling confirmed, clean sampler2.28% of full40 time; no force-argmax/full-vocab gather/generic vocabulary TopK |
|24 DRAM-sharded LM head | r3/K4 selected including norm/layout costs; legal r1/r2 controls | Measured; stored duplicate legacy weight/capacity must remain disclosed |
|25 Appropriate precision experiments | Real stage5 weight/activation evidence carried; stage7 head/scan/slab preserve arithmetic and dtypes | User-specified scope satisfied without broad search; final frontier belongs to datatype-sweep |
|26 Reconciled accounting/perf_summary | Candidate2 queued/observed/model-only/sampler/HF-teacher numbers separate; useful-byte floor passes; current optimized layer estimate available | Final perf_summary.json and nonblocking layer budget present; native attribution/advice complete in final_profile_audit.md; separate controls are not exact decomposition |
|27 Preserve batch≤32 | B3 production slab length129/16decode exact pass; final/b32.json passes all40 fixed slots and chip0 inactive state | B32 bounded gate and all40 mixed/serving checks pass; exact state snapshot scopes remain disclosed |
|28 Multi-row/resumed prefill/warm buckets | Full-model contract inherited; final runner all40 replay covers lengths/resume/mixed with Watcher10; semantic-key CPU perturbation checks | Final all40 Watcher replay passes17cases/32checkpoints; context32769/context65536 smoke passes (final/prefill_context32769.json) |
|29 optimized-vLLM serving-specific gates | User explicitly excludes vLLM integration | Not applicable to this stage |

## Evidence and final-audit lines outside the checkbox block

| Required group | Concrete disposition |
|---|---|
| Functional/PCC, paged cache, stress, representative kinds | Exact selected layer0/layer3 and layer34/B3 controls exist; slab128-step rounded/unrounded×DRAM/L1 four-rank bitwise checks pass. Fresh all40 prefill/teacher top5/top100 pass; final all40 replay and serving v3 pass |
| Warm prefill/decode and context | Before/final clean decode and128-token TTFT are separate rows. Same-context boundary TTFT passes;32769/context65536 smoke passes (final/prefill_context32769.json); unchanged KV avoids unnecessary maximum rerun |
| Watcher | Isolated changed kernels have Watcher10 evidence. Final all40 Watcher10 replay passes fresh/resumed/mixed/inactive/sampled paths; logs preserved. Separate changed-kernel Watcher controls cover physical padding/all-rank exactness |
| Shared qualitative/degenerate output | Fresh exact-ID AIME and six matched128-token shared controls have bounded human PASS verdicts above; degenerate check0. No completed-task accuracy or broad HF quality equivalence claimed |
| Native report/advice | Reduced real0/3 plus terminal profile validates29440 executions. final_profile_audit.md records native policy, every advice disposition and measured router_advice_comparison.md controls (no kept gain) |
| Layout and host boundaries | Runtime audit plus component/current queued guards; final_profile_audit.md identifies remaining casts, head reshard, padding and stable-logit copy. Host logits compatibility never substitutes for measured device sampling |
| Trace lifetime/acknowledgments/preparation | Tracking1/skip0 final queue, tracker positive control with `--generate`, all40 changed inputs and cross-request reuse. Any last buffer/sampler/CCL edit invalidates earlier acceptance and needs repeat |
| Explicit config and async serving contract | Exact source descriptors retained; existing serving command probe covers asynchronous output leases/device token paths. No vLLM work is implied |
| Local review/commit | Independent round1 review: clean-pass, no required work. Local checkpoint/SHAs are recorded in work_log.md. |
| vLLM benchmark/no-profiler exceptions | Not applicable; this non-serving stage still requires tt-perf-report |

## Untried versus blocked versus required

**Review complete:** independent round1 clean-pass, with no required work. Local
checkpoint is recorded in work_log.md; no failed gate was waived. First L1 failures were repaired into legal
head controls; no entire reader family is rejected merely from allocation failure.

**Not tried (Stop Rule3 handoff):** embedding replication/batched alternatives; head row-parallel or
fused norm/head; wider output-core/subblock combinations at selected fidelity;
shared GU packed-versus-separate producer/consumer comparison; broader B32
per-role geometry/grouped-expert combine tuning; whole-stack public batched
residual interface; additional prefill2D/chunk geometry; wider FP32-router core/subblock cross-products;
explicit sampler broadcast padding/native partial-packet repair. These are not claimed
slower or impossible. If the final profile identifies one as applicable material
advice, it becomes current work unless a documented Stop Rule governs the handoff.

**Policy-owned work:** a new expert/shared/attention/KV datatype Pareto frontier
belongs to `$datatype-sweep`. Rejected residual/dtype/fidelity policies stay
rejected here. The historical owner-carried stage5 linear speed limitation does
not waive stage7's full-model floor or any correctness requirement.
