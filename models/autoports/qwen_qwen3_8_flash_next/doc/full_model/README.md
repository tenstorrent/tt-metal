# Full autoregressive model — independently reviewed clean-pass

Clean all 48 batch-1 prompt 128/generate 128 performance is **TTFT 7038.67ms** and
**7.207 tokens/s/user** token-out decode (`acceptance_perf_final_v1.json`).
The separate model-only fixed-token trace measures 7.238 tokens/s/user; it is
not teacher-forcing accuracy. Steady decode reuses model/sampling traces with
zero token, position or page-table refreshes, captures or releases. All 12 QSA
layers on all 4 ranks end at position 255. Tracker, watcher and profiler are off.

Current all 48 B1 HF accuracy passes on the recorded production source:
prefill top-1/top-5/top-100 is **97%/100%/100%**, and traced teacher-forcing
decode is **96%/100%/100%**, each over 100 AIME24 chat-template predictions
(`acceptance_accuracy_resume_final_v1.json`). The installed retained-state
resume check passes starts 32/68/256 of a 257-token natural prompt, with 100%
top-1 agreement and minimum reported PCC 0.999965677. Both `no_history` and
`fresh_suffix` controls fail as expected at starts 32/68; the one-position
start 256 case does not run controls under the existing 64-position control rule.
The run uses allocation tracking and is accuracy evidence, not clean performance.

The integrated production generator passes the reduced tracked serving suite
(`serving_probe_v7.json`, 16/16 checks) and all 48 B2 shared-pool prefill/signature
checks (`prefill_factored_full48_v1.json`). The latter includes the shared
prefill workspace and bounded factored decode: resumed starts 31→265 and 68→257
are bitwise exact, mixed/reversed outputs agree, matched-seed decode state and
token feedback agree, and trace IDs/program counts remain stable. These are
scoped serving and retained-state correctness results. The separate full 48 B32
HF/shared-pool check now also passes (`batch32_hf_factored_v2.json`): each of two
matched histories scores 2645/2704 top-1 matches (97.818%) and 2704/2704 top-5
and top-100 matches. The metric is actual traced TT greedy-token membership in
HF top-k. Every per-row/per-phase accuracy group passes the unchanged bar. These are correlated predictions from one cached AIME reference, not 2704
independent prompts. Repeated/migrated logits, tokens and final request state
are exact; dormant state and all-inactive behavior are preserved. Four additional
resident sampled-feedback steps pass with no token/position/page uploads, capture
or release. Their position readback covers all 12 QSA layers and all fixed slots
on rank0; the separate clean B1 run checks all four ranks. This tracked run is
correctness evidence, not B32 performance.

Qualitative review is complete for the recorded all 48 B1 text revision:
`qualitative/verdict.json` reports HF **0/10** and TT **0/10** blind-scored
failures for each of six prompts over matched seeds 0–9. The owner also read all
six greedy pairs and both AIME completions. Both backends show coherent output
within the matched 128-token window; checkpoint thinking and cap truncation
limit final-answer/code assessment. The text run predates B>1 factoring and
formatting, with the B1 semantic path retained. It saved all outputs but exited 1
in its final reporter after device teardown; the raw failure and reporter-only
repair remain recorded in `acceptance_text_v1_process_status.json`.

Reduced C262144 terminal attribution is complete, with eager/profile versus
full48 clean-trace boundaries documented below. Once-only maximum-context
execution passes. [Independent review](stage_review_round1.md) returned **clean-pass**
with no required work; [checkpoint records](commits.json) track local closure. The older all 48 HF result is
97/100/100% top1/top5/top100 (`full_accuracy_v2.json`), for its earlier source.
The frozen cross-active-width PCC diagnostic remains FAILED. The owner
[disposition](batch_geometry_disposition.md) records matched original-HF
width/history sensitivity and independent unchanged HF top-k passes; it does
not relabel the equality test or change precision. Independent review accepts that scoped known-limitation
disposition; historical failures remain failed. The B32 test supplies same-history state/accuracy
evidence and does not itself adjudicate the cross-width diagnostic. Advertised
262144-token cache allocation is not maximum-context execution.

## Model and inherited execution policy

`tt/model.py` implements HF token embeddings, all 36 GDN and 12 QSA blocks,
layer1 PLE, the terminal hyperconnection mixer and the untied vocabulary head.
It uses the accepted optimized `MultichipDecoder`; each of four Blackhole
chips retains one complete 2560-wide hyperconnection stream between layers.
Embedding width and LM-head vocabulary are sharded. There is no host model
compute or single-chip/replicated decoder fallback.

The inherited TP4 attention, EP4 BFP4/LoFi routed experts, selected DRAM
projection configs, BF16 activation/CCL and FP32 recurrence remain unchanged.
Paged QSA K and V both use BF16/page32, as selected at stage5. The historical
BFP8 KV candidate was rejected for latency and is not reinstated here.
Selected-indexer and packed PLE policies and the decoder rejection ledger
remain authoritative. The HF checkpoint revision is
`812d96e10a0dd82b69d749e68693698cf6c36daa`.

## Generator and sampling

`tt/generator.py` implements the readiness generator and explicit low-level
prefill/decode. Its persistent fixed-slot caches hold attention, recurrence,
convolution and PLE token-history state. Caller-owned cache can be bound at
construction. Logical prompt tails 1..32 use six canonical dyadic buckets
(1/2/4/8/16/32); host reads slice
to the requested lengths. One persistent prefill output remains separate from
the sampler canvas; each request's last logical row is copied directly into its
physical sampler slot before workspace reuse. Positions advance within device
decoder traces.

Compact device checkpoints preserve state before an odd prompt's final
singleton. A matching retained-state continuation restores that checkpoint,
replays the old tail with new input through the existing multi-row path, and
omits the replayed prefix output. This preserves the accepted singleton FP32
versus multi-row BF16 MoE arithmetic. Reset/decode invalidate only affected
requests; slot migration carries checkpoint ownership. The integrated all 48 B2
factored/shared-workspace result confirms exact starts 31/68 continuation after
the earlier checkpoint revision first closed the historical odd-tail failure.

Current source uses bounded factored decode for configured batches above one:
fixed per-slot attention/state traces, power-of-two MoE buckets, and fixed
per-slot terminal traces. Only valid MoE activation rows are repeated for bucket
padding; padded rows do not execute attention or mutate request caches. Standalone
B1 retains its whole-model trace. Traces remain retained until explicit release;
there is no subset-cache eviction, eager fallback, or switch to a replicated
decoder. Production integration passes reduced servingv7, all 48 B2
prefill/submission-order checks, and the all 48 B32 HF/shared-pool test at C512.
The B32 test retains 1857 model segments, six prefill traces and one sampling
trace without request-time releases; 2179 programs remain unchanged. Its measured
allocation is scoped to that C512 pool geometry. Current retained-memory and
worker-expanded-pool arithmetic are reconciled in `final_capacity_reconciliation.md`;
expanded545/8194-block pools remain calculated budgets, not executed settings.
B1 maximum-context execution now passes separately; no B32 clean-throughput claim is made.

Model and sampling traces are separate. Common `SamplingGenerator` receives
the persistent decode-token tensor as `tt_out_tok`. Seed progression uses an
integer device helper implementing the common request-seed hash; common code
still performs token sampling. Device seed-only replay passed on all 4 ranks
(`device_seed_device_v1.json`). Actual common sampled-token integration also
passes with allocation/program-cache tracking (`seeded_sampling_device_v2.json`):
42 checks, 6 captures and 41 replays cover reset, row order, migration, inactivity
and sampled prefill. The first attempt exposed and repaired missing seed-op
binary loading before capture; `AUTOFIX.md` retains the failure evidence.
The explicit `host_sampling=True` compatibility path implements host request
features and is excluded from measured device token-out throughput.

`sampling_contract_audit.md` compares the two common sampler implementations.
`serving_contract.json` pins the read-only plugin audit and records each row
with its exact supported/unsupported or pending scope and evidence. B32 results
do not automatically establish unrelated capacity or serving features. No vLLM
adapter, installation or server is implemented.
KV-only prefix reuse has the documented recurrent-state exclusion. Retained-state
resumed prefill passes starts 31/68 exactly in the integrated all 48 B2
`prefill_factored_full48_v1.json`; it does not reconstruct recurrent state from
shared KV blocks. The serving ledger records42 supported generator equivalents,
21 unsupported declarations and no missing rows. The capacity API is supported
by actual maximum-context execution and explicitly calculated worker expansion;
no adapter/server execution or independent review verdict is implied.

## Current integrated evidence

| Artifact | Observed result |
|---|---|
| `acceptance_max_context_final_v1.json/log`, `max_context_attempt.json` | Once-only full48 B1 public nonaligned262143-token prefill + one decode/two sampled outputs consumes262144 context.8196 prefill replays,1 model replay,2 sampling replays; all12 QSA layers/all4 ranks at262144 and terminal logits finite on all4. JSON PASS, completion markers and clean UMD close; outer monitor interrupted, no numeric process-exit claim. No quality or performance claim. Post-finiteness final allocated/free/largest per bank4099126528/173214848/165250368. |
| `batch32_hf_factored_v2.json/log` | Current all 48 B32/C512, 12 independent global QSA K/V pools: baseline and reversed-slot migration each score 2645/2704 top1 and 2704/2704 top5/top100. Prefill top1 30/32; decode top1 2615/2672; both phases and all per-row groups reach 100% top5/top100. One correlated cached AIME trajectory. Same-history logits/tokens/final state and dormant state exact; all-inactive no-op; four resident feedback steps pass. 1857 model + 6 prefill + 1 sampling traces retained, 2179 programs stable. Exit 0; 1296.826s tracked diagnostic wall time, not throughput or maximum-context evidence. |
| `acceptance_perf_final_v1.json/log` | Current all 48 B1/context262144, clean tracker0,128 prompt/128 generated: TTFT 7038.672ms, token-out138.754269ms (7.206985t/s/u), model-only fixed-token138.152610ms (7.238372t/s/u). Steady127 model and127 sampling replays with no host feedback/position/page refresh or capture/release. All 12 QSA layers across four ranks end at position255. |
| `acceptance_accuracy_resume_final_v1.json/log` | Current all 48 B1/context262144, serial tracker1: prefill97/100/100 and teacher-forcing96/100/100 over100 predictions each. Source hashes match the inspected production files. Resume passes;360.81s diagnostic wall time. No clean-performance or maximum-context claim. |
| `../../readiness/prefill_resume_check.json` | Current all 48 L257: starts 32/68 each compare129 positions, start 256 compares1; all top1 agreement1.0 and last-token equality, minimum PCC 0.999965677. Both negative controls reject at32/68; none at256 because its one-position suffix is below control_min_window64. Retained full state, not KV-only prefix reuse. |
| `serving_probe_v7.json/log` | Production factored decode with checkpoints, feedback and shared workspace; reduced0/1/3 B2/context128:16 tracked checks pass,2180 programs unchanged,22 captures retained during requests and released on explicit unload. Instrumented 49.39s is not performance evidence. |
| `prefill_factored_full48_v1.json/log` | Integrated all 48 shared-pool B2/context384: starts 31→265 and 68→257 bitwise exact; mixed/reversed logits and matched-seed decode order/state/token feedback pass, six prefill signatures retained,2182 programs unchanged, no request-time release or late compilation. 272.66s is diagnostic wall time; no HF top-k/B32/stage claim. |
| `qualitative/verdict.json` | Complete blind 120-example sampled comparison: each of six prompts HF 0/10 and TT 0/10 failures; six greedy pairs and AIME pair directly reviewed. Exact reused HF controls, pinned chat template/recipe/seeds, frozen hashes and source scope recorded. Final mathematical/factual/code completeness is not assessed at 128 tokens. |
| `acceptance_text_v1_process_status.json` | All 48-layer B1 AIME 128-token and TT 66-request outputs saved before final reporter TypeError; actual process exit 1 preserved. Reporter-only fix passes 8 CPU tests; no generation rerun or process-pass relabeling. |

## Completed profiler evidence and performance boundaries

[terminal_final_analysis.md](terminal_final_analysis.md) and its JSON record the
completed reduced layers0/1/3 B1/C262144 **eager** attribution: one measured
iteration after two warmups, no live traces, and very short live history.
Coverage independently reconciles21488 host/device/final operations, all six
signposts, positive durations and equal per-region operation counts on all four
ranks. This is distinct from the clean full48 traced token-out result above;
neither eager merged kernel sums nor signpost wall times are advertised throughput.

Split greedy remains selected: merged kernel sum0.429263ms versus force-argmax
2.886563ms (6.724× slower), with both returning the actual greedy token243975
on the same frozen logits. Split is3.673% of the reduced model-plus-split kernel
sum. These results do not show sampler dominance. Current embedding/gather costs
0.036766ms, final mixer0.372330ms, head including layouts0.668237ms and native
row assembly0.045470ms. The head retains BF16/HiFi4/FP32 DEST; no head or sampler
advice remains. No new sampler or precision change is selected.

The inherited QSA scorer costs2.387150ms at allocated C262144 versus historical
C128:0.025220ms, while the native causal selector remains approximately0.01017ms.
[performance_boundary_audit.md](performance_boundary_audit.md) distinguishes this
capacity-dependent scorer work from the short live-history selector and the
historical weighted-decoder estimate. Exact full48 critical-path decomposition
remains unmeasured; multiplying reduced costs by12 would not establish it.
Some BinaryNg shape annotations disagree with source/adjacent metadata; the
analysis discloses this and derives no FLOP/bandwidth claims from those fields.

Raw capture `terminal_final_tracy_v1/` is retained uncompressed in place;
[raw_artifacts.json](raw_artifacts.json) records its persistent location and
retention boundary. `terminal_final_perf_v1/coverage.json` and final reports
preserve coverage and durations; analysis JSON records their hashes. Historical
failed captures remain retained. Maximum-context execution passes separately;
fresh independent stage review and local commit remain unfinished.

## Historical evidence and controlled limitations

These artifacts retain their original source scope and outcomes. A later scoped
pass does not rewrite a historical failure or imply full-stage acceptance.

| Artifact | Observed result |
|---|---|
| `full_accuracy_v2.json/log` | Earlier head/generator revision: all 48 prefill100 and traced teacher100 pass97/100/100%; splits32/68/256 pass. Preserved historical evidence; refreshed current accuracy is listed above. |
| `full_accuracy_v1.json/log` | Historical revision:95/100/100%; resume failed at68 before the native repairs. |
| `ragged_prefix_dependency_v1.json/log` | All48 B2/context2048: lengths300/668, starts283/68 and24 predictions per row pass numerical resume equivalence; top1agreement100%, minimum PCC0.9999663,10908 program entries unchanged, tracking enabled and clean close. No semantic recall verdict. |
| `generator_prefill_trace_v1.json` | Reduced0/1/3 deterministic traced feedback and forced tokens; earlier generator revision. |
| `device_seed_device_v1.json` | Exact64-bit hash, carry, inactive rows, migration, repeated tracked draws pass. |
| `seeded_sampling_device_v2.json` | Common sampler integration passes42 checks with tracked replay and request lifecycle. |
| `serving_probe_v6.json/log` | Integrated checkpoint/token-feedback/shared-workspace revision before production factoring, reduced0/1/3 B2/context128:16 tracked checks pass,2177 programs unchanged,10 traces released once and clean caller mesh close. Instrumented43.90s is not performance evidence. |
| `prefill_checkpoint_full48_v1.json/log` | All48 shared-pool B2/context384 checkpoint revision before workspace/factoring: starts 31→265 and 68→257 bitwise exact; mixed/reversed logits and matched-seed decode order/state/tokens pass, six prefill trace IDs retained,2182 programs unchanged. Scope is prefill signature correctness, not refreshed HF accuracy or stage acceptance. |
| `serving_probe_v5.json/log` | Earlier head/row-fill/six-bucket revision, reduced0/1/3 B2/context128:16 tracked checks pass,2179 programs unchanged,12 traces release once; tracker negative control and final replay pass. Instrumented38.80s is not performance evidence. |
| `prefill_signatures_full48_v2.json/log` | Historical pre-checkpoint all 48 shared-pool B2/context384 failure: mixed/reversed logits exact, six retained prefill signatures reused without release,2181 programs unchanged. Resume68 exact; resume31 fails top1 0.952991/PCCmin0.682564. Decode row-order logits/tokens/feedback exact; unseeded-row harness defect prevents overall state verdict. The later checkpoint artifact repairs resume and seeds; this failed artifact is retained unchanged. |
| `serving_probe_v4.json/log` | Reduced real0/1/3, B2/context128:16 tracked checks pass, including explicit host mode, resident feedback, changed/unchanged pages, sampled migration, immutable delayed snapshots and unload.10905 program entries unchanged;20 traces release once. |
| `serving_watcher_v1.json/log` | Watcher corroboration passes the same16 checks,10905 program entries unchanged and clean close; no ERROR/ASSERT/CORRUPT hits. Raw watcher evidence retained separately. |
| `shared_pool_generator_v5.json/log` | Exact uninterrupted/migrated B2 and private/pool lifecycle state; overall fails later-token B1/B2 decode accuracy (token43 PCC0.97965, top1/top5 agreement0.5). Raw failure unchanged; current cross-width HF controls and owner disposition are recorded separately in `batch_geometry_disposition.md`, pending independent review. |
| `sampled_prefill_lifetime_audit.md` | Historical v3 live temporary-row allocation repaired before sampling replay; v4 confirms that revision under unchanged tracking. Current streamed workspace drops each slice before the next replay; servingv7 supplies reduced production evidence and prefill_factored_full48_v1 supplies the scoped all 48 B2 confirmation. |
| `reduced_resume_aligned.json` | Historical diagnostic: alignment alone improved split68 but top1agreement94.6% failed; superseded by native repair evidence below. |
| `reduced_resume_rms_v5.json/log` | Scoped RNE and native row-reduction repair: splits32/68/256 pass100% agreement; HF smoke100/100/100. |
| `rms_native_regression.log` | Ten native row-placement, BF16, integer and max regression cases pass. |
| `capacity_plan.json`, `full_load_v1.json` and `full_accuracy_v2.json` | Earlier full262144 cache allocation fits. Historical all 48 run records loaded/prepared/final DRAM; current head/checkpoint/workspace/factored charges require final high-water reconciliation; short workload only, not max-context acceptance. |
| `qualitative/hf_outputs.json`, `hf_validation.json` | Original66 HF controls (6 greedy,60 sampled) completed and validated; reused by exact cache match for the completed paired verdict above. |
| `serving_upload_triage/` | Live dispatch upload stall evidence, before model execution. |

The earlier diagnostic accuracy timing included allocation tracking and concurrent
HF CPU controls. Current acceptance accuracy is serial but still uses tracker1;
its teacher-forcing TTFT11606.33ms, decode6.7161t/s/u and end-to-end3.7955t/s/u
include instrumented reference-token feedback and must not be quoted as clean
token-out performance. The completed clean performance run above separately
measures public TTFT, model-only fixed-token replay and token-out decode for
batch1 prompt128/generate128. Reduced-stack terminal attribution and performance
analysis are complete as separate evidence above. The advertised context remains
262144; `../context_contract.json` records arithmetic and measured allocation,
with maximum-context execution now passed and historical measurements preserved.

Exact commands, failures, recovery and artifact provenance are in
`work_log.md`. Full 48 B32 cache/allocator geometry and matched-history request
isolation now have the scoped C512 evidence above. The serving ledger owns
individual feature dispositions. Current capacity arithmetic and the once-only
maximum-context execution are reconciled. B1 final free1385718784 bytes/rank
minus786432 bytes of calculated worker pool expansion leaves1384932352; the
expanded8194-block pool itself was not executed. Reduced C262144 terminal
attribution is complete with the eager-versus-traced limits above.
Independent review remains pending; stage review has not passed, and no stage
completion or local stage commit is claimed.
