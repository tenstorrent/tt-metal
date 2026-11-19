# Optimized full model — clean-pass

| Warm batch-1, 128 input / 128 output | Before | Final measured path |
|---|---:|---:|
| TTFT, including first sampled token delivery | 2787.782 ms | 2729.973 ms |
| Queued traced token-out, persistent device feedback | 25.081181 ms/token; 39.8705 t/s/u | 21.416868 ms/token; 46.6922 t/s/u |
| Caller-observed token delivery, including per-token readback | 25.158932 ms/token; 39.7473 t/s/u | 21.461293 ms/token; 46.5955 t/s/u |
| Fixed-token model trace, without sampling | 24.601166 ms/token | 20.934685 ms/token |
| Traced HF teacher inputs to logits, uploads included; no sampling | 24.599893 ms/token; 40.6506 t/s/u | 20.933991 ms/token; 47.7692 t/s/u |

The final default reproduces the stage-7 FUNCTIONAL floor: 21.416868 ms/token
against the 21.631299 ms ceiling (46.2293 t/s/u), reducing decode latency by
14.61%. Accuracy, runtime, performance and context gates pass. Independent
[stage review](stage_review_round1.md) returns **clean-pass**, with no required work. All40 accuracy, Watcher replay, B32, qualitative and serving-command
checks pass. TTFT is 2.07% lower in this final run; this small difference lies
within observed run-to-run variation, so it is not evidence of a robust TTFT gain.

The rows come from [before.json](before.json) and [final/performance.json](final/performance.json), measured without profiler,
watcher or trace-allocation instrumentation. Both model and sampling traces run
nonblocking with persistent token/position/RoPE/page-table state. The queued loop
has zero host refreshes, readbacks or captures. Teacher-forcing decode, which
uploads forced tokens and omits sampling, is a separate metric in that artifact;
it is not the token-out number above.

The target is Qwen/Qwen3.6-35B-A3B revision
`345d467b117d1fd5eda9b879a5988de9ae5f47a7`, all 40 layers on a 1×4 Blackhole mesh.
Preserve the selected decoder arithmetic, BF16 KV and CCL policy, replicated BF16
inter-layer residual contract, rejection ledger, fixed slots and non-aligned
prompts. No vLLM integration or datatype frontier search is part of this stage.

[Topology and bandwidth accounting](topology_audit.md),
[terminal audit](terminal_audit.md), [validation plan](validation_plan.md), and
[commands and decisions](work_log.md) track the work and outstanding evidence.

Fresh all40 AIME checks on the frozen selected source pass both prefill and
teacher forcing: top-1 98%, top-5 100%, top-100 100% over100 predictions.
See [prefill](final/prefill_accuracy.json) and [teacher](final/teacher_accuracy.json).
Their diagnostic timings do not replace the clean performance rows above.

| AIME24 chat-template accuracy,100 predictions | Completed full model | Optimized full model |
|---|---:|---:|
| Prefill top-1 / top-5 / top-100 |98% /100% /100% |98% /100% /100% |
| Teacher forcing top-1 / top-5 / top-100 |98% /100% /100% |98% /100% /100% |

The prior accepted controls are [prefill](../full_model/prefill_accuracy_lifetime_fixed.json)
and [teacher forcing](../full_model/teacher_accuracy_lifetime_fixed.json).

## Complete model and generator decisions

| Boundary | Selected stage-7 path and evidence |
|---|---|
| Embedding and residual | Hidden-sharded BF16 embedding followed by the existing gather; the selected replicated BF16 residual remains in its L1 width-sharded carrier between B1 layers. No new inter-layer gather. See [topology audit](topology_audit.md). |
| Norms and sensitive linear projections | Preserve compensated/ordered arithmetic, FP32 products and state, BF16 consumption boundaries, and selected exact SiLU. Narrow coefficient reads and selective K512 coefficient caching remove movement without changing the arithmetic. |
| Recurrent attention | A guarded state-slab kernel combines decay, ordered contractions, delta update, state commit and terminal contraction. All four rounded/unrounded and persistent/temporary variants have 128-step Watcher exactness evidence; full-model accuracy and replay qualify the integrated path. |
| Full attention and MoE | Retain selected packed projections, paged SDPA, BF16 KV, BFP8/HiFi2 ordinary projections and BFP8/LoFi active experts. Indexed TP4 B1 and grouped EP4 batched routing remain selected. No datatype frontier or rejected-policy substitution. |
| Collectives | Shared persistent decoder workspace remains enabled. Redundant singleton padding/slicing is removed. The optional explicit-buffer sampler broadcast is rejected by a raw gather repro and native partial-packet diagnosis; see [rejection](sampler_persistent_broadcast_rejection.md). |
| LM head | Vocabulary-parallel BFP8 DRAM-sharded head, HiFi2 with FP32 accumulation; three readers/bank, block4, 16 input cores and per-core N32. Real-weight exactness, legal reader/block controls and full-stack L1-pressure controls are in [candidate search](candidate_search.csv). Legacy head storage remains for other row geometries. |
| Logits and sampling | Resident sampler-ready local-vocabulary logits feed the separate sampling trace, which writes `tt_out_tok` for the next model trace. Correct greedy split sampling wins over power-of-two padding and force-argmax; [final comparison](final/sampler_timing.json). |
| Cache and orchestration | Explicit slot/cache/page/position/prompt-length state remains intact. Position/RoPE advance on device; unchanged page tables cause no upload. Semantic preparation keys deduplicate equivalent compute policies. Resident token-out queues both traces nonblocking with zero per-token host conversions, refreshes or readbacks. |

[Runtime contract audit](runtime_contract_audit.md),
[optimization checklist](optimization_checklist.md), and the original
[selected-policy ledger](../optimized_multichip_decoder/optimization_decisions.md)
record the exact interfaces, guards and rejected alternatives. The canonical
[serving contract](../full_model/serving_contract.json) remains the full-model
handoff contract; no vLLM files or registration are added.

## Correctness and supported behavior

| Evidence | Final result and scope |
|---|---|
| AIME24 chat prefill / teacher forcing | Both top1 98%, top5 100%, top100 100% over100 predictions, unchanged from completed full-model acceptance. |
| AIME autoregressive | Exact161 saved prompt IDs and100 greedy tokens; paired coherent but divergent reasoning, truncated before a final answer. [Verdict](final/aime_autoregressive/verdict.json). |
| Shared qualitative suite | Six128-token matched cases; all fresh TT texts exactly match stage6 TT controls. Both HF/TT remain in reasoning at the cap. Degeneracy check passes; shared q03 phrase repetition is recorded in the [limitations ledger](../eval_readiness/limitations.json). |
| Full40 Watcher replay | 17 cases and32 stable-cache checkpoints: fresh/resumed tails, mixed-versus-solo, changed pages, inactive middle row, greedy and per-row top-k/top-p/seed isolation. [Artifact](final/replay_all40.json). |
| Fixed32 slots | Full40 capacity, reordered continuation and inactive-state smoke. Detailed state snapshots cover chip0; separate kernel controls cover all ranks. [Artifact](final/b32.json). |
| Serving commands | Nine full40 checkpoints, including all10 full-attention layers' changed30/unchanged0 page-table updates, dormant remap, asynchronous output leases and RNG state. [Passing rerun](final/serving_commands_all40_v3.json). |
| Trace lifetime | Tracking1/skip0 acceptance plus a deliberately unsafe allocation that the positive control rejects; clean timing uses tracking0/skip0 separately. |
| Short/tail public prefill | Context4096 lengths64,255,256,257,1025,2049 pass, followed by two resident model/sampling replays with no refresh/readback/recapture. [Artifact](final/prefill_boundaries.json). |

[Correctness acceptance](final/correctness_acceptance.json) reconciles the
completed original queue with the affected serving-test rerun. The original
failed assertions remain recorded: a reduced-model counter constant, followed
by a test-local name-shadowing mistake. Both are fixed in the passing rerun;
production code did not change between these final gates.

## Accounting, context and evidence

The retained-carrier, nonblocking representative layer controls give
`30 × 0.586715590 + 10 × 0.306684235 = 20.668310051 ms`.
Whole token-out is3.62% above that estimate before allowing terminal work,
below the requested10–15% gap threshold. These real-weight, fixed-input/position
controls are an engineering lower-bound estimate; natural routing and evolving
positions differ. Separate head/sampler controls are not an exact additive
decomposition. See [layer budget](final/layer_stack_budget.json) and
[performance accounting](perf_summary.json). The [native audit](final_profile_audit.md)
validates29,440 executions from real layers0/3 plus the complete terminal and
sampler. Rank2 model/sampling operation work is1.266784/0.436511ms per replay;
these reduced instrumented sums are not all40 wall latency. The clean sampler
control is2.28% of full40 queued time. Native rows confirm the selected head,
custom recurrence, precision and collectives; no force-argmax, generic vocabulary
TopK or full-vocabulary gather executes. Advice and untried geometry are
explicitly dispositioned in that audit. The [same-profile derivation](native_accounting_derivation.md)
reconstructs per-kind ownership and extrapolates20.631607ms of operation work
for40 layers plus terminal work. Its0.785261ms difference from clean wall time
is not measured host overhead. The separate reduced-run ideal/work/wall triplet
is0.350018/1.703295/2.015464ms.

The [context contract](../context_contract.json) retains262144 aggregate tokens.
KV dtype/layout is unchanged, so this stage uses a32769-token/context65536 smoke
instead of repeating maximum-context acceptance. The [smoke](final/prefill_context32769.json)
passes all40: two resident feedback steps reach position32771 with zero refresh,
readback or recapture. Prepared/final DRAM allocation stays21,024,343,552bytes
per device, with12,954,371,584bytes free. The [memory audit](context_memory_audit.md)
discloses the extra135,380,992bytes/device of resident head allocation and keeps
historical maximum acceptance separate from projected headroom. This repeated
AIME-token fixture is capacity/contract evidence, not long-context quality.
Capacity means the configured aggregate pool across fixed
slots, not32 simultaneous maximum-length requests. Existing resumed-prefill
HF-reproduced limitations remain in the original contract.

The [evidence manifest](evidence_manifest.json) links commands, gates, failures
and raw locations, and embeds exact paired qualitative text/metadata for local
checkpoint readability. Raw logs, large native captures and byte-sensitive
generated files remain in the persistent workspace; they are not normalized or
archived. [Final commands](final_gate_commands.md),
[serial measurement script](run_final_measurements.sh),
[AutoFix record](AUTOFIX.md), and [work log](work_log.md) give reproduction and
decision details. [Prefill analysis](prefill_performance_analysis.md) and
[before/after boundary timings](before_after_prefill.csv) show the unchanged
logical-tail support and bounded physical chunk contracts. Single-sample boundary
changes range−0.03% to+3.43%; no robust prefill speedup is claimed. Native custom
prefill program roles are only attributed to the resolution supported by metadata.
Stop Rule3 applies after the user’s stage-cap wrap-up; untried work is recorded
in [the work log](work_log.md). Independent review and local checkpoint are complete.

Local implementation/evidence checkpoint: `08977a6f21c68e383ead5f4675f980eb45cfbed8`.
The following documentation-only commit records this SHA and final status.
Nothing was pushed.
