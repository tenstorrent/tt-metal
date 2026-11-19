# Qwen3.6-35B-A3B full model

**Full-model warmed TTFT: 2919.990325 ms. Trace-verified batch-1 caller-observed
decode: 39.795818834 t/s/u (25.128267976 ms/token). Queued device-token production:
39.869407085 t/s/u (25.081887921 ms/token). True HF logits-only teacher forcing:
40.649627178 t/s/u (24.600471626 ms/step).**

[Final isolated performance](performance/lifetime_fixed.json) passes on the
current all-40-layer source, with clean closure and an idle-device check afterward.
The run reports 435.988 s (approximately 440 s parent-observed wall time). The
headline workload uses 128 input tokens and 128 generated tokens: one prefill
prediction followed by 127 decode steps, next position 128 → 255. Caller-observed
timing includes 127 token deliveries; queued timing retains intermediate outputs
on device and reads none inside timing. Both use 127 model and 127 sampling trace
replays, zero token/position/page/activity refreshes and zero recaptures. Both
finish with token 318; 8452 program-cache entries stay unchanged. Token delivery
never supplies feedback to the next replay; there is no host argmax or full-logit
readback in either token-out path.

| Measurement | Recorded value | Boundary |
|---|---:|---|
| Warm TTFT | 2919.990325 ms | Public prefill through first token delivery; loading, prepare, warmup, reset and tokenization excluded |
| Caller-observed decode | 39.795818834 t/s/u; 25.128267976 ms/token | Token delivery, Python loop and audit overhead included; resident device feedback |
| Queued device production | 39.869407085 t/s/u; 25.081887921 ms/token | Boundary synchronization; intermediate tokens not delivered |
| True HF logits-only teacher forcing | 40.649627178 t/s/u; 24.600471626 ms/step | 99 actual forced HF inputs; input uploads included, sampling/readback excluded |
| Model-only component | 24.600408906 ms/replay | Fixed input token with advancing positions; **not teacher forcing** |
| Sampler-only component | 0.487787433 ms/replay | Common sampling trace over fixed real decode logits |
| Packaged teacher diagnostic | 39.61108 t/s/u; 4246.728 ms TTFT | Separate final accuracy run while CPU-only profiler conversion was active; not isolated headline performance |

True HF teacher forcing uses the full 161-token AIME prompt, then 99 reference
inputs through embedding, all 40 layers, terminal norm/head and resident logits.
It records 99 model replays and forced input uploads, zero sampling/readbacks or
other refreshes/captures, positions 161 → 260 and final persistent input 393,
matching reference index 98. This timing is separate from both fixed-token replay
and the packaged accuracy check in `teacher_accuracy_lifetime_fixed.json`.
Final-source prefill and teacher accuracy each pass 98% top-1 and 100% top-5/top-100
over 100 predictions. Earlier `performance/result.json` remains historical.

The all 40 fixed-logit sampler control is approximately 1.9448% of queued token-out
time. Separate controls are not an additive latency decomposition. The unprofiled
run has allocation tracking 1, program-cache checking enabled, Watcher/profiler
unset and both traces released; see [driver log](logs/performance_lifetime_fixed.log).

The [v4 native component profile](performance/profile_reduced_v4_drains/component_summary.json)
is validated after completed conversion and six tt-perf-report exports
(approximately 3040 s total; capture/workload 916.515 s). It covers real layers 0/3,
full embedding/norm/LM head and the greedy sampler at batch 1/context 1024, with
32,896 native execution UIDs, 16 matching sessions per component/rank on four
ranks and 748 validated device/program metadata entries. Native operands and
complete kernel/fidelity inventories are in `native_program_metadata.json`;
`ops.csv`, component operation-group CSVs and perf reports remain under the same
directory. No marker/source-overflow loss was accepted. Failed v1/v2 OOM/source
overflow and v3 marker-loss captures remain preserved.

On the rank with maximum combined kernel work, the reduced model/sampler sums
are 1.471036125/0.4548093125 ms: sampler 23.6161%, not dominant. This is reduced-stack
operation-work attribution, **not all 40 share or wall-time decomposition**.
Four parallel ranks are never summed into latency. It must not replace the
separate all 40 unprofiled headline.

**Stage status: complete under the owner-revised contract; [round2 review](stage_review_round2.md): clean-pass.**
The owner accepts the three HF-reproduced resumed-prefill PCC failures as a known
limitation. All seven TT top-1 criteria and ten applicable negative controls pass;
raw results and thresholds are unchanged. See [owner decision](owner_decision.md),
[acceptance evidence](owner_acceptance.json), and the limitation record. The four
existing continuation/chunk paths are accepted with this scope and their capability
declaration is enabled. No device code, numerical policy, maximum-context setting,
or trace graph changed. No further work on this gate or vLLM implementation is requested.

## Implementation and preserved policy

The HF autoregressive path comprises embedding, all 40 selected TP4/EP4 decoder
layers, terminal RMSNorm with HF's `1 + weight` offset, and untied vocabulary
head. It runs on the required 1x4 Blackhole mesh. Inter-layer residuals remain
replicated BF16; linear recurrent state is FP32; full-attention paged KV is
BF16. Selected weight/fidelity, activation and collective policies and the
optimized decoder rejection ledger remain the baseline. The full stack shares
immutable lookup/RoPE storage and collective workspace; unreachable inherited
single-chip routing buffers are released at construction.

Configured batch 1 retains its optimized path. Configured larger batches use a
fixed slot bucket and preserve inactive convolution/recurrent/token/position/RNG
state. Batched grouped-MoE rounding is explicitly deterministic to eliminate
cross-request dependence from a launch-global stochastic stream; accepted
weight dtype, BF16 arithmetic and LoFi expert fidelity remain unchanged. The
batch-3 accuracy result below validates this repair. See `AUTODEBUG_MIXED.md`,
`AUTOFIX.md` and `batched_teacher/result.json`.

The public generator owns nonaligned prefill, padding, masking, cache writes,
absolute positions and logical output slicing. Low-level serving interfaces
expose explicit cache/page tables, positions, prompt lengths, fixed slots and
request state. `serving_contract.json` is the pinned plugin interface inventory;
all 86 rows are resolved, with numerical continuation limits recorded explicitly.
Automatic KV-only prefix caching remains excluded for recurrent state. Retained-state
continuation is supported with the owner-accepted PCC sensitivity described below.

The selected common `SamplingGenerator` uses split greedy with tiled local
candidates and generator-owned model/sampling traces. The separate Sampling1D
implementation was rejected for the request-state integration cost. The same
real-terminal probe measured selected split greedy at 0.4883 ms/replay, padded
split greedy at 0.5218 ms and full-vocabulary force-argmax at 2.6647 ms. These are
sampler component measurements, not full-model throughput. The common graph's
true-maximum and tested tie behavior pass; see `sampler_contract_probe.json`
and `sampler_comparison.md`. Explicit `host_sampling=True` remains a test
compatibility mode and is excluded from the measured device token path.
`host_sampling_probe.json` records its actual reduced standalone-discovery PASS:
three model replays, zero sampling replays, and explicit host token/logit boundary
activity. This verifies compatibility mode separately; it is not an automatic
fallback or a performance result.

## Accuracy and qualitative evidence

The immutable HF revision is `345d467b117d1fd5eda9b879a5988de9ae5f47a7`.
`readiness_aime24_chat.refpt` contains the HF-tokenizer chat-template AIME24
prompt (161 tokens), 100 generated tokens and top-100 predictions. Generation
used `--model-dir`; its exact metadata/cache match is recorded in
`hf_reference/reference_summary.json`. HF cached generation itself agrees
with the separate full-forward reference at 99% top-1 and 100% top-5/top-100.

| Check | Samples | Top-1 | Top-5 | Top-100 | Artifact |
|---|---:|---:|---:|---:|---|
| Final-source all 40 layers, prefill | 100 | 98% | 100% | 100% | `prefill_accuracy_lifetime_fixed.json` and adjacent provenance; `logs/prefill_accuracy_lifetime_fixed.log` |
| Final-source all 40 layers, B1 teacher forcing | 100 | 98% | 100% | 100% | `teacher_accuracy_lifetime_fixed.json` and adjacent provenance; `logs/teacher_accuracy_lifetime_fixed.log` |
| All 40 layers, prefill before lifetime change | 100 | 98% | 100% | 100% | `prefill_accuracy_final.json` and provenance; 381 s run |
| All 40 layers, B1 teacher forcing | 100 | 98% | 100% | 100% | `teacher_accuracy.json` and provenance |
| All 40 layers, B3 combined, each slot | 100 | 98% | 100% | 100% | `batched_teacher/result.json` |
| B3 first prefill prediction, each slot | 1 | 0% | 100% | 100% | Same artifact |
| B3 decode-only, each slot | 99 | 98/99 | 100% | 100% | Same artifact |

The first two rows are the completed final-source refreshes. Earlier prefill/B1
and B3 results remain historical evidence from before the lifetime change.
The acceptance bars are top-5 >= 98% and top-100 = 100%. B3 uses disjoint physical
pages 0–8, 9–17, 18–26 and reaches position 260 in each slot. It performs 99 model
trace replays with unchanged 8491 program-cache entries and no recapture/page
refresh. Its explicit HF input uploads and logits readbacks are diagnostic
teacher forcing; they do not measure token-out sampling or throughput.

All six canonical prompts have paired HF/TT 128-token completions in
`qualitative/tt_greedy/q*/`. The recorded review finds relevant reasoning,
no mechanical loop or wrong-language drift, and no incoherent early divergence.
First mismatch indices are 64, 85, 53, 107, 19, 74; q05 first differs in formatting.
Both HF and TT repeat the isolated q03 opening phrase and use English reasoning
before the French translation task. Both consume the cap on reasoning; this
smoke does not establish final-answer correctness or distributional quality.
See `qualitative/tt_qualitative_verdict.json`, `paired_review.md` under
`qualitative/tt_greedy/`, and `qualitative/degenerate_output_check.json`.
No quality failure was selected for debugging. Any later failure requires the
locked rubric and 10 matched-seed blind HF/TT controls before implementation work.

## Serving, capacity and acceptance

| Gate | Result/artifact |
|---|---|
| Final-source prefill and B1 teacher accuracy | Both PASS: `prefill_accuracy_lifetime_fixed.json`, `teacher_accuracy_lifetime_fixed.json`, adjacent provenance and matching logs; each 100 predictions, top1 .98/top5 1.0/top100 1.0 |
| Configured batch 32 | PASS `full_batch32_trace_budget.json`; 512 MiB trace reservation, 8551 program entries with zero growth; 31 inactive slots checked on chip 0, computation on all 4 chips |
| Retained-state resumed prefill | **Owner-accepted known limitation**: `owner_acceptance.json`; raw `../../readiness/prefill_resume_check.failed.json` remains FAIL. HF reproduces the three PCC failures; all TT top-1 and negative controls pass |
| Public nonaligned/mixed/reordered traces | Latest-source reduced nonaligned/mixed/page/seed replay PASS: `generator_replay_probe/lifetime_fixed_watcher.json`; separate reordered serving evidence remains `serving_commands_probe/result.json` |
| External-cache ownership/reset/release | Reduced evidence: `external_cache_probe/result.json`; historical source/coverage limits reconciled in the runtime audit |
| Caller-observed TTFT/decode | Final-source PASS: `performance/lifetime_fixed.json`, `measurements.token_out_observed`; 128 input / 128 output, 39.795818834 t/s/u |
| Queued decode, HF teacher forcing and component controls | Final-source PASS in the same artifact; true HF logits-only teacher forcing is separate from fixed-token model-only and fixed-logit sampler controls |
| Explicit host-sampling compatibility | PASS `host_sampling_probe.json`; reduced standalone discovery only |
| Native component profile | V4 validated PASS: `performance/profile_reduced_v4_drains/component_summary.json`; 32,896 native executions, 16 sessions/rank/component, 748 metadata entries. Failed v1/v2/v3 retained |
| Full context 262144 | **PASS** [context_acceptance/final.json](context_acceptance/final.json): one 262143-token prefill plus one traced decode at 262143 → 262144; clean closure, no capability reduction. Original `context_acceptance/result.json` failed during warmup |
| Full DRAM arithmetic | Prepared/after-prefill/after-decode census agrees: 25372641280 allocated and 8606073856 free bytes/chip. Partial tensor inventory in `context_acceptance/final.json`; completed census/capacity reconciliation in `../context_contract.json` |
| Reduced full-model generator Watcher | **PASS** [watcher/summary.json](watcher/summary.json), [replay result](generator_replay_probe/lifetime_fixed_watcher.json): real layers 0/3, context 1024, capacity 3; 90 clean dumps, all 4 chips attached/observed/detached. Not all 40 or maximum-context Watcher evidence |
| Runtime fallback audit | Source/evidence audit in `runtime_fallback_audit.md/.json`; final runtime counters and guards are recorded separately in `performance/lifetime_fixed.json` |
| Pinned serving inventory | 56 supported / 30 documented not_supported / 0 missing. Four continuation/chunk rows accepted with explicit numerical limitation; `supports_chunked_prefill=True`, prefix caching remains False under recurrent_state |
| Stage06 gate | Historical exit2 preserved in `logs/stage06_final_gate.log`. CPU-only owner-acceptance stage06 check PASS exit0: `logs/owner_acceptance_stage06.log`; no numerical/device rerun |
| Independent review | Round2 `stage_review_round2.md`: clean-pass, no Required Work; prior round1 retained |
| Local commits | Shared MoE: `69a03c857a5366dd575dc9a7a655b7248c15fad8`; shared profiler: `958dbadc5c4eb4b6b6ce73f792a0b862ba8f9e35`. Incomplete stage checkpoint: `b0cbe4ab9cd932bd1d2bb43d030c21cba9799d25`; no push |

The four formerly blocked rows are `prefill_logits_continuation`,
`supports_chunked_prefill`, `chunked_prefill` and `resumed_prefill`. Their paths
already executed in the recorded evidence. Owner acceptance closes the PCC
condition without changing the raw failures or claiming a numerical repair.
The original gate log and schema-reconciliation record remain historical evidence.


The reduced Watcher replay reports 894.928 s (approximately 899 s parent-observed
wall time), with `TT_METAL_WATCHER=10`, allocation tracking 1,
`TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0` and device profiler unset. Its eight
recorded source hashes match the inspected files at this documentation update.
Nonaligned lengths and continuations, solo-versus-mixed requests, changed and
unchanged page tables, an inactive middle slot and per-slot seed/parameter
isolation pass. Program-cache entries remain 8489 with no growth; model and
sampling traces each replay 155 times and both release at cleanup. Raw evidence:
[Watcher log](watcher/replay/generated/watcher/watcher.log) and
[driver log](logs/generator_replay_watcher.log). This is reduced state/replay
and Watcher coverage, not all 40 HF accuracy, performance or resumed-prefill PCC
acceptance.

The raw resumed-prefill gate fails PCC at 257/start 128, 1025/start 260 and
1025/start 1024. Before diagnosis, the pinned original HF model was run on the
same natural book, token IDs, selected logit positions and actual prefix cache.
HF reproduces all three case-level failures under unchanged thresholds; all HF
and TT cases still meet top-1 >= .98 and final top-1 equality. HF per-position metrics
are preserved, but exact TT/HF worst-row overlap is unknown because the original
TT artifact retains only aggregates. The blind numeric rubric and deterministic
seed-label reuse are explicit; this is not a sampled-text statistical waiver.
See `qualitative/hf_resume_control/verdict.json`, its README and the
`original_model:same` limitation in `../eval_readiness/limitations.json`.
AutoFix found no verified policy-preserving production repair; see
`AUTODEBUG_RESUME.md` and `AUTOFIX_RESUME.md`. The raw resume result remains
failed, with thresholds unchanged. The explicit2026-11-16 owner decision accepts
these PCC failures for stage6; no further work on this gate is requested.

The public context remains 262144. The all-40-layer maximum-context run passes
in [context_acceptance/final.json](context_acceptance/final.json), exits 0 and
releases both traces cleanly. It makes exactly one long prefill call with 262143
tokens and one traced decode at 262143; the next position 262144 is a sentinel,
never executed. All 10 full-attention layers on all 4 chips have finite/nonzero
last-page rows 0–30 before decode, zero row 31 before decode and written row 31
after decode, with preceding rows unchanged. All 240 linear-state/chip checks
are finite. The decode uses device token feedback with zero token/position/page/
activity refreshes, zero readbacks and zero captures inside the measured replay.

The prefill takes 5883.448 s and the boundary decode 33.730 ms. These are repeated-token
capacity diagnostics, **not headline TTFT/decode performance or qualitative
accuracy evidence**. The prepared and final allocator census is unchanged:
25372641280 allocated and 8606073856 free bytes per chip (about 8.606 GB free),
including the retained model, KV/request state, sampler, prefill workspaces and
live traces. Actual tensor inventory and separate trace-reservation accounting
are recorded in the artifact; they must not be added again to the census.
There is no capacity reduction or additional context sweep. The original
full-model warmup failure remains preserved in `context_acceptance/result.json`.

The all-40-layer batch-32 acceptance completed in 424 s. Its short fixed-slot workload
checks active output, reversed row ordering and inactive recurrent/conv/token/
seed/position preservation. It is capacity/state evidence, not B32 HF accuracy
or long-request throughput. The initial 200 MB trace reservation failure remains
recorded; increasing the mesh-owner reservation to 512 MiB resolved capture
without reducing batch or context capability.

The prior isolated-decoder reference budget is 24.99350229 ms/token
(30 × 0.719504474 + 10 × 0.340836807). It excludes terminal work, sampler and orchestration
and uses a different workload; it is a comparison estimate, not an additive
measurement of this full model. Reduced component profiling preserves the real
terminal and sampler contracts, but its sampler share cannot stand in for the
all-40-layer share. If full-model token-out sampling dominates, repair the terminal
contract before completing the stage.

## Recorded commands and artifacts

These commands document completed runs, not further authorized experiments.
The final isolated performance invocation, after activating `python_env` and
sourcing `/home/ttuser/qwen36-35b-a3b/lane-env.sh`, was:

```bash
"$TT_MODEL_BRINGUP_ROOT/scripts/wait_for" --cmd 'env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 timeout 3600 python -u -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_full_model_performance --output models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/performance/lifetime_fixed.json' --file models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/logs/performance_lifetime_fixed.log --timeout 3650 --tail 8
```

The final accuracy provenance records these Python argv, each with allocation
tracking 1 and program-cache skipping 0; shell timeout wrappers are not inferred:

```bash
python /home/ttuser/qwen36-35b-a3b/tt-metal/models/autoports/qwen_qwen3_6_35b_a3b/tests/run_full_model_readiness.py --mode prefill --output models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/prefill_accuracy_lifetime_fixed.json
python /home/ttuser/qwen36-35b-a3b/tt-metal/models/autoports/qwen_qwen3_6_35b_a3b/tests/run_full_model_readiness.py --mode teacher --output models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/teacher_accuracy_lifetime_fixed.json
```

The exact native capture, conversion and six report-export commands are in
[v4 commands.log](performance/profile_reduced_v4_drains/commands.log); capture
environment and source pins are in the adjacent `performance.json`. The validated
summary links native raw trace/device reports, `ops.csv`, operand/kernel metadata
and their hashes. Earlier failed profiler attempts are retained separately.

The max-context command and one-run policy are in `context_acceptance_plan.md`;
profile commands and timing semantics are in `profile_components_commands.md`.
`acceptance_checklist.json` records the exact installed stage06 gate and broader
user requirements. The original gate exited2 under the earlier contract. The
owner-acceptance follow-up performs only CPU manifest/check/review work; measured
runtime evidence remains unchanged. Exact commands and local SHAs are in the work log.

## Checkpoint evidence storage

Reports that exceed the repository file-size limit or require exact native whitespace are committed as lossless gzip in [checkpoint_evidence](checkpoint_evidence/README.md). The manifest records original paths and SHA256 hashes; its restore recipe recreates the unchanged files, including generated HF/TT completions. Original files are already present in this workspace. Large native Tracy/device captures and expanded ops CSV remain local at the recorded paths.

## Final review disposition

The prior [round1 review](stage_review.md) found one combined blocker: the
resumed-prefill acceptance and four missing serving rows. The owner has now
accepted that HF-reproduced PCC limitation explicitly. The independent diff-only [round2 review](stage_review_round2.md) returns
**clean-pass** with no Required Work and confirms the original finding is closed
under the revised contract. Earlier raw failures and AutoFix conclusions remain intact.
Nonblocking notes remain under `Carried to next stage` in the work log.

Original local checkpoint: `b0cbe4ab9cd932bd1d2bb43d030c21cba9799d25`; SHA-log
commit `35169ad54c7acbf0016ab155c5a4c08f9b1ec016`. Owner-acceptance checkpoint
`63c0b7d46b1d03eb49023d216f312f885b72eb16`; commit hooks PASS. The following documentation-only SHA-log commit records it. No push.
