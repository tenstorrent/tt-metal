# Optimized full model — stage 7 owner-accepted performance limitation

| Warmed batch 1, 128 input / 128 output | Before | Final measured path |
|---|---:|---:|
| TTFT | 7035.653 ms | 4705.951 ms |
| Resident token-out decode | 7.214029 tokens/s/user | 9.702619 tokens/s/user |
| Resident decode latency | 138.618793 ms | 103.064960 ms |
| Caller-read token-out decode | 7.204990 tokens/s/user | 9.692458 tokens/s/user |
| Traced teacher forcing, tracked caller boundary | 6.716107 tokens/s/user (stage 6) | 8.790475 tokens/s/user |

**Owner accepts the current decode speed as a known limitation for this run.**
The [2026-11-16 decision](owner_decision_2026_11_16.json) accepts 9.702619 tokens/s/user
against the original 15.589612 FUNCTIONAL floor and directs no further tuning
for this floor. The measured fraction remains 6.2238% versus 10%; the numerical
floor is not relabeled as passed. The [limitation registry](../eval_readiness/limitations.json)
records the decision and evidence. Resident throughput improved 34.50% and TTFT
fell 33.11%. [AutoFix](AUTOFIX_functional_floor.md) and the [round-1 review](stage_review_round1.md)
remain historical evidence. All required device gates finished; the independent [round-2 review](stage_review_round2.md)
returned **clean-pass** under the amended acceptance contract. Stage 7 is complete.

[Baseline](baseline_resident.json) and [final measurement](final_resident_v1.json)
use all 48 layers on the 1×4 Blackhole mesh, capacity 262144 and the same clean
128/128 harness. All 128 greedy tokens match. Resident timing includes model,
split sampling and device token/position feedback: 127 model and 127 sampler
replays, with no interior host reads, synchronization, input refresh, capture or
release. Public resident `decode_forward` separately measures
9.705529 tokens/s/user. Teacher forcing includes tracking and
caller sampling/readback and is a separate metric, not the resident benchmark.
The fixed-token logits-only row in `baseline_perf.json` is not teacher forcing.

## Selected implementation and correctness

The selected path adds exact causal scoring, fixed-policy LM-head geometry,
exact Q1 GR-up/GDN-output projection kernels, residual writes, local rank sums
and hyperconnection norm epilogues. It retains TP4 attention, EP4 whole experts,
four distinct rank-owned residual streams, inherited precision/fidelity and
BF16 KV/activation/CCL policy. Synchronous sampling gathers beat the tested
barrier-async alternative; persistent trace buffers and device feedback remain
selected. The rejected GR-down compaction is disabled and contributes no gain.
[Topology and dispatch audit](operation_topology_audit.md) and
[checklist](optimization_checklist.md) contain decisions and rejection evidence.

[Full-model validation](final_validation_v2.json) passes AIME24 chat-template
prefill at 97% top-1 and teacher forcing at 96% top-1; both reach 100% top-5 and
top-100. These equal the completed stage-6 [accuracy baseline](../full_model/acceptance_accuracy_resume_final_v1.json): prefill 97/100/100% and teacher forcing 96/100/100% for top-1/top-5/top-100. Resume, autoregression and the complete runtime fallback audit pass.
[Fresh blinded qualitative review](final_validation_v2/qualitative/verdict.json)
finds 0/60 failures on each backend under the shared rubric. Both HF and TT
have capped/incomplete answers; this is not an AIME solved-answer score.

[Batch-32 validation](final_batch32_hf_summary_v1.json) passes both reordered
2,704-prediction runs at 100% top-5/top-100, with exact matching logits and final
state across slot migration, inactive rows and resident sampled feedback.
These are staggered histories from one correlated AIME reference, not 32
independent prompts. The 32769-token non-aligned context smoke passes, including
an extra feedback replay without host refresh/readback/synchronization.
[Context contract](../context_contract.json) remains 262144; KV layout/dtype is
unchanged, so maximum context was not rerun.

[Measured source manifest](final_sources_v4.json) includes a rejection-only guard
for unsupported custom tiles added after numerical validation; standard-tile
math and kernels are unchanged. Its 25 CPU metadata cases and repository hooks
pass. The final resident benchmark records unchanged model, native, shared and
harness sources across setup and measurement. [Full48 mixed-prefill validation](final_prefill_signatures_v1.json) also passes after the guard: exact mixed/reversed logits, non-aligned resumes, retained trace IDs and no late compilation.

## Performance accounting

[Representative component traces](component_trace_v2.json) give
35×L0 + 1×L1 + 12×L3 = 102.507076 ms. Entry, terminal work and
split sampling add 1.576551 ms, totaling
104.083628 ms versus 103.064960 ms measured full-model token-out.
This shows no positive >10–15% orchestration gap. Frozen inputs in the reduced
model make this a representative stack estimate, not an exact mathematical
lower bound for every layer. Split sampling takes 0.481938 ms and is not dominant.
The strict 6.414528 ms bandwidth floor is kept separate from this estimate.

| Prompt length | Before TTFT (ms) | Final TTFT (ms) |
|---:|---:|---:|
| 31 | 2485.757 | 1845.539 |
| 32 | 1783.999 | 1202.206 |
| 33 | 1922.814 | 1304.506 |
| 128 | 7035.653 | 4705.951 |
| 1024 | 56733.396 | 38102.021 |

[Performance JSON](perf_summary.json) separates all timing boundaries and the
failed strict floor. Selected-source [decode profiling](terminal_profile_analysis_v2.md) reconciles 99.939984 ms of weighted device work with resident latency. [Six-bucket prefill profiling](prefill_profile_analysis_v1.md) records actual reduced-model kernels and advice; it does not extrapolate whole-model device TTFT. [Work log](work_log.md) records
commands, failures, source changes and exact artifact paths. No vLLM work or push was performed. Local checkpoint SHAs are recorded in the work log and `checkpoint_commits.log`.

[Batch-32 resident replay](batch32_resident_v1.json) passes at 0.390840 tokens/s/user (12.506878 aggregate), 2558.592232 ms/step over 127 steps. This all48/C256 regression row uses Q1 attention/head per slot and packed32 MoE; it is not a batch-32 latency improvement claim. All384 slot/layer positions and four-rank page tables agree, with stable traces/buffers/program cache and zero interior metadata uploads, host reads or synchronizations.

The local checkpoint includes a [formatting-only native-source delta](checkpoint_sources_v5.json) applied by the commit hook after measurement: line wrapping and braces around existing single statements. The full diff was inspected; math, operands, execution order and configuration are unchanged. Numerical/performance evidence retains its measured v4 hashes; no binary-equivalence or post-format hardware measurement is claimed.

Closure evidence: [CPU audit](closure_audit_2026_11_16.json), [exact formatter delta](checkpoint_clang_format_delta.patch), and [source verification](checkpoint_format_delta_verification.json). Untried optimization ideas remain in the work log as hand-off notes under the owner-confirmed stop rule.
