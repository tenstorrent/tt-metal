# Qwen3.6-27B optimized full model — stage7

**Warmed batch1, prompt128/output128, context allocation262144, all64 real layers,
Blackhole TP4:** same-harness TTFT **87.031→86.889ms**; readback-inclusive token-out
**30.976→30.957t/s/user**. The selected resident path delivers **31.000t/s/user**
(**32.258ms/token**) with TTFT **86.851ms**. These are small, near-noise changes;
the stage does not claim a material full-model speedup.

| Measurement boundary | Before | Selected default | Evidence |
| --- | ---: | ---: | --- |
| Whole-request warmed TTFT, same generate harness |87.031ms |86.889ms | `baseline.json`, `readback_final.json` |
| Split token-out, device feedback, every-token readback |30.976t/s/u |30.957t/s/u | same files; three repeats |
| Split token-out, resident feedback, no per-token read/sync | old API ignored no-read flag |31.000t/s/u | `performance_final.json`; three127-step windows |
| Resident harness TTFT, first token read included | — |86.851ms | `performance_final.json` |
| Logits-only trace, no sampler/readback; timing-only fixed token |31.507steps/s |31.504steps/s | `baseline.json`, `readback_final.json` |
| AIME24 traced teacher forcing, reference updates + token readback |31.05t/s/u (stage6) |31.04t/s/u | `teacher_forcing.log` |

TTFT includes reset, request configuration, all prompt chunks, final norm/head,
sampling and first-token readback. Preparation/compilation/capture is excluded.
Resident decode submits nonblocking model and sample traces, feeds `tt_out_tok`
directly back, and synchronizes once after the measured window. Diagnostic
endpoint reads are outside those windows. Logits-only is neither autoregression
nor teacher forcing. Readback-inclusive before/after ranges overlap
(30.956–30.977 versus30.954–30.965t/s/u); its median difference is−0.063%.
All numbers above are uninstrumented, except the explicitly labeled profiler
analysis below. This stage has no vLLM integration or serving benchmark.

## Implementation and preserved contract

`tt/generator.py` now honors `read_from_device=False` by returning borrowed
persistent token/logit device storage. It adds one prepared B1/P128 first-chunk
prefill trace and skips identical padded temperature/k/p parameter uploads.
Seeds still reset as requested; slot remapping invalidates the parameter cache.
`tt/prefill_trace.py` binds existing input/state/logit buffers and introduces no
allocation exemptions. General lengths, resumed prefill and multi-row requests
retain the prepared bounded path. The borrowed tensor must be consumed before
another replay overwrites it.

The full TP4 path remains: hidden-sharded embedding → hidden-sharded64-layer
stack → final norm/hidden gather → vocab-sharded BF16/HiFi4 head → split candidate
sampling → persistent token feedback. Each layer carries1280 residual elements
per rank into its next consumer. Selected BFP8 weights/KV, BF16 residual/CCL
(with retained FP32 GDN intermediates), GDN attentionHiFi4, GDN MLPHiFi2 and full-attentionLoFi are unchanged. The existing
precision rejection ledger, persistent collectives, inter-layer layout and
page32 cache contract are preserved. No broad datatype frontier was run.

See [mesh_plan.md](mesh_plan.md), [runtime_audit.md](runtime_audit.md),
[sampling_decision.md](sampling_decision.md) and [checklist.md](checklist.md).
The executed-op dtype/fidelity/program evidence is
[perf/runtime_matmul_policy.csv](perf/runtime_matmul_policy.csv).

## Full-path performance accounting

The optimized decoder-layer wall bound is
48×0.509915878 +16×0.411218298 = **31.055455ms/token**. It uses prefix512 layer
probes, so it is a reference budget, not an exact subtraction from P128 all64.
The new reduced real-weight profile extrapolates29.064–29.376ms layer kernel
work plus2.113–2.119ms terminal/IO/sampling, totaling **31.183–31.490ms** per rank.
The resident all64 wall time is32.258ms, only0.768ms above the conservative
worst-rank kernel projection and3.87% above the layer-only wall reference.
There is no>10–15% excess over stack plus terminal requiring another gap repair.

The vocabulary head costs1.55–1.57ms, sampler0.453–0.473ms of kernel work. Fresh
serial sampler tests retain split greedy **0.512781ms** over force-argmax
**2.725168ms**. The trace contains specialized large-index TopK and two small
candidate gathers, no generic TopK or full-vocabulary gather. Existing same-policy
adapted DRAM-head candidates lose whole-terminal latency; their exact evidence
remains in `../full_model/perf_terminal/lm_head_decision.json` and
`../full_model/lm_head_uninstrumented.json`. Equal-value TopK ties may select a
maximum other than PyTorch's lowest index; this unchanged contract is explicit
in the sampler decision.

The compulsory-traffic roofline estimate is **15.515526ms/token**,
**64.4516 theoretical t/s/u**, using per-chip512GB/s and all64 active matrices,
recurrent/cache traffic and selected embedding rows. Its10% FUNCTIONAL floor is
6.4452t/s/u, well below31.000. This is a theoretical estimate, not measured
bandwidth. `perf_summary.json` collects the final regime and caveats.

[perf/analysis.md](perf/analysis.md) reconciles actual per-rank kernel sums,
modeled traffic and host timing. The instrumented probe uses one real layer of
each kind, not all64; never sum the four TP ranks into latency. Profiler-drain
op-to-op gaps are excluded. Its source predates only request-boundary sampling
parameter caching; decode/prefill/sample device graphs and program policies are
identical to the selected runtime. The later one-line absolute-import repair
only enables the readiness standalone loader; `import_resolution_proof.json`
proves package-mode helper identity and the exact source delta. Completed
package-mode measurements retain their original hashes; readiness gates rerun
on the repaired source. `profile.command.json` and
`perf/command_receipt.json` retain the exact profiling/report commands.

## Prefill lengths and retained optimization decisions

Same-process paired medians from `performance_final.json`, three repeats. Maximum
physical chunk512; logical lengths are not rounded at the API. Only exactP128
selects the additional trace, so other paired differences measure the same eager
path and are not attributed to an optimization.

| Logical tokens | Chunks | Helper disabled TTFT ms | Selected TTFT ms |
| ---: | ---: | ---: | ---: |
|33 |1 |89.013 |88.622 |
|127 |1 |89.637 |89.850 |
|128 |1 |86.856 |86.650 |
|129 |1 |92.165 |92.093 |
|511 |1 |137.727 |137.949 |
|512 |1 |132.691 |131.404 |
|513 |2 |182.170 |186.466 |
|1025 |3 |302.020 |302.029 |
|32769 |65 |11597.753 |11602.076 |

P128 trace improves its paired median0.2365%; parameter caching changes86.730679
→86.746810ms (no meaningful gain) while eliminating redundant parameter uploads.
These last two retained changes meet optimize Stop Rule2 (<3% each). Optional
tuning stops; no significant-gain or universal-optimum claim follows. Untried
families are listed under `Not tried (stop rule)` in [work_log.md](work_log.md).
The per-size/role kernel table distinguishes current measured128, historical512
and unmeasured alternatives in
[perf/prefill_kernel_selection.md](perf/prefill_kernel_selection.md).

## Acceptance evidence

AIME24 chat-template numerical checks pass over100 reference continuation
positions: prefill top1/top5/top100=100/100 each; teacher forcing=99/100,100/100,
100/100, unchanged from stage6. These are token-agreement checks, not an AIME
problem-solving score. Teacher forcing cold TTFT93.504s includes preparation
and is not the warmed TTFT headline.

The all64 B32 regression passes complete preparation, identical-prompt row
logits,128 resident steps with31 active slots, inactive state, remap and release.
It records16.237918t/s/active user at context1024 (`slots32.json`), a synthetic
fixed-slot regression rather than a serving/natural-answer benchmark.

All20 bounded acceptance jobs pass; `hard_gates.json` and `collect_evidence.log`
verify exits, source lineage and numerical thresholds. Independent stage review returned [clean-pass](stage_review_round1.md), round 1.
The matched HF wording observation is registered in
[limitations.json](../eval_readiness/limitations.json). Local checkpoint SHAs are
recorded in [commit_receipt.json](commit_receipt.json).

| Additional gate | Result / exact artifact |
| --- | --- |
| Full-stack tracking and reduced complete-shape battery | Stable traces/programs/inputs, no exemptions; `full_trace_tracking.json`, `trace_contract.json`, positive-control log |
| Watcher and no-host-boundary sentinels | Pass; `watcher.json`, `runtime_tracking.json`, matching logs |
| All64 mixed33/128/513 and preemption rebuild | Pass, preserved live slots; `mixed_all64.json` |
| Resumed natural2560 prefill | Starts32/68/2528/2559 pass; both eligible negative controls fail; `prefill_resume.json` |
| Explicit host compatibility then return to device | Pass under tracking; `host_compatibility.json` |
| Free-running readiness |128 tokens exactly equal HF and stage6; `autoregressive_verdict.json`, `autoregressive/` |
| Shared greedy/sampled suite | All12 TT histories equal stage6; matched HF metadata and actual texts reviewed; `qualitative_verdict.json`, `qualitative_tt.json` |
| Installed stage check | No degeneracy; `stage_gate.log` and command receipt |
| Device health and release | Pass, zero holders, no reset/recovery; `device_health_final.json` |

Shared-suite128-token windows mostly remain in default thinking; they do not
establish final-answer task success or statistical quality equivalence. The
matched story prompt's initial doubled word occurs in HF as well. Resume
case-mean PCC is at least0.999431966; the worst individual0.985754633 passes the
unchanged runner floor0.98 and exactly reproduces stage6. These observations and
the repaired import failure are classified in [anomalies.md](anomalies.md).

The context contract remains262144. KV dtype/layout is unchanged, so the prior
maximum-context acceptance is retained and this stage refreshes a nonaligned
32769-token smoke at full262144 allocation. Four trace IDs,24456 programs and
persistent input addresses stay stable. The final prepared snapshot records
19,071,786,496allocated and15,006,936,576free bytes per device; it is not an allocator
high-water mark. Exact lineage is in `../context_contract.json`.

## Reproduction and artifacts

From the checkout root, activate `python_env` and source
`/home/ttuser/qwen36-27b/lane-env.sh`. `run_acceptance.py` serializes bounded jobs;
each `<name>.command.json` records the exact command, source hashes, environment,
start/end and exit. `baseline.command.json` explicitly labels its retrospective
command receipt. The original source was commit
`7dde27883c7cf72ed1268a9ef5845b19134549c3`; final checkpoint SHAs are in the work log.
No device reset/recovery or external push was performed.

Compact report tables/CSVs and provenance are retained in `perf/`. Original raw
Tracy/device data stays local under
`tracy/reports/2026_11_14_06_17_56/ops_perf_results_2026_11_14_06_17_56.csv` and
`tracy/.logs/`; it is not embedded in the source commit. The eight report commands
and analysis scripts regenerate the compact accounting from this exact capture.
