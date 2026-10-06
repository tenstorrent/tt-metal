# Optimized vLLM work log

Stage 10; model google/gemma-4-26B-A4B-it. Starting tt-metal commit
58b7d77f12; vLLM commit 7f72b1c6e905f5137fe3377f2e7b42738d3f271d.
Preserve operator-owned PIPELINE_INTERVENTIONS.md and embedded vllm checkout.

Primary workload: 4096 input / 128 output, B1 C1, greedy on device;
secondary CI: 100 input / 100 output, 32-request burst.
Server: P300x2, TP4/DP1, max_num_seqs32, max_model_len262144,
trace_region_size1000000000, FABRIC_1D, async scheduling, decode_only tracing.
Exact launch: server_command.json. Datatype: head4_inner_all4_shared_down4.
No context reduction. No serving profiler, Tracy or ReadDeviceProfiler.

## Baseline

Preserved preceding final runner JSON in before/. These are unchanged-source
stage09 measurements, not newly measured stage10 results. Full sampling72pass,
1 documented all-vocab logprob skip; existing qualitative and contract evidence
remains under ../vllm_integration and ../../readiness_vllm.
A fresh baseline server launched using the same command; server_runner.log.

## Topology audit

| Boundary | Current path | Candidate/action |
| --- | --- | --- |
| Decoder | Selected packed QKV, paged SDPA, routed indexed expert TP4, persistent CCL buffers | Preserve selected datatype and prior optimized layer contracts; no new kernel tuning justified by serving gap |
| Terminal | Vocab-sharded LM head, canonical tiled local TopK32/candidate gather, greedy split sampling, persistent token feedback | Preserve full-model terminal path; force-argmax disabled |
| Replay | Model trace then sampler trace, both nonblocking | Preserve async split and deferred token-only read |
| Output | Eager device slice into persistent public token tensor | Investigate eliminating dispatch while preserving stable async output shape/identity |
| Scheduler inputs | Persistent token/position state; equal page tables skipped | Investigate repeated equality checks only with changed-page correctness retained |

Full-model selected autoregressive control is 51.6473t/s/u at4096/128/B1/C1,
vs inherited serving50.7283t/s/u. Different prompt content/harness; indicative
orchestration comparison, not an identical-workload speedup claim.

## Pending

Candidate discrimination, fresh before/after benchmarks, final checks,
independent clean-pass review, and local commits remain required.

## Public-token formatting candidate

AutoDebug found one eager public-token slice per decode after the two split
trace replays. AutoFix moved that copy into a model-local wrapper around the
canonical sampler's captured sampling method. Persistent public output shape
and tensor identity are preserved; host sampling retains the eager copy. Direct
tensor bindings avoid a generator/sampler reference cycle. Page-table refresh,
sampling math, precision, and plugin code remain unchanged.

The focused pre-change experiment reproduced the extra device-sampling eager
copy; post-change full host sampling contracts pass **84/84**. See
`AUTODEBUG.md`, `AUTOFIX.md`, and `host_sampling_contract.log` for commands and
limits. Black and whitespace checks pass; isort is unavailable. Hardware async
replay and before/after serving measurements remain parent-owned pending checks.

## Candidate contract

AutoFix source diagnosis investigates sampler-trace token formatting. Keep the
same canonical SamplingGenerator algorithms, greedy localTopK32 representation,
request seeds/penalties, persistent output tensor and async event boundary.
The output slice is currently eager after sampler replay. Candidate captures
that same copy inside the sampler trace, removing one per-token Python/TTNN
op dispatch without a new replay or changed sampling semantics.
Only full-model token-out/serving boundary work is in scope here. Existing
selected matmul/dtype/topology evidence remains authoritative; serving profiling
is intentionally disabled. No kernel performance improvement is claimed.

## Before and focused validation

`python doc/optimized_vllm/run_baseline.py` (full model path prefix implied)
ran the exact shared benchmark command twice on the unchanged loaded server;
first_use/ retains the first run, before/ retains the warmed repeat. Each ran
both primary4096/128/B1/C1 and CI100/100/32. Server PID63645, API63675,
EngineCore63714 exited after runner SIGINT; ps verified no remaining owned PIDs.

Candidate host sampling-contract suite:84passed in36.33s (host_sampling_contract.log).
Reduced adapter command: `OMP_NUM_THREADS=8 TT_METAL_TRACE_ALLOC_TRACKING=1
TT_METAL_TRACE_ALLOC_TRACEBACKS=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0
TT_METAL_LOGS_PATH=<stage>/runtime_logs python -m
models.autoports.google_gemma_4_26b_a4b_it.tests.check_vllm_adapter
--output <stage>/adapter_changed_pages.json` exited0. The initial attempt
without TT_METAL_LOGS_PATH failed creating the image-owned logs directory;
retry changed only the log path. No reset needed, mesh closed normally.
Changed/unchanged pages, stale token/position inputs, minimal async reads,
external-cache identity and standalone token equality pass with allocation tracking.

Watcher reduced multirow check initially failed before model construction:
ACTIVE_ETH program28464 bytes exceeds26624 config buffer, then error cleanup
segfaulted (exit139); batch_cache_watcher.log. No live process remained.
Prescribed scoped retry uses TT_METAL_WATCHER=10 and
TT_METAL_WATCHER_DISABLE_ETH=1, same command/test, writable runtime_logs.
No serving profiler or device-profiler collection was attempted.

Scoped Watcher retry exited0; batch_cache_watcher.json matches two standalone
controls for31/63-token prompts and three decode steps, trims32 wire rows to2
logical rows, preserves external pool geometry. Watcher checked all four
compute devices with no assertions; ETH instrumentation disabled as documented.
Candidate full-model server launched with candidate_server_command.json;
PID88237. Candidate source hashes: candidate_source_manifest.json.

Full candidate first-use shared benchmarks passed both profiles, saved in
candidate_first_use/. Primary TPOT17.7136ms is provisional; final warmed
checks supply acceptance metrics.
`python -m models.autoports.google_gemma_4_26b_a4b_it.tests.check_vllm_requests
--output models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_vllm/requests_final.json
--reference models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/requests_full_final.json`
exited0. All nine96-token requests completed; distinct31/63/95-token prompts
matched isolated and prior validated tokens when concurrent;33/1057/33 reuse
matched. These cross real allocator-driven page boundaries, unlike the direct
changed-unused-page diagnostic, and exercise changing batch/trace output shapes.
Final shared checks now run final_checks_command.json in order
qualitative,benchmark,sampling; sampling profile full.

Repository pre-commit passed on generator.py and both changed test files. All candidate runtime hashes remain unchanged; precommit_code.log. No C++ build required (Python/docs-only change).

## Final warmed comparison

comparison.json derives metrics from before/ and after/ raw shared-runner JSON.
Primary4096/128/B1/C1: TTFT2642.792698 ->2498.284880ms;
meanTPOT19.666982 ->19.747422ms; medianITL19.597341 ->19.598357ms;
aggregate24.898760 ->25.566518tokens/s; TPOT-derived50.846643 ->50.639523t/s/u.
CI100/100/32: TTFT14702.596603 ->14511.986676ms;
meanTPOT681.289169 ->681.786750ms; aggregate39.111150 ->39.177649tokens/s.
Both profiles completed every required request. RawP99 values are retained.
Decode is effectively flat; no throughput speedup or causal prefill improvement
is claimed from a one-request primary benchmark. The retained change removes
proven eager dispatch without changing decoder math or adding tensor storage.

First-use candidate TPOT17.713591ms versus meanITL19.5620ms was inconsistent;
review identified streaming-chunk timing as a plausible explanation. Final
warmed TPOT19.747422 and meanITL19.747424 agree. First-use numbers are excluded
from performance claims, with evidence retained in candidate_first_use/.
Final serving50.639523t/s/u is98.05% of selected full-model autoregressive
51.647281t/s/u at4096/128/B1/C1; prompt/harness differences are explicit.

All6 greedy qualitative responses match the prior full-model-serving control
exactly; all6 sampled responses inspected. Degeneracy checker exited0.
qualitative_verdict.md records controlled wording and truncation limitations.

Final shared runner exited0: full sampling72passed,1skipped,zero failures in
1119.96s. The single skip is the documented default-cap all-vocabulary chat
logprob case. final_validation.json and after/sampling_tests.log record it.
Stage10 checker exited0: no degeneracy, full262144 context preserved.
SIGINT to runner88237 stopped API88260 and EngineCore88297; all owned PIDs gone
(cleanup.json). No reset or profiler used. Final queued-read/rebind probe now
runs after server shutdown with the same allocation-tracker flags as before.

Final queued-read allocation-tracker probe exited0 (adapter_queued_reads.json).
Two decode/read submissions occur before the first host wait; their distinct
host buffers retain correct values/shapes after B1-to-B2 slot rebind. Persistent
device identities and unchanged-state refresh counters remain stable. Devices
closed normally. No serving process remained before this probe. Shutdown
nanobind105instances/973types/4455functions matches inherited baseline counts;
this is a binding teardown warning, not a leftover-process claim.

Independent final stage review returned clean-pass (stage_review.md), with no
required work. Python-only implementation: no C++ build required. Host contract
tests84/84 and code pre-commit passed; evidence pre-commit checked separately.
Local checkpoint SHAs are recorded in checkpoints.json; no push.

Local implementation/evidence checkpoint: `8148e15d639eaf777953f096f1f570adc53455a7`.
Unmodified vLLM checkpoint: `7f72b1c6e905f5137fe3377f2e7b42738d3f271d`. Commit hooks passed.
