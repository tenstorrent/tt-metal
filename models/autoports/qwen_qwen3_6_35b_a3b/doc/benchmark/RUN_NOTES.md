# Qwen3.6-35B-A3B final benchmark work log

Stage started 2026-11-18 02:10 UTC from tt-metal `6dcf319c`. Stage10 owner-accepted implementation remains selected. Four Blackhole p300c devices were idle; bounded tt-smi listing confirms all four. No reset or foreign-process termination. All hardware work is serial on the full 1x4 mesh. Prefix caching remains disabled under the recurrent-state serving contract. Context contract is preserved.

## Setup before benchmark_stage invocation

Provisioned separate lm-eval 0.4.13/transformers<5 client and vLLM 0.26 performance client. First vLLM wheel download hit a 30s network timeout; explicit 300s HTTP timeout retry succeeded. The TT server environment remains unchanged (vLLM 0.31.0). Setup command logs are retained under setup/. Exact elapsed setup will be recorded before invocation.

The packaged common reasoning manifest is reused. GPQA preparation failed because Idavidrein/gpqa requires an authenticated account with dataset access; no credentials were exposed or changed. Select MMLU-Pro and IFEval over the same packaged common questions. MMLU-Pro published reference: 85.2%, https://huggingface.co/Qwen/Qwen3.6-35B-A3B . No exact-model IFEval reference is listed on that card. Native general-task thinking policy: temperature1, top_p0.95, top_k20, min_p0, presence_penalty1.5, repetition_penalty1; output budget32768, native template enable_thinking=true once on server. Upstream prompt/scorer retained, few-shot as structured multiturn; textual stops removed to preserve reasoning and native EOS.

The installed runner requires max(8,3*b) 4096/128 requests per profile and two 32768/128 lone-user requests. Historical serial prefill~22ms/token projects beyond the default7200s client clock even without accuracy. An asynchronous owner question requests an explicit clock decision; no extended client budget has been assumed.

AutoDebug phase-collector investigation uses the selected plugin launcher. Sandbox preflight failed (unprivileged namespaces disabled); unrestricted-authorized retry with AUTODEBUG_SKIP_CHILD_SANDBOX=1 retains fresh-process inspection isolation. Logs retained; no device execution in investigator.

Initial full40 server startup completed in275.5485s, before benchmark_stage invocation (setup/configuration-initial-b32.json). Worker grid observed11×10 on each of deviceIDs0,1,2,3; imported module path is the generated autoport. Context check passes at262144; historical nested-cap mentions remain advisory.

Pilot helper initially bypassed the upstream JsonChatStr-to-messages conversion and failed before any HTTP request. Corrected conversion matches TemplateAPI.create_message; failed preparation log retained. Four pilot prompts use unchanged frozen documents; compact harness docIDs map to original IDs in setup/pilot-native-template.json. Native presence_penalty1.5 selects the existing supported host-logit sampling route; greedy performance remains device sampled.

Pilot completed: four requests at concurrency4 on the32-slot server in485.6345s. Outputs1916,3360,3817,3908tokens; four normal stops; no blank final or truncation. Server prompt tokens1947,1607,87,48 exactly match the pinned native template. The reduced frozen manifest retains14/12032 MMLU-Pro questions (one per subject) and18/541 IFEval questions. Selection is a prefix of the common seeded ranking, verified by document IDs/content hashes; no scores informed selection. Proposed timed configuration retains the7200s client limit, sets a separate10800s setup limit, and makes no batch skips. No extended clock authorization has arrived.

Local source commits:6ef68e29 adds opt-in phase instrumentation;9ae7a8c2 adds server control, native pilot and accounting tools. Initial server imported6ef68e29 model source; later tool-only commit did not change the running model. No commits were pushed.

Attempt1 invocation prepared at 2026-11-18T02:43:22.874056+00:00. Pre-invocation setup wall time 2002.874s from documented02:10UTC start (minute precision). Configuration:7200s client clock,10800s separate setup clock. Fresh native pilot and collector smoke completed before invocation. No extended client budget assumed.

Attempt1 failed before requests:0.002096s client,0.063737s setup; command spelling integration fault (underscore alias not recognized by evidence checker), not a capacity mismatch. Preserved under attempts/attempt1. Corrected launcher records canonical flag with identical capacity; commit b08fe50aea6f34b9790d8eec5005ed3d7d37df6d. Fresh launch and full identity/collector validation precede authorized attempt2.

Repaired smoke2 (setup only):32server slots,2concurrent requests,2completed, exact8192input/256output tokens;188.6663s measured HTTP wall. Collected prefill182.459984s and decode6.192962s with matching request lifetime/chunk/position accounting; no new synchronization or profiler. It is not a required batch32 performance row. Raw evidence:setup/collector-smoke-tokenizer-fixed/. Original tokenizer-fault smoke is retained separately.

Attempt2 invocation prepared 2026-11-18T02:49:42.084706+00:00. Fresh server launch 275.533s completed before invocation. Observed launch/configuration and collector readiness checks pass. Pre-invocation wall 2382.085s from documented02:10UTC start (minute precision), including setup, repairs and negligible failed attempt1. Client7200s and setup10800s bounds remain unchanged.

Timing clarification: upstream vLLM0.26 prints “Starting initial single prompt test run” but with default ready_check_timeout_sec=0 it skips that HTTP probe; num_warmups=0 also adds no hidden requests. The stage runner supplies its own explicit warmup workload. Fresh2-request smoke had182.459984s full prefill =91.229992s per4096tokens. Fixed performance repetitions total176prompts, projecting16056.479s of prefill alone; this excludes accuracy and the long lone-user row. This is a measured clock conflict, not a reason to reduce context, change precision or omit an unmeasured batch silently.

Upstream protocol identities:lm-eval0.4.13 MMLU-Pro task metadata3.1,5-shot validation first_n, subject-specific CoT descriptions and custom-extract regex `answer is \(?([ABCDEFGHIJ])\)?` then take_first; exact_match ignores case/punctuation. IFEval metadata4.0,train split as evaluation population,zero-shot,upstream process_results and four strict/loose prompt/instruction metrics. Frozen documents/few-shot content and hashes remain in the manifests. Copies of upstream recipe/scorer sources are in setup/task-recipes.json. Native generation overrides are explicit in run_config.json; only final-answer content is scored.

Weight provenance:tt/model.py pins HF_REVISION=345d467b117d1fd5eda9b879a5988de9ae5f47a7 and Checkpoint defaults to HF_HOME/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/<revision>; the generated adapter uses that default when hf_config._name_or_path is the model ID. Startup logs record each loaded layer0–39, and initialization records the exact selected precision policy. Server argv pins both model and tokenizer revisions. Tokenizer/config asset identities and installed package records are retained in setup/. Stage9 installation manifest and pinned plugin docs/install-vllm-tt.sh establish vLLM0.31.0 from the PyPI sdist built for empty target; installed version is0.31.0+empty, with no embedded git metadata. The release commit is labeled as such.

Attempt2 32-slot warmup completed all32 requests:2977.948277s HTTP wall,131072 input/4096 output tokens (exact4096/128 per request),32 concurrent requests. Accuracy started around03:39UTC on the client clock. This explicit warmup is setup evidence, not the repeated measured batch32 row.

Attempt2 failed during shared accuracy: the installed evaluate.py hardcodes a3600s per-request HTTP timeout. At04:39:42UTC the outstanding IFEval document59 raised aiohttp TimeoutError; the shared gather aborted before scoring, despite only3616.330224/7200 client seconds used. All14 MMLU-Pro and17/18 IFEval raw responses were retained; all31 normal stops, no blank finals or token limits. The missing document is a transport failure, not a model answer or zero score. Original failed summary/report are preserved separately before offline recovery. No third fresh inference attempt was started.

Independent AutoFix audit confirmed the hidden child timeout precedes the parent deadline. The source repair is committed in ../skills-v24 as f61cb04660b549b370258c9d253a55421f4d4cfa; the installed cached package remains unchanged. Offline upstream replay matches frozen request-instance hashes and recovers MMLU-Pro11/14=78.5714%, extraction14/14. IFEval retains all18 selected questions;17 collected answers may be scored individually, but its frozen-subset headline remains unavailable.

The server became idle after cancellation. Buffered phase evidence was flushed before the owned idle-server stop; shutdown_record.json and stop-owned-server.log confirm cleanup, and fuser found no remaining device holders. No reset was used. Warmup-only accounting reconciles32requests/131072input/4096output,64prefillchunks and158decodecalls:2898.122657s full prefill and79.790499s full decode, estimated0.217549% compute/8.725437%DRAM rooflines. This is explicitly setup evidence, not a required warmed profile.

Evidence directories were moved under run/ with relative symlinks at the original paths, retaining frozen path references and file bytes. Post-failure offline scoring/reporting time will be added to the client clock; no clock reset or fresh run is used.

## Final offline handoff

The workflow timeout repair propagates the remaining parent client deadline to the accuracy HTTP client, retaining the parent process deadline. Local skills-v24 commit `f61cb04660b549b370258c9d253a55421f4d4cfa` passed 744 tests with 1 skip, package/version/catalogue checks and whitespace validation. The plugin source version is 0.1.25; installed 0.1.24 was not mutated. No hardware claim is made for this repair. See run/setup/workflow-timeout-AUTOFIX.md and run/workflow-timeout-fix.json. Nothing was pushed.

Offline scoring forbids network/model calls and matches all 31 saved request-instance hashes. All 32 frozen inputs remain in run/benchmark-inputs.jsonl. Complete MMLU-Pro is 11/14 (78.5714%); partial IFEval scores are explicitly separate from the unavailable frozen 18-question result. The report's answer statistics were read: all retained answers are extracted, normal stops, nonblank and not cap-limited.

Final context check exits 0 with full 262144-token capacity. Evidence check exits 2, CRITICAL: benchmark stage did not complete. No warmed batch profiles or lone-user measurements are claimed. Independent offline handoff audit confirmed linked evidence, scores, frozen denominators, setup/performance distinction, continuous clocks, raw-response hashes and absence of an accuracy acceptance verdict. Raw logs/full snapshots remain locally intact and are enumerated in run/artifact-retention.json.

Remaining work requires another timed inference attempt after installing the timeout repair and resolving the measured client-clock conflict. Both authorized fresh attempts have been used. The selected benchmark run contract says “Ask only before a third.” No third attempt or extended budget was assumed.

Repository end-of-file checks added exactly one final newline to three upstream performance JSON files. Their original bytes remain under run/raw-originals/; run/raw-formatting-provenance.json records both hashes. JSON values and all accuracy response bytes are unchanged. These original files join logs/snapshots in the local retention manifest.

Benchmark evidence and recovery tool committed locally as `4e10ec4cbde1d5bebb88b72680d21be03863cf35`; all required repository hooks passed. Prior source and workflow repair SHAs are logged in run/commits.json. Final bookkeeping follows in a separate commit. Nothing was pushed.

Continuation preparation: run/retry-proposal/ contains an inactive third-attempt configuration and bounded timing proposal. Current runner rejects the proposed 8-hour client budget; no limit was bypassed and no new inference was launched. Previous failed-run clocks/report remain unchanged. Explicit approval is pending for the third attempt and client-budget extension.

Owner Vladan approved attempt 3 with an eight-hour client bound, existing three-hour setup bound and raised enclosing wall cap. Full protocol remains unchanged. Owner also directs stage completion with explicit partial results and known limitations if this final attempt cannot produce a row within its limits; no fourth attempt. Exact approval is recorded in run/owner-approval.json. Third-attempt preparation is timed separately from invocation. Prior identity is preserved in run/failed-execution/identity.json. New server states/logs will use run/attempt3-support/ without overwriting prior evidence.

Third-attempt workflow: skills-v24 commit f2873162ea1b8b4971aa9b41c83a551e7c1d53e5 (version 0.1.26) adds the explicit owner-approved budget; 766 CPU tests passed, 1 skipped, package/version/host checks passed. Existing HTTP timeout repair f61cb046 is retained. Model-side launch/support changes committed378452e4. No installed cache edits or pushes. The default budget guard and measurement checks remain enforced.

Attempt3 warmup completed around06:09:50UTC: all32 requests, exact4096 input/128 output each,131072/4096 totals,2994.001424s wall. Accuracy now runs at concurrency32 on32 server slots using the unchanged native-thinking generation dictionary. Observed child command includes --request-timeout-seconds28800, confirming the timeout override reached the live evaluation process. Completion of the full accuracy pass remains pending. This warmup is setup evidence only.

Attempt3 accuracy completed around07:13:55UTC (shared accuracy wall3844.048666s): all32 frozen questions retained and scored upstream,14 MMLU-Pro and18 IFEval. MMLU-Pro11/14=78.5714%; IFEval prompt15/18=83.3333% and instruction29/32=90.625%, identical strict/loose scores. Both extraction shares100%; all32 normal stops, no blank finals or token limits. Official answer-statistics helper reports MMLU tokens median1521/p902295/max2560 and IFEval median2438/p906025/max16352; selected/recommended cap32768. No statistics advisories. Document59 completed after the former3600s HTTP limit, validating the timeout repair on the live serving path. Statistics are preserved in attempt3-support/setup/completed-accuracy-statistics.json; the final runner report will be inspected again. The96-request4K/128 batch32 measurement is now running; offline input export is deferred until timed serving work ends.

### Third attempt: measured batch 32 completed

At 2026-11-18 09:46 UTC, all 96 measured requests completed, each with 4096 input and 128 output tokens; no request errors. Measured duration 9141.294454 seconds. The collector finished before the runner advanced to the concurrency-1 FUNCTIONAL warmup, reporting full-phase prefill 8918.901485 seconds and decode 222.369775 seconds, with matching 2189432639719680 prefill FLOPs and 39562640716909.4 estimated decode DRAM bytes. Compact statistics are retained in `run/attempt3-support/setup/completed-b32-statistics.json`; all raw performance and accounting files remain under `run/attempt3/`. Independent audit of recorded replacement-phase events is pending. No server or client was restarted for this milestone.

The independent batch-32 accounting audit subsequently passed: 608 complete calls, all 96 lifetimes with exactly 127 decode positions, and all 190 prefills following initial decode beginning with no pending decode interval. No cross-phase overlaps or missing completions; phase/work totals and 31 detail-file hashes reconcile. See `run/attempt3-support/setup/completed-b32-accounting-audit.md`. The 32-slot, concurrency-1 FUNCTIONAL profile also completed all four 128/128 requests with no errors: 28.069544 seconds total, mean TTFT 3032.590436 ms, mean TPOT 31.373871 ms. Its raw result is `run/attempt3/perf-b32-c1-functional.json`. The runner proceeded to the long-context warmup.

### Third attempt: measured long-context row completed

At 2026-11-18 10:24 UTC, both concurrency-1 long-context requests completed on the 32-slot server, each with 32768 input and 128 output tokens and no errors. The preceding warmup completed one 32768/128 request in 748.269114 seconds (setup only). Raw measured output is `run/attempt3/perf-b32-c1-long_context.json`, with compact statistics in `run/attempt3-support/setup/completed-long-context-statistics.json`. The runner is proceeding to the batch-16 server/profile.

### 2026-11-18T10:30:15.923011+00:00 — batch-16 launch and warmup

Batch-32 owned shutdown completed before batch-16 launch. The new server identity passes `benchmark_stage.evidence.validate_server` against the batch-32 baseline, including configuration SHA; imported generator-module SHA also matches. All 40 layers, baseline_bfp8, full 262144 context and prefix caching false are preserved. Collector preparation passed before the 16-request 4096/128 warmup. Warmup client PID 3619420, creation time 1794997767.07; setup remains on the separately bounded setup clock. No required batch-16 measurement has completed yet.

### 2026-11-18T10:54:03.074416+00:00 — batch-16 warmup complete; measurement started

Warmup validated 16/16 exact 4096-input/128-output requests with zero errors, duration 1455.9872566470003 seconds, setup-only. Raw bytes are preserved; compact statistics are in `run/attempt3-support/setup/completed-b16-warmup-statistics.json`. The runner started measured client PID 3690586 (creation 1794999227.79) for 48 requests at concurrency 16 on the validated 16-slot server. Required measurement and phase collection are pending.

### 2026-11-18T12:07:22.549451+00:00 — measured batch-16 complete

Required warmed profile validated 48/48 exact 4096-input/128-output requests, concurrency 16 on the validated 16-slot server, zero errors. Duration 4390.070053438016 seconds. Identity/configuration and baseline continuity pass; prefix caching remains false. Raw bytes are preserved; compact statistics are in `run/attempt3-support/setup/completed-b16-statistics.json`. The runner is collecting full-phase accounting before the owned batch-16 server stops.

Batch-16 accounting collection completed before the runner switched servers: 48 measured requests, prefill 4283.864746723993 s / 1094716319859840 FLOPs; decode 106.18672229800723 s / 20518017126111.207 estimated DRAM bytes. Derived roofline fractions: 0.220764553870367% prefill and 9.4348547850855% decode. Independent saved-artifact audit requested. At 2026-11-18T12:08:02.138433+00:00 the runner is launching batch 8 (setup helper PID 3902579, creation 1795003623.64).

### 2026-11-18T12:09:27.772061+00:00 — batch-16 independent audits

Saved-artifact accounting audit passed: 48 request lifetimes, 496 complete calls, 127 decode positions per request, no unfinished or cross-phase-overlapping intervals, and all 24 detail hashes/work totals/phase times reconcile. See `run/attempt3-support/setup/completed-b16-accounting-audit.md`. A source/saved-timing audit explains upstream `max_concurrent_requests=18` (b16) and 34 (b32): inclusive one-second occupancy buckets include sequential replacements. The configured semaphore limits are 16 and 32; reconstructed unrounded stream overlap peaks are 16 and 32. Raw metrics remain unchanged. Report these as configured concurrent requests; if the derived bucket metric is shown, label its semantics explicitly. See `run/attempt3-support/setup/concurrency-metric-audit.md`.

### 2026-11-18T12:12:17.069046+00:00 — batch-8 launch validated; warmup started

Batch-8 server passes identity/baseline continuity, observed launch capacity, referenced configuration SHA and imported module SHA checks. It retains all 40 layers, baseline_bfp8, full 262144 context and prefix caching false. Collector preparation passed before the eight-request 4096/128 warmup. Warmup client PID 3915369, creation time 1795003910.21. Required 24-request measured profile remains pending.

### 2026-11-18T12:24:16.971377+00:00 — batch-8 warmup complete; measurement started

Warmup validated 8/8 exact 4096-input/128-output requests with zero errors, duration 725.1626215970027 seconds, setup-only. Raw bytes are preserved; compact statistics are in `run/attempt3-support/setup/completed-b8-warmup-statistics.json`. The runner started measured client PID 3950672 (creation 1795004641.32) for 24 requests at concurrency 8 on the validated eight-slot server. Required measurement and phase collection are pending.

### 2026-11-18T13:00:36.756738+00:00 — measured batch-8 complete

Required warmed profile validated 24/24 exact 4096-input/128-output requests, concurrency 8 on the validated eight-slot server, zero errors. Duration 2174.825375843997 seconds. Identity/configuration and baseline continuity pass; prefix caching remains false. Raw bytes are preserved; compact statistics are in `run/attempt3-support/setup/completed-b8-statistics.json`. The runner is collecting full-phase accounting before the owned batch-8 server stops.

Batch-8 accounting collection completed before the runner switched servers: 24 measured requests, prefill 2120.819512908027 s / 547358159929920 FLOPs; decode 53.989997582975775 s / 10732133410658.77 estimated DRAM bytes. Independent saved-artifact audit requested. The runner is launching batch 1 (setup helper PID 4055809, creation 1795006821.58).

Independent batch-8 audit passed: all 24 lifetimes, 440 complete calls, 21 detail hashes, positions, full phase times and modeled work reconcile; no pending or cross-phase-overlapping intervals. Exact stream overlap peaks at 8; inclusive one-second occupancy buckets reproduce raw max_concurrent_requests=10. See `run/attempt3-support/setup/completed-b8-accounting-audit.md`.

### 2026-11-18T13:05:42.389203+00:00 — batch-1 launch validated; warmup started

Batch-1 server passes identity/baseline continuity, observed launch capacity, referenced configuration SHA and imported module SHA checks. It retains all 40 layers, baseline_bfp8, full 262144 context and prefix caching false. Collector preparation passed before the one-request 4096/128 warmup. Warmup client PID 4069328, creation time 1795007128.15. Required eight-request measured profile remains pending.

### 2026-11-18T13:07:13.890131+00:00 — batch-1 warmup complete; measurement started

Warmup validated 1/1 exact 4096-input/128-output request with zero errors, duration 86.19283088899101 seconds, setup-only. Raw bytes are preserved; compact statistics are in `run/attempt3-support/setup/completed-b1-warmup-statistics.json`. The runner started measured client PID 4073818 (creation 1795007219.33) for eight requests at concurrency 1 on the validated one-slot server. Required measurement and phase collection are pending.

### 2026-11-18T13:18:51.836655+00:00 — measured batch-1 complete

Required warmed profile validated 8/8 exact 4096-input/128-output requests, concurrency 1 on the validated one-slot server, zero errors. Duration 693.2189016269986 seconds. Identity/configuration and baseline continuity pass; prefix caching remains false. Raw bytes are preserved; compact statistics are in `run/attempt3-support/setup/completed-b1-statistics.json`. Final phase collection and runner reporting are pending.

### Third attempt: serving complete and offline handoff

Runner exit 0; all four required profiles and both lone-user rows completed. Original runner clocks: client 21806.454037914024 s / 28800, setup 7209.513573397999 s / 10800; elapsed 29015.967611312022 s. Original report/summary preserved in `run/attempt3-support/setup/runner-final/`. Continuous finalization time is charged to the client from the original summary mtime, including the observation gap. The report answer statistics were read: MMLU-Pro 14/14 and IFEval 18/18 extraction, no blank-at-cap answers or truncations. Inputs-only offline export retained all 32 contexts and verified all 32 response links without rescoring. All four owned server shutdowns passed, and fuser found no device holders; no reset or reservation release. Batch-1 accounting independently passed all eight lifetimes, 1032 complete calls and 37 detail hashes with no pending or cross-phase overlap. Its exact stream overlap is 1; raw inclusive one-second buckets reproduce 2. Canonical result paths now link exclusively to attempt3; old attempt2 files were archived with a path map. Root identity again matches the largest measured server.

Repaired workflow stage checker exits 0: benchmark evidence passes; context checker reports zero critical findings and full 262144 capacity. Historical log-scanning context advisories are retained. Context contract bytes are unchanged from HEAD. Installed cached 0.1.24 evidence checker exits 2 solely because it rejects budgets above 7200; its exact output is preserved and the enclosing runner incompatibility is explicit. The approved 0.1.26 checker validates the 28800 approval without relaxing measurement checks. Raw performance bytes are frozen before hooks; a narrowly anchored EOF-fixer exclusion preserves collector-bound upstream JSON and the original runner report, while all hooks remain enabled.

Normal hook preflight formatted the finalization helper and exposed two compact raw phase snapshots without final newlines. Their original staged bytes were restored, and those two exact collector-source paths were added to the visible EOF-only exception. All raw performance hashes remain unchanged; the complete hook suite is rerun after this correction. No hook was disabled or bypassed.

Post-hook validation passed: complete normal hook suite exit 0; repaired stage checker exit 0 with zero critical context findings and 61 historical log-scan advisories. All 12 immutable performance files match frozen hashes in both worktree and staged Git blobs; retained response/input hashes, all 32 input records, report links and canonical links pass. A fresh independent stage review is next; no further inference is planned.

Independent Stage 11 review returned clean-pass with no Required Work. Reviewer inspected all 182 performance requests, 32 accuracy responses, raw metrics, protocol, identity, phase work/timing, clock continuity and cleanup. Report: `run/attempt3-support/setup/stage-review.md`. No accuracy acceptance verdict was assigned. Remaining actions are the prescribed final clock refresh and local checkpoints; no more inference or review round is required for bookkeeping.

Evidence/report checkpoint committed locally on tt-metal main as `ed1a846828af7068d3e38ca01e36bbea7750a8f8`; all normal hooks passed and all 12 committed raw performance blobs match their frozen hashes. Workflow source remains clean at `f2873162ea1b8b4971aa9b41c83a551e7c1d53e5`. The final bookkeeping checkpoint SHA and post-commit clock stamp are recorded in `run/attempt3-support/setup/closing-record.log` and the final handoff response. Nothing was pushed.

Closing evidence/context check again exited 0 after the evidence checkpoint. Exact output is retained in `run/attempt3-support/setup/stage-check-closing.log`; final check metadata is in `run/checks.json`. Continuous client time still includes all finalization since the original runner summary. The old cached-checker budget incompatibility remains the only enclosing-runner integration limitation; it does not change the successful approved-budget measurement checks.

## Runner-check remediation

The cached 0.1.24 exit-2 budget finding was reproduced and refuted against the original owner approval. The approval-aware 0.1.26 stage check passes on unchanged measurements; 29 focused budget/provenance tests pass. See [work_log.md](work_log.md) for original-message provenance, exact commands, causal isolation, fresh check outputs and runner disposition. No new inference attempt or hardware operation was performed.
