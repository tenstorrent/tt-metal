# Stage 12 — active

| Suite | functional_at | Evaluation hours/run | Score (95% Wilson interval) | Reference |
|---|---:|---:|---|---|
| GPQA Diamond | 10 | 0.17846 at C4 | 9/10, 90% (59.58–98.21%) | Published84.3%, GPU83.33% |
| SWE-bench Verified | 2 | 0.10416 | 0/2, 0% (0–65.76%) | NA |
| Terminal-Bench 2.1 | 2 | 0.63756 | 0/2, 0% (0–65.76%) | NA |

These are banked subset scores, not release results. No dry run or release has started. Small intervals are not implied by repeated evaluations of the same tasks; these samples cannot establish full-suite parity with references. Reference provenance is retained in survey/preflight evidence; unavailable comparable references remain NA.

GPQA C4 is selected from a current-runtime, same-host pair: C2 took938.039seconds and C4 took642.446seconds, a31.51% reduction. Both scored9/10 with identical visible response hashes and equal executed token work. C4 used0.29556device-job hours and0.30111workflow hours. A nested20-task rung is still running at its original C2 setting. Full198 was stopped after10 reported completions: empirical projections24.55–29.46hours exceeded12hours. Those uncertain projections are not proof of looping or mathematical impossibility. See `runs/gpqa-c4-scored10/RESULT.md` and `runs/full-gpqa-observable/stop.json`.

SWE retains its valid two-task smoke. Original nightly5 was held for network contamination and missing cap evidence. Repeat42228448171 completed in1.12276evaluation hours,1.21083device-job hours and1.21583workflow hours, against7.7925hours forecast. Raw1/5 (Wilson95%3.62–62.45%) is not accepted: four tasks are valid, but a missing Astropy response aligns with32767decode steps and cannot be classified as repetition versus interrupted progress. The frozen5 is not banked, and four audited tasks are not a replacement cohort. No external retrieval was observed in this repeat; all220saved responses retain token IDs matching usage. The model-owned token-retention repair passed paired exactness and is published at11a387. The same frozen5 is queued as42247098210 with a new image; forecast8.54417hours includes build, full agent tail and queue. See `runs/nightly-swe-retained-repeat/audit.json`.

Terminal retains its valid two-task smoke. Nightly10 was stopped at two reported completions after2.2evaluation hours: completed-throughput plus observed queue/setup projected12.1486workflow hours, beyond the12hour limit. No nightly score or confirmed loop is claimed; this is a measured stop with functional_at2. See `runs/nightly-terminal-observable/stop.json`.

Extended mode has48hours from2026-11-06T04:28:49UTC to2026-11-08T04:28:49UTC for release start, and12hours per release run. Operator decision8 authorizes CI for Docker tasks, GPQA and final dry/release; local Docker remains unavailable. Task lists, immutable run SHAs, failed attempts, actual times and current projections remain in `cohorts.json`, `compute.json`, `estimate.json` and `plan.json`. The unresolved tool-suite dry-route and disk-admission questions are recorded in `blocked.json`; GPQA-only early-stop dry procedure is conditional on the final go set.

## Effort predictions and results

Costs are device-job hours, excluding build/queue unless stated. Savings are per projected release list; guesses remain marked in efforts.json. Each linked result owns its evidence.

| Effort | Predicted / used device hours | Change / problem | Confirmation evidence |
|---|---:|---|---|
| local-sandbox-runtime | 0.00000 / 0.00000 | none / solved | runs/local-runtime-repair/result.json; setup/AUTOFIX.md |
| gpqa-local-credential | 0.00000 / unknown | not_run / solved | runs/smoke2-r1_gpqa_diamond/audit.json |
| agent-capacity-and-contention | 0.25000 / 0.07763 | none / open | runs/agent-capacity-and-contention/REPLAY_RESULT.json |
| source-only-packaging | 0.00000 / 0.00000 | kept / solved | runs/source-packaging/AUTOFIX.md |
| harness-registration | 0.00000 / 0.00000 | kept / solved | runs/harness-registration/result.json |
| suite-registration | 0.00000 / 0.00000 | kept / solved | runs/suite-registration/RESULT.md |
| ci-release-route | 0.00000 / 0.00000 | none / open | runs/ci-release-route/RESULT.md |
| local-serving-check | 0.20000 / 0.09213 | none / solved | runs/local-serving-check/RESULT.md |
| local-host-contention | 0.10000 / 0.09849 | none / solved | runs/local-host-contention/RESULT.md |
| nightly-registration-preparation | 0.00000 / 0.00000 | none / solved | runs/nightly-preparation/RESULT.md |
| local-long-context | 0.15000 / 0.25401 | none / solved | runs/local-long-context/RESULT.md |
| parser-protocol | 0.10000 / 0.10580 | kept / solved | runs/parser-protocol/RESULT.md; runs/parser-protocol/AUTOFIX.md |
| ci-fabric-topology | 0.21670 / 0.00000 | none / solved | runs/ci-topology/result.json |
| native-tool-roundtrip | 0.00278 / 0.00052 | none / solved | runs/native-tool-roundtrip/result.json |
| ci-cache-routing | 0.08333 / 0.05278 | kept / open | runs/ci-startup-analysis/paired-audit.json |
| immutable-harbor-release-source | 0.00000 / 0.00000 | kept / solved | runs/release-dependency-pins/AUTOFIX.md |
| nightly-current-base-preparation | 0.00000 / 0.00000 | none / solved | runs/nightly-refreshed-preparation/result.json |
| real-history-capacity-preparation | 0.00000 / 0.00000 | none / open | runs/real-history-capacity/RESULT.json |
| stage12-zero-cost-checker | 0.00000 / 0.00000 | none / open | runs/stage12-checker-zero-cost/RESULT.md |
| explicit-autoport-dispatch | 0.00000 / 0.00000 | kept / solved | runs/ci-implementation-selection/result.json |
| harbor-model-pin-confirmation | 0.00000 / 0.00000 | kept / solved | runs/harbor-model-pin/RESULT.md |
| gpqa-server-sized-preparation | 0.00000 / 0.00000 | none / solved | runs/gpqa-server-sized-preparation/RESULT.md |
| nightly-resource-preparation | 0.00000 / 0.00000 | none / solved | runs/nightly-resource-preparation/result.json |
| prefill-under-load | 0.15000 / 0.15284 | none / solved | runs/prefill-under-load/RESULT.md |
| long-prefill-decode-interference | 0.25000 / unknown | not_run / pending | runs/prefill-under-load/RESULT.md |
| docker-startup-timeout-leaks-own-container | 0.00000 / 0.00000 | kept / solved | runs/ci-container-startup/AUTOFIX.md |
| terminal-current-base-preparation | 0.00000 / 0.00000 | none / solved | runs/terminal-current-base-preparation/RESULT.md |
| gpqa-terminal-state-capacity | 0.10000 / 0.00000 | none / open | runs/gpqa-terminal-state-capacity/RESULT.json |
| gpqa-current-base-confirmation | 0.00000 / 0.00000 | none / solved | runs/gpqa-server-sized-current-base/RESULT.md |
| nightly-current-base-reconfirmation | 0.00000 / 0.00000 | none / solved | runs/nightly-current-base-preparation/RESULT.md |
| gpqa-server-sized-c2-preparation | 0.00000 / 0.00000 | none / solved | runs/gpqa-server-sized-c2-preparation/RESULT.md |
| gpqa-nightly-runtime-analysis | 0.00000 / 0.00000 | none / open | runs/gpqa-nightly-runtime-analysis/RESULT.json |
| ci-phase-autoexport | 0.15000 / 0.10390 | kept / solved | runs/ci-phase-autoexport/result.json; runs/ci-phase-autoexport/path-followup-result.json |
| swe-network-validity | 0.00000 / 0.00000 | kept / solved | runs/swe-network-validity/AUTOFIX.md |
| django-cap-loop-diagnosis | 0.25000 / 0.54024 | none / open | runs/django-cap-loop-diagnosis/result.json |
| real-history-capacity-replay | 0.05000 / 0.01613 | none / solved | runs/real-history-capacity/REPLAY_RESULT.json |
| measured-workload-prefill | 0.15000 / 0.27580 | kept / solved | runs/measured-workload-prefill/RESULT.md |
| terminal-smoke-runtime-analysis | 0.00000 / 0.00000 | none / solved | runs/terminal-smoke-runtime-analysis/result.json |
| next-rung-after-terminal-preparation | 0.00000 / 0.00000 | none / solved | runs/next-rung-after-terminal/RESULT.md |
| native-history-retention | 0.00000 / 0.00000 | none / solved | runs/native-history-retention/RESULT.md |
| terminal-server-sized-retention-preparation | 0.00000 / 0.00000 | none / solved | runs/terminal-server-sized-retention/VALIDATION_REBASED.json |
| swe-phase-accounting-confirmation | 0.00000 / 0.00000 | none / solved | runs/swe-phase-accounting-confirmation/result.json |
| larger-cohort-preparation | 0.00000 / 0.00000 | none / solved | runs/larger-cohort-preparation/RESULT.md |
| swe-valid-runtime-analysis | 0.00000 / 0.00000 | none / solved | runs/swe-valid-runtime-analysis/result.json |
| ci-disk-admission-basis | 0.00000 / 0.00000 | none / open | runs/ci-disk-admission-basis/RESULT.md |
| measured-workload-idle-regime | 0.20000 / 0.10905 | none / solved | runs/measured-workload-idle-regime/RESULT.md |
| serving-shortfall-pricing | 0.00000 / 0.00000 | none / open | runs/serving-shortfall-pricing/RESULT.json |
| ci-image-reuse | 0.00000 / 0.00000 | none / open | runs/ci-image-reuse/result.json |
| current-runtime-benchmark-confirmation | 0.16900 / 0.08735 | none / solved | runs/current-runtime-benchmark/result.json |
| native-token-retention | 0.00000 / 0.00000 | none / open | runs/native-token-retention/result.json |
| native-token-retention-confirmation | 0.12000 / 0.06232 | kept / solved | runs/native-token-retention-confirmation/result.json |
| malformed-response-retention | 0.00000 / 0.00000 | none / open | runs/malformed-response-retention/result.json, runs/malformed-response-retention/AUTOFIX.md |
| nightly-swe-network-reach | 0.00000 / 0.00000 | none / open | runs/nightly-swe-network-reach/result.json, runs/nightly-swe-network-reach/AUTOFIX.md |
| gpqa-c4-capacity | 0.18000 / 0.14046 | none / open | runs/gpqa-c4-capacity/RESULT.json |
| gpqa-c4-scored10 | 0.50889 / 0.76167 | kept / open | runs/gpqa-c4-scored10/RESULT.json, runs/gpqa-c4-scored10/RESULT.md |
| terminal-nightly-tail-diagnosis | 0.00000 / 0.00000 | none / open | runs/terminal-nightly-tail-diagnosis/RESULT.json, runs/nightly-terminal-observable/stop.json |
| gpqa-larger20-preparation | 0.00000 / 0.00000 | kept / solved | runs/gpqa-larger20-preparation/RESULT.json |
| gpqa-only-dry-route | 0.00000 / 0.00000 | none / solved | runs/gpqa-only-dry-route/RESULT.md |
| gpqa-dispatch-route-repair | 0.00000 / 0.00000 | kept / solved | runs/gpqa-dispatch-route-repair/AUTODEBUG.md, runs/gpqa-dispatch-route-repair/AUTOFIX.md |
| swe-repeat-audit-preparation | 0.00000 / 0.00000 | kept / solved | runs/swe-repeat-audit-preparation/RESULT.json |
| astropy-capped-prefix-preparation | 0.00000 / 0.00000 | none / solved | runs/astropy-capped-prefix-preparation/RESULT.json |
| gpqa-current-ten-baseline-preparation | 0.00000 / 0.00000 | none / solved | runs/gpqa-current-ten-baseline-preparation/RESULT.json |
| astropy-prefix-discriminator | 0.12000 / 0.05006 | none / open | runs/astropy-prefix-discriminator/RESULT.json, runs/astropy-prefix-discriminator/RESULT.md |
| model-token-retention | 0.12000 / 0.13158 | kept / open | runs/model-token-retention/RESULT.json, runs/model-token-retention/RESULT.md, runs/model-token-retention/AUTOFIX.md |
| partial-tool-dry-route | 0.00000 / 0.00000 | none / open | runs/partial-tool-dry-route/RESULT.md |
| retained-runtime-benchmark | 0.09000 / 0.09668 | none / solved | runs/retained-runtime-benchmark/RESULT.json |
| retained-token-capacity | 0.00000 / unknown | running / pending | runs/retained-token-capacity/plan.md |
| gpqa20-c4-retained-preparation | 0.00000 / unknown | running / pending | runs/gpqa20-c4-retained-preparation/plan.md |

## Configuration and remaining limits

The fresh full60 autoport uses HF/tokenizer revision842da3794eaa0b77d5f08bae87a17459d91ff475, advertised context262144, mesh1x4 and stock plugin ca7a18246e4111629ed6cb4a7d1192002749fccf. Current runtime publication11a387692d2b434198e79dd92fbb8ed308c3ca47 adds default-off bounded retention of already-host-visible decode IDs. The selected diagnostic config enables it with1second periodic export. Prefill and host-sampled tokens are excluded; physical rows are not request IDs or proof of client emission. The paired check matched583expected IDs and all three native responses, including overlap and remapping, without additional device synchronization.

Selected GPQA concurrency is4 atB8; tool suites remain2 atB8. The active twenty-task GPQA run retains its originalC2. Batches1/8/16/32 are validated; B8 is the smallest supported batch for2 or4 clients. Prefix caching, chunked prefill and asynchronous scheduling remain enabled. Decode and final prefill use the device sampler; mixed/intermediate prefill uses the accepted stock host sampler. Served defaults are temperature1/top_p0.95/top_k32; exact eval values and limits remain in serving_config.json. Task CPU, memory and storage come from task definitions.

The accepted-runtime headline measured eight exact4096-input/128-output requests atB1/C1: TTFT753.316ms, decode35.3181tokens/s/user. The unchanged stage11 TTI matrix completed8/8 in27.08457seconds; its nominal128-token prompts become141–143 adapter tokens through the stock chat route. Total local device hold was0.09668hours versus0.09forecast, within0.18box. No vLLM profiling was used; device-time and roofline fields remain unknown. Evidence: runs/retained-runtime-benchmark/RESULT.json.

GPQA C4 capacity was measured with four resident131072-input contexts,8/8 requests, no preemptions and99.854% return-prefix reuse. This does not establish four full262144 contexts or full-suite accuracy. Earlier C2 synthetic requests reached262136prompt+8output tokens without preemptions or new captures. Actual saved SWE stress histories reached107585tokens; those measurements do not rehabilitate quarantined scores. Measured idle-host cold TTFT at4096/31755/107585tokens was0.783/8.284/48.065seconds. A107585-token prefill overlapped only28delivered anchor tokens over49.259seconds. These are host observations, not isolated device phases or causal release-speed estimates.

C4 GPQA's measured saving on the banked ten was0.0821093hours versus0.0657648predicted; total effort cost0.761667device hours versus0.508889forecast includes the wrong-route attempt. The same-host pair matched2390executed prefill tokens and56042active decode-row steps, but does not prove hidden generated token identity or represent independent accuracy replication.

Astropy's reconstructed31001-token full60 initial-distribution comparison matched the top token, withKL0.317021 andJS0.0412971; truncated sampling supports differed. Device cost0.0500586hours versus0.12forecast includes a cache-launch failure and successful health/retry. The earlier Django42823-token comparison hadKL0.135185. These conditional measurements and short non-looping replays do not explain the original missing capped trajectories. The next useful causal replay requires a newly preserved failing request rather than guessed original seeds.

The stock CI route remains unchanged. GPQA uses workflow=evals; tool suites use agentic; release benchmarks use benchmarks, without spec tests. Two earlier GPQA dispatches to agentic produced no score; AutoDebug/AutoFix corrected dispatch only. Image reuse is allowed only for unchanged TT/plugin runtime and audited TTI configuration. The new retention runtime requires the new image now building for SWE.

Tool-suite exact early-stop dry evidence and disk admission remain unresolved operator questions in blocked.json. Stock logs do not prove that every tool environment has built before grades occur; partial-run source analysis rejected that shortcut. A GPQA-only early-stop dry procedure is conditional on the eventual go set. A tested checker defect rejects legitimate zero-device-hour efforts; the installed skill package remains unchanged pending the recorded operator decision. No dry/release is claimed.

Execution changes are committed locally, with SHAs in the stage commit manifest. This execution checkout has no remotes and is never pushed. Separate authorized checkouts publish source-only commits. The local telemetry packet7ef7a2c1-3dce-40e4-b6d3-0c83a631ac7e contains measured headline values, banked subset scores and missing reasons; raw tokens/logs are excluded.
