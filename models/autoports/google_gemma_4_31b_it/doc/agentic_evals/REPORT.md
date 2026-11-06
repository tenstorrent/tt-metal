# Stage 12 — active

| Suite | functional_at | Evaluation hours/run | Score (95% Wilson interval) | Reference |
|---|---:|---:|---|---|
| GPQA Diamond | 10 | 0.26304 | 9/10, 90% (59.58–98.21%) | Published84.3%, GPU83.33% |
| SWE-bench Verified | 0 | Unknown | Not yet validly scored | NA |
| Terminal-Bench 2.1 | 0 | Unknown | Not yet validly scored | NA |

The GPQA nightly list completed10 samples in946.951seconds,1.159times its smoke-based forecast. Its CI device job took0.7789hours including setup/server/client, and workflow1.6606hours including build/queue. All ten responses have complete final answers; one is wrong. Dataset metadata198 is not the completed count. This subset cannot establish parity with full-suite references. See `runs/nightly-gpqa-explicit-autoport/audit.json`.

SWE smoke2 (42173587147) is running and Terminal lifecycle-repair smoke2 (42190891568) is building. Terminal42182632490 selected the correct implementation but failed before tasks: its delayed Docker launch survived a30s timeout and the retry collided with port8000. Generic fix `fix/docker-launch-ownership` passed a failing-baseline regression,51 focused tests and83 combined tests/fullhooks; publication2c4a262 is queued for live confirmation. Terminal retry42175367781 failed before tasks because its dispatch omitted the implementation and selected stock; GPQA nightly42176391791 had the same omitted argument and is disqualified before score. Corrected replacements preserve all source SHAs/cohorts/limits and explicitly select the autoport. The first Terminal attempt failed before model initialization on a CI fabric-topology fault and has no score. The earlier SWE registration was superseded after an actual tool request exposed HTTP400; it was not cancelled for queueing. Frozen cohorts and sources are in `cohorts.json`, immutable run SHAs in `compute.json`.

No dry run or release has started. This is extended mode,48hours, starting2026-11-06T04:28:49UTC and ending2026-11-08T04:28:49UTC for release start; each release run has12hours. Next rungs depend on valid scores and measured projections. GPQA next server-sized run retains the same10 tasks and conservative C2/B8: C8 admission was rejected because saved content omits separated reasoning. Measured C2 evaluation is946.951seconds; full198 linear estimate5.208hours excludes tail and overhead and is not a final-release projection. Operator decision8 authorizes CI for Docker tasks, GPQA and final dry/release; local Docker remains unavailable. The stock CI early-stop/evidence limitation and proposed completed no-score rehearsal remain an unanswered operator question in `blocked.json`.

## Effort predictions and results

Device hours below are recorded effort costs, not end-to-end CI build/queue time. Unknown savings stay unknown until real workload timing supports a release-level estimate. Each effort's original confirmation and evidence are retained in `efforts.json`.

| Effort | Predicted device hours | Used device hours | Change / problem | Confirmed result |
|---|---:|---:|---|---|
| local-sandbox-runtime | 0.0000 | 0.0000 | none / solved | success: False; result: Backend unavailable; host-side provisioning required.; resolution: Operator decision8 removes local Docker requirement by authorizingCI; environment remains unchanged, no runtime repair claimed. |
| gpqa-local-credential | Unknown | Unknown | — / not_run | Official dataset access confirmed by valid CI smoke; local credential provisioning excluded by operator. |
| agent-capacity-and-contention | Unknown | Unknown | — / not_run | Validity first; no real task timings for ranking speed. |
| source-only-packaging | 0.0000 | 0.0000 | kept / solved | policy: byte-identical selected policy; generator: complete class AST unchanged; host_checks: 4 packaging checks,7 precision tests,existing host contract passed; boundary: Device-dependent imports stubbed; no native live CI/serving validation |
| harness-registration | 0.0000 | 0.0000 | kept / open | host_tests_passed: 110; live_validation: False |
| suite-registration | 0.0000 | 0.0000 | kept / open | host_tests_passed: 7; live_validation: False |
| ci-release-route | 0.0000 | 0.0000 | none / open | benchmarks: Stock local existing-server path resolves exact stage11 matrix; no Docker required; eval_route: Stock per-suite CI workflows omit spec tests; remaining: CI exact early-stop dry-run control/artifact retention not exposed; no route change made |
| local-serving-check | 0.2000 | 0.0921 | none / solved | success: True; registration: full60 autoport;14 published hashes match; synthetic_growing_clients: 2; final_context_tokens: 32768; late_hit_fraction: 0.9965343876604676; client_wall_seconds: 76.469; real_agent_capacity: not yet measured |
| local-host-contention | 0.1000 | 0.0985 | none / solved | success: True; requests_per_wave: 32; idle_tokens_per_second: 985.554416731418; busy_tokens_per_second: 645.0344483642087; host_metric: process CPU/step, not wall/device time |
| nightly-registration-preparation | 0.0000 | 0.0000 | none / solved | source_patches_prepared: 3; tests_per_candidate: 23; ruff: pass; published: False; dispatch: requires valid smoke first |
| local-long-context | 0.1500 | 0.2540 | none / solved | success: True; phase_seconds: [140.41494534397498, 443.72461584897246]; max_prompt_tokens: 262136; generated_tokens: 8; concurrency: 2; preemptions: 0; new_trace_captures: 0; quality: not measured; deterministic continuation stress |
| parser-protocol | 0.1000 | 0.1058 | kept / solved | swe_baseline_http: 400; candidate_http: 200; actual_mini_parser: pass; actual_terminal_parser: pass; exact_generated_token_pairs: [168, 168]; focused_tests: 30; full_host_hooks: pass; ci_score_validity: pending |
| ci-fabric-topology | 0.2167 | 0.0000 | none / open | Host diagnosis excludes mesh alias error; physical channel deficit observed; corrected retry42182632490 is executing on a different runner; topology result pending |
| native-tool-roundtrip | 0.0028 | 0.0005 | none / solved | HTTP200 and valid parsed submissionaction, no commands executed |
| ci-cache-routing | 0.0833 | 0.0528 | kept / open | Pinned cache files identical; full60healthy;2x168tokenIDs exact. CIstartup saving still unmeasured. |
| immutable-harbor-release-source | 0.0000 | 0.0000 | kept / open | Fullhosthooks pass; generic optionalpin locallyintegrated. Model pin published at8a3c949; live pin-hook confirmation awaits next normal rung. |

| nightly-current-base-preparation | 0 | 0 | none / solved | Frozen SWE5/TB10 patches preserve current parser/cache/pins; host checks passed. |
| real-history-capacity-preparation | 0 | 0 | none / open | Two single-turn GPQA histories cannot establish multi-turn C8 capacity. |
| stage12-zero-cost-checker | 0 | 0 | none / open | Seven regression tests pass for prepared repair; installed package remains read-only pending operator decision. |
| explicit-autoport-dispatch | 0 | 0 | none / open | Host resolver proves explicit implementation fixes selection; corrected CI confirmations pending. |
| harbor-model-pin-confirmation | 0 | 0 | kept / open |58 focused tests/full hooks;8a3c949 published, live pin hook pending. |
| gpqa-server-sized-preparation | 0 | 0 | none / solved | Same10 tasks/C8/B8 patch;32 focused tests/full hooks; live admission pending. |
| nightly-resource-preparation | 0 | 0 | none / solved |15 task definition hashes checked; actual assigned-runner admission still pending. |
| prefill-under-load | 0.1500 | 0.1528 | none / solved | Six requests complete;84Ki prefill35.41s, decoder maximum delivery gap2.43s; no speedup claimed. |

Missing original prediction fields are retained explicitly in `efforts.json`; current conservative planning credit is not a retroactive forecast. No unmeasured savings are reported as results.

## Current configuration and measured limits

Full60-layer fresh autoport, HF/tokenizer revision842da3794eaa0b77d5f08bae87a17459d91ff475, context262144, mesh1x4. Runtime publication08ee8790bcbcd9e85cb799593b2bbaf40b785134 matches execution source. Stock plugin ca7a18246e4111629ed6cb4a7d1192002749fccf remains unchanged. Native Gemma reasoning/tool parsers and automatic tool choice are enabled. Served defaults retain temperature1/top_p0.95 with operator top_k32; suite limits/settings are independently recorded in `serving_config.json`.

Smoke and current nightly runs use planned_clients2 and the smallest supported batch at or above it,8. Validated batches are1/8/16/32. Prefix caching, chunked prefill and asynchronous scheduling remain enabled. Decode and all-final prefill use the canonical device sampler; mixed/intermediate prefill may use the accepted stock host sampler. GPQA smoke observed5480 device-decode submissions and2 device-prefill submissions, no host sampling; submissions are not emitted-token counts. Exact real-agent working sets and fastest concurrency remain unmeasured. The GPQA saved histories are single-turn and unsuitable for a multi-turn capacity replay.

Local synthetic C2 traffic reached262136prompt+8output tokens without preemptions or new captures. This establishes a physical envelope, not real-agent capacity or long-context accuracy. Host CPU contention reduced synthetic C32 delivery from985.55 to645.03tokens/s; release hours saved by a remedy cannot yet be priced because real tool generation/CPU shares are pending.

Local synthetic prefill interference is measured in `runs/prefill-under-load/RESULT.md`:4K/32K/84Ki probes took1.43/9.52/35.41seconds to first token while an anchor decoded. Longer probes coincided with up to2.43second stream-event gaps. Short sequential baselines and HTTP event coalescing limit attribution; release cost remains unknown until actual task overlap is available.

The unchanged stage11 TTI benchmark matrix completed8/8 requests in27.121seconds after parser registration. Fresh full-model vLLM headline4096input/128output/B1/C1 completed8/8: mean TTFT752.844ms and decode35.304tokens/s/user. These are separate from agent-suite scores. Device time and rooflines are unknown: live vLLM profiling is prohibited, and host-wall phase estimates are not device measurements. Evidence: `runs/parser-tti-benchmark/result.json`.

Cache routing to existing persistent mounts passed identical-file checks, live environment checks and two exact168-token output pairs; TTI publication8437f18 includes it. CI startup savings are not yet measured. A default-preserving generic Harbor source-pin repair passed full host hooks on defect branch `fix/model-scoped-harbor-revision` at7079fe2 and is locally integrated at eac7465. Model-specific pin e76c762fca5267dfb36e1f900f0a7724ed72e735 is published at8a3c949 after58focusedtests and fullhooks. This is the source observed in the active SWE job; live pin-hook confirmation remains pending on the next normalrung. Transitive package identity is not claimed frozen.

Execution changes are committed locally and logged in the multigoal stage commit manifest. This execution checkout has no remote and is never pushed. Source-only CI publications use the separately authorized checkouts/branch. The compact telemetry packet7ef7a2c1-3dce-40e4-b6d3-0c83a631ac7e exists locally with measured headline values, GPQA subset accuracy and explicit missing measurements. Historical blocked attempts are preserved in Git and setup diagnostics; they are not current completion claims.
