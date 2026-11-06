# Stage 12 — active

| Suite | functional_at | Evaluation hours/run | Score (95% Wilson interval) | Reference |
|---|---:|---:|---|---|
| GPQA Diamond | 2 | 0.04538 | 2/2, 100% (34.24–100%) | Published84.3%, GPU83.33% |
| SWE-bench Verified | 0 | Unknown | Not yet validly scored | NA |
| Terminal-Bench 2.1 | 0 | Unknown | Not yet validly scored | NA |

The GPQA smoke is a two-sample subset, too small to establish parity with full-suite references. Its CI device job took0.5772hours including setup/server/client; evaluation alone took163.384seconds. Actual sample JSONL contains2 records even though lm-eval's dataset metadata says198. See `runs/smoke2-r1_gpqa_diamond/audit.json`.

GPQA nightly10 (42176391791), SWE smoke2 with native parser registration (42173587147), and Terminal smoke2 infrastructure retry (42175367781) are active. The first Terminal attempt failed before model initialization on a CI fabric-topology fault and has no score. The earlier SWE registration was superseded after an actual tool request exposed HTTP400; it was not cancelled for queueing. Frozen cohorts and sources are in `cohorts.json`, immutable run SHAs in `compute.json`.

No dry run or release has started. This is extended mode,48hours, starting2026-11-06T04:28:49UTC and ending2026-11-08T04:28:49UTC for release start; each release run has12hours. Next rungs depend on valid scores and measured projections. Current next-GPQA estimate is816.92seconds evaluation plus measured CI overhead and margin, not a final-release projection. Operator decision8 authorizes CI for Docker tasks, GPQA and final dry/release; local Docker remains unavailable. The stock CI early-stop/evidence limitation and proposed completed no-score rehearsal remain an unanswered operator question in `blocked.json`.

## Effort predictions and results

Device hours below are recorded effort costs, not end-to-end CI build/queue time. Unknown savings stay unknown until real workload timing supports a release-level estimate. Each effort's original confirmation and evidence are retained in `efforts.json`.

| Effort | Predicted device hours | Used device hours | Change / problem | Confirmed result |
|---|---:|---:|---|---|
| local-sandbox-runtime | 0.0000 | 0.0000 | none / solved | success: False; result: Backend unavailable; host-side provisioning required.; resolution: Operator decision8 removes local Docker requirement by authorizingCI; environment remains unchanged, no runtime repair claimed. |
| gpqa-local-credential | Unknown | Unknown | — / not_run | Operator decision8 authorizes gated dataset on qb2-ci; smoke job42166117505 building. No local credential requested. |
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
| ci-fabric-topology | 0.2167 | 0.0000 | none / open | Host diagnosis excludes mesh alias error; physical channel deficit observed; recovery pending retry42175367781 |
| native-tool-roundtrip | 0.0028 | 0.0005 | none / solved | HTTP200 and valid parsed submissionaction, no commands executed |
| ci-cache-routing | 0.0833 | 0.0528 | kept / open | Pinned cache files identical; full60healthy;2x168tokenIDs exact. CIstartup saving still unmeasured. |
| immutable-harbor-release-source | 0.0000 | 0.0000 | kept / open | Fullhosthooks pass; generic optionalpin locallyintegrated. Modelpin/actualCIcheckout confirmation pending. |

## Current configuration and measured limits

Full60-layer fresh autoport, HF/tokenizer revision842da3794eaa0b77d5f08bae87a17459d91ff475, context262144, mesh1x4. Runtime publication08ee8790bcbcd9e85cb799593b2bbaf40b785134 matches execution source. Stock plugin ca7a18246e4111629ed6cb4a7d1192002749fccf remains unchanged. Native Gemma reasoning/tool parsers and automatic tool choice are enabled. Served defaults retain temperature1/top_p0.95 with operator top_k32; suite limits/settings are independently recorded in `serving_config.json`.

Smoke and current nightly runs use planned_clients2 and the smallest supported batch at or above it,8. Validated batches are1/8/16/32. Prefix caching, chunked prefill and asynchronous scheduling remain enabled. Decode and all-final prefill use the canonical device sampler; mixed/intermediate prefill may use the accepted stock host sampler. GPQA smoke observed5480 device-decode submissions and2 device-prefill submissions, no host sampling; submissions are not emitted-token counts. Exact real-agent working sets and fastest concurrency remain unmeasured. The GPQA saved histories are single-turn and unsuitable for a multi-turn capacity replay.

Local synthetic C2 traffic reached262136prompt+8output tokens without preemptions or new captures. This establishes a physical envelope, not real-agent capacity or long-context accuracy. Host CPU contention reduced synthetic C32 delivery from985.55 to645.03tokens/s; release hours saved by a remedy cannot yet be priced because real tool generation/CPU shares are pending.

The unchanged stage11 TTI benchmark matrix completed8/8 requests in27.121seconds after parser registration. Fresh full-model vLLM headline4096input/128output/B1/C1 completed8/8: mean TTFT752.844ms and decode35.304tokens/s/user. These are separate from agent-suite scores. Device time and rooflines are unknown: live vLLM profiling is prohibited, and host-wall phase estimates are not device measurements. Evidence: `runs/parser-tti-benchmark/result.json`.

Cache routing to existing persistent mounts passed identical-file checks, live environment checks and two exact168-token output pairs; TTI publication8437f18 includes it. CI startup savings are not yet measured. A default-preserving generic Harbor source-pin repair passed full host hooks on defect branch `fix/model-scoped-harbor-revision` at7079fe2 and is locally integrated at eac7465. Its model-specific pin and CI confirmation remain pending; transitive package identity is not claimed frozen.

Execution changes are committed locally and logged in the multigoal stage commit manifest. This execution checkout has no remote and is never pushed. Source-only CI publications use the separately authorized checkouts/branch. The compact telemetry packet7ef7a2c1-3dce-40e4-b6d3-0c83a631ac7e exists locally with measured headline values, GPQA subset accuracy and explicit missing measurements. Historical blocked attempts are preserved in Git and setup diagnostics; they are not current completion claims.
