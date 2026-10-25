# Stage Review

Verdict: more-work-needed

Independent initial review of Stage 10 optimized-vLLM for Qwen/Qwen3.8-27B,
as of 2026-10-25 13:16 UTC. Live branch `exp26-qwen38-bringup`, starting HEAD
`eacda58896fae0c866816fc2a5cba9ff5df554c3`. The owner explicitly identified
the final B1/B8/B16 measurements, quality gates and documentation as ongoing.
This verdict identifies that remaining work; it is not evidence of a new
runtime failure or permission to stop the stage.

No concrete runtime correctness defect was found in the current terminal-work
suppression change. The owner must obtain a fresh final review after completing
the required evidence below. This initial review cannot authorize stage closure.

## Required Work

- P1: Complete the final-source primary B1 measurement and comparison.
  Evidence: `before_b1_vllm_result.json` and its command manifest exist, but
  `after_b1_vllm_result.json`, the final B1 command manifest, `comparisons.json`
  and `perf_summary.json` were absent at the review cutoff. The completed
  `after_b32_vllm_benchmark.json` is one concurrent request on a physical B32
  server, despite the runner's generic `single_user_decode` profile label.
  `batch_configurations.json` correctly keeps B1 pending.
  Why this matters: the required headline is exactly 4096 input / 128 output /
  B1 / concurrency 1. The physical B32 result cannot substitute for it.
  Required next step: finish the planned tracking-disabled, profiler-free B1
  run; preserve its raw/normalized artifacts and exact source/environment
  manifest; establish like-for-like before/after settings and report TTFT,
  TPOT, ITL, throughput and `1000 / mean_tpot_ms`. Produce the full-model
  autoregressive comparison and final performance summary. The inspected
  summarizer selects the appropriate Stage 08 4096/128/B1 autoregressive control
  (36.1709028801 tokens/s/user), with the timing-boundary distinction stated.

- P1: Finish and inspect final-source sampling and prompt-correct quality gates.
  Evidence: `readiness_vllm/sampling_tests.log` was still being appended at the
  cutoff and had no final suite outcome. The planned B8 manifest enables explicit
  host compatibility (`QWEN38_HOST_SAMPLING` and seeded-host routing), as in
  Stage 09. The new `tests/run_shared_serving_quality.py` is present, but no final
  Stage 10 shared-suite output/verdict existed. Existing qualitative verdicts
  and full-model controls belong to prior validation.
  Why this matters: a passing prior suite is a control, not the optimized path's
  acceptance evidence. Synthetic lifecycle text does not establish language
  quality. Full sampling in compatibility mode must not be presented as proof
  that performance used device-only sampling.
  Required next step: retain the completed full suite and classify skips or
  failures; collect the planned packaged and shared chat-template outputs;
  inspect actual greedy/sampled text against the matching controls; record
  rendering metadata, a concrete verdict and the mechanical-degeneracy check.
  Keep quality/device-route evidence and host-compatibility sampling scope
  explicit. Resolve any failure before final review.

- P2: Finish tuned B8/B16 acceptance and the final reviewable evidence packet.
  Evidence: `batch_configurations.json` still marks B8 and B16 pending. B8 is
  running at budget 32768 and B16 is planned at budget 65536; neither is yet a
  final validated configuration in this snapshot. There is no Stage 10 README,
  finalized checklist, cleanup certificate or final packet at the review root.
  `doc/agentic_serving.json` deliberately remains `in_progress` and retains old
  primary/CI rows under its current serving section pending replacement.
  Why this matters: inherited batch capability and old metric rows do not
  validate the newly selected per-capacity scheduler settings or final process
  cleanup. Copying archived normalized JSON also leaves original canonical
  `raw_result_file` paths that will be overwritten by subsequent runs.
  Required next step: complete the planned B8/B16 checks; attach exact manifests
  and outputs to each configuration; replace pending status/old final metric
  references; publish a README led by the exact primary shape and final values,
  a completed evidence-backed optimization checklist, anomaly ledger and
  cleanup evidence. Map each archived normalized result to its immutable raw
  counterpart. Write the user-required packet from the supplied exact template.
  Then request a fresh final stage review. Create local checkpoint commits only
  after clean-pass, record the SHAs, and never push.

## Other Concerns

- There is no demonstrated wall-time speedup from the completed B32 runs.
  Primary-at-B32 TPOT changes from 133.159232 to 133.115619 ms; secondary CI TPOT
  from 112.060190 to 112.049889 ms; CI throughput from 237.214519 to 237.238314 tokens/s.
  These are within noise for single runs. Exact terminal-call elimination proves
  removal of unused work. It does not prove improved end-to-end latency. The
  current anomaly ledger correctly declines a speedup claim. The final report
  must preserve this distinction and must not attribute Stage 09's scheduler
  improvement to Stage 10.
- The configured B32 prefix pool does not establish the full planning target
  of 32 conversations at 32768 tokens each. The current measured workload is
  32x4130 followed by 33-token appends. The existing ledger describes the
  configured-layout shortfall and explicitly avoids calling it a universal
  physical model limit. Retain that limitation; native advertised context 262144
  is preserved and is a separate capability.
- The reduced terminal-suppression probe compares exact final logits, sampled
  tokens, three decode outputs and all cache tensors, including an inactive
  peer. Its first decode is teacher-forced with token 100, so this probe alone
  is not a direct proof that the preceding sampled token feeds decode step 0.
  The production code, unchanged full-model split-feedback implementation,
  inherited stale-input/page-growth controls and current serving replay cover
  the separate feedback contract. Do not describe the reduced probe more
  broadly than it measures.

## Hard-Check Gaps

- Runner profile names alone do not identify physical server capacity. The
  B32 single-request result is labeled `single_user_decode`; its command manifest
  is needed to identify `max_num_seqs=32`. Final tables must include server
  capacity beside workload concurrency.
- Passing lifecycle status alone would miss output anomalies: raw synthetic
  SSE output includes replacement characters and prompt-byte prefixes. The
  previous-stage raw-output comparison below supplies the relevant control.
- This initial review did not execute the stage checker or any model test. The
  final stage check result, with advisory/critical rows classified, still belongs
  in the owner's closure evidence. A scripted pass cannot replace the missing
  final metrics and quality review.

## Anomaly Ledger

- Observed anomaly: generic potential-corruption allocation warning with live traces.
  Evidence: `after_b32_server.log:182`.
  Affected path: allocations during the tracking-disabled performance run.
  Control or comparison: identical runtime source hashes in
  `after_b32_tracked_command.json`; tracking 1 and skip-program-cache 0; 112
  requests, all four signatures, unchanged capture count and no retirement.
  Likely subsystem: generic allocator warning, not a detected tracker violation.
  Investigation performed: compared manifests/current hashes, inspected
  generator preparation and raw mixed-traffic events, checked the absence of
  newly introduced allocation exemptions.
  Resolution: controlled within the exercised variants; retained in
  `anomaly_ledger.md`.

- Observed anomaly: nonlinguistic text and decoded prompt bytes in synthetic SSE fixtures.
  Evidence: `after_b32_lifecycle.requests.json`, first137/138/139 and long/prefix cases.
  Affected path: deliberately synthetic low-token-ID completion requests.
  Control or comparison: exact text equality against
  `readiness_vllm/full_b32_budget131072_lifecycle.requests.json` for all seven
  checked cases: first137/138/139, chunked8195/12291, prefix cold/hit4130.
  Likely subsystem: synthetic token fixture and completion/detokenization behavior;
  no Stage 10 regression established.
  Investigation performed: read first SSE chunks, complete text and usage;
  compared the seven raw outputs directly.
  Resolution: controlled against prior serving; not qualitative evidence.

- Observed anomaly: maximum mixed-traffic SSE pause remains 31.916 s.
  Evidence: `after_b32_lifecycle.json`, versus 31.959 s in the prior same workload.
  Affected path: large fixed-B32 eager prefills while another request decodes.
  Control or comparison: unchanged scheduler settings, physical 1024-token
  prefill chunks and full 64-layer source; the new candidate only omits terminal work.
  Likely subsystem: full-model prefill cost and scheduling boundary, consistent
  with retained prior timeline evidence.
  Investigation performed: verified actual concurrent prefill submissions,
  continuation offsets, chunk count and before/after latency metrics.
  Resolution: controlled limitation; no low-latency service bound or material
  speedup claimed.

- Observed anomaly: initial probe environment failure and pre-model mesh-open failure.
  Evidence: `probe_cache_environment_failure.log`, `probe_mesh_open_failure.log`,
  `recovery_list_before.log`, `recovery_reset.log`, `recovery_list_after.log`,
  `recovery_mesh_smoke.log` and work log.
  Affected path: probe startup before model validation.
  Control or comparison: corrected environment, bounded recovery and subsequent
  successful reduced control/full 64-layer B32 serving.
  Likely subsystem: environment setup and Ethernet initialization.
  Investigation performed: inspected retained failure/recovery descriptions
  and successful follow-up evidence; no hardware commands run by this reviewer.
  Resolution: fixed/recovered by the stage owner.

- Observed anomaly: watcher ACTIVE_ETH image exceeds kernel configuration capacity.
  Evidence: `terminal_suppression_watcher.log`; 29264 bytes versus 26624 bytes.
  Affected path: watcher mesh initialization.
  Control or comparison: prescribed `TT_METAL_WATCHER_DISABLE_ETH=1` retry;
  `terminal_suppression_watcher_noeth.log` and matching JSON pass.
  Likely subsystem: watcher instrumentation capacity.
  Investigation performed: inspected scoped retry and exact-state result.
  Resolution: controlled tooling limitation; compute watcher coverage exists,
  Ethernet watcher coverage does not.

## Scope Inspected

- Goal/skill paths: parent's explicit Stage 10 contract, including user override
  to 4096/128/B1/concurrency1; installed 0.1.21 skills `stage-review`,
  `model-bringup` startup, `vllm-integration`, `optimize`, `agentic-serving`,
  `tt-device-usage`, `tt-enable-tracing`, `qualitative-check`; checkout
  `tech_reports/AdvancedPerformanceOptimizationsForModels/TraceCorrectness.md`.
- Artifact paths: Stage 10 work log, topology audit, checklist, batch configs,
  anomaly ledger, terminal-suppression controls/logs, B1 baselines, B32
  before/after raw and normalized benchmarks, command/source manifests,
  tracked server events and lifecycle raw requests; current sampling log;
  context/agentic contracts and selected precision; relevant Stage 09 README,
  runtime audit, lifecycle control, qualitative verdicts/full-model controls;
  Stage 08 autoregressive performance control.
- Code paths: full current adapter and relevant generator construction,
  preparation, prefill, continuation, sampling, decode and teardown code;
  Stage 10 diff; `probe_prefill_terminal_suppression.py`,
  `test_serving_prefill_contract.py`, `run_serving_lifecycle.py`,
  `run_shared_serving_quality.py`, `summarize_optimized_serving.py`,
  `serve_vllm.sh`.
- Commands run: read-only `git status/diff/rev-parse`, `rg`, `cat`, `sed`,
  `tail`, directory listing and small standard-library Python JSON/hash/output
  analyses. The only write is this requested review report. No TTNN imports,
  tests, server launches, devices, hardware tools, resets or profiling.
- Independently verified: before/after B32 argv and selected environment are
  equal; only generator/adapter source hashes differ; every after-manifest
  runtime source hash matches the current checkout. Generator SHA256:
  `91c8ad90268bcbb42185e9c3ad4cb143d798917316c93ba7b464410812ccacbb`;
  adapter SHA256:
  `315bd0fe2ac27fb0c9cbfc6efa655664ddee53bfb8c880d0c10d9b9f2b746307`.
- Raw event findings: 904 events, one worker PID 1655993, 672 decode submissions,
  112 releases, 59 prefill pieces (35 suppress terminal, 24 retain it), 30 prefill
  submissions; captures 4 throughout, retirements 0, host-logits reads 0. Actual
  starts 4096/8192/12288 and final 3-token tails occur with another active decode
  slot. Initial/final capture signatures map to the same four trace IDs.
  Previous-prefix reuse is 131072/132160 = 0.99176755448.
- Source findings: terminal suppression changes no cache update, persistent
  allocation, dtype or trace signature. Final-emitting pieces still compute
  terminal logits. Off-grid bridge decode remains terminal-capable as declared.
  Generator-wide terminal-capable preparation precedes capture; persistent
  token/position/RoPE/page-table/sampling bindings and nonblocking decode/sampler
  replay remain. Deferred output reads transfer one sampled-ID replica.
  Host-logit compatibility requires an explicit opt-in absent from performance
  manifests. No new acknowledgment or allocation scope was found.

## Residual Risk

- The worktree and evidence were live during review. Findings apply to the
  recorded source hashes and artifact snapshot, not later edits or uncollected
  outputs. Final-source manifests and fresh final review are required.
- Native maximum-context capacity is inherited from unchanged Stage 08; the
  current runs validate affected serving paths, not a new maximum-context
  accuracy claim. Reduced exact controls use real layers 0/3 and B2; full 64-layer
  lifecycle and final sampling/quality provide the complementary serving scope.
- Diagnostic tracking, event logging and watcher timings are not headline
  performance. No serving device-time or roofline claim can be established here.
- Current remaining work is completion of already planned gates and evidence,
  not a request for additional broad sweeps, profiling or a long soak.
