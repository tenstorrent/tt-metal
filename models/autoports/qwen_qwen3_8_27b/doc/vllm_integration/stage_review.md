# Stage Review

Verdict: clean-pass

Independent review of stage 9, Qwen/Qwen3.8-27B, completed 2026-09-14.
The reviewer inspected source and existing artifacts, wrote this report, and ran
small host artifact analyses. The reviewer did not import TTNN, execute the model,
start a server, operate hardware, change implementation files, or delegate review.

## Required Work

- None. The sensitive concurrent-request control resolves the review's original
  weak page-growth oracle finding. Final sampling, native quality/performance,
  seed/trace/cache controls, context gate, and process cleanup are supported.
## Other Concerns

- The final canonical sampling profile is **73 passed** in explicit optional
  all-host compatibility mode, at max sequences 32. Native supported sampling
  has separate final-source stochastic, greedy, qualitative, and benchmark
  evidence. Mixed host/native batches do not promise the same RNG algorithm;
  the README distinguishes all three compatibility modes.
- Primary native S128/G128/N1, max sequences 1, concurrency 1, temperature 0,
  ignore EOS, and one unmeasured same-shape warmup measures **85.8247 ms TTFT**,
  **24.20998 ms mean TPOT**, **41.3053 tokens/s/user**, and **40.4966 aggregate
  output tokens/s**. Raw and normalized JSON agree. One measured request makes
  TTFT/TPOT P99 a single observation, not a population tail estimate.
- Secondary native S100/G100/N32 completes 32 requests and 3200 output tokens:
  **121.6156 aggregate tokens/s**, **201.6713 ms mean TPOT**. Its 4.9586 derived
  tokens/s/user is correctly secondary. This burst has no warmup and disables
  vLLM chunked prefill; it is not comparable to the primary single-user workload.
- Sampled haiku meter is imperfect (5/7/4); the earlier selected-policy TT
  control also misses meter (5/7/6), while HF and current greedy serving satisfy
  5/7/5. This is a controlled instruction-quality limitation, not a universal
  quality pass or proof of the cause of every sampled word choice.
- Raw logs, vectors, and JSON are ignored local artifacts. Their persistent
  paths and hashes exist now; copying only the source commits will not copy
  the complete runtime evidence.

## Hard-Check Gaps

- The installed stage checker establishes context and mechanical-degeneration
  gates; it does not establish answer completeness, correct haiku meter, or
  cache ownership. Independent text/source inspection and targeted controls
  supply those additional checks.
- Concurrent HTTP submission order does not directly observe scheduler row
  placement. The direct adapter control forces a drained permutation, and the
  standalone logit control explicitly swaps slots 0/1. The serving full-vector
  logit comparison covers prefill; decode is additionally exercised by shared
  logprobs and the 100-token synchronous control.
- Short lifecycle cache occupancy returns to zero. This is not an exhaustive
  TT DRAM allocator or arbitrary-shape long-duration memory audit. Existing
  allocation ownership and tracker evidence are sufficient for the changed
  paths, with their limits stated in the README.
- Maximum context remains 262144. This stage verifies that served configuration,
  cache geometry, and non-aligned requests preserve it; maximum-length execution
  is inherited from the unchanged full-model/datatype evidence. Batch 32 short
  requests and batch 1 maximum context are separate capacity points.

## Anomaly Ledger

- Observed anomaly: reduced active output streams were only token 220/spaces.
  Evidence: `adapter_device_final.json`, `reduced_concurrent.json`, and
  `reduced_concurrent_control_final.json`.
  Affected path: active-request async feedback and allocator-driven page growth.
  Control or comparison: final `readiness_vllm/full_concurrent_control.json`.
  Likely subsystem: reduced-model numerical sensitivity, not demonstrated cache
  corruption.
  Investigation performed: independently compared all 12 full-model text and
  token arrays; four diverse chat streams match native async, logprob-forced
  synchronous greedy, and reordered repeats across multiple page boundaries.
  Structural device-table/position/inactive-state controls remain useful.
  Resolution: fixed evidence gap.

- Observed anomaly: mixed-parameter seeded reproduction failed once; native
  seeds could rewind on authoritative batch changes.
  Evidence: `AUTOFIX_native_sampling.md`, seed host regressions, and retained
  failed/targeted sampling logs.
  Affected path: sampling resets, remaps, companion changes, and mode changes.
  Control or comparison: `native_seed_continuity.json`, final 73-test all-host
  suite, and `adapter_device_after_seed_fix.json`.
  Likely subsystem: seed anchoring plus shared batch-level RNG backend selection.
  Investigation performed: reviewed absolute-output-position seed anchoring,
  preserved steady seeds, int32 headroom, and host-to-native refresh. Independently
  compared six native seeded continuations. Final device evidence has 68 model
  and sampling replays, two authoritative token/position/RoPE/seed refreshes,
  six changed-page refreshes, and 14 distinct owned snapshots.
  Resolution: native seed defect fixed; cross-backend reproducibility controlled
  by explicit all-host compatibility for the shared suite.

- Observed anomaly: first-use packed-prefill buffers conflicted with existing
  trace scratch; generic allocation warnings remain in final server logs.
  Evidence: `AUTOFIX_prefill_sampling_trace.md`, original tracker failure,
  `full_host_profile_server.log:166`, and `final_native1_server.log:165`.
  Affected path: prefill packing and subsequent decode replay.
  Control or comparison: tracker-enabled repaired representative smoke and final
  adapter probe; partial tracked all-layer evidence and successful final runs.
  Likely subsystem: persistent program-buffer versus trace-scratch lifetime.
  Investigation performed: inspected signature/cache invalidation and release
  before unseen packing; inspected ownership of transient prefill intermediates.
  Final uninstrumented full-suite success is not treated as a complete tracker
  audit. Generic allocation alone does not establish retained unsafe buffers.
  Resolution: concrete persistent-buffer defect fixed; generic warning controlled
  with explicitly limited ownership/tracker evidence.

- Observed anomaly: prefill RoPE metadata and synchronous/host-async output
  boundaries initially raised errors.
  Evidence: prefill, nonoverlap-read, and host-async AutoFix reports and red/green
  host logs.
  Affected path: plugin/adapter interface translation.
  Control or comparison: final real-plugin host tests, full shared suite, and
  native/synchronous exact-token serving control.
  Likely subsystem: scalar tensor metadata and host/device output formatting.
  Investigation performed: inspected zero text RoPE offsets, minimal one-replica
  device reads, completed-host-logit handling, and plugin finalization.
  Resolution: fixed.

- Observed anomaly: worker teardown lacked explicit mesh closure; startup
  cancellation could orphan EngineCore; earlier Ethernet startup failures and
  nanobind exit warnings were visible.
  Evidence: worker/startup AutoFix reports, guard tests, archived shutdown logs,
  `final_process_audit.json`, `final_device_list.log`, `final_mesh_smoke.log`.
  Affected path: worker, runner descendants, and device lifecycle.
  Control or comparison: normal/exceptional worker close, guarded final exits,
  empty ownership, and successful exact ring reopen without final reset.
  Likely subsystem: explicit lifetime management; earlier heartbeat causality
  remains confounded by external telemetry/firmware behavior.
  Investigation performed: reviewed drain/model-close/mesh-close ordering and
  marker/start-time/pidfd ownership. Raw closure markers support the audit.
  Resolution: lifecycle defects fixed; historical hardware cause and nanobind
  warnings controlled by documented successful cleanup/recovery, not asserted
  to be fully diagnosed.

- Observed anomaly: shared 256-token quality requests truncate reasoning or
  answers; raw reasoning markers and haiku-meter errors are visible.
  Evidence: all 12 shared completions, eight extensions, matching HF/selected-TT
  controls, and `qualitative_review.md`.
  Affected path: prompt formatting, generation budget, and answer quality.
  Control or comparison: six exact rendered prompts/token-ID sequences and
  prior selected-policy/HF text; final greedy and seed-42 sampled extensions.
  Likely subsystem: checkpoint template, output budget, and model instruction
  quality rather than demonstrated stale-token behavior.
  Investigation performed: read every serving completion in full; all extensions
  stop coherently, four greedy prefixes match, translation/science remain correct,
  and three inspected Fibonacci functions execute correctly. Sampled extensions
  are explicitly new seeded requests, not claimed continuations of unseeded
  originals. Prior TT meter failure and current sampled failure are retained.
  Resolution: truncation/template behavior controlled; meter limitation controlled.

- Observed anomaly: benchmark CLI failed on absent wheel distribution metadata.
  Evidence: metadata failure log, CLI diff/version check, successful raw primary
  and CI benchmark records.
  Affected path: CLI setup before HTTP benchmark execution.
  Control or comparison: narrow `PackageNotFoundError` fallback to package version.
  Likely subsystem: source-checkout packaging metadata.
  Investigation performed: inspected the seven-line diff and completed retries;
  no model execution or sampling path changed.
  Resolution: fixed.

## Scope Inspected

- Goal/skill paths: this directory's `goal_contract.md`; installed
  `tt-model-bringup/0.1.4/skills/{stage-review,vllm-integration,tt-device-usage}`
  and model-bringup startup; repository qualitative-check and supporting stage
  skill texts. Installed allocator-growth and lifecycle requirements govern.
- Artifact paths: final README/work log/qualitative review; AutoDebug/AutoFix
  reports and host/device logs; all principal `readiness_vllm/` sampling, native
  seed, concurrent, logit, lifecycle, qualitative, benchmark, run-config and
  cleanup artifacts; stage 8 raw qualitative/context/precision controls.
- Code paths: `tt/generator_vllm.py`, generator changes and relevant model/cache
  methods; stage test/probe/guard helpers; sibling plugin `platform.py`,
  `model_runner.py`, `async_decode.py`, `worker.py`, sampling-capability and worker
  tests; `vllm/entrypoints/cli/main.py` and reference adapter/contract code.
- Commands run: read-only `rg`, `sed`, `cat`, `nl`, `find`, git status/diff/HEAD
  inspection; Python JSON/hash/token/prompt/metric comparisons and restricted
  execution of already-inspected generated Fibonacci functions. Both repos'
  `git diff --check` pass. Runtime tests and hardware checks were performed by
  the stage owner; this reviewer inspected their evidence.
- Provenance: both branches are `mvasiljevic/qwen38-full-bringup`. Review started
  before the operator's history cleanup; final tt-metal base is
  `125d29dda46fb77a409e6401da6cfc533341aea1`, sibling vLLM base is
  `03fa3af2e15b5f8dc07cbaa67d92f979aa00be11`. All source hashes in final B1, B32,
  and full-host run configs match the reviewed files. All 23 artifact hashes and
  byte counts in `readiness_vllm/final_artifact_manifest.json` verify.

## Residual Risk

- This is a stage-9 serving integration pass with finite correctness and quality
  evidence, not a release-scale evaluation or long-running memory guarantee.
- Prefix caching remains disabled. Native sampling supports its documented
  subset; optional compatibility covers broader API sampling parameters.
- Raw thinking markers and controlled haiku-meter errors remain visible in
  diagnostic output. No unresolved serving-specific regression was established.
- Local commits must isolate stage changes from operator-owned root
  `AUTODEBUG.md` and `PIPELINE_BLOCKERS.md`, then record both repository SHAs in
  the work log. They must not be pushed by this stage.
