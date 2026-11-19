# Stage Review

Verdict: pass-with-carried

Stage 10 optimized-vLLM, `ornith-ai/Ornith-1.5-9B`, independently reviewed on the live `main` worktree based on `fffd4918cd09d2b58b0f1cbcdba4a8ddefc727ab`. This round uses the explicit latest user instruction in `review_round1_brief.txt`: the reached cap permits nonblocking carries now. It does not waive correctness or required gates. No blocking correctness, accuracy, stale-evidence, or missing-gate defect was found in the inspected final path. No hardware, servers, profiling, tests, or implementation edits were performed by this reviewer.

## Required Work

None from this review. The owner still performs the already-planned local checkpoint commits and records their SHAs before declaring the stage complete. This is finalization after an accepted review, not a request for another review or device run. Never push from this stage.

## Other Concerns

- **Carry the untried optimization inventory without claiming rejection or optimality.** `work_log.md` under `Not tried (stop rule)`, `checklist_draft.md`, and `rule1_disposition_draft.md` identify alternate grids/repadding/K values, collective/residual families, phase placement, sampler epilogues/history construction, dense batched prefill, and query-only padding reduction. Rule 3 fired at 05:07:49 UTC; no further experiment is required for this review. Gains remain unknown. Copy these notes into `Carried to next stage` when recording this verdict.
- **Preserve the latency scope.** The final primary row is one warmed 128-input/128-output request at concurrency 1. Its 28.94% TTFT reduction is reproduced in the recorded matched comparison, but the 0.019% decode-rate difference is effectively unchanged and has no confidence interval. CI p99 ITL rises from 73.715 to 85.381 ms. The separate mixed workload reaches a 1225.932 ms maximum pause. The selected-budget bound is specific to the 31-background-decoder sweep, not a universal one-second latency guarantee. The current README makes these distinctions correctly; retain them at handoff.
- **Retain quality-control limits in the next stage.** All 12 final HTTP answers were read directly, as were the six final standalone sampled answers and selected original-HF/story controls. They are coherent answers or capped prefixes, without an unexplained serving-corruption signature. The 20-row blind story scores contain zero pronoun/material-corruption failures and 18 capped incomplete outputs. The HF arm uses original-model batch 10, while HTTP samples are serial; the native/custom sampler control covers only 16 tokens at batch 1. These are useful controls, not proof of population equivalence or complete-answer quality. `doc/eval_readiness/limitations.json` records the inherited thermodynamics and cap limits with `original_model: "same"` and evidence paths.
- **Replace planning assumptions at benchmark/release handoff.** The 0.43-hour eval forecast is a planning estimate. GPQA's 12000-token typical answer extrapolates beyond the measured 6000-token load. Keep the current explicit distinction from executed accuracy evaluations and substitute the actual release plan before scheduling.

## Hard-Check Gaps

- The packaged stage gate exits 1 for advisories rather than zero. Inspection of `final_stage_gate.log`, `final_stage_gate.exit`, and `final_gate_disposition.json` supports the stated disposition: context 262144 validates; P4/P5 pass; the remaining messages are historical 512-token fixture references and optional dense-prefill ratio targets. No dense batching implementation was adopted, and the user's contract makes it optional. This is not a failed mandatory gate. A future checker improvement could separate advisory exit status from failure; no rerun is required here.
- The mixed trace workload deliberately uses a periodic token-ID prompt. Its repetitive token continuation is an expected stress-input behavior, not natural-language quality evidence: `tests/vllm_mixed_trace_probe.py` constructs `[9707 + i % 13 ...]`, and the raw request/response records preserve that pattern. Natural-language quality is assessed separately by the shared suite and matched controls.
- Sampling is 67 passed, 75 skipped, not 142 passed. The raw `sampling_tests.log` retains each skipped case; the scope includes unsupported speculation and opt-in coverage. The required chunk scheduling and changed-position paths have separate mixed/device evidence. No new skipped-case gate is invented by this review.

## Anomaly Ledger

- Observed anomaly: an interrupted page-table transfer could leave device rows changed while the old host snapshot authorized an unsafe retry.
  Evidence: `page_refresh_failclosed_red.log`, `page_refresh_failclosed_focused_host.log`, `page_failure_guard_host_suite.log`, `tests/vllm_page_failure_test.py`, and `tt/generator.py`.
  Affected path: commanded page reload and forward-input remap.
  Control or comparison: injected failure mutates the failing destination; retries with both old and new tables are rejected before further writes. The 53-test focused log and 160-test suite pass; five final guarded device probes pass on the final 23-file manifest.
  Likely subsystem: host/device page ownership metadata.
  Investigation performed: inspected the fail-closed guards in prepare, decode, capture, replay, resident admission, and remap; inspected the failure-injection assertions and final device results.
  Resolution: fixed. Failed state requires generator recreation; normal close/drain remains available.

- Observed anomaly: a device-backed output could be overwritten by another replay before its deferred read was queued.
  Evidence: `tt/generator.py` around `decode_forward`, `_submit_read`, `read_decode_output`, `replay`, and `close`; `guarded_device/vllm_adapter_decode_probe.json`; focused deferred-read tests.
  Affected path: async output lifetime and public standalone replay.
  Control or comparison: unqueued reads reject a subsequent decode/replay before state writes; queued CQ0 transfers preserve prior outputs and permit overlap; prefill/reset transitions require drained submissions.
  Likely subsystem: borrowed persistent output lifetime.
  Investigation performed: inspected ordering and first-shard token transfer, explicit host-logit compatibility, and actual serving route events.
  Resolution: fixed. Raw measured windows contain 127/127 primary, 261/261 capacity, 148/148 P5, and 6511/6511 P4 device decode/async-read events, all with `read_from_device=False` and trace enabled; no captures/releases occur in those windows.

- Observed anomaly: older mixed latency was affected by bytewise SSE consumption.
  Evidence: `mixed_stream_client_anomaly.json`, corrected `tests/vllm_mixed_trace_probe.py`, and both final `mixed_probe.py.source` snapshots.
  Affected path: client latency measurement, not model arithmetic.
  Control or comparison: corrected client requires uncompressed chunked transport and uses chunk-aligned reads; final tracked/untracked runs are source-bound and pass all ten mixed checks.
  Likely subsystem: HTTP client buffering.
  Investigation performed: inspected the corrected stream reader, admission order, raw request tokens/events, and snapshot hashes.
  Resolution: fixed for final metrics. Historical bytewise latency is excluded.

- Observed anomaly: a pre-guard unseeded story changed protagonist pronouns.
  Evidence: `story_control/batched_run/blind/scores.json`, `unblinded_results.json`, HF and HTTP `prompt03_seed*.json`, and `guarded_http_qualitative_review.json`.
  Affected path: sampled text quality.
  Control or comparison: ten matched seeds per side, same prompt/template and sampling recipe, fixed strictly-greater-than-20-percentage-point excess rule, blind scoring before unblinding; zero failures on either side.
  Likely subsystem: stochastic model content; no demonstrated TT excess.
  Investigation performed: read final story text and selected control outputs; independently counted all 20 score flags and 18 incomplete outputs.
  Resolution: controlled with the small-sample/batch-shape limitations above. No new regression claim is supported.

- Observed anomaly: `run_matched_before.py` fingerprint changed during packaging.
  Evidence: `run_matched_before_provenance.json`, `run_matched_before_preformat.py.source`, and `run_matched_before_formatting.patch`.
  Affected path: orchestration helper, not measured model runtime.
  Control or comparison: exact recovered premeasurement source differs only in Black's multiline formatting of the six-name list; baseline restore manifest records successful restoration.
  Likely subsystem: repository formatting hook.
  Investigation performed: independently compared parsed Python ASTs, which are identical, and inspected the exact formatting diff. All 221 inputs hashed by the final serving audit still match disk.
  Resolution: controlled; no measurement invalidation.

- Observed anomaly: canonical mixed/qualitative artifacts changed after promotion for repository packaging.
  Evidence: `canonical_mixed_compaction.json`, `canonical_qualitative_newline.json`, promotion annotations, and the three raw-evidence bundle manifests.
  Affected path: evidence storage representation.
  Control or comparison: labelled audited originals are unchanged; canonical mixed checks, metrics, trace signatures, and request token/text outputs match the full tracked report exactly; qualitative parsed JSON is identical.
  Likely subsystem: file-size and final-newline hooks.
  Investigation performed: independently decompressed and verified every referenced object: historical 893 paths/376 objects, guarded 530 paths/293 objects, metadata 27 paths/27 objects, all with correct SHA-256 and byte lengths.
  Resolution: controlled and losslessly recoverable.

## Scope Inspected

- Goal/skill paths: `goal_contract.txt`, `review_round1_brief.txt`, installed `stage-review`, `vllm-integration`, `optimize`, `tt-enable-tracing`, `tt-device-usage`, `qualitative-check`, and model-bringup startup instructions under `/home/ttuser/ornith-9b/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24/skills/`.
- Artifact paths: final README/work log/checklist/stop disposition; final source/audit/promotion/gate/cleanup and formatting receipts; before `stage10_final_before`, after `stage10_guarded_after`, tracked `stage10_guarded_tracked` run summaries, raw benchmark results, server logs, environments and mixed reports; guarded device probes/full-model comparison; compact/prefill-refresh/attention-input full-stack controls; guarded budget selection, P4/P5; final HTTP and standalone qualitative artifacts, blind story scores and selected HF controls; limitations and prefix-cache exclusion.
- Code paths: all six changed `tt/*.py` files (`generator`, `generator_vllm`, `prefill_contract`, `prefill_ops`, `prefill_plan`, `prefill_runtime`), page-failure/deferred-read tests, mixed probe, and final audit implementation. The prefill controls each contain two 772-comparison cases with exact finite outputs/state and unchanged traces: aligned and awkward tails at 1024/1033, plus final 4096/4105 segments through 32768 and next decode. The final selector retains the 8192 large numerical branch and rejects the failed 6144 candidate.
- Re-derived measurements: raw primary results are TTFT 248.984129 to 176.926293 ms, TPOT 8.701514 to 8.699881 ms, and `1000 / TPOT` 114.922523 to 114.944095 t/s/u. Raw CI capacity is 839.794117 to 969.500991 output tok/s for 100/100/32/concurrency unset. Before/after B1 and B32 server command arrays are identical, including context 262144, budget 4096, mesh and sampling. Full-model comparison contains all 32 layers and three streamed repetitions; median 8.743032 ms/token gives a served/full-model ratio of 0.995065.
- Trace/context coverage: allocation tracking is enabled with program-cache allocations included in the final tracked environment and disabled for performance. The positive control detects a live unsafe allocation. Final mixed workloads retain six initial traces, complete 15 requests of 70 outputs at logical lengths 129/130/131/8193/32769, change continuation offsets/order/modes, and overlap prefill with decode. Prefix caching retains the documented `recurrent_state` exclusion, not a claimed prefix-cache pass. Context remains 262144; P5 exercises through 131072 tokens and nonaligned 127/128, with all requests completed.
- Commands run: read-only `git diff HEAD`, `git status`, `rg`, `cat`, `sed`, `tail`, and small Python analyses of JSON, logs, ASTs, SHA-256 and gzip objects. Some exploratory file-name reads returned missing paths; the actual corresponding logs/tests were located and inspected afterward. Only this requested report was written. No tests or gates were rerun.

## Residual Risk

- This is an inspection-based review of recorded runs, not an independent device reproduction or release-accuracy certification. The next benchmark stage owns its accuracy results.
- The final serving default is supported by raw route events, changed/unchanged state tests, allocation tracking, full-stack exactness, and real HTTP/load results. Deliberate host compatibility remains outside the measured default windows.
- Evidence supports the tested request/mode transitions and bounded load. It does not establish every possible scheduler history, statistical performance confidence, global kernel optimality, or a universal latency SLA.
- Record this verdict and the four carry groups above in the work log, then checkpoint only stage-owned changes locally. Cleanup evidence reports no remaining vLLM/EngineCore/device owners, no resets, an unchanged plugin, and no push; local commit SHAs were intentionally pending at review time.
