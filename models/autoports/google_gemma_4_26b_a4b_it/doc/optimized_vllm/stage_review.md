# Stage Review

Verdict: clean-pass

Independent final review of stage 10, optimized-vLLM,
`google/gemma-4-26B-A4B-it`, on the live tt-metal worktree based on
`58b7d77f12`. Runtime sources match `candidate_source_manifest.json`.
The vLLM checkout is unchanged and clean at
`7f72b1c6e905f5137fe3377f2e7b42738d3f271d`.

## Required Work

None. The pending runtime gates from `stage_review_initial.md` are complete.
No actionable correctness defect was found in the implementation change.

The pass covers removal of an eager public-token copy dispatch, preservation
of serving correctness, and the accurately limited final performance report.
It does not certify a decode speedup: final warmed decode is effectively flat.
Post-review local checkpoint commits and their work-log entries belong to the
stage owner's normal closure sequence.

## Other Concerns

- Primary before/after measurements are one request each. Their percentiles
  and lower observed TTFT do not establish statistical or causal improvement.
  The final report states this explicitly and excludes the first-use anomaly.
- Shared all-vocabulary chat logprobs skips under the server's configured
  maximum of 20; supported logprob cases pass. Constrained/logprob requests use
  explicit host compatibility, while the measured greedy path remains device
  sampled. Stochastic device top-k remains capped at 32, as documented.
- Full Ethernet Watcher instrumentation does not fit this configuration.
  Compute-device Watcher coverage passed with Ethernet instrumentation disabled.
  The report does not claim Ethernet coverage or a general memory soak.

## Hard-Check Gaps

No unresolved required gate for this bounded change.

- The real plugin loads all 30 model layers on mesh 1x4, with
  `sample_on_device_mode=all`, `trace_mode=decode_only`, async scheduling,
  `max_num_seqs=32`, and `max_model_len=262144`; these are recorded in
  `after/server.log`, not inferred only from launch JSON. The measured engine
  environment has no reduced-layer override or serving profiler enabled.
- The local sampler wrapper delegates canonical sampling/penalties, then
  copies to the preallocated public token tensor (`tt/generator.py:24-49`).
  Binding releases old traces first; exact-shape warmup precedes capture
  (`tt/generator.py:347-405`). Model and sampler replays remain nonblocking.
  The plugin queues the deferred read after sampler replay and before the
  next submission; the adapter returns a submission-shaped tensor rather
  than consulting a later batch size during host finalization.
- `adapter_queued_reads.json` and its test source prove two decode/read pairs
  are submitted before the first wait, return distinct correct token values
  `[2, 47]`, use independent host storage, and preserve persistent device
  tensor identity and unchanged-state refresh counts. A later B1-to-B2 rebind
  with only slot 1 active matches the standalone control, and the previous
  host outputs retain their values and B1 shape. The allocation-tracked run
  retains program-cache checks and closes normally.
- The manual changed-page test is explicitly limited to unused columns.
  Actual allocator growth is covered separately: all nine full-model
  isolated/concurrent/lifecycle requests complete 96 tokens and exactly match
  the prior token reference. Prompt lengths are 31/63/95 and 33/1057/33.
  The unchanged plugin invalidates overlap on nonempty block growth and
  drains pending output before reloading host token/position state.
- Full sampling log: 72 passed, one documented skip, zero failures in
  1119.96 seconds. The host contract log has 84 passes. Changed Python files
  pass repository pre-commit checks. No C++ build is required for these changes.
- All 12 final qualitative outputs were read. Independent JSON comparison
  confirms identical prompts/chat mode and exact equality of all six greedy
  strings against the prior serving control. Sampled responses remain coherent
  and relevant; inherited malformed wording and explicit truncation are
  documented. The shared prompt metadata includes rendered prompts/token IDs
  and pinned HF/full-model controls. Degeneracy and full-context gates pass.
- Raw before/after primary and CI JSON were compared field-by-field against
  `comparison.json`; all recorded metrics and derived decode rates match.
  Primary benchmark commands are identical. All four runs have zero failures,
  with 1/1 primary requests and 32/32 CI requests completed.

| Workload | Metric | Before | After |
| --- | --- | --- | --- |
| 4096 input / 128 output, B1/C1 | TTFT P50/P99 ms | 2642.793 | 2498.285 |
| Same primary | TPOT mean/P99 ms | 19.667 | 19.747 |
| Same primary | ITL P50/P99 ms | 19.597 / 24.302 | 19.598 / 27.104 |
| Same primary | Aggregate output tokens/s | 24.899 | 25.567 |
| Same primary | TPOT-derived decode tokens/s/user | 50.847 | 50.640 |
| 100 input / 100 output, 32-request burst | TTFT P50/P99 ms | 14702.597 / 14703.811 | 14511.987 / 14513.015 |
| Same CI burst | TPOT mean/P99 ms | 681.289 / 752.464 | 681.787 / 752.376 |
| Same CI burst | ITL P50/P99 ms | 677.224 / 729.325 | 677.243 / 735.973 |
| Same CI burst | Aggregate output tokens/s | 39.111 | 39.178 |

Final serving decode is about 98.05% of the recorded 51.6473 tokens/s/user
standalone autoregressive reference. Different prompts and timing boundaries
make this a contextual reference, not a matched speedup experiment.
No serving profiler or additional hardware run is needed for this review.

## Anomaly Ledger

- Observed anomaly: first-use TPOT 17.7136 ms disagrees with mean ITL 19.5620 ms.
  Evidence: `candidate_first_use/vllm_result.json`; benchmark client source.
  Affected path: streamed primary benchmark accounting.
  Control or comparison: warmed baseline and final candidate.
  Likely subsystem: response-chunk accounting; exact first-use cause unproven.
  Investigation performed: recomputed implied 115 versus 127 stream intervals
  and inspected per-chunk ITL versus token-count TPOT calculations. Final
  warmed mean TPOT 19.747422 ms and ITL 19.747424 ms agree.
  Resolution: controlled. First-use number is retained and excluded from claims.

- Observed anomaly: Watcher ACTIVE_ETH image exceeds its configuration buffer;
  initial error cleanup segfaults before model construction.
  Evidence: `batch_cache_watcher.log`, retry log, and result JSON.
  Affected path: Ethernet instrumentation startup.
  Control or comparison: same reduced multirow probe with Ethernet Watcher disabled.
  Likely subsystem: instrumentation image capacity.
  Investigation performed: inspected successful four-device compute Watcher
  checks, exact token controls, and normal retry closure.
  Resolution: controlled with the Ethernet coverage limitation preserved.

- Observed anomaly: allocation-after-trace warning during request lifecycle.
  Evidence: `after/server.log`, `adapter_changed_pages_retry.log`, and
  `adapter_queued_reads.log`.
  Affected path: capture/allocation lifecycle.
  Control or comparison: tracker-backed reduced replay, queueing and rebind
  probes; exact full-server request controls.
  Likely subsystem: generic warning about allocations while a trace exists.
  Investigation performed: inspected persistent tensor ownership and trace
  release/capture order; tracker runs with program-cache checks report no
  unsafe-survivor failure and close normally.
  Resolution: controlled for exercised paths, without an all-shape memory claim.

- Observed anomaly: greedy prose contains “own-contained” and over-hyphenated
  “tiny-brass-heart”; several longer answers end at the output cap.
  Evidence: final qualitative JSON, inherited control, and prompt metadata.
  Affected path: model prose and configured 256-token output limit.
  Control or comparison: all six final greedy strings exactly match prior
  validated serving output; prior full-model controls explain the same wording.
  Likely subsystem: inherited model/precision behavior and generation limits.
  Investigation performed: read all greedy/sampled outputs and compared prompts
  and greedy strings. No new mechanical repetition, gibberish, wrong-language
  drift, or cross-request leakage was observed.
  Resolution: controlled as a serving-regression check, not prose/science certification.

- Observed anomaly: shutdown force-stops EngineCore and reports nanobind leaks.
  Evidence: `after/server.log`, `cleanup.json`, inherited shutdown evidence.
  Affected path: abort/timeout-zero API and interpreter teardown.
  Control or comparison: unchanged counts of 105 instances, 973 types, and
  4455 functions in the inherited shutdown; all three owned serving PIDs gone.
  Likely subsystem: binding-reference teardown and runner shutdown policy.
  Investigation performed: inspected final shutdown and cleanup artifacts;
  subsequent reduced probe opens and closes normally.
  Resolution: controlled. No graceful device teardown or per-request leak claim.

## Scope Inspected

- Goal/skill paths: supplied stage-10 contract; installed stage-review,
  optimize, vllm-integration, tt-enable-tracing, qualitative-check; TT review
  core/router and trace/serving domain checks.
- Artifact paths: stage README, checklist, work log, AutoDebug/AutoFix,
  runtime/anomaly audits, source manifest, environment records, before/after
  raw and normalized metrics, sampling logs, final validation, qualitative
  outputs/controls, request lifecycle, reduced adapter/queueing/Watcher
  evidence, context/degeneracy gate logs, formatting logs, and cleanup.
- Code paths: complete model generator and vLLM adapter, changed contract and
  queued-read tests, common sampler capture/replay, plugin page-growth and
  async submission/finalization, request harness, benchmark accounting, and
  preallocated slice behavior.
- Commands run: read-only git status/diff/revision/whitespace checks, local
  file inspection, SHA-256 comparisons, and small JSON comparison scripts.
  This reviewer ran no hardware, server, reset, profiler, or test command and
  modified only review documents.

## Residual Risk

- One primary request per side cannot establish a small latency improvement;
  final decode is effectively unchanged. Low-level device timing is intentionally
  absent under the serving-stage profiler restriction.
- The tested lifecycle is bounded. It is not universal long-context or long-soak
  memory verification. Context and persistent tensor capacity are unchanged.
- Stochastic top-k and logprob compatibility limits, output wording/caps,
  Ethernet Watcher coverage, and shutdown behavior remain disclosed.
- These limits do not leave required work for the reviewed stage change.
