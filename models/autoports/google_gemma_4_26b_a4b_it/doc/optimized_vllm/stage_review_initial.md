# Stage Review

Verdict: more-work-needed

Initial independent review of stage 10, optimized-vLLM,
`google/gemma-4-26B-A4B-it`, on the live worktree based on tt-metal
`58b7d77f12`. The vLLM checkout is clean at
`7f72b1c6e905f5137fe3377f2e7b42738d3f271d`. Candidate source hashes match
`candidate_source_manifest.json`. Final candidate checks are still running;
this is not a final rejection of the implementation.

## Required Work

- P2: Complete the candidate's final serving gates before stage closure.
  Evidence: `checklist.md:9-24` still records final benchmarks, full sampling,
  qualitative, allocator/lifecycle, and default-path evidence as pending;
  `context_contract.json:430` marks candidate validation in progress.
  Why this matters: inherited stage-09 output cannot validate the changed
  sampler trace's final serving behavior. The available fresh reduced
  probes and 84 host tests validate useful subsets, not the complete stage.
  Required next step: finish the already-running/planned full-model checks,
  inspect prompt-correct greedy and sampled outputs against existing controls,
  preserve all required metrics and request results, then request final review.
  During this review the final warmed primary and nine request-lifecycle
  cases completed and were inspected; final CI, full sampling, qualitative
  review, queued-read proof, and cleanup remain pending.
  This is pending authorized work, not a discovered source defect.

## Other Concerns

- The final warmed primary resolves the first-use performance anomaly:
  `readiness_vllm/vllm_result.json` dated `20260927-171032` reports
  19.7474 ms mean TPOT and mean ITL, 19.5984 ms median ITL, 2498.28 ms TTFT,
  and 25.5665 output tokens/s. Baseline values are respectively 19.6670,
  19.5973, 2642.79, and 24.8988. Decode is effectively flat (mean TPOT about
  0.4% higher), with no supported 11% speedup. No extra primary repeat is
  required by this review if the final report uses the warmed measurements
  and limits the claim accordingly.
- No actionable implementation defect was found in the bounded source change.
  `_Gemma4SamplingGenerator._run_sampling` delegates canonical sampling and
  penalty bookkeeping before copying public tokens. `_bind` releases old
  traces before rebinding both tensors; `_capture` warms the exact slice
  shape and sampler before capture. Both replay submissions remain
  nonblocking, and the plugin enqueues the deferred read immediately after
  `decode_forward`, before another submission can overwrite the public buffer.
- Final documentation should remove stale statements: `work_log.md:61`
  still describes the output slice as currently eager; `checklist.md:19`
  says the Watcher retry is pending although the retry log and JSON pass.
  `before/final_validation.json` is inherited stage-09 validation metadata
  adjacent to newly measured stage-10 baseline benchmarks; label that origin
  explicitly so it is not mistaken for fresh baseline sampling/quality evidence.

## Hard-Check Gaps

- `tests/check_vllm_adapter.py:66-69` finalizes every asynchronous read before
  the next decode. It does not implement the two queued decode/read submissions
  before host finalization proposed in `AUTODEBUG.md`. The parent plans a
  focused queued-read probe after the serving run. Cover the stable public
  tensor/view identity and distinct per-submission host values, including a
  padded multirow case; no additional full-model startup is requested here.
- The manual page edit at `tests/check_vllm_adapter.py:78-89` changes unused
  pages and sets `reset_batch=True`. It proves copying and trace retention,
  not allocator-driven page growth during overlap. The parent plans the
  existing full request test completed during this review: distinct
  31/63/95-token prompts with 96 output tokens, isolated/concurrent controls
  and the 33/1057/33 lifecycle sequence. Independent JSON comparison verifies
  all nine token arrays match `readiness_vllm/requests_full_final.json` exactly
  in `requests_final.json`. The unchanged plugin source correctly
  invalidates steady decode on nonempty block growth and drains pending output
  before host-state resets. This gap is resolved for the affected serving path.
- No serving profiler artifact is required or requested. The optimization
  contract explicitly substitutes serving metrics and trace correctness.

## Anomaly Ledger

- Observed anomaly: candidate first-use TPOT and ITL imply different rates.
  Evidence: candidate first-use primary JSON and the benchmark client source.
  Affected path: streamed primary performance measurement.
  Control or comparison: warmed baseline has matching mean TPOT and mean ITL.
  Likely subsystem: streaming/timing accounting; cause not yet established.
  Investigation performed: independently recomputed implied stream interval
  counts from E2EL, TTFT, and mean ITL; traced the client metric formulas.
  The client records ITL per response chunk at
  `vllm/vllm/benchmarks/lib/endpoint_request_func.py:221-249` and derives TPOT
  from total token count at `vllm/vllm/benchmarks/serve.py:423-444`. The final
  warmed row has matching TPOT and mean ITL, with approximately 127 intervals.
  Resolution: controlled; first-use result is retained but excluded from the
  final speedup claim. No material steady decode improvement is established.

- Observed anomaly: Watcher ACTIVE_ETH image exceeds its configuration buffer
  and its failed initial cleanup segfaults.
  Evidence: `batch_cache_watcher.log`, `batch_cache_watcher_retry.log`, and
  `batch_cache_watcher.json`.
  Affected path: Ethernet Watcher instrumentation, before model construction.
  Control or comparison: scoped retry with Ethernet instrumentation disabled
  succeeds, checks all four compute devices, and exactly matches both controls.
  Likely subsystem: instrumentation image capacity.
  Investigation performed: inspected retry log and numeric output arrays.
  Resolution: controlled with explicit limitation: Ethernet was not instrumented.

- Observed anomaly: live server emits the generic allocation-after-trace warning.
  Evidence: `readiness_vllm/server.log:157`; current reduced allocation-tracked
  `adapter_changed_pages_retry.log` completes both split traces and closes normally.
  Affected path: trace-buffer allocation lifecycle.
  Control or comparison: existing stage-09 trace-allocation investigation plus
  the fresh candidate probe with tracking and program-cache allocations enabled.
  Likely subsystem: generic allocation warning rather than demonstrated corruption.
  Investigation performed: inspected candidate binding/capture ordering and
  passing tracker-backed probe. The public output remains preallocated.
  Resolution: controlled for exercised paths; retain this scope in the final audit.

## Scope Inspected

- Goal/skill paths: supplied stage-10 contract; installed `stage-review`,
  `optimize`, `vllm-integration`, `tt-enable-tracing`; review core/router plus
  trace and vLLM domain checks. No new model-math optimization was requested.
- Artifact paths: `doc/optimized_vllm/` work log, checklist, AutoDebug/AutoFix,
  source manifests, baseline and candidate-first-use benchmark JSON, host test
  log, reduced adapter and Watcher results/logs; inherited vLLM runtime audit
  and stage review for unchanged contracts; current server launch/log.
- Code paths: complete `tt/generator.py`, `tt/generator_vllm.py`, changed host
  tests, common sampler capture/replay, reduced adapter/batch probes, request
  lifecycle harness, plugin page-growth/async submission/finalization, benchmark
  client accounting, and slice preallocated-output handling.
- Commands run: read-only git status/diff/revision checks, file reads/searches,
  SHA-256 source-manifest comparison, and small JSON metric calculations.
  No test, hardware, server, reset, or profiler command was run by this reviewer.
  Only this review document was written.

## Residual Risk

- Final all-layer sampling, qualitative review, CI benchmark, queued-read,
  and cleanup evidence remains pending. No clean-pass is granted.
- Source inspection supports unchanged selected precision, 262144 context,
  canonical split greedy sampling, device-owned token/position feedback,
  explicit host compatibility, and the real plugin adapter path. It does not
  substitute for the pending final runtime evidence.
