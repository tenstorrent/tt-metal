# Stage Review

Verdict: more-work-needed

Round 1 of 3. Independent inspection of stage 4, Qwen/Qwen3.6-35B-A3B,
on the live `main` worktree at starting HEAD
`2901d8a506198dc69a9e99714db3a9c590b6ff95`. No device was opened, no test or
measurement was rerun, and no implementation or test file was changed.

## Required Work

- **P2: Supply the required trace-allocation positive control.**
  Evidence: the selected skill's `tt-enable-tracing/SKILL.md:205,262`
  explicitly requires a deliberately live unsafe device buffer to make replay
  raise, in the same environment, with output saved at
  `doc/<stage>/trace_alloc_tracking/positive_control.log`. That file and directory
  are absent here. A search including ignored files under the model's `doc/`
  found no positive-control artifact, and the stage manifest/work log contain
  no such result. `natural_l{0,3}_b32`, both stress runs and the mixed-stack
  provenance do correctly record tracking=1 and skip-program-cache=0, but
  `tests/summarize_multichip_validation.py:120-124` verifies only those environment
  settings; its 26 gates never verify a rejection control. The runner acknowledges
  exact final outputs at `tests/run_multichip_decoder.py:263-268` and then runs
  clean replays, without attempting the required negative case.
  Why this matters: clean tracked execution is silent. The required control
  distinguishes an active allocation safety gate from settings recorded in
  provenance without an effective checker. This is a missing required gate
  artifact, not evidence that the decoder is numerically wrong.
  Required next step: run one small, bounded control on the same TTNN build and
  target mesh, with `TT_METAL_TRACE_ALLOC_TRACKING=1` and
  `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0` set before importing TTNN. Capture a
  prepared trace, retain a non-acknowledged device allocation made after capture,
  and assert that replay raises the expected unsafe-allocation RuntimeError.
  Save the command, environment and observed exception at the prescribed path,
  link it from the evidence manifest, and make the acceptance check require the
  passing control. A clean replay after releasing the injected allocation is a
  useful recovery check. The existing decoder correctness/timing/profile runs do
  not need to be repeated merely to add this control.

## Other Concerns

- The constructor census quoted in `mesh_plan.md:139-140` is stale:
  465,528,832 / 450,160,640 bytes versus the selected mixed-stack artifact and
  capacity plan's 465,569,792 / 450,185,216 bytes. The actual capacity calculation
  uses the latter and still has 4,135,717,888 bytes of planned headroom. Correct
  the prose; this small discrepancy does not invalidate the capacity result.
- Stage 5 has concrete performance work to start from: per-request attention at
  B32, approximately 200 us of output splitting, and unmeasured B32 shared/grouped
  geometry. The mesh's multirow projections currently use automatic programs,
  whereas the optimized baseline explicitly tuned prefill programs; adapting
  those to the local shapes is a useful candidate. The supplied profiles identify
  these costs and do not claim those unmeasured families were rejected. These are
  performance follow-ups, not additional correctness blockers for this review.

## Hard-Check Gaps

- The missing positive control is the required gap above. The 26/26 result
  otherwise agrees with the saved artifacts; every listed evidence path exists.
- The gate compares current decoder identity but does not require the measured
  runner's identity to be current. I investigated the mismatch rather than
  treating it as stale evidence: removing only the optional `prefill_states`
  export blocks and its CLI flag from the current paired runner reproduces the
  exact recorded hash `8e1e0437b146aadce7aa518dd0f820e39bef16aed70f8d8296d4b561cd084f7e`.
  That option is disabled in the earlier acceptance runs, so their execution
  remains applicable. No rerun is needed for this historical runner change.
- The profiler summary and checker share summarization helpers. Independently
  streaming the CSVs in this review verified all selected device-row timings
  against the original C++ timing CSV and recomputed each rank's mean; they
  matched. This is a check-design observation, not a request for another gate.

## Anomaly Ledger

- Observed anomaly: allocation tracking is claimed without its required positive
  rejection control.
  Evidence: tracked provenance, absent control artifact and checker source above.
  Affected path: allocation safety acceptance for new mesh traces.
  Control or comparison: none recorded.
  Likely subsystem: validation coverage.
  Investigation performed: read the tracking skill, tracker source, runners,
  manifest, work log and gate code; searched ignored evidence files too.
  Resolution: more-work-needed; the sole Required Work item.

- Observed anomaly: selected timing/correctness runs have an older paired-runner hash.
  Evidence: run provenance versus current `tests/run_multichip_decoder.py`.
  Affected path: acceptance provenance.
  Control or comparison: exact hash reconstruction after removing the later
  optional profiler state export.
  Likely subsystem: additive evidence tooling.
  Investigation performed: CPU reconstruction matched the original hash exactly;
  baseline and selected decoder sources also match their recorded identities.
  Resolution: controlled; no affected default runtime work.

- Observed anomaly: broad B32 profiling failed at the source-location limit;
  later CPU conversion exited 137 and an unfiltered export had malformed rows.
  Evidence: preserved failure captures, `AUTOFIX.md`, narrow correctness JSON,
  compaction manifests and `postprocess_multichip_profile.py`.
  Affected path: profiler collection/conversion, not the selected clean timing.
  Control or comparison: accepted eight-step B32 state seeds, two-step narrow
  eager/trace comparisons and original `cpp_device_perf_report.csv` timings.
  Likely subsystem: Tracy source metadata and CPU report memory use.
  Investigation performed: inspected native `tools/tracy/process_ops_logs.py`
  predicates and timing join, confirmed original C++ timing files are symlinked
  into processing, and independently matched all 340,812 selected B32 device
  rows to their original op names and durations. B1's 22,860 selected rows also
  match. Rank means reproduce both summaries. The representative-request
  prefill scope and omitted optional host-child columns are disclosed.
  Resolution: controlled; no device rerun is needed for these tooling failures.

- Observed anomaly: large profiler gaps and incomplete roofline modeling could
  be misread as decoder latency or zero DRAM traffic.
  Evidence: actual human tables, raw CSVs, `profile_findings_b1.md` and
  `profile_findings.md`.
  Affected path: performance interpretation.
  Control or comparison: clean serialized final timing JSON and raw rank sums.
  Likely subsystem: profiler drains, merged-rank reporting and unmodeled ops.
  Investigation performed: checked table warnings, signpost scopes, raw counts,
  timing provenance and the distinction between kernel work and elapsed latency.
  Resolution: controlled by explicit scope; no invalid speedup claim found.

- Observed anomaly: local dispatch's linear-prefill screen is 0.79% slower,
  while the selected path improves decode.
  Evidence: `local_dispatch_l{0,3}_{false,true}.json`, mesh plan and final timing.
  Affected path: EP dispatch selection.
  Control or comparison: matched policy pairs and fresh single-chip/mesh finals.
  Likely subsystem: whole-layer noise/tradeoff around a movement optimization.
  Investigation performed: checked selected source, described absolute times and
  the separate final latency rows. No universal prefill improvement is claimed.
  Resolution: controlled; the reported tradeoff is retained.

## Scope Inspected

- Goal/skill paths: `review_contract.md`; installed v0.1.24 `stage-review`,
  `model-bringup`, `multichip`, `tt-device-usage`, `optimize` and
  `tt-enable-tracing` skills; local LLM report section 3.3 and TraceCorrectness.
- Artifact paths: stage README, work log, gate summary, manifest, mesh plan,
  capacity/context contracts, latency CSV, both profiler summaries/findings,
  MoE shares, geometry/topology audits, AutoDebug/AutoFix evidence, final and
  natural B1/B32 results, stress/boundary/context/mixed-stack results and
  provenance, watcher logs, and the optimized baseline README.
- Code paths: all of `tt/multichip_decoder.py`; inherited prefill planning and
  relevant optimized contracts; paired/context/mixed-stack/profile runners;
  acceptance drivers, runtime audit, provenance, capacity and summary scripts;
  native profiler conversion, trace tracker and count-controlled expert/dispatch
  source. Native C++ files are unchanged by this stage.
- Commands run: read-only `rg`, `sed`, `cat`, `nl`, `git diff --cached --stat`
  and `git status`; CPU Python JSON/path/hash checks, Torch tensor comparisons
  without importing TTNN, and streaming CSV-to-native-timing comparisons.
  Only this report was written.
- Independent numerical checks: recomputed whole-output PCC for four final
  cases and both 128-step stress cases from tensor artifacts. Differences from
  recorded PCC were at most 2.55e-14. Lowest stress decode PCC was
  0.999768122 (linear) / 0.999777332 (full); mixed-stack minimum output PCC
  recomputed as 0.999335206. All exceed 0.995.
- Independent timing checks: recomputed medians, ratios and four-chip efficiency
  for all eight final latency rows. They match the CSV and README. Profiler row
  counts per rank/replay are B1 177/79 and B32 3636/1588 for linear/full; each
  phase's raw rank means match the saved summaries and original device timing.
- Implementation contracts inspected: local GQA head duplication, linear
  head/state partitioning, FP32 linear partial reduction, global gate-selected
  IDs, count-controlled local EP dispatch, worst-case capacity padding,
  zero-initialized combine, internal nonaligned MoE padding, bounded concat,
  shared serial CCL workspace, and shared physical KV ownership. No unexplained
  numerical or ownership defect was found in the inspected evidence.

## Residual Risk

- This is decoder-stage evidence. The 40-layer resident footprint is extrapolated
  from measured constructor censuses; the 6 GiB activation/trace allowance is a
  budget, not a full-model peak measurement. Maximum full-model context, model
  quality, generation and serving remain later-stage obligations.
- Clean timing is short-context, standalone, blocking decoder trace timing;
  long-context tests are eager correctness evidence. Neither implies production
  token throughput or long-context latency.
- The shared workspace assumes serial CQ0 use and retained backing resources.
  The next stage must preserve that lifetime and its preparation contract when
  adding more layers, trace signatures or concurrent execution.
- The planned action after a passing review is local stage-owned checkpoint
  commits only. No push is authorized; no push history was inferred. At this
  round, the missing positive control prevents a passing verdict.
