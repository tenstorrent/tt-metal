# Stage Review

Verdict: clean-pass

Independent review, round 1 of 3, of stage 1 functional-decoder for `Qwen/Qwen3.6-35B-A3B`. Reviewed the live worktree on `main`, HEAD `a407e7d3ec6a00693e1be369d8a395f63c9c2c62`, including the uncommitted stage changes. The reviewed decoder SHA256 is `ff3307ea3a3a4dcc93400fdd2a79a47a32e216d9f1bc47626e875dc7b561f99b`.

## Required Work

- None. The inspected implementation and execution artifacts satisfy the functional-decoder acceptance contract. This verdict permits the stage owner to perform the remaining local commit/report steps; it is not a claim that those subsequent steps have already occurred. The supplied run plan authorizes local stage commits and explicitly forbids pushing.

## Other Concerns

- The dense 256-expert bank and serial per-row batch methods are substantial performance opportunities for the optimization stage. Their current semantics include the actual device-selected top eight experts and gated shared expert. No functional-stage latency target requires replacing them here.
- The trace orchestration contract is deliberately a decoder-layer boundary: callers prepare/warm paths and retain the input, state, page-table, and position allocations before capture. Later model/generator work must implement that preparation and lifetime ownership centrally. The current stage demonstrates this contract in its harnesses; it does not claim a production generator or token-feedback loop.

## Hard-Check Gaps

- `tests/check_evidence.py` primarily validates recorded summaries and does not itself bind every accepted result to current source hashes or recompute report totals. This review independently checked those bindings, raw profiler sequences, report hashes/totals, watcher logs, and archive hashes. Incorporating these existing checks into a single future gate would reduce stale-artifact risk; no new device evidence is needed for this review.
- The original linear long-context result predates provenance sidecars. It is supported by its preserved execution log and unchanged linear arithmetic, rather than a fabricated retrospective sidecar. I independently compared the current decoder AST against the checkpoint: only `_full_attention`, `new_kv_state`, and `prepare_prefill` changed. The linear planner branch remains equivalent, and current-source profiling provides additional real-weight linear correctness evidence. This does not invalidate the existing long-context result.

## Anomaly Ledger

- Observed anomaly: original full-attention long-context execution was interrupted by the investigation/recovery sequence, with a later halted-device dispatch observation that could not establish a pre-existing model hang.
  Evidence: `AUTOFIX.md`, preserved triage material, `recovery.json`, recovery logs, and the final `logs/long_tail_regression_fixed.log`.
  Affected path: device execution and recovery, before accepted full-context evidence.
  Control or comparison: authorized reset, successful arithmetic mesh smoke, and subsequently completed full-context and regression runs.
  Likely subsystem: device/debugger state; an original model-hang cause was not established.
  Investigation performed: preserved host/device diagnostics, source audits, explicit correction of the earlier inference, and bounded recovery.
  Resolution: controlled. No acceptance claim relies on the interrupted run.

- Observed anomaly: full-attention long-context decode and boundary tokens had poor PCC with the original SDPA geometry; a grid-only repair still left failing prefill rows.
  Evidence: `sdpa_precision_probe.json`, `real_tail_probe.json`, `real_prefill_tail_probe.json`, `full_long_grid_only.json`, `long_tail_expanded_failure.json`, final `full_long.json`, and corresponding logs.
  Affected path: native SDPA's long reductions and BF16 intermediate accumulation.
  Control or comparison: isolated geometry probes and real-weight attention probes, followed by end-to-end TT-produced cache validation.
  Likely subsystem: numerical accumulation in SDPA, rather than cache address overflow.
  Investigation performed: source inspection, controlled grid/key-chunk comparisons, every final 64-row checks, and refreshed sibling paths affected by the key-chunk change.
  Resolution: fixed by the model-local 8x8 decode grid and 256-token key chunks. Final prefill aggregate PCCs are 0.999887458/0.999887432 for lengths 262143/262144; the minimum checked trailing-row PCC is 0.998649944. Final traced decode PCC at position 262143 is 0.999947275. All exceed the unchanged 0.995 bar.

- Observed anomaly: small logical capacities required page-table padding for 256-token SDPA reads, and native cache-update validation also required the physical pool to accommodate the padded table width.
  Evidence: `page_extent_regression.json`, `padded_trace_regression.json`, final allocation/planning code, and native page-cache/SDPA validation.
  Affected path: small/nonaligned capacities, read extent, and ownership.
  Control or comparison: capacities 1/31/32/33/255/256/257/511/512/513; zero versus large finite future-row sentinels; changed page assignments through one tracked trace; shared-pool K/V comparisons and untouched guards.
  Likely subsystem: physical layout and native operation contracts.
  Investigation performed: inspection of bounds and alias rules, plus independent host execution of the actual planner for 86,632 valid start/length/chunk combinations. Multi-token starts were always 256-aligned, lengths were tile-aligned and at most 256, and all plans covered exactly their logical range.
  Resolution: fixed. Padding aliases the final owned finite page and is causally masked; valid writes remain within owned logical pages. Logical capacity is unchanged. Decode position validity remains the documented caller precondition.

- Observed anomaly: initial profiler runs dropped device markers.
  Evidence: historical overflow artifacts marked invalid; final raw `tracy/layer_{0,3}/ops.csv`, profiler logs, filtered reports, and `profiler_completeness.json`.
  Affected path: measurement completeness, not accepted model numerics.
  Control or comparison: final collection drains the profiler between complete replays.
  Likely subsystem: profiler buffer capacity.
  Investigation performed: independently reconstructed every replay sequence and the signposted measurement windows from raw CSVs. Each kind has 20 identical replay sequences, of which 16 are measured: 144 operations per linear replay and 99 per full-attention replay. Final logs have no dropped-marker warnings.
  Resolution: fixed. Independently recomputed device durations agree with the summaries: prefill 69685.930/68881.854 microseconds and decode 60378.858/60137.201 microseconds per pass, linear/full respectively. Microsecond units, division by 16, and separate dispatch-gap reporting are correct. These are device-duration sums, not host latency or model throughput.

- Observed anomaly: a successful initial watcher/reuse workload failed during MeshTraceId JSON serialization.
  Evidence: preserved failed log and completed replacement `trace_reuse_{0,3}.json` with provenance and logs.
  Affected path: result serialization.
  Control or comparison: complete reruns with string trace IDs.
  Likely subsystem: test reporting.
  Investigation performed: inspected the final harness and execution results. Both runs retain one trace across nine changed/repeated requests and release it only at teardown.
  Resolution: fixed. Both logs contain the deliberate allocation-hazard rejection; accepted replays include program-cache allocation tracking. Independent watcher-log inspection found 15 completed dumps per kind, matching hashes and no corruption/assert/address/overflow/error messages.

## Scope Inspected

- Goal/skill paths: original `multigoal_logs/01-01-functional-decoder.prompt.txt`; installed 0.1.23 `stage-review`, `functional-decoder`, `tt-device-usage`, and `tt-enable-tracing` skills; applicable startup instructions; checkout `tech_reports/AdvancedPerformanceOptimizationsForModels/TraceCorrectness.md`.
- Artifact paths: stage README/work log, `../context_contract.json`, hard gates, both long-context results, boundary/partition/continuation/batch results, refreshed trace reuse, padded-page regressions, synthetic JUnit and result artifacts, runtime/source/watcher audits, profiler reports and raw CSVs, provenance sidecars, anomaly evidence, and archive manifest.
- Code paths: complete `tt/functional_decoder.py`; reference, long-context, boundary/batch, trace-reuse, padded-page, profiling, provenance, audit and evidence-check harnesses; installed HF `qwen3_5_moe` decoder/MoE/norm/attention code; relevant native gated-delta, scatter, slicing, paged-cache and SDPA code. The installed HF modeling source matches its distribution RECORD hash.
- Commands run: read-only `cat`, `sed`, `rg`, Git status/show and small standard-library Python analyses of ASTs, JSON, logs, CSV, JUnit, hashes, and planner behavior. All 138 archived files, including the split linear CSV, were checked against recorded decompressed hashes and available original bytes. No TTNN import, device open/reset, hardware test, server, or implementation edit was performed by this reviewer. The only write was this report.

The two layer kinds use the real target geometry and real layer-0/layer-3 weights. The inspected source matches HF residual order, normalization conventions, partial RoPE, full-attention gating, recurrent convolution/history semantics, and routed/shared expert composition. Prefill supports arbitrary valid logical lengths through aligned chunks and exact boundary tokens. The contract retains the advertised 262144 context for both kinds.

The numerical gates are meaningful: continuation checks subtract the input residual and require every resumed row to pass, while corresponding dropped-history controls fail. Final minimum continuation-contribution PCCs are 0.997672597 for linear attention and 0.999403081 for full attention. Shared-pool batch evidence additionally compares K/V contents and guards, rather than relying only on output PCC. Two synthetic target-shape tests pass without failures or skips.

Trace harnesses preallocate persistent inputs/state, warm the actual paths before capture, refresh the captured tensors in place, and consume output before another replay. Exact output acknowledgments have a simple single-live-trace lifetime; no broad allocation scope is used. Batch decode is itself captured and replayed three times, with per-row HF comparisons and bit equality. The audited runtime call tree contains no Torch or host-conversion calls; dynamic guards and inspected native helper branches support that result.

## Residual Risk

- This is inspection of completed evidence, not an independent hardware rerun. Acceptance applies to the reviewed source hash and recorded environment.
- Representative layers cover both computational kinds, not every distinct checkpoint layer's weights. Full-model integration and quality evaluation remain later-stage work.
- The exact full context is tested, including its nonaligned predecessor, but finite input/page/position samples cannot prove every possible state. Callers must preserve finite caches, distinct page ownership, matching recurrent history, valid current positions, and trace buffer lifetimes.
- Performance is a functional single-device baseline. Profile drains and diagnostic guards affect surrounding overhead; the report correctly limits its claim to measured device-operation duration.
