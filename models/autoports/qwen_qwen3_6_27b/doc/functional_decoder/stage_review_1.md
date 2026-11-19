# Stage Review

Verdict: more-work-needed

Independent review, round 1 of 3, of stage 1 functional-decoder for
`Qwen/Qwen3.6-27B`. Reviewed live worktree on `main`, base commit
`a0b0bd54e198d81676662b5a2d3a489e29bc9ab9`. Decoder SHA256 verified as
`476f9d99fa90a1268f264964037dbc6837b320adc9f48d172dc1156b8b35962d`.

The correctness evidence supports both layer kinds, the advertised context,
continuation, paged ownership, and traced decode. One explicit measurement
requirement remains open. This review found no additional demonstrated decoder
correctness defect. Existing correctness evidence need not be rerun for a
measurement-only correction.

## Required Work

- P2: Collect the final performance evidence with trace allocation tracking disabled.
  - Evidence: Both `perf/linear_attention/metrics.json:27` and
    `perf/full_attention/metrics.json:27` record
    `TT_METAL_TRACE_ALLOC_TRACKING="1"`. The corresponding
    `layer_0_profile_final.json` and `layer_3_profile_final.json` agree.
    `work_log.md:54` exports tracking before the profiling command at line 62.
    The tracing skill's Trace Allocation Safety Gate explicitly says,
    "Collect performance numbers separately with tracking and diagnostics
    disabled, recording the settings for each run."
  - Why this matters: These are the stage's accepted warmed performance numbers,
    including inter-op gaps. Allocation accounting is active during eager
    prefill and replay submission; an uninstrumented measurement is absent.
    The CSV arithmetic is correct, and this finding does not assert that the
    reported kernel durations are numerically wrong. It is an unmet explicit
    measurement contract, not a missing correctness test.
  - Required next step: Repeat only the two short, batch-1, length-512 warmed
    profiling runs with `TT_METAL_TRACE_ALLOC_TRACKING=0` and
    `TT_METAL_TRACE_ALLOC_TRACEBACKS=0`, Watcher disabled, retaining the
    5000-program profiler capacity and between-pass drains. Preserve the
    existing tracked runs as correctness evidence. Regenerate each kind's
    human tables, filtered CSV, compressed raw ops, provenance and marker
    audit, and update the accepted timing values. Record the disabled
    diagnostic flags explicitly. No decoder change or full-context rerun is
    requested for this finding.

## Other Concerns

- Long-context HF validation is selected-row validation. The complete TT
  decoder executes every prompt token, and the reference advances every K/V
  or recurrent-state update, but only boundary rows and the final 32 rows are
  compared. This is permitted by the functional-decoder skill and accurately
  disclosed. It does not establish all-layer or generated-text accuracy.
- Native full-attention arithmetic remains sensitive to reduction geometry.
  The final 512-key policy passes the required real-weight evidence; its
  minimum selected long-context contribution PCC is 0.9977730592. Preserve
  that gate during later optimization. This is a handoff risk, not a request
  for a precision or geometry sweep in this stage.
- Serial mixed-request prefill and up to 511 exact token steps for unaligned
  large-cache continuation are explicit functional baselines. Their latency
  is not represented by the aligned length-512 performance row. No optimized
  latency claim is made, so they do not block this stage.

## Hard-Check Gaps

- `tests/evidence.py` records tracking but omits
  `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE` and
  `TT_METAL_TRACE_ALLOC_TRACEBACKS`. The inspected run commands and lane
  environment do not set a program-cache exclusion; native
  `tt_metal/llrt/rtoptions.cpp:442` defaults that exclusion to false. There is
  no evidence that the accepted checks skipped program-cache allocations.
  Recording these flags would remove ambiguity in future artifacts; their
  absence alone is not a second required rerun.
- Trace reuse counters are literal summary values, but the inspected harness
  independently supports them: one capture before six requests, the same
  stable bindings and trace ID for each replay, no release in the request
  loop, and release only after the final replay. Extra runtime counters are
  not required for this layer-level harness.

## Anomaly Ledger

- Observed anomaly: Accepted profiling ran with allocation tracking enabled.
  Evidence: Both final performance provenance objects and final profiling JSONs.
  Affected path: Warmed prefill and traced decode measurements.
  Control or comparison: No final tracking-disabled measurement supplied.
  Likely subsystem: Measurement instrumentation and dispatch overhead.
  Investigation performed: Inspected runner signposts, tracker entry point,
  recorded environment, raw ops hashes and filtered CSV totals.
  Resolution: more-work-needed; Required Work above.

- Observed anomaly: Earlier linear-attention length-511 contribution PCC failed.
  Evidence: `AUTOFIX.md`, `gdn_contract_final.json`, norm/inverse diagnostics.
  Affected path: Folded RMSNorm scaling and decoder contribution.
  Control or comparison: The same seeded contract improved from
  0.9940785704 to 0.9980010967 after preserving folded scales in FP32.
  Likely subsystem: Premature scale rounding.
  Investigation performed: Compared implementation with installed HF norm
  semantics and checked the final per-row array and all seven resume controls.
  Resolution: fixed on the reviewed source.

- Observed anomaly: Earlier long full-attention decode PCC was 0.9563047020.
  Evidence: `full_context_262143_retry.json`,
  `long_attention_debug/{chunk_32,chunk_256,chunk_512,all_tokens_chunk512}.json`,
  and the two `full_final_context_26214*.json` artifacts.
  Affected path: Native long-context attention reduction.
  Control or comparison: HF-cache tail probes distinguish key-block geometries;
  the full-context HF oracle control is bit equal to installed HF at its
  selected rows and decode. Final all-token TT runs use no diagnostic override.
  Likely subsystem: Reduced-precision running attention state; individual
  rounding contributions are not separately quantified.
  Investigation performed: Inspected oracle code and control JSON, geometry
  probe metrics, final chunk/padding/unaligned-start rules, and final raw logs.
  Final decode PCC is 0.9995742584, with eager/replay bit equality; exact-max
  prefill PCC is 0.9998768214. Final high-position continuation also passes.
  Resolution: fixed for the advertised stage contract, with the numerical
  residual risk noted above.

- Observed anomaly: The first device-only tiny integer fill triggered Watcher.
  Evidence: `full_contract_device_fill.log`, `watcher_audit.json`,
  `position_fill.json`, and `full_contract_final.json/log`.
  Affected path: Unaligned full-attention continuation position construction.
  Control or comparison: Final code fills a physical 1x32 row before slicing;
  35 exact-value fill cases and the final continuation contract pass.
  Likely subsystem: Native small-row circular-buffer sizing versus aligned NoC write.
  Investigation performed: Inspected final construction, native audit and
  compressed Watcher evidence, including historical/final separation.
  Resolution: fixed by the target-local padding workaround.

- Observed anomaly: Earlier profiler runs dropped markers.
  Evidence: Historical profile logs, `profiler_audit.json`, both final
  compressed profile logs, and compressed raw ops.
  Affected path: Profiler capacity and collection completeness.
  Control or comparison: Final runs increase capacity to 5000 and drain
  between complete passes; final warning lines contain no marker-drop failure.
  Investigation performed: Read final warnings, verified raw ops hashes,
  recalculated all four filtered timing totals and checked rendered tables.
  Resolution: fixed for collection completeness; tracking settings remain the
  distinct Required Work item.

- Observed anomaly: Earlier long-context timeout followed triage that can halt cores.
  Evidence: Chronological work log, `AUTOFIX.md`, and final long-context logs.
  Affected path: Historical execution/diagnosis, not accepted numerical results.
  Control or comparison: Subsequent normal mesh smoke and final full-context
  runs complete without triage or diagnostic chunk drains.
  Likely subsystem: Historical cause remains uncertain because triage perturbed execution.
  Investigation performed: Checked that the owner does not claim a proven
  original dispatch defect or treat the timeout as a capacity limit.
  Resolution: controlled; no unresolved failure appears in accepted runs.

- Observed anomaly: Final logs include motherboard-discovery/subset-MMIO
  warnings, an optional Tracy WASM-copy warning, a report category warning,
  and dependency deprecations.
  Evidence: Final console logs, human performance tables and synthetic pytest log.
  Affected path: Discovery labels, optional visualization copy, report category
  classification and host dependencies.
  Control or comparison: Local physical-chip-0 runs close normally; raw ops
  generation, checksum checks and four synthetic tests succeed. Watcher logs
  contain no error/assert/sanitize report. The deliberate allocation rejection
  is identified as the positive control, followed by a successful replay.
  Likely subsystem: Host tooling and intentional diagnostic control.
  Investigation performed: Separated these lines from actual model/device
  failures and inspected their surrounding successful results.
  Resolution: controlled.

## Scope Inspected

- Goal/skill paths: Parent-provided original stage contract;
  installed `tt-model-bringup/0.1.23/skills/{stage-review,functional-decoder,
  tt-device-usage,tt-enable-tracing}/SKILL.md`; model-bringup startup;
  checkout `tech_reports/AdvancedPerformanceOptimizationsForModels/TraceCorrectness.md`.
- Artifact paths: `doc/context_contract.json`, pinned `doc/hf_config.json`,
  `doc/functional_decoder/{README.md,work_log.md,acceptance.json}`; both final
  contract/reuse sets; both kinds at lengths 262143 and 262144;
  `full_final_context_4097`; final batch-32, mixed-position Watcher and profile
  results; synthetic pytest and hooks logs; runtime/native, Watcher and
  profiler audits; both complete performance directories; AutoFix/AutoDebug
  investigations and HF-oracle/geometry control results.
- Code paths: Entire `tt/functional_decoder.py`; all eleven test/helper files;
  relevant installed HF Qwen3.5 decoder, norm, gated-delta and attention
  implementations; trace allocation checker and native option default.
- Commands run: Read-only `git status --short`, `git rev-parse HEAD`, `rg`,
  `sed`, `cat`, `nl`, `wc`, and Python standard-library artifact analysis.
  Python checked source/harness/HF hashes, JSON gates, compressed raw-log/ops
  hashes and filtered CSV sums. No TTNN import, device open, hardware test,
  reset, process termination or server launch was performed. Only this report
  was written.
- Verified results: Final decoder and current harness/HF hashes match their
  accepted artifacts. Each layer has 12 length cases, seven successful resume
  windows with seven failing dropped-history controls, and mixed rows. Each
  retained trace survives six changing requests. Batch 32 and mixed positions
  pass, with repeated bit equality. Maximum-context evidence is real-weight,
  all-token TT execution without a context reduction. Four synthetic tests pass.
- Recomputed performance totals, pending the instrumentation correction:
  linear prefill/decode kernel sums 15996.625/2732.151 microseconds and
  kernel-plus-gap sums 18809.156/2795.138 microseconds; full attention
  6306.836/2333.960 and 6460.895/2379.251 microseconds, respectively.
- Push/checkpoint state: Run plan requires a local checkpoint only after a
  passing review and prohibits pushing. The only reported dirty path is the
  untracked target autoport. No commit or push was performed in this review.

## Residual Risk

- This is an inspection review of completed evidence, not an independent
  hardware reproduction. It accepts the reduced long-context oracle within
  its documented scope.
- Real target shapes and weights are exercised with seeded synthetic layer
  inputs; later full-model stages must validate actual propagated activations
  and generation. Those later-stage gates are not imposed here.
- The correctness evidence remains valid if remediation changes only profiler
  settings/reporting. Any decoder change would need the affected checks,
  rather than automatic reuse of this source's evidence.
