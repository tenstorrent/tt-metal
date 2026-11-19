# Stage Review

Verdict: clean-pass

Independent round 1 review of stage 4, multichip-decoder, for
`ornith-ai/Ornith-1.5-9B`. This verdict covers the delivered 1x4 Blackhole decoder
and its documented serial-stack interface. It does not certify a full model,
generator, serving adapter, or concurrent command queues.

Reviewed live worktree on `main`, base HEAD
`f3b556f11d7e84e13c4640e9b8d667319d99c14e`. Final implementation SHA256:
`bc8386dc4e5c8f6fb18f24addca003e9cd37efe2b57341e6be9b06948aef88fc`.
The optimized single-chip baseline is unchanged relative to HEAD and hashes to
`b678803e78b74ede66d61e7609c6ca9147b2977dbba702f8431eb621ef997c4e`.

## Required Work

- None. The inspected source and existing runtime evidence satisfy the supplied
  decoder-stage contract. No hardware rerun is required by this review.

## Other Concerns

- Closed during review: profiler aggregation was inaccurately described as the
  maximum device duration for every operation. The installed
  `tt_perf_report/perf_report.py::merge_device_rows` uses the maximum for
  noncollectives and the mean for collectives. The owner corrected README,
  `tests/summarize_multichip.py`, and `perf_summary.json`, and recorded the
  correction in the work log. I checked the corrected files and unchanged
  implementation hash. The underlying CSV values were valid; this correction
  requires neither new measurements nor another review round.
- Smaller-chunk direct prefill all-reduce remains an explicitly unmeasured
  optimization candidate. The whole-1024-row operation has an exact L1 blocker:
  `all_reduce_async_program_factory.cpp:273-278` reserves the output-page scratch
  per link, giving 4 MiB per worker at two links for BF16 1024x4096 output.
  That exceeds the stated physical worker L1. The stage does not claim that
  chunked direct all-reduce was measured or rejected.

## Hard-Check Gaps

- No missing required current-stage gate was found. The artifact audit is useful
  but does not independently prove every claim: I additionally inspected the
  runners, compared logged numerical rows with their JSON sidecars, verified
  profiler raw-source hashes, and recomputed performance and capacity arithmetic.
- The final shell runner's reuse test keys only on the implementation hash. A
  future change to a test, its arguments, or reference data should explicitly
  invalidate affected artifacts; that reuse mechanism alone would not establish
  freshness. This is not evidence that the reviewed final run is stale: its
  queue log records new execution of the final jobs, their current-source hashes
  agree, and the inspected logs match the sidecars.
- Current trace evidence intentionally has one live trace per standalone test
  fixture, or one trace covering the complete two-layer batch. It establishes
  warmed decoder replay and request-state refresh on that scope. It does not
  establish a future generator's simultaneous trace variants or eager-prefill
  interleaving under live traces. Those are subsequent generator-stage gates.

## Anomaly Ledger

- Observed anomaly: an oversharded K3072 decode matmul hung during the geometry
  search; the initial informal attribution named the wrong configuration.
  Evidence: `AUTOTRIAGE.md`, `AUTOFIX.md`, `geometry_0_before_repair.json`,
  `repair_geometry.json`, and the resumed geometry artifacts.
  Affected path: experimental down projection with 64 input storage cores.
  Control or comparison: exact 32-core input coverage passed; the 48-input,
  64-output repair passed both blocks with minimum PCC 0.9999999404.
  Likely subsystem: DRAM-sharded matmul sender-coordinate accounting.
  Investigation performed: source/live-evidence triage, independent exact-K
  analysis, repair probes, and the completed 312-case precision-locked sweep.
  I independently evaluated the pure host planner over K1024/3072/4096 and all
  requested core counts and legal shard-block divisors. Input coverage is exact.
  Resolution: fixed in the model-local planner. The shared oversharding kernel
  limitation is not claimed repaired; the corrected configuration attribution
  is explicit in the retained evidence.

- Observed anomaly: resident fused-prefill failed the watcher NoC exit check.
  Evidence: `watcher_triage/AUTOTRIAGE.md`, `watcher_triage/AUTOFIX.md`, retained
  original failure artifacts, planner sweep, constrained-family measurements,
  and final `watcher_{0,3}.json/log`.
  Affected path: the former resident matmul/reduce-scatter candidate.
  Control or comparison: separate minimal matmul plus RS/AG passes; 64/96-KiB
  constrained fused alternatives pass numerical tests but are slower.
  Likely subsystem: dedicated-NoC ownership and per-RISC read accounting.
  Investigation performed: source diagnosis and host planning analysis, bounded
  recovery, adapted fused candidates, then complete final default revalidation.
  The final source disables this fusion by default and constrains the optional
  path to 64 KiB. I inspected the two final watcher logs: both attach the four
  devices, complete their runs, and detach without an error/assertion entry.
  Resolution: fixed for the selected model path; shared resident fusion is not
  claimed repaired and watcher checks remain enabled.

- Observed anomaly: TTNN compacts requested matmul output grids; UMD emits a
  motherboard-mapping warning.
  Evidence: `anomalies.md`, final logs, mesh provenance, and output/state records.
  Affected path: returned local layouts and machine discovery.
  Control or comparison: all-rank output/contribution PCC and reconstructed
  local state pass; all four devices participate in final CCL work.
  Likely subsystem: output-layout selection and UMD topology discovery.
  Investigation performed: the implementation consumes returned layouts and
  converts at explicit boundaries; the stage retained repeated successful
  open/execute/close and numerical evidence on this machine.
  Resolution: controlled within the documented hardware/interface scope.

- Observed anomaly: generic profiler advice says minimal matmul has no program
  config; decode compute utilization can exceed 100%.
  Evidence: `tracy/{0,3}/*_perf_report.*`, explicit `MinimalMatmulConfig` in source,
  and the installed report's DRAM-sharded core-count override.
  Affected path: report interpretation.
  Control or comparison: raw matmul rows show BF16 inputs, BFP8 weights, the
  intended HiFi2/LoFi policies, and measured DRAM utilization.
  Likely subsystem: report recognition and utilization denominator.
  Investigation performed: source/report comparison. The explicit prefill
  M4/K16/N8, subblock1x2 configuration exists; the report uses eight DRAM-interface
  workers as its compute denominator even for multi-reader configurations.
  Resolution: controlled reporting limitations; no physical utilization claim
  above 100% is made.

- Observed anomaly: profiler aggregation label treated collective means as maxima.
  Evidence: installed report merge function and the corrected summary files.
  Affected path: README and summary metadata only.
  Control or comparison: the operation CSV numerical values were unchanged.
  Likely subsystem: documentation.
  Investigation performed: direct report-source inspection and owner correction.
  Resolution: fixed during this review.

The early mesh-binding, fused-gather worker-count, fused-scatter descriptor, and
fixture-metadata errors in `anomalies.md` are also classified and followed by
successful adapted runs. No unresolved core-path warning was found in the final
acceptance logs inspected.

## Scope Inspected

- Goal/skill paths: supplied stage-4 contract; installed 0.1.23
  `stage-review/SKILL.md`, `multichip/SKILL.md`, `optimize/SKILL.md`,
  `tt-enable-tracing/SKILL.md`, `tt-device-usage/SKILL.md`, and model-bringup startup
  instructions. Repository references included `tech_reports/LLMs/llms.md`
  section 3.3 and `TraceCorrectness.md`.
- Artifact paths: stage README, mesh plan, work log, anomaly ledger,
  `final_audit.json`, final queue/hooks logs, context contract, memory plan,
  environment provenance, topology/geometry comparisons, baseline/final timing
  JSON, final contracts/stress/long/stack JSON and logs, watcher logs, profiler
  TXT/CSV/provenance/raw operation CSVs, and both AutoTriage/AutoFix investigations.
- Code paths: complete `tt/multichip_decoder.py`; relevant optimized baseline
  and baseline test helpers; multichip probe, long, stack, geometry, topology,
  memory, profiler, audit, summary, and shell runners; common CCL manager;
  all-reduce scratch-allocation source; installed profiler merge/accounting code.
- Commands run: read-only `cat`, `sed`, `rg`, `wc`, `sha256sum`, `git diff`, and
  `git rev-parse`; small standard-library Python analyses for artifact/log
  equality, source/profiler hashes, pure-host planner invariants, geometry
  ranking, timing medians, and storage arithmetic. No TTNN imports, device jobs,
  resets, servers, or long experiments were run by the reviewer. This report
  is the only file written by the reviewer.

Important results independently checked:

- All 16 final artifact families carry the current implementation hash. There
  are 3,848 recorded PCC metrics with minimum **0.9997044802**, exceeding 0.995.
  Logged output comparisons match their JSON records in contracts, stress, and
  long-context runs. The runs use real checkpoint weights and matched synthetic
  hidden inputs against the unchanged optimized TTNN reference.
- Both layer kinds cover the 11 specified logical lengths, eight changing decode
  inputs per length, all replicated output ranks, and local state reconstruction.
  Stress covers 128 evolving steps. Long runs execute complete lengths 262139,
  262143, and 262144, compare selected windows, and test bit-identical restored
  terminal replay. The context contract remains 262144.
- B1/B32 layer0-to-layer3 traces consume the actual preceding layer's output,
  with ragged prefill, unaligned continuation, shuffled tables, distinct
  positions, and shared workspace. Batch32 is correctly described as serial
  per-row work, not vectorized throughput.
- Tracked acceptance runs enable allocation tracking with program-cache tracking
  retained. Each contract log contains 11 actual rejected unsafe-allocation
  positive controls. Trace IDs 0 through 10 each have three fresh-request
  refresh comparisons in their fixture. Persistent state/input/page-table
  storage is allocated before capture and refreshed in place.
- Allocation acknowledgments target exact outputs only; no allocation scope
  exempts a forward/capture/warmup. With the sole live trace in these fixtures,
  each acknowledged output is overwritten by its producing trace and consumed
  before the next replay. Stack outputs remain referenced during the entire
  capture, so later row allocations cannot reuse their backing storage. State
  and the shared CCL receive buffers are not exempted. The shared workspace is
  allocated once before capture; two reductions per layer restore the ping-pong
  index, and one command queue orders reuse across layers/rows.
- All 312 geometry rows pass and retain the intended precision. Selected
  per-role configurations are the measured minima. Coherent sharded-residual,
  fused-gather, fused-scatter, combined, and replicated alternatives were
  exercised through their norm/residual consumers, not merely gathered back at
  the producing boundary.
- Final warmed decode is 0.3078585/0.2884595 ms for linear/full layers, versus
  0.6437405/0.6240834 ms single-chip. Prefill is 2.1632379/2.2194451 ms versus
  3.1796093/3.6146101 ms. The final implementation reproduces the selected
  two-link candidate within 0.5%; reported speedup and efficiency arithmetic
  agrees with the timing artifacts. Timing runs exclude watcher/profiler/tracker.
- Profiler raw-source hashes match provenance. Actual matmul rows carry BFP8
  weights, BF16 activations, HiFi2 attention and LoFi MLP. Category sums agree
  with the operation CSVs and are correctly labeled as instrumented aggregates.
- Independently recomputed stored-weight, KV, recurrent, convolution, table and
  reserve arithmetic gives **17,544,937,472 bytes (16.339996 GiB) per chip** against
  34,225,520,640 usable DRAM bytes. The separate shared L1 allocation is 2 MiB per
  chip, 64 KiB per participating core. The report explicitly presents whole-stack
  capacity as an analytic expectation rather than a measured 32-layer allocation.

Push state follows the supplied run plan: the stage agent never pushes; only
stage-owned local checkpoint commits follow a passing review. No commit or push
was performed by this reviewer, and no push state was inferred from reflogs.

## Residual Risk

- Full-stack accuracy, all-layer allocator peak, generation quality, serving,
  vectorized batching, and concurrent execution are outside this completed
  decoder-stage evidence. The 262144-token memory plan budgets one maximum-length
  resident request, not 32 simultaneous maximum-length caches.
- The default constructor can create a private workspace for an isolated layer.
  Full-model assembly must follow the documented shared `MultichipCollectives`
  contract to avoid multiplying persistent L1 receive buffers across 32 layers.
- Long-context comparisons sample output windows; they do not compare every
  long-sequence output row. The complete prefill executes, and short/medium tests
  compare all output elements and reconstructed state. This matches the stated
  evidence scope and does not justify a broader whole-model correctness claim.
- Future generator integration must preserve all captured backing allocations,
  prepare all supported variants before capture, and validate actual mixed-request
  trace reuse with its own trace ordering. The present decoder evidence should
  not be relabeled as that later-stage gate.
