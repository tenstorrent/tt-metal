# Stage Review

Verdict: clean-pass

## Required Work

- None.

## Other Concerns

- The mandatory fixed 1x4 mesh is not a universal latency win. All measured
  prefills and layer-12 sliding decode at S=128 regress versus 1x1. The reports
  disclose those rows, and the current topology matrix supports all-reduce +
  EP4 as the fastest measured 1x4 decoder among the material supported
  candidates.
- The layer-13 single-chip decode baseline is about 6.75 ms, producing raw
  10.632x and 6.484x 1x4 ratios. The reports correctly label these as artifacts
  of a distinct one-device full-attention path, not mesh scaling or efficiency.
- Long-position decode has one warmed trace per 64-token page bank. The
  accepted 500-replay latency is steady-state graph replay within a bank; it
  excludes trace capture, bank-transition recapture, and host input refresh.
  The separate sequential test proves mutable hidden/position buffers and exact
  K/V writes through a bank transition, while README/context documentation
  states the recapture boundary.
- The strongest full-context gate performs real-weight decode at position
  131071 with fully allocated local caches and reverse page mapping, but does
  not populate all preceding 131071 tokens. Nonzero arbitrary-page cache
  correctness is bounded at S=2048; the endpoint adds capacity, full-prefix
  graph execution, finite-output, routing, and last-page placement evidence.
- Full Ethernet watcher instrumentation is unavailable on this platform because
  the instrumented ACTIVE_ETH program is 27,920 bytes versus a 25,600-byte
  kernel-config buffer. Worker/Tensix watcher, current ring-heavy runs, ARC
  heartbeat checks, and final board discovery compensate, but they are not
  equivalent to an ETH-instrumented watcher pass.

## Hard-Check Gaps

- Most current JUnit XML retains pass/fail and testcase identity rather than
  captured PCC stdout. It proves the required PCC thresholds and route checks;
  several exact decimal minima in README/work_log remain summary values rather
  than independently recomputable fields in the JUnit.
- `check_eth_status` exits under a triage `pass` wrapper while every per-device
  Ethernet read is explicitly skipped because of the installed UMD
  `noc_read` signature mismatch. The stage documents it as skipped, not passed.
- The flat evidence manifest hashes the authoritative artifacts and frozen
  source files but does not embed the source hashes inside each runner artifact.
  Final post-freeze suite, timing, topology, profile, watcher, and endpoint
  ordering plus manifest hashes provide the provenance chain.

## Anomaly Ledger

- Observed anomaly: The initial multichip prefill precision policy dropped
  precision-sensitive behavior from the optimized baseline.
  Evidence: Round-1 review, `AUTODEBUG.md`, `AUTOFIX.md`, final implementation
  branches, the frozen default suite, and the S=128/129/2048 precision cases.
  Affected path: Full-attention S=128/129/2048 prefill and following decode.
  Control or comparison: Real current `OptimizedDecoder` captures for both
  layer kinds, including attention, routing/top-4, output, logical K/V, and
  following decode.
  Likely subsystem: Full-attention accumulation and layer-aware expert/projection
  fidelity.
  Investigation performed: Compared current implementation policy to the
  optimized baseline and inspected the current real-weight testcase matrix and
  authoritative JUnit.
  Resolution: fixed

- Observed anomaly: The original long-position trace specialized the graph to
  one host-side cache position.
  Evidence: Round-1 review versus current `_manual_paged_decode_attention`,
  `test_warmed_long_position_trace_replay_matches_eager`, frozen-suite JUnit,
  boundary/trace JUnit, and watcher JUnit/log.
  Affected path: Sequential decode at and above position 127.
  Control or comparison: Eager outputs and per-rank physical K/V rows at
  positions 128-131, 191, and 192-193 for sliding and full layers.
  Likely subsystem: Trace lifecycle, page-bank shape specialization, device
  position masking, and paged cache updates.
  Investigation performed: Followed Python bank selection and mutable device
  position use through source and tests; verified one capture covers each bank
  and the next bank is recaptured once.
  Resolution: fixed

- Observed anomaly: The accepted timing JSON was previously overwritten by a
  short profiler run, and the earlier hardware evidence predated later source.
  Evidence: Round-1 review versus current `MULTICHIP_PERF_RESULT_PATH`,
  20/500 timing JSON/JUnit pairs, profile-specific JSON/raw CSV, 52-entry
  manifest, and exact frozen source hashes.
  Affected path: 1x1/1x4 latency, speedup, efficiency, profile, context, and
  watcher provenance.
  Control or comparison: All current authoritative artifact hashes match the
  live files; the frozen source hashes are
  `25fee42d...`, `3df7e045...`, and `f3797d18...`.
  Likely subsystem: Artifact naming and run ordering.
  Investigation performed: Recomputed every manifest hash, checked all
  authoritative XML for failures/errors, compared JSON repeat counts and
  values with reports, and checked source/artifact timestamps.
  Resolution: fixed

- Observed anomaly: Collective, carried-residual, fused O, and TP4-expert
  alternatives were initially rejected using historical evidence.
  Evidence: Current candidate source/tests, eight-case topology JUnit, candidate
  JSON, and 20/500 whole-layer S=128 timing JSON/JUnit for selected AR+EP4,
  padded RS+AG+EP4, attended-AG/local-O+EP4, and AR+TP4.
  Affected path: O-projection collective, expert collective/ownership, residual
  layout, next norm/router/QKV boundary, and persistent buffers.
  Control or comparison: Current selected all-reduce + EP4 decoder on the same
  frozen source, weights, mesh, sequence, and trace regime.
  Likely subsystem: Multichip topology selection.
  Investigation performed: Inspected candidate code, residual contracts,
  payload dtypes/bytes, persistent buffers, real-weight PCC/top-4, and
  whole-layer latency. Confirmed fused matmul+reduce-scatter is source-gated on
  Blackhole by issue #46181 rather than dismissed after a first API failure.
  Resolution: fixed

- Observed anomaly: The first 131072 endpoint gate covered only sliding
  attention; a later pre-fix full-layer endpoint run failed at PCC 0.934595.
  Evidence: Superseded `final_context_multichip_131072.junit.xml`, explicit
  classification in `AUTOFIX.md`/work_log, exact-manual and default-native
  capture JUnit/PT hashes, and passing
  `autofix_h5_endpoint_final_postformat.junit.xml`.
  Affected path: Full-attention decode and routing at position 131071.
  Control or comparison: Current default-native and exact-manual
  `OptimizedDecoder` endpoint controls for both layer kinds.
  Likely subsystem: Long full-attention projection fidelity and a near-tied
  top-4 route.
  Investigation performed: Inspected both failed and superseding artifacts,
  endpoint source/tests, reverse-page K/V checks, final manifest membership,
  and current documentation.
  Resolution: fixed

- Observed anomaly: Full ETH watcher fails before model execution.
  Evidence: `watcher_eth_attempt_final_frozen.log` records ACTIVE_ETH 27,920
  bytes exceeding 25,600 bytes; worker watcher JUnit/log passes four current
  CCL-heavy cases; triage reports healthy approximately 10 Hz ARC heartbeats;
  final `tt-smi` lists four P300c boards.
  Affected path: Ethernet watcher observability for ring collectives.
  Control or comparison: Worker/Tensix watcher on all four devices, repeated
  successful 1x4 ring collectives/traces, ARC checks, and board discovery.
  Likely subsystem: Watcher instrumentation size, not decoder execution.
  Investigation performed: Inspected full/partial watcher initialization,
  failure point, attached devices, pass/close sequence, triage skips, heartbeat
  data, and final health listing.
  Resolution: controlled

- Observed anomaly: The reported full-layer raw speedup and efficiency are
  superlinear.
  Evidence: Current 1x1/1x4 JSON and performance tables show about 6.75 ms
  single-chip versus 0.63-1.04 ms multichip decode.
  Affected path: Performance interpretation for layer 13.
  Control or comparison: Layer-12 rows and source inspection of the distinct
  one-device full-layer path.
  Likely subsystem: Baseline path selection, not physical four-chip scaling.
  Investigation performed: Recomputed ratios from authoritative JSON and
  checked the explicit report qualification.
  Resolution: controlled

- Observed anomaly: The mandatory 1x4 path regresses every measured prefill and
  S=128 sliding decode.
  Evidence: Frozen wall-clock JSON/report and the current topology matrix.
  Affected path: Prefill at S=17/128 and sliding decode at S=128.
  Control or comparison: Current 1x1 optimized baseline and four current 1x4
  candidate families.
  Likely subsystem: Collective overhead and fixed-mesh utilization.
  Investigation performed: Verified like-for-like repeats, source state,
  sequence, weights, candidate settings, and selected latency.
  Resolution: controlled — explicitly reported limitation; no faster supported
  current 1x4 candidate was hidden.

- Observed anomaly: The merged four-device prefill profiler model reports
  107.5% DRAM roofline.
  Evidence: Raw Tracy CSV, final prefill operation/category summaries, and
  `perf/perf_report.md`.
  Affected path: Profiler interpretation only.
  Control or comparison: The report identifies this as a merged four-device
  model, not one device exceeding physical peak.
  Likely subsystem: Aggregated roofline accounting.
  Investigation performed: Cross-checked raw/profile summary hashes, operation
  calls, dtype/fidelity rows, and report totals.
  Resolution: controlled

- Observed anomaly: The watcher process reports nanobind leaks during Python
  teardown.
  Evidence: Tail of `watcher_final_frozen.log`.
  Affected path: Python binding shutdown.
  Control or comparison: All four tests pass, watcher checks complete, all four
  devices detach, and the cluster closes normally.
  Likely subsystem: Binding lifetime/teardown rather than decoder runtime.
  Investigation performed: Inspected the surrounding pass, watcher detach, and
  device-close sequence.
  Resolution: controlled

- Observed anomaly: Ethernet triage labels its wrapper `pass` while every
  device read is skipped.
  Evidence: `triage_eth_arc_final_frozen.log`.
  Affected path: Supplemental Ethernet health evidence.
  Control or comparison: Documentation explicitly calls the reads skipped and
  relies on current ring execution, worker watcher, ARC, and board listing for
  compensation.
  Likely subsystem: UMD Python binding compatibility.
  Investigation performed: Read every device result and compared it to
  README/work-log wording.
  Resolution: controlled

## Scope Inspected

- Goal/skill paths:
  - Exact multichip-decoder goal supplied for openai/gpt-oss-20b.
  - `.agents/skills/stage-review/SKILL.md`
  - `.agents/skills/multichip/SKILL.md`
  - `.agents/skills/tt-device-usage/SKILL.md`
  - `tech_reports/LLMs/llms.md`, section 3.3 Multi-Device
- Artifact paths:
  - `doc/multichip_decoder/stage_review_round1.md`
  - `doc/multichip_decoder/AUTODEBUG.md`
  - `doc/multichip_decoder/AUTOFIX.md`
  - `doc/multichip_decoder/README.md`
  - `doc/multichip_decoder/work_log.md`
  - `doc/multichip_decoder/evidence_manifest.json`
  - `doc/context_contract.json`
  - Every authoritative artifact referenced by the final evidence manifest,
    including current timing/candidate JSON, JUnit, endpoint PT controls,
    watcher/health logs, raw Tracy CSV, and compact profiler summaries
  - `doc/multichip_decoder/perf/perf_report.md` and final frozen CSV tables
  - `doc/functional_decoder/multichip_provenance.json`
- Code paths:
  - `models/autoports/openai_gpt_oss_20b/tt/multichip_decoder.py`
  - `models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py`
  - Relevant policy branches in `tt/optimized_decoder.py`
  - Blackhole fused-MM/RS gate in
    `models/demos/gpt_oss/tt/attention/operations.py`
- Commands run:
  - Read-only `sed`, `nl`, `grep`, `find`, `wc`, `head`, `stat`, `sha256sum`,
    `jq`, and shell filtering over source/artifacts.
  - Read-only `git branch`, `git rev-parse`, `git status`, and source comparison
    against base commit `e78e3cd110810695ca1172002deffcc6ddb97e43`.
  - No test, TT device, reset, watcher, profiler, server, or vLLM command was
    run by this reviewer.

## Residual Risk

- The reviewed decoder is deliberately batch-one and fixed to the complete
  local 1x4 Blackhole P300c ring; behavior on a different mesh or topology is
  rejected rather than generalized.
- Full-prefix nonzero correctness is qualified through S=2048, while 131072
  evidence is an endpoint capacity/execution/control gate. Later full-model
  integration should retain long-context end-to-end coverage, but no
  full-model/vLLM work belongs in this stage.
- Page-bank capture cost is amortized once per 64 tokens and is not included in
  the steady-state decoder timing. A later generator can remove or quantify
  that orchestration overhead without changing this layer-stage verdict.
- The main agent must now add this clean-pass report to the stage work log and
  create the requested local stage checkpoint commit from the isolated staged
  files, recording branch and SHA. No push is authorized.
