# Stage Review

Verdict: clean-pass

## Required Work

- None.

## Other Concerns

- The implementation intentionally targets exactly the available Blackhole
  `MeshShape(1, 4)` ring. Other mesh sizes are rejected rather than supported
  dynamically.
- Full 32-layer model execution, full-model weight loading, generation, and
  serving are downstream stages. This stage proves the direct layer boundary
  and shared 32-layer CCL-owner contract, not a loaded 32-layer hardware run.
- The host reports a 64 MiB `/dev/shm` MPI-capacity warning, firmware 19.8.0 is
  newer than the latest fully tested 19.5.0 bundle, the repo-local Inspector
  directory is not writable, and nanobind prints process-exit leak diagnostics.
  These did not produce a correctness, watcher, device-close, or repeatability
  failure in the retained gates.

## Hard-Check Gaps

- No unresolved gap in the multichip-decoder stage contract. The 131,072-token
  capability is supported here by an exact per-device full-stack capacity plan
  and paged-cache behavior through page boundaries; a live full-length,
  full-stack run remains downstream evidence.
- `tt-perf-report` leaves several CCL operations in its unclassified category,
  but the retained per-op CSV and summary rows still expose and account for the
  exact reduce-scatter and all-gather times used by the report.

## Anomaly Ledger

- Observed anomaly: Every final TP4 projection was labeled `SLOW`, and the
  DRAM-sharded rows exposed no output-subblock field to `tt-perf-report`.
  Evidence: `tracy/analysis/multi_decode.csv`, the losslessly compressed
  canonical raw op CSV archive under
  `tracy/reports/2026_07_18_09_34_42/`, and
  `profiler_geometry_audit.md`.
  Affected path: Final BFP4/LoFi TP4 prefill and decode projections.
  Control or comparison: Eight-way DRAM geometry regressed decode to
  0.331311 ms; a full interleaved BFP4/LoFi 1-D decoder regressed prefill/decode
  to 0.989849/0.405666 ms versus the final 0.733909/0.320058 ms.
  Likely subsystem: DRAM-sharded matmul factory geometry and profiler
  introspection.
  Investigation performed: The factory source was traced through DRAM-bank
  worker assignment, its 80-core bounding launch, and internal subblock
  selection. Exact `1x6`, `1x8`, `1x7`, `1x7`, and `1x8` output subblocks and
  maximal legal input blocks were derived; same-policy alternatives were run.
  Resolution: controlled.

- Observed anomaly: The fused all-gather-matmul TP4 candidate hung.
  Evidence: `AUTOTRIAGE.md`, `AUTOFIX.md`, and `topology_results.csv`.
  Affected path: Fractured residual plus distributed RMSNorm and fused next
  projection.
  Control or comparison: The shape-faithful separate fractured boundary passed
  at minimum-rank PCC 0.9999708059 but was 1.066% slower than the replicated
  boundary.
  Likely subsystem: Generic fused AGMM transfer accounting.
  Investigation performed: Autotriage reduced the failure to a TP8-hardcoded
  eight-slice ledger on a four-rank ring; the unsafe candidate was removed and
  the devices were recovered.
  Resolution: controlled.

- Observed anomaly: Fused matmul plus reduce-scatter was initially not earned
  as a rejected topology.
  Evidence: `fused_mm_rs_audit.md`, `logs/fused_mm_rs_probe.log`, and
  `topology_results.csv`.
  Affected path: Exact TP4-local O (`32x1024x4096`) and down
  (`32x3584x4096`) row-parallel boundaries.
  Control or comparison: Matching separate interleaved BFP4/LoFi matmul plus
  reduce-scatter.
  Likely subsystem: Experimental fused CCL/matmul APIs.
  Investigation performed: The generic API was adapted to both shapes and was
  correct at minimum-rank PCC 0.999999940395, but was 26.6342% and 14.5637%
  slower. The other minimal-strided API is source-gated on Blackhole for the
  repository-tracked producer/consumer race.
  Resolution: controlled.

- Observed anomaly: The first current-source default-suite rerun failed the
  mocked 32-layer shared-CCL test because its fake base constructor did not set
  the newly read `use_advisor_1d` field.
  Evidence: `work_log.md` and the final `logs/pytest_final.log` (`5 passed, 5
  skipped in 12.89s`).
  Affected path: Static stack ownership test only; hardware default execution
  had already passed.
  Control or comparison: The real `OptimizedDecoder` constructor initializes
  the field.
  Likely subsystem: Test double maintenance.
  Investigation performed: The fake constructor contract was updated and the
  complete current-source suite rerun.
  Resolution: fixed.

- Observed anomaly: Active-ETH watcher instrumentation produced an archived
  TP2 teardown false positive after successful fabric execution.
  Evidence: `watcher_summary.txt`, archived TP2 evidence referenced there, and
  fresh `logs/watcher_final.log`.
  Affected path: Watcher teardown instrumentation, not decoder output.
  Control or comparison: Fresh TP4 run with only ETH watcher instrumentation
  disabled kept compute, dataflow, and NoC coverage enabled; all four devices
  completed ten bitwise-deterministic trace replays and detached cleanly.
  Likely subsystem: Blackhole active-ETH watcher teardown.
  Investigation performed: The earlier run was separated from model execution,
  and the final controlled watcher command was rerun on the live source.
  Resolution: controlled.

- Observed anomaly: Inspector initialization, MPI shared-memory capacity,
  firmware-version, motherboard-discovery, and nanobind teardown warnings recur
  in hardware logs.
  Evidence: Fresh fallback, default-suite, watcher, performance, fused-probe,
  and Tracy logs.
  Affected path: Host diagnostics and test-process teardown.
  Control or comparison: All required runs passed, the full four-device ring
  was detected consistently, and every device closed cleanly.
  Likely subsystem: Host environment, discovery metadata, Inspector
  permissions, and Python binding teardown.
  Investigation performed: Warnings were compared across independent runs and
  checked against device lifecycle, watcher, fallback, PCC, and determinism
  results.
  Resolution: controlled.

## Scope Inspected

- Goal/skill paths: `.agents/skills/stage-review/SKILL.md`,
  `.agents/skills/multichip/SKILL.md`,
  `.agents/skills/tt-device-usage/SKILL.md`, and section 3.3 of
  `tech_reports/LLMs/llms.md`.
- Artifact paths: `doc/multichip_decoder/README.md`, `mesh_plan.md`,
  `work_log.md`, `final_gate_results.txt`, `candidate_results.csv`,
  `topology_results.csv`, `profiler_geometry_audit.md`,
  `fused_mm_rs_audit.md`, `AUTOTRIAGE.md`, `AUTOFIX.md`, all final logs, the
  canonical Tracy raw CSV, and all four filtered/summary/provenance/table report
  families; also `doc/context_contract.json` and functional-decoder provenance.
- Code paths: `tt/multichip_decoder.py`, `tests/test_multichip_decoder.py`, the
  single-chip optimized and functional decoders, common `TT_CCL`, DRAM-sharded
  matmul factory/config source, both fused matmul-reduce-scatter APIs and tests,
  and the Blackhole minimal-strided source gate.
- Commands run: Read-only `git status`, `git diff --check`, Black formatting
  check, path/mtime checks, and targeted source/log/CSV inspection with shell
  search and text tools. No device, server, reset, or hardware command was run
  by the reviewer.

## Residual Risk

- The next full-model stage may expose aggregate 32-layer DRAM fragmentation,
  trace-region pressure, or long-lived CCL-buffer interactions not visible in a
  one-layer hardware test and static shared-owner test.
- Live cache tests reach the 63/64/65 page boundary rather than 131,072 tokens;
  maximum context currently rests on the verified capacity formula and page-pool
  contract.
- TP4 speedup is real but communication-limited: the final long run reaches
  1.694317x prefill and 1.817101x decode speedup, while the current profile
  attributes 20.76%/12.67% of TP4 decode time to reduce-scatter/all-gather.
