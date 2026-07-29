# Final independent stage review

Verdict: **clean-pass**

## Required work

None.

## Evidence reviewed

- The original optimized-decoder goal and the `optimize`, `tt-device-usage`,
  `shard-advise`, and `stage-review` skill contracts.
- `tt/optimized_decoder.py`, its direct optimized-path tests, the context
  contract, and committed checkpoint `ed12648255e`.
- Required shard-advisor `report.json` and `final_ir.mlir`, plus the exact
  advisor-seed versus selected-default real-weight A/B at batches 1 and 32.
- Correctness, non-aligned/context, repeated-run, watcher, Tracy raw CSV, and
  advice-enabled performance-report evidence.
- The README remediation that removes the stale prefill-regression statement
  and consistently reports the retained replay as 62 device operations.

## Controlled anomalies and residual risk

- Nanobind leak diagnostics occur during Python teardown after passing tests
  and successful device close; the watcher log has no kernel assert, error, or
  hang.
- Tracy buffers fill after the retained signposted window. That window is one
  complete 62-device-op replay, and independent 200-replay timing reproduces
  the selected latency.
- The matmul factory's round-robin output grid is restored to the explicit
  rectangular 16-core grid required by RMSNorm. The conversion is visible in
  the profile and the whole-layer path passes PCC and performance acceptance.
- The advisor's exact output geometry is legal and correct but slower than the
  selected geometry at both logical batches. The DRAM-sharded family was
  retained; the exact geometry was rejected with real-weight evidence.
- Batch 1 and batch 32 both pad decode activation height to one 32-row tile.
  Capture at batch 32 plus exact-seed real-weight A/B at both logical batches
  controls the single-capture risk.
- Phi-3.5 Mini has one homogeneous dense decoder-layer kind; the tested layer
  is representative.

No TT hardware was used by the reviewer. Unrelated dirty paths were confirmed
outside this stage's scope.
