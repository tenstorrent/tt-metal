# Initial independent stage review

Date: 2026-07-19

Verdict: `more-work-needed`

## Required work

- P1: The canonical final prefill measurement used the test-only 32-token MLP
  chunk rather than the production `MultiChipConfig` default of 4096. Refresh
  final correctness, warmed performance, profiler provenance, and the docs on
  the production default. Keep 32+7 only as a separate nonaligned-tail check.
- P1: The current pass did not include lower-core gate/up and down projection
  geometries. Exercise legal reduced-core variants, adapt L1 failures, and
  retain before/after evidence.
- P2: The profiler's prefill-L1-input advice was rejected using a decode-only
  shard-advisor chain. Run the whole-prefill family with all advised matmul
  inputs moved to L1 and measure it directly.

## Other concerns

- The most complex rejected candidates are represented by test harnesses and
  detailed logs rather than retained production implementations. Preserve
  enough source/configuration and exact failure or timing provenance for a
  reviewer to reconstruct each decision.
- The README reported a healthy post-watcher `tt-smi -s` state without a
  retained health artifact. Capture the final report.

## Hard-check gaps

- The reviewer did not have a separate hardware allocation and therefore
  relied on the retained XML, logs, profile CSVs, and source-backed contracts.
- Wall latency and device-profile rows come from intentionally separate
  invocations, so their comparison must be identified as accounting rather
  than a single simultaneous capture.

## Anomaly ledger

- Firmware 19.8.0 is newer than the latest fully tested 19.5.0 bundle.
- The B850M-C host uses the known bus-topology fallback; `/dev/shm` headroom is
  low; nanobind emits the pre-existing shutdown leak diagnostic.
- These notices accompanied successful runs and no watcher or model error.
- The live workspace changed after the review and requires a fresh review of
  the final source and artifacts.

## Scope inspected

The reviewer compared the requested optimized-multichip-decoder contract with
the production decoder, model-local tests, context contract, baseline and
candidate XML/log evidence, profiler CSVs, and the draft README/work log.

## Residual risk

Until the three required evidence gaps are closed and the exact final default
path is rerun, the stage cannot claim a clean pass.
