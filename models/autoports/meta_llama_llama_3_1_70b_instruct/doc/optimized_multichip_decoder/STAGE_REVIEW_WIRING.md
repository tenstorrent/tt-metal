# Independent stage rereview: SDPA wiring

Date: 2026-07-19

Verdict: `more-work-needed`

## Required work

- P1: Production prefill SDPA wiring and retained final evidence disagreed.
  `MultiChipConfig` enabled `explicit_sdpa_program_config`, and the attempted
  `prefill_sdpa_program_config=None` override had been placed in
  `Provenance2DDecoder`, not production `MultiChipDecoder`. The retained raw
  Tracy prefill row consequently showed an explicit 8x8/64/64 program while
  README/work log claimed implicit prefill. Move the override to the production
  constructor, add a recurrence assertion, and rerun paired A/B, canonical
  final, Tracy/`tt-perf-report`, watcher, health, and static gates. The refreshed
  raw prefill row must record `program_config=std::nullopt`.
- P2: The mandatory wall/device/theoretical three-number performance accounting
  was incomplete. Report theoretical roofline latency from modeled stored-dtype
  bytes divided by aggregate bandwidth, cross-checked against device time
  multiplied by modeled DRAM utilization.

## Other concerns

- Complex fused-CCL and residual candidates retain detailed XML/log provenance
  but not their transient final implementations; exact source reconstruction is
  weaker than for durable harness candidates.
- Exclude the generated
  `shard_advise/__pycache__/advise_llama70b_tp4_local.cpython-312.pyc` from the
  checkpoint commit.
- The stage checkpoint commit was correctly pending the clean review gate.

## Hard-check gaps

- This read-only review did not use hardware and relied on retained JUnit,
  logs, raw/filtered profiler CSVs, source, and health output.
- Wall and profiler measurements are separate invocations; the docs identify
  this correctly.
- Optimized runtime evidence is batch 1. Code remains batch-parameterized, but
  no current-pass larger-batch optimized run is retained.
- No standalone compile log existed, though the final suite imported/executed
  both files and `git diff --check` passed during review.

## Anomaly ledger

- Explicit prefill in the retained raw profile versus documentation claiming
  implicit prefill: unresolved at review time; production wiring/config
  propagation was the likely subsystem.
- Large first-op/signpost Tracy gaps: controlled by separating range-entry
  artifacts from subsequent steady replay gaps.
- Firmware 19.8 versus tested 19.5, B850M-C topology fallback, low `/dev/shm`,
  and nanobind shutdown diagnostics: controlled environment/tooling notices.

## Scope inspected

The reviewer inspected the goal and selected skills, README/work log, both
prior review records, context contract, all candidate evidence, baseline/final
Tracy tables and raw CSVs, canonical JUnit/log, watcher/health evidence,
shard-advisor outputs, production multichip/base optimized decoder code, and
model-local tests. No files were edited and no hardware command was run.

## Residual risk

The chunk-4096/tail, lower-core geometry, whole-prefill L1-input, and retained
health findings were otherwise closed. A fresh review after regenerated
artifacts was required to establish a stable clean pass.
