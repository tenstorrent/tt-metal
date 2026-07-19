# Final independent stage review

Date: 2026-07-19

Verdict: `clean-pass`

## Required work

None.

## Other concerns

- Several complex fused-CCL/residual candidates retain exact XML/log evidence
  but not their transient implementations. Their adapted retries,
  configurations, failures, and successful final measurements are sufficient
  to support the recorded rejections.
- `*.log` and `*.csv` are gitignored. The checkpoint must force-add the
  evidence tree while excluding the unrelated
  `.agents/skills/forge-functional-decoder-from-ir/SKILL.md` change.

## Hard-check gaps

- This read-only review used no hardware; it relied on source, JUnit, logs,
  compressed raw Tracy data, rendered reports, watcher output, and device
  health evidence.
- Wall and device-profile measurements are adjacent but separate invocations.
  The documentation labels this correctly.
- Optimized runtime evidence is batch 1. This matches the accepted advertised
  context contract; larger-batch runtime evidence was not retained.

## Anomaly ledger

- Production prefill wiring previously contradicted its documentation. The
  override is now in `MultiChipDecoder.__init__`, not `Provenance2DDecoder`;
  final raw prefill records `program_config=std::nullopt`, while decode records
  explicit 8x8 HiFi2. Resolution: fixed.
- A stale pre-fix `final_tracy.log` referenced deleted 07:53 output. The
  authoritative 08:14 compressed raw CSV and rendered reports superseded it,
  and the stale file was removed. Resolution: fixed.
- Three decode gaps of 5671.646, 5671.403, and 5671.030 us appeared in merged
  Tracy output. The remaining five-replay steady gaps total 238.385 us versus
  597.632-us wall and 583.918-us device replay. These are cross-device
  signpost/range-entry artifacts. Resolution: controlled.
- `tt-perf-report` modeled 106,954,752 bytes/replay, less than the complete
  120,324,096 stored BFP4 weight bytes/device. The analytic 235.03-us lower
  bound is kept separate from the 208.896-us profiler-model cross-check.
  Resolution: controlled.
- Firmware 19.8 versus tested 19.5, B850M-C topology fallback, low `/dev/shm`,
  and nanobind leak diagnostics remain environment/tooling notices. Clean
  watcher, passing final suite, and zero GDDR errors control the risk.

## Scope inspected

- Skills and contract: `$stage-review`, `$optimize`, `$tt-device-usage`, and
  the original optimized-multichip-decoder goal.
- Artifacts: README/work log, both earlier review records, all candidate JUnit
  XML and relevant logs, final default, refreshed Tracy raw/tables/CSVs/plots,
  watcher, health, static gates, shard-advisor report/IR/decision trace, and
  `doc/context_contract.json`.
- Code: production `tt/multichip_decoder.py`, model-local
  `tests/test_multichip_decoder.py`, and relevant base SDPA configuration.
- Commands: read-only git, find/stat/grep/sed, and artifact/JUnit/profile
  parsing. No hardware, server, test, mutation, or commit command.

The refreshed evidence is newer than the production fix: source at 08:10,
paired A/B at 08:11, canonical final at 08:13, raw Tracy at 08:14, watcher at
08:16, health at 08:17, and static gates at 08:20.

The final evidence establishes:

- explicit decode program/compute, implicit prefill program, default chunk
  4096, and separate 32+7 tail coverage;
- paired prefill `1.267830 -> 1.266436 ms`, eager decode
  `1.572428 -> 1.454311 ms`, traced decode `0.603287 -> 0.598957 ms`, PCC
  `1.0/0.9999979552`, and K/V `1.0/1.0`;
- canonical TP4 prefill/eager/traced latency
  `1.281216/1.507943/0.597632 ms` with accepted absolute PCC;
- 1018.519-us prefill device work, 583.918-us decode device replay, about
  235.03-us analytic decode roofline, and 208.896-us profiler-model check;
- persistent async RS+AG, packed projections, DRAM-sharded BFP4/LoFi matmuls,
  and no inter-layer mesh collective;
- adapted evidence for all material residual, fusion, packing, collective,
  precision, fidelity, advisor, lower-core, and prefill-L1 families;
- clean fallback audit, nonaligned/paged/dynamic coverage, full 131072-token
  batch-1 contract, watcher stress, and post-run device health;
- no full-model, LM-head, generation, or vLLM work.

## Residual risk

- Evidence uses one representative real layer kind plus direct two-layer
  composition rather than an 80-layer construction, correctly outside scope.
- Batch greater than one remains unmeasured in this pass.
- Transient rejected-candidate source reconstruction is weaker than the
  retained final/default harness, though evidence is sufficient for decisions.
