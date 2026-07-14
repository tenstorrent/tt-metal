# AutoFix: batch-32 packed MLP and persistent-scratch coexistence

## Starting evidence

- Source diagnosis: `batch32_packed_AUTODEBUG.md`.
- Original failure: `final_suite.log` / `final_suite.xml`.
- The late batch-32 packed gate/up compile had static CB high-water
  1,163,136 while retained persistent L1 began at 854,272, an overlap of
  308,864 bytes.

## Hypothesis experiments

### Logical-M-aware gate/up topology

- Hypothesis: keep packed BFP8 byte-for-byte for M=1, but halve the dominant
  N-dependent matmul CB family for M>1 with the resident separate gate/up
  weights. Spill each BFP8 projection to DRAM before launching the next.
- Experiment: force the separate topology for the ordered batch-32 tests.
- Result: sliding batch 32 passed at PCC 0.9999113. The subsequent full-layer
  prefill still failed before decode in RMSNorm because sliding decode scratch
  remained resident: static CB end 1,237,760 versus L1 start 1,069,760.
- Verdict: verified for the packed-matmul failure, but refuted as a complete
  phase-coexistence fix. Evidence: `../candidates/batch32_separate_retry.log`
  / `../candidates/batch32_separate_retry.xml`.
- Kept fix: the final source selects packed BFP8 only for logical M=1. M>1
  uses separate BFP8 device projections, DRAM spill, device GELU/reshard,
  multiply, down projection, and TP reduction. There is no host data fallback.

### Tail-grid, physically canonical persistent scratch

- Hypothesis: reshard both row-projection partials to the final 24 row-major
  Blackhole workers before async all-reduce. The two collectives are serialized
  and have identical physical TP4 capacity, so one 57,344-byte/core scratch
  tensor can be shared while distinct semaphores retain independent epochs.
- Fix: `_persistent_ccl_memory_config` owns the tail-grid contract. The pool key
  is physical shard geometry/dtype rather than role, semaphore slot, or logical
  M; captured buffer addresses remain stable.
- Result: isolated ordered sliding/full batch-32 tests passed with PCC
  0.9998972132/0.9998811795. The full prefill ran after sliding decode created
  and replayed the shared scratch, closing the original phase-coexistence gap.
  Tests assert one 24-core scratch and an unchanged address through replay.
- Verdict: verified. Evidence: `batch32_autofix.log` /
  `batch32_autofix.xml`.

## Batch-1 regression and final performance

The M=1 packed projection is unchanged, but the collective-boundary reshard is
new. The final focused run passed both PCC tests and both warmed benchmarks:

| layer kind | PCC | prefill-128 | traced decode |
|---|---:|---:|---:|
| sliding | 0.99994830 | 2.3852985 ms | 0.4745425 ms |
| full | 0.99988662 | 2.1796695 ms | 0.5251030 ms |

This is a 1.05%/1.15% decode regression versus the earlier non-coexistent
0.469603/0.5191525 ms candidate, but remains 9.87%/8.78% faster than the
unchanged Stage-05 baseline of 0.5264925/0.575653 ms. Prefill does not regress.
Evidence: `final_latency.log` / `final_latency.xml`.

## Final status

Fixed with target-mesh PCC, trace determinism, stable persistent addresses,
prefill-after-decode coexistence, and final-default latency evidence. Remaining
gates are the normal full suite and watcher run; no speculative fixes remain.
