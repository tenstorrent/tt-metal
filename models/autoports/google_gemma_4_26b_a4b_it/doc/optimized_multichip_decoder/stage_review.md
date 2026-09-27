# Stage Review

Verdict: clean-pass

Independent inspection of Stage 05, optimized multichip decoder for
`google/gemma-4-26B-A4B-it`, completed 2026-09-27 against the live worktree
based on `adcb0e8f21`. This verdict covers the decoder stage and its stated
TP4 contract. It does not certify a complete model or serving integration.

The reviewed runtime SHA256 is
`6938c6d512d19cf25c3c9d7fb5cee29f2b3f46d174426b715180527b16e02cd3`;
the final runner is
`3ea427ba49605809a48bdcc0bebc5c65fb99eeefb60c34c7d7307449a24d8bec`.
The native all-gather writer is
`14ca782e9b60777f75e3d5a8e3ed8700bc2508cc6af0b0efaac86941d35b490a`;
the matmul placement utility is
`c97504644b95b8ade64522597fb9f7c4405dcc6211fc53da5a10adcdadd805e1`.

## Required Work

None for this stage. The owner can now record acceptance and create the
authorized local, stage-owned checkpoint commits. No push is part of this
verdict. Status and commit bookkeeping following this review do not change
the reviewed runtime or measurements.

## Other Concerns

- Accuracy has limited margin in some expanded cases. Real adjacent sliding
  layers 0→1 at 33 tokens reach a minimum PCC of **0.995049733**, against the
  unchanged 0.995 threshold. All tested values are finite. This is disclosed
  numerical risk, not evidence of a failed gate.
- The final ordinary host decode medians are **653.344 µs sliding** and
  **701.861 µs full**, with prefill **93346.919 / 79004.977 µs**. Full decode
  improves 3.17% over the reproduced baseline. Sliding is 0.46% slower than
  the original lower-precision baseline and 0.48% faster than the
  accuracy-matched unoptimized control. The report correctly retains this
  tradeoff and does not claim the earlier, rejected candidate's speed.
- A shared `CollectiveBufferPool` is valid only for serial layers on the same
  mesh and command queue, with private semaphore sets and lifetime extending
  through every captured trace. Concurrent requests, threads, or queues must
  not share it. The final layer output has independent DRAM storage.
- Maximum-context reservation tests preserve the 262144-token contract.
  They are not a proof of a complete model's allocation order, fragmentation,
  or 32 simultaneous maximum-context requests. The documentation makes these
  distinctions explicit.

## Hard-Check Gaps

- The required Docker build wrapper was attempted but Docker was unavailable.
  The changed matmul translation unit compiled and the native library linked
  with the existing toolchain; the endpoint writer JIT-compiled in successful
  Watcher probes. Three native reader-count regressions and two native
  endpoint regressions passed. This supports the changed paths, while the
  standard CI-image build remains unverified for an environment reason.
- Raw native CSV/Tracy captures and tensor fixtures remain local. Converted
  tt-perf-report CSVs/tables, native policy rows, whole-layer window CSVs,
  compact logs, and exact source-delta patches are preserved, including gzip
  copies where required by repository hooks, with a SHA256 manifest. This
  review inspected the raw captures before closure; it does not imply they
  are included in the checkpoint.
- This reviewer performed source and artifact analysis only. No TTNN import,
  device opening, test execution, server, or subreviewer was used. Hardware
  results below are checked owner-produced evidence, not claimed independent
  reruns.

## Anomaly Ledger

### Secondary DRAM-reader mesh query

- Observed anomaly: multi-reader DRAM matmul placement passed a mesh to a
  physical-device hop query.
- Evidence: `AUTOFIX_dram_mesh.md`, native build/link records, reader-count
  regression, and real `fixed_*dram*` candidates.
- Affected path: DRAM-sharded candidate projections; matmul placement utility.
- Control or comparison: reader counts 1/2/3 on the actual 1×4 mesh, all
  replicas compared to the reference, followed by real 4096/128 candidates.
- Likely subsystem: physical worker placement.
- Investigation performed: inspected the narrow device-selection change,
  compilation provenance, regression source, and retained results.
- Resolution: **fixed**. Adapted DRAM candidates remain slower; API failure
  alone is not used to reject them.

### L1 residual mixed-precision corruption

- Observed anomaly: the mixed FP32/BF16 fast add corrupted the adapted L1
  residual candidate.
- Evidence: `AUTOFIX_l1_residual.md`, controlled residual results.
- Affected path: optional L1 residual family.
- Control or comparison: the supported SFPU add repairs correctness; the
  coherent whole-layer candidate remains slower.
- Likely subsystem: native BinaryNg register capacity for this combination.
- Investigation performed: reviewed source diagnosis, isolated controls, and
  the retained supported workaround.
- Resolution: **controlled**. The selected default does not use the faulty
  candidate path; no claim is made that the general native defect was fixed.

### CCL worker coverage and buffer capacity

- Observed anomaly: an inherited 8×8 semaphore grid omitted Blackhole workers
  at x=8/9, causing replica divergence; private persistent payloads for all
  layers also collided with prefill L1 allocations.
- Evidence: `AUTODEBUG_ccl_semaphore_grid.md`, `AUTODEBUG_ccl_pool.md`,
  `pool_equivalence.json`, final stack/capacity results, CPU regression.
- Affected path: asynchronous collectives and inter-layer reuse.
- Control or comparison: full 11×10 private semaphore coverage repairs the
  real adjacent-stack case; pooled/private outputs agree bitwise in the
  controlled comparison. All four final maximum-context cases pass with the
  actual primed pool and 30 semaphore sets.
- Likely subsystem: semaphore address ownership and aggregate L1 lifetime.
- Investigation performed: inspected worker coverage, pool keys, all reuse
  dependencies, retained outputs, and physical reservation accounting.
- Resolution: **fixed** for the documented serial pool contract. The selected
  pool is 2690688 bytes/device; conservative DRAM bound is 29206137856
  bytes/device, including 1754480640 extra resident weight bytes.

### Precision and invalid mixed-stack fixture

- Observed anomaly: cheaper sliding policies failed real adjacent 0→1
  accuracy; full WO BFP4 failed batch 32. A separate 0→5 diagnostic skipped
  four real layers and could not legitimately veto full QKV BFP4.
- Evidence: `AUTODEBUG_stack_precision.md`, real adjacent fixtures and final
  stack/contract JSONs, retained diagnostic results.
- Affected path: attention/expert precision and accuracy harness context.
- Control or comparison: sliding QKV/expert gate BFP8 with BF16 attention
  CCL passes actual 0→1. Full QKV BFP4 with WO BFP8 passes actual 4→5 using
  HF layers 0–3 to construct the exact input history.
- Likely subsystem: cumulative arithmetic sensitivity and fixture semantics.
- Investigation performed: inspected fixture generation, routing controls,
  candidate dtype propagation, and native final dtype/fidelity rows.
- Resolution: **controlled**. Synthetic nonadjacent failures remain diagnostic;
  real adjacent, batch, cache, and replay gates select the current policy.

### Fused gather K44 corruption

- Observed anomaly: fused QKV produced rank-local cache NaNs and bad outputs.
- Evidence: original `final_policy_sharded_qkv_agmm.json`,
  `AUTODEBUG_sharded_regression.md`, K22 control and six `repaired_sharded_*`
  results.
- Affected path: fused gather/matmul candidate geometry.
- Control or comparison: K22 alone restores correctness; K44 exceeds the
  22-tile local readiness slice. Six final-policy QKV AGMM, WO AGMM, and MMRS
  comparisons pass at roughly 803–809 µs sliding and 860–884 µs full.
- Likely subsystem: fused kernel readiness/block divisibility.
- Investigation performed: checked the constructor guard, exact source delta,
  original anomaly preservation, and compatible carried-width704 residuals.
- Resolution: **fixed**. Corrected families lose whole-layer latency without
  an immediate replicated restore inside the measured layer.

### Watcher endpoint assertion

- Observed anomaly: one-worker Linear all-gather asserted while looking up a
  nonexistent outward fabric connection. NCRISC's CRBW was downstream, not
  the asserting site.
- Evidence: preserved Watcher failure/triage, `AUTOFIX_watcher_ag.md`,
  focused BF16/BFP8 probes, native regression, `final_watcher_summary.json`.
- Affected path: native minimal all-gather writer at mesh endpoints.
- Control or comparison: eight traced focused replays have zero mismatches
  on all four ranks; both final layer kinds and both actual adjacent stacks
  pass Watcher after the fix.
- Likely subsystem: unconditional lookup of an unused endpoint connection.
- Investigation performed: independently checked every pointer use against
  target/loop guards. Required routes retain their assertions; allocation,
  local writes, CB draining, and required data movement are unchanged.
- Resolution: **fixed**. Reuse of earlier capacity evidence is supported by
  the exact native source-equivalence record and unchanged allocations.

### NaN-sensitive acceptance predicate

- Observed anomaly: Python `min()` could ignore a NaN after a finite prefix;
  the failed fused run demonstrated this in its cache vector.
- Evidence: `finite_gate_source_delta.patch`, `finite_gate_audit.json`,
  `finite_gate_regression.log`, updated output/cache/batch/stack gates.
- Affected path: host acceptance math, not device execution.
- Control or comparison: the actual helper rejects NaN, both infinities,
  below-threshold and empty vectors. Two durable host tests pass; all 42
  previously accepted final vectors remain finite and passing.
- Likely subsystem: host numeric predicate.
- Investigation performed: raised the finding, inspected the repair and
  host-only equivalence proof, and checked retained vectors.
- Resolution: **fixed**. Original measurement/source hashes are preserved.

### Small placement gains and profiler inflation

- Observed anomaly: initial producer-L1 gains were small; profiled prefill
  device windows greatly exceeded ordinary warmed host timing.
- Evidence: `prefill_producer_summary.json`, 24 result/command pairs, final
  native captures, `final_perf_findings.md`.
- Affected path: performance selection and reporting.
- Control or comparison: marginal placement orderings reverse in repeated
  15-sample runs; combined full QKV/WO/router L1 gains only 0.061% against a
  30-sample control. Shared-input L1 loses about 2.3–2.4%. Historical and
  current profiled prefill windows both include substantial dispatch gaps.
- Likely subsystem: measurement variability and instrumentation overhead.
- Investigation performed: inspected producer interception/memory evidence,
  rederived all 24 medians and gates, and recomputed native whole-layer spans.
- Resolution: **controlled**. DRAM producers remain selected; no untried
  placement advice is deferred and no prefill speedup is claimed.

## Scope Inspected

- Contract and skills: Stage05 task contract; repository `AGENTS.md`;
  `.agents/skills/{stage-review,optimize,tt-device-usage}/SKILL.md`.
- Implementation: `tt/multichip_decoder.py`, relevant inherited optimized and
  fused helpers; main, batch and stack harness changes; finite-PCC and CCL
  CPU tests; DRAM matmul placement change/test; native all-gather writer/test.
- Evidence: Stage04 baseline/topology/advice records; Stage05 candidate
  commands, JSONs and investigations; README, work log, residual/context and
  memory contracts; final validations and source-equivalence proofs; both
  final native profiles and compact telemetry packet.
- Commands: read-only `git diff`, `grep`, `sed`, `find`, and Python standard
  library JSON/CSV/hash/statistics/gzip analysis. No target code was executed.
- Final audit: all 18 enumerated validation artifact hashes and pass flags
  match; all 24 placement records match passing source JSONs and their
  medians; all 307 preserved archives decompress to recorded hashes. Packet
  top-level keys match its supplied template, and per-kind latency/roofline
  values match final profiles; undefined dataset population is explained.
- Native profiling: independently reconstructed all 516 phase/session/device
  groups per kind, with four devices and 128 decode sessions. Whole-layer
  device times exactly match **329194.676 / 710.513 µs sliding** and
  **339373.403 / 744.896 µs full**. All 20 provenance hashes per profile match.
  The denominator includes complete layer gaps, not matmul subtotals. Useful
  active-eight FLOPs and estimated four-device DRAM traffic are explicitly
  distinguished from hardware counters and mixed-fidelity theoretical peaks.

## Residual Risk

The reviewed scope demonstrates the stated decoder workload, direct BF16
replicated inter-layer boundary, indexed top-eight decode execution, B32
ownership/replay checks, nonaligned inputs, maximum context with reservations,
and Watcher-clean selected paths. It does not establish full-model generation
quality, full-stack allocation order, or concurrent shared-pool safety.
Future integration must preserve the documented ownership and layout
contracts and revalidate cumulative accuracy; the narrow 33-token PCC margin
should remain visible. No unresolved anomaly blocks this stage.
