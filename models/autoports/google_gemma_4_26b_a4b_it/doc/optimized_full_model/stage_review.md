# Stage Review

Verdict: clean-pass

Independent inspection of Stage07 optimized-full-model for
`google/gemma-4-26B-A4B-it`, revision
`4d7ae4984b7db7de8f8457170b3f1a419ee76d52`, TP4 on four Blackhole P300c
devices. Reviewed the live `gemma-4-26b-a4b-it` worktree starting from
`941b23a758`. The reviewer performed no hardware execution, server launch,
device open/reset, or implementation edit.

## Required Work

None. The implementation and completed validation evidence satisfy this stage's
contract. Local checkpoint and telemetry finalization follow this review under
the stage owner's workflow; this verdict does not claim those later actions
have already happened.

## Other Concerns

- Long fixed-length outputs have quadratic recorder traffic. At
  `tt/generator.py:373`, indexed-fill creates an updated history and copy writes
  that history back on each replay. `AUTOFIX.md` acknowledges this, and the
  reduced profile measures at most 9.821 us for the required 127-row history.
  This is not a demonstrated bottleneck at the 128-token target. It remains a
  performance risk for much longer outputs and for short requests reusing a
  previously large capacity.
- Ethernet Watcher coverage is unavailable in this environment. The initial
  unmodified Watcher attempt fails before model execution because the ACTIVE_ETH
  program is 28,464 bytes versus a 26,624-byte config buffer. The scoped retry
  retains worker assertions and trace allocation tracking, but disables ETH
  instrumentation. This is an evidenced environment limitation, not a full
  Ethernet Watcher pass.

## Hard-Check Gaps

- Full 30-layer device latency and utilization are not measured. The explicit
  safety contract prohibits all-layer profiling. Reduced layers 0/5 plus the
  real terminal, sampler, and recorder provide the device diagnostic; full-model
  timing comes from uninstrumented autoregressive requests. Null full-model
  device-time fields correctly avoid extrapolating a reduced measurement.
- Batch 32 exercises all 30 layers and all slots for feedback/positions; detailed
  isolated logits controls cover slots 0 and 31. Inactive-slot and changed-table
  checks use the two-layer real-wrapper probe. The new buffered public generation
  path remains the existing single-request API; larger batches use the preserved
  low-level API.
- Maximum-context execution is inherited from the accepted Stage06 capacity
  tests at 262,143 and 262,144 tokens. The new history allocation is accounted
  for within the existing reserve, but maximum-length buffered generation was
  not run. No context cap or alignment restriction was introduced.
- Readiness covers 100 continuation positions from one chat-rendered AIME24
  entry. It is not full-dataset AIME accuracy; the documentation says so.

## Anomaly Ledger

- Observed anomaly: headline throughput changes by only about 0.20%.
  Evidence: `baseline/performance.json`, `performance_comparison.json`, and
  `performance.json`; reviewer recomputed both three-request medians.
  Affected path: full 4096-input/128-output/B1/C1 autoregressive generator.
  Control or comparison: original baseline 49.256080 t/s/user, same-head
  streaming control 49.264327, selected buffered default 49.354078.
  Likely subsystem: host output boundary was a contract issue rather than a
  large latency bottleneck at this workload.
  Investigation performed: exact control token comparisons, replay/readback
  counters, source inspection, reduced recorder profile.
  Resolution: controlled. The README calls throughput essentially flat and
  claims the verified removal of per-token reads.

- Observed anomaly: allocation warnings occur with an active trace in ordinary
  runs, including batch 32 and profiling.
  Evidence: `trace_full_batch32.log`, `profile.log`,
  `buffered_extended_watcher.json`, and `trace_page_tables.json`.
  Affected path: trace setup and request reuse.
  Control or comparison: focused allocation-tracked recorder, growth/rebind,
  request reuse, sampled/greedy alternation, reset, and changed-table checks.
  Likely subsystem: generic allocator warning versus lifetime of temporary
  allocations outside capture.
  Investigation performed: source confirms capacity growth releases/rebinds
  traces; tracker-enabled tests pass; buffered outputs equal controls and
  persistent feedback state advances correctly.
  Resolution: controlled for the changed path. The review does not represent
  the untracked batch-32 run as a full allocation-tracker audit.

- Observed anomaly: unmodified Watcher cannot initialize fabric.
  Evidence: `buffered_extended_watcher.log:27`; scoped retry log reports
  `disabled features: ETH` and completes successfully.
  Affected path: instrumentation setup before model execution.
  Control or comparison: successful list/reset/list and mesh smoke; passing
  worker-Watcher plus allocation-tracked reduced-path checks.
  Likely subsystem: instrumented Ethernet firmware/config capacity.
  Investigation performed: exact overflow retained; scoped workaround and
  unavailable coverage disclosed in the work log.
  Resolution: controlled environment limitation, with residual coverage gap.

- Observed anomaly: reduced profiled prefill is much slower than ordinary
  uninstrumented execution, and the Tracy GUI copy warns about a missing file.
  Evidence: `profile.log`, `profile/summary.json`, and native CSV whose SHA256
  matches the summary.
  Affected path: diagnostic profiling/report export.
  Control or comparison: 667.192 ms whole-device prefill window fits inside
  the 667.644 ms host signpost; native report generation succeeds after the
  optional GUI-copy warning. Decode is 3.083 ms device versus 3.281 ms host
  for the same reduced signposted window.
  Likely subsystem: eager instrumented dispatch and optional GUI artifact copy.
  Investigation performed: raw report availability/hash checked; terminal
  dtype/config rows and trace IDs inspected; full-model numbers remain separate.
  Resolution: controlled. No reduced-profile time or per-op utilization is
  presented as full-model performance.

## Scope Inspected

- Goal/skill paths: supplied Stage07 contract; `.agents/skills/stage-review`,
  `optimize`, `multichip`, `full-model`, `tt-enable-tracing`, `qualitative-check`,
  and `tt-device-usage` skill instructions; shared Tenstorrent review evidence
  rules. The specific contract preserves Stage05's policy/rejection ledger and
  forbids all-layer profiling and broad datatype/vLLM work.
- Implementation: full `tt/generator.py` and `tt/model.py`, their diff from
  the accepted checkpoint, common sampling trace execution, runtime audit,
  indexed-fill validation/factory/kernel contract, and the new/used test
  runners. Changes are Python/documentation only; no native build is needed.
- Accuracy and text: `readiness.json`, verified reference hash and metadata,
  all six actual TT/HF shared qualitative texts and prompt rendering metadata,
  Stage06 token controls, buffered-prefix comparisons, refreshed sky completion,
  and degeneracy output. Prefill top1/top5/top100 is 96/100/100%; traced
  teacher forcing is 94/100/100%. All six Stage07 TT token sequences exactly
  match Stage06; buffered prefixes match EOS-stopping controls.
- Trace/state: recorder-only and real-wrapper reports, extended worker-Watcher
  run, all-layer batch-32 run, mixed active/inactive slots, page-table refresh,
  reset/repeated requests, seeded top-k16/top-p0.9 mode alternation, and prompt
  lengths 31/32/33, 127/129, 1023/1024/1025, and 4097. Source and audit evidence
  preserve common split sampling with `tt_out_tok`, device-owned positions,
  nonblocking replay, and zero per-token host refresh/readback in the measured
  fixed-length loop. One final output transfer remains included in timing.
- Optimization: terminal sampler comparison, all five native LM-head candidate
  JSONs/logs and the 44-row CSV, operation-topology/advice disposition, native
  reduced profile and compact tables, stack comparison and context accounting.
  Reviewer verified 29 passing head cases and 15 exact allocation rejections;
  passing cases retain all four local top-1 tokens and PCC at least 0.999.
  The runtime head row confirms BF16/HiFi4 and K block 4. Adapted DRAM-sharded
  families lose whole-path latency; rejection is supported beyond first errors.
- Preserved decoder contract: accepted Stage05 README, residual contract, policy,
  final performance findings and summaries; decoder implementation has no Stage07
  diff. No new inter-layer gather, precision fallback, or rejected layout is
  introduced. Stack diagnostic is 19.842905 ms versus full token-out 20.261750
  ms, a 2.11% cross-run host-wall difference, correctly labeled as such.
- Provenance/checks: raw-profile SHA256, reference SHA256, terminal fixture
  SHA256 across candidate files, JSON medians, six-prompt token equality,
  compressed-table byte identity, retained pre-commit/stage-gate logs, and
  `git diff --check`.
- Commands run by reviewer: read-only `git status`, `git diff`, `git log`,
  `git branch`, `cat`, `sed`, `grep`, `find`, `head`, `tail`, `nl`, `sha256sum`,
  and small Python JSON/CSV/hash/gzip analysis scripts. `rg` was unavailable.
  No test runner or model code was executed by the reviewer.

Reviewed implementation SHA256 values:

```text
tt/generator.py 90e3bb5d36ac933447f65771254c86bfbbd346d66d25af9639960f7d39e26b28
tt/model.py     1b0f45ba362b47c5411451ef588878b23a08bee76c8b94c11c10e62b7ddddb20
```

## Residual Risk

The accepted decoder precision policy has finite accuracy margin and the current
quality gate is deliberately small. Long-output recorder scaling, full Ethernet
Watcher coverage, and performance at larger batch/context combinations are not
established by this stage. These limits are explicit and do not contradict the
validated Stage07 target, preserved capability contract, or reported results.
