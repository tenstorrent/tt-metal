# Multichip decoder stage review

Final independent `$stage-review` verdict: **clean-pass**.

Required work: none.

The reviewer inspected the goal and skill contracts, current implementation
and tests, compiler provenance, context plan, final PCC/cache/non-aligned/trace
artifacts, heterogeneous and long-context gates, all 13 current-hash geometry
sweeps, warmed latency, selected Tracy/`tt-perf-report` evidence, Watcher
records, the complete residual-boundary probes, AutoFix analysis, and the
two-link fused-AGMM triage/recovery record.  It confirmed the final
implementation and primary-test hashes
`b249847705594bbf49e795eecdc1669a0016929929952d0767c9939cef1dd573`
and `b6b10e32580ff5e4f9fdb465099272727a147049d5f191e91bd0dcae91ded557`.

## Findings closed in the completion audit

- The capacity ledger now counts physical BF16 norm tiles and full-sequence
  buffers retained around chunked prefill.  The batch-1 resident-plus-reserve
  total is 3,307,954,432 bytes per device, including a 512-MiB transient
  reserve.  Full logical context 32,768 remains physically executed.
- Every tuning artifact was refreshed against the final hashes.  The omitted
  12-core MLP midpoints were measured; 8/8/24/8 cores, BF16 CCL, and two links
  remains the measured winner.
- Explicit and fused residual-sharded families now carry the full boundary
  through distributed RMSNorm and the real next QKV.  Replicated is
  0.097550 ms, explicit fractured is 0.109961 ms at PCC 0.999898, and one-link
  fused is 0.110681 ms at PCC 0.999859.  All completed traces are bitwise
  deterministic.
- The two-link fused candidate's trace hang was triaged before terminating
  only the probe processes.  All four devices returned after a bounded reset
  and the `1x4` mesh smoke passed.  This slower experimental operation is
  absent from the selected decoder.
- The exact Watcher stdout is preserved and hash-bound.  Active-Ethernet
  instrumentation remains disabled only because its expanded router exceeds
  the Blackhole kernel-config buffer; worker/dispatch checks, the ring fabric,
  and both selected collectives remained active in the passing run.

## Controlled residual risks

- The experimental fused AGMM kernel has a stale debug-only transfer-count
  assertion.  Release-mode one-link execution is correct but slower, and the
  two-link trace hangs, so the candidate is rejected rather than integrated.
- Watcher stdout contains generic nanobind teardown leak warnings also seen in
  the optimized baseline.  The test passed, all four devices detached, and
  Watcher reported no decoder-path errors.
- Paged-cache behavior and heterogeneous batch positions are independently
  covered, not combined into one batch-greater-than-one case.
- Trace replay proves fixed-input determinism.  Mutating position tensors
  between replays belongs to the later generator/trace-integration stage.
- Full-context batch 32 was not executed and is not claimed.  The advertised
  32,768-token context was executed and capacity-gated at batch 1.
- Full-model stacking is outside this stage.  The replicated layer boundary is
  shape-compatible and exercised by the stacked-decoder check.

The reviewer performed no file edits or hardware actions.  The unrelated
dirty skill file remains outside stage ownership, and no push is authorized.
