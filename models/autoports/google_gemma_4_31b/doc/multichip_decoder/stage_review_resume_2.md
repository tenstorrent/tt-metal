# Stage 04 post-correction independent review

Verdict: `clean-pass`

## Required work

None. An additional irregular-batch per-layer PCC test is not required: the
repair changes exact core-grid selection and concat-heads subcore routing, not
decoder math. Exhaustive grid/wiring tests plus real-weight, full-stack P150x4
trace capture and replay across every affected batch close the defect.

## Other concerns

- The Stage 11 Meta accuracy/readiness blockers are downstream release issues,
  unrelated to Stage 04 decoder correctness.
- The immutable Stage 04 snapshot remains historically defective; the clean
  verdict applies to the cumulative state incorporating correction commit
  `97a16e1c982a27fbc2f4e27b65dbd6b077f9e34f`.

## Hard-check gaps

None.

The original fixed-batch-32 gap is closed by:

- 146 host tests covering batches 1-32, exact irregular ranges, bounds, core
  counts, subcore selection, and all decoder call-site wiring;
- tracked P150x4 trace-ready evidence for active batches 13, 17, 19, 23, 26,
  29, and 31; and
- completion of 541/541 IFEval requests and all 17 benchmark points with zero
  failed requests, including explicit concurrency 13 and 26.

## Anomaly ledger

### Dynamic decode head grid

- Observed anomaly: the immutable Stage 04 decoder required one rectangular
  head grid and raised `ValueError: max() arg is an empty sequence` for seven
  supported active batches on the target 11x10 worker grid.
- Evidence: implementation `683adda7a3d`, the dynamic-batch `$autodebug`
  report, and the original active-batch-19 release failure.
- Affected path: TP4 decode head concatenation for irregular dynamic batches.
- Control: factorable batches preserve the rectangular path; exhaustive 11x10
  and 14x10 matrices validate every batch.
- Likely subsystem: core-grid planning and `nlp_concat_heads_decode` subcore
  configuration.
- Investigation: `$autofix` derived the exact failing set and TTNN contract;
  commit `97a16e1c982` added exact row-wise grids and matching subcores.
- Resolution: fixed. Target-mesh trace replay and complete release workloads
  cover every affected batch.

### Ethernet watcher instrumentation

- Observed anomaly: full Ethernet watcher instrumentation exceeded the
  Blackhole firmware buffer; the no-inline attempt timed out restoring an
  instrumented router.
- Affected path: watcher instrumentation, not normal decoder execution.
- Control: worker/NoC watcher passed both layer kinds with only Ethernet
  instrumentation disabled.
- Resolution: controlled by the retained failure, recovery, and clean watcher
  evidence.

### Profiler slow rows

- Observed anomaly: Stage 04 profiler reports flag decode MLP and prefill
  attention rows as slow.
- Affected path: performance only.
- Control: precision-locked geometry, placement, fabric-link, attention, and
  residual-topology sweeps selected the measured winners; TP4 beats the
  single-chip baseline in all four modes.
- Resolution: controlled.

## Scope inspected

- Original prompt and `$stage-review`, `$multichip`, `$tt-device-usage`
  contracts, plus section 3.3 of `tech_reports/LLMs/llms.md`.
- Immutable checkpoint `e1a3f724877`, implementation `683adda7a3d`, all Stage
  04 correctness/performance/watcher/profiler artifacts, and prior reviews.
- Correction commit `97a16e1c982`, `tt/decode_head_grid.py`,
  `tests/test_decode_head_grid.py`, correction provenance, tracked server log,
  release workflow log, and release notes.
- Read-only `git`, `rg`, `sed`, `awk`, `jq`, `sha256sum`, XML-header, JSON, and
  shell arithmetic commands. The reviewer used no hardware or writes.

All original and correction hashes rederived correctly.

## Residual risk

- No isolated per-layer irregular-batch PCC artifact exists. Risk is low
  because numerical operations are unchanged, structural routing is
  exhaustively tested, and the real-weight traced 60-layer stack exercised
  both attention kinds at every affected batch.
- Ethernet watcher instrumentation remains unavailable for the documented
  firmware-capacity reason; worker and NoC coverage are clean.
