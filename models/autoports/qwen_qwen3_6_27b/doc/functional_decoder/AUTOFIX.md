# AutoFix Report

## Starting Evidence

- Source: `AUTODEBUG.md` and the first independent stage-review verdict.
- Failure: zero-weight fixed-input traces could not prove numerical replay,
  cache/page routing, batch-row isolation, or long linear state.

## Hypothesis Experiments

- Hypothesis: the trace failure was an evidence-harness issue.
  Experiment: nonzero synthetic weights, eager compile, exact mutable-state
  reset, stable device-buffer updates, two sequential replay steps.
  Result: verified and fixed. Full/linear batch 1/32 replay PCC spans
  0.998809–0.999991. Artifacts: `autofix_trace_cache/`.
- Hypothesis: paged allocation/routing could alias at non-aligned context.
  Experiment: seq65 prefill then pos65 decode with page table `[[1,0]]`, HF
  comparison, and physical cache-slot assertions.
  Result: allocation bug verified and fixed:
  `batch * ceil(context/page)`. Routing PCC 0.999905286.
- Hypothesis: linear prefill retained O(sequence) live output references.
  Experiment: model retention at advertised context and exact reduced recurrence
  through sequence 262145.
  Result: the first balanced-concat fix was insufficient: a 262143 run stayed
  CPU-bound after ~36 minutes. A second AutoFix replaced per-token dispatch
  with 64-token vectorized convolution and a logarithmic affine scan. Target
  scan PCC is 0.999763906; full-layer seq5/seq65 PCC is
  0.999998050/0.999997842. Public target-shape seq192511 passes in 474.957 s.
  Seq262143 fails at a hard 9,126,805,504-byte MLP DRAM allocation.
- Hypothesis: final numerical paths are watcher/fallback clean.
  Experiment: batch32 nonzero traces with `TT_METAL_WATCHER=10`; batch1 traces
  with `throw_exception_on_fallback=True`.
  Result: verified. Watcher logs contain no fatal/assert/NoC/L1/CB/sanitize
  signature and both fallback-audit runs pass.

## Final Status

Fixed. Nonzero traced correctness, batch32 isolation, cache-dependent paged
prefill/decode, bounded long linear state, watcher-10, and dynamic fallback
evidence pass. Device recovery reset only exact failing boards 3 and 2; a
post-reset 1x1 mesh smoke passed.
