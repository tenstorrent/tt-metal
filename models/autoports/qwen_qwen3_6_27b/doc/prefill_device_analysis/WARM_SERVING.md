# Warm serving TTFT

Status: production implementation under device validation; no new HTTP result yet.
This is prompt-directed follow-up work, not a completed pipeline stage.

## Hypothesis and measurement contract

The selected baseline at4ea57c41431 measured728.482ms median HTTP TTFT for
ISL128/OSL252/C1. Full64 B1 generator prefill measured119.301ms. A matched
full64 adapter experiment attributed~533ms to request-boundary decode setup.
The pre-registered hypothesis for the integrated warm serving path is200–300ms
HTTP TTFT, preserving real streaming delivery, full context/pool allocation,
all prompt computation and disabled prefix caching. Warmup duration is excluded
from steady-state samples and reported separately.

## Implementation

- `generator_vllm.py` now runs real startup prefill warmup for explicit
  `QWEN36_WARMUP_PREFILL_LENGTHS` (default128,4096 within context), with logical
  lengths and physical padding logged. Decode compilation/restoration runs
  without trace capture; a second phase retains model and sampler traces.
- `trace_reuse.py` retains verified warmed C1 request envelopes, with normal
  recapture for unsupported transitions. Request inputs remain authoritative:
  tokens, positions, page-table contents and both active masks refresh in place.
- Process-start allocation tracking remains enabled, including program-owned
  buffers. Both trace maps must be empty of unsafe live allocations before
  model replay. The sampler checks again; a late failure aborts rather than
  repeating a partially executed token. Python garbage collection is omitted
  from this local checked executor; uncollected allocations remain unsafe.
- Checked replay is separate from cross-request reuse eligibility so ordinary
  sampled/seeded modes do not acquire a GC pause on every decode step.
  Shared sampling accepts an optional executor; its default remains unchanged.
- The mesh has one serialized device-submitting owner. The controller rejects
  cross-thread use; unrelated external mesh submissions are unsupported.
  Bucketed sampler traces with corruptible allocation exemptions are excluded.

## Results so far

Five host warmup tests and twelve existing seed/reload/history tests pass.
Formatting checks pass. The integrated four-layer device A/B, including startup
warmup, passed all active token/rank, cache and position comparisons across
four alternating requests. Eight recaptures became one capture plus seven
reuses. Median prefill plus first decode fell57.453→24.282ms; first decode
fell40.760→8.348ms. Both arms used the same complete checks without GC, isolating
the recurring setup change. Startup warmup measured396.175ms in this reduced
configuration. These are adapter timings, not HTTP or full-model timings.

The full64 production A/B also passed: all four alternating request comparisons
matched active tokens across four ranks, all512 cache/rank digests and positions.
Eight captures became one plus seven reuses. Median prefill plus first decode
fell798.811→248.156ms; first decode631.824→90.822ms. Full64 startup warmup
measured1178.128ms in the max-context256 harness. These remain adapter timings.
Thirteen additional host guard tests pass, including unsafe allocations, late
sampler abort, program growth, slot mask refresh and bucket rejection.

Actual HTTP validation is running with the unchanged full serving capacity,
startup lengths128/4096 and all-mode sampling. Its result is pending. The previous
tracked experiment's success does not replace validation of this integrated
implementation, startup lifecycle, changed lengths, sampling modes and serving
asynchronous output path.
