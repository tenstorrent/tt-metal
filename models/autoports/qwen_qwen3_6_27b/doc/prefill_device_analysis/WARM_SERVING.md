# Warm serving TTFT

Status: both HTTP benchmarks, canonical sampling and qualitative review complete.
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
Sixteen additional host guard tests pass, including unsafe allocations, late
sampler abort, program growth, slot mask refresh and bucket rejection.

Actual HTTP validation completed with unchanged full serving capacity,
startup lengths128/4096 and all-mode sampling. Results follow below. The previous
tracked experiment's success does not replace validation of this integrated
implementation, startup lifecycle, changed lengths, sampling modes and serving
asynchronous output path.


## Actual short-request HTTP result

Checkpoint4a02bf62cf5 measured **177.361ms median HTTP TTFT** at128/252/C1,
down from728.482ms: **4.107× faster**, saving551.120ms. All4 requests completed
with1008 requested output tokens. MeanTTFT180.004ms,p99195.039ms;
meanTPOT88.740ms remains essentially unchanged. HTTP is now **1.487×** the
119.301ms B1 generator prefill measurement. The60ms target still misses2.956×.

Server logs show one startup capture and four request reuses. Prefix caching is
explicitly disabled, and the configured max context262144 / total cache pool
1728448tokens is unchanged. Startup warmup took381.696ms for128,1498.708ms for
4096,286.473ms for compile-only decode and537.079ms for retained capture. Model
loading and cache-pool allocation are separate startup costs, not included in
those warmup spans or steady-state TTFT.

The result beats the pre-registered200–300ms hypothesis. The full64 adapter's
248.156ms includes first-decode readback; after removing blocking setup, HTTP
can publish the first token before that entire decode step finishes. This is an
inference from the timing boundaries, not a new server-span measurement.
The4096/252/C8 burst completed8/8 requests and2016tokens at11285.561ms median
TTFT and90.355ms meanTPOT. The previous median was11443.820ms; the1.38%
change is not attributed to reuse because C8 uses the conservative fallback.
Canonical sampling passed3 tests with1 skip in24.30s. All six greedy strings
match the device-fill checkpoint byte for byte. All sampled strings are coherent,
but the haiku stops before its final poem, stories and thermodynamics exhaust
the256-token budget, and the sampled Fibonacci example truncates. Both complete
Fibonacci functions passed n=0,1,10 checks. These are not twelve completed tasks.
The runner exited0 and both server processes closed normally.

See `artifacts/warm_serving_summary.json` and
`artifacts/warm_serving_qualitative_review.json` for structured evidence.
A third CI run34622652794 is running the same13 requested points with all eight
required lengths warmed at startup, using the fixed4a02 tag, inference-server
ed2ef012 and reused native image. All build jobs were observed skipped. The
previous two CI runs remain active on other runners.

After the measured run, two host-tested changes harden invalid-input cleanup
and telemetry: malformed slots release retained traces before raising, and the
capture counter now counts all actual captures rather than only reuse-eligible
C1 captures. They do not change the measured valid-request path. The final host
suite includes18 trace-reuse/sampler tests,12 reload/history tests and5 warmup
tests. No C++ or CMake changes require a build.
