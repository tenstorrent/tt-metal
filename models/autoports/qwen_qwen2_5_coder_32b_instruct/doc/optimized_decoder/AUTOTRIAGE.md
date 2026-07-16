# AUTOTRIAGE

## Diagnosis

- The original functional trace attempt hit a capture-time program-cache miss because warmup used `current_pos=17` while capture used `current_pos=18`; TTNN slice bounds are part of the program signature.
- That fatal capture left pytest in mesh-device teardown, waiting forever for a worker-completion count that the aborted trace could no longer satisfy. A second benchmark was started while that first process still owned the device, so it was not a valid independent repro.

## Triage Evidence

- Live `tt-triage` completed and reported ARC heartbeats, DDR status, watcher ringbuffer, and broken-component checks as passing. Its detailed worker reads were unavailable because the installed Exalens/UMD binding rejected the triage tool's `noc_read` signature; this limits device-side stop-site evidence.
- Host GDB attached to the original hung PID. The main thread was blocked in `FDMeshCommandQueue::clear_expected_num_workers_completed()` from `FDMeshCommandQueue::~FDMeshCommandQueue()`, reached through `MeshDevice::close()` during pytest teardown.
- The first test log had already emitted `Writes are not supported during trace capture` followed by `Reads are not supported during trace capture`, so the teardown wait is downstream of a trace-capture abort, not evidence of a decoder kernel deadlock.

## Source Evidence

- The benchmark originally warmed decode at position 17 and captured at position 18. Both functional and optimized decode slice RoPE tables and position indices with Python integer bounds, and those bounds select a distinct cached program.
- `$tt-enable-tracing` explicitly identifies integer slice begins/ends as compile-time signature fields and requires an exact-value warmup immediately before capture.
- The test now defines one `trace_pos`, runs a complete eager forward at that exact value, synchronizes, and captures the identical operation sequence and arguments.

## Downstream Effects

- The first process remained in device close after the fatal capture.
- The second benchmark overlapped the same physical board and therefore could not provide meaningful device or latency evidence.
- No timing from either process is retained.

## Proposed Fix

- Terminate both stale pytest processes only after triage and host backtrace capture, reset the board, then rerun one benchmark at a time.
- Keep exact-value warmup and capture positions identical.
- Wrap future trace benchmarks in a bounded timeout so a fatal capture cannot silently leave an overlapping owner.

## Uncertainty

- The installed triage stack's `noc_read` ABI mismatch prevented per-core call-stack and running-op reads. The host teardown stack plus the preceding fatal trace messages are sufficient to classify this incident, but a future independent device hang should be captured with the matching triage environment.
