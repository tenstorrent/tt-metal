# B1 local validation

## Host checks

Candidate worker and helper scripts pass Python AST parsing. Routing preparation validates128expert IDs,20480assignments,4unique experts/token and per-device assignment counts for each case. Strict extractor synthetic check accepted complete8-device×10-iteration Dispatch/Combine coverage and rejected a missing operation row. These are not accelerator validations.

## First local launch: run-layer18-captured-xgQuTx

The launch on2026-09-11 reached TTNN import but was interrupted during initial pytest conftest loading. Traceback ends in transformers initialization recursively scanning its model directory for `image_processing_*.py`, with `KeyboardInterrupt`; Tracy wrapper reports nonzero exit at19:23:07UTC.

No pytest test case or accelerator kernel measurements are present in this attempt. The host Tracy file does not constitute a successful B1 capture, and no timing results should be extracted or compared from it. This demonstrates neither candidate correctness nor a candidate implementation failure. A completed local retry is still required.

The first interruption was issued by the root agent to inspect the slow-import stack. A second directory, run-layer18-captured-0JwbHk, records a shell launcher path problem (relative `$0` after changing directories), before test launch. Neither is a device failure.

A third attempt, run-layer18-captured-mFo19z, started at19:23:47UTC using the absolute launcher path. Validation pending; the launcher is allowing the imports to finish.

The third attempt passed imports and entered iteration0, then stopped making progress after dispatch returned and surrogate tilize compilation completed. No `B1_END` or kernel CSV was produced. After more than5minutes without the configured timeout firing, root attempted SIGINT and then terminated the test child; wrapper exit4. Root checked device holders empty; no reset was performed. No valid performance result exists from this attempt.

Root subsequently launched the manager-free diagnostic as run-layer18-captured-9aKzOp, preserving `worker.py` in the run directory. That run is pending validation.

## Manager-free attempt: run-layer18-captured-9aKzOp

The manager-free diagnostic did not reach the test body. At19:31:00UTC device0 failed firmware initialization after a10000ms wait for physical cores2-2,3-2,5-2,6-2. During teardown, device7 active Ethernet core22-25 failed to change its heartbeat, and runtime recommended a board reset. No `B1_START` appeared and no timing CSV was produced.

This leaves the subdevice-manager hypothesis untested: the new process failed before candidate dispatch/combine execution. Device holders being empty after terminating the preceding run did not establish healthy firmware state. Hardware recovery must precede another test; no reset was performed by this validation agent.

## Successful local capture: run-layer18-captured-FfQ27z

Following root's targeted recovery, the manager-free candidate completed10iterations with exit0. Official operations CSV: profiler/reports/2026_09_11_19_34_20/ops_perf_results_2026_09_11_19_34_20.csv. `timing.json` validates160target rows (8devices ×10iterations ×2operations), complete matching signpost boundaries, exactly one Dispatch followed by Combine on every device in each window, and strictly positive finite durations.

The initial extractor incorrectly expected identical raw call IDs across devices. Actual CSV encodes a device offset, e.g. first dispatch2048+d and first combine4096+d. Updated extraction uses explicit B1_START/B1_END windows as primary grouping and verifies constant call-minus-device within each operation and monotonic call progression. It preserves raw IDs for audit. It does not align unrelated rows by positional merging.

After discarding iteration0, mean of per-iteration device maxima: Dispatch1.656553ms and Combine1.725759ms. These are isolated eager kernel measurements, not summed model latency or throughput. No controlled-routing comparison has yet been established in this validation entry.

Fail-closed tests derived from the real CSV rejected each of: missing operation, duplicate operation, NaN duration, inconsistent call ID, and missing final END signpost. The successful unmodified CSV passes.

During the second round, run-layer23-captured-VMOmCU failed during device initialization/ethernet handshake and stalled during teardown. Root terminated that test child and initiated targeted recovery before retrying remaining cases. This unsuccessful pre-test sample is excluded from the timing matrix. Hardware reset occurred between successful repeats, so fresh-process repeats also differ in recovery state; the experiment does not isolate the effect of that reset. The matrix script includes only exit0 runs with complete signposted coverage and the identical worker hash.

## Final controlled matrix

All12 requested fresh-process captures completed and passed strict extraction: two per layer/routing case, with one identical archived worker hash. Aggregate validates1920 Dispatch/Combine rows, excluding192 warm-up rows from retained timing summaries. Per-case archived routing hashes also match across repeats. The matrix is reproducible with aggregate_timings.py; final conclusions and recovery-state limits are in LOCAL_FINDINGS.md.
