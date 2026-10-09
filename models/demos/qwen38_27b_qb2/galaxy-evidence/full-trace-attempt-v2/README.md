# Whole-model profiling attempt: incomplete

The 32K/B32 test loaded all 64 layers and passed deterministic warmup/trace
readback checks on four ranks. Its hardware test completed in 519.65 seconds.
CPU validation passed 367 tests plus 40 subtests.

This is **not a valid full-model timing profile**. Device profiler buffers
overflowed and dropped markers; the export then exceeded the existing 1-GiB
per-file budget (`tracy/.logs/tracy_ops_times.csv`). The guard stopped the capture,
and remaining 8K/B1/B16 cases did not run. The large incomplete export remains
on the host, not in Git. The queue and hardware receipts are preserved here.

An exit/signal race also produced a secondary ProcessLookupError in a helper;
the helper now preserves the original error if a child exits before signaling.
Earlier v1 stopped before hardware because of the missing tt-smi PATH entry.

The profiler needs reduced metadata volume and enough per-core capacity, or a
different attribution strategy, before full traced TPOT can be reconciled.
No P0, compute-utilization, or launch-overhead percentage is qualified by this run.
