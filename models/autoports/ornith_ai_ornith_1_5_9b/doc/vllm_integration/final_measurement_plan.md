# Tracking-off serving measurements

The executable source of commands is [run_final_measurements.sh](run_final_measurements.sh). It records exact shell invocations and environment in `<label>_measurement_execution/`. The optional first argument is an artifact label only; it changes no serving configuration. Default `final` produces `final_b32_untracked/` and `final_b1_untracked/`. Fresh directories are required, and root artifacts are copied before any collection.

Prerequisites: reviewed full-model tracked sampling, mixed-shape/lifecycle evidence and qualitative results for the source being measured; all previous device work finished; no competing hardware or heavy CPU work. Allocation tracking is disabled before worker imports. Watcher/profiler are absent.

The driver serializes:

1. B32 server: context262144, block32, TP4/FABRIC_1D, decode trace, async scheduling, selected precision, sample_on_device_mode=all, operator default top_k20, thinking off.
2. P2 unchanged harness profiles at1/32 users and128 output tokens. Any failure is preserved; independent greedy benchmarks still run and the driver ultimately returns the P2 failure code. This is collection, never a waiver.
3. B32 auxiliary128/128/1/concurrency1, CI100/100/32 with concurrency unset, and prefill burst128/32/32/concurrency32. No self-comparison serial-prefill ratio is manufactured.
4. Two fresh idle metric observations, normal owned shutdown and device-owner check.
5. B1 server with the same configuration apart from capacity, then primary128/128/1/concurrency1, greedy, ignore_eos, P50/P99 timing metrics. CI is explicitly omitted.
6. Idle-only shutdown and artifact validation. Neither B1 metadata nor results are merged into B32 records.

The primary figure is1000/mean_TPOT_ms, with TTFT, TPOT and ITL P50/P99 and aggregate throughput alongside its exact workload. B32 burst TPOT is secondary because admission and chunked prefill affect it. All normalized JSONs retain their raw benchmark path and command.

No artifact is promoted automatically. Root reviews results and keeps original launch summaries, then selects the B1 primary and B32 burst files explicitly. A sampler change requires a fresh label/run and matching correctness validation. Teacher-forcing numbers are lower-bound context only, not equivalent serving throughput.
