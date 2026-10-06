# Profiler close abort investigation

Source/CPU-only investigation, 2026-09-25. No hardware commands or production-source edits were performed. The original `profile_sliding.log`, `profile_sliding.json` and `tracy/sliding/raw/.logs/` were preserved.

## Starting evidence

The real sliding S4096 run with 128 successive traced decode inputs passed its recorded correctness checks, printed `PERF_DECODE_END`, released its trace, and then aborted during close (`profile_sliding.log:20300-20346`). Neither `cpp_device_perf_report.csv` nor `profile_log_device.csv` was written. No readable core dump was found in the checked workspace, `/tmp`, or `/var` crash locations; the kernel core pattern points to an Apport pipe. There is no native abort backtrace in the log.

The saved host-event export completes 114 `readControlBufferForCore` zones and the containing `readControlBuffers` zone, but no completed `readProfilerBuffer`/`readResults` zone. Source close already calls `ReadMeshDeviceProfilerResults(...LAST_FD_READ)` (`tt_metal/distributed/mesh_device.cpp:964-967`). That function drains queues and reads results (`tt_metal/impl/profiler/tt_metal_profiler.cpp:1205-1248`); the read path performs control reads immediately before the bulk DRAM read (`tt_metal/impl/profiler/profiler.cpp:2593-2598`). A missing explicit harness read does not explain the failure.

## Verified source defect: uint32 host-allocation overflow

`DeviceProfiler::setProfileBufferBankSizeBytes(uint32_t size, uint32_t num_dram_banks)` allocates with:

```cpp
profile_buffer.resize(size * num_dram_banks / sizeof(uint32_t));
```

Both multiplication operands are uint32, so the product wraps **before** conversion/division by `sizeof(uint32_t)` (`tt_metal/impl/profiler/profiler.cpp:2392-2394`). The read still requests the full per-bank size into this vector (`:1369-1381`).

The first run compiled with `PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC=7200000`, matching 150000 programs × 48 bytes/program (`profiler_state_manager.cpp:58-80`). Blackhole uses five processors per core (`tt_metal/hw/inc/internal/tt-1xx/blackhole/core_config.h:34`) and this run has eight DRAM banks. Allocation multiplies by the ceiled total TENSIX+ETH core count per bank (`metal_soc_descriptor.cpp:369-388`; `tt_metal_profiler.cpp:788-798`).

The 114 observed control reads establish a lower bound on physical core count, not equality: the read list contains logical compute cores and active Ethernet cores, while allocation includes all TENSIX and Ethernet cores (`tt_metal_profiler.cpp:1054-1083`). The run's compiler defines show 110 L1 banks, and Blackhole has at most 14 ETH cores (`tt_metal/llrt/hal/tt-1xx/blackhole/bh_hal.cpp:44-49`). The applicable ceiled count is therefore 15 or 16. Both overflow:

| Programs supported | Cores/bank | First bank read, bytes | Required host allocation, bytes | Actual wrapped allocation, bytes |
| --- | --- | --- | --- | --- |
| 150000 | 15 | 540000000 | 4320000000 | 25032704 |
| 150000 | 16 | 576000000 | 4608000000 | 313032704 |
| 100000 | 15 | 360000000 | 2880000000 | 2880000000 |
| 100000 | 16 | 384000000 | 3072000000 | 3072000000 |

For either first-run case, even the first bank read exceeds the entire host allocation. This is a concrete memory-corruption defect and matches the last completed zone. The exact aborting instruction remains unverified without a core/backtrace. C++ changes are outside this task's permitted scope.

## Workload count provenance

CPU parsing of the preserved CSV files is recorded in `profiler_source_probe.json`:

- `tracy_ops_data.csv` contains **560** `TT_METAL_TRACE_ENQUEUE_PROGRAM` records and **130** replay records: two correctness replays plus **128 between `PERF_DECODE` and `PERF_DECODE_END`**.
- The measured workload is therefore **560 × 128 = 71680 program executions**.
- `tracy_ops_times.csv` contains **78391 `Program_*` records**: 5591 runtime IDs occur once, and 560 captured runtime IDs occur 130 times. Thus `5591 + 560 × 130 = 78391`, including setup/warmup/prefill.
- `Program_*` records originate from the separate realtime dispatch profiler (`tt_metal/impl/dispatch/realtime_profiler_tracy_handler.cpp:24-44`). They are device events, but their presence does **not** prove the conventional per-RISC profiler buffer was read or exported. No host duration was substituted for the requested device report.

## Minimal controlled workaround and verification

Retry the same workload and all 128 uninterrupted trace replays, changing only **`--op-support-count 150000` to `--op-support-count 100000`**. Save the new output under `tracy/sliding/raw_retry/` and a separate retry log/JSON. This avoids the demonstrated allocation wrap and leaves nominal capacity above the observed 78391 programs. No production or harness edit, intermediate dump, or reduction in the measured workload is required for this first experiment.

Prediction: close proceeds beyond bulk profiler read and writes the device CSV/report. Verify clean process exit, no dropped-marker/buffer-capacity warnings, both expected signpost regions, exactly 128 decode replay windows, and complete native device timing tables. Correctness JSON alone does not establish successful profiling. The main agent owns this hardware experiment; its result was pending when this report was written.

Adding `ReadDeviceProfiler` with the original 150000 allocation would merely encounter the same undersized buffer earlier. Smaller collection batches are a secondary option only if the corrected allocation exposes an independent capacity issue; preserve all 128 successive inputs and derive each whole-layer window from device timestamps, excluding inter-replay dump gaps.

## Retry result: workaround succeeded

The main agent ran the unchanged 128-step workload with support count 100000. I inspected `profile_sliding_retry.log`, `profile_sliding_retry.json`, `tracy/sliding/raw_retry/.logs/cpp_device_perf_report.csv`, and the generated `tracy/sliding/ops.csv`; this audit did not execute that run. The retry completed device export and report generation, with passing recorded correctness checks. The generated signposted windows contain 2501 prefill operations and 71680 decode operations across 128 replay sessions. This supports the allocation-overflow diagnosis and the command-only workaround. Original first-attempt artifacts remain intact.

## Follow-up: stale `PROGRAM CACHE HIT` metadata

All measured rows display `False`, but the serializer contains a separate verified reporting defect:

1. On the first encounter with an operation hash, `assemble_device_op_json` caches a compact message prefix containing the **current** `program_cache_hit` flag (`ttnn/cpp/tools/profiler/op_profiler_json.cpp:174-180`). For a first cold operation that flag is false.
2. Later encounters return the stored prefix with only a new operation ID appended; the new `program_cache_hit` argument is ignored (`ttnn/api/tools/profiler/op_profiler_serialize.hpp:331-335,377-380`).
3. The report parser reads that stale flag from each compact message (`tools/tracy/process_ops_logs.py:469-484`). Trace expansion copies the captured host operation metadata into each replay row (`:690-704`), so the 71680 decode flags are copies of 560 capture records, not independent runtime cache lookups.

The actual operation dispatcher computes the hit using the mesh program cache (`ttnn/api/ttnn/device_operation.hpp:402-430`) and passes true on its cache-hit enqueue path (`:290-291`). The cache defaults to enabled (`tt_metal/api/tt-metalium/program_cache.hpp:162`), and this harness contains no disable call. The first stored false prefix therefore cannot establish that the runtime cache was disabled or missed repeatedly.

CPU artifact parsing is recorded in `profiler_cache_metadata_probe.json`. It found 365 initial JSON messages and 5786 compact messages, all serialized false. More specifically:

| Phase | Native host operation records | Previously unseen operation/device/hash keys | Sequence comparison |
| --- | --- | --- | --- |
| Measured prefill | 2501 | 0 | Exactly equals the preceding warmup's 2501-operation suffix |
| Decode capture | 560 | 0 | Exactly equals all 560 decode-warmup operations |

This rules out changing **serialized program hashes** in the measured paths and establishes that the same ordered signatures were warmed. It is not a direct observation of runtime cache hits: mesh caching also includes coordinates and a canonical key (`mesh_device_operation_adapter.hpp:999-1044`). The `False` report column must not be presented as a measured cache-miss rate or silently rewritten to true.

The main agent plans a bounded runtime confirmation: after warmup, forbid program cache misses during the second prefill and trace capture, then restore the policy for unrelated setup operations. The dispatcher explicitly throws on a forbidden miss (`device_operation.hpp:418-421`). A control with uncached metadata serialization (`TT_METAL_PROFILER_NO_CACHE_OP_INFO=1`, exposed by Tracy's `--no-op-info-cache`) bypasses the faulty prefix reuse and records each actual flag. These controls were **pending** when this section was written; no production fix or device command was performed by this audit.
