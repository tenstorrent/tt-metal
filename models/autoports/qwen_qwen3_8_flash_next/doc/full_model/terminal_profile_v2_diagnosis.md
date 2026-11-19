# Terminal profile v2: incomplete capture and bounded recollection

CPU-only diagnosis, 2026-11-14. `terminal_ops_v2.json` reports successful model/sampler execution and equivalent greedy IDs243975 for both common sampler paths. This is not a valid per-op profile: the host capture ends before every target signpost and the device log reports dropped markers. Source and artifact hashes are in [diagnosis JSON](terminal_profile_v2_diagnosis.json). No device execution or native runtime build was performed by this author.

## Two independent limits

The log records an earlier capture failure that the final missing-op assertion obscures:

> Instrumentation failure: Too many source locations. You cannot have more than32K static or dynamic source locations.

The capture saves at03:46:55 after51s; `PERF_MODEL` starts at03:47:04.287. Retained host messages end at51,001,088,213ns and global op33603587, while the device report continues through op80388099. `import_tracy_op_logs` finds **zero signposts**, not six. This is not an exporter omission: the instrumented process kept running after the capture disconnected. The unchanged CSV exporter successfully parses the data that was actually saved.

The realtime profiler names every invocation `Program_<runtime_id>` and interns the resulting source location. Full `QwenGenerator.prepare()` runs all32 prefill buckets and all sampler configurations, creating over32K unique locations in this eager-only experiment. Legacy BRISC/ERISC marker source functions can also carry per-op IDs. The runtime already carries per-event Op ID metadata, so stable source locations could be a general repair, but **no native changes were made**: this stage needs a bounded reduced-layer profile and can avoid collecting unrelated preparation shapes.

Separately, the device DRAM marker buffer uses the default1000-program capacity. It overflows during preparation and again after the isolated measured model window: `PERF_MODEL_END` is03:47:04.369, followed by dropped-marker warnings at03:47:04.469–.473 on allfour devices, core1,2, BRISC/NCRISC/TRISC. Thus excluding warmup alone does not establish target completeness.

The final missing op1037312 is `ReduceDeviceOperation`, device0, host timestamp44,944,632,887ns: it belongs to preparation before the lost measured windows. Its absence does not identify the only missing data. The current capture cannot be recovered by selecting target intervals because their host records and markers are absent. No missing-op assertion was disabled, no target event was invented and no incomplete report is accepted.

## Repeated reads are not overwrites

`DeviceProfiler::readResults` reads DRAM and resets the control buffers; `processResults` writes each collected marker batch, then clears that in-memory batch. `writeProgramsPerfResultsToCSV` opens an existing report with `std::ios_base::app` under a mutex. Explicit log deletion belongs to constructor/new-log setup or `freshDeviceLog`, not the repeated reads used here. The observed source/marker limits explain the evidence; no repeated-read overwrite defect is established.

## Profile-only repair, hardware proof pending

`tests/profile_full_model_terminal.py::prepare_eager_profile` now allocates the required inactive-logits tensor `[1,1,1,62080]` (BF16 tiled DRAM), prepares the device seed helper, resets state, warms only `_decode_body((0,))`, then precompiles and warms both exact sampler wrappers. Sampling collectives and both sampler states are still constructed first. It drains between preparation stages and never marks `gen.prepared=True` or captures a trace. Common sampler arithmetic and real tensor shapes/policies are unchanged.

`time_region` now drains after each profiler warmup, then keeps drains outside measured signposts. The existing collector setting `--op-support-count 8192` is required for the bounded eager profile because even one reduced model body overflowed the default1000 capacity. This is additional profiler storage, not a change to advertised model context or serving settings. Final completeness still requires an actual run; the value is not a claim that all marker streams have been measured complete.

The clean traced-timing branch retains full `gen.prepare()`, sampler preparation and capture behavior. The small eager helper is test instrumentation and cannot be used to advertise general generator readiness. Three CPU tests execute the actual extracted helpers without importing TTNN: required inactive/seed initialization, exact scope, draining outside target intervals, and zero profiler reads in clean timing. Syntax and formatting pass.

Ready serialized collection after the device owner restores the accepted environment/mesh descriptor, with watcher absent:

```bash
TT_METAL_DEVICE_PROFILER=1 TT_METAL_TRACE_ALLOC_TRACKING=0 \
TT_METAL_TRACE_ALLOC_TRACEBACKS=0 timeout 3600 python_env/bin/python -m tracy \
-r -p -v --no-web-server --op-support-count 8192 \
-o models/autoports/qwen_qwen3_8_flash_next/doc/full_model/terminal_tracy_v3 \
-m models.autoports.qwen_qwen3_8_flash_next.tests.profile_full_model_terminal \
--profile-ops-only --iterations 1 \
--output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/terminal_ops_v3.json
```

Before interpreting any timing, require all six target signposts in retained host data, no capture instrumentation failure, and complete positive per-op device timings on allfour ranks for each target interval. If only unrelated preparation is incomplete in a subsequent capture, explicit signpost-scoped report generation is permitted only after proving every selected host/device op is present; the global assertion must remain intact. The v2 capture does not satisfy that precondition. Final `tt-perf-report` and clean trace latency remain separate evidence.
