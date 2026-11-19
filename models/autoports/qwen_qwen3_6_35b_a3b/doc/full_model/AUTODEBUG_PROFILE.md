# Reduced profiler host OOM: source diagnosis

CPU-only AutoFix investigation, 2026-11-14. No device, build, model change, or new profiling run was performed by this investigator. Outcome: two independent profiler faults are supported: automatic real-time Tracy program zones exceed its source-location limit and terminate capture early; ordinary device-profiler reads retain parsed host markers without mid-run dumps. Both need attention before a new capture. Neither failed capture establishes a model numerical failure or a physical context limit.

## Failure and retained evidence

Both runs requested real layers 0/3 on the 1x4 mesh, context 1024, batch one, native per-op metadata, and 16 replay samples each for token-out, model-only, and sampler-only. They failed during eager preparation, before valid component measurements.

- `performance/profile_reduced_v1/{capture.log,host_oom.log,failure.json,performance.json}`: the first preparation read encountered dropped markers; Linux killed workload Python PID 3826202 with 188465852 kB anonymous RSS. The 100000-program buffer allowed a large interval before its first read.
- `performance/profile_reduced_v2/{capture.log,host_oom.log,failure.json,performance.json}`: the harness reduced the buffer count to 5000 and read after every preparation attention/MoE component. Linux nevertheless killed workload Python PID 3927700 with **188621280 kB anonymous RSS**. Last printed preparation progress was linear attention rows 32 (`capture.log:53118`); rows 1 and 16 appear at lines 19281 and 36707. `performance.json` still says RUNNING with no measurements, so `failure.json` is the final status.
- At inspection, v2's `raw/.logs/tracy_profile_log_host.tracy` was 308720108 bytes; v1's was 6821666 bytes. Neither raw directory contained `profile_log_device.csv` or `cpp_device_perf_report.csv`. Small/compressed host capture size is not a measure of the workload's live device-marker objects.
- Bounded arithmetic/open/close recovery smokes passed separately (`logs/mesh_smoke_after_profile.log`, `logs/mesh_smoke_after_profile_v2.log`). They do not validate profiling.

## Proven early collector exit, separate from host marker retention

Both logs explicitly contain **`Instrumentation failure: Too many source locations. You cannot have more than 32K static or dynamic source locations.`**: v1 `capture.log:16537`, v2 `capture.log:35500`. The collector then saves a partial trace while the workload continues compiling. V1 reports 302407 zones over 1:40.9; v2 reports 185768835 zones over 2:53.3. Parent-observed file modification times (v1 14:07:54, v2 14:22:10) precede workload OOM (14:16:57 and 14:24:55). This fully explains the frozen early trace files; they were completed **partial** captures, not progressively written live streams.

A bounded CPU-only native export of the 6.8 MB v1 trace completed in 0.2 seconds:

```bash
timeout 20 build/tools/profiler/bin/tracy-csvexport \
  models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/performance/profile_reduced_v1/raw/.logs/tracy_profile_log_host.tracy \
  > /tmp/qwen36_profile_v1_zones.csv
```

Counting raw lines matching `^Program_([0-9]+),` found **32766 source-location rows with 32766 distinct runtime IDs**. Generic kernel source strings contain unquoted commas/newlines in this statistics export, so naive CSV row counts are invalid; the anchored program-name count is the relevant observation.

The matching source is `tt_metal/impl/dispatch/realtime_profiler_tracy_handler.cpp:51`: `make_marker` names every record `Program_<runtime_id>` and attaches concatenated kernel-source strings as its file field (`:237`). `HandleRecord` emits start/end markers (`:253`). The automatic handler is installed by `tt_metal/distributed/realtime_profiler_manager.cpp:909`. It is **separate from** `DeviceProfiler::pushTracyDeviceResults`; its callback checks only `tracy::GetProfiler().IsConnected()` (`realtime_profiler_tracy_handler.cpp:87`), not `TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY`. Thus the collector's existing disable flag does not suppress these program lanes. Tracy rejects dynamic source locations above signed-16-bit capacity (`server/TracyWorker.cpp:3893`); static locations have a separate corresponding check (`:3499`). The 32766 distinct program lanes plus other dynamic locations corroborate the failure precisely.

The capture executable's loop ends when the worker disconnects, prints instrumentation failure, writes a trace, and **returns zero even after that failure** (`capture/src/capture.cpp:134–181`). Therefore the Python wrapper's nonzero-exit check does not catch it. Its wait happens only after workload completion (`tools/tracy/__main__.py:439`), allowing the workload to continue. No `-s` duration or `-m` memory cap was passed in either recorded capture command.

Disconnected-client queue growth is **not proven** by current source: the non-on-demand path calls `SuppressEmit()` and repeatedly `ClearQueues()` after connection loss (`public/client/TracyProfiler.cpp:2215`); `build/CMakeCache.txt` has `TRACY_ON_DEMAND=OFF`. The real-time callback also stops emitting when disconnected. This makes persistent marker-set retention below the stronger explanation for continued workload RSS growth. Installed binary/source correspondence and exact heap ownership have not been measured, so do not claim the queue hypothesis mathematically excluded.

The exit warning about missing `generated/profiler/.logs/tracy_profile_log_host.tracy` is an independent copy-path bug: `tools/tracy/common.py:11–23,58` computes `PROFILER_LOGS_DIR` at import, before the outer CLI processes `-o`; `__main__.py:473` later uses that stale imported constant for the optional WASM copy. The actual capture command uses the requested artifact directory. That warning neither caused the early disconnect nor invalidates the existing partial trace file at its real path.

## Host-memory mechanism supported by source

**`ReadDeviceProfiler` does not discard accumulated host markers by default. The new per-component reads need `TT_METAL_PROFILER_MID_RUN_DUMP=1`.**

1. `tests/collect_full_model_components.sh:17` enables the device profiler and disables pushing device data to Tracy. Lines 18–19 set the 5000-program capacity and enable op metadata/trace tracking. The wrapper never sets `TT_METAL_PROFILER_MID_RUN_DUMP`, nor passes Tracy's `--dump-device-data-mid-run`. The runtime default is false (`tt_metal/llrt/rtoptions.hpp:241`; `rtoptions.cpp:1265`). The failed result schema does not record that environment variable, so inherited environment cannot be reconstructed conclusively; absence of both device CSV outputs is consistent with the default path.
2. `ttnn/cpp/ttnn-nanobind/device.cpp:768` exports the op-name map and calls `ReadMeshDeviceProfilerResults(..., ProfilerReadState::NORMAL, ...)`. The mesh implementation (`tt_metal/impl/profiler/tt_metal_profiler.cpp:1194`) finishes queues, reads every device, then processes results.
3. `ProcessDeviceProfilerResults` (`tt_metal_profiler.cpp:1017`) parses the new device data, but calls `profiler.dumpDeviceResults(true)` only when `dumpDeviceProfilerDataMidRun` succeeds. That predicate immediately returns false without the mid-run flag (`:997`). Reading/draining the device ring therefore transfers markers into host memory without freeing earlier host markers.
4. `DeviceProfiler::readDeviceMarkerData` inserts into a `std::set<tracy::TTDeviceMarker>` (`tt_metal/impl/profiler/profiler.cpp:1994`). Each marker owns operation/file/name strings and JSON metadata (`tt_metal/third_party/tracy/public/common/TracyTTDeviceData.hpp:139`), in addition to set-node overhead. Across cores, RISCs, four chips and the preparation geometries, the live set grows with all markers read so far.
5. `DeviceProfiler::dumpDeviceResults` writes native analyses/CSV, optionally pushes to Tracy, waits for workers, then clears `device_markers_per_core_risc_map` (`profiler.cpp:2479–2524`). **This is the missing host release.** The same retention mechanism is explicitly described in the existing background debug-dump implementation (`profiler_state_manager.cpp:343`); that debug-dump thread is a separate mode, not enabled by this harness.
6. Reducing `PROGRAM_SUPPORT_COUNT` changes the device ring and raw host read-buffer allocation (`profiler_state_manager.cpp:44`, `profiler.cpp:2383`), not the lifetime of parsed marker sets. Reading more frequently without enabling dumps can avoid ring overflow while still accumulating the entire run on the host. This explains why v2 still OOMed.

This is a source-supported causal mechanism, not a measured heap attribution of all 188 GB. A heap profile was not captured. Other caches and Tracy allocations also exist; the next experiment must demonstrate bounded growth and successful native reports.

## Candidate assessment

| Candidate | Assessment |
| --- | --- |
| Enable mid-run dump, retain current component reads | Required alongside the separate real-time Tracy gate. Activates the existing write/analyze/clear path, without changing the model graph, dtypes, fidelity, cache or sampler. |
| Set `TT_METAL_PROFILER_NO_CACHE_OP_INFO=1` | Rejected as a memory repair. `op_profiler_serialize.hpp:339` bypasses the compact cache-hit message and rebuilds full JSON on every call; `op_profiler_json.cpp:174–185` still inserts compact prefixes. It neither frees device-marker sets nor disables native program caching, and can increase message traffic. Tracy CLI normally unsets it unless `--no-op-info-cache` is given (`tools/tracy/__main__.py:252`). |
| Disable device-data push to Tracy | Already done. The check is in `pushTracyDeviceResults` (`profiler.cpp:2667`), downstream of marker allocation, and does not release the host set. |
| Disable C++ runtime analysis | Not the first fix: the original failure precedes the gated dump/analysis path. Existing conversion explicitly requires `cpp_device_perf_report.csv` (`tests/profile_full_model_components.py:65`). Retain native analysis for this experiment. It retains a smaller per-dump analysis vector (`profiler.cpp:2472`), so monitor growth after the marker fix. |
| Toggle `TTNN_OP_PROFILER` around warmup | Not a reliable off switch: `is_op_profiler_env_var_set()` latches true once enabled (`op_profiler_serialize.hpp:252`). This also does not control device marker retention. |
| Trace-only instrumentation / skip all-geometry warmup | Defer. `TT_METAL_TRACE_PROFILER` is explicitly incompatible with mid-run dumps (`tt_metal_profiler.cpp:1003`), and changing what gets prepared/captured adds unnecessary evidence risk. `-p` controls Python zones, not the native device marker lifetime. |
| Increase host memory or decrease buffer count again | Does not repair the growing host set. No reason to repeat either as the next experiment. |

## Exact next experiment (not executed)

**Do not retry mid-run dumping alone:** it cannot repair the demonstrated 32K program-location overflow. The smallest additional profiler-only source fix is to make the real-time Tracy handler honor the existing `TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY` runtime option, while leaving real-time record collection/dispatch behavior and the traditional per-op CSV path intact. Gate its record callback before constructing `Program_<runtime_id>` markers, and consistently gate its sync/stall/calibration Tracy emission. Prefer the context's parsed runtime option over a new environment variable; default behavior remains unchanged when the existing flag is unset. Avoid guessing/unregistering a callback handle from Python, suppressing all Tracy messages, or changing model preparation. The source repair described below is now applied; a focused host build is needed before hardware. No build/install was performed by this investigator.

A profiler-runner guard should also check native collector status/connection or scan the capture log for instrumentation failures while preparation runs, and stop the owned workload cleanly before further profiling if the collector exits early. Final report acceptance must reject instrumentation failure even when `tracy-capture` exits zero.

After that profiler-only gate is built, the parent finishes the independently authorized context run, and idle ownership is verified, run the collector once with mid-run dumping into a fresh directory:

```bash
source /home/ttuser/qwen36-35b-a3b/lane-env.sh
TT_METAL_PROFILER_MID_RUN_DUMP=1 bash \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/collect_full_model_components.sh \
  models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/performance/profile_reduced_v3_midrun
```

The current collector preserves this environment variable. Alternatively add `--dump-device-data-mid-run` to its Tracy invocation; do not combine it with `--device-trace-profiler` or dispatch-core profiling. Keep device files enabled, C++ analysis enabled, op metadata cached, trace tracking enabled, 5000-program capacity, four chips, layers 0/3, and all 16 replays. Keep the existing preparation and measured replay drains; no full-stack capture.

Small profiler-only observability additions before execution are useful: retain collector/workload PIDs and collector liveness; record the mid-run flag, support count, and C++ analysis flag in `performance.json`; periodically record current `/proc/self/status` VmRSS and `resource.getrusage(...).ru_maxrss` from the preparation drain wrapper. Current RSS can remain at a high-water allocator plateau even when objects are freed; distinguish bounded reuse from monotonic run-length growth. Record native CSV sizes after completed drains. The CLI enables C++ analysis in `tools/tracy/__main__.py:306`, so do not infer that flag solely from the outer shell.

Acceptance for this experiment:

1. No real-time `Program_<runtime_id>` lanes are emitted with the disable flag; the collector stays connected throughout preparation/measurement and saves only after completion. Device CSV and native C++ report appear/grow during preparation, proving the dump path executed. RSS growth becomes bounded enough to finish preparation and capture; report actual high-water values, not an assumed limit.
2. No dropped-marker, mismatched-zone, or incomplete-report warning. If one component exceeds the ring, identify that component before changing drain granularity; do not accept partial timing.
3. Existing converter/summarizer validates all four ranks and **16 replay sessions per requested component**, matching op signatures, trace IDs, cache-entry stability, token/position counter assertions, and required per-op dtype/fidelity fields.
4. Retain raw traces, CSVs, logs, source hashes and command. Preserve the two failed attempts. Fenced instrumented times remain diagnostic; headline TTFT/tokens-per-second come from the separate unprofiled full-model run.

Mid-run dumping weakens cross-device timestamp synchronization (`tt_metal_profiler.cpp:321` warns about sensitive device-device event analysis). Use per-rank native duration/op-work attribution and the existing report's stated non-additive interpretation; do not infer exact inter-chip CCL skew from these timestamps. The change does not remove native dtype/fidelity metadata or replay identity. If memory still grows after confirmed dumps, investigate retained C++ analysis vectors and Tracy/client caches with bounded allocation evidence; that outcome is not yet observed.

## Inspection pins

- `tests/run_full_model_performance.py`: `6a096f6cd87c046d9a09d39f0de7a7e3b57334afd23f73ebe6da5390f45cabe6`
- `tests/collect_full_model_components.sh`: `bd63658276952dbb809c5fb76c91ad7caa9a0724edf360c02bf550a6a63bc79a`
- `tt_metal/impl/profiler/profiler.cpp`: `b8b1ea765325b104b5f5732fc088e51d008aeac2e1ccfc5104b605954dd1e7b8`
- `tt_metal/impl/profiler/tt_metal_profiler.cpp`: `91cd3044bbc8391ad596b585c9313a7940269174ccb257420607abe91ecd69bb`

Additional inspected pins: real-time Tracy handler `5cebd76eeed0846114dd2ddaae88dd65cb32d80f813515476712854e4df5b7e6`; Tracy client `f77c7b7f40548db7c032f4c139a6d2c26c79aa6d3b6d12fef380a55b09d7ce82`.

## Applied profiler-only repair

At the parent's follow-up request, the following changes were applied after the source diagnosis:

- `tt_metal/impl/dispatch/realtime_profiler_tracy_handler.cpp`: read the existing parsed `get_profiler_disable_push_to_tracy()` option from the default MetalContext (the same context used by the existing callback registration). Skip record callbacks and context creation when disabled; return no context from the shared lookup used by record/sync/stall/calibration emissions; suppress the separate pending-stall Tracy message flush. With the option unset, existing behavior is unchanged. Real-time device collection and non-Tracy callbacks remain active. No model numerical or scheduling path changed.
- `tests/collect_full_model_components.sh`: set `TT_METAL_PROFILER_MID_RUN_DUMP=1`. Preserve the workload status and reject instrumentation failures, dropped markers, or partial-report warnings before conversion, including a source-location failure followed by a zero capture return code. This guard checks after capture exits; it is **not** a live workload abort monitor.
- `tests/run_full_model_performance.py`: update the documented command to support-count 5000 and mid-run dumping; require the mid-run flag for profile mode and record it, support count, C++ analysis and op-info-cache settings in the artifact environment.

Validation performed: `clang-format --dry-run --Werror` and `git diff --check` for the C++ change; `bash -n` for the collector; `python -m py_compile` for the performance harness; four CPU-only executions of the actual shell guard covering a clean capture, zero-exit source-location failure, dropped markers, and propagation of timeout exit 124. All those checks passed. The installed newer `black --check --target-version py312` proposes unrelated formatting changes in existing blocks; no broad formatting rewrite was applied. No C++ compile, link, install or hardware validation was performed here.

Build advice for the parent once it owns an idle runtime: `cmake --build build --target tt_metal --parallel 2`, then the lane's existing library installation procedure. The source belongs to the `impl` object library; the current unity translation unit is `build/tt_metal/impl/CMakeFiles/impl.dir/Unity/unity_18_cxx.cxx`, and the linked target is `build/tt_metal/libtt_metal.so`. Updating that library is required; restarting Python alone does not apply this C++ change. Do not rebuild/install beneath the in-flight context or device test.

Post-edit SHA256:

- Real-time Tracy handler: `174896228620b4428d1a55ba6721aa4e30192134bbcd55e2ad8d02e772ec77c6`.
- Collector script: `49752c4cad27b9d570e57c235d63ce50d3b87e0a044efa3dd1516b95b11190d4`.
- Performance harness: `3b74f5ae69afad367c1b6660640b4495ff313a1e9f3df9f1d280ae84afcad5f9`.

The hardware outcome remains unverified; the next capture must still satisfy every acceptance item above.
