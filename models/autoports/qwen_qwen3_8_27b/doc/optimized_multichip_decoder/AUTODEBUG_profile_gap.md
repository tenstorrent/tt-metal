# AutoDebug: layer-3 host/device timing gap

Status: localized in existing host traces; underlying cause still unverified. This investigation read source and saved artifacts only. It did not import TTNN, open devices, or run the model. The separate redundant-BF16-cast repair is outside this diagnosis.

## Evidence that changes the diagnosis

The excess time is inside **C++ `FDMeshCommandQueue::finish_nolock`**, not an unexplained Python/signpost overhead. These values come from `.logs/tracy_ops_times.csv`, selecting host zones between the `PERF_DECODE` and `PERF_DECODE_END` timestamps in `.logs/tracy_ops_data.csv`:

| Profile | Python timed host, µs | C++ execute_trace, µs | C++ finish, µs | Nested finish_nolock, µs | Longest per-device worker span, µs |
|---|---:|---:|---:|---:|---:|
| final_profile_l3 | 643.209 | 33.033 | 592.662 | 590.779 | 314.483 |
| final_profile_l3_host_repeat | 910.209 | 23.314 | 871.966 | 869.410 | 311.471 |
| profile_trace_prefill_l3 (decode window) | 366.150 | 19.938 | 329.931 | 326.994 | 315.064 |
| final_profile_l0 | 520.242 | 41.169 | 458.445 | 452.194 | 436.750 |

The bad profiles have only 17.514 and 14.929 µs left after subtracting their C++ execute and finish zones from the Python timer. GIL reacquisition, Python wrapper work, and timer bookkeeping therefore cannot explain their approximately 329/599 µs host-minus-worker-span gaps. The tail of `finish` outside `finish_nolock` is only 1.883/2.556 µs, also excluding its realtime-profiler callback and distributed barrier as the dominant cause in these saved runs.

All three layer-3 profiles contain 43 device ops. Per-device spans in the repeated bad run are 309.774, 309.978, 311.471, and 311.337 µs. Device accounting is stable while C++ completion latency changes markedly. These are per-device worker intervals, not a calibrated host-to-device completion interval; they do not measure trace launch latency, the final dispatch/event tail, or completion-reader notification.

The repeated bad run's signpost timestamps are 6151152851 and 6152342773 ns. Relative to the first signpost, `execute_trace` begins at 284.834 µs, and `finish_nolock` spans 313.829–1183.239 µs. The first approximately 285 µs include signpost logging before `profile_start`; they are **excluded** from the 910.209 µs Python metric. The original bad run similarly has approximately 385 µs before C++ execute. That preceding idle time may affect worker wake state, but should not be added to or subtracted from the measured decode latency.

## Source path

Paths below are relative to the repository root; line numbers describe the inspected source.

1. `tests/run_multichip_decoder.py` under the model root, lines 558–570: refresh/restore, synchronize, and `ReadDeviceProfiler` precede the signpost and timer. The timer encloses one `execute_trace(blocking=False)` and one synchronize. The trailing profiler read follows the end signpost. Lines 317–325 and 502–512 show the same execute/synchronize pair for normal traced-prefill and decode samples; host equality/readback is outside those timers.
2. `tt_metal/impl/profiler/tt_metal_profiler.cpp:1220–1258`: profiler reads finish the queues, read device records, enqueue result processing, and **wait for the mesh dispatch thread pool**. The debug-dump alternative is also synchronous with its explicit read request (`profiler_state_manager.cpp:176–183`). Pending processing from this API is not established by source; another uncontrolled profiler thread would require independent evidence.
3. `tt_metal/distributed/fd_mesh_command_queue.cpp:715–767`: `finish_nolock` submits a host completion event, then waits on `reads_processed_cv_`. Its caller later performs the realtime-profiler sync check and distributed barrier. Existing host traces resolve the aggregate `finish_nolock` duration but do not separate its event submission from its reader wait.
4. **An additional thread-pool handoff exists inside that timed completion path.** `enqueue_record_event_helper` at lines 922–958 enqueues one event-command task per device into `dispatch_thread_pool_`, then waits for that pool. Only after it returns does `enqueue_record_event_to_host_nolock` publish the completion descriptor and wake the completion reader (974–979). A drained profiler pool can still take time to wake for these new tasks; this is distinct from unfinished profiler processing.
5. The completion reader is a separate C++ thread (`fd_mesh_command_queue.cpp:212`). It waits on a condition variable, visits the per-device events, decrements `num_outstanding_reads_`, and notifies the main thread (1008–1049). Event observation sequentially visits local devices and polls their completion queues (1116–1138). Source therefore admits delayed event submission, reader wakeup, main-thread wakeup, and an actual late event arriving from dispatch; it does not prove which occurred.
6. `tt_metal/distributed/mesh_device.cpp:95–101` selects a device-bound pool for TP4 unless `TT_MESH_PASS_THROUGH_THREAD_POOL=1`; the latter runs tasks synchronously. `tt_metal/impl/threading/thread_pool.cpp:190–253` uses a brief spin followed by atomic wait/notify. Do not attribute these runs to the obsolete growing-sleep implementation described in the comment: the inspected implementation already uses atomic waits. Workers receive CPU affinity at lines 226–227. Actual affinity and runnable delay must be measured.
7. `execute_trace(blocking=True)` calls the **same** `finish_nolock` from `enqueue_trace` (`fd_mesh_command_queue.cpp:1296–1297`). It removes a Python/API boundary, but does not bypass event-command dispatch or the completion reader. Its success alone would not identify a specific wait bug.
8. Both trace execute and synchronize release the GIL in nanobind (`ttnn/cpp/ttnn-nanobind/operations/trace.cpp:56–66`, `device.cpp:607–627`), but the existing zone arithmetic above bounds their uninstrumented API-return overhead tightly.
9. `profile_optimized_multichip_decoder.sh` uses Tracy `-p`, meaning partial profiling; `tools/tracy/__init__.py:77–81` does not enable global Python function tracing in that mode. A generic Python profiler-overhead explanation is unsupported. The wrapper records environment before Tracy enables the profiler, so its `TT_METAL_DEVICE_PROFILER: null` is not proof that the child ran without device profiling.

## Ranked hypotheses and focused controls

### H1: completion-event dispatch through the device-bound pool is delayed

This is the first control because it directly removes one unresolved handoff inside the slow zone without changing model math or rebuilding.

Run the normal runner with fixed code/policy in A/B/A order, 30 traced-prefill and decode samples per process:

```bash
env TT_MESH_PASS_THROUGH_THREAD_POOL=0 models/autoports/qwen_qwen3_8_27b/tests/run_optimized_multichip_experiment.sh gap_pool_default_a --layer 3 --length 128 --batch 1 --repeats 30 --prefill-repeats 30 --trace-prefill
env TT_MESH_PASS_THROUGH_THREAD_POOL=1 models/autoports/qwen_qwen3_8_27b/tests/run_optimized_multichip_experiment.sh gap_pool_passthrough --layer 3 --length 128 --batch 1 --repeats 30 --prefill-repeats 30 --trace-prefill
env TT_MESH_PASS_THROUGH_THREAD_POOL=0 models/autoports/qwen_qwen3_8_27b/tests/run_optimized_multichip_experiment.sh gap_pool_default_b --layer 3 --length 128 --batch 1 --repeats 30 --prefill-repeats 30 --trace-prefill
```

Record the pool environment explicitly; the existing wrappers' environment allowlists omit it. Repeat the signposted profile under both values. Prediction: removing the handoff substantially reduces the tail of `finish_nolock`, while the worker spans and correctness remain stable. Refutation: the same large tail persists with pass-through. A positive result localizes the pool boundary but does not by itself distinguish worker wakeup from runnable delay or identify a thread-pool correctness defect; the option affects other mesh pool uses too. Keep it as a diagnostic until that distinction and broader serving implications are checked.

### H2: completion reader or main thread resumes late after events are available

Use `host_gap_probe.py` to collect main-thread wall/CPU/rusage measurements and per-thread `/proc/self/task/*/{schedstat,status}` deltas. First compare the pool A/B with the same probe enabled, then confirm conclusions using untouched timing runs. A long wall time with little main-thread CPU is expected for a wait and is not proof of scheduling contention. Require increased **runnable wait** in `schedstat`, a wake-to-run trace, or equivalent scheduling evidence for the implicated C++ worker/reader/main TID. Main-thread context switches alone cannot identify the delayed thread.

If still needed, add narrow C++ zones (one diagnostic build) around: (a) event command enqueue plus pool wait, (b) outstanding-read wait; on the reader, (c) receipt of the descriptor, (d) each completion-queue wait, and (e) main-thread notification. Prefer these bounded zones to broad instrumentation. They distinguish a late dispatch task, an actual late device event, delayed reader execution, and delayed return after notification. Run only one fixed workload while observing these events. Parent reports no CPU cgroup throttling (`cpu.max=max 100000`, `nr_throttled=0`) and no simultaneous hardware job; those observations exclude those specific explanations, not host scheduling effects generally.

### H3: final dispatch/event completion is delayed beyond the reported worker interval

If H1 is refuted and reader runnable delay is absent, use a synchronized host/device profile plus dispatch-core collection (`python -m tracy ... --sync-host-device --profile-dispatch-cores ...`). Ensure capture succeeds and inspect final worker completion → dispatch wait/event write → reader observation. Default profiles are **not** guaranteed to align GPU and host clocks: `tt_metal/impl/profiler/profiler.cpp:2818–2854` falls back to an arbitrary host anchor when no sync frequency exists. Do not infer a precise host/device tail by visually lining up default Tracy lanes.

Prediction: the event itself is generated late, and the delay appears before its host-visible completion. Refutation: dispatch has completed promptly and the lateness is in host observation/notification. Worker spans alone cannot settle this hypothesis.

### H4: extra Python/API boundary causes the anomaly

This is already strongly disfavored by saved C++ zone durations. The probe supports `QWEN_GAP_BLOCKING_TRACE=1`, changing originally nonblocking replays to blocking replays and consuming their immediately following synchronize call. Compare with the unmodified pair, preserving restore, correctness readback, and the surrounding timer. The same internal wait remains, so persistence is expected under H1/H2/H3. Do not apply this altered timing contract to final stage metrics unless explicitly adopted and compared consistently.

## Unprofiled traced-prefill anomaly

`trace_prefill_l3_s128.json` has 30 traced-prefill samples, median 2.009047 ms, minimum 0.749322 ms, maximum 6.049253 ms; 18 exceed 1 ms. Its decode samples range 0.303932–2.994228 ms. The first 20 decode samples have median 0.703463 ms, whereas the last 10 have median 0.308284 ms. `profile_trace_prefill_l3.json` contains only **one** traced-prefill sample, 0.772044 ms, and its good signposted decode has a 326.994 µs C++ completion wait. The single good profile does not demonstrate that profiling fixed the repeated timing distribution.

Both phases share the same host completion path and perform host Torch equality/readback between samples (`torch.set_num_threads(8)` is set in the runner). A common host completion cause is a testable hypothesis, not a demonstrated explanation. Use H1's same-process traced-prefill/decode controls and counters first. Only pursue a one-Torch-thread control or explicit thread affinity if counters implicate host CPU activity; source alone does not show contention.

## Result and remaining work

No implementation fix is justified yet. Preserve the original timing outliers and report the existing localization: the saved anomalies reside in `finish_nolock`; profiler reads, signpost logging, and GIL-return overhead do not account for the excess. First run the pool bypass A/B plus the probe; escalate to narrow completion-path zones only if those controls fail to identify the cause. Keep paired host and per-device evidence from each run, and do not replace distributions with a favorable one-shot profile.

### Parent-run controls reported while this investigation was in progress

These observations were supplied by the parent, not executed by this source-only investigation. After the separate BF16-cast guard, all 31 correctness cases passed. With 100 decode and 60 prefill samples, the ordinary default-pool run reported eager/traced-prefill/decode/queued medians of 1.165571/0.738887/0.305313/0.293246 ms. The pass-through control reported 1.089836/0.734078/0.306680/0.293472 ms. Thus the earlier 2.009 ms traced-prefill median did not reproduce in the new default run, and this control supplies **no decode latency advantage** for bypassing the pool. H1 remains a hypothesis about the saved anomalous completion waits; it is not a demonstrated default-runtime performance problem. The profiled A/B and diagnostic counters remain useful because those original signposted anomalies were repeated.

The probe is now available as `doc/optimized_multichip_decoder/host_gap_probe.py`. The parent added wrapper support for `QWEN_PROFILE_MODULE` and pool-environment provenance. Use:

```bash
QWEN_EXPERIMENT_ENTRY=models/autoports/qwen_qwen3_8_27b/doc/optimized_multichip_decoder/host_gap_probe.py
QWEN_PROFILE_MODULE=models.autoports.qwen_qwen3_8_27b.doc.optimized_multichip_decoder.host_gap_probe
```

Pass these as environment variables to the corresponding wrapper, retaining the same model arguments. The sidecar replaces the output `.json` suffix with `.host_gap.json`. Its timer proxy collects `/proc` before the runner's start timestamp and after its end timestamp. Synchronize records contain API-only wall time and main-thread CPU/rusage; per-TID deltas for timed calls cover their enclosing execute-plus-sync interval. Standalone sync calls get separate thread snapshots. Small counter calls remain inside normal runner timing, and the snapshots change host wake state even though their cost is excluded. The sidecar labels these limits; use its counters to design attribution, and preserve untouched runs for performance claims. All-zero scheduler wait counters are inconclusive when scheduler statistics are unavailable.
