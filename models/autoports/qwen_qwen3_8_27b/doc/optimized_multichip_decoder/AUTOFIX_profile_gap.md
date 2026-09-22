# AutoFix: host completion timing gap

Status: completion-path sensitivity demonstrated; pass-through pool selected for measured eager-prefill benefit. The precise cause of the historical signposted outliers remains unproven. This follow-up analyzed completed artifacts only; it did not run hardware, change the probe, or edit implementation code.

## Starting evidence

`AUTODEBUG_profile_gap.md` localized the original layer-3 timing anomalies to C++ `FDMeshCommandQueue::finish_nolock`. Its duration was 590.779/869.410 µs in two anomalous profiles, versus 326.994 µs in an existing healthy profile. Worker-kernel spans stayed near 314 µs. Profiler reads completed before the timer, and the C++ zone arithmetic excluded signpost logging and GIL-return overhead as dominant explanations.

H1 concerned the new event-command submission through the device-bound dispatch pool **inside** `finish_nolock`. H2 concerned scheduling of the pool workers, completion reader, or main thread. The pass-through control removes the dispatch/read pool workers while retaining device completion semantics.

The parent subsequently chose `TT_MESH_PASS_THROUGH_THREAD_POOL=1` as the default of the optimized launchers. That choice is supported by the eager-prefill measurements below, not by a claimed steady-state decode speedup. The parent reported 31/31 correctness cases passing after the separate BF16 cast guard; final expanded correctness, stress, watcher, capacity, and profile refreshes are being handled by the parent.

## Verified input artifacts

`gap_counter_pool_0`, `gap_counter_pool_1`, `gap_profile_passthrough`, and the six ordinary paired runs listed below all have recorded exit status 0. Both counter sidecars report `error: null`, `incomplete_interval: null`, no blocking-replay override, and the expected pool values 0/1. Their runner reports pass exact trace, changed-input replay, and post-decode-state equality. Minimum reported PCC in the counter and pass-through profile runs is 0.9999994039535522.

All these new runs record model source SHA256 `211b70db973294f7660ba9cd3c4ef0601ffbac32dab4866e3511b4804ff76213` and runner SHA256 `22b17036b15d2c85626b4113f397d9f2ddc2274517ba5c9f116ae0b9196fa294`. The historical bad profiles precede the BF16-cast guard and have a different model source hash; their layer-3 decode profiles and the new profile nevertheless each contain 43 ops. The ordinary `gap_threadpool_default` environment records the pool variable unset, selecting the source default; the later `_0` controls explicitly record 0.

## Ordinary timing controls: selected behavior

Each row is a separate completed process, with 60 eager-prefill and 100 decode samples. Layer 3 additionally has 60 traced-prefill samples. Queued measurements are the runner's existing amortized supplementary metric. Times are milliseconds; rounded values below come directly from each named JSON.

| Workload / artifact pair | Pool | Eager prefill | Traced prefill | Decode | Queued decode |
|---|---:|---:|---:|---:|---:|
| Full attention: `gap_threadpool_default` | default | 1.165571 | 0.738887 | 0.305313 | 0.293246 |
| Full attention: `gap_threadpool_passthrough` | 1 | 1.089836 | 0.734078 | 0.306680 | 0.293472 |
| Linear: `gap_pool_linear_0` | 0 | 1.325071 | — | 0.423690 | 0.408693 |
| Linear: `gap_pool_linear_1` | 1 | 1.228877 | — | 0.421215 | 0.408988 |
| Stack: `gap_pool_stack_0` | 0 | 2.599765 | — | 0.704711 | 0.691692 |
| Stack: `gap_pool_stack_1` | 1 | 2.376624 | — | 0.704641 | 0.691328 |

Observed eager-prefill reductions are 6.498%, 7.259%, and 8.583% across the three workloads. Decode medians remain essentially unchanged; full-attention decode is slightly higher with pass-through. These are measured within-process sample medians from one process per setting, not confidence bounds over independent process repetitions. They support the launcher's local performance choice without establishing a general runtime-wide advantage.

The previous 2.009047 ms unprofiled traced-prefill median does **not** reproduce in the new untouched default runner: its traced-prefill median is 0.738887 ms even before bypassing the pool. Therefore pass-through must not be credited with fixing that historical median regression.

## Counter probe A/B

The parent ran the following commands with the recorded environment; each collected 30 eager-prefill, 30 traced-prefill, and 30 decode intervals:

```bash
env TT_MESH_PASS_THROUGH_THREAD_POOL=0 QWEN_EXPERIMENT_ENTRY=models/autoports/qwen_qwen3_8_27b/doc/optimized_multichip_decoder/host_gap_probe.py models/autoports/qwen_qwen3_8_27b/tests/run_optimized_multichip_experiment.sh gap_counter_pool_0 --layer 3 --length 128 --repeats 30 --prefill-repeats 30 --trace-prefill
env TT_MESH_PASS_THROUGH_THREAD_POOL=1 QWEN_EXPERIMENT_ENTRY=models/autoports/qwen_qwen3_8_27b/doc/optimized_multichip_decoder/host_gap_probe.py models/autoports/qwen_qwen3_8_27b/tests/run_optimized_multichip_experiment.sh gap_counter_pool_1 --layer 3 --length 128 --repeats 30 --prefill-repeats 30 --trace-prefill
```

These are **diagnostic** measurements. The probe excludes `/proc` snapshot cost from the runner timer but changes the surrounding thread wake state. The small per-sync counter reads also add overhead to the enclosing runner measurement. Do not replace ordinary performance results with these medians.

All times in the following table are µs. Sync wall time measures the wrapped synchronization API, whereas main CPU uses `thread_time_ns` deltas. Context switches are `RUSAGE_THREAD` deltas around the sync call, summed over the 30 samples.

| Phase | Pool | Runner median / max | Sync median / max | Sync main CPU median | Sync voluntary / involuntary switches, total |
|---|---:|---:|---:|---:|---:|
| Eager prefill | 0 | 1629.752 / 2406.896 | 38.830 / 1037.631 | 21.024 | 77 / 0 |
| Eager prefill | 1 | 1569.673 / 2129.687 | 33.569 / 79.021 | 22.051 | 31 / 0 |
| Traced prefill | 0 | 756.500 / 1250.579 | 721.954 / 1215.983 | 22.603 | 77 / 6 |
| Traced prefill | 1 | 755.212 / 788.546 | 723.311 / 738.460 | 25.823 | 30 / 1 |
| Decode | 0 | 329.374 / 2227.925 | 300.829 / 2210.622 | 18.130 | 69 / 32 |
| Decode | 1 | 329.840 / 353.675 | 302.227 / 310.654 | 21.506 | 30 / 0 |

The default probe has 7/30 decode intervals above 0.5 ms, including 2 above 1 ms; pass-through has 0/30 in either category. Default traced prefill has 2/30 intervals above 1 ms; pass-through has none. Medians are nearly unchanged while the sampled tail and extra voluntary synchronization switches disappear with the pool bypass. This is positive evidence for **sensitivity of diagnostic completion tails to the pool boundary**.

### Exact counter examples and attribution limits

Default-run main TID is 843351. Eight threads have single-CPU affinity: 844870→3, 844873→2, 844884→1, 844885→0, 844886→3, 844887→2, 844888→1, and 844889→0. The pass-through run has 70 observed threads versus 78 in the default run, and none has single-CPU affinity. This matches removal of the two four-worker device-bound mesh pools in source. The generic `python` thread names do not independently identify which pool or the completion reader owns a TID.

| Default probe interval / sync ID | Phase | Sync wall, µs | Main CPU during sync, µs | Runnable-wait counters in enclosing snapshot, µs |
|---|---|---:|---:|---|
| 10 / 21 | Eager prefill | 1037.631 | 20.730 | CPU-2-pinned TID 844873: 1010.528 |
| 46 / 95 | Traced prefill | 1179.543 | 37.581 | CPU-2-pinned TID 844873: 1114.343 |
| 68 / 139 | Decode | 1008.406 | 28.172 | CPU-0-pinned TID 844885: 954.843 |
| 64 / 131 | Decode | 2210.622 | 14.205 | CPU-3-pinned TID 844870: 1198.211; unbound TID 844890: 985.080 |

The longest decode interval has only 31.699 µs of main-thread CPU across execute plus sync, two voluntary switches inside sync, and no main-thread involuntary switch inside sync. This is a blocked completion interval, not prolonged main-thread computation. Its broad snapshot reports only 8.015 µs of main-thread runnable wait; the long runnable waits appear on other threads. Source plus those counters support delayed worker/reader progress as a plausible contributor.

However, `/proc` snapshots occur sequentially before/after the measured timer and include collection time. Their deltas cannot be summed and subtracted from a single sync to claim exact causal decomposition. A counterexample is pass-through interval 83: the entire timed decode is just **324.811 µs**, while its enclosing snapshot reports **2877.835 µs** of runnable wait for unbound TID 848195 and **2037.236 µs** for `RealtimeProfile` TID 848445. Those waits clearly cannot all belong to the measured decode. This control prevents treating a large unrelated per-thread counter as proof of critical-path delay.

Both runs include threads named `RealtimeProfile` and `RtProfilerConsu`, despite `TT_METAL_DEVICE_PROFILER` being unset for these ordinary launches. Their existence and activity are observable; neither the environment variable nor their presence establishes that they caused the anomaly. `RUSAGE_THREAD` user/system CPU deltas also do not consistently track these sub-millisecond intervals tightly; use the recorded nanosecond thread CPU clock for the CPU comparisons above. Scheduler counters and their snapshot scope remain in the original sidecars for audit.

## Pass-through signposted profile

The recorded profile command was:

```bash
env TT_MESH_PASS_THROUGH_THREAD_POOL=1 models/autoports/qwen_qwen3_8_27b/tests/profile_optimized_multichip_decoder.sh gap_profile_passthrough --layer 3 --length 128 --repeats 1 --prefill-repeats 1 --trace-prefill
```

`gap_profile_passthrough.json` reports traced prefill 0.755472 ms, ordinary profiled decode 0.348336 ms, and signposted decode 0.387580 ms. Per-device decode accounting is:

| Device | Ops | Worker kernels, µs | Inter-op gaps, µs | Worker span, µs |
|---|---:|---:|---:|---:|
| 0 | 43 | 273.083 | 38.302 | 311.385 |
| 1 | 43 | 274.819 | 37.879 | 312.698 |
| 2 | 43 | 276.020 | 38.301 | 314.321 |
| 3 | 43 | 278.411 | 38.244 | 316.655 |

The same-run host-minus-longest-worker-span difference is **70.925 µs**, consistent with the ordinary approximately 60–80 µs overhead seen in the healthy profiles. In `.logs/tracy_ops_times.csv`, between signpost timestamps 6248401859 and 6249062430 ns, C++ execute takes 38.874 µs, finish takes 330.090 µs, and nested finish_nolock takes 327.886 µs. Thus the original 591/869 µs C++ completion tail is absent in this profile.

The healthy earlier default-pool `profile_trace_prefill_l3` already had finish_nolock 326.994 µs and a host-minus-worker-span difference of 51.086 µs. Consequently, one healthy pass-through profile is evidence that the selected path can run cleanly; it does not establish a necessary or sufficient cure for every historical outlier. The new profile also uses traced prefill and post-guard source, so it is not a perfectly matched rerun of the two original eager-prefill profiles.

## Hypothesis verdicts

| Hypothesis | Verdict | Evidence and limit |
|---|---|---|
| The saved anomaly was profiler-read processing or logging charged inside the host timer. | Refuted as the dominant explanation. | Reads/logging are outside the timer; existing C++ zones account for almost all its duration. |
| The saved anomaly was GIL/API-return overhead or the outer finish callback/barrier. | Refuted as the dominant explanation. | Bad profiles retain only approximately 15–18 µs outside execute+finish, and only approximately 2 µs between finish_nolock and outer finish. |
| Event-command submission through the device-bound pool can produce long completion tails in this workload. | Supported by the diagnostic A/B; boundary sensitivity verified. | Removing the pool removes eight pinned workers, extra voluntary waits, and sampled probe completion tails. Ordinary decode medians do not improve. |
| Scheduling delay on worker/reader threads explains the precise two historical outliers. | Still uncertain. | New slow default intervals have substantial runnable-wait counters on pool-associated threads, but snapshot timing is broader than the operation and historical runs lack those counters. |
| The original 2.009 ms traced-prefill median was a stable device-compute regression fixed by pass-through. | Refuted as stated. | Untouched default control already measures 0.738887 ms; new worker spans are stable. |
| Pass-through is useful for the selected launchers. | Verified for observed eager-prefill medians; selected by parent. | 6.498–8.583% lower eager-prefill medians in three paired workloads with passing correctness and essentially unchanged decode. No general runtime claim. |

## Final status and conditional follow-up

The selected launcher change is an evidence-backed host-overhead optimization, not a proven repair of a device kernel or a thread-pool correctness bug. The original completion anomaly is conservatively classified as a **transient host completion-path tail, with new evidence of pool/scheduling sensitivity**. Preserve its original artifacts and this qualification.

No further diagnostic experiment is necessary to justify the selected prefill improvement while the parent's final correctness and normal performance/profile refreshes pass. If large completion tails recur in those untouched final runs, one precise next experiment is justified: capture a wake-to-run scheduling trace with named pool/reader TIDs and add narrowly scoped timestamps around event-command pool wait, completion-reader event observation, and notification. Correlate those with the existing finish_nolock zone in the **same slow sample**. That would distinguish delayed event submission, a late device completion event, delayed reader execution, and delayed main-thread wakeup without relying on broad `/proc` intervals. Blocking replay alone is not sufficient because it enters the same finish_nolock implementation.
