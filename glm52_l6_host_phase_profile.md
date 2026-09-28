# GLM-5.2 L6 warm host-phase profile

Representative untraced full layer 6, 5,120-token chunk at 51,200-token KV depth, 8×4 Blackhole mesh, real GLM-5.2 weights and cached prefix. The run used the same CI-like settings as `glm52_l6_host_op_overhead.md`: `TT_METAL_SHM_TRACKING_DISABLED=1`, `LOGURU_LEVEL=ERROR`, and `TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0`. Ten warmups preceded ten measured iterations. Values below are medians of the ten measured calls (or paired internal calls) in a separate instrumented run, in microseconds. Temporary C++ timers added a little overhead, so these figures explain the split of host time and should not be compared directly with the uninstrumented latency baseline.

| Outer TTNN call | Output creation | Cache key | Runtime-arg patch | Enqueue | Within enqueue: cached compile/validate | FD command update | FD command write |
|---|---:|---:|---:|---:|---:|---:|---:|
| `all_reduce_async` gate (RS + AG) | 57 | 9 | 146 | 313 | 105 | 88 | 78 |
| `deepseek_prefill.combine` | 12 | 6 | 187 | 239 | 71 | 86 | 55 |
| `deepseek_prefill.dispatch` | 29 | 7 | 151 | 247 | 68 | 91 | 43 |
| `reduce_scatter` post-combine | 58 | 4 | 127 | 203 | 69 | 55 | 58 |
| `deepseek_prefill.offset_cumsum` (AG + primitive) | 53 | 8 | 45 | 167 | 56 | 33 | 50 |

The enqueue subcolumns are contained within **Enqueue**. Other enqueue work includes dispatch-command generation (about 10–18 µs per listed call), queue preflight and per-program bookkeeping. Cache-hit validation and cached-workload lookup outside enqueue were each below 2 µs per direct op. The `all_reduce_async` row sums the internal `ReduceScatterMinimalAsyncDeviceOperation` and `AllGatherAsyncDeviceOperation`; their enqueue times were 173 and 136 µs, respectively. The `offset_cumsum` row sums its internal `AllGatherDeviceOperation` and `OffsetCumsumDeviceOperation`, so its outer wrapper, reshape, and layout conversion are outside the table. The measured outer Python host calls in the instrumented fastest chunk were 577, 454, 452, 426, and 388 µs in table order.

The warm **cached compile/validate** stage does not JIT-compile kernels. `MeshWorkloadImpl::compile()` still walks each cached program and calls `compile_and_allocate()`, whose warm path validates circular-buffer core ranges, circular-buffer regions, and dataflow-buffer regions against live device state. Its 56–105 µs cost is therefore per-enqueue validation on this mesh. The fast-dispatch command queue updates each program's dispatch commands and writes the command sequences to local devices, accounting for another 80–169 µs across the listed calls. Workload tracking, runtime-ID setup, and Tracy bookkeeping together measured about 3 µs per direct call.

The runtime-argument update remains substantial: 187 µs for combine, 151 µs for dispatch, 127 µs for post-combine reduce-scatter, and 146 µs across the two gate collectives. Combine's descriptor fast path already avoids rebuilding descriptors; it applies resolved buffer bindings across per-coordinate programs. The remaining reduction would require fewer per-core bindings or a different program/runtime-argument layout. The gate's CCL semaphores cycle between calls, so their addresses must be refreshed. Standard `reduce_scatter` owns cache-stable semaphores and could avoid rewriting those particular fields, although that is only part of its 127 µs patch time.

Raw phase data from the temporary instrumentation: `/tmp/glm52_l6_host_phases3.csv`, `/tmp/glm52_l6_enqueue_phases3.csv`, `/tmp/glm52_l6_fd_phases3.csv`. The same run passed `scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/test_prefill_transformer_chunked.py::test_glm52_prefill_single_moe_layer_perf -k notrace -xvs`.

## Cache-hit patching experiments

All follow-up runs used the same 10-warmup/10-measured untraced L6 test and CI-like environment. The numbers below are outer host-call times from each run's fastest measured chunk. They include enqueue, so small differences are noisy; they are useful as a guard against a whole-call regression, not a direct patch-stage measurement.

| Variant | Best layer (ms) | Gate all-reduce (µs) | Combine (µs) | Dispatch (µs) | Post-combine RS (µs) | Offset-cumsum (µs) |
|---|---:|---:|---:|---:|---:|---:|
| Before this investigation | 14.787 | 556 | 450 | 459 | 387 | 395 |
| Cache each Buffer address once per descriptor patch | 14.921 | 569 | 449 | 442 | 399 | 400 |
| Hoist CCL semaphore address reads; skip stable all-gather semaphore writes | 14.904 | 570 | 457 | 447 | 418 | 391 |
| Cache each kernel's runtime-arg table across cores | 15.102 | 587 | 444 | 461 | 419 | 393 |
| Direct write to descriptor's validated runtime-arg slot, run 1 | 14.849 | 574 | 430 | 436 | 425 | 384 |
| Same direct-slot build, run 2 | 15.044 | 571 | 461 | 444 | 417 | 409 |
| Direct-slot build plus direct writes in CCL helpers | 14.821 | 572 | 446 | 436 | 390 | 389 |
| All-gather tensor addresses moved to common args (compact per-core layout) | 14.721 | 560 | 430 | 426 | 409 | 367 |

Only the descriptor direct-slot change was retained from the argument-patching experiments. Its dispatch call was lower in both repeats, while the other variants had no consistent target-call improvement. The common-arg change also altered kernel argument layout, so it was reverted without a gate-call improvement. The later CB-generation enqueue change is described below. JSON and test logs are in `/tmp/glm52_{addr_cache,ccl_patch,table_patch,slot_patch,slot_patch_repeat,all_slots,ag_common_compact}.{json,log}`.

## Why warm enqueue is still expensive

`MeshWorkloadImpl::compile()` runs on every enqueue. Its warm path calls `ProgramImpl::compile_and_allocate()` for every program, and that function explicitly rechecks circular-buffer core ranges, circular-buffer regions, and dataflow-buffer regions against live L1 allocations and service-core claims. The code comments say these checks cannot currently be skipped because device state may have changed since the previous enqueue. This accounts for 56–105 µs per listed outer call in the instrumented profile, even with JIT compilation cached.

Fast dispatch then updates cached command sequences and writes them. `update_program_dispatch_commands()` refreshes every cached local and remote CB configuration entry, DFB configurations, cross-node configurations, and any RTA copies. Update plus write measured 80–169 µs per listed call. A follow-up run with temporary timers inside that function split its update block further (medians over the ten measured iterations, summed across internal programs, µs):

| Outer call | CB config refresh | DFB config refresh | Cross-node config refresh | RTA copies | RTA update count / bytes |
|---|---:|---:|---:|---:|---:|
| Gate all-reduce | 56.3 | 1.9 | 2.2 | 1.5 | 0 / 0 |
| Combine | 64.2 | 1.1 | 2.2 | 0.8 | 0 / 0 |
| Dispatch | 73.8 | 1.2 | 1.5 | 0.8 | 0 / 0 |
| Post-combine reduce-scatter | 36.1 | 1.1 | 1.8 | 0.8 | 0 / 0 |
| Offset-cumsum | 20.2 | 1.1 | 1.4 | 1.0 | 0 / 0 |

Thus RTA copying is not the update bottleneck for these calls. The CB refresh loop is. At baseline, the source constructs `local_cb_config_updates` and `remote_cb_config_updates` for every CB when it assembles a command sequence, then rewrites address, size, page count, and page size on every enqueue. `UpdateDynamicCircularBufferAddress()` is used by other ops, and total size/page size can also be changed through host APIs, so skipping unchanged CBs requires explicit tracking. For validation, an allocator/service-core state generation would allow rechecking cached regions only when the relevant live state changed; there is no such generation gate in the current warm path. Raw split profiles: `/tmp/glm52_fd_update_split.csv` and `/tmp/glm52_fd_update_split2.csv`; temporary timers were removed after measurement.

## CB-generation change

`CircularBufferImpl` now increments a configuration generation when its local address, dynamic global address, total size, or page size changes. A cached command sequence records the generation already present in each CB payload. Fast dispatch refreshes a payload only when its generation differs. This leaves dynamic address and size/page updates visible on relaunch while avoiding the full static-CB loop for these warm programs.

| Run | Best layer (ms) | Median layer (ms) | Gate (µs) | Combine (µs) | Dispatch (µs) | Post-combine RS (µs) | Offset-cumsum (µs) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 14.787 | 14.865 | 556 | 450 | 459 | 387 | 395 |
| CB generation, run 1 | 14.528 | 14.629 | 550 | 398 | 377 | 384 | 374 |
| CB generation, run 2 | 14.534 | 14.649 | 539 | 395 | 389 | 393 | 388 |

Both runs used ten warmups and ten measured iterations with the same CI-like flags and passed the full L6 test. A freshly rebuilt `MeshWorkloadTestSuite.MeshWorkloadCBUpdate` also passed, including cached relaunches after CB total-size and page-size changes. Test outputs: `/tmp/glm52_cb_generation{,_repeat}.{json,log}` and `/tmp/glm52_cb_update_gtest_fresh.log`. The remaining enqueue costs are live-state validation (56–105 µs per listed outer call in the baseline instrumented run) and command-sequence write (43–78 µs).

## Cached lockstep L1 validation

After a successful full layout validation, a warm program with lockstep L1 allocation and no service-core claims caches the largest static CB/DFB region end. Each subsequent enqueue checks the current L1 allocation frontier once against that bound. A newly overlapping allocation still enters the full validator for the detailed error; hybrid allocation, service-core claims, and sub-device-manager changes also use full validation.

| Run | Best layer (ms) | Median layer (ms) | Sum of 114 op calls (ms) | Gate (µs) | Combine (µs) | Dispatch (µs) | Post-combine RS (µs) | Offset-cumsum (µs) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Previous, run 1 | 14.528 | 14.629 | 10.664 | 550 | 398 | 377 | 384 | 374 |
| Previous, run 2 | 14.534 | 14.649 | 10.643 | 539 | 395 | 389 | 393 | 388 |
| Cached validation, run 1 | 13.945 | 14.071 | 9.901 | 490 | 340 | 329 | 361 | 359 |
| Cached validation, run 2 | 13.925 | 13.982 | 9.868 | 482 | 355 | 330 | 342 | 350 |

Each row is the fastest measured chunk after ten warmups and ten measured iterations, except the median column. Both new runs passed the full untraced L6 test with the same CI-like flags. Outputs: `/tmp/glm52_simple_l1_validation{,_repeat}.{json,log}`. The improvement in best layer time was 0.58–0.61 ms (4.0–4.2%); summed host op time fell by 0.74–0.80 ms. The measurement includes normal run variation and has no temporary C++ timers.

`UnitMeshCQSingleCardFixture.TensixTestSubDeviceCBAllocation` passed after extending it to exercise a cached program before and after a new overlapping L1 allocation. Output: `/tmp/glm52_simple_l1_collision_gtest.log`.

## Command-write follow-up

Temporary C++ timers split `FDMeshCommandQueue::write_program_commands_to_devices` and `program_dispatch::write_program_command_sequence` during the same L6 workload. In the diagnostic run, the already-packed 32-device path took about 16 µs per call for a median 5.5 KB command, including roughly 4.4 µs each for the device copies, issue-queue publication, and fetch-queue notification. The single-device path took about 1.95 µs per call for a median 5.3 KB command split into nine fragments; fragment copies accounted for about 0.98 µs. The run invoked 1,591 packed multi-device writes and 16,704 single-device writes across setup, warmups, and measurements. Raw diagnostic logs: `/tmp/glm52_cq_write_profile.log` and `/tmp/glm52_cq_single_profile.log`.

An experiment packed each single-device one-shot command into one contiguous host buffer before writing it. Both runs passed the untraced L6 test, but the performance effect was not consistent:

| Variant | Best layer (ms) | Median layer (ms) | Sum of 114 op calls (ms) |
|---|---:|---:|---:|
| Before, run 1 | 13.945 | 14.071 | 9.901 |
| Before, run 2 | 13.925 | 13.982 | 9.868 |
| Packed single-device write, run 1 | 13.791 | 13.867 | 9.791 |
| Packed single-device write, run 2 | 13.906 | 13.958 | 9.915 |

The single-device packing experiment and all temporary timers were removed. There is no retained command-write change. Test outputs: `/tmp/glm52_cq_single_packed{,_repeat}.{json,log}`.

## Real-time profiler isolation and ring reduce-scatter patching

`TT_METAL_DISABLE_REALTIME_PROFILER=1` now skips real-time profiler manager initialization at mesh creation. The flag was tested alone, with the same release build, CI-like logging and memory-shim settings, and ten warmups plus ten measured untraced L6 runs. Afterward, a separate ring `reduce_scatter_minimal_async` change hoisted worker-shared addresses and wrote to the existing runtime-argument storage directly. Both variants passed the single-layer test.

| Variant | Best layer (ms) | Median layer (ms) | Sum of 114 op calls (ms) | Five minimal reduce-scatters (ms) |
|---|---:|---:|---:|---:|
| Profiler on, run 1 | 14.326 | 14.442 | 10.325 | 1.356 |
| Profiler off, run 1 | 13.718 | 13.806 | 9.715 | 1.308 |
| Profiler on, run 2 | 13.886 | 13.945 | 9.916 | 1.308 |
| Profiler off, run 2 | 13.628 | 13.724 | 9.547 | 1.258 |
| Profiler off + reduce-scatter patch, run 1 | 13.498 | 13.714 | 9.431 | 1.127 |
| Profiler off + reduce-scatter patch, run 2 | 13.523 | 13.603 | 9.430 | 1.137 |

The profiler comparison is an isolated A/B: the only runtime difference within each pair was the new flag. The disabled runs registered 1,578 JIT cache hits versus 1,580 with the profiler enabled, consistent with skipping its setup programs. The reduce-scatter patch lowered the measured sum of its five calls by 0.12–0.18 ms relative to the profiler-disabled runs; whole-layer results are subject to normal run variation. Outputs: `/tmp/glm52_rt_profiler_{on,off}{,_repeat}.{json,log}` and `/tmp/glm52_rs_patch{,_repeat}.{json,log}`.

## Full L78 profiler A/B

The 11-chunk, 10-iteration untraced GLM-5.2 prefill test passed twice on the same build with the ring reduce-scatter patch above. Both runs used the CI-like memory-shim and logging settings; the only runtime difference was `TT_METAL_DISABLE_REALTIME_PROFILER=1`.

| Profiler | Mean of 11 chunk medians (s) | Median of 11 chunk medians (s) | Chunk-median range (s) |
|---|---:|---:|---:|
| Enabled | 0.55355 | 0.551 | 0.533–0.584 |
| Disabled | 0.55309 | 0.551 | 0.533–0.584 |

The 0.45 ms per-chunk difference is too small to establish an end-to-end gain from one pair. An older run before the reduce-scatter patch averaged 0.561 s across chunk medians, so that comparison cannot isolate the profiler. Logs: `/tmp/glm52_full_prefill_rt_profiler_{on,off}.log` and `/tmp/glm52_full_prefill_simple_l1_validation.log`.

## Isolated profiler synchronization cost

A temporary 8×4 torus test called `ttnn.synchronize_device(mesh_device)` 100 times to warm up, then timed 500 empty synchronizations per process. The same release build and mesh settings were used for each on/off pair. The temporary test was removed after measurement.

| Run | Profiler enabled: median / p90 (µs) | Profiler disabled: median / p90 (µs) | Median difference (µs) |
|---|---:|---:|---:|
| Pair 1 | 255.58 / 279.90 | 173.70 / 187.40 | 81.88 |
| Pair 2 | 229.56 / 265.36 | 190.30 / 204.71 | 39.26 |

The profiler adds a measurable 39–82 µs to an otherwise empty mesh synchronization. The full-model untraced loop synchronizes once per chunk, so this direct cost is much smaller than a millisecond per chunk. It does not establish a meaningful profiler cost for the model's operation dispatches; the earlier single-layer op-sum difference is likely sensitive to run variation. Logs: `/tmp/glm52_rt_sync_{on,off}{,_repeat}.log`.

## High-bandwidth all-gather follow-up

The current seven `high_bw_all_gather` host calls total 1.051–1.067 ms in the two L6 runs after the ring reduce-scatter patch. The op already uses caller-provided outputs and common runtime arguments for tensor addresses. An older internal phase profile put cached key/validation at about 4 µs/call and its runtime-argument override at about 40 µs for five calls and 8–12 µs for two calls; those internal timings predate the generic dispatch improvements.

A temporary control-change probe on the current L6 path observed 135 gather overrides, each covering 32 programs. None changed `input_batch_index` or `gathered_dim_size`, so the scalar schedule was already stable. The five slower overrides correspond to the code path that patches changing external semaphore pairs across worker cores.

Moving those two semaphore addresses to common runtime arguments and reading them from common arguments in the kernels passed twice, but worsened the seven-call host sum from 1.051–1.067 ms to 1.095–1.129 ms. Best layer time rose from 13.498–13.523 ms to 13.627–13.692 ms. The experiment and temporary counters were reverted, and the release build was restored. Outputs: `/tmp/glm52_ag_ctrl.log`, `/tmp/glm52_ag_common_sem{,_repeat}.{json,log}`.

This leaves no demonstrated large gather-specific cache-hit opportunity in this workload. The remaining time is mainly the seven workload submissions plus the required semaphore/address refresh; further progress likely needs a shared enqueue improvement or fewer gather calls.

## Default all-gather and residual follow-up

The MoE glue `all_gather_async` cache-hit override now fetches runtime-argument tables and tensor/semaphore addresses once per program, then writes each worker's existing argument storage directly. The kernel is unchanged. Two untraced L6 runs with the CI-like settings and ten warmups plus ten measured runs passed. That one call fell from 266.0–267.4 µs in the ring reduce-scatter baseline to 221.8–237.2 µs. A full L78, 11-chunk, 10-iteration run passed but had the same rounded per-chunk medians as the preceding baseline (mean 0.553545 s), so it does not establish an end-to-end benefit.

The two block residual adds were then changed to `ttnn.add_`. The L6 test passed with a best chunk of 13.387 ms and 9.318 ms summed host op time. The two residual calls took 229.7 µs together, compared with 244.5–257.3 µs for the out-of-place pair in the two preceding runs. The MoE glue add remains out-of-place because its routed intermediate can be returned separately. GLM-5.2 L1 and L10 PCC checks passed in both untraced and traced modes with the in-place residuals. The full L78, 11-chunk, 10-iteration run passed; its mean of rounded chunk medians was 0.553545 s, identical to the preceding run. This does not establish an end-to-end gain. Logs: `/tmp/glm52_add_inplace.json`, `/tmp/glm52_add_inplace.log`, `/tmp/glm52_add_inplace_pcc.log`, and `/tmp/glm52_full_prefill_add_inplace.log`.

Two ring indexer argument-patching experiments were reverted after no improvement: a local per-kernel runtime-argument table cache and replacing the custom binder with the shared descriptor binder. Instrumentation showed the current override binds 544 tensors over 32 programs, with a median 86.9 µs in bindings and 17.4 µs in scalar controls. The failed candidates left the indexer source unchanged.

## Sparse MLA all-to-all output reuse

The two `all_to_all_async_generic` calls in sparse MLA totalled 456.3 µs in the preceding L6 run. The op already accepts a persistent output tensor, so an experiment reused one output per transpose shape from the model's shared `tt_ccl` object. The first L6 run passed with those two calls at 438.6 µs; the repeat passed at 433.0 µs. Best layer times were 13.563 and 13.331 ms respectively, against the preceding 13.387 ms. GLM-5.2 L10 PCC passed both untraced and traced. The full L78 run passed, with mean of rounded chunk medians 0.553455 s versus 0.553545 s before the experiment. That 0.09 ms change is not a measurable model improvement at 1 ms table precision, while the two persistent outputs retain significant DRAM across layers. The experiment was reverted. Logs: `/tmp/glm52_persistent_all_to_all{,_repeat}.{json,log}`, `/tmp/glm52_persistent_all_to_all_pcc.log`, and `/tmp/glm52_full_prefill_persistent_all_to_all.log`.

## MoE glue residual

With `return_intermediates=False`, the routed output has no later reader after the shared expert sum. An in-place add in this path lowered the MoE glue add host call from 162.4 µs in the preceding L6 run to 102.1 and 105.6 µs in two repeated runs. The other path still uses out-of-place `add` so its separately returned routed intermediate remains intact. The two L6 runs passed with best layer times 13.371 and 13.438 ms and summed host op times 9.221 and 9.271 ms. GLM-5.2 L10 PCC passed in untraced and traced modes. The full L78, 11-chunk, 10-iteration run passed with exactly the same eleven rounded chunk medians as before this change (mean 0.553545 s). Thus the call-level host gain did not become a measurable end-to-end gain. Logs: `/tmp/glm52_moe_glue_add_inplace{,_repeat}.{json,log}`, `/tmp/glm52_moe_glue_add_inplace_pcc.log`, and `/tmp/glm52_full_prefill_moe_glue_add_inplace.log`.
