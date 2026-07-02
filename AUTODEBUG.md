# AutoDebug Report

Scope: inspection-only diagnosis for the failed `Qwen/Qwen3-4B optimized-multichip-decoder` review. I did not open TT devices, run hardware tests, start servers, or reset devices. The implementation file changed while this inspection was in progress; the code references below describe the latest on-disk state, while the stage artifacts still reflect the reviewed state.

Status after remediation: the findings below drove the follow-up repair pass. The final optimized decoder now uses the lower persistent L1 all-reduce path for decode, removes the decode-down L1-to-DRAM-to-L1 bounce, proves trace replay with a replay-specific input and capture-output delta, includes Qwen TP4 fused CCL probes, measures a coherent sharded residual stack and rejects it with performance evidence, and regenerates final correctness/perf/watcher artifacts under `models/autoports/qwen_qwen3_4b/doc/optimized_multichip_decoder/`.

## Headline Findings

### H1. Persistent/preallocated all-reduce is not stage-earned, and the current lower-overload path still has avoidable movement

Type: code issue plus missing experiment.

Evidence:

- The reviewed README says row-parallel reductions use high-level `ttnn.experimental.all_reduce_async` on cluster axis 1 (`models/autoports/qwen_qwen3_4b/doc/optimized_multichip_decoder/README.md:16`) and says explicit semaphore/persistent-buffer setup was inspected but not measured (`README.md:151-154`).
- Current code now has a persistent-buffer path: `_all_reduce_hidden(..., use_persistent=True)` converts the partial to L1 width-sharded memory, obtains a scratch buffer plus semaphore, and calls the lower `ttnn.experimental.all_reduce_async(input, buffer, cluster_axis, mesh_device, semaphore, ...)` overload (`models/autoports/qwen_qwen3_4b/tt/multichip_decoder.py:602-622`). Decode WO and decode down call it (`multichip_decoder.py:917`, `multichip_decoder.py:744`), but prefill WO/down still use the high-level cluster-axis path (`multichip_decoder.py:785`, `multichip_decoder.py:753`).
- TTNN exposes both overload families: the cluster-axis overload and the buffer/semaphore overload are bound separately (`ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/all_reduce_async_nanobind.cpp:83-134`). The high-level cluster-axis overload without explicit semaphores lowers to `ttnn::reduce_scatter` plus `ttnn::all_gather` (`ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/all_reduce_async.cpp:349-425`). The buffer overload calls `ttnn::prim::all_reduce_async` (`all_reduce_async.cpp:453-482`).
- The lower primitive validator requires Blackhole input not be DRAM, and requires input, buffer, and output memory layouts to be `WIDTH_SHARDED`; the scratch buffer grid must contain the output grid and the scratch shard volume must be at least `output_shard_volume * ring_size` (`ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/all_reduce_async_device_operation.cpp:24-72`).
- Current decode down produces an L1 width-sharded matmul output (`multichip_decoder.py:733-742`), immediately converts it to DRAM (`multichip_decoder.py:743`), then `_all_reduce_hidden(..., use_persistent=True)` converts it back to L1 (`multichip_decoder.py:608-610`). Decode WO writes the partial to DRAM (`multichip_decoder.py:908-914`) and then stages it to L1 for the persistent all-reduce.

Why this explains a review failure:

The stage evidence still does not prove a persistent/preallocated CCL-buffer implementation. The current worktree partially implements one for decode only, but that implementation is not reflected in the stage docs/artifacts, is not A/B measured against the reviewed high-level path, and still introduces L1 -> DRAM -> L1 movement on decode down.

Smallest focused experiment/potential fix:

- Add a narrow CCL mode switch, for example `QWEN3_4B_MULTICHIP_CCL=highlevel,persistent_l1`, so the same correctness/perf harness can compare the reviewed high-level path and the lower buffer path without unrelated changes.
- For the persistent all-reduce candidate, use shape `[1, 1, M, 2560]`, `cluster_axis=1`, `topology=Ring`, `num_links=1`, BF16 payload, and L1 `WIDTH_SHARDED` output. With the current 8-core helper, expected output shard is `[32, 320]`; the scratch buffer shape is `[1, 1, M, 10240]` with shard `[32, 1280]`, satisfying the ring-size 4 volume rule. `M=16` prefill and `M=1` decode both pad to 32 rows in the memory config.
- Remove the decode-down DRAM bounce before measuring. Either let the lower all-reduce consume the existing 4-core L1 width-sharded down output `[32, 640]` with a matching scratch shard `[32, 2560]`, or do an L1-to-L1 reshard to the 8-core `[32, 320]` all-reduce layout. For WO, try making the matmul output directly match `_all_reduce_l1_memory_config(M, 2560)`.
- If this fails, the useful blocker is the exact validator/runtime error. The likely blockers are: Blackhole DRAM input, non-`WIDTH_SHARDED` input/output/buffer, scratch grid not containing output grid, or scratch shard too small for `ring_size=4`.

### H2. Fused CCL+matmul rejections are based on unrelated gates, not Qwen TP4 shape blockers

Type: missing experiment; not yet a legitimate limitation.

Evidence:

- The stage rejects `all_gather_matmul_async` because the common 1D attention config auto-enables it only for `num_devices == 8` (`README.md:137-140`, `models/common/modules/attention/attention_1d.py:1729-1764`). That common config is TP8-shaped, including a hard-coded `dim // 8` decode memory config (`attention_1d.py:1757-1764`).
- The actual `all_gather_matmul_async` API takes `persistent_output_buffer`, `dim`, `multi_device_global_semaphore`, all-gather and matmul memory configs, a matmul program config, and compute config (`ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_matmul_async/all_gather_matmul_async_nanobind.cpp:27-84`). Its validator requires rank-4 tensors, `dim=3`, batch size 1, and a 1D or 2D multicast matmul program config; it does not statically require TP8 (`ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_matmul_async/device/all_gather_matmul_async_device_operation.cpp:30-66`).
- The stage rejects `minimal_matmul_strided_reduce_scatter_async` because the GPT-OSS path gates it off on Blackhole (`README.md:139-140`, `fused_ccl_probe.log:3-8`). That gate documents a Blackhole race at `M_tiles=32` for S=1024 (`models/demos/gpt_oss/tt/attention/operations.py:120-138`). Qwen decoder/prefill rows in this stage are `M=1` or `M=16`, padded to one tile, so the cited gate is not row-specific evidence for this shape.
- The fused matmul-RS validator requires `dim=3`, batch size 1, Ring topology, three semaphores, and `MM output N_tiles % ring_size == 0` (`ttnn/cpp/ttnn/operations/experimental/ccl/minimal_matmul_strided_reduce_scatter_async/device/minimal_matmul_strided_reduce_scatter_async_op.cpp:26-130`). Qwen WO/down output hidden width is 2560, so `N_tiles=80`; `80 % TP4 == 0`.

Why this explains a review failure:

The review asked for an adapted TP4 fused all-gather-matmul/local-output candidate for WO/down, or an exact blocker. The inspected artifacts only show that a TP8-biased common helper and a GPT-OSS Blackhole gate decline the path. They do not test the Qwen TP4 tensor shapes.

Smallest focused experiment/potential fix:

- Build a Qwen-local `all_gather_matmul_async` probe from `tests/nightly/t3000/ccl/test_minimal_all_gather_matmul_async.py`. Use `dim=3`, Ring, one link, a persistent all-gather output buffer, and a `MatmulMultiCoreReuseMultiCast1DProgramConfig`.
- WO local-output candidate: input local attention `[1, 1, M, 1024]`; all-gathered matmul input `[1, 1, M, 4096]`; output-sharded weight equivalent to `[4096, 640]`; matmul output `[1, 1, M, 640]`. Try TP4-compatible output grids where `Nlocal_tiles=20` divides cleanly, such as `CoreGrid(x=4,y=1), per_core_N=5, out_subblock_w=5` or `CoreGrid(x=5,y=1), per_core_N=4, out_subblock_w=4`.
- Down local-output candidate: input local gated activation `[1, 1, M, 2432]`; all-gathered input `[1, 1, M, 9728]`; output-sharded weight equivalent to `[9728, 640]`; output `[1, 1, M, 640]`; use the same `Nlocal_tiles=20` output-grid family.
- If the replicated residual contract is kept, add a final all-gather of the `[1, 1, M, 640]` local output and compare against current matmul plus all-reduce. If pursuing lower movement, feed the local output into the sharded residual/RMSNorm family in H3.
- Build a Qwen-local `minimal_matmul_strided_reduce_scatter_async` probe from `tests/nightly/t3000/ccl/test_minimal_matmul_strided_reduce_scatter_async.py`. Use current BFLOAT4_B/LoFi policy, `dim=3`, `cluster_axis=1`, Ring, `N=2560`, `N_tiles=80`, and compare fused matmul-RS plus trailing all-gather against current matmul plus all-reduce. If it fails on Blackhole at `M_tiles=1`, that exact failure is the row-specific blocker the review requested.

### H3. Lower-movement residual-layout rejection only disproves a naive shape mix, not a stack-compatible sharded residual

Type: missing experiment; plausible stack family remains open.

Evidence:

- The current residual probe reduce-scatters a replicated hidden tensor to local width `hidden_size // 4`, then proves that naive operations fail: adding a width-sharded tensor to a replicated tensor, applying full-width RMSNorm gamma to a width shard, and feeding the sharded norm into a full gate/up matmul (`models/autoports/qwen_qwen3_4b/tests/test_multichip_decoder.py:438-507`).
- Distributed RMSNorm paths already exist. The common 1D path runs `ttnn.rms_norm_pre_all_gather`, gathers stats, then runs `ttnn.rms_norm_post_all_gather` with distributed weight (`models/common/modules/rmsnorm/rmsnorm_1d.py:274-315`). The 2D decode path does the same with `cluster_axis=1` and sharded decode memory configs (`models/common/modules/rmsnorm/rmsnorm_2d.py:158-205`).
- A fused CCL RMS path exists as `ttnn.fused_rms_minimal`; the nightly decode-config test constructs sharded input memory, a persistent stats tensor with shard shape `(32, 32)`, sharded gamma, and verifies PCC (`tests/nightly/tg/ccl/test_distributed_rms_norm_decode_configs.py:84-196`).

Why this explains a review failure:

The stage rejection only shows that a local width shard cannot be dropped into the existing full-width residual/RMSNorm/gate-up contract. It does not test a consistent residual family where residual, RMSNorm gamma, RMS stats, and next projection inputs are all distributed coherently through the next norm/MLP boundary.

Smallest focused experiment/potential fix:

- Probe a decode first, then prefill, stack family:
  - Keep post-WO or post-down output as a local hidden shard `[1, 1, M, 640]` after reduce-scatter or local-output fused matmul.
  - Keep the residual/base hidden in the same TP4 width-sharded layout so residual add is shard-to-shard, not shard-to-replicated.
  - Run distributed RMSNorm: `ttnn.rms_norm_pre_all_gather(sharded_hidden)`, gather stats across `cluster_axis=1`, then `ttnn.rms_norm_post_all_gather(..., weight=sharded_gamma)`, or use `ttnn.fused_rms_minimal` with a persistent stats tensor.
  - Feed the distributed norm output either to local-output QKV/gate/up candidates or all-gather exactly at the first unavoidable full-width boundary.
- Exact likely blockers to capture if the family fails: `fused_rms_minimal` topology/cluster-axis support on this 1x4 Blackhole mesh, stats tensor shape/memory-config mismatch, gamma sharding mismatch, or the next matmul requiring full K with no local-output equivalent.

### H4. Dominant matmul geometry sweeps are partially scaffolded in code but still lack stage evidence under the final policy

Type: missing experiment; some code scaffolding exists in the current worktree.

Evidence:

- The final prefill tt-perf rows still show dominant BFLOAT4_B/LoFi matmuls with DRAM input and small subblocks: QKV `32 x 2560 x 1536` at 41.509 us, WO `32 x 1024 x 1024` at 21.873 us, gate/up `32 x 2560 x 2560` at about 50.38/50.40 us, and down `32 x 2432 x 2432` at 49.491 us (`tt_perf_report_prefill.csv`).
- The traced decode table still has a QKV `32 x 2560 x 1536` row at 41.511 us with DRAM input and `out_subblock_w=1`, plus the 8392.649 us gap (`tt_perf_report_traced_decode.csv`). Decode down is improved by the DRAM-sharded path, reporting `32 x 2432 x 2560`, L1 width-sharded input/output, and 14.697 us.
- Current code added `QWEN3_4B_MULTICHIP_GEOMETRY` candidates for decode QKV (`qkv_1d_dram_24c_i5_s2`, `qkv_width_l1_10c_i8_s5`, `qkv_width_l1_8c_i10_s6`), gate/up (`gate_up_1d_l1_20c_i10_s4`, `gate_up_1d_l1_10c_i8_s8`, `gate_up_width_l1_10c_i8_s8`), WO (`wo_1d_explicit_8c_i4_s4`), and down (`down_dram_2c_i38_n40`) (`multichip_decoder.py:199-307`). Those modes are not represented in the reviewed README/work log or CSV artifacts.
- Prefill QKV, WO, gate/up, and down still use default DRAM-output matmuls with no explicit program-config sweeps in the inspected code paths (`multichip_decoder.py:650-656`, `multichip_decoder.py:711-728`, `multichip_decoder.py:745-753`, `multichip_decoder.py:777-784`).

Why this explains a review failure:

The review finding is not just that tt-perf gave advice. The stage has not closed the loop by showing measured geometry candidates under the final BFLOAT4_B/LoFi policy, especially for prefill. Current decode candidates may be useful, but without correctness, trace, and perf artifacts they are unearned.

Smallest focused experiment/potential fix:

- Run the current geometry modes one at a time with the same correctness plus `test_multichip_perf_signposts` harness and append one CSV per mode. Keep CCL mode fixed while sweeping geometry.
- Decode QKV shape: `M=32 padded`, `K=2560`, `N=1536`, `K_tiles=80`, `N_tiles=48`. Test the current 24-core DRAM and 8/10-core L1-staged modes, then keep only modes that improve traced replay timing and preserve PCC.
- Decode WO shape: local input width 1024, full output width 2560. The current explicit mode uses 8 cores, `in0_block_w=4`, `per_core_N=4`, `out_subblock_w=4`; measure it against default plus persistent CCL with movement accounted for.
- Decode gate/up shape: `M=32 padded`, `K=2560`, local `N=2432`, `N_tiles=76`. Current candidate grids use 20 or 10 cores and larger subblocks. Measure both gate and up, because the same program config applies to two expensive matmuls.
- Decode down shape: `K=2432`, `N=2560`, `K_tiles=76`. The existing 4-core DRAM-sharded path is legal because 76 divides by 4; the current 2-core candidate changes `in0_block_w=38`, `per_core_N=40`. Measure it and include whether it changes the all-reduce input layout/bounce.
- Add prefill-specific sweeps or state exact blockers. Prefill remains the slower final mode relative to a candidate in the artifacts, and its dominant matmuls are still the ones with DRAM-input and subblock advice.

### H5. Performance accounting is internally inconsistent for traced decode and final candidate selection

Type: evidence/accounting issue plus possible test gap.

Evidence:

- The README treats the Tracy 8392.649 us QKV op-to-op gap as a profiling/signpost artifact while also using the traced decode device table as final evidence (`README.md:104-110`).
- In `tracy/optimized_multichip_ops_final.csv`, the `PERF_MULTICHIP_TRACE_DECODE` signpost starts at host timestamp `37678807425`, but the row used in `tt_perf_report_traced_decode.csv` for QKV global call count `339971` starts at `37671402963`, inside the earlier `PERF_MULTICHIP_DECODE` capture window (`tracy/optimized_multichip_ops_final.csv:873-875`, `:1043`, `:1086`). The down row global call count `377856` also starts before the trace replay signpost (`optimized_multichip_ops_final.csv:1024`, `:1165`).
- `trace_decode_once` captures a `decode_forward`, ends trace capture, executes the trace under `PERF_MULTICHIP_TRACE_DECODE`, and returns the `output` tensor from the capture call, not an explicit replay-produced output (`multichip_decoder.py:941-958`). The existing trace test compares eager output to that returned tensor (`test_multichip_decoder.py:430-433`).
- The kept async all-reduce candidate reports prefill 2.391728 ms and traced decode 0.411478 ms, but the final default retest reports prefill 2.551568 ms and traced decode 0.412838 ms (`work_log.md:94-102`, `README.md:160-169`, `async_all_reduce_down_dram_perf.csv`, `perf_host_timings.csv`). Prefill regressed by about 0.160 ms from the candidate that was kept.

Why this explains a review failure:

The 8392 us gap may indeed be a profiling artifact, but the artifact currently appears to be from trace capture decode, not trace replay execution. That makes the traced decode per-op table mislabeled or at least ambiguous. Separately, the final default does not reproduce the faster kept prefill candidate, so candidate selection is not adequately accounted for.

Smallest focused experiment/potential fix:

- Re-profile with separate, unambiguous signposts for warmup decode, trace-capture decode, and `execute_trace` replay. Only call the device op table "traced replay" if the raw op timestamps fall inside the replay signpost window; otherwise report replay as host timing only and label the per-op table as capture-time decode.
- Make `trace_decode_once` or the trace test prove replay output correctness, not just capture output correctness. If `execute_trace` updates preallocated outputs in place, document and assert that exact tensor identity/content after replay.
- Repeat final default and the kept async candidate in one controlled A/B run, preferably 5-10 warmed iterations each, same commit, same env, same CCL mode, same geometry mode. Use min and median. If the 2.391728 ms prefill result was noise or from a different implementation, record that explicitly.

## Other Potential Issues

- The current worktree diverges from the stage docs in several ways: persistent decode all-reduce, decode geometry environment modes, and DRAM-sharded down support are in code, while the reviewed README/work log still describe a simpler high-level async all-reduce path. Before final review, pin the exact commit/worktree used for artifacts.
- The persistent all-reduce scratch tensor is created with logical shape `[1, 1, m, width * tp]` while the memory config pads `m` to 32 rows (`multichip_decoder.py:309-349`). This is likely acceptable because TTNN pads tile tensors, but if the persistent path fails, first check whether logical `m=1` or `m=16` conflicts with the padded L1 shard shape.
- The local-output fused all-gather-matmul path is not guaranteed to win. It moves the all-gather before the matmul and needs output-sharded weights. The issue is not that it must be adopted; the issue is that the current rejection is not based on a Qwen TP4 measurement or exact validator failure.
- `minimal_matmul_strided_reduce_scatter_async` fuses only the matmul plus reduce-scatter half. If the layer boundary stays replicated, a trailing all-gather is still needed. The fair comparison is fused matmul-RS plus all-gather versus current matmul plus all-reduce; the lower-movement comparison is fused matmul-RS feeding a sharded residual stack.

## Suggested Experiment Order

1. Freeze the worktree used for the next artifact pass and reconcile docs with code. Do not mix the reviewed CSVs with newly added geometry or persistent-buffer code.
2. Close persistent all-reduce first: A/B high-level versus lower buffer overload for decode and prefill, remove the decode-down DRAM bounce, and collect correctness, trace, and perf.
3. Fix traced-decode accounting: split capture and replay signposts, verify replay output, and re-label or regenerate the per-op table.
4. Run decode geometry modes already scaffolded in `QWEN3_4B_MULTICHIP_GEOMETRY`, one at a time, with CCL mode fixed.
5. Add prefill matmul geometry sweeps for QKV, WO, gate/up, and down, or record exact blockers if prefill cannot use those program configs.
6. Probe fused matmul-RS for Qwen WO/down with `M_tiles=1`, `N_tiles=80`, TP4 Ring, BFLOAT4_B/LoFi. Keep the exact Blackhole failure if it fails.
7. Probe TP4 local-output `all_gather_matmul_async` for WO and down, then decide whether to all-gather back to replicated residual or feed a sharded residual path.
8. Probe the sharded residual/RMSNorm family through at least one next norm plus gate/up or QKV boundary. Reject it only with an op-specific blocker.

## Files/Artifacts Inspected

- `.agents/skills/autodebug/SKILL.md`
- `.agents/skills/autofix/SKILL.md`
- `.agents/skills/optimize/SKILL.md`
- `models/autoports/qwen_qwen3_4b/tt/multichip_decoder.py`
- `models/autoports/qwen_qwen3_4b/tests/test_multichip_decoder.py`
- `models/autoports/qwen_qwen3_4b/doc/optimized_multichip_decoder/README.md`
- `models/autoports/qwen_qwen3_4b/doc/optimized_multichip_decoder/work_log.md`
- `models/autoports/qwen_qwen3_4b/doc/optimized_multichip_decoder/fused_ccl_probe.log`
- `models/autoports/qwen_qwen3_4b/doc/optimized_multichip_decoder/perf_host_timings.csv`
- `models/autoports/qwen_qwen3_4b/doc/optimized_multichip_decoder/async_all_reduce_down_dram_perf.csv`
- `models/autoports/qwen_qwen3_4b/doc/optimized_multichip_decoder/down_dram_candidate_perf.csv`
- `models/autoports/qwen_qwen3_4b/doc/optimized_multichip_decoder/down_dram_decode_only_perf.csv`
- `models/autoports/qwen_qwen3_4b/doc/optimized_multichip_decoder/num_links2_down_dram_perf.csv`
- `models/autoports/qwen_qwen3_4b/doc/optimized_multichip_decoder/bfloat8_cache_down_dram_perf.csv`
- `models/autoports/qwen_qwen3_4b/doc/optimized_multichip_decoder/tt_perf_report_prefill.csv`
- `models/autoports/qwen_qwen3_4b/doc/optimized_multichip_decoder/tt_perf_report_traced_decode.csv`
- `models/autoports/qwen_qwen3_4b/doc/optimized_multichip_decoder/tracy/optimized_multichip_ops_final.csv`
- `ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/*`
- `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_matmul_async/*`
- `ttnn/cpp/ttnn/operations/experimental/ccl/minimal_matmul_strided_reduce_scatter_async/*`
- `ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/*`
- `tests/nightly/t3000/ccl/test_minimal_all_gather_matmul_async.py`
- `tests/nightly/t3000/ccl/test_minimal_matmul_strided_reduce_scatter_async.py`
- `tests/nightly/tg/ccl/test_distributed_rms_norm_decode_configs.py`
- `models/common/modules/attention/attention_1d.py`
- `models/common/modules/mlp/mlp_1d.py`
- `models/common/modules/rmsnorm/rmsnorm_1d.py`
- `models/common/modules/rmsnorm/rmsnorm_2d.py`
- `models/common/rmsnorm.py`
- `models/demos/gpt_oss/tt/attention/operations.py`
