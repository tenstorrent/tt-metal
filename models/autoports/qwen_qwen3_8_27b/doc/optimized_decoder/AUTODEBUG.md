# AutoDebug: retained L1 warmup output changes trace preparation

Inspection date: 2026-10-23. Checkout HEAD: `22612a5fc90813932f3212bc3d8d842fa9142706`.

## Findings

**The minimal fix is to release the eager decode output before beginning capture.** The original runner kept `warm` alive. With `carry_residual=true`, that output is a BF16 L1 width-sharded tensor; capture therefore started with a different live L1 allocation state from warmup. The parent added only `del warm` after the eager PCC check while this investigation was running. Its new layer-0 artifacts show successful capture, deterministic replay, changed-input correctness, and normal teardown. A second layer-0 run and a layer-3 control also pass.

**The strongest specific cache mechanism is the convolution-history TILE-to-ROW_MAJOR conversion in layer 0.** It lowers to native `untilize_with_unpadding`, whose operation attributes include an L1-occupancy-dependent `enough_space_height` flag. A retained output at a low L1 address can flip that flag, producing a new program-cache entry during capture—even when both flag values select the same block-interleaved factory. The source and concrete sizing support this explanation; the failed run did not log the offending operation/hash or L1 addresses, so the exact cache miss remains an inference.

**A separate capture-exception cleanup defect remains.** If `decode()` throws between begin/end capture, the runner closes the mesh without ending the capture. The runtime remains in bypass-recording mode and queue destruction can wait for a completion event that was never sent to the device. This provides a source-supported explanation for the reported cleanup hang without requiring a stuck decoder kernel, although no saved host stack identifies the exact blocking frame.

This investigator ran no device workloads, resets, or hardware probes and edited no source. Hardware results below are parent-produced artifacts, inspected read-only. Three xhigh subagents independently audited lifecycle, lowering/cache keys, and layout conversion. This report incorporates a claim-review pass.

## Evidence and revision boundaries

The original `record_control_l0.log`, read at investigation start, contained:

```text
PREFILL_PCC 0.9999707275339088
EAGER_DECODE_PCC 0.9999801774045264
2026-10-23 16:06:44.485 ... Cannot load new binaries during trace capture.
This program is not yet in program cache. Warm up before capturing a trace.
Terminated
```

The parent subsequently reran the sweep and overwrote that filename with a passing log. The original error above is an observation from the initial read, not the current contents of that path. The preserved `runner_before_warmup_fix.py` contains the original runner. A byte-for-byte diff against the current runner shows exactly one added line: `del warm` at current `tests/run_optimized_decoder.py:314`.

| Artifact | Direct observation |
| --- | --- |
| `runner_before_warmup_fix.py:312-318` | One eager decode; host PCC; restore; synchronize; capture, with `warm` still referenced. |
| `released_warm_l0.json:2-42` | B1/S129, layer 0, real weights, recorded activations, original `record_control` policy; decode PCC 0.9999801774; changed-input PCC 0.9999913878; deterministic replay. |
| `released_warm_l0.log:28-32` | Normal device close completes after the result. |
| New `record_control_l0.json` | Same policy, provenance, and correctness as `released_warm_l0.json`; only the host timing field differs. |
| New `record_control_l3.json:2-42` | Same fixed runner, decoder provenance, and policy; decode PCC 0.9999848675; changed-input PCC 0.9999935130; deterministic replay. Its log completes normal close. |
| `norm_kv_l3.json` | Earlier successful full-attention trace, synthetic activations, no `carry_residual` entry, older decoder SHA. Supporting contrast, not a one-variable control. |

Source fingerprints:

- Original runner/saved copy: `9962d9c54f25ba0ae5ecfb8f0ba89671b872fb4ce40231cf0d4de377f9734a7b`.
- Fixed runner: `eacb0a7c91a6c3b27e6e2a811bb8cf4b0c14b36c854fffd6dcccf2ab1b194b55`.
- Decoder inspected at start and recorded by the new passing runs: `5b56310d150c0040e8a329a1d9941597e36d0c27db76c375aa976af16caf6af3`.
- Earlier `norm_kv_l3.json` decoder: `036fb1fde51c0f34ef7846ed1d722de283ce06fa0a8599aab30d0789e589a393`.

The parent continued optimization during the investigation. The later live decoder was observed at SHA `e919e3eeff00ca4777d83b9f9acde19bedb007b932d7c43ca21b44d0a81ee4fa`, with additional weight-padding and MLP-layout options. **All `optimized_decoder.py` line references and allocation reasoning in this report refer to the preserved [candidate2.py](candidate2.py) snapshot**, verified to have SHA `5b56310d...` matching the inspected decoder and successful control artifacts. They do not refer to the moving live file or certify those subsequent changes. The runner's saved/current diff still consists only of `del warm`.

The failed run did not emit a provenance JSON: the runner writes it only after successful trace work. Thus its exact decoder fingerprint is inferred from the prepared source, not proven by a failed-run manifest. The successful one-line intervention is strong evidence at the caller-lifetime boundary, but it does not establish which operation missed.

## Allocation and layout chain

The scoped failure is `record_control_l0`: batch 1, prefill length 129, then one-token decode. The sweep policy is DRAM-sharded weights, LoFi matmul, BFP4 gate/up, default BFP8 down/attention weights, `block=4`, sharded norms, residual carry, and BFP8 KV. Layer 0 is linear attention: its state is BF16 convolution history plus FP32 recurrence; the KV dtype setting is not exercised there.

Source: [preserved optimized_decoder.py snapshot](candidate2.py), especially lines 69-83, 133-148, 156-234, 236-308, and 364-406.

| Value/operation | Effective decode configuration |
| --- | --- |
| Input hidden state | BF16 TILE, logical `[1,1,5120]`, DRAM. |
| Input/post-attention norm | Reshape to `[1,1,1,5120]`; eight row-wise cores; width shards `[32,640]`; BF16 L1 output, non-inplace; HiFi4/FP32 compute. |
| Projections | Explicit `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`; BF16 output regardless of weight dtype. Input shard widths: 640 for mix/gate/up, 768 for out, 2176 for down. |
| Mix output | Linear-attention packed width 16512; converted back to DRAM even with residual carry. |
| Out/down outputs | Logical `[1,1,5120]`; eight `[32,640]` L1 shards retained with `carry_residual=true`. |
| Final residual addition | `_finish` returns `ttnn.add(x, mlp, memory_config=mlp.memory_config())`; output remains L1. |
| Eager `warm` | 20 BF16 tiles per shard = `20 * 2048 = 40960` bytes on each of eight cores. It owns a distinct output allocation. |
| Prefill result | Each chunk's `_finish` output is explicitly converted to DRAM before concatenation, including the one-token tail. |

The matmul output geometry is derived in `ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:2834-2876`: `per_core_M=1`, `per_core_N=ceil(5120/(32*8))=20`, eight row-wise cores, shard `[32,640]`. Carrying residuals changes allocation lifetime and placement; it does not change those output dtypes to BFP4.

The original runner's state sequence was:

| Phase | Completion boundary | Live output state |
| --- | --- | --- |
| Prefill | `ttnn.to_torch(actual)` completes the host read | `actual` is DRAM. |
| Eager decode | `ttnn.to_torch(warm)` completes the host read used for PCC | `warm` stays referenced in L1. |
| Restore | Copies backup state; explicit `synchronize_device(mesh)` follows | Does not release `warm`. |
| Begin/capture | Programs are recorded; they are not executed as an ordinary eager decode | Starts with the extra L1 allocation; a new uncached workload is rejected. |
| Exception | Skips end/release; outer `finally` closes mesh | Capture mode still active. |

The explicit synchronization before capture matters: eager PCC is not the only evidence of completion. There is no need to explain this failure as a latent earlier decoder kernel still running when capture begins.

## Specific lower-level mechanism: untilize admission flag enters the cache key

The layer-0 decode history update at `optimized_decoder.py:252-254` evaluates:

```python
ttnn.to_layout(projected[..., :10240], ttnn.ROW_MAJOR_LAYOUT)
```

Its input is BF16 TILE, DRAM interleaved, logical `[1,1,10240]`, padded `[1,32,10240]`. Output is BF16 ROW_MAJOR DRAM, logical `[1,1,10240]`. It supplies the last history row before concatenating/copying the three-row convolution state.

Lowering and cache evidence:

1. `ttnn/cpp/ttnn/operations/core/to_layout/to_layout_op.cpp:169-182` directly invokes `untilize_with_unpadding` with inclusive logical ends. This path does not pass through the ordinary `untilize` codegen dispatcher.
2. `.../data_movement/untilize_with_unpadding/untilize_with_unpadding.cpp:168-208` computes `enough_space_height` before dispatch and passes it as an operation attribute. The output reservation is zero because output is DRAM (`.../data_movement/common/common.cpp:547-555`).
3. `.../data_movement/common/common.cpp:614-631` tests **strictly greater than** the full-row input/output CB estimate. Here it is `320 tiles * (2048 + 2048) = 1310720 bytes`.
4. `.../operations/core/program_cache_l1.hpp:18-34` bases available space on the **lowest occupied L1 address minus CB base**, capped by bank capacity. It is not total free memory. A small surviving tensor low in the address space can exclude large holes above it from the estimate.
5. `UntilizeWithUnpaddingParams` contains `enough_space_height` (`.../device/untilize_with_unpadding_device_operation_types.hpp:11-17`). The op has no custom hash; `ttnn/api/ttnn/mesh_device_operation_adapter.hpp:250-258` hashes the attributes and input.
6. `.../device/untilize_with_unpadding_device_operation.cpp:88-118` chooses the block-interleaved factory when the flag is false. When true, this one-tile-high, 320-tile-wide input also selects block-interleaved through the wide-row/core-utilization heuristic. **Different cache key does not imply different selected factory or kernel source.**
7. A newly created workload without committed binaries hits the exact observed guard in `tt_metal/distributed/mesh_workload.cpp:171-200`. A disk JIT-cache hit cannot replace the required device-resident, committed program workload.

The selected factory is `UntilizeWithUnpaddingMultiCoreBlockInterleavedProgramFactory`. It sizes a shared `BlockPlan`, creates input/output dataflow buffers, and binds `reader_unary_interleaved_wh_multicore_metal2.cpp`, `writer_unary_stick_layout_wh_multicore_metal2.cpp`, and `untilize_wh_metal2.cpp` (`.../factories/untilize_with_unpadding_multi_core_block_interleaved_program_factory.cpp:80-143,168-231`). The cache failure occurs before dispatch of the newly created workload; these kernel names identify the candidate lowering, not a proven stuck kernel or CB mismatch.

### Concrete placement example and its uncertainty

Source-derived default Blackhole values are L1 ceiling 1572864 and CB base 111744, giving empty program span 1461120 bytes (`tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h`; `tt_metal/llrt/hal/tt-1xx/blackhole/bh_hal_tensix.cpp:43,68-69`; default L1_SMALL is zero). The following is an illustrative ordinary top-down allocation walk, **not an observed address dump**:

| Allocation/lifetime | Example start, expressed below L1 ceiling T |
| --- | --- |
| Input norm output; input conversion freed afterward | `T - 81920` |
| Mixer output reuses the top free 40960-byte slot | `T - 40960` |
| Converted residual, first residual sum, post norm | `T - 122880`, `T - 163840`, `T - 204800` |
| Down input `[32,2176]`, then down output `[32,640]` | `T - 344064`, `T - 385024` |
| Down input freed; final add reuses the high end of that hole | `T - 245760` |

Python keeps the outer input norm and mixer, plus `_finish`'s residual, first sum, post norm, and down result, alive while allocating the final add. After return, their holes can be freed while the final output remains low. In this example:

- Eager conversion sees `1490944 - 111744 = 1379200 > 1310720`: flag true.
- Capture with retained `warm` sees at most `1327104 - 111744 = 1215360 < 1310720`: flag false.

This is why subtracting only the output's 40960-byte size from the empty L1 capacity understates its possible effect. Actual cache-owned buffers, allocator policy, and hole reuse can alter these addresses. The artifact lacks `warm.buffer_address()`, live L1 bounds, and the operation hash at failure. Treat this calculation as a concrete mechanism consistent with the intervention, not proof of the exact original allocation sequence.

## Why cleanup can hang after the host-side error

The current runner still has unguarded begin/decode/end at lines 317-319, a happy-path release at line 376, and unconditional mesh close at lines 402-403. Preserved runner locations are 316-318, 375, and 401-402.

| Runtime transition | Source and consequence |
| --- | --- |
| Begin capture | `tt_metal/distributed/fd_mesh_command_queue.cpp:1389-1406` saves/resets dispatch state, sets trace ID, enables sysmem bypass. |
| Capture normal programs | `:439-452` appends trace nodes and returns, without normal execution. |
| Failed new binary load | `mesh_workload.cpp:194-200` throws while capture remains active. |
| Normal end, skipped on failure | `fd_mesh_command_queue.cpp:1654-1673` clears trace state, restores dispatch counters/config state, disables bypass. |
| Mesh close | `mesh_device.cpp:1093` destroys queues. |
| CQ destructor | `fd_mesh_command_queue.cpp:240-247` invokes blocking completion cleanup; `:367-404` emits an event and waits for its completion. |
| Bypass still enabled | `tt_metal/impl/dispatch/system_memory_manager.cpp:585-605` writes into the host bypass buffer and advances its offset, rather than submitting that command to hardware. |

The resulting producer/consumer mismatch is a host completion waiter expecting an event whose producer command remains recorded in host memory. `release_trace` alone does not perform `record_end` or restore bypass state (`mesh_device.cpp:1589-1607`).

Saved triage is consistent with an idle worker side and dispatch waiting for commands: no running model op, no active model CB failure, prefetch in `fetch_q_get_cmds`, dispatch waiting for upstream data (`triage/trace_warmup.txt:199-236`). It does **not** prove a clean system: the detailed capture reports a NoC mismatch at lines 218-220 and several failed symbol reads/core-state inconsistencies. The summary's `pass` entries mean script completion, not that every underlying hardware check passed. Those anomalies do not identify the failing program and do not establish a decoder NoC bug. There is no saved host stack to prove the destructor event wait was the observed blocking frame.

## Minimal fix and remaining work for the parent

The one-line lifetime fix is already present in the current runner and supported by the new parent-run artifacts:

```python
warm = decode()
print('EAGER_DECODE_PCC', pcc(ttnn.to_torch(warm), expected_decode), flush=True)
del warm
restore()
ttnn.synchronize_device(mesh)
tid = ttnn.begin_trace_capture(mesh, cq_id=0)
```

Keep the captured output alive for replay. Explicit `ttnn.deallocate(warm)` is an alternative when ownership is known, but is not needed to enlarge the demonstrated fix. Extra warmup iterations that retain prior outputs can themselves change allocation state; prefer identical output ownership at warmup and capture.

Add exception-safe trace lifecycle handling separately: if the capture body throws, best-effort end the active partial capture, release it without replay, preserve the original exception, then close. Put trace release in outer cleanup for failures after successful end as well. An error while ending capture itself requires care; this investigation does not claim that a Python cleanup guard can recover every partially failed runtime state. No exception-cleanup fix or fault-injection test was performed here.

If exact operation attribution is still needed, use a bounded parent experiment that records the candidate untilize's `enough_space_height`, program hash/cache hit, lowest occupied L1 address, and output buffer address in both calls. An uncaptured second decode with the old retention policy can reveal the new entry without intentionally failing capture. Release its output and restore state between controlled trials. Stable total cache-entry counts alone are weaker than identifying the changed operation.

The B1/S129 success does not certify the other sweep variants, larger batches, long contexts, profiler behavior, or later full-model output ownership. For the reported case there is no evidence requiring weight/dtype changes, disabling residual carry, disabling program-cache allocation tracking, resetting hardware, or patching a compute kernel.

## Other potential issues and hypotheses rejected during review

| Candidate | Adjudication |
| --- | --- |
| Sharded 5120 RMS allocator-driven plan | Its hash uses attributes/tensors directly; only the interleaved RMS branch computes the live-L1 plan (`layernorm_device_operation.cpp:27-51`). It is not the identified direct key mechanism. |
| Small width-128 RMS in layer 0 | Source sizing gives small-program footprints 163840 bytes for FP32 L2 norm and 131072 for BF16 output norm. Extreme pressure would fail the streaming-footprint check rather than create the proposed valid alternate small-width plan. We found no sequence-matched evidence for this explanation. |
| Automatic matmul config changes | Decode supplies an explicit DRAM-sharded program configuration; the automatic planning path is bypassed (`matmul_device_operation.cpp:3229-3247`). |
| Every changed buffer address changes the cache key | False. DeviceStorage reflection is empty; generic Tensor hashing does not include device buffer addresses. Binary keys also use tensor/config metadata, not raw addresses. |
| Custom recurrence hashes runtime buffer addresses | False for the called local implementation at `optimized_decoder.py:409-496`: the addresses are runtime argument values; generic-op hashing includes runtime argument sizes, not those values (`operations/generic/device/generic_op_device_operation.cpp:53-68,116-123`). |
| Ordinary untilize codegen tier caused this conversion miss | Wrong lowering for this call: `to_layout` goes directly to `untilize_with_unpadding`. |
| `trace_region_size=0` means no trace capacity | False: mesh trace code explicitly treats zero as dynamic allocation mode (`mesh_device.cpp:1647-1651,1677-1684`). |
| KV precision or recorded activation values explain layer-0 capture rejection | KV is unused by linear attention, and this candidate key depends on layout/capacity, not numerical values. Eager/reference results and the one-line lifetime fix argue against changing precision. |
| Remove `enough_space_height` from all cache keys | Unsafe blanket fix: the flag can select different factories for other shapes. A future core improvement could canonicalize irrelevant planner attributes or hash the effective plan, but that is broader than the demonstrated harness fix. |

Final review checked the source chain, parsed the runner/decoder as Python syntax without imports, verified the one-line runner diff and the three parent-result provenance/accuracy records, and recomputed the CB and illustrative L1 arithmetic. The snapshot was matched by SHA-256. No target code or hardware-dependent test was run by this investigation.

The final diagnosis is strongest at the caller allocation-lifetime boundary. The exact untilize cache-key transition and exact cleanup blocking frame are explicitly unresolved runtime details, with concrete source-backed predictions above.
