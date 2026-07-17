# AUTOTRIAGE

## Diagnosis

- This is not a trace-capture or trace-replay hang: the baseline trace completed, then the first fused warm invocation reached its final host synchronization and did not complete. No fused trace had begun.
- The first concrete reference-contract mismatch is the persistent workspace passed to `matmul_reduce_scatter_async`. The row-parallel O matmul produces `[1, 1, 32, 3072]`, so the reference Ring workspace uses that shape and its output is `[1, 1, 32, 768]`. The candidate supplies an intermediate of `[4, 1, 32, 3072]`. AutoFix source review subsequently established that `validate_intermediate_tensor` does not check shape, so the oversized workspace may be harmless surplus rather than the cause; it remains a contract correction to test, not a proven root cause.
- Root-cause confidence is **low** from the initial capture: the captured triage could not read call stacks or operation state, and the candidate enqueues reduce-scatter, statistics all-gather, distributed RMSNorm, and all-gather-matmul before its only completion boundary. Consequently the initial evidence identifies an unsupported fused invocation and a plausible first stuck stage, but cannot prove which device kernel was the first waiter. The candidate must not be accepted or timed in this form.

## Triage Evidence

- Reproduction: `FALCON3_RUN_MULTICHIP_FUSED_BOUNDARY=1` with `test_multichip_decoder.py::test_fused_reduce_scatter_distributed_norm_all_gather_matmul_boundary` on the TP4 `1x4` Ring mesh.
- The ordinary replicated boundary compiled, captured, replayed, and completed. This proves that the same mesh, input/weight tensors, O and QKV matmul program configurations, and ordinary Ring all-reduce were usable immediately before the failure.
- During the fused warm call, host logs showed compilation of `matmul_reduce_scatter_async`, distributed RMSNorm stages, and `all_gather_matmul_async`; no later progress appeared for more than 60 seconds. In `_trace_mesh_callable`, the next action after `function()` is `ttnn.synchronize_device(mesh_device)`, so the observed host waiter is warm completion at `test_multichip_decoder.py:191-192`, before `begin_trace_capture` at line 193.
- `logs/fused_boundary_hang_triage.txt` does **not** provide a device stop-site. Every attempted BRISC/TRISC/NCRISC call-stack read failed in the triage host layer because `noc_read` was invoked with a `memoryview` signature unsupported by the loaded UMD binding. The summary marks the other probes as run, but it contains no usable running-op, semaphore, kernel-PC, or call-stack record from which to identify the first stalled core.
- The process was terminated only after capture, devices were reset, and the four devices subsequently reported healthy. That makes permanent hardware failure unlikely; it does not distinguish an RS, AG, RMSNorm, or AGMM protocol wait.
- Explicit `sub_device_ids=[SubDeviceId(0)]` versus the candidate's bare `ttnn.synchronize_device(mesh_device)` is **not** a causal difference. `select_sub_device_ids` resolves an empty list to `device->get_sub_device_stall_group()` (`tt_metal/impl/buffers/dispatch.cpp:1830-1838`), and the test set that stall group to `[SubDeviceId(0)]` at lines 1040-1043.

## Source Evidence

### Shape and ownership contract

- Candidate O input and local O weight are `[1, 1, 32, 768]` and `[1, 1, 768, 3072]`; the matmul result consumed by reduce-scatter is therefore `[1, 1, 32, 3072]`.
- TP4 Ring reduce-scatter along dimension 3 produces one local hidden shard `[1, 1, 32, 768]`. The candidate's `rs_output` has this correct shape (`test_multichip_decoder.py:1053-1059`).
- For Ring, `ReduceScatterMinimalAsyncDeviceOperation::compute_output_specs` leaves `inter_shape` equal to the input padded shape and divides only the output scatter dimension by ring size (`reduce_scatter_minimal_async_op_device_operation.cpp:64-106`). Its validator calls `validate_intermediate_tensor` for a supplied workspace and requires exactly three global semaphores (lines 33-61), but `validate_intermediate_tensor` checks placement/layout properties rather than exact shape.
- The CCL reference fused test builds `persistent_intermediate_buffers` from the reduce-scatter input shape and builds the persistent output by dividing dimension 3 by `num_devices` (`tests/ttnn/unit_tests/operations/ccl/test_new_matmul_reduce_scatter.py:71-97`).
- The candidate instead allocates `rs_intermediate` as `[TENSOR_PARALLEL_SIZE, 1, rows, hidden]`, or `[4, 1, 32, 3072]` (`test_multichip_decoder.py:1045-1052`). The TP degree is not a batch multiplier for a Ring workspace; the tensor is already replicated to four devices.
- The fused wrapper validates the matmul, scatter dimension, and matmul program type only (`matmul_reduce_scatter_async_device_operation.cpp:23-50`). It neither calls `reduce_scatter_common_validates` nor validates persistent intermediate/output shapes or the three-semaphore count. Exact shape validation against the matmul result is absent in both the fused path and the helper used by standalone RS, so acceptance of the oversized workspace does not by itself explain the hang.

### Producer/consumer and semaphore ledger for the first warm call

| Stage | Producer | Consumer / expected count | Persistent state used by candidate |
|---|---|---|---|
| Fused O matmul -> RS | O matmul produces 96 width tiles per rank (`32x3072`) and its internal fused signaler wakes RS workers | Four-rank Ring reduces four partials and emits 24 width tiles per rank (`32x768`) | RS semaphore set A: three global semaphores; malformed intermediate `[4,1,32,3072]`; correct output `[1,1,32,768]` |
| RMS statistics AG | `rms_norm_pre_all_gather` produces local BF16 statistics | Four ranks must contribute once; `rms_norm_post_all_gather` consumes gathered statistics once | AG semaphore set A: two global semaphores; barrier A |
| Fused normalized-input AG -> QKV matmul | Each rank provides local normalized `[1,1,32,768]`; Ring AG forms logical `[1,1,32,3072]` | QKV matmul consumes all four hidden shards and produces local `[1,1,32,1280]` | AG semaphore set B: two global semaphores; barrier B; correct AG persistent output `[1,1,32,3072]` |
| Host completion | All workers in configured subdevice 0 | One completion event after the entire chain | Bare synchronize resolves to the configured stall group |

- `TT_CCL` creates double-buffered banks for three cluster-axis choices. Each RS bank has three semaphores, each AG bank has two, and each bank has a barrier (`models/common/modules/tt_ccl.py:81-122`). This matches the source validators.
- On the first fused warm call, RS obtains bank A. The two AG calls obtain AG A then AG B, and barrier A then barrier B. There is no same-call global-semaphore or barrier alias between the statistics all-gather and the fused all-gather-matmul. A semaphore-count or first-call bank-collision hypothesis is therefore refuted.
- The chain has no intermediate host completion, signpost, or one-op readback (`test_multichip_decoder.py:1106-1171`). The final synchronization can only prove that at least one upstream producer failed to satisfy a downstream wait; it cannot identify which stage.

### Trace helper risk, separate from the observed warm hang

- If warm completion is repaired, the current timing helper presents a second unsupported protocol. It captures one set of global semaphore addresses and then enqueues 100 executions of that trace before one synchronization (`test_multichip_decoder.py:197-202`, called with `iterations=100` at lines 1184-1189). Host-side `TT_CCL` cycling does not run during `execute_trace`, so all 100 replays use the same captured RS/AG/barrier addresses.
- The fused RS reference executes a captured trace once before synchronizing (`test_new_matmul_reduce_scatter.py:248-262`). The fused AGMM reference synchronizes after every replay (`test_minimal_all_gather_matmul_async.py:251-270`). Neither reference establishes that 100 unsynchronized replays with one semaphore bank are safe. This cannot explain the current pre-trace warm hang, but it would invalidate the proposed latency measurement and could create a later semaphore-generation wait.

## Downstream Effects

- A reduce-scatter producer that fails to finish leaves `local_projected` unavailable. The add, pre-RMS statistics, stats AG, post-RMSNorm, and AGMM are then downstream victims even if their own semaphore contracts are correct.
- A completed RS followed by a stats-AG wait leaves post-RMSNorm and AGMM downstream. A completed distributed RMSNorm followed by AGMM failure leaves only final output/completion waiting. With no usable call stacks and no stage-local synchronization, these fanouts are observationally identical at the host.
- Trace teardown, tensor cleanup, and subdevice-manager reset in the `finally` block are not reached while warm synchronization is stuck. They are effects of the hang, not candidate root causes.
- The oversized RS workspace also consumes four times the required DRAM. Even if an implementation happens to ignore its surplus pages, relying on that is outside the documented/tested persistent-buffer contract and makes the candidate unsuitable for production evidence.

## Proposed Fix

1. Correct the Ring workspace before any further hardware run: allocate `rs_intermediate` as `[1, 1, rows, hidden]`, exactly the fused O matmul output shape. Keep `rs_output` as `[1, 1, rows, hidden / TP]`.
2. Turn the candidate into a staged diagnostic before restoring the full boundary:
   - Run only `matmul_reduce_scatter_async`, synchronize `sub_device_ids=[sub_device_id]`, and read back/PCC both returned tensors.
   - If that fails, run the same geometry as ordinary matmul plus `reduce_scatter_minimal_async` with the corrected buffers. Then try fused RS with `num_links=1`. This separates fusion from TP4/two-link Ring routing.
   - After RS passes, add local residual and distributed RMSNorm, synchronize/read back after the statistics all-gather, and compare to ordinary RMSNorm.
   - Run `all_gather_matmul_async` alone with a known local input and persistent AG output. Start with one link if the two-link fused case lacks a passing reference, then test two links.
   - Only after each producer/consumer pair passes should the whole boundary be chained and timed.
3. For trace validation, begin with one captured replay followed by subdevice synchronization. Then repeat with a synchronization after every replay, matching the fused AGMM reference. Do not enqueue 100 replays per completion until a focused stress test proves semaphore rearming is safe; otherwise capture multiple alternating semaphore banks or retain per-replay completion for this candidate.
4. Clear the loaded subdevice manager as the CCL reference tests do after resetting the stall group. This is cleanup hardening, not a diagnosis for the current first warm call.
5. Outside this stage-local model scope, harden `MatmulReduceScatterAsyncDeviceOperation::validate_on_program_cache_miss` to validate the reduce-scatter contract against the **matmul output spec**, including persistent intermediate/output shapes and the three-semaphore count. Its current output-spec path also constructs RS specs from the matmul input at lines 61-69; that should be audited because the RS program actually consumes the matmul output.
6. For another hang, run one isolated stage under `scripts/run_safe_pytest.sh --dev` and fix or align the triage/UMD `noc_read` binding first so running-op, worker call-stack, Ethernet, and semaphore evidence is captured before termination.

## Uncertainty

- No device kernel PC, running operation, semaphore value, Ethernet credit state, or watcher record was recoverable. The exact first waiting kernel is therefore unproved.
- The RS workspace mismatch is the first verified source-contract violation and the best initial repair, but because it is oversized rather than undersized, the implementation may only address the required prefix. If isolated corrected RS still hangs, the next highest-value discriminators are `num_links=1` versus `2`, fused versus unfused RS, then standalone stats AG and standalone AGMM.
- TP4 Ring and two links are individually exercised by the production decoder's ordinary collectives, but the located fused RS and AGMM references use an eight-device mesh with one link. They do not prove the exact Blackhole TP4/two-link fused geometry.
- The bare versus explicit subdevice synchronization spelling is equivalent in this setup and should not be presented as a fix.
- The repeated-trace semaphore concern is source-supported but prospective: it cannot cause the observed warm hang because trace capture had not begun.
