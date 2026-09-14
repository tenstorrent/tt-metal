# AUTOTRIAGE

2026-09-13, stage 8 datatype sweep, Qwen/Qwen3.8-27B. Fresh isolated
investigation against live checkout HEAD
`624f6352a9ca1c339870dbe517e564910bb71b15` plus stage-owned precision changes.
This report was written before any proposed implementation edit. The investigator
read source, logs and process records only; the coordinator owns all device
capture, process termination, recovery and reruns.

## Diagnosis

The earliest proven stalled boundary is a **blocking host-to-mesh tensor upload
waiting for a command-queue completion event during model initialization**.
The current source and native main-thread stack tie this upload to the automatic
pinned-memory path. The precise missing device transition and root cause remain
unproven because the current device capture timed out with no output.

This is not evidence that BFP4 KV allocation, cache updates, attention reads or
traced decode failed: none of those candidate-dependent operations has started.
The stalled candidate cannot yet supply an accuracy or performance rejection.

Original command:

```bash
QWEN_PRECISION_CONFIG=models/autoports/qwen_qwen3_8_27b/doc/datatype_sweep/configs/kv_bfp4.json bash models/autoports/qwen_qwen3_8_27b/tests/run_datatype_experiment.sh kv_bfp4_evaluated models.autoports.qwen_qwen3_8_27b.tests.run_datatype_candidate
```

## Triage Evidence

- `kv_bfp4_evaluated.log` last contains `LOAD_LAYER 63`, with no `MODEL_LOADED`.
  It was 8,274 bytes, last modified at 22:51:38 UTC, and remained unchanged
  when inspected after 22:56 UTC. `LOAD_LAYER` precedes the corresponding layer
  constructor, so this marker does not prove layer 63 finished loading.
- Before `LOAD_LAYER 0`, the same log records four rounds of device 1 realtime
  profiler synchronization failure. Each round has three 2,000 ms response
  timeouts; a subsequent 3,000 ms sync check also times out. These are earlier
  device/host communication anomalies, not model arithmetic errors. The
  candidate environment records `TT_METAL_DEVICE_PROFILER=null` and watcher
  variables unset; these messages arise from the runtime's realtime subsystem.
- Coordinator capture under `triage_kv_startup/` ended with exit 124 after its
  180-second bound. Both `tt-triage.txt` and `capture.log` are zero bytes; no
  `triage-summary.txt` was produced. Therefore this run provides **no current
  device PC, running operation, NoC counter, CB state or Ethernet health result**.
- `triage_kv_startup/host_gdb_main.log` is a successful native attachment and
  `bt40`. Main-thread frames 5–13 show:
  `wait_for_outstanding_reads` -> `FDMeshCommandQueue::finish_nolock` ->
  `MeshCommandQueueBase::enqueue_write_shards_nolock` -> `enqueue_write_shards`
  -> `MeshCommandQueue::enqueue_write_tensor` -> `to_device` ->
  `create_distributed_tensor<bfloat16>`.
- `triage_kv_startup/host_gdb.log` independently identifies the busy worker
  TID 599822 as `Cluster::read_from_sysmem` -> `read_cq_host_ptr<true>` ->
  `SystemMemoryManager::completion_queue_wait_front` ->
  `FDMeshCommandQueue::read_completion_queue_event`. This is a completion
  reader, not ongoing Torch packing.
- `triage_kv_startup/host_snapshot.json` at 22:55:14 UTC records PID 599757
  sleeping in `futex_do_wait`, 70 threads, about 5.43 GiB RSS and zero swap.
  Process counters alone do not diagnose a device root cause; the native stacks
  are the discriminating host evidence.
- The coordinator preserved those files before sending SIGTERM to PID 599757.
  Recovery actions and outcomes belong in `triage_kv_startup/recovery_actions.json`
  and the stage AutoFix/work log; no successful recovery is claimed here.

The historical `doc/optimized_full_model/AUTOTRIAGE_head_startup.md` describes
device 1 prefetch waiting in a pinned linear relay. It is a useful hypothesis
source, but its saved device state is from another run and is not reused as a
current stop-site or root-cause proof.

## Source Evidence

### Initialization precedes the KV candidate

`tt/model.py:69` prints `LOAD_LAYER` before `MultichipDecoder.from_state_dict`.
After the layer loop, the constructor uploads embeddings, final norm and head,
converts head weights into DRAM shards, computes the full-context rotary tables,
and uploads those tables before `MODEL_LOADED` at line 126.

`tt/generator.py:706` constructs `QwenModel` before `QwenGenerator`.
Cache allocation is later, through `QwenModel.allocate_cache` at `tt/model.py:140`
and `OptimizedDecoder.allocate_state` at `tt/optimized_decoder.py:239`.
`kv_dtype` is consumed to allocate the full-attention key/value tensors there.

Host-only JSON comparison found that `configs/kv_bfp4.json` differs from
`configs/baseline_bfp4_lofi_head_bfp8_hifi2.json` only in `config_id` and
`kv_cache_dtype`. Run snapshots of `model.py`, `generator.py`, `precision.py`,
`multichip_decoder.py` and `optimized_decoder.py` all matched current files at
inspection. Consequently the measured startup path has no intended KV-dependent
allocation or kernel difference from the passing baseline.

### Upload and completion ledger

| Transition | Producer | Consumer and required condition | Current evidence |
| --- | --- | --- | --- |
| Python input to distributed tensor | `ttnn/core/tensor/py_to_tt_tensor.cpp:230`, `ttnn/core/distributed/distributed_tensor.cpp:747` | Mesh mapper creates host shards, then `to_device` uploads them | Main native frames identify this path. Template argument `bfloat16` is the host element type; it does not identify the final destination dtype or tensor name. |
| Automatic pinning | `tt_metal/impl/tensor/tensor_apis.cpp:42,152` | Total host shard bytes must exceed 32 MiB; memory pinning must be available; at least one shard pin must succeed before the blocking branch | `enqueue_write_tensor` calls `enqueue_write_shards(..., blocking=true)` only in that successful-pin branch. The native stack enters this branch. Whether each device transfer actually used pinned NoC reads is not captured. |
| Shard write submission | `tt_metal/distributed/mesh_command_queue_base.cpp:230` | Per-shard dispatch tasks finish submission; pinned writes invalidate the prefetch cache manager; `blocking` then calls `finish_nolock` | Main thread is past dispatch-thread-pool submission, inside finish. There is no evidence that a taskflow worker failed to submit a shard. |
| Host completion event | `tt_metal/distributed/fd_mesh_command_queue.cpp:715` | Finish enqueues a host event, then waits for outstanding reads to drain | Main thread is in the corresponding wait. |
| Completion reader | `fd_mesh_command_queue.cpp:1014,1116` | Read one event for every local coordinate in its descriptor; after processing queued descriptors decrement `num_outstanding_reads_` and notify the waiter | Worker is inside `completion_queue_wait_front`, before its current event can be consumed. |
| Completion FIFO readiness | `tt_metal/impl/dispatch/system_memory_manager.cpp:756` | Host write-pointer/toggle must differ from the reader's pointer/toggle | Busy worker polls sysmem for this readiness condition. Missing device event, device dispatch stall, wrong observed pointer and preceding command failure are not distinguished by this stack. |

This source already retains pinned memory via transfer objects and uses a
blocking finish for the observed tensor-upload path. It already invalidates
prefetch cache state after actual pinned transfers. Removing that wait or adding
another unconditional synchronization is not a supported repair.

The historical prefetch theory was also checked against current
`tt_metal/impl/dispatch/kernels/cq_prefetch.cpp:1707`: relay linear already uses
separate scratch-half TRIDs 6/7, barriers each half before reuse and checks the
maximum transaction count. Those protections are present and cannot be claimed
as a missing fix. There is no current device capture to prove this kernel is
the present waiter.

### Passing contrasts

The baseline, `head_bfp4_lofi_evaluated`, `ccl_bfp8_evaluated`,
`activation_bfp8_evaluated` and `smoke_kv_bfp4` logs all reach `MODEL_LOADED`
without these realtime synchronization warnings. `smoke_kv_bfp4.json` records
successful reduced-layer S31/S33/S129 generation with BFP4 KV; its repeated
reduced-model tokens are not a full-model quality result. These contrasts support
trying recovery before attributing the startup stall to the candidate dtype.

## Downstream Effects

The main condition-variable waiter and completion reader are two ends of the
same upload completion dependency. They do not prove two independent host bugs.
No current capture establishes a CCL/fabric deadlock, page-table error, matmul
failure, cache numerical failure or teardown defect. The realtime profiler
messages prove missing synchronization responses, but do not by themselves prove
that the profiler caused the later upload stall or that device 1 owns the missing
CQ event.

## Proposed Fix

No implementation change is justified by the current evidence. The failed
device-capture attempt forced source/host-stack inspection as a fallback in this
fresh isolated investigation; it did not supply a device root-cause diagnosis.
Use the following experiments in the coordinator's serialized AutoFix loop:

1. **Recoverable device/transport state.** Complete bounded owned-process
   recovery, list/reset/list and exact Ring 1x4 open/close smoke, then rerun the
   same full-model `kv_bfp4` policy. A successful unchanged run establishes
   recovery of this incident and permits the actual KV sweep to proceed. It does
   not establish a native source fix or retrospectively identify a lost NoC read.
2. **Repeatable upload boundary.** If the same stall recurs, schedule
   `faulthandler.dump_traceback_later(..., repeat=True)` before `build_generator`,
   and mark before/after the remaining layer, embedding, norm, head and rotary
   uploads. `PYTHONFAULTHANDLER=1` alone does not schedule timed tracebacks.
   Identify the actual tensor name, source/destination dtype, logical/physical
   shape and transfer size before altering layout, precision or chunking.
3. **Pinned-transfer versus broader device communication failure.** On a
   recurrence, capture the blocked CQ event coordinate and ID, completion
   pointer/toggle, dispatcher/prefetch stop-site, then the exact pinned host range,
   encoded NoC address, owner lifetime and outstanding transactions. Test an
   exact-shape upload through a controlled non-pinned transfer only once the
   relevant current API path is identified. A failure of both paths would refute
   a pin-only explanation; a differential result would narrow the source boundary
   but still require a lifetime/address/count ledger before a native edit.

Do not reject BFP4 KV or reduce the advertised context to work around this
initialization stall. All kill/reset/device work stays with the coordinator.

## Uncertainty

- Device triage timed out; no current kernel, NoC, CB or Ethernet state is known.
- The host stack identifies the pinned upload branch, but not its Python tensor
  name, actual transfer dtype/size, failing coordinate, event ID or device command.
- `create_distributed_tensor<bfloat16>` cannot distinguish BF16 destination from
  a BF16 source subsequently converted/packed to a lower destination precision.
- Source lifetime/accounting errors and device/transport failure remain competing
  hypotheses. Historical similarity and a successful future retry would not prove
  an exact root cause.
- The report provides no candidate accuracy, traced performance or successful
  recovery claim. Those require the coordinator's current full-model rerun.

Only this Markdown report was authored by the investigator. No implementation
edit, build, device test, reset or process termination was performed here.
