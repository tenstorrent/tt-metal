# AutoDebug: bounded scratch for Qwen3.8-27B TP4 direct all-reduce

Inspection date: 2026-10-23. Scope: decoder and collective/trace implementation only. No device execution, reset, reproduction, source repair, or full-model runtime was performed. This report is the only file written by this investigation.

## Findings

1. **The current scratch ownership cannot scale to 64 layers.** Each decoder retains separate `out` and `down` scratch tensors for each logical batch shape. For the reported BF16 configuration, that is **320 KiB per selected L1 core per layer**, or **20 MiB per selected core for 64 layers**, before activations, program buffers, or other state. The earliest problematic decision is the per-instance, per-logical-shape cache in `_reduce`, not a collective size calculation. This is a source-proven capacity problem; no 64-layer allocation failure was reproduced.
2. **Replacing those allocations with one immediately reused buffer is unsafe.** The selected minimal `all_reduce_async` has no all-rank entry barrier. Its global semaphore signals incoming data readiness and resets before compute finishes reading scratch. A fast rank's next collective can overwrite a slow rank's current reduction input. This is a source-derived permitted execution, not an observed corruption in the supplied runs.
3. **Two persistent alternating slots shared across complete decoders are sufficient under a precise ordering contract.** For the sequence `out → down → out → down`, the intervening collective proves every rank completed the previous use of the slot before anyone can write its next use. The minimal decoder repair therefore need not add a barrier or per-token host synchronization: share two slots and preserve the contract below. Their combined scratch cost is **320 KiB per selected core / 2.5 MiB per device**, independent of layer count and, with canonical physical capacity, batch sizes 1–32 and the number of serialized live traces.
4. **Trace capture does not retain transient scratch allocations.** Captured commands retain addresses; ordinary deallocation returns tensor storage to the allocator. A pool must remain allocated while any referencing trace or queued operation can use it. Allocating a new persistent pool after an older trace was captured can overlap that trace's recorded intermediates. `release_trace` itself is not a completion fence.

The two-slot design is the recommended minimal repair for this decoder. A single-slot design needs a genuine device-side all-rank consumption/entry fence before reuse. Arbitrary transient allocation is not a substitute for either protocol ordering or trace address ownership.

## Evidence and revision boundaries

The checkout HEAD is `ad9ed5d3c46f649ae4a0fa0804e899e7999d8180`. The decoder and runner were already modified, and the optimized-decoder evidence directory was untracked. Findings concern the working tree, not just HEAD.

The decoder changed externally during inspection. Initially its SHA256 was `fdde2c776cb16b82724cdcb9dd1c71d325000221dfb977f7f6d2738856f2d12e`, matching the layer-3 evidence. At report preparation it was `36ab8a8df410e00ed072b26f4438f2f3e8c15af3eabf95f89bd3855d04e0d567`; the inspected change added `_prefill_phase` gating and shifted line numbers. The scratch allocation and direct-AR call remained unchanged. Layer 0 records a different SHA, `a5516e6d80bdb35c0e6dc6d5d9607538ea5229b64ec55fd41a1fc43403ac478c`; its exact historical delta was not established.

The runner initially matched the artifacts' `5cf5c449652844d34d54f854180f2f4c0eee21b52da08780b3e7a972287a0efb`, but changed externally to `39f8f8976aed5c0fc4a09e5b5f2ce1cc6257e1adca958c8661d0140c89ee8dd0` during review, adding `--async-stress` and effective-policy reporting. References below use these later decoder/runner line numbers; observations from the supplied JSONs concern their recorded earlier versions. No execution result for the newly added stress branch is inferred here. The `optimized_decoder.py` hash in both artifacts matches the inspected file.

### Recorded hardware observations, not new measurements

| Artifact | Workload | TP4 median decode host time | Prefill PCC | Minimum of 128 decode PCCs | Final state PCC |
| --- | --- | ---: | ---: | ---: | --- |
| [Layer 0](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/doc/optimized_multichip_decoder/topology_direct_ar_l0.json) | Linear attention; B1; 4096 prefill + 128 decode | 473.347 µs | 0.9999618667 | 0.9999663803 | Conv 0.9999384100; recurrent 0.9998668592 |
| [Layer 3](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/doc/optimized_multichip_decoder/topology_direct_ar_l3.json) | Full attention; B1; 4096 prefill + 128 decode | 391.540 µs | 0.9999755888 | 0.9999912666 | Key 0.9999380142; value 0.9999260777 |

Both use `phase_specific`, `direct_all_reduce=true`, `stack=false`, and report three bitwise restored replays plus replica equality. The logs identify Blackhole: [layer-3 log](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/doc/optimized_multichip_decoder/topology_direct_ar_l3.log:5). The measured times are blocking replay wall times, not isolated collective/device times. PCC compares against the single-chip `OptimizedDecoder`, not directly against HF: [runner](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tests/run_multichip_decoder.py:90), lines 272–282.

The approximately 490/408 µs TP4 comparison is supported by [before_l0.json](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/doc/optimized_multichip_decoder/before_l0.json:309) (489.869 µs) and [before_l3.json](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/doc/optimized_multichip_decoder/before_l3.json:309) (408.252 µs). Those runs also use B1/4096+128, with policy `{}` and decoder SHA `4b8b89ab05452f6c9194049ed6baa155265a3d1d1288d7043f8fe25c731eca90`. The `baseline` objects inside the two direct-AR JSONs instead contain single-chip medians of approximately 794/689 µs. These comparisons must not be conflated; none is a measurement of the proposed shared pool.

The evidence-producing runner warms and reads one eager decode, captures one complete decode, then **synchronizes before each token and executes every replay with `blocking=True`**, followed by output reads. That ordinary path remains at [runner lines 198–221](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tests/run_multichip_decoder.py:198). Thus these observations demonstrate synchronized reuse of existing private scratch. They do not test cross-layer sharing, multiple live traces, an unsynchronized replay stream, or transient scratch reuse. Phase-specific prefill uses sharded residuals and bypasses direct AR; its passing PCC is not direct-AR coverage.

The report's memory telemetry is DRAM-only: [runner lines 96–99](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tests/run_multichip_decoder.py:96). It cannot establish L1 feasibility. At initial inspection, the batch/stack/watcher suites in [validate_optimized_multichip.py](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tests/validate_optimized_multichip.py:30) did not enable `direct_all_reduce`; their existence was not coverage for this candidate. Later external suite edits likewise need their own matching execution evidence.

## Exact allocation and lowering

Each instance creates its own `_ccl_buffers` and `TT_CCL`: [decoder lines 30–31](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tt/multichip_decoder.py:30). The prefill shallow copy shares these objects, rather than adding a second cache. `_linear` reduces only `out` and `down`: [decoder lines 209–216](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tt/multichip_decoder.py:209).

For the reported policy, the lowered path is:

| Item | Concrete value |
| --- | --- |
| Mesh / collective | 1×4, TP axis 1, ring, two links, minimal scratch-taking overload |
| Logical partial | `[B,1,5120]`, BF16; compacted to `[1,1,B,5120]` |
| Input and output placement | L1, width-sharded on logical cores `(0..7,0)`, row-major shard orientation, shard `[32,640]` |
| Scratch | `[1,1,B,20480]`, BF16 tiled, same eight cores, shard `[32,2560]` |
| Cache key | `("ar", role, tuple(x.shape), str(x.dtype))` in each decoder instance |
| Semaphore | `TT_CCL.get_and_cycle_barrier_semaphore_handle()` passed as `multi_device_global_semaphore` |

Source: [decoder lines 158–181](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tt/multichip_decoder.py:158). Direct AR allocates persistent scratch regardless of the separate `persistent_ccl` policy switch. The high-level memory selection earlier in `_reduce` does not keep this scratch in DRAM: `ar_mem` installs the sharded L1 configuration.

The selected [wrapper lines 541–572](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/all_reduce_async.cpp:541) forwards directly to the minimal primitive. It does not take the other overload's reduce-scatter/all-gather composite path. The [primitive lines 255–290](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/all_reduce_async_device_operation.cpp:255) derives ring size 4 from axis 1 and launches the device operation. Validation requires width sharding, output-grid containment in scratch, and at least `output_shard_volume × ring_size` scratch capacity: [validation lines 46–73](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/all_reduce_async_device_operation.cpp:46).

For B=1 through 32, one BF16 tile occupies 2048 bytes. Per core:

- Input/output: `32×640 / 1024 = 20` tiles, 40,960 bytes.
- Scratch: `4×20 = 80` tiles, `32×2560×2 = 163,840` bytes = 160 KiB.
- Two roles: 327,680 bytes = 320 KiB; across eight cores, 2.5 MiB/device; across four devices, 10 MiB.
- Sixty-four separate layer owners: 20 MiB per selected core, 160 MiB/device, for scratch alone. The HF configuration specifies 64 layers: [config](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/doc/functional_decoder/hf_config.json:98). Blackhole exposes only 1536 KiB of L1 per core before reservations: [memory map](/home/mvasiljev/tt-metal/tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h:33).

The logical-batch key adds another avoidable multiplier: B1/B8/B16/B32 retain four equally sized pairs, totaling 1.25 MiB per selected core for even one decoder. Exact failure timing depends on other allocations and fragmentation; no particular failing layer count is claimed.

### Kernel buffer/state ledger

The factory binds reduction CB1 directly to the supplied scratch; output CB2 is a separate output allocation: [factory lines 359–382](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/all_reduce_async_program_factory.cpp:359). This is not a private copy protecting the scratch from later writes.

For two links, each sender handles four output cores, 80 input tiles, with zero starting offset; the two link assignments cover cores 0–3 and 4–7. See [factory lines 176–201, 269–350](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/all_reduce_async_program_factory.cpp:176). Sender cores are selected outside the output grid (lines 217–225). Their staging CBs are additional program workspace, not extra copies of the persistent reduction scratch.

| Phase | Producer/consumer action | What completion establishes |
| --- | --- | --- |
| Sender reader | Reserve 20 tiles per input core, read them, NoC read barrier, push; four cores per link | Input data is available in sender staging CB; [reader lines 49–67](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/kernels/dataflow/worker_reader.cpp:49) |
| Sender writer | Wait/pop corresponding staging tiles; write source-rank block at `scratch + rank×40960 + tile_offset×2048` on each destination | Each source owns a disjoint 20-tile block; final shard packet signals its arrival; [writer lines 113–170](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/kernels/dataflow/worker_writer.cpp:113) |
| Reduction receiver | Wait for count 4, publish 80 scratch tiles with `cb.push_back(80)`, reset global semaphore to 0 | All four contributions are ready to read; **not** that they were consumed; [receiver lines 31–36](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/kernels/dataflow/reduction_receiver.cpp:31) |
| Compute | Wait for 80 tiles; reserve 20 output tiles; reduce rank blocks in pairs, output groups 8, 8, 4; push 20 output tiles | Scratch is read after publication, through the reduction loop; no remote consumed acknowledgment; [compute lines 20–67](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/kernels/compute/reduction.cpp:20) |

The 20/80-tile geometry is consistent; the last compute group is four tiles, not an overrun. The missing guarantee is between invocations, not an identified intra-invocation count mismatch. The scratch CB is scoped to a program; absence of a final input pop does not create the remote reuse fence.

## Cross-device skew: what is unsafe and what is sufficient

### One immediately reused slot: a legal bad schedule

Let invocation q use scratch S. Rank B sends its q data, receives all q contributions, and publishes its scratch CB, but its compute is delayed. Rank A receives all q contributions and finishes its q program. A can then launch q+1 and overwrite A's source block of S on B while B is still reading q. B can consume a mixture of q and q+1. Distinct arrival semaphores do not prevent this data overwrite.

There is no entry wait in [worker_writer.cpp lines 85–164](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/kernels/dataflow/worker_writer.cpp:85). The caller's name `barrier_semaphore_handle` does not change its role: here it is the reduction's arrival counter. The writer reads an `out_ready_sem_wait_value` but never waits on it. Its separate per-program reduction semaphore is set, not used for an all-rank scratch-consumed handshake.

Fabric `open_finish` and `close_finish` coordinate worker/EDM connection state, not remote reduction compute: [adapter open lines 522–563 and close lines 615–632](/home/mvasiljev/tt-metal/tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp:522). Flushing outgoing packets likewise does not prove remote scratch reads finished. Therefore neither Python call order, allocator reuse, a local event, semaphore reset, nor connection closure repairs immediate single-slot reuse.

### Two alternating slots: why an intervening collective is enough

Local completion is stronger than “sender done.” The next program's GO waits for the prior local worker completion count: [ordinary dispatch](/home/mvasiljev/tt-metal/tt_metal/impl/program/dispatch.cpp:3156), [trace dispatch](/home/mvasiljev/tt-metal/tt_metal/impl/program/dispatch.cpp:3348), and [device wait](/home/mvasiljev/tt-metal/tt_metal/impl/dispatch/kernels/cq_dispatch_subordinate.cpp:618). BRISC waits for NCRISC and TRISCs before reporting completion: [firmware lines 325–335, 550, 595–610](/home/mvasiljev/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisc.cc:325). Early kernel setup does not bypass this gate: [BRISC kernel loader](/home/mvasiljev/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisck.cc:75) waits for GO before `kernel_main`. Replay boundaries also wait for previous local kernels: [trace replay dispatch lines 103–146](/home/mvasiljev/tt-metal/tt_metal/impl/trace/dispatch.cpp:103).

Let `Cj(q)` mean whole-program completion of collective q on rank j; `Sj(q)` means j sends q's contribution. With slot `q mod 2`, for **every** source/destination pair i,j:

```text
Cj(q)
  → Sj(q+1)                  [local program order on j]
  → rank i receives j's q+1  [all-rank arrival requirement]
  → Ci(q+1)
  → Si(q+2)                 [local program order on i]
```

Thus every rank has finished reading slot q before any rank starts writing that slot for q+2. Completion of q everywhere also means all its required writes arrived at their destinations. One does not need all ranks to finish q+1 before q+2 starts: q+1 uses the other slot.

This proof requires identical participating ranks, identical collective order, whole-program ordering in the same execution domain, nonaliasing slots, no other scratch users, and distinct arrival counters for adjacent invocations. It is not a general assertion that an asynchronous all-reduce is a global completion barrier.

The decoder satisfies the required role sequence when complete decodes are composed: [decode_forward](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tt/optimized_decoder.py:514) calls one mixer then `_finish`; the linear/full mixers end in `self.out` ([decoder](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tt/multichip_decoder.py:371), line 431), and `_finish` performs one `self.down` ([decoder](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tt/multichip_decoder.py:258), line 286). Complete trace A followed by complete trace B, or repeated complete traces, preserves `out/down` parity.

`TT_CCL` already cycles two arrival handles, although the decoder currently owns a separate instance per layer: [tt_ccl.py lines 75–95](/home/mvasiljev/tt-metal/models/common/modules/tt_ccl.py:75). For an explicit shared pool, pairing a dedicated semaphore with each role slot makes the invariant independent of other calls into the generic CCL semaphore cycle. Trace replay uses captured addresses; a Python counter does not advance during replay.

A sibling implementation uses two shared role slots in [GPT-OSS DecodeAllReduce](/home/mvasiljev/tt-metal/models/demos/gpt_oss/tt/batched_decode.py:36). Its comment was checked against the protocol above rather than accepted as proof or as hardware validation for Qwen.

### If one slot is required: exact additional synchronization

Before **any** q+1 writer touches S on any rank, establish `Cj(q)` for **all** ranks j. A real all-rank entry exchange at q+1 can do this because a rank may advertise entry only after its preceding full program completed. All sender workers/links must be covered; releasing one link without gating the others is insufficient. Keep entry/consumption counters separate from data-arrival counters and make their epoch/reset reuse safe.

A separate device barrier program ordered after q's complete reduction, or an entry handshake in q+1's sender kernels, can provide that guarantee without a host wait. The sibling [all-gather sharded writer lines 105–130](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/llama_shapes_sharded_writer.cpp:105) demonstrates an explicit all-rank handshake before data writes. It is an implementation pattern, not an option secretly enabled in minimal AR. Adding it to minimal AR requires an actual implementation/API change; passing a differently named existing semaphore is not enough.

An end barrier is also sufficient only if each rank announces **completion of all scratch-consuming compute cores**, not merely completion of sender traffic or receiver publication. A host mesh synchronize between calls is a useful diagnostic control but does not meet the requested runtime design.

## Scratch and trace ownership

Normal tensor release returns L1 addresses to the ordinary allocator: [allocator lines 321–353](/home/mvasiljev/tt-metal/tt_metal/impl/allocator/allocator.cpp:321). A globally bound CB stores an address and raw pointer, not an owning tensor reference: [CB configuration lines 232–238](/home/mvasiljev/tt-metal/tt_metal/impl/buffers/circular_buffer_config.cpp:232). `MeshTraceDescriptor`/`MeshTraceBuffer` retain command bytes and trace storage, not a set of scratch tensors: [mesh_trace.hpp lines 29–49](/home/mvasiljev/tt-metal/tt_metal/distributed/mesh_trace.hpp:29). Finalized capture nodes are cleared at [record_end](/home/mvasiljev/tt-metal/tt_metal/distributed/fd_mesh_command_queue.cpp:1652).

The documented contract is explicit: replay reads/writes the captured addresses without allocator operations or running factories again; freed temporaries are invisible to the allocator despite future replay using them. See [TraceCorrectness.md lines 37–62](/home/mvasiljev/tt-metal/tech_reports/AdvancedPerformanceOptimizationsForModels/TraceCorrectness.md:37).

Consequences for the proposed alternatives:

| Design | Ordering requirement | Address/lifetime requirement | Assessment |
| --- | --- | --- | --- |
| Per-call transient scratch, free after AR | Immediate reuse can alias q and q+1 remotely; ordinary local queue order alone fails | Replays keep historical addresses; later allocations may overlap | Not a safe drop-in repair |
| Transient scratch inside a fully planned capture | Explicit entry fence or a proven nonaliasing two-slot schedule | All intermediate overlaps and late persistent allocations must be planned; incompatible concurrent traces need disjoint arenas | Possible in principle; capture alone supplies neither proof |
| Persistent one-slot mesh pool | All-rank consumption/entry fence before every immediate reuse | Allocate before traces; retain until retirement | Bounded and supportable with added protocol work |
| Persistent two-role mesh pool | Complete ordered `out/down` sequences, phase-separated semaphores | Allocate before traces; retain until retirement | Smallest repair for this decoder |

The allocation tracker is useful but not a reservation mechanism. A trace registers at [capture end](/home/mvasiljev/tt-metal/tt_metal/distributed/mesh_device.cpp:1655); later allocations are tracked by ID and deallocations remove them: [tracker lines 183–215](/home/mvasiljev/tt-metal/tt_metal/impl/allocator/trace_allocation_tracker.cpp:183). It does not prove remote-consumer ordering or detect every overlap between concurrent captured temporaries. The multi-trace limitations are documented in [TraceCorrectness.md lines 64–95](/home/mvasiljev/tt-metal/tech_reports/AdvancedPerformanceOptimizationsForModels/TraceCorrectness.md:64).

The runner's `trace_region_size=0` enables dynamic DRAM storage for trace commands; its high-water-mark checks protect that command storage, not L1 scratch: [mesh_trace.cpp](/home/mvasiljev/tt-metal/tt_metal/distributed/mesh_trace.cpp:65). Increasing the trace region cannot solve this L1 ownership problem.

## Minimal repair specification

No repair was applied. The following is a concrete implementation contract for a subsequent decoder-only change:

1. **Create one explicit pool owner per ordered TP4 execution domain**, passed/shared by sequential decoder instances and all compatible trace owners. Keep it separate from the generic `_ccl_buffers` cache, which also stores AG/RS buffers. Scope it to the actual mesh lifetime, physical participants/axis, subdevice/queue ordering, input dtype/tile format, L1 shard grid/orientation/capacity and compatible collective configuration. A process-global key based only on a recycled numeric mesh ID is insufficient lifetime ownership.
2. **Preallocate two role slots and two dedicated arrival semaphores before the first trace.** Use `out → slot 0`, `down → slot 1`. Each BF16 scratch can have canonical logical shape `[1,1,32,20480]` and shard `[32,2560]`. The minimal op validates grid/capacity and uses scratch's address; its loops/output shape derive from input/output, not scratch's logical B. See [validation/output spec](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/all_reduce_async_device_operation.cpp:38) and [factory binding](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/all_reduce_async_program_factory.cpp:359). Consequently B1/3/8/16/32 can share the same pair without a scratch view or allocation per B. Do not include layer identity or logical B in this physical-capacity key.
3. **Make `_reduce` borrow a slot, not allocate/zero/cache it per layer.** Pass the borrowed scratch and paired semaphore to the existing minimal overload. All 80 scratch tiles are overwritten before readiness, so scratch contents need not survive between uses; backing addresses must. Setup can use `empty` instead of repeated zeros. Preserve separate output tensors and other decoder math. Runtime cache-hit address rebinding already exists: [factory lines 618–633](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/all_reduce_async_program_factory.cpp:618).
4. **Enforce balanced complete-decoder use.** Reject unsupported standalone/partial reuse or give it a separate owner/barrier. An `out`-only trace replayed twice, or a role sequence `out → out`, breaks the fixed two-role proof. Every participating rank must follow the same schedule. New dtype/layout families require capacity admission; an unbounded dictionary of signatures is not a bounded pool. The stated B1–32 BF16 family needs only one pair.
5. **Prepare all persistent state and warm all trace variants before capture.** Allocate the shared pool, semaphores, inputs/state, retained outputs and persistent helper/program-cache buffers up front. Creating B's retained tensors after capturing A can overlap A's temporaries even when scratch itself is correctly pooled. A captured copy into a preallocated output can avoid this: release the temporary returned output and all its aliases after enqueueing that copy, retaining only the preallocated destination. Keeping the temporary alive would preserve the late-allocation hazard. Sharing scratch does not solve every other tensor's trace lifetime.
6. **Serialize compatible replays without per-token host synchronization.** Same-CQ complete traces may be enqueued with `blocking=False`. Multiple live traces can share this pair because liveness is not simultaneous execution. Truly concurrent subdevices/streams need disjoint pools or explicit serialization of the entire relevant work. Cross-CQ event dependencies can preserve local ordering, but are not all-rank consumption barriers: [CQ ownership](/home/mvasiljev/tt-metal/tt_metal/impl/dispatch/cq_shared_state.cpp:10), [event waits](/home/mvasiljev/tt-metal/tt_metal/distributed/fd_mesh_command_queue.cpp:1012). Event waits cannot simply be inserted inside captured traces on this path.
7. **Retire in-flight uses before releasing ownership.** Stop submissions, wait once for affected queued work at teardown, release referencing traces, then drop pool/semaphore ownership. Releasing only one of several live traces must not free the pool. [release_mesh_trace lines 1589–1607](/home/mvasiljev/tt-metal/tt_metal/distributed/mesh_device.cpp:1589) unregisters/erases trace state without finishing queued work. Captured bindings cannot be repaired by later replacing the Python pool object.

This leaves two scratch allocations rather than 128 for 64 layer owners. It introduces no new collective or per-token host wait under the balanced schedule. Preserving the reported latency is a prediction to benchmark, not an established result.

## Discriminating validation, still decoder-only

These are proposed follow-ups, not tests run by this investigation. Do not substitute a full-model run.

1. **Ownership/capacity check.** Bind many lightweight layer owners to the pool without loading a 64-layer model. Check that all `out` bindings share one allocation and all `down` bindings another. For B1/3/8/16/32 and multiple retained traces, record backing addresses and L1 usage, not only DRAM. Expected persistent scratch growth is one 2.5 MiB/device pair, plus small semaphore overhead; subsequent compatible layers/batches/traces add no scratch. Account separately for activation, output and program workspace.
2. **Controlled skew at the decisive boundary.** In a minimal TP4 all-reduce test at the exact Qwen geometry, use changing, rank/epoch-tagged values with exactly representable BF16 sums. Delay one rank's q reduction compute after arrivals, while allowing its sender to finish. Delaying that rank before it sends can hide the race by preventing fast ranks from advancing. Compare one slot, one slot plus a real entry fence, and two alternating slots with separate semaphores. Inspect every retained epoch/output shard, not only the final tensor. Predict corruption for the deliberately skewed one-slot control; expect both repaired protocols to preserve every value. This separates scratch lifetime from numerical tolerance and semaphore aliasing.
3. **Direct-AR decoder accuracy and composition.** Explicitly enable `direct_all_reduce`; rerun current-revision layer 0 and layer 3 at B1/3/8/16/32, with 4096+128 and relevant boundary/short sequences. Compare private-buffer control and pooled candidate on identical inputs/state. Check all replicas, per-step output PCC and final conv/recurrent/K/V state. Exercise both two-layer stack orders, 0→3 and 3→0. Existing suites need the policy enabled to count.
4. **Remove the synchronization that masks reuse.** Capture several complete dependent decoder passes, or queue many complete trace replays without per-token host reads/synchronizes. Preserve per-step results in distinct preallocated device history slots, using ordered copies between replays or a supported device-indexed write, and inspect after one final completion wait. A fixed history address inside a repeated trace would overwrite earlier steps. Include repeated same trace and A→B→A→B across different batch trace variants, all prepared in one mesh lifetime. Verify changing inputs, state restoration, exact restored replay, and both outputs/state against the private-buffer control.
5. **Trace lifetime contrast.** Prepare persistent resources before all captures, keep A and B live, release one only after its work completes, and verify the survivor still works with the same pool. Run with `TT_METAL_TRACE_ALLOC_TRACKING=1` and program-cache tracking enabled. For a proposed transient alternative, deliberately force freed scratch addresses to be reused and include a late persistent allocation/canary control. Confirm the checker catches the late-live-allocation case; do not treat a clean checker result as proof of the skew protocol.
6. **Contract rejection and performance.** Ensure out-only/odd-phase traces and unordered overlapping streams cannot silently borrow the fixed role pool. Benchmark the accepted two-slot path and, if implemented, the one-slot barrier path using the same host/device timing definitions as the control. Recheck L1 headroom under each supported batch. Numerical agreement alone does not validate resource bounds or preservation of the 473/392 µs performance observation.

The existing generic [all-reduce test](/home/mvasiljev/tt-metal/tests/ttnn/unit_tests/operations/ccl/test_new_all_reduce.py:100) uses eight persistent scratch/semaphore pairs (allocation at lines 155–167, cycling at 192–197); its eager path synchronizes each call. It is useful coverage of the primitive, but not evidence for one-slot reuse, this two-slot Qwen pool, or arbitrary transient lifetime.

## Other potential issues and adjudicated alternatives

- **Arrival semaphore namespace:** the decoder supplies `cluster_axis=1` but calls the getter without an axis, selecting the no-axis namespace. This is a naming/ownership concern, not an established defect: the resulting global semaphore still covers the cores and is passed explicitly. Adjacent complete decode calls alternate handles. A dedicated pair avoids coupling to unrelated uses.
- **Fabric readiness ordering:** remote fused payload write and arrival atomic use the same destination core/ordered NoC path: [packet handler](/home/mvasiljev/tt-metal/tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp:214), [same-core contract in the fabric tests](/home/mvasiljev/tt-metal/tests/tt_metal/tt_fabric/test_infra/kernels/tt_fabric_test_kernels_utils.hpp:1267). Source flushing is not destination completion, but that fact alone does not prove the current data-ready protocol broken. No independent write-before-ready defect was established. The two-slot proof assumes the primitive's existing readiness semantics are correct.
- **Geometry/dtype variants:** the examined B1–32 BF16 geometry passes the capacity relation. No scratch indexing mismatch explains the reported passing runs. Other dtypes, unequal grids, larger batches or participant groups need their own capacity/protocol validation; they are not silently covered by the canonical BF16 pool.
- **Passing PCC is not evidence against the one-slot hazard:** the recorded runs have two distinct private role buffers and host synchronization between tokens. The bad schedule needs a changed ownership/aliasing condition absent from those observations. Conversely, no current failing decoder output is alleged by the evidence.
- **One slot plus an actual entry barrier is valid:** the concern that an entry barrier inherently cannot protect the preceding reduction's reads is false when entry is ordered after full local program completion. The missing barrier in this minimal path, rather than the mere label “async,” is the relevant fact.
- **Two slots are not inherently unsafe because ranks skew:** the intervening collective proves the necessary completion relation. The narrower single-slot counterexample must not be generalized to a correctly phased pair.
- **Transient tensors are not universally forbidden in traces:** ordinary planned intermediates are supported. What is unsupported by this proposal is assuming deallocation/capture automatically reserves a remote scratch lifetime or isolates unrelated live traces.

## Report review and confidence

The first draft was rechecked against the actual overload, validation, factory, sender/receiver/compute kernels, local dispatch/firmware completion, allocator, and trace ownership. Independent xhigh reviews agreed on the single-slot counterexample, conditional two-slot proof, and absence of automatic trace retention. Review corrected the architecture capacity reference to Blackhole, located the exact earlier TP4 timing artifacts, updated the externally changing runner provenance, and clarified temporary-output release and history-buffer indexing. The scratch capacity calculation was independently recomputed.

A cheap host-only dependency-DAG check instantiated four ranks and three collectives: prior q completion precedes q+1 sends for only the four same-rank pairs, while it precedes q+2 sends for all 16 rank pairs under the all-rank-arrival and local-program-order assumptions. This checks the stated proof's logic, not hardware timing. JSON parsing and SHA256 checks verified the recorded metric/provenance distinctions above.

**Confidence:** high in the capacity arithmetic and source-level ordering/lifetime contracts; the proposed pool's performance and behavior under deliberately skewed, nonblocking multi-trace execution remain unverified on hardware. The supplied observations are compatible with these findings and do not establish the proposed repair's acceptance tests.
