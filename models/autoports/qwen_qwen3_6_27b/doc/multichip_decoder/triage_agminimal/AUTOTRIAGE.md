# AUTOTRIAGE

## Diagnosis

The first observed missing producer is the last device's activation relay, not the matmul weight reader or an Ethernet credit waiter. Device 2 has consumed exactly its own ten K blocks and waits for the first remote block; device 1 has consumed twenty blocks and device 0 thirty. Device 3 has no running worker program in this snapshot. This is a trace/repeated-launch liveness failure: the running op is `AllGatherMinimalMatmulAsyncOp (trace id: 1)`, not the first eager probe.

The strongest model-local hypothesis is unsafe reuse/admission across Linear fused-gather launches: the caller supplies no persistent gather buffer, rotates semaphores shared with ordinary AG, and immediately captures/replays after asynchronous warmups. The native Linear path has **no entry barrier**, despite accepting a `barrier_semaphore`, and resets readiness counters at exit. Source and this single snapshot do not establish whether device 3 never entered this particular replay, finished an earlier replay, or lost a readiness epoch. Accordingly this report identifies the first missing dependency and a discriminating caller workaround, not a hardware-proven root-cause repair.

## Triage Evidence

Evidence: `tt-triage.txt`, `summary.txt`, and `../topology_agminimal_mlp_s512_v2.log`.

- `tt-triage.txt:244–263`: op 48, trace ID 1, executes on physical devices 0, 1, 2; device 3 is idle. The op window contains earlier fused ops 39, 46, 47. Given the current probe's control flow, reaching the fused trace implies its initial eager PCC assertion passed; the exact PCC was not printed, so no numerical value can be recovered.
- Shape: activation `[1,1,512,1280]` BF16; local weight `[1,1,5120,4608]` BFP8. Two links; 8×8 grid; M/K/N blocks 4; subblocks 2×2.
- Every one of eight activation injector cores `(0,y)` on each active device stops in `compute_actual_k_block<true>` / `noc_semaphore_wait_min`, `matmul_dataflow_common.hpp:171`, called from `dm_in0_sender.cpp:433`.
- Each of 64 compute cores per active device has the following CB counts. CB0 is activation, CB1 weight. CB0 blocks have 16 tiles.

| Device | CB0 pushed/popped | Complete K blocks | Forward-ready counter | Next required counter | CB1 occupancy |
|---|---:|---:|---:|---:|---:|
| 0 | 480/480 | 30 | 20 | 21 | 32/32 |
| 1 | 320/320 | 20 | 10 | 11 | 32/32 |
| 2 | 160/160 | 10 | 0 (not emitted in nonzero dump) | 1 | 32/32 |
| 3 | no active-op CB records | unknown | unknown | unknown | unknown |

- Global nonzero forward counters at address `0x17fd80` are 20 on all eight device-0 injectors and 10 on all eight device-1 injectors. There are no nonzero global records on device 2 or 3.
- CB2 output and CB3 intermediate counters are zero on the active cores: they have not completed their first full-K output block.
- Weight BRISC waits at `dm_in1_sender_out.cpp:366` for space in its full CB1; TRISC0 waits at `compute.cpp:488` for CB0. These are backpressure from the missing activation.
- Device 3's prefetch/dispatch cores wait for commands/data (`cq_prefetch.cpp:943`, `cq_common.hpp:664`); it is not observed inside a fused compute kernel or close-mux routine.
- The detailed dump reports `check_noc_status.py: fail` for Ethernet software/hardware accounting snapshots, even though `summary.txt` lists that script as pass. The report also says such differences can reflect live traffic. Binary/core magic and lightweight assertions pass. Do not report this capture as a watcher-clean or fully healthy run.

## Source Evidence

Paths below are relative to the checkout.

1. **Block and route ledger.** In `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/device/kernels/dm_in0_sender.cpp:199–215`, local K is 1280/32=40 tiles, hence ten K blocks per device; global K is forty blocks. For each of the eight M-axis workers, M=16 tiles split over eight rows gives two live M tiles; its padded CB block remains M_block×K_block=16 tiles. All workers execute one M block. Each rank reads ten local blocks then consumes thirty remote blocks. `matmul_dataflow_common.hpp:124–140,160–172` orders slices `(rank+device_iter)%4` and waits for one forward counter increment per remote block.
2. **Forwarding.** `dm_in0_sender.cpp:511–580` sends only the first thirty K iterations. Ranks above zero short-send each block to their predecessor, one hop. Rank zero long-sends to rank three, three hops. The factory's Linear route override at `all_gather_minimal_matmul_async_program_factory.cpp:801–814` already sets this long-send route; it is not an absent-wrap-route bug. Each payload is followed by a forward-ready atomic to the receiving injector. Two mux links group four M workers each. This matches the observed ten-block staircase: device 2 lacks the first device-3 slice; devices 1 and 0 have only the slices that could flow before it.
3. **Not a divisibility failure.** Local K40 is divisible by K block4, and M partial blocks are explicitly supported. The factory requires `ceil(in0_axis/num_workers_per_link)==num_links`; `ceil(8/4)==2` satisfies it. The all-core semaphore coverage repair is already present in the caller's model setup. Neither reducing links nor changing transpose repairs an identified count mismatch.
4. **Launch admission differs by topology.** `dm_in0_sender.cpp:294–322` documents the ownership hazard of a following gather reusing preceding DRAM and implements an entry barrier only for `Topology::Ring`. Linear bypasses it even when the caller passes a barrier semaphore. At lines 772–773 both out-ready counters are reset to zero. The code does not provide a Linear invocation-wide admission fence around scratch reuse.
5. **Scratch ownership.** `all_gather_minimal_matmul_async_device_operation.cpp:523–528` creates a new activation gather intermediate unless `persistent_output_buffer` is provided. The public operation removes that intermediate from returned tensors at lines 689–695. Therefore `values[0]` in the probe is correctly the first **matmul** output, not the gathered activation; do not change it to `values[1]` based on older test comments.
6. **Caller.** `models/autoports/qwen_qwen3_6_27b/tests/probe_multichip_topology.py:61` omits `persistent_output_buffer`. Lines 70–75 enqueue two warmups, delete outputs, and start capture/replay without a mesh synchronization boundary. Lines 79–81 enqueue fifty nonblocking executions with only one final synchronization. No per-phase prints distinguish capture, first replay, and repeated replay in the current log.
7. **Blocking replay itself is not a per-device synchronization bug.** `tt_metal/distributed/fd_mesh_command_queue.cpp:1418–1425,1583–1623` submits trace commands to all local devices before `finish_nolock`; merely changing `blocking=True` to False is not an explained fix.

## Downstream Effects

The earliest observed consumer is device 2's activation injector, waiting for the device-3 relay. All weight-CB reservations, TRISC waits, the corresponding waits on devices 1/0, and eventual host synchronization stalls follow from it. The fabric muxes on active devices are still running, but the evidence does not show a send-slot, credit, or close-connection stop as the first failure. Blaming two-link routing, transpose, or mux teardown without a contrasting experiment would be premature.

## Proposed Fix

Parent owns experiments; no source or device mutations were performed by this reviewer. Apply this bounded caller-only experiment before rejecting the fused path:

1. Allocate and retain a dedicated replicated BF16 TILE DRAM gather buffer `[1,1,512,5120]` before warming. Pass it as `persistent_output_buffer` for every fused call. Keep it alive through trace release. The cost is 5,242,880 bytes/device (5 MiB). Give the fused probe its own two full-grid zero-initialized readiness semaphores, separate from ordinary AG; hold the pair fixed for a one-op trace. Keep existing 8×8 geometry, two links and Linear topology initially so the experiment isolates lifetime/admission.
2. Add phase prints immediately after eager PCC, each synchronized warmup, capture end, first replay, and replay iteration. Synchronize after each warmup, before capture, and after capture. First run repeated replay with `blocking=False` followed by `synchronize_device(mesh)` **each iteration**. This is a diagnostic launch-isolation control; report it as such, not as pure device latency.
3. If that passes, remove only the per-replay host fences while retaining persistent scratch and dedicated semaphores. If it then fails, the remaining issue is cross-launch ordering, not scratch allocation. A model-local traced device fence before the fused call is preferable to a shared-kernel edit: use a tiny existing Linear collective with its own persistent buffers/semaphores and include its cost in the candidate path. Verify its actual entry-barrier semantics first. Alternatively keep the separate AG+matmul path if the required fence makes fusion slower.
4. If synchronized persistent replay still stalls, preserve another snapshot and obtain a host stack/phase marker. Check whether rank 3 was submitted before changing kernels. Only then try one link (`num_workers_per_link=8`) or explicit transposed grid as independent topology controls; a passing variant would be evidence, not a source-proven fix in advance.

A fast unsynchronized trace must pass PCC and sustained replay after any workaround. Reusing the previous log as numerical/performance evidence would be incorrect.

## Uncertainty

- The capture does not contain the host Python/native stack or a replay iteration number. It cannot prove why physical device 3 is idle, nor distinguish an invocation-admission race from a trace submission/lifecycle issue.
- Mesh rank-to-physical-device order is inferred from the source route and exact CB staircase; record it explicitly in the next run.
- No shared kernel fix, hardware reset, TTNN import, device run, or performance claim was made here. Only source inspection and Python standard-library parsing of the existing dump were used.
- Proposed persistent-buffer/fence changes are unverified. This is the source-inspection portion of AutoFix, with concrete experiments required before calling the failure repaired.
