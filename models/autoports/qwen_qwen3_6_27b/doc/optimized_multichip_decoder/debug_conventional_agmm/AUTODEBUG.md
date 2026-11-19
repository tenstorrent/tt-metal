# AutoDebug: conventional AGMM on TP4 Linear Qwen decode

Inspection date: 2026-11-13. Checkout HEAD: `8502f42bca20dcad1507fbc8cebf7032220b7d8e`.
Scope: `ttnn.experimental.all_gather_matmul_async`, not the minimal or Llama AGMM operations.
No target commands, TTNN imports, device opens, hardware reproductions, or source edits were performed. Only this report was written. References below are repository-relative paths and current source line numbers.

## Findings and verdict

**Conventional AGMM can plausibly be adapted to this exact TP4 Linear contract without changing the transport to Ring. It is not safe to enable it by removing the guard.** The smallest defensible repair is a Linear-specific full-slice receiver shared by the interleaved A and B readers, plus an explicit cross-rank admission boundary for persistent scratch. Preserve the existing transport's complete-slice publication, worker barriers, and write/atomic completion rules. Initially admit only the interleaved MULTI configurations described below. Correctness, deadlock freedom on silicon, and performance remain unverified.

The independently confirmed blockers are:

1. **Present failure: deliberate host rejection.** `all_gather_async/device/all_gather_async_default_program_factory.cpp:219-220` rejects Linear with a fused signaler. The retained `ag_mcast1d_l0_b32.log:20-46` records FABRIC_1D, successful baseline replay, then precisely this assertion during candidate eager setup. No conventional candidate latency or output was obtained.
2. **Latent TP4 receiver size contradiction.** `all_gather_matmul_async/device/all_gather_matmul_async_program_factory.cpp:54` hardcodes four transfers. `operations/ccl/kernel_common/worker_sync_utils.hpp:194-220` interprets that as eight slices. With local K1280 = 40 tiles and Kblock4, it asserts `8 * 10 == 40`, which is false. Removing the host guard exposes this contradiction in both 1D and 2D kernels. This is a device `ASSERT`, not a guaranteed ordinary host validation failure; builds without effective assertions must not be assumed safe.
3. **Transfer count alone cannot repair Linear.** The existing receiver alternates directions and wraps rank offsets (`worker_sync_utils.hpp:229-258`). Changing four transfers to two gives the correct slice count but asks ranks 0, 1, and 3 to wait for traffic their Linear producers never publish. Rank 2 happens to match the four-step schedule; this does not make the collective safe.
4. **Repeated persistent launches need generation ownership.** The conventional candidate passes persistent gather scratch, but the factory disables its supplied startup barrier (`all_gather_async_default_program_factory.cpp:632-641`). A faster rank can begin the next launch's remote writes while another rank's previous matmul still reads that same scratch. Ready-credit preservation does not preserve tensor generations. This is a source-supported hazard of bypassing the guard, not an observed conventional runtime failure.

The existing source audit gets the size problem and proposed MULTI wait table right. It needs two corrections: the endpoint self-write semaphore is initialized on every launch, so an unused rank-3 credit is not a demonstrated accumulation bug; and `align_to_slice_and_sync` is a separate sharded route, not the selected interleaved 1D consumer. The initial repair can reject that route instead of changing it.

## Observations, evidence limits, and actual lowering

The retained JSON artifacts contain these completed synthetic MLP-boundary measurements (milliseconds):

| Family, layer 0 / B32 | Baseline | Candidate | Recorded result |
| --- | ---: | ---: | --- |
| separate distributed norm | 0.264523 | 0.305686 | pass |
| minimal fused AGMM | 0.264881 | 0.533441 | pass |
| minimal fused AGMM/SwiGLU | 0.264407 | 0.536692 | pass |
| conventional mcast1d | 0.264560 | unavailable | baseline only; host rejection |

These are parsed from `doc/optimized_multichip_decoder/ag_{separate,minimal,swiglu,mcast1d}_l0_b32.json`. The conventional JSON remains `status="running", pass=false`; the log supplies the termination reason. Successful cases record fixed-input eager/first/last equality after 50 queued replays. They neither measure conventional AGMM nor prove changed-input or adversarial rank-skew safety. Candidate SHA256 `01de0c6d43e5f6305811945e86ee3de2419c1bfba8928e9a04d13f66cd6c5d85` matches the inspected candidate. The measured decoder hash differs from the currently edited decoder: retain these as historical measurements, not a fresh comparison against every current decoder change.

The actual Qwen path is:

`attention RS -> hidden-sharded residual add -> distributed RMSNorm -> local normalized A -> fused AG + packed gate/up -> SwiGLU -> local down -> RS -> hidden-sharded residual add`.

`tt/fused_ag_candidate.py:132-200` retains the hidden-sharded residual boundary. Its conventional branch at 172-188 uses tiled interleaved DRAM local A `[1,1,M,1280]`, persistent gather scratch `[1,1,M,5120]`, and setup-converted contiguous gate/up B `[1,1,5120,8704]`. M is logical 1 or 32 and physically 32 rows, not an outer batch of 32 independent matrices. Weights retain each rank's local N shard; the K extent on each rank remains the full 5120. Gate/up are contiguous halves here; the paired layout used by minimal fused SwiGLU is a different input contract.

The Python API has no cluster-axis argument. `all_gather_matmul_async.cpp:34-59` canonicalizes topology, and `all_gather_matmul_async_device_operation.cpp:204-225` constructs whole-tensor-group AG attributes with `cluster_axis=nullopt`. On the asserted 1x4 full tensor placement, this is the intended TP group. Rank is the index in the tensor's coordinate list (`operations/ccl/ccl_common.cpp:268-303`), and neighbor selection uses that same list (`:355-368`). The schedule must use these ranks, not PCI IDs or an independently assumed physical chip ordering. Do not generalize this repair to arbitrary 2D mesh axes.

The operation creates matmul attributes against the gathered A (`all_gather_matmul_async_device_operation.cpp:227-256`). There is no implicit operand exchange in this inspected lowering. The factory builds matmul first to learn receiver cores/semaphores, then AG, and merges them (`all_gather_matmul_async_program_factory.cpp:73-151`). It selects **CCL descriptor builders** in `operations/experimental/matmul/ccl_fusion/device/ccl_fusion_mcast_{1d,2d}.cpp`, which bind legacy shared matmul dataflow kernels. Editing only the normal matmul Metal 2.0 factories will not repair this path.

Source references beginning `all_gather_*` below are under `ttnn/cpp/ttnn/operations/experimental/ccl/`; `worker_sync_utils.hpp` is under `ttnn/cpp/ttnn/operations/ccl/kernel_common/`, outside that experimental directory.

## Concrete producer ledger: ranks, readiness, and forwarding

For P=4 and local rank r, Linear forward distance F=`3-r`, backward distance L=`r` (`ccl_common.cpp:1260-1267`). Direction names refer to kernel direction values, not an inferred clockwise physical route:

- Reader dir0 receives `r-1, r-2, ...`; reader dir1 receives `r+1, r+2, ...` (`minimal_default_reader.cpp:163-170,222-229`).
- Only dir1 writers write the local slice into local gather scratch (`minimal_default_writer.cpp:406-429`). After all writer workers finish, they publish one local credit on fused semaphore 1 (`:496-504`).
- Each reader direction publishes once per complete received slice (`minimal_default_reader.cpp:402-410`); the factory maps it to the same-numbered fused semaphore (`all_gather_async_default_program_factory.cpp:611-624`).

The following counts are **fused matmul readiness counters**, not the caller's packet-ready GlobalSemaphores:

| Rank | (F,L) | dir0 receives / forwards | dir1 receives / forwards | Final (fused sem0, fused sem1), including local |
| --- | --- | --- | --- | --- |
| 0 | (3,0) | 0 / 0 | 3 / 0 | (0,4) |
| 1 | (2,1) | 1 / 1 | 2 / 2 | (1,3) |
| 2 | (1,2) | 2 / 2 | 1 / 1 | (2,2) |
| 3 | (0,3) | 3 / 0 | 0 / 0 | (3,1) |

Interior workers forward received slices whenever the opposite outgoing neighbor exists; endpoints still receive and publish slices even when no forwarding is needed (`minimal_default_reader.cpp:240-255,370-410`; writer `:507-513`). Do not remove endpoint readers or local dir1 writers merely because their outgoing connection is absent.

The exact legal **consumer order** is the audit's table. Each entry is `(source_rank, fused_semaphore_direction, minimum_counter_value)`:

| Local rank | Slice 0 | Slice 1 | Slice 2 | Slice 3 |
| --- | --- | --- | --- | --- |
| 0 | (0,1,1) | (1,1,2) | (2,1,3) | (3,1,4) |
| 1 | (1,1,1) | (0,0,1) | (2,1,2) | (3,1,3) |
| 2 | (2,1,1) | (1,0,1) | (3,1,2) | (0,0,2) |
| 3 | (3,1,1) | (2,0,1) | (1,0,2) | (0,0,3) |

Generate it as local first, then increasing distance, lower rank before higher rank, skipping out-of-range ranks. Lower source at distance d waits on sem0 >= d; higher source waits on sem1 >= d+1. The local source waits on sem1 >= 1. Both directions can publish independently; dir0 may publish before local sem1. The table is not a claim that transport publishes in this total order.

The first dir1 remote publication waits for the local-write handshake (`minimal_default_reader.cpp:405-408`). Thus sem1's cumulative prefix cannot authorize remote work before local publication has been issued. All local worker slices have completed local `async_write_barrier()` before the writer group signal (`minimal_default_writer.cpp:426-429,496-500`). `OpSignaler::synchronize_workers_and_signal_op` gathers every worker before publishing a slice (`worker_sync_utils.hpp:67-112`). Keep this barrier across all links/workers, including zero-work participants if admitted.

Remote readiness follows flushed payload writes: the writer configures its ready atomic with the Fabric flush contract (`minimal_default_writer.cpp:364-370`), and readers wait for the whole assigned slice before signaling matmul. The final reader and writer write/atomic barriers (`minimal_default_reader.cpp:420-425`, writer `:702-703`) are part of completion. Do not replace acknowledged completion with a mere flush when adapting the path. Actual Fabric visibility and watcher behavior still require hardware validation.

For contrast, a four-step legacy Ring receiver with `num_transfers=2` consumes `(r,1,1), (r-1 mod4,0,1), (r+1 mod4,1,2), (r-2 mod4,0,2)`. On Linear, rank0 first blocks at nonexistent sem0>=1; rank1 later needs sem0>=2 although its maximum is 1; rank3 needs sem1>=2 although its maximum is 1. Rank2's four entries happen to be valid. Leaving `num_transfers=4` additionally fails the size assertion on every rank.

## Exact A/B address and compute schedule

Use ordinary 32x32 tiles. Full Kt=160, local Kt=40, Nt=272, Kblock=4. There are 10 blocks per source slice, four slices, and 40 total K blocks. Each source contributes all 40 tiles exactly once.

`InterleavedRingAllGatherTensorSlicer` derives A's slice stride as 40 tile IDs (`ccl_common.hpp:458-486`); the fused factory derives B's reduction-slice stride as `40 * 272 = 10880` tile IDs (`all_gather_matmul_async_program_factory.cpp:55,69-70`). These are page/tile coordinates, **not bytes** and not per-core N widths.

| Receiver rank | A source-slice bases, in processing order | B source-slice bases, in processing order |
| --- | --- | --- |
| 0 | 0, 40, 80, 120 | 0, 10880, 21760, 32640 |
| 1 | 40, 0, 80, 120 | 10880, 0, 21760, 32640 |
| 2 | 80, 40, 120, 0 | 21760, 10880, 32640, 0 |
| 3 | 120, 80, 40, 0 | 32640, 21760, 10880, 0 |

For scheduled source s, block-within-slice j=0..9, A row h, K position k=0..3, and B output-column position n:

```text
A page_id = A_outer_base + 40*s + 4*j + 160*h + k
B page_id = B_outer_and_core_column_base + 10880*s + 1088*j + 272*k + n
```

For the selected one-block decode cases, A's outer base is zero. B's core-column base is `34*x` for 2D core `(x,0)`, or `9*i` for 1D work-core index i=0..30. It must remain in the address after the source offset is applied. Each A block contains 4 tiles for Mtile1; each B block contains `4*per_core_N` tiles, with existing N-tail handling on the last 1D core. The source offset jumps only at block 0,10,20,30; the existing within-slice block increments then advance A by4 and B by1088.

Both readers must use the **same source at the same block ordinal**. Reordering only A multiplies it against a different B K slice; reordering both preserves the contraction. B is resident and can follow this schedule without waiting on AG. A waits before its slice is read. CB backpressure may limit B's lead; it does not require adding an AG wait to B.

For example, rank3 processes blocks0..9 from source3, 10..19 from source2, 20..29 from source1, and 30..39 from source0, waiting on sem1>=1, then sem0>=1/2/3. Rank0 uses source0/1/2/3 with sem1 thresholds1/2/3/4. This accounts for all160 K tiles without padding the public hidden width.

## Distinguish the 1D and 2D consumers

| Selected path | A delivery and readiness | B delivery and scheduling | Geometry |
| --- | --- | --- | --- |
| CCL 2D multicast | Interleaved A sender uses `update_current_block_start_tile_id`, waits; A multicasts across its row | B sender/writers use the same method without waiting | `(8,1)`, M1/N34, Kblock4, subblock1x2; eight work cores, exact N272 |
| CCL 1D `mcast_in0=true` | One interleaved A sender uses that same method and waits; other A cores receive multicast | Every work core reads its own B block, same source order without waiting | `(8,4)`, M1/N9, Kblock4, subblock1x1; 31 work cores, N tail |
| Block/width-sharded A | Uses `align_to_slice_and_sync` and rotating source ownership | Different ownership/readiness contract | Exclude from initial Linear support |

The selected A/B kernel sites are `operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp:106-113,233-242` and `reader_bmm_tile_layout_in1_sender_writer_padding.cpp:147-155,346-361`. The CCL 1D builder appends A/B receiver args at `ccl_fusion_mcast_1d.cpp:954-955,1022-1023`; 2D at `ccl_fusion_mcast_2d.cpp:1217-1218,1292-1294`. Preserve `use_in1_offset=false/true` respectively.

`align_to_slice_and_sync` is called by `reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp:66-98`. In sharded 1D, `ccl_fusion_mcast_1d.cpp:509-516` selects SINGLE signaling: a source signals only its selected sender core, and the helper waits for1 on that core. The cumulative MULTI thresholds above cannot be applied to each SINGLE core. Furthermore the 2D block-sharded branch does not initialize the fused signaler or append A's fused descriptor (`ccl_fusion_mcast_2d.cpp:744-784,1193-1218`). It must not become accidentally reachable under an interleaved-only Linear exception. Supporting sharded Linear later needs a separate owner-to-source mapping and both helper contracts audited together.

The candidate's 1D N272/N9 partition has 31 work cores; `(6,3)` holds two valid N tiles, while multicast reaches the 8x4 bounding rectangle including dummy `(7,3)`. The multicast helper distinguishes physical destinations from handshake participants, so a missing dummy ACK is not an established deadlock (`ccl_fusion_mcast_1d.cpp:240-256,1054-1061`; `ttnn/cpp/ttnn/kernel_lib/mcast/host/mcast_impl.cpp:228-246`). Nevertheless reserve the whole rectangle from AG/other work because dummy L1 receives data/readiness writes. Current AG offsets `(0,5)` for1D and `(0,2)` for2D are separated from compute (`fused_ag_candidate.py:179`). Descriptor merge checks kernel-core overlap (`tt_metal/impl/program/program_descriptors.cpp:88-100`), which is not by itself a check of every physical multicast destination.

## Minimal repair contract and ABI changes

No public Python layout flag or new Ring fabric is necessary. The following is one concrete, bounded implementation specification; it has not been implemented here.

1. Add a host schedule kind to `MatmulFusedOpSignaler`, defaulting to the existing Ring behavior, and carry the matching capability into `AllGatherFusedOpSignaler`. Select `LINEAR_FULL_SLICE_V1` only after validation. Propagate it to both CCL descriptor builders and both shared interleaved dataflow readers, for example through a Linear-only kernel define. Gate the current Linear assertion on this explicit capability; unrelated fused signalers keep their current rejection. Do not key kernel behavior on mesh size alone.
2. Keep the two existing producer readiness semaphores and their publication ABI. Linear needs no new producer ordering or source-to-destination transport. For a minimal receiver ABI, retain the existing nine-word footprint from `ccl_op_fusion.cpp:676-692`: under the explicit Linear mode, word0 is **actual full-slice count P**, derived from `ag.ring_size`, not a literal4 or per-direction transfers; retain group size, rank, local slice width, operand-specific slice stride, legacy last-stride/direction slots (ignored by Linear), and sem0/sem1. The new parser consumes all nine words for both waiting A and nonwaiting B. Ring serialization/parsing remains unchanged. An explicit four-entry tuple table would also encode this schedule, but is unnecessary: P and rank already derive every entry. Document the changed word0 semantics so a legacy parser cannot consume a Linear record.
3. In Linear mode derive the source schedule above without wrapping. At each slice boundary choose that source, set the current tile ID to the supplied tensor base plus `source * operand_slice_stride`, and wait on that entry's threshold only for A. Require `local_Kt % Kblock == 0`, `num_blocks_per_slice > 0`, `P * num_blocks_per_slice == total_K_blocks`, exactly P unique source ranks, and thresholds bounded by `(r,P-r)`. Finish after P slices. Do not merely retain the old alternating `curr_dir` state with a shorter loop.
4. Initially admit TP4, Linear on FABRIC_1D, dim3 width gather, full four-rank 1x4 placement, 32x32 tiles, nontransposed A/B, interleaved DRAM A/scratch/B/output, outer batch product1, MULTI signaling, and one output h/w block per core (`out_block_h=per_core_M`, `out_block_w=per_core_N`). Require explicit supported 1D mcast-in0 or 2D configs, disjoint physical worker regions, and the admission contract below. This includes both candidate B1/B32 geometries. Keep `gather_in0`, 1D mcast-in1, sharded A/B, extra outer loops, and other fabric/group/topology variants rejected with precise messages. FABRIC2D uses a different neighbor-barrier target and is outside this initial proof (`all_gather_async_default_program_factory.cpp:197-208,495-496`).
5. Make the single-sweep restriction explicit in host validation. Receiver state currently survives across `b/bh/bw` loops; A's update uses `in0_tensor_start_tile_id`, and B's uses `in1_batch_tile_id`, dropping later h/w-block offsets (`reader_bmm_tile_layout_in0_sender_padding.cpp:232-242`, `reader_bmm_tile_layout_in1_sender_writer_padding.cpp:334-361`). Candidate defaults make those loops singleton. If resource tuning introduces multiple output blocks, use stateless slice indexing/reset per sweep **and** the correct outer base before claiming support. A one-line schedule rewind alone is insufficient.
6. Preserve tensor binding and cache-hit refresh in `all_gather_matmul_async_program_factory.cpp:182-197`: all A/B/scratch/output addresses and caller global semaphore/barrier addresses must rebind. Preserve AG reader/writer as descriptor kernels0/1. Topology is already hashed (`all_gather_matmul_async_device_operation.cpp:156-170`); include any independently selectable new schedule/admission attribute in the hash, and ensure the kernel variant records the ABI mode. Infer the fixed initial mode from topology if no user option is needed.

This confines changes to the conventional fused factory/validation, signaler host metadata/serialization, the CCL 1D/2D descriptor bindings, shared receiver/interleaved kernels, and the existing AG startup admission logic. Compute math and Linear forwarding need no redesign.

## Admission, semaphore lifetime, and deadlock hazards

**Persistent scratch requires a barrier before writes, not just ready credits after writes.** The candidate supplies `barrier_semaphore`, but `barrier_semaphore.has_value() && !using_persistent_buffers` disables it (`all_gather_async_default_program_factory.cpp:635`). Unlike minimal, the conventional Python branch never calls `_admission_fence()` (`fused_ag_candidate.py:150-151,172-188`). A local merged program completes only after all its matmul, AG writers, and mux kernels finish; that does not imply another rank's program has finished. Completion of the previous matmul on the target rank is the condition that permits reuse of its gather scratch.

For the initial native Linear implementation, require the supplied global barrier, enable the existing full-range FABRIC_1D writer admission even with persistent scratch, and place every scratch write after it. Every rank enters its next admission only after its preceding merged program has completed on the same ordered CQ/subdevice context. Arrival therefore proves previous local matmul reads are finished. This also protects against early writes into a fresh next-call output allocation before its owner is ready. Retain one owner and one CQ; a concurrent queue needs a separate ownership protocol.

**Do not enable that barrier unchanged.** It currently waits for `barrier_target_count` then does `set(0)` (`minimal_default_writer.cpp:258-307`). A fast rank can finish its full program and send a future admission credit while a slow worker has passed its old wait but not yet cleared its semaphore. The clear erases the future credit and can deadlock a later launch. Consume this epoch by atomic subtraction of exactly its expected count and flush/acknowledge the atomic before exit; for this full-range TP4 Linear barrier the target is3 per participating worker. Preserve future credits, as the packet-ready reader already does (`minimal_default_reader.cpp:414-425`). A software fence/admission collective is a conservative alternative for the first hardware control, but its cost must be included and its own repeated-launch contract verified.

**Keep the three semaphore lifetimes distinct:**

| State | Producer / consumer | Launch lifetime |
| --- | --- | --- |
| Caller global packet-ready sem0/sem1 | neighbor Fabric writes / AG readers | survives cached/queued calls; subtract consumed packet credits |
| Caller global startup barrier | peer admission writers / local AG writers | survives calls; preserve future admission credits |
| Program-local fused sem0/sem1, worker barriers, self-write handshake | local AG kernels / local matmul or AG kernels | initialized from descriptor on every program launch |

Program-local initialization is encoded in `tt_metal/impl/program/dispatch.cpp:1276-1304,2442-2468` and the config command sequence is processed at3121-3122. All producers of these local fused/handshake semaphores belong to the same merged program. Consequently, rank3's unused self-write credit is not a demonstrated next-launch accumulation issue: its dir1 reader receives zero remote slices and never executes the handshake wait, but the next program reinitializes it. Suppressing the unused increment or consuming it in a common dir1 epilogue makes accounting balanced; it is cleanup, not the cause of the observed guard rejection. Do not fix it by skipping local publication, which rank3 matmul requires.

**Do not create a transport/compute dependency cycle.** Matmul readiness waits must follow already-visible slice publication and must not prevent transport from draining its CB. The current producer never waits for matmul; AG worker cores and CBs are separate from matmul's. Adding an acknowledgement from matmul before forwarding can deadlock the line. Keep reader full packet reservations/pushes matched to writer full packet waits/pops, even on partial final payloads (`minimal_default_reader.cpp:137-153`; writer `:383-388,450-454`). The two-worker physical 40-tile input is split20/20; each direction's worker barrier publishes one complete 40-tile source, not two independent ready events. With four K tiles per matmul block, each source is consumed as ten A/B blocks, while each matmul CB reserves/publishes full configured blocks and existing compute drains the same 40-block stream. Tail N validity changes payload use, not this K block count.

**Worker coordinate/index agreement is another bounded requirement.** The factory constructs directional sets then converts them to coordinate vectors (`all_gather_async_default_program_factory.cpp:231-232,256-259,335-343`), while runtime indices are `worker+link*workers` (`:614-624`). Across rows/links these orders can differ: an 8-wide rowwise placement with two links and two workers yields dir1 loop order `(4,0),(5,0),(2,1),(3,1)`, but x-first set order `(2,1),(3,1),(4,0),(5,0)`. The supposed master then waits on one core while slaves increment another (`worker_sync_utils.hpp:57-72,103-108`). The actual one-link/two-worker placement keeps each group in one row and avoids that example. Initially validate this mapping; to support general placement retain the direction vector in allocation order or compute each current-core index from the exact serialized vector. This is not an explanation for the present host rejection.

## Resource feasibility and bounded validation

Do not equate Kblock dividing40 with L1 feasibility. From the descriptor CB formulas (`ccl_fusion_mcast_1d.cpp:168-205`, `ccl_fusion_mcast_2d.cpp:243-264`) and BF16 A/output, BFP8_B B, FP32 intermediate defaults (`tt/multichip_decoder.py:109-114`), approximate A/B/output/partial allocations per active core are:

| Configuration | A | B | Output | FP32 partials | Sum |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2D N34, Kblock4 | 16,384 | 295,936 | 69,632 | 139,264 | 521,216 bytes |
| 1D N9, Kblock4 | 16,384 | 78,336 | 18,432 | 36,864 | 150,016 bytes |

These exclude other buffers, semaphores, binaries, runtime configuration, and concurrent resident resources. BF16 B raises the totals to782,336 and219,136 bytes. Kblock40 would make the 2D B CB alone about2.96MB. Verify actual allocation against the active subdevice; a legal smaller K block or more N cores is an adaptation, not grounds for a blanket family rejection. Do not evade L1 failure by adding extra output-block sweeps without implementing their schedule/base contract. Each prepared B signature also owns327,680 bytes/device of persistent gathered BF16 scratch; retain accounting for the setup interleaved weight copy and all existing decoder weights.

The following tests are bounded follow-ups for the owner after implementation; none were run here:

1. **CPU descriptor/schedule checks first.** All four ranks, both factories, Kblock4 and8, M1/32, N272. Assert 4 unique sources,40/20 total K blocks respectively, per-direction producer limits, identical A/B source order, and nine-word parser cursor agreement. Check source/base formulas and preservation of B's per-core column base. Validate all initial unsupported layouts/outer-loop variants fail on the host. Include worker-coordinate agreement and physical multicast destination exclusion. Retain TP8 Ring descriptor/ABI regression; a TP4 Ring is not a substitute for Linear.
2. **One fresh-cache hardware smoke per factory.** Actual TP4/FABRIC_1D/Linear, M32/K5120/localK1280, first a small legal N then N8704. Use rank-distinct A and independently K-slice-distinct B with multiple output columns; check gathered scratch and each rank's matmul output separately against separate AG + matmul and a host reference. Constant identical slices, or only a full-output aggregate PCC, can hide an A/B permutation. Include one-source-at-a-time patterns and a signed/cancellation pattern. Cover endpoint0/interior1/interior2/endpoint3, not just one output shard. Use the model's unchanged numerical gates; source-order reduction can differ from sequential unfused accumulation.
3. **Cover both exact candidate geometries and M1/M32.** In particular the 1D31-core N tail and dummy physical multicast destination, plus 2D8x1 exact N coverage. Run Kblock4 first; Kblock8 is a bounded second geometry. Check actual L1 placement. For an initial full-rectangle 1D control, per_core_N17 uses16 work cores, but report it as a different geometry rather than silently changing the candidate baseline.
4. **Separate watcher and completion tests.** Run watcher without profiler on a fresh cache. Synchronize explicitly after eager calls; a Python return only enqueues device work. Require blocking replay or final device synchronization and readback before claiming completion. A timeout must preserve the owner's existing hang evidence before any recovery; this investigation does not authorize recovery commands.
5. **Cache and repeated-launch ownership.** Reuse a compiled signature while replacing input, weights, gathered scratch, resulting matmul-output allocation, caller packet semaphores, and barrier; confirm addresses refresh and program-cache count stays stable where intended. Use alternating rank-coded inputs whose expected results differ. Then replay at least50 launches without per-launch host synchronization, with bounded final completion and changed-data validation at checkpoints. Exercise rank skew (for example, an ordered extra workload on one rank) under the admission contract. Same-input first/last equality cannot prove generation safety. Test persistent and newly supplied scratch, and verify consumed global ready/admission credits and endpoint local initialization.
6. **Only after correctness, measure the coherent family.** Compare conventional1D/2D and separate/minimal controls at B1/B32, layer kinds0/3, same selected precision, actual weights/recorded activations, same hidden-sharded residual input/output, and admission cost included. Measure local/remote overlap with profiler separately from watcher, plus complete attention-RS/add/norm/AG/projection/SwiGLU/down/RS/add latency. Track resident copies, trace resources, and layer-stack ownership. No immediate replicated-residual restoration is required. Conventional speed relative to the current DRAM-sharded projection is unresolved.

Existing conventional tests are in the misleadingly named `tests/nightly/t3000/ccl/test_minimal_all_gather_matmul_async.py`: its fused call is at278 and the principal mesh at394 is1x8 with Ring fabric at529-534. Eight slices match that setup; its trace loop synchronizes after each replay (`:317-320`), so it does not establish this TP4 Linear queued contract. Llama AGMM tests and minimal AGMM tests exercise different operations.

## Other potential issues and claims not promoted

- **B parser cursor mismatch outside the bounded layout.** The existing serializer emits nine words, while a nonwaiting receiver consumes only seven (`ccl_op_fusion.cpp:679-692`; `worker_sync_utils.hpp:194-205`). Interleaved B has no subsequent sequential parser that uses that cursor; multicast uses an explicit offset (`reader_bmm_tile_layout_in1_sender_writer_padding.cpp:250-254`). DRAM-width-sharded B does parse further sequential args at181-189 and would misinterpret the semaphore words. This is a real adjacent ABI defect, not the present interleaved candidate's guard failure. The proposed Linear parser consumes the fixed record, and the initial layout restriction excludes that path.
- **Sharded owner mapping and helper thresholds.** SINGLE mode and `align_to_slice_and_sync` need their own source-to-owner proof. This report does not infer their correctness from the MULTI table, and does not require them to ship with initial interleaved Linear support.
- **Physical Ring availability is unproven.** Saved device inventory and successful neighbor Linear traffic do not prove a closing edge or its fabric order. No physical topology discovery was performed. Even a valid TP4 Ring would still hit the hardcoded eight-slice receiver mismatch. Keep FABRIC_1D for this repair.
- **Performance and historical minimal hangs.** Slower measured minimal families do not prove conventional will be slower; historical persistent-minimal workarounds do not prove the same low-level failure occurred here. The conventional path never reached execution. Its newly identified persistent hazards are repair requirements established by source ordering, with runtime manifestations still to be tested.

## Static checks performed and report adjudication

Read the supplied audit independently against API, validation, workload descriptor factory, signaler serialization, AG worker kernels, CCL1D/2D descriptor builders, shared legacy matmul readers, program semaphore dispatch, tests, and retained artifacts. Three isolated xhigh inspection agents independently checked transport, consumers, and competing explanations; none edited files or executed the target.

A stdlib-only Python ledger enumerated all TP4 schedules, checked every threshold against final producer counts, checked unique full K coverage for block4 and8, and verified each A tile start times272 equals its B K-row base. It also checked full valid-column coverage for the 2D N34 and1D N9 partitions. All assertions passed. It reproduced the legacy four-step schedule's impossible waits on ranks0/1/3, and the stock eight-slice arithmetic `80 != 40`. Existing JSON metrics and candidate SHA were parsed without importing TTNN. These are static contract checks, not a simulator or hardware proof.

After drafting, headline claims were rechecked against reachability and lifetime: the host guard is the only observed conventional failure; receiver assertions/absent waits and persistent-generation hazards are masked by that guard. Endpoint local-counter accumulation, the nonselected sharded helper, and the B parser defect were explicitly separated from that causal chain. The proposed intervention matches the actual failing boundary while retaining the measured hidden-sharded MLP family contract.
