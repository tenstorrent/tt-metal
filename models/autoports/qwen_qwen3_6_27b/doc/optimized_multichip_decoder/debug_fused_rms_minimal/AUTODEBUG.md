# AutoDebug: fused_rms_minimal TP4 stats mapping and consumer contract

Inspection date: 2026-11-13. Checkout HEAD: `8502f42bca20dcad1507fbc8cebf7032220b7d8e` (workspace contains unrelated existing changes). This investigation read the prepared tree; it did not import TTNN, execute the target, build, use devices, reset anything, or edit production/helper code. Only this report was written. Three isolated xhigh subinvestigators audited mapping/fabric, the compute consumer, and admission/tests; their important findings were checked against source here.

## Leading diagnosis

1. **The current stats-core subset construction has host undefined behavior for every requested TP4 C/P case that reaches the factory.** Factory F:893 computes `C + 4 - (1 / 4) = C + 4`, then F:896–897 constructs a vector using iterators beyond a C-element vector. This happens before target execution. Invalid tensor capacities may be rejected earlier; real hardware or a larger nominal stats grid cannot make this expression correct.
2. **Replacing that expression with a ceiling count is insufficient.** The writer destination is determined by the global gathered *page* index. For one page per rank, source rank `r` must target stats core `floor(r/P)` and page `r%P` on every device. The existing `floor(C/4)*r` subset start is a different quantity and is wrong for C2/P2, among other configurations.
3. **Transport capacity and compute capacity are different contracts.** C4/P1 and C2/P2 can store the four gathered pages in aggregate after correcting the mapping, but the current compute consumer reads four contiguous pages locally on its sender core. Its dynamic CB requests 16,384 bytes on that one core; backing shards of one/two FP32 pages have only 4,096/8,192 bytes. Dynamic-CB validation rejects these layouts. There is no cross-core stats staging reader. C1/P4 on the sender is the minimal compatible storage arrangement, with actual padded stats width 128.
4. **Even that canonical layout does not clear all source issues.** The seven ordinary compute workers in the requested 8×1 input grid run an unconditional post-stage `compute_kernel_hw_startup(c19,c14,c17)` although c19/c17 exist only on the sender. Also, this second startup violates the startup API's explicit call-once contract. Invalid metadata use is proven; a particular hang or numeric signature is not established without hardware.
5. **Only `num_links == 1` is implemented by this one-page path.** Positive values greater than one pass validation but multiply a semaphore target without creating additional sending workers. TP4 links2 can generate four credits against a wait for eight.

These findings identify repair boundaries for this optimization family. They do not reject fused distributed RMS normalization, establish its performance, or claim that fixing the iterator expression alone makes the target correct.

## Evidence notation and observation boundary

Line references below use these source aliases, relative to the repository root:

| Alias | File |
|---|---|
| F | `ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/device/rms_allgather_program_factory.cpp` |
| V | `ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/device/rms_allgather_device_operation.cpp` |
| W | `ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/device/kernels/dataflow/rms_writer.cpp` |
| K | `ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/device/kernels/compute/rms_compute.cpp` |
| S | `ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/device/kernels/dataflow/rms_sender_reader.cpp` |
| R | `ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/device/kernels/dataflow/rms_receiver_reader.cpp` |
| H | `ttnn/cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp` |
| B | `ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/rms_allgather_nanobind.cpp` |

Direct observations here are source facts and stdlib-only arithmetic checks. The requester supplied the intended TP4 Linear/local1280/physical32 matrix and the suspect expression. No runtime failure artifact with a matching invocation was supplied or generated. The iterator UB, placement contradictions, CB capacity check, missing-CB startup, and credit deficit are independently demonstrated from the current source; the final runtime manifestation of each remains unmeasured.

The autoport's existing `tt/hidden_norm_fused_candidate.py` and `tests/probe_hidden_norm_fused.py` do **not** call this native op. They use custom `generic_op` kernels and local width5120. The probe creates separate width5120 activations per rank and uploads a concatenated width20480 tensor that shards back to width5120 per rank (`probe_hidden_norm_fused.py`:339–355,412–428). Those artifacts are not a passing control for TP4 shards of1280. The documentation audit `remaining_bottleneck_audit24.md`:63–67 proposes investigating this family; it is context, not execution proof.

## Concrete lowering of the requested target

The public C++ wrapper directly forwards its operands (`rms_allgather.cpp`:12–47). There is no operand swapping or hidden conversion. The primitive obtains ring size from the selected mesh axis, defaults links to1, and selects this factory (V:330–366). Use cluster_axis1 on a 1×4 mesh, or cluster_axis0 on a 4×1 mesh; the selected axis must have extent4. A rank's source index is `mesh_coord[cluster_axis]` (F:90–112).

| Object / value | Required concrete contract for the bounded candidate |
|---|---|
| Input logical / padded | Per-device `[1,1,M,1280]`, `M=1..32`; padded `[1,1,32,1280]` |
| Input storage | BF16 TILE, L1 WIDTH_SHARDED, ROW_MAJOR orientation, full rectangle `(0,0)..(7,0)` |
| Per-core input shard | `[32,160]` = 5 tiles; local width1280 = 40 tiles = 8×5 |
| Program configuration | grid8×1, block_h1, block_w5, subblock_w1, out-of-place for a clean initial control |
| Compute / stats | `fp32_dest_acc_en=true`; FP32 stats required by V:33–44 |
| Gamma | FP32 ROW_MAJOR, local shape `(40,32)` or `(1,1,40,32)`, physical volume1280; each stick128 bytes |
| Stats | FP32 TILE L1 WIDTH_SHARDED ROW_MAJOR; actual padded local shape `[1,1,32,128]`; canonical grid only `(0,0)`, shard `[32,128]`, C1/P4 |
| Output | BF16, same shard spec as input for the bounded control; logical shape stays local `[1,1,M,1280]` |
| Global semaphore | Initial0, grid covers all eight input workers, valid for each participating device; links1 |
| Topology | Explicit Linear; four devices along the selected cluster axis |

V:47–102,164–221 enforce most input/layout/program facts. For block5 the positive divisors are1 and5. FP32 with the usual half-DST configuration allows subblock≤4 (F:213–218), leaving **subblock1**. With these values, `num_subblocks_w=5`, first-stage blocks8, second-stage blocks0, all-to-all workers1, ordinary workers7 (F:58–76,265–272,337–355). Every worker's `num_reduce_tiles_per_block_h` is5, including the last: `40 - 7*5 = 5` (F:915–920). No hidden-width padding branch differs across M1..32.

M1 and M32 lower to the same one-tile-row work counts. M is a logical row count, not a split of the 1280-wide reduction; all rows reduce independently. Correctness checks must compare only the live logical rows while retaining honest physical32 allocation.

The local output shape is copied from the input (V:235–259); stats all-gather does not all-gather activations. Reassembling four local outputs along hidden width yields the global5120 tensor. Do not label a local1280 output as5120.

## Global gathered tile-to-core mapping, with bounds

Let R=4 ranks, C be the number of coordinates in `corerange_to_cores(stats.grid, ..., true)`, P be pages per stats shard, and T=4096 bytes per full FP32 tile. F:197–199 currently computes P as `shard_h*shard_w/1024`. For the audited WIDTH_SHARDED one-row stats layout this means `shard_h=32`, `shard_w=32*P`; positive, full tile geometry must be checked before using that quotient.

Each device reduces its local1280 channels to one tile of per-row statistics. Gathering does not pack four statistics into four adjacent scalar columns in one tile. It deposits **one full tile per source rank**. The global gathered page order is rank order, so global page `g=r` satisfies the Euclidean decomposition

```
g = core_index * P + local_page
core_index = g // P
local_page = g % P
destination = (physical(stats_cores_vec[core_index]), stats_base + local_page*T)
```

Bounds are `P>0`, `C>0`, `0<=r<4`, `core_index<C`, and `0<=local_page<P`. All four destinations must fit the actual backing allocation; for the uniform complete shard vector the aggregate condition is `C*P>=4`. The op also needs the four-page logical/padded extent and the consumer-local conditions below; aggregate capacity alone is not admission.

The same source-rank destination is used on **every receiving device**, including the local device. It is not selected using the receiving rank, an input worker index, a fabric hop, or `C/R`.

| Pages/core P | r0 `(core,page)` | r1 | r2 | r3 | Minimal C |
|---:|---|---|---|---|---:|
| 1 | (0,0) | (1,0) | (2,0) | (3,0) | 4 |
| 2 | (0,0) | (0,1) | (1,0) | (1,1) | 2 |
| 4 | (0,0) | (0,1) | (0,2) | (0,3) | 1 |

For a future producer with Q pages/rank, use global interval `[r*Q+a, r*Q+b)` for its assigned local interval `[a,b)`. A nonempty interval touches core indices `g0//P` through `(g1-1)//P`, with first offset `g0%P`. Its bounds are `0<=g0<g1<=R*Q`, `g1<=C*P`, and both endpoint core indices within the vector. Empty intervals have no destination. A ceiling of the *number of cores divided by ranks* cannot replace these page-space calculations. The present kernel is hardcoded to one page and one destination; merely giving it more coordinates does not implement a multi-page loop.

### Current expression and ceiling-only counterexamples

F:893–897 currently does:

```
count = C + R - (1/R)
start = (C/R)*rank
subset = vector(begin + start, begin + start + count)
```

All arithmetic is integer. At R4 the count is C+4. C4 gives end indices8,9,10,11 in a vector of length4; C2 gives end6 in length2; C1 gives end5 in length1. The iterators/range construction are already invalid even before F:1020 indexes the subset. The defect is not a requirement to have at least four stats cores.

Changing only count to `ceil(C/4)` leaves start `(C//4)*rank`:

- C4/P1 then addresses the correct four cores, but the consumer still cannot bind four pages to a one-page bank.
- C2/P2 makes every start0. r0/r2 both overwrite core0/page0; r1/r3 both overwrite core0/page1. Core1 is never filled. Semaphores can still count four contributions, so counts do not detect the data collision.
- C1/P4 happens to use core0/pages0..3 correctly once the bad range construction is removed.
- C4/P4 is overprovisioned, but valid page mapping uses core0 for all ranks. The unchanged start instead sends ranks to cores0/1/2/3, so even this capacity-rich case is wrong.

Using `ceil(C/4)*rank` as start instead is also incorrect: with C1/P4 ranks1..3 immediately require nonexistent cores. For one page/rank, each rank needs exactly the single bounded coordinate `stats_cores_vec[rank//P]` and offset `rank%P`.

### Full requested C/P capacity matrix

The transport column assumes the iterator and page mapping are repaired, stats padded width128, and actual allocation matches the declared shards. **Every row that reaches this factory currently also hits the iterator UB.**

| C | P | C×P | Correct transport capacity | Current contiguous consumer, after mapping repair |
|---:|---:|---:|---|---|
| 4 | 1 | 4 | Fits; pages span four cores | Reject: c19 wants16KiB, bank4KiB |
| 2 | 1 | 2 | Invalid: r2 needs core2 outside C2 | Invalid capacity |
| 1 | 1 | 1 | Invalid: r1 needs core1 outside C1 | Invalid capacity |
| 4 | 2 | 8 | Fits; pages use first two cores | Reject: c19 wants16KiB, bank8KiB |
| 2 | 2 | 4 | Fits; pages use both cores | Reject: c19 wants16KiB, bank8KiB |
| 1 | 2 | 2 | Invalid: r2 needs core1 outside C1 | Invalid capacity |
| 4 | 4 | 16 | Fits; pages use only core0 | Can satisfy storage contract if core0 is sender and D=4; excess grid entries unnecessary |
| 2 | 4 | 8 | Fits; pages use only core0 | Same qualification |
| 1 | 4 | 4 | Fits; pages use core0 | Minimal compatible arrangement; other canonical blockers still apply |

C0/P0, non-tile geometry, insufficient actual allocation, wrong orientation, and a first stats core different from the sender must also be rejected or explicitly handled before indexing/addressing. Invalid total capacities may already fail general TensorSpec construction; they must not be admitted by a future factory mapping helper merely because its ceiling count looks plausible.

## Actual writer and fabric ownership

The eight input workers are not eight fabric producers. F:1006–1064 supplies destination coordinates and fabric connections only to input worker index `i==0`; the connection is opened on link index0. W:109 parses that input-worker index as `iteration_number`; only `iteration_number==0` enters the fabric body. All other writers wait for stats-ready (W:208–210). The sole compute producer has already combined eight input-core partials into one c7 tile (K:143–160).

W:118–125 parses the first page offset and coordinate arrays, but W:151–172 initializes `core_id=0`, waits exactly one c7 tile, and performs exactly one send. There is no per-core traversal or packet loop despite the generic-looking coordinate-array arguments and unused counters.

The helper H:235–267 performs one local NoC write to that selected coordinate/page, and sends the same payload/address in forward/backward fabric packets. It does not advance to a new stats-core owner at a hop. The packets have **chip multicast** routing (W:145–148), with **NoC unicast** destination on each chip (H:251–261). These are distinct multicast domains. It does not multicast stats data across local stats cores.

For Linear TP4, F:175–180 derives these forward/backward chip counts:

| Source rank | Forward devices | Backward devices | Other chips receiving source tile |
|---:|---:|---:|---:|
| 0 | 3 | 0 | 3 |
| 1 | 2 | 1 | 3 |
| 2 | 1 | 2 | 3 |
| 3 | 0 | 3 | 3 |

Each receiver therefore gets the three remote source tiles, at their source-owned gathered-page destinations, plus its own locally written tile. F:1024–1026 sets the drain core to physical input worker0 on every device. That core is the semaphore recipient even when a source tile's storage destination would be a different stats core. Do not conflate data ownership with notification ownership.

## Stats CB reader ownership and the bounded storage contract

F:303–307 derives `D = stats.padded_shape[-1]/32` and uses D as `num_distributed_devices`, not `ring_size`. F:325 allocates D accumulator-format pages for c19, and F:593–595 binds c19 directly to the stats buffer on **sender_cores only**. In the canonical input rectangle sender is logical `(0,0)`; c17 variance and c18 reciprocal RMS are also sender-only (F:585–591).

After a c20 token, the sender computes a contiguous D-tile reduction from c19 using `NoWaitNoPop`, then explicitly pops D tiles (K:173–175,191–210). The raw incoming NoC writes are the data producer; c19 has no ordinary CB push/wait handshake. This is an intentional direct-memory alias pattern here, not evidence of a missing c19 push. Its read pointer is local to the sender; it is not a distributed tensor accessor.

S:100–134 gathers **input-core partials c2 into c4**, not gathered rank stats into c19. S:153–178 later multicasts the final **reciprocal RMS c18 into c9** across the input workers. Neither this reader nor R stages rank tiles from other stats cores into sender-local contiguous memory.

The dynamic-CB backing rule is explicit: `CircularBufferConfig(CBDescriptor)` applies the buffer and total size (`tt_metal/impl/buffers/circular_buffer_config.cpp`:72–84), and binding requires `total_size <= buffer.aligned_size_per_bank() - offset` (:206–229). Therefore C4/P1 and C2/P2 with D4 encounter the local bank-capacity error after fixing the earlier mapping UB. If that check were bypassed, the sender would still read missing local tiles; that is not a solution.

The minimal compatible storage contract is **all four rank tiles on sender `(0,0)`, D4, P4**, with C1 avoiding unused grid entries. Strict C==1 is a reasonable bounded support choice, but is **not mathematically necessary**: nominal C2/P4 or C4/P4 can satisfy the same local storage/read contract if their first stats core is sender and padded width remains128. General TensorSpec width-shard validation allows extra grid cores (`tt_metal/impl/tensor/spec/tensor_spec.cpp`:62–73). Such layouts do not demonstrate arbitrary C/P support; the live four pages still all belong to the first core. Supporting actual P1/P2 distribution requires a staging/reduction consumer change, not just the correct writer map.

### Padded stats width is a separate invariant

The documented stats shape `[1,1,32,num_devices]` (B:36–39) is misleading for this kernel. With standard TILE alignment and TP4, a logical width4 pads to32, so D becomes1. Default TILE alignment is only tile height/width (`tt_metal/impl/tensor/spec/layout/page_config.cpp`:51–53); padded shape rounds the logical extent to that alignment (`tensor_layout.cpp`:311–324). Increasing nominal C or shard capacity does not by itself establish D4.

The bounded candidate should explicitly obtain/verify actual padded local width128, four pages per gathered stats tensor (one per source rank), FP32 tile size4096, one-row stats geometry, sender placement, and per-bank capacity≥16,384 bytes. Using actual logical width128 is the straightforward unambiguous setup. A deliberately custom-padded smaller logical width would need its padded spec verified. Extra allocated space with D1 is insufficient: even if writes fit, K consumes only one tile and uses scale1 instead of1/4. P1 alone does not imply collisions: corrected C4/P1 mapping places the four page0 writes on four distinct cores, but D1 still consumes just one of them. No zero/one initializer repairs these semantics.

## Semaphore and CB state ledger

### Canonical one-link sequence

| Phase | Producer / ownership | Wait, publication and drain |
|---|---|---|
| Per-core pre-RMS | Each of8 compute workers squares5 BF16 input tiles into c6 and produces1 FP32 partial in c2 | K:111–140 waits/pops5 c6 tiles; c2 pushed once |
| Gather on chip | Seven reader workers notify sender; sender reads8 c2 tiles into its c4 | R:92–97 waits c2, sends readiness, waits release. S:82–113 waits its c2 plus7 readiness credits, resets readiness, releases workers, reserves/pushes8 c4 pages after read barrier |
| Per-device stat | Sender compute reduces8 c4 pages into1 c7 | K:143–160 `WaitAndPopPerTile` drains all8 c4 pages; S:134 and R:147 drain each worker's c2 |
| Gather between chips | Only writer0 reads1 c7 tile, writes source rank's destination locally and on3 peers | W:158–175 waits/pops1 c7; fabric close begins after enqueue |
| Drain notification | Every chip's sender gets3 remote increments | F:996 sets target4; links1 W:181–183 waits target−1=3, omitting a local atomic |
| Stats-ready publication | Sender writer resets global counter and multicasts ready to8 input workers | W:192–207 reset, multicast local stats-ready semaphore with loopback, write barrier, finish fabric close; ordinary writers wait ready. Each writer reserves/pushes1 c20 token (W:208–214) |
| Post-RMS on sender | Sender compute reads4 contiguous raw-written c19 pages, reduces with1/4, adds epsilon, computes reciprocal sqrt into c18 | K:173–218 drains1 c20; reads/pops4 c19 pages without a normal input wait; produces1 c18 |
| Broadcast reciprocal RMS | Sender reader copies1 c18 tile to c9 on all8 workers | S:173–178 waits c18, reserves1 local c9, multicasts with loopback and write barrier, pushes its c9, pops c18, publishes post-ready. R:149–151 reserves1 c9 and publishes after post-ready |
| Normalize and affine | Each compute worker consumes its local input and reciprocal RMS, then gamma | K:221–243 consumes5 input tiles,1 c9,5 c15 intermediate tiles, produces5 output tiles;5 gamma pages remain resident for kernel duration |

For the matching-shard output c10 aliases the output buffer; no reshard writer runs (F:149,547–554; W:239–242). Input c5/c12 are direct aliases; they intentionally bypass ordinary input CB waits. Local per-program semaphores are descriptor-initialized to INVALID (F:435–445). The caller-owned global semaphore is initialized0 by the caller and explicitly reset by writer0 after the remote wait.

W:163 uses the fused remote helper with `flush=false`, which is not on its own a proven ordering bug. The receiver implementation issues the data write then atomic on the same NoC and VC (`tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp`:236–256; router variant `router_local_delivery.hpp`:187–191). Repository NoC ordering documentation guarantees program order for writes on the same NoC/VC (`docs/source/tt-metalium/tt_metal/apis/kernel_apis/data_movement/ordering.rst`:22–28). H's helper is nonblocking and has no immediate source drain; W starts connection close, pops c7, later performs a write barrier and finishes close before publishing its own c20. There is no second c7 producer within this invocation, so the early pop alone does not demonstrate overwrite of its payload. Hardware testing can confirm this protocol but source does not justify headlining an independent flush race.

### Why multiple links cannot complete

F:1009–1012 computes a nominal distribution of a single page across links, but only i0 reaches that code and only i0 receives connection args. For any positive links, its assigned interval is `[0,1)`. Other workers neither send empty completion packets nor own additional links in W.

| num_links | TP4 target in F | Writer wait | Credits actually available per chip | Result under otherwise corrected setup |
|---:|---:|---:|---:|---|
| 1 | 4 | 3 | 3 remote; local data completion handled by sender's drain | Counts agree |
| 2 | 8 | 8 | 3 remote +1 local =4 | Deficit4; wait cannot complete normally |
| 4 | 16 | 16 | 3 remote +1 local =4 | Deficit12 |
| 0 | — | — | — | V:197 rejects |

For links>1 W:185–189 sends just one local atomic and waits for `4*num_links`. The source defect is missing admission/multi-link ownership, not a fabric link-count performance setting. Default None resolves1 (V:361); explicitly require1 for this bounded path. Arbitrary external increments or overlapping invocations are not legitimate missing credits.

Global reset is not an overlap protocol: invocations sharing stats/global-semaphore storage must preserve their queue ordering/lifetimes. Source inspection here does not establish cross-queue safety. A host call returning, descriptor query success, or trace enqueue is not device completion. Sibling ordinary correctness paths use `synchronize_device` or `to_torch`; some timing paths use nonblocking trace replay. The parent should preserve its serialized checkpoint and perform explicit completion after each initial control before attributing a later wait/teardown failure to that later operation.

### Canonical buffer sizes and ownership

For no residual, matching output sharding, FP32 accumulation/gamma/stats, block5:

| CB | Pages / format | Scope and purpose |
|---|---|---|
| c0,c1 | 1 each BF16 | All8 cores; pre scalers |
| c2,c3 | 1 each FP32 | All8; partial / unused second-stage intermediate |
| c4 | 8 FP32 | Allocated all8; sender uses8 input-core partials |
| c5,c12 | 5 each BF16 | All8; dynamic input views |
| c6 | 5 FP32 | All8; squared input |
| c7 | 1 FP32 | Allocated all8; sender's transport stat |
| c8 | 16 packet-header pages | All8; sender uses3 headers; size is not a tensor-tile size |
| c9 | 1 FP32 | All8; reciprocal RMS broadcast |
| c10 | 5 BF16 | All8; dynamic output |
| c11 | 5 FP32 | All8; gamma |
| c13,c14 | 1 each BF16 | All8; epsilon and post scaler |
| c15 | 5 FP32 | All8; normalized intermediate |
| c17,c18 | 1 each FP32 | Sender only; variance / reciprocal RMS |
| c19 | 4 FP32 =16,384 bytes | Sender only; dynamic stats alias |
| c20 | 1 page of2 bytes | All8; readiness token |

F:310–326,486–597 produce20 CB descriptors:17 across all8 cores,3 sender-only. Reshard adds c16; residual fusion adds c21/c22. This is a descriptor/ownership calculation, not a measured L1 high-watermark or compiled resource report.

## Additional source blockers after honest C1/P4 setup

### Receiver post-stage configuration uses absent CBs

K:180 calls `compute_kernel_hw_startup(cb_stats, post_cb_scaler_global, cb_var)` outside the `if constexpr(is_allgather_worker)` at K:191. Factory's separate ordinary-worker compute kernel has is_allgather_worker=false (F:838–842), but only sender owns c19/c17 (F:585–595). Thus all seven ordinary workers run startup with absent input19/output17.

This is actual hardware configuration, not an unused handle declaration. `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h`:72–79 invokes unpack/math/pack configuration. Per-kernel compilation populates CB metadata from that kernel's core ranges (`tt_metal/impl/program/program.cpp`:2958–2971,3394); absent formats remain Invalid (`tt_metal/jit_build/hlk_desc.hpp`:42–50). Blackhole unpack configuration reads c19's local FIFO page metadata and its format (`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_unpack_common_api.h`:44–81), and pack configuration uses output17 metadata (`llk_pack_common_api.h`:45–70).

The smallest intervention belongs at per-role post-stage configuration: only sender should configure the stats reduction buffers; ordinary workers must configure valid local operands for their next normalization phase. Adding four-page dynamic stats CBs to every input core would not follow the actual consumer ownership and needlessly broadens allocation. Invalid configuration access is proven; exact corruption, assert, or hang behavior is unverified.

The same K:180 is a second full startup after pre-stage arithmetic on the sender too. The startup API explicitly requires exactly one call at the beginning and warns against mid-kernel use (`compute_kernel_hw_startup.h`:29–47). Existing TODO comments at K:178–179 acknowledge this. An appropriate phase-local reconfiguration must preserve formats and required destination synchronization without rerunning call-once startup. This is a proven API-contract violation; it is not evidence of a specific race signature on this target.

### Stats admission must describe the actual reader

V currently checks stats presence and accumulator dtype (28–44), but not the TILE/L1/WIDTH_SHARDED/ROW_MAJOR geometry, D==ring_size, selected sender location, four-page local capacity, or P/C bounds subsequently assumed by F. F unconditionally dereferences stats shard metadata (197–199). Fixing only the final iterator messenger leaves these earlier missing contract checks. Admission must either enforce a documented bounded sender-local layout or route distributed stats into a genuinely compatible consumer.

For canonical BF16 input + FP32 accumulation, FP32 stats is already required and correct. Do not weaken that check or use old BF16-stats examples to avoid capacity: F's stats reader aliases accumulator-format pages and the transport page size follows stats dtype.

### Setup and output contracts can otherwise invalidate the comparison

Gamma must be converted once to local ROW_MAJOR sticks containing the correct1280 slice. V:112,129–144 requires last dimension32 and local physical volume1280, accepting FP32; a tiled5120 gamma from an older candidate does not qualify. W:216–236 fetches each stick's two facelines and copies its second half to the tile's second horizontal face. For FP32 it uses64-byte facelines,128-byte reads,1024-byte second-face offset, and4096-byte CB page strides. No source-level gamma-address-unit mismatch was found for honest `(40,32)` FP32 sticks.

The public `persistent_output_tensor` documentation says it is a32×32 intermediate (B:45–46), but V:368–374 binds it as `preallocated_output` and V:264–265 returns it directly. For this target it must be the actual normalized output, with logical local1280/M and matching storage; the safe initial control omits it. This doc discrepancy and missing specific preallocated-output validation are real, but conditional on using that optional argument; they do not explain the current iterator failure.

## Other Potential Issues and limits

- **FP32 storage/accumulator is not an exact-FP32 arithmetic guarantee.** F:831–835 leaves unpack modes empty. Mixed BF16/FP32 formats select TF32 as the ordinary FP32 unpack destination (`tt_metal/jit_build/genfiles.cpp`:1070–1086; `data_format.cpp`:147–155,211–217). The public config here does not request an all-operations exact FP32 path. BF16 scalar CBs also represent the local1/160 reduction factor and epsilon with BF16 storage. This can make results differ from a frozen custom reduction tree; it is not evidence of incorrect stats ownership or a numerical bug by itself. Use a global5120 RMS reference and the intended stage's numerical criterion before deciding whether precision needs further work.
- **`dst_full_sync_en` is dropped.** F:210–230 validates limits using this user-config flag, but the ComputeConfigDescriptor at831–835 omits it; `tt_metal/api/tt-metalium/program_descriptors.hpp`:108 defaults false. Therefore a user enabling full DST to admit subblock5 does not get that actual kernel mode. The bounded subblock1/false control avoids this separate proven config-loss defect.
- **Unmatched c3 pop on ordinary readers.** R:152 pops c3 even though the is_all_to_all_worker=false branch never produced it. This is an unmatched bookkeeping operation. `dataflow_api.h`:260–275 increments acknowledgement and advances/wraps the allocated one-page CB without waiting for occupancy; c3 has no subsequent use in this path. It is not sufficient evidence for a hang or wrong RMS result and should not outrank the direct blockers above.
- **Empty reader descriptor.** With8×1, `num_cores_all_to_all=1` but use_mcast=true. F:397 constructs an empty all-to-all-except-sender set and F:1224–1225 appends that reader descriptor. Inspected descriptor/kernel construction has no rejecting empty-range check in this path. This is not a demonstrated blocker. A future conditional omission must preserve kernel-index/runtime-binding assumptions.
- **Ragged/offset/two-stage grids are outside this bounded target.** The full8×1 rectangle avoids phantom-cell and two-stage ownership questions. Fixes aimed at those other configurations are not needed to explain the canonical stats/core mismatch. No general-grid correctness claim is made.
- **Fabric ordering and buffer lifetime remain runtime verification subjects, not replacement diagnoses.** Same-NoC/VC evidence above defeats a blanket claim that flush=false necessarily publishes remote stats early. Shared stats/semaphore concurrency, trace replay, and next-op completion should be tested after the source contracts are corrected, with serialized initial controls and explicit waits.

## Tests, review, and suggested parent handoff

In-tree test source is not a passing execution record for this checkout. `tests/ttnn/unit_tests/operations/ccl/fusion_subtests/rms_test.py`:824–1030 uses M32 and BF16 gamma, chooses FP32 stats when requested, and does not pass its helper's num_links argument in the native call kwargs (989–1008). Its stats comments at885–888 do not establish the required padded-width contract. `tests/ttnn/unit_tests/gtests/test_graph_query_op_constraints.cpp`:1295–1403 skips mock mode because of the subset overflow; it uses C1 with a large stats width and performs a resource query, not numerical/device-completion proof. Its comment assigning this to insufficient mock hardware is not an explanation for the unconditional integer-range defect.

This session ran only cheap stdlib Python arithmetic (`python3 -I -S` with a heredoc): all9 requested C/P combinations,4 ranks each; uniqueness and capacity equivalence of `(r//P,r%P)`; current out-of-bounds slice ends; ceiling-only destinations; all32 logical-M values padding to32;40/8=5 tiles;4×4096=16,384 bytes. All assertions passed. No target/import/build/device test was run. The first report draft was revisited against F/W/K/S/R and backing-CB/JIT code; overprovisioned P4 layouts, unmatched c3 pops, empty descriptors, and flush=false were explicitly adjudicated rather than promoted into unsupported headline causes.

The smallest defensible parent work boundary is:

1. Correct global-page destination selection with checked P/C/index bounds; do not replace it with a core-count ceiling. Cover C4/P1, C2/P2, C1/P4 and invalid capacities in source-level mapping checks, while distinguishing transport-valid from consumer-supported layouts.
2. Document/admit a bounded TP4 sender-local D4/P4 stats path, links1, honest `[32,160]` input shards on8×1, block5/subblock1, proper gamma sticks and local output shape. If admitting P1/P2, implement stats staging/reduction ownership instead of bypassing CB capacity checks. A strict canonical C1/P4 contract is a scope choice; extra unused P4 core entries are not inherently numerically invalid.
3. Repair post-stage initialization by role and use valid phase-local configuration in place of the second call-once startup. Keep the unrelated lower-priority findings separate unless a selected parameter reaches them.
4. After the parent's serialized queue checkpoint, build and test an explicit one-link C1/P4 control using a global5120 reference, compare M1,3,17,32 first then allM1..32, nonconstant rank-distinct inputs/gamma, stats ownership sentinels, explicit device completion, repeated queue calls, fresh allocations/cache hits, and trace replay. These are follow-ups, not results of this investigation. Include any input/gamma conversion and output gathering in honest candidate accounting at the appropriate boundary.

The first divergent calculation is the host subset/mapping setup. Correcting it exposes the independent local stats-CB contract and receiver phase initialization issues. No available observation justifies collapsing those into a hardware problem or declaring the family unsupported in principle.
