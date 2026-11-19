# AutoDebug: decode norm owners exhaust the shared L1 address allocator

**Leading finding:** the minimal decode norm owner assumes that placing successive owners on different worker rows reduces their persistent L1 allocation cost. Ordinary TTNN sharded L1 allocations use a single lockstep address allocator. Different shard rows therefore still reserve different intervals in the same address space. The allocator is correctly rejecting an infeasible persistent inventory; the earliest error is the owner's capacity/placement assumption, before the failing `ttnn.empty`.

The source-derived construction ledger reproduces **1,454,848 B allocated, 5,440 B free, and a failing 10,240 B per-bank request exactly**. Sixty-four owners require **3,407,872 B of native tensor address reservations**, before semaphores, against the reported **1,460,288 B** capacity.

**Smallest practical repair:** retain all 64 owner objects, their independent phase semaphores, FP32 gamma, and persistent DRAM activation-gather outputs. Make **both** minimal-family statistics and native normalized outputs temporary within the complete `normalize → activation gather → canonical conversion` call. The native output already supports ordinary operation allocation; stats must be explicitly allocated as temporary scratch. This changes the scratch-storage contract and its receipts, not decoder identity. Details and required lifetime conditions follow. No repair was implemented or executed in this investigation.

## Scope and evidence provenance

Investigation date: 2026-11-13. Inspection only: source reads, artifact parsing, hash/AST checks, and a standard-library allocation simulation. No device operations, hardware reproductions, builds, resets, or source edits were performed. Only this report and its requested copy were written. The AutoDebug launcher was not invoked.

Evidence:

- `models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/norm_resources26_minimal_b1.json`
- The adjacent `.log`, `.command.json`, and `norm_resources26_minimal.policy.json`.
- `fused_decode_norm_integration_audit.md` and its CPU receipt were examined as claims to check, not runtime proof.

The reported and inspected Git HEAD is `8502f42bca20dcad1507fbc8cebf7032220b7d8e`, with substantial uncommitted model/native work. Git HEAD alone is insufficient provenance. Initially, all ten source hashes in the failure JSON matched the checkout. During inspection the three Python paths `fused_decode_norm.py`, `multichip_decoder.py`, and `check_multichip_stack_resources.py` were reformatted outside this investigation; their hashes subsequently differed. Re-reading confirmed the relevant allocation sizes, construction ordering, and norm/collective flow remained present. The allocator sources still match the preparation command's native-source manifest, including every file under `tt_metal/impl/allocator/`, and the relevant RMS native factory/device-operation hashes still match the failure JSON.

The failed helper hash is `8d1488567401528596dcbf7cca3698335e157dbea70830b8fafd6c542cb65713`; the subsequently formatted helper hash is `b60e24a299e35f7176977f0e2bc65b07de14da527957687014c726e7c6f03b37`. Helper references below use the formatted file unless explicitly labeled historical. The failing `norm_outputs` allocation was historical line 102, now lines 146–155. No execution result is attributed to the reformatted sources.

## Direct observations versus deductions

**Directly recorded:** TP4 mesh `[1,4]`, Blackhole, decode batch 1, activation gather in DRAM, `distributed_decode_norm="minimal"` for both layer kinds, trace tracking enabled, program-cache tracking not skipped. The failure stack is `run → install_decode_norm → FusedDecodeNormOwner.__init__ → ttnn.empty(norm_outputs)`. It requests 81,920 B over eight banks, requiring 10,240 B per bank. Capacity is 1,460,288 B; allocated is 1,454,848 B; total and largest contiguous free space are both 5,440 B.

The last successful memory sample is `two_representative_layers`: L1 allocated window 109,248 B, 110 banks. There is no `all64_communication_owners` sample, no prefill warmup, no eager reference, and no main replay result. `resource_counts.layers=64` is the requested inventory, not proof that 64 owners were constructed. The separate trace-tracker positive control passed and was torn down before decoder construction.

**Deduced from source:** the common address allocator explains the failure without prefill, a trace leak, or a kernel execution failure. The exact ledger predicts failure while installing logical layer index **26**. That index is an inference, not a logged coordinate. The first output in that owner's output list is predicted to fail. This reconstruction is unusually strong because both the earlier 109,248 B sample and the exact final allocation count agree.

## 1. Earliest incorrect assumption: row placement is not allocator partitioning

At `models/autoports/qwen_qwen3_6_27b/tt/fused_decode_norm.py:113–130`, the comment and construction spread owners onto `worker_row = layer_idx % grid.y`, using eight worker cores `(0..7,row)` and a single sender `(0,row)`. This changes physical shard placement. It does not select another allocator.

The actual lowering is:

1. `ttnn.empty` creates a device tensor (`ttnn/cpp/ttnn/operations/creation/creation.cpp:365–372`; `ttnn/core/tensor/tensor_ops.cpp:101–122`).
2. Tensor allocation builds a `MeshBuffer` with page size, buffer type and sharding arguments, without a subdevice allocator ID (`tt_metal/impl/tensor/tensor_impl.cpp:40–56`).
3. The mesh allocator creates an owning backing buffer and supplies its address across the mesh (`tt_metal/distributed/mesh_buffer.cpp:279–324`).
4. Ordinary L1 allocation always selects **allocator ID 0** (`tt_metal/impl/allocator/allocator.cpp:287–295`). The passed compute grid is the allocator's compute grid; the shard row does not create a new address pool.
5. `BankManager::allocate_buffer` uses `num_shards` to choose the divisor for `size_per_bank`, then calls the same allocator's `allocate(size_per_bank, ...)` (`bank_manager.cpp:469–496`). In the default lockstep configuration this is one allocator, not a set of per-core free lists (`allocator_types.hpp:49`; `bank_manager.hpp:41–53`).

Consequently, moving a stats tensor from row 0 to row 1 changes the core holding its bytes but still consumes another 16,384 B interval from allocator ID 0. Moving an eight-shard output changes its cores but still consumes another 10,240 B interval.

The audit's maximum sender payload of **372,736 B** (`fused_decode_norm_integration_audit.md:34–42`) is correct physical payload arithmetic for seven owners on a row. Its use as the effective allocation-capacity model is wrong for ordinary lockstep allocations. The code does not reserve that maximum once and overlay the other rows inside it. It makes independent owning allocations and sums their intervals.

`capacity()` at `fused_decode_norm.py:17–33` reports 196,608 B of L1 tensor payload per minimal owner. That is not the same quantity as the common-address reservation. The broader planner records this inventory but adds only norm DRAM into its resident DRAM calculation (`tests/optimized_multichip_capacity.py`, `norm_inventory`/`norm_dram` calculation). There is no 64-owner L1 admission check accounting for all future owners, semaphores, transient inputs, and native circular buffers. Add an explicit allocator-window inventory and runtime peak budget; do not relabel payload as reservation or rely on a per-core table as admission.

## 2. Concrete sizing and exact failure ledger

All batches B1..32 have one physical 32-row tile. The constructor always allocates the B32 backing storage. Batch 1 does not reduce this persistent demand; B32 has the same construction problem.

| Minimal owner resource | Physical payload per owner | Shared allocator window per owner |
|---|---:|---:|
| Two FP32 TILE stats `[1,1,32,128]`, one sender shard each | 32,768 B | 32,768 B |
| Two BF16 TILE normalized outputs `[1,1,32,1280]`, eight shards each | 163,840 B | 20,480 B |
| Native tensor subtotal | **196,608 B** | **53,248 B** |
| Eight independent global semaphores | Core-dependent payload | **512 B** |

Source: `fused_decode_norm.py:135–155`. One output is forty 2,048-byte BF16 tiles divided into eight five-tile shards: **81,920 / 8 = 10,240 B**. A stats tensor is four 4,096-byte FP32 tiles on one sender: **16,384 B**. These exactly match the reported request; there is no logical/padded size mismatch in the failing allocation.

Global semaphores also allocate real sharded L1 buffers (`tt_metal/impl/buffers/global_semaphore.cpp:84–108`). Their four-byte pages are subject to a **64-byte minimum allocation**, independent of how many cores contain a semaphore. The source chain is `l1_banking_allocator.cpp:150–158` → `bank_manager.cpp:120–128,194–196` → `algorithms/free_list_opt.cpp:97–98`: FreeListOpt uses DRAM alignment as its minimum. Blackhole's DRAM alignment is 64 B (`tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:374–394`).

The resource test first creates the representative layers 0 and 3, then reuses their weights in shallow copies. Each later copy installs a fresh norm owner **before** constructing its new legacy CCL/convolution resources (`tests/check_multichip_stack_resources.py:168–199`). The legacy CCL manager uses twelve semaphores per layer (`models/demos/gpt_oss/tt/ccl.py:42–59`); a linear-attention clone adds three convolution semaphores.

```text
Two complete prototypes:
  2 × (53,248 native tensor window + 8 × 64 norm semaphore window)
  + 24 × 64 legacy CCL semaphore window
  +  3 × 64 convolution semaphore window
  = 109,248 B                                     [matches recorded sample]

Then 19 complete linear clones and 5 complete full-attention clones:
  109,248 + 19 × (53,248 + 23 × 64)
          +  5 × (53,248 + 20 × 64)
  = 1,421,568 B

Installing the next norm owner, inferred logical index 26:
  + 8 × 64 norm semaphore reservations
  + 2 × 16,384 stats reservations
  = 1,454,848 B                                   [exact recorded count]

  1,460,288 − 1,454,848 = 5,440 B free
  First normalized output needs 10,240 B → OOM     [exact recorded request]
```

The layer-kind sequence was obtained from `doc/hf_config.json` and checked with a pure standard-library simulation; no model or TTNN module was imported. The preconstructed layer-3 owner is included even before logical index 3 is visited. There are 26 complete norm owners at this point, plus the partially constructed next owner.

For the intended complete stack:

| Reservation | Common L1 address window |
|---|---:|
| 128 stats buffers | 2,097,152 B |
| 128 native outputs | 1,310,720 B |
| 512 norm semaphores | 32,768 B |
| 768 legacy CCL semaphores | 49,152 B |
| 144 convolution semaphores | 9,216 B |
| **Combined** | **3,499,008 B** |

Even stats alone exceed the available capacity. Free space equals the largest free block at failure, so this request is failing total capacity, not merely fragmentation. Synchronizing or compacting cannot make the inventory fit. Conversely, multiplying the memory view's allocated window by its 110 banks (the test's `allocated_bytes_per_device`) is an allocator-envelope measurement, not the actual sparse sharded tensor payload.

The two DRAM gather outputs and two RM FP32 gammas consume 665,600 B per owner, 42,598,400 B for 64 owners per rank. They do not explain this L1 failure.

## 3. Minimal repair with 64 independent identities

For the **minimal** family only:

1. Keep a new owner for every decoder, including shallow clones. Retain `owner_id`, `layer_idx`, phase/batch checks, the poisoned partial-call state, all independent norm/gather semaphores and barriers, FP32 gamma, and the existing DRAM `ag_outputs`/`ag_views`.
2. Store the sender stats memory configuration as metadata. Remove constructor allocations for minimal `self.stats`, `self.norm_outputs`, and their `self.norm_views`. Do not merely drop the output bases while retaining the views: reshape views keep the underlying allocation alive (`ttnn/core/tensor/tensor_ops.cpp:469–478`; `ttnn/core/tensor/storage.cpp:157–161`; shared-root deallocation handling in `ttnn/core/tensor/tensor.cpp:125–148`).
3. Allocate one temporary FP32 TILE `[1,1,32,128]` stats tensor with the existing sender memory config inside each minimal normalization. Stats is required by validation/factory code, so omitting `stats=` is not a fix (`rms_allgather_device_operation.cpp:28`; `rms_allgather_program_factory.cpp:124–126`).
4. Omit `persistent_output_tensor` for `fused_rms_minimal`. `RMSAllGatherDeviceOperation::create_output_tensors` already allocates the correct width-sharded output when it is absent (`rms_allgather_device_operation.cpp:300–348`). Keep the same BF16 output, FP32 gamma/accumulation, geometry, conversion to `ag_memory`, full activation gather, and canonical projection-input conversion.
5. Hold temporary stats through the end of `normalize`; retain each temporary tensor through its last consumer submission. Keep scratch out of persistent owner/view caches. Update ownership receipts and tests to describe persistent resources and temporary peak requirements honestly.

This removes **3,407,872 B** of persistent tensor-window reservations while retaining all **1,424 independent semaphore identities**. It does not prove the resulting full forward peak fits; runtime inputs, other decoder tensors, and native CBs must still be measured.

Removing only one class is insufficient:

- **Native outputs only:** persistent stats plus all semaphores still need 2,188,288 B, so construction still cannot fit.
- **Stats only:** outputs plus semaphores need 1,401,856 B, leaving 58,432 B. One native input shard (10,240 B) and temporary stats (16,384 B) leave 31,808 B before native scratch. Three unbacked CBs alone require **73,728 B/core** with the actual FP32 accumulation/gamma: `c4=8×4096`, `c6=5×4096`, `c11=5×4096`. See `rms_allgather_program_factory.cpp:238–248,301–325,499–505,557–559`, and `multichip_decoder.py:181–186` for FP32 configuration. Therefore stats-only removal can move the failure from construction to native program capacity; it is not a usable repair by itself.

### Asynchronous and trace lifetime contract

Host return and phase-counter updates do not establish device completion. The proposed temporary allocation is valid only with the existing complete, serialized **CQ0** operation chain, including the activation gather. The source-level ordering argument is:

| Boundary | Required state / evidence |
|---|---|
| Norm statistics exchange | `rms_writer.cpp:181–207` waits rank arrivals, resets its phase semaphore, drains multicast writes, and finishes fabric closure; compute is then released. |
| Norm stats consumption | `rms_compute.cpp:172–244` consumes gathered stats before normalized output can be supplied to the following gather. |
| Activation gather | `all_gather_async/device/kernels/minimal_default_reader.cpp:132–169,209,388–425` transfers the local slice and waits for every expected remote slice. Gather completion establishes that every peer entered the gather after its preceding norm finished consuming stats. |
| Next local write/reuse | Dispatch waits for prior worker completion (`tt_metal/impl/dispatch/kernels/cq_dispatch_subordinate.cpp:628–663`, plus the FDS branch at `594–602`). Subsequent scratch writes must remain in CQ0 with the same participating subdevice. |

Thus next-normalization reuse follows a distributed completion dependency, not merely Python variable destruction. **Do not generalize this to consecutive isolated RMS calls, another command queue, concurrent callers, or externally supplied writes.** A persistent standalone activation gather skips its startup barrier (`all_gather_async_default_program_factory.cpp:196`); the argument uses gather completion, not an assumed entry barrier. Retain separate phase semaphores and do not add ad hoc resets after failed calls.

The cache does not silently retain temporary RMS tensors: its factory resource descriptor populates only programs (`rms_allgather_program_factory.cpp:1253–1260`); the adapter moves those descriptors into programs and retains binding metadata (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:597–626`). Dynamic CB configuration holds a raw Buffer pointer, not shared ownership (`tt_metal/impl/buffers/circular_buffer_config.cpp:230–235`). On each cache hit the RMS override patches current semaphore, stats, gamma, input and output CB addresses (`rms_allgather_program_factory.cpp:1295–1345`).

Trace capture snapshots runtime arguments and current dynamic CB addresses (`tt_metal/impl/program/dispatch.cpp:3258–3285`); capture command generation copies the captured CB payloads into commands (`:2984–2995`). Later cache rebinding does not edit old captured command bytes. Temporary buffers can be released in the host capture walk and their recorded addresses reused during a later ordered replay. RMS scratch storage is not retained by these cached programs or trace nodes: a trace node retains the ProgramImpl (`tt_metal/impl/trace/trace_node.hpp:33–42`), whose dynamic buffer references are raw pointers. Keep persistent owners, gamma, gather buffers and semaphores alive through all referring trace lifetimes; do not retain post-capture temporary allocations across replay.

The allocation tracker records allocations made while traces are live and removes their unsafe IDs on deallocation (`tt_metal/impl/allocator/trace_allocation_tracker.cpp:183–215`). Keep tracking enabled with `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`; warm every changed signature/worker row before the first retained trace. Do not suppress the tracker or acknowledge live allocations to make a test pass. Temporary lifetime and multiple-trace safety still require runtime validation; source inspection is not a hardware pass.

## 4. Persistent arenas and explicit safe reuse alternatives

If private persistent stats/output storage is itself a required contract, use an explicitly owned reservation rather than pretending that a shard grid is an arena.

**One owning slab, partitioned by row:** reserve a common-address slab once, then create typed non-owning views with explicit addresses and actual shard grids. Use slot `layer_idx // grid.y` within row `layer_idx % grid.y`. A conservative slot occupies 53,248 B: two disjoint 16,384-byte stats intervals and two disjoint 10,240-byte output intervals. Reserve `ceil(64 / grid.y) × 53,248 B`. If `grid.y == 10`, as the earlier CPU audit assumes, seven slots require **372,736 B** once. The reported 110 banks alone does not verify that grid height. Different rows intentionally reuse the same address offsets but have disjoint physical storage; owners on the same row have separate slots. With ten rows, including the existing 91,136 B semaphore inventory gives 463,872 B of persistent address window before other tensors.

`LockstepSRAMArena` supplies a real owning MeshBuffer reservation (`tt_metal/impl/sram_arena/sram_arena_impl.hpp:76–95`), but typed fixed-address TTNN tensor creation needs native plumbing. Explicit-address MeshBuffers are non-owning (`tt_metal/api/tt-metalium/mesh_buffer.hpp:198–231`; `distributed/mesh_buffer.cpp:325–328`). Keep the sole reservation strongly owned until all views, in-flight work, and traces finish. An unreserved fixed address can overlap allocator tensors or native CBs. An owning reservation participates in program buffer/CB collision checks (`tt_metal/impl/program/program.cpp:2593–2599,2695–2702`). Address-only uniqueness tests must become physical `(rank, core, address interval)` ownership checks for this design.

**One range-lockstep arena per owner or row is not enough:** its implementation sets the range-lockstep flag and creates an ordinary MeshBuffer (`sram_arena_impl.hpp:97–122`), which still reserves through allocator ID 0. The flag narrows conflicts with per-core allocations; it does not overlay separate ordinary allocations. Nor does a trace allocation-tracker scope create or reserve an arena.

**Explicit shared scratch:** one stats/output pair on one common worker row costs 26,624 B; two phase pairs cost 53,248 B. Sharing such a reservation can preserve 64 owner objects and separate semaphores, but changes their storage-isolation contract. Bind scratch as a distinct strongly owned resource, prohibit overlapping calls/CQs, retain it across all trace lifetimes, and prove last-reader/last-writer completion before reuse. Do not implement this by sharing an entire owner or mutating `owner_id` on a shallow copy. Per-row shared scratch also needs either an explicit slab or an honest sum of ordinary row reservations.

**Subdevices:** true disjoint subdevice allocators can provide separate address accounting (`tt_metal/impl/sub_device/sub_device_manager.cpp:263–338`); they require an explicit subdevice ID when allocating a buffer (`buffers/buffer.cpp:354–371,507–512`). Existing tensor `empty` does not infer that ID from its shard grid. Partition/program mapping and manager lifetime make this a larger repair.

**HYBRID per-core allocation:** this genuinely uses bank-specific allocator IDs (`allocator.cpp:157–191`), but is not a flag-only repair here. RMS minimal has no per-core allocation support declaration; the runtime rejects unsupported per-core input tensors (`ttnn/api/ttnn/device_operation.hpp:568–574`; `ttnn/core/device_operation_detail.cpp:236–254`). Its stats binding also uses a single buffer address. An operation-level address audit and changes would be needed.

## 5. Ownership, tests, and follow-up validation

The current shallow-clone path correctly installs a new owner before executing the clone; the model identity assertion prevents reuse of the original owner. No accidental shared-owner leak is needed to explain the OOM. The two prototypes intentionally remain live. Precreated logical B1..32 views share backing storage and do not multiply device capacity by 32.

Tests must preserve real independence while changing the scratch contract. In particular, `tests/check_optimized_multichip_evidence.py` currently requires eight persistent minimal tensor records; `check_multichip_stack_resources.py` checks stable/unique persistent addresses; `probe_fused_decode_norm_owner.py` checks persistent native output views and hashes. Update those assumptions for the selected repair. Keep model/gamma rebinding, independent semaphores, phase poisoning, per-rank correctness, changed-input replay, and tracker positive controls. A receipt must not claim eight independent persistent tensors after making four of them temporary.

Suggested parent follow-up, **not performed here**:

1. A static allocator-window budget check should reject the original 64-owner inventory and distinguish payload, reservation, semaphore minimums, and peak native CB needs. Recheck the exact construction ledger after the repair.
2. Confirm all 64 distinct owners construct at B1 and B32; sample memory after construction and at warm/capture boundaries. Confirm temporary scratch actually releases and no native CB collision replaces the constructor OOM.
3. Warm all changed signatures and offset rows. Exercise complete input/post pairs with real independent gammas and changed inputs at B1/3/17/32, serialized queued calls, and multiple retained batch traces. Preserve exact replay/state checks and unchanged numerical tolerances.
4. Run the existing 64-owner trace test with allocation tracking and program-cache tracking enabled, no acknowledgments, and the positive control. Then verify full decoder/context/stress capacity and warmed latency. The resource test shares representative weights and inputs; it cannot establish full-model numerical or full-weight capacity by itself.

## Other potential issues / limits of the diagnosis

- Native offset-row routing and terminal-traffic correctness remain hardware validation obligations, but cannot explain this recorded pre-execution allocation failure. No independent offset bug was established here.
- DiT uses a different stats/storage implementation. It avoids these particular persistent minimal L1 tensors when gather is DRAM, but this failure is not evidence that DiT has passed 64-owner execution, CB capacity, or trace validation. Switching families is an explicit alternative configuration, not the minimal repair to this family's ownership/storage contract.
- The proposed ephemeral path relies on serialized full normalization plus activation-gather completion. If callers later bypass the gather or overlap queues, re-establish the lifetime protocol; owner identity and a Python phase counter alone are insufficient.

## Draft claim review

After drafting, the causal claims were rechecked against allocator placement, semaphore allocation minimums, clone ordering, native output creation, cache retention/rebinding, CB sizing, and gather completion. The exact arithmetic was rerun without TTNN imports. The primary finding explains the failing operation, requested size, allocated/free counts, successful two-prototype sample, and absence of prefill/capture. Unsupported alternatives—fragmentation, view allocation multiplication, a clone leak, or an already-running native kernel failure—are not headline findings. Repair effectiveness and hardware trace correctness are explicitly unverified.
