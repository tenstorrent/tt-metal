# AUTODEBUG: TP4 fused CCL candidate stall

## Diagnosis

**The captured stall is explained by a concrete worker-coordinate ordering bug in fused all-gather setup.** With two links and two workers per direction per link, the CCL allocation crosses the Blackhole worker-grid row boundary. The factory assigns worker indices in allocation order, but serializes their synchronization coordinates in an independently sorted order. The worker acting as master waits on one core while every slave increments a different core. The first local-slice barrier cannot complete.

This explains the exact master/slave locations on all four captured devices, the all-gather readers waiting for local-write completion, and the matmul waiting for its first activation slice. It also explains why separate all-gather plus matmul passes: that path does not execute this fused worker barrier. **Reduce-scatter, output-block sizing, persistent-buffer allocation, and local/global K are not the cause of this captured stop.** They were inspected separately below.

Confidence is high in the source diagnosis and its correspondence to the prepared capture. No hardware was accessed, no reproduction was run, and no implementation was changed. Clearing this first defect has not been shown to make the entire fused decoder pass.

## Scope and evidence boundary

Inspection date: 2026-10-23. Checkout HEAD: `7c620da56e3475f64e8656d70cde0a9b6464bb20`, with pre-existing uncommitted autoport work. The implementation inspected is the working-tree candidate plus tracked CCL/matmul source. The untracked harness changed during this investigation; references to its synchronization boundaries below use recognizable statements rather than assuming its current line numbers are those of the failed run. The diagnosed ordering code and captured kernel stop-sites agree.

The AG worker factory, `worker_sync_utils.hpp`, and `core_coord.cpp` were checked byte-for-byte against HEAD. Inspected candidate SHA-256: `91864cc42ed7b1916c0797c9044d2f30c5178adba9b9c7f5f11a00a4bd5a8d3d`. Prepared `tt-triage.txt` SHA-256: `e3bf427c71b5dcd2d02367106b9e1e3f17d01df8d2b23a3278c9e9141d08496b`. These identify inspected artifacts, not an unavailable failed-run provenance manifest.

Direct observations:

- `fused_both_l0.log:20-21,29` records baseline prefill/decode completion and multichip `PREFILL_DONE`. Lines 30-45 contain matmul configuration warnings; line 46 is `Terminated`, not successful decode completion.
- `triage_fused/tt-triage.txt:271` identifies running op **434, `AllGatherMatmulAsyncDeviceOperation`**, preceded by op **433, `LayerNormPostAllGatherDeviceOperation`**, on four devices. Its operands are BF16 DRAM `[1,1,1,1280]` and BFP4 DRAM `[1,1,5120,4608]`. This identifies the linear-attention `mix` projection, before recurrence, output projection, or MLP.
- On every device, logical core `(4,6)` is the AG writer master waiting at `worker_sync_utils.hpp:72`; `(5,6)`, `(10,6)`, and `(0,7)` are slaves waiting at line 108. Their caller is `minimal_default_writer.cpp:485`. The corresponding readers wait at `minimal_default_reader.cpp:394`. See capture lines 51-58, 117-124, 183-190, and 249-256.
- The matmul in0 sender waits at `MatmulOpReceiver::update_current_block_start_tile_id`, `worker_sync_utils.hpp:253`; see capture lines 8, 74, 140, and 206. Other matmul cores wait for multicast/input tiles; weight readers back up in `cb_reserve_back`.
- Recorded Ethernet links are up; the shown CRC, uncorrectable-codeword, resend/drop, and `ERR_STAT` values are zero (`tt-triage.txt:275-290`). This supports investigating the local protocol first, without proving every hardware condition healthy.

Evidence limits:

- The capture does not contain semaphore values, CB counters, or actual runtime-argument arrays. `summary.txt` saying diagnostic scripts passed is not a substitute for their omitted data.
- The failed run has no result JSON/provenance hash or complete saved argv. Batch 1 and the operand dimensions are directly observed; the filename/request identify layer 0 and `--fused both`.
- Saved `sharded_l0.json` proves a sharded **33-token + 2-decode**, batch-1 pass. `workload_l0.json` proves **4096 + 128**, batch-1, **replicated**. The broader stated sharded 4096+128 result is not established by those two artifacts. Both saved controls use nonfused operations.

## Exact causal chain

Source aliases in this section:

- **AG workers**: `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_default_program_factory.cpp`.
- **Sync**: `ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp`.
- **Fusion setup**: `ttnn/cpp/ttnn/operations/ccl/ccl_op_fusion.cpp`.

### 1. Allocation order and synchronization order diverge

The candidate supplies `num_links=2`, `num_workers_per_link=2`, and offset `(0,6)` (`tests/multichip_fused_candidate.py:37-44`). This argument becomes workers **per direction per link**. AG workers:815-838 assigns two directions and one mux core per direction when there is more than one worker. Therefore the allocation is:

`2 links × 2 directions × (1 mux + 2 workers) = 12 cores`.

For the captured eleven-column worker grid, allocation uses `(0..10,6)` and `(0,7)`. `choose_worker_cores` supplies the ordered core vector (`ttnn/cpp/ttnn/operations/ccl/ccl_common.cpp:514-538`). AG workers:914-938 consumes it in link/direction/worker order:

| Link | Direction | Mux | Workers in assigned-index order |
|---|---|---|---|
| 0 | 0 | `(0,6)` | `(1,6)`, `(2,6)` |
| 0 | 1 | `(3,6)` | `(4,6)`, `(5,6)` |
| 1 | 0 | `(6,6)` | `(7,6)`, `(8,6)` |
| 1 | 1 | `(9,6)` | `(10,6)`, `(0,7)` |

The factory preserves that allocation order in `sender_worker_core_ranges`, a vector (AG workers:935,958), and uses it when assigning runtime arguments (1208,1236-1237). It supplies index `worker + link * num_workers_per_direction` (1288-1298,1359-1363).

However, it separately stores direction-specific workers in `std::set<CoreRange>` (905-906,930-933), then calls `corerange_to_cores(set, ..., true)` to obtain the synchronization coordinate list (1007-1014). **The `true` argument does not globally sort ranges in row order.**

- `CoreRange::operator<` compares start coordinates (`tt_metal/api/tt-metalium/core_coord.hpp:104-107`).
- `CoreCoord` aliases `tt_xy_pair` (same header:31), whose comparison is **x first, then y** (`tt_metal/third_party/umd/tt-umd/api/tt-umd/types/xy_pair.hpp:32-34`).
- `CoreRangeSet(std::set)` preserves set iteration order; its span/vector construction preserves insertion order (`tt_metal/common/core_coord.cpp:158-165`).
- `corerange_to_cores` preserves range order and applies `row_wise` only inside each range (same file:606-627). These direction-specific ranges are singletons.

Thus direction 1 has two different orderings:

```text
Assigned worker indices:   [(4,6), (5,6), (10,6), (0,7)]
Serialized sync list:      [(0,7), (4,6), (5,6), (10,6)]
```

Fusion setup:39-42 translates the serialized list into each device's NoC coordinates without reordering it. Different physical harvested-core mappings across devices do not restore the lost correspondence.

### 2. The master waits on a different core from the slaves' destination

`OpSignaler` decides whether it is master by comparing **list[index]** with **list[0]**, rather than comparing the actual executing core with the master coordinate (Sync:44-62). Consequently:

| Actual executing core | Assigned index | Coordinate represented by that index | Branch taken | Action |
|---|---:|---|---|---|
| `(4,6)` | 0 | `(0,7)` | Master | Wait for 3 on its own local semaphore at `(4,6)` |
| `(5,6)` | 1 | `(4,6)` | Slave | Increment semaphore at list[0], `(0,7)`; wait locally |
| `(10,6)` | 2 | `(5,6)` | Slave | Increment `(0,7)`; wait locally |
| `(0,7)` | 3 | `(10,6)` | Slave | Increment `(0,7)`; wait locally |

The master uses `worker_sync_sem.wait(num_workers_to_sync - 1)` (Sync:69-73). All three slaves use the coordinates at list entry 0 for their increment (103-108). No slave supplies the expected increments to actual master `(4,6)`. Its local semaphore therefore cannot reach 3 in this protocol. The misplaced counter at `(0,7)` may also interact with that slave's equality wait/reset; the master's missing increments are sufficient to prove the deadlock without assuming a particular value survives there.

The capture shows **exactly these branches at exactly these logical cores on all four devices**. This is stronger evidence than a generic hang explanation or an assumed correspondence between unindexed samples.

### 3. Downstream waits follow directly

The direction-1 writer performs this barrier before publishing its local-write completion (`minimal_default_writer.cpp:481-488`). The direction-1 reader waits for that publication before issuing its first remote-slice signal (`minimal_default_reader.cpp:389-397`). The master also never sends the local-slice ready signal to matmul (Sync:75-90).

Matmul's activation sender consequently waits at Sync:253 before its first local K slice; its receivers and compute engines wait for A, while B readers fill their double buffers and then block reserving more space. These are consequences of the missing producer signal, not separate evidence of a BFP4 compute failure.

### 4. Why the visible sequence and controls agree

The candidate gates fusion on decode (`multichip_fused_candidate.py:25-28`). The supplied prefill completed through nonfused operations. The harness prints `PREFILL_DONE` after a blocking prefill replay and output comparison. It subsequently executes `eager=forward(); read(eager); restore()`: the full layer can enqueue/compile later operations before the host blocks reading eager output. Later AG/RS compilation warnings do not prove device progress beyond op 434.

Separate `all_gather_async` does not create the fused `OpSignaler` worker barrier. Thus the same two-link/two-worker nonfused transfer can succeed while the fused transfer stalls. The diagnosed geometry also explains why a one-link fused example need not encounter the defect.

## Shape, output-block, ring-signal, and buffer audit

### Effective operands and matmul blocks

All candidate operands are explicitly converted to interleaved DRAM and reshaped to rank 4. A is BF16, B is the existing local BFP4 weight, output is BF16; LoFi, `fp32_dest_acc_en=False`, and packer L1 accumulation are requested (`multichip_decoder.py:31-32,56`; candidate:14-15,30-38). There is no transpose or operand swap on this path. The mesh is `1×4`; the fused APIs use no cluster axis and the CCL factory linearizes that coordinate set.

For captured batch 1, logical M=1 pads to 32. Allowed matmul grid is `8×6`, but **actual work occupies eight columns and one row**. The Python program-config constructor defaults omitted `out_block_h/w` to `per_core_M/N` (`ttnn/cpp/ttnn/operations/matmul/matmul_nanobind.cpp:167-168`).

| Role | A K before fused op | Matmul K | Local physical weight N | `in0_block_w` | K iterations | `per_core_N` | `out_block_w` | Output width blocks/core |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Linear mix | 1280 | 5120 after AG | 4608; logical 4160 | 10 | 16 | 18 | 18, default | 1 |
| Full-attention mix | 1280 | 5120 after AG | 3584 | 10 | 16 | 14 | 14, default | 1 |
| Gate/up | 1280 | 5120 after AG | 8704 | 10 | 16 | 34 | 34, default | 1 |
| Out | 1536 | 1536, local TP K | 5120 | 6 | 8 | 20 | 10, explicit | 2 |
| Down | 4352 | 4352, local TP K | 5120 | 17 | 8 | 20 | 10, explicit | 2 |

Model construction establishes these local dimensions (`multichip_decoder.py:17,44-56,61-95`). In AG, the factory actually passes the **gathered** A to the matmul helper (`all_gather_matmul_async_program_factory.cpp:81-84`), so global K=5120 is intentional. RS uses each rank's local K contribution; replacing 1536/4352 with global 6144/17408 would change the computation incorrectly.

The fused matmul helper derives `num_blocks=K/in0_block_w` and output width blocks as `per_core_N/out_block_w` (`experimental/matmul/ccl_fusion/device/ccl_fusion_mcast_2d.cpp:115,146-150`; descriptor counterpart:1973,2022-2026). All listed divisions are integral. RS's `20/10` geometry is also used by the in-tree TP4 Blackhole cache regression (`tests/ttnn/unit_tests/operations/ccl/blackhole_CI/box/nightly/test_matmul_reduce_scatter_cache_bh.py:43-45,86-101`). That test uses M=128 and different precision settings, so it establishes a relevant configuration contract, not observed success for this candidate.

### Queue and signal ledger

For first mix, each AG direction has four workers. The local tile count is `32×1280 / 1024 = 40`, split into ten tiles per worker. There are no zero-work participants. The local writer group needs three slave arrivals and one master release; the coordinate mismatch breaks that first barrier.

For matmul, global K is 160 tiles and the K block is 10 tiles. Each local TP slice is 40 tiles, hence four blocks/slice. The first K pass would request these ready counters:

| K blocks | Slice order for rank r | Wait |
|---|---|---|
| 0-3 | local r | direction 1 >= 1 |
| 4-7 | r-1 modulo 4 | direction 0 >= 1 |
| 8-11 | r+1 modulo 4 | direction 1 >= 2 |
| 12-15 | r-2 modulo 4 | direction 0 >= 2 |

The source-level producer counts match these waits when local synchronization completes: TP4 routes two remote slices through direction 0, and one remote slice plus the local slice through direction 1 (`ccl_common.cpp:1817-1820`; `minimal_default_reader.cpp:158-165,389-397`; writer:481-488). The even-ring split-forwarding optimization is explicitly disabled for fusion (`minimal_default_reader.cpp:180`), so it does not introduce partial-slice ready signals here. There is no demonstrated first-pass credit deficit from TP4 itself. The current failure happens before the required local-slice ready signal on direction 1; direction-0 signals can occur independently.

At the matmul CB boundary, first-mix A capacity is two blocks of ten tiles; B capacity is two blocks of `10×18=180` BFP4 tiles. The activation producer has not passed its first slice wait. Compute waits for A and cannot drain B, so the B reader eventually blocks in `reserve_back(180)`. This accounts for the captured CB wait without treating it as a CB-size defect. Source: helper:146-178; in0 reader:265-273; in1 sender/writer:459-461; compute `bmm_large_block_zm_fused_bias_activation.cpp:338-340`.

RS sends its fused ready signal **after the whole matmul batch**, not once per output-width block (`reader_bmm_tile_layout_in1_sender_writer_padding.cpp:771-774`). Request batch B is packed into M by `[1,1,B,K]`; the matmul's outer batch count is one, so one release is expected. `ReduceScatterOpReceiver::wait_for_matmul_batch` waits for `batch_idx+1` (Sync:315-318). Thus two width blocks do not imply two required RS signals. No contrary count mismatch was established for out/down. The single active matmul row is explicitly supported by the `SKIP_MCAST` branch when there are no in1 receiver cores (descriptor helper:2512-2514).

### Persistent buffers and cache state

- AG `persistent_output_buffer=None` is supported. The op allocates a gathered activation tensor and uses it for matmul (`all_gather_matmul_async_device_operation.cpp:99-113,195-208`). The supplied startup barrier remains enabled when buffers are nonpersistent (AG workers:1309). If testing an explicit persistent AG buffer, its shape must be `[1,1,B,5120]`, not the matmul's N-dependent output shape.
- RS requires persistent intermediate/output buffers (`matmul_reduce_scatter_async_device_operation.cpp:174-175`). The candidate retains separate per-role buffers `[1,1,B,5120]` and `[1,1,B,1280]` (candidate:16-20,48-53). The full-width buffer is RS scratch; the op separately creates the local partial matmul output. These are the correct width domains.
- Current RS cache override reapplies tensor bindings **and** caller global semaphore arguments (`matmul_reduce_scatter_async_program_factory.cpp:158-172`). A missing-cache-refresh claim would contradict this code.
- Global AG, RS, and barrier semaphores from `TT_CCL` are distinct from the local fused-worker synchronization semaphore (`models/common/modules/tt_ccl.py:75-107`; Fusion setup:27-42). Changing global semaphore cycling or buffer persistence cannot correct the wrong local master destination.
- First eager execution is already stuck; there is no evidence that trace replay, buffer reuse across replays, or state restoration triggered this failure.

## Smallest repair and discriminating adapted configurations

**Source repair:** preserve direction-specific worker vectors in the same order used to assign runtime worker indices, and pass those vectors to `init_all_gather`. Alternatively, derive every worker index by finding its actual coordinate in the serialized list. Apply the same consistent mapping to both directional reader signalers and the local-writer signaler. The current AG factory does neither. Do not change global `CoreRangeSet` enumeration semantics to compensate for this caller's index/list mismatch.

A focused host-only regression should exercise workers spanning multiple rows and verify that each serialized index names its actual assigned core. The concrete failing forward list above is a useful case. Hardware confirmation should then keep the original geometry and demonstrate the barrier releases with the repair.

Before a source repair, these **proposed, unrun** candidate adaptations isolate the diagnosis. They are API settings to change in a diagnostic candidate, not currently supported command-line flags:

| Control | Concrete adaptation | Expected discrimination |
|---|---|---|
| Preserve two physical links | AG `num_workers_per_link=1`, `num_links=2` | Four CCL cores, no mux; direction-1 workers `(1,6),(3,6)` retain the same index/list order. Predicted to clear this barrier only. |
| Preserve two workers | AG `num_workers_per_link=2`, `num_links=1` | Six CCL cores; direction-1 workers `(4,6),(5,6)` stay in one row. Predicted to clear this barrier only. |
| Isolate each fusion | Run `--layout sharded --fused ag` and `--layout sharded --fused rs` separately | Original AG settings should retain the diagnosed stop; RS-only avoids the faulty fused AG path. A later RS-only failure would require its own capture. |
| Keep model math fixed | Keep local real weights, BFP4/LoFi, original K blocks, and buffer shapes in the first controls | Prevents precision or local/global-K changes from obscuring the geometric contrast. |
| RS output-block contrast | Original `per_core_N=20`, compare `out_block_w=10` with `20`, subblocks `1×1` | Changes two width blocks to one; useful only after AG is bypassed or repaired. It does not address the captured first AG barrier. |
| RS K-block contrast | Out: `6→4` (8→12 K iterations); down: `17→8` (8→17 iterations) | Tests different exact local-K decompositions without substituting global K or changing model dimensions. |
| Small-M contrast | Isolated RS M=1/padded32 versus M=128, retaining K=1536 or4352 and N=5120 | Compares one-row work with the nearby M128 test geometry; also changes RS chunk partitioning, so it does not isolate only matmul height. Full-layer batch changes introduce additional factors. |
| Persistence contrast | Explicit AG full-width BF16 buffer versus `None`, after worker mapping is safe | Checks reuse/startup-barrier behavior; not a repair for the coordinate mismatch. Keep independent role buffers alive through trace use. |

Moving only the CCL y offset does not eliminate a twelve-core allocation wrapping an eleven-column row. Changing matmul output blocks, BFP4 fidelity, or local/global K does not change the faulty barrier participants. The one-link nightly AG examples, including explicit two-worker cases in `tests/nightly/t3000/ccl/test_minimal_all_gather_matmul_async.py:364-473`, do not cover this crossing geometry.

For any future device check, synchronize/read immediately after the isolated fused operation before adding the rest of the decoder; then validate eager output, repeated cache hits, and trace replay separately. Predicted barrier release is not a numerical-correctness or whole-decoder acceptance claim.

## Other potential issues and adjudicated alternatives

| Candidate | Direct code fact | Relevance to captured failure | Smallest intervention/check |
|---|---|---|---|
| **Worker index/coordinate mismatch** | Proven divergent order; exact captured master/slave locations match | **Leading diagnosis; explains first missing signal and downstream waits** | Consistent ordered worker vectors or coordinate-based index lookup |
| Hardcoded AG transfer count | `all_gather_matmul_async_program_factory.cpp:58` fixes `num_transfers=4`; Sync:207-220 asserts `8 slices × 4 blocks = 16 blocks`, which is false for TP4 | Real TP4 assertion-contract defect, but `num_transfers/num_tensor_slices` do not drive live waits in `update_current_block_start_tile_id`. It cannot explain the observed local writer barrier | Derive supported even-ring transfer count (TP4 needs 2) and retain validation; do not claim this alone fixes the stall |
| Additional AG output blocks | `MatmulOpReceiver` state persists across outer output-block loops (in0 reader:259-268) | Potential later-use issue if configuration introduces multiple output blocks; current AG defaults have exactly one | Keep AG `out_block_h=per_core_M`, `out_block_w=per_core_N` while isolating this bug; inspect reset/reuse before adding outer blocks |
| RS output spec uses A | `matmul_reduce_scatter_async_device_operation.cpp:63-67` constructs RS specs from input A, not matmul output | A separate API/spec discrepancy: K-local/4 would be384 or1088 rather than required N/4=1280. Actual creation returns supplied output (76-79), and factory uses actual matmul/output tensors. No causal path to this earlier AG stop | Correct spec derivation; separately inspect graph/spec consumers |
| Allowed-worker-core warnings | CCL helper auto-populates omitted `allowed_worker_cores` (helper:1657-1668,3485-3496) | Warning explicitly provides a fallback; captured kernels ran. Does not change AG worker-coordinate list ordering | Explicit `CoreRange(0,0)-(7,5)` or proper program normalization is cleanup, not the diagnosed repair |
| Persistent lifetime/cache, BFP4/LoFi, RS split output blocks, hardware transport | No corresponding source contradiction shown for this stop | Cannot explain why all devices stop in the same first local AG worker barrier as precisely as the proven mismatch | Retain as downstream validation dimensions, not headline causes |

The hardcoded transfer count deserves a separate correction before assertion-enabled validation. For this TP4 first pass, the live ring wait sequence is nevertheless four slices, not eight: `num_transfers` is only consumed by the constructor's assertion, whereas the K loop and slice width determine the runtime wait sequence. Reporting the hardcode as the demonstrated deadlock would overstate its role.

## Review and validation performed

The first diagnosis was challenged against constructor semantics, global versus within-range ordering, NoC coordinate conversion, actual per-core stop-sites, and passing nonfused paths. Independent inspection confirmed that none of those layers repairs the mismatch. A cheap host-only Python calculation expanded the 12-core placement, index/list permutation, safe one-link/one-worker alternatives, and concrete K/output block counts. A CSV parse of the prepared callstack dump confirmed the same master/slave/self-write-done mapping on all four devices.

Claims requiring hardware remain explicitly unverified: successful execution after either workaround or source repair, numerical accuracy, throughput, subsequent fused RS behavior, and replay safety. The report's headline is the earliest source contract violation that explains the captured operation and its exact waits; secondary real-looking issues have not been promoted into that causal chain.
