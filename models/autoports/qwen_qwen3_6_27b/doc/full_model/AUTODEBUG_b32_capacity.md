# AutoDebug: B32 preparation retains a large warmup working set and fails a fragmented all-position head allocation

**Leading finding:** the reported failure is an avoidable all-position diagnostic-head allocation during preparation, under high resident pressure and fragmented DRAM. It is not evidence that the preserved all64/TP4 model, B32 slots, virtual context262144 and physical8224-page pool are intrinsically infeasible. The exact failed request is **32 ×192 ×62080 ×2 =762,839,040 bytes/device** of BF16 logits. The inferred failing logical length is161, padded to192: the preceding progress marker is160, and the final MMRS compile arguments corroborate161. Production last-position prefill needs a much smaller **127,139,840-byte** head output for this batch.

Two source-established contributors compound this peak: preparation keeps **1,342,177,280 bytes/device** of distinct head-concat inputs alive across its entire shape loop, and the preceding B1 full preparation is expected to leave **2,206,203,904 bytes/device** of persistent CCL buffers across64 layers. Native cached workloads also own additive DRAM kernel binaries. Their byte total is not in the failing artifact: **do not label the remaining approximately10.78GB inventory gap as a proven native leak**.

**Smallest recommended repair boundary:** retain the accepted B32 decoder execution and its policies; bound diagnostic-head work and shorten concat-input lifetime in `tt/bounded_prefill.py`, with a small host-readback lifetime adjustment in `tt/generator.py`. First measure native cache ownership in a separate pre-trace diagnostic. No complete full1..512 capacity fix is demonstrated by this inspection.

Inspection only, 2026-11-14. No devices, target execution, builds, production edits, resets, or AutoDebug relaunch. Only `./AUTODEBUG.md` was written; the previous root report concerns an older stage and is superseded here as requested. Paths prefixed `Q/` below mean `models/autoports/qwen_qwen3_6_27b/`; other paths start at the repository root. Native CCL paths abbreviated with `.../` are under `ttnn/cpp/ttnn/operations/`.

## Direct observations and provenance

- `Q/doc/full_model/commands.json`, row `slots32_all64_pool`, records all64, batch32, prefill33, context262144, physical-pages-per-slot257, full preparation, trace region100000000, profiler off, allocation tracking on, and program-cache allocations included. These are evidence, not instructions executed by this investigation.
- `Q/doc/full_model/slots32_all64_pool.log:21-85` loads all64 layers and reports8 banks with4,259,840,384 general-DRAM bytes/bank. Loaded allocation is1,732,213,760 bytes/bank, or **13,857,710,080 bytes/device**.
- The log's first preparation pass completes every32-token marker through512 (`:86-102`). The second completes through160 (`:455,1072,1777,2460,3165`), then fails (`:3188-3223`). Its stack is `probe_full_slots.py:49 → generator.py:440 → bounded_prefill.py:354 → model.py:139 → MatmulDeviceOperation::create_output_tensors`.
- The error reports a762,839,040-byte request distributed as95,354,880 bytes/bank, with4,081,165,120 allocated,178,675,264 free, and51,515,008 largest free block **per bank**. The request was not allocated; do not add it to the reported live bytes.
- The failed run does not finish the JSON-writing path in `probe_full_slots.py:106-122`; its log has no final owned-CCL inventory, native-byte inventory, or successful full preparation/capture receipt. Values below explicitly distinguish measurements from source estimates.

Current Git HEAD is `7be8c4348666622686a6bdf5efa72e73ecf4c9b0`. The full-model Python files are untracked in this prepared workspace, so HEAD alone does not pin them. Current SHA256 values are:

| Source | SHA256 |
| --- | --- |
| `Q/tt/model.py` | `1e4bad696340d1eef9ca9c289452661ed9cc9e2cbe101c60b9ff06f0c8a32254` |
| `Q/tt/generator.py` | `5f6af66b5d71d1566896547718ad58b68316003d35c2d11ccf6a5470e191e392` |
| `Q/tt/bounded_prefill.py` | `8269b97febeefd0d3bbecb7a69042662a24ff3d850be4ed8109cb3a26caa866b` |
| `Q/tt/multichip_decoder.py` | `2d4ba9dd20c86f6ae9d710c5fe79c259bdd66fe6007b6af4be5437ca01554c0b` |
| `Q/tests/probe_full_slots.py` | `6916fce62d5c989a8a6f8dc9a63f7d0524efd569caf06c6ffa1de83cfecd707f` |

The current call sites match the failure stack, and the failed byte count independently matches the derived tensor. The artifact lacks a completed source-hash manifest, so exact historical source identity beyond this correspondence remains unproved.

## 1. Exact trigger: diagnostic logits and concat inputs overlap unnecessarily

`Q/tt/generator.py:438-440` prepares B1 before uniform B32. Both helpers allocate all512 ID/position shapes in advance (`bounded_prefill.py:99-104`) and prepare logical lengths1..512 (`:308-310,336`). Three representative layers are used in this model: layer0 GDN, ordinary GDN at layer1, and attention at layer3. Layer0 is distinct because final-norm weights are added only to `model.io == layers[0]` (`model.py:62,81-83`), and the representative signature includes the complete weight/shard-norm inventories (`bounded_prefill.py:264-276`). This is representative execution with all64 layers' weights and states still resident, not64 full decoder executions per length.

For each logical length, preparation executes representative decoder paths, warms an unaligned attention-head gather/step, warms concat, then calls both `terminal(result)` and `terminal(result[:, -1:, :])` (`:339-355`). The first is the failing all-position head. `model.py:73-79` shards the head's vocabulary dimension across TP4; `:137-145` gathers/normalizes hidden rows and requests BF16 DRAM output. For logical `[B,S,1280]` hidden, the output is logical `[B,S,62080]`, with the sequence independently tile-padded in each batch row.

| Head output or warmup storage | Bytes/device |
| --- | ---: |
| B1 all-position logits at S512 | 63,569,920 |
| B32 all-position logits at S160, padded160 | 635,699,200 |
| B32 all-position logits at S161, padded192 | **762,839,040** |
| B32 all-position logits at S512 | 2,034,237,440 |
| B32 one-position logits, sequence padded32 per user | **127,139,840** |
|512 distinct B32 concat inputs, each logical `[32,1,1280]` BF16 | **1,342,177,280** |

The logical length161 is inferred from the sequential loop and the final MMRS compile arguments (`slots32_all64_pool.log:3167,3178`), not an explicit failure-length print. Those reader arguments encode a161-row tile geometry; the factory derives `Mt` from flattened input rows divided by32 (`ttnn/cpp/ttnn/operations/experimental/ccl/matmul_reduce_scatter_prefill/device/matmul_reduce_scatter_prefill_program_factory.cpp:383-389,535`). The requested byte count alone identifies padded192 and would also match logical lengths162..192.

The final table row is not512 × a logical one-row payload. Each user's one-token sequence occupies a full32-row tile: each clone allocates2,621,440 bytes. `concat_inputs` is constructed once at `bounded_prefill.py:348-352` and remains strongly referenced until `prepare()` returns. The expression results of `terminal()` are discarded, so there is no Python list retaining every previous logits output. The retained concat list, the current value/result/rope/one-row tensors, and the current terminal normalization are nevertheless live when the next output is allocated. Terminal logits do not need the512 distinct concat inputs. A full512 concat output itself is logical `[32,512,1280]` and costs41,943,040 bytes; its inputs carry independent tile padding. Native concat sums logical dimensions (`ttnn/cpp/ttnn/operations/data_movement/concat/device/concat_device_operation.cpp:280-297`) and batches large arities through temporary outputs (`:441-484`).

This predicts the important contrast: B1 head outputs are small and its concat inputs total41,943,040 bytes; B32 pays32 times those padded batch costs and crosses a sequence-tile boundary at161. Actual generation already calls the head only on the last position unless `return_all_logits=True` (`bounded_prefill.py:436-438`). The failing command has not reached its requested33-token prefill or decode; preparation performs a broader workload than the request.

**Confidence:** exact failed tensor, padding, lifetime overlap and unnecessary generation-preparation peak are established by source and stack. Eliminating this peak has not been shown to make every later prepared shape fit.

## 2. Persistent CCL: real ~2.206GB ownership, mostly fixed before B32

`Q/tt/multichip_decoder.py:469-494` persists collectives only when packed rows are<=32. Its key includes reduction flag, exact logical shape, dtype and memory config. Thus B1 logical M1..32 retains32 separate signatures even though their storage all pads to32 rows. These buffers belong to each layer's `_ccl_buffers`, not a temporary local. `bounded_prefill.py:279-297,359-360` allocates matching private resources on all other layers after the representative warmup. Per-layer ownership is intentional; sharing mutable scratch across layers would change the existing trace/resource contract.

Expected full B1 inventory under the selected BF16 CCL/default sharded-residual policy:

| Per short-M signature | Physical padded shape | Allocated bytes/device |
| --- | --- | ---: |
| RS intermediate | `[2,1,32,5120]` | 655,360 |
| RS output | `[1,1,32,1280]` | 81,920 |
| Hidden AG output | `[1,1,32,5120]` | 327,680 |
| Statistics AG output where used | `[1,1,32,128]` | 16,384 |

GDN's distributed norm uses statistics plus hidden gathers; attention's B1 `gather_norm="decode_or_single"` uses a hidden gather directly (`multichip_decoder.py:592-633`). Statistics output width32 per rank is defined in `ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/layernorm_pre_all_gather_device_operation.cpp:91-101`; gathering four ranks gives width128. Eight-bank minimum allocation makes its8,192-byte payload cost16,384 bytes.

- 48 GDN layers ×32 ×(655360+81920+327680+16384) =1,660,944,384 bytes.
- 16 attention layers ×32 ×(655360+81920+327680) =545,259,520 bytes.
- Total **2,206,203,904 bytes/device**, expected 5,632 signatures and 7,680 distinct tensors, before later sampler variants. These counts are source-derived, not a recovered runtime inventory.

At uniform B32/S1, packed M32 hidden/RS signatures already exist from B1. The full-attention representative can add a single16,384-byte statistics buffer. For every uniform S>1, packed M=32*S>32 disables this persistence. Therefore **progressive B32 S1..161 residency is not growth of this Python persistent-CCL map**. The approximately2.206GB fixed ownership still matters to available space and potentially the allocation layout.

The lowered operations honor supplied storage: `.../experimental/ccl/all_gather_async/device/all_gather_async_device_operation.cpp:281-287` returns the supplied persistent output, and `.../reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_op_device_operation.cpp:256-279` adopts the supplied intermediate/output. No hidden second DRAM copy is established there. Fused MMRS persistent resources are one set per mesh: four L1/L1_SMALL semaphores and an L1 mailbox (`.../matmul_reduce_scatter_prefill/device/matmul_reduce_scatter_prefill_program_factory.cpp:960-999`), not a per-shape DRAM pool.

The old69,206,016-byte one-signature CCL estimate is therefore unsuitable for full1..512 preparation. `Q/doc/full_model/capacity_accounting.md:80-82` already identifies this category as incomplete; treat it as a lower inventory, not a promised capacity admission result.

## 3. Native programs retain real DRAM; the missing total needs measurement

This is separate from Python `_ccl_buffers`:

1. Program-cache entries own `CachedMeshWorkload`/`AdaptedCachedMeshWorkload`, which own `MeshWorkload` (`tt_metal/api/tt-metalium/program_cache.hpp:66-86`). The cache inserts distinct keys without eviction (`:145-169`).
2. `MeshWorkloadImpl::load_binaries`, `tt_metal/distributed/mesh_workload.cpp:183-219`, takes the maximum coordinate-program binary-data size, allocates one replicated DRAM `MeshBuffer` with2048-byte pages and `bottom_up=false`, and stores it in the workload's `kernel_bin_buf_`. It remains live while the cached workload remains live. The explicit-address program views at`:225-246` reference that same allocation; do not count them again.
3. For raw maximum binary bytes `K` in one cached workload, eight-bank consumption is `8 * ceil(ceil(K/2048)/8) * 2048`. A new cached shape can therefore add persistent device DRAM even when all Python activation results have died.
4. `ttnn/api/ttnn/device_operation.hpp:630-646` allocates outputs before launching/cache processing. The observed failed request is an activation output, **not a binary allocation**. Earlier cached workloads may still contribute to the state in which it fails.

The inspected AG/RS/RMS workload descriptors populate programs without parking their previous input/output tensors in owned descriptor-buffer vectors. Global CB bindings alone are raw buffer references, not ownership proof. The current AG/RS/RMS hashes also exclude semaphore addresses, so different layer semaphore identities do not by themselves create64 copies of every native program. These facts rule out two tempting but unsupported multiplication/leak stories.

The failed log contains no native retained-byte sum. Another reduced-model B1 receipt (`Q/doc/full_model/trace_contract_final.json:210,743`) has23,705 constant program-cache entries; it demonstrates a large preparation repertoire but does **not** measure this all64/B32 run or its binary footprint. Entry counts, JIT compiler hits, host build-cache files and device DRAM bytes are different quantities.

`TT_METAL_TRACE_ALLOC_TRACKING=1` and `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0` do not supply a full pre-capture allocation ledger. Allocation recording is conditional on tracked traces (`tt_metal/impl/allocator/allocator.cpp:312-314`, `trace_allocation_tracker.cpp:183-193`). This failure precedes generator trace capture (`generator.py:434-466`). Native ownership is a concrete budget term, but assigning the entire unaccounted total to it remains an interpretation requiring the experiment below.

## 4. Allocator verdict and reconciled resource ledger

The allocator error is internally consistent. `tt_metal/impl/allocator/bank_manager.cpp:481-511` computes bytes/bank and reports the underlying free-list statistics. `algorithms/free_list_opt.cpp:424-451` sums allocated and free blocks; adjacent free blocks coalesce (`:267-319`), but live allocations are not relocated.

| Observed allocator quantity | Bytes/bank | Bytes/device across8 banks |
| --- | ---: | ---: |
| General DRAM | 4,259,840,384 | 34,078,723,072 |
| Allocated before failed output | 4,081,165,120 | **32,649,320,960** |
| Total free | 178,675,264 | **1,429,402,112** |
| Largest free block / corresponding striped capacity | 51,515,008 | **412,120,064** |
| Failed request | 95,354,880 | **762,839,040** |

Total free exceeds the request; the largest usable contiguous range does not. With ideal compaction this one request would leave666,563,072 bytes/device free. The immediate error is therefore **fragmentation at high occupancy**, rather than proof of insufficient aggregate capacity for this one tensor. It does not prove that ideal compaction would sustain subsequent operations or S512.

For P8224, the existing capacity note's source model and measured loaded baseline give this partial reconciliation. The rows are additive; the failed output is excluded.

| Live category | Bytes/device | Evidence type |
| --- | ---: | --- |
| Loaded model including embedding/head/shared RoPE/setup | 13,857,710,080 | Measured before generator construction |
| Physical K/V plus one deduplicated reset-zero source | 2,362,195,968 | `287232 *8224`, source model |
| B32 fixed state/helper/table/logit/remap categories | 2,012,768,768 | Capacity note`:59-72`, source model |
| Five wide penalty arrays allocated by common sampling setup | 87,457,792 | Capacity note`:76`, source model |
| Subtotal before CCL, native growth and current temporaries | **18,320,132,608** | Incomplete lower inventory |
| Expected complete B1 persistent CCL | 2,206,203,904 | Derived above |
| Uniform-preparation concat inputs | 1,342,177,280 | Derived above |
| Known partial total at failing phase | **21,868,513,792** | Mixed observation/source inventory |
| Unassigned remainder to observed allocation | **10,780,807,168** | Includes native growth and other live tensors/controls |

This remainder is **not** all fragmentation: free-list fragmentation describes the arrangement of the free1.429GB, whereas the32.649GB statistic counts allocated blocks. Nor is it an isolated measured binary total. Current decoder/normalization/concat outputs, rope, small controls, representative scratch, added stats and programs must complete the ledger. The failed log cannot uniquely partition them.

The pool is genuinely8224 pages: `probe_full_slots.py:39-43` uses a permutation of32*257 physical IDs in a virtual `[32,8192]` table; `MultichipDecoder.create_paged_state` sizes storage by max ID+1 (`multichip_decoder.py:766-780`). It does not allocate dense B32*262144 K/V. Helpers alias the existing attention K/V while owning FP32 GDN worksets (`bounded_prefill.py:105-115`). The general-DRAM total already excludes reserved trace storage (`allocator.cpp:54-94`); do not subtract the100MB command region again. No live generator traces have yet been captured.

Host calls enqueue device work; reaching a progress print is not a device-completion proof. However, the inspected tensor→storage→mesh-buffer→buffer destruction chain frees host allocator entries directly (`ttnn/core/tensor/tensor.cpp:121-151`, `storage.cpp:234-241`, `tt_metal/distributed/mesh_buffer.cpp:424-483`, `tt_metal/impl/buffers/buffer.cpp:767-800`, `allocator.cpp:318-346`). An unbounded asynchronous deferred-free queue is not established by this source. Synchronization remains necessary for clean measurements and safe cache-clear controls.

## Minimal repair proposal, preserving the optimized decoder

Make the following focused change in the parent implementation, then verify it. This is a proposal, not an applied or hardware-proven fix.

1. **Separate concat signature warming from terminal warming.** Once a full-attention one-row result exists, call a small helper that owns distinct clones, executes each required concat arity, and returns. The helper's clones/results must die before terminal and later decoder work. Retain the ordinary `_layer_chunk` result for terminal; do not overwrite it with concat solely for warmup. Preserve the distinct-input alias partition: `[one]*N` is explicitly a different concat signature (`bounded_prefill.py:348-352`). For full preparation this removes1.342GB of unnecessary overlapping B32 lifetime, without deleting required concat signatures. It does not remove concat kernel binaries.
2. **Prepare only the B32 one-position head; stream B32 all-position diagnostics through it.** Keep B1 all-position `terminal(result)` preparation and keep the existing last-position head for both batches. For uniform B32 `run(return_all_logits=True)`, execute the existing full decoder chunk, then iterate its logical positions using runtime `row_index` and `ttnn.gather(value,1,row_index)`, yield `terminal(row)` as `(absolute+i,1,logits)`, and consume/release each result. Existing preparation already warms gather for every `[B,S,1280]` source shape (`:344`) and a `[B,1,1280]` terminal shape (`:355`). Generator host assembly ignores interval labels, reads each yield and concatenates along sequence (`generator.py:585-602`); all-position device output is already rejected (`:553-554`). Thus public host diagnostics can retain `[B,S,V]` behavior without allocating a full B32 chunk's logits on-device.
3. **End yielded-logit lifetimes promptly at the host boundary.** Delete the host-only consumer's device `logits` reference after `_read_logits`, and drop the engine's previous yielded reference on resumption. Otherwise the last127MB one-position output can overlap allocation of the next one. Keep the existing sampled-prefill handling and returned-device ownership intact.

Do not merely delete `terminal(result)` while leaving the B32 diagnostic runtime path unchanged: that removes warm signatures and can cause late native allocations after traces exist. Validate the exact gather/slice output specs, normalization and terminal signatures; the equivalence is source-feasible but still needs cache-count/allocation-tracking and numerical checks, especially after resumed unaligned prefill. Per-position norm/head math is independent by row, but a different matmul shape may select different tiling, so numerical equivalence must be measured with existing tolerances.

Preserve accepted attention HiFi4, projection/activation precision, BF8 KV, BF16 published/FP32 live GDN recurrence, packed corrections, private persistent CCL, residual sharding, uniform B32 decoder execution and virtual context. Reducing physical pages, disabling persistence, clearing the native cache in production, or lowering context on this first error is not justified.

The proposal removes proven excess peak and diagnostic repertoire. Native binaries for all remaining decoder shapes can still exhaust capacity later. If measurements identify them as the dominant remaining term, the next repair boundary is shape/signature reuse or bounded program residency with an explicit capture-safe preparation contract; a blind cache clear invalidates that contract and is not a finished repair.

## Reduced discriminating experiments for the parent (not executed)

### A. Existing CLI: exact head shape with a small repertoire

Run the same environment and hardware configuration as the failing manifest, but use:

```bash
python -u -m models.autoports.qwen_qwen3_6_27b.tests.probe_full_slots \
  --batch 32 --all-layers --prefill-length 161 \
  --context 262144 --physical-pages-per-slot 257 \
  --output models/autoports/qwen_qwen3_6_27b/doc/full_model/slots32_all64_diag161.json
```

**Omit `--full-prepare`.** The test then prepares only `{1,161}` for both helpers (`probe_full_slots.py:45-49`), while still allocating all64 weights, all states,8224 physical KV pages, the full virtual context, and512 concat clones. Its head at161 requests the exact762,839,040 bytes. This is an intentionally reduced diagnostic, not an acceptance receipt for full1..512.

- A pass disproves an unavoidable S161 head peak on this pool and implicates accumulated preparation resources/layout. Both short-M CCL ownership and native programs shrink in this contrast; it does not distinguish them by itself.
- A failure must be compared by phase/request, not exit code alone. If the same output fails, inspect free/largest and ownership there. A different earlier failure is a different observation.

### B. Isolate native cache ownership before any trace

In a separate diagnostic process with the same all64/B32/C262144/P8224 setup, stop after B1 full preparation or after a chosen uniform prefix such as1..160; do not call the remaining generator capture path. Drop diagnostic local outputs, synchronize the mesh, and record `model.memory_view()`, `ccl_inventory(model)`, program count and DRAM block table. Keep model, cache, helpers and all Python `_ccl_buffers` alive. Then call `mesh.clear_program_cache()`, synchronize again, and take the same snapshots.

`MeshDeviceImpl::clear_program_cache` only clears the map (`tt_metal/distributed/mesh_device.cpp:1333-1336`); synchronization is not implicit. The allocated-byte drop measures released cache-owned storage collectively. It does not by itself identify every factory's resource type. CCL/helper inventories should remain unchanged; the MMRS global L1 resources should also survive. Measure the largest-block change as well as total bytes to distinguish pressure from placement. Before any subsequent execution/capture, restore the required warm programs in this diagnostic or exit the process. Do not reuse an already-captured trace for this control.

For the closest failure-matched sample, a diagnostic hook immediately before the S161 terminal can retain the exact concat/current-hidden set while comparing snapshots around a synchronized cache clear, followed by the same terminal. This distinguishes release of native ownership from simply letting1.342GB of concat inputs go out of scope. It is a test-only control, not an edit to make in production.

### C. Attribute the peak repair, then close the full contract

With a reduced preparation set around `{1,160,161,512}`, collect synchronized allocated/free/largest and owned-CCL/program counts before and after concat warming, decoder warmup, terminal normalization/output and helper return. Compare original versus shortened concat lifetime and bounded diagnostic head independently. Check both B1 and B32 diagnostics, final-position equality, resumed unaligned prefill, slot/state preservation, unchanged program-cache count and strict allocation tracking after trace capture. Then rerun complete all64 B1/B32 full1..512 preparation and the original slot probe on8224 pages. Only that full result can establish that the proposed repair is sufficient and support a physical capacity statement.

## Other potential issues and rejected explanations

- **Persistent CCL growth at every B32 length:** contradicted by the packed-row<=32 guard. Its large B1 baseline is real; progressive B32 map growth is not.
- **Native cache retains all previous AG/RS/RMS activation outputs:** not established by the inspected descriptors. Retained binaries are proven; activation retention needs a specific owner.
- **Each layer's semaphore forces a separate native cache entry:** excluded by current AG/RS/RMS compile-attribute hash definitions. Private Python CCL tensor ownership still multiplies across layers.
- **Pure allocator defect or bank-count multiplier error:** reported totals and request match the allocator calculation and exact output shape. No defect was found in this exercised allocation/error path.
- **All intermediate temporaries from all64 layers are simultaneously live:** not the preparation control flow; representative execution and normal local lifetimes must be respected.
- **Warmup table aliases force dense KV or invalid full512 writes:** this command's pool has more than32*16 pages; current warmup tables use disjoint physical writes and are restored in `finally` (`bounded_prefill.py:315-322,361-366`). This does not explain terminal allocation failure.
- **Trace allocation tracking itself measures or pins the missing10.78GB:** no complete pre-capture ledger exists in these logs; trace capture has not started.
- **Full diagnostics are semantically unnecessary:** they remain a public host diagnostic API. The repair must bound their device working set while retaining behavior, not silently remove support.

## Post-draft claim review

Rechecked the failing stack, both progress sequences, the exact logits size and per-bank arithmetic, the persistent-row guard, representative signatures, concat ownership, native workload binary ownership and cache-clear lifecycle against source. A standard-library-only static check extracted `tensor_bytes` from the existing capacity model and asserted the762,839,040-byte output,2,206,203,904-byte CCL estimate,1,342,177,280-byte clones, P8224 KV/reset total, and free-versus-largest inequality. No TTNN/torch import or device work was used.

A fresh xhigh reviewer independently checked the report against source and found no material contradiction. Its logical-length qualification and fuller mesh-buffer deallocation citation were incorporated. Source hashes were rechecked at the end and match the table above.

The review deliberately leaves the10.78GB remainder unassigned, labels CCL counts as expected rather than measured, and separates the source-proven immediate failure from unverified sufficiency of the repair. No context/pool reduction or new supported-capacity claim follows from this report.
