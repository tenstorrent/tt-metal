# AutoDebug: batch32 precise GDN trace capture

## Finding

**The leading diagnosis is a trace stream exceeding the implementation's 32-bit byte-count limit after the precise projections expand the captured graph.** There is a definite source bug: the assembled host vector retains its full length, but allocation uses a byte count truncated to `uint32_t`. A large stream therefore gets an undersized device buffer, and uploading the full stream produces exactly `Region must be within buffer`.

The supplied artifacts do not record the assembled stream length. Thus the source defect is proven, while this execution crossing 4 GiB is a strong, sequence-compatible inference requiring one runtime measurement. Under the inspected single-device, single-stream path, ordinary sub-page padding cannot produce this mismatch below 4 GiB. Do not describe an exact batch32 byte count as measured.

**Smallest model-local candidate for the failing implementation:** change `PreciseGDNProjection`'s output block from 32 to 128, retaining both precise projections inside the existing complete batch32 trace. This reduces block-scaled graph construction fourfold without changing the reduction order along K. It is a workaround for the trace size ceiling, not a repair to shared trace code, and has not been executed here. Increasing `trace_region_size` cannot correct the allocation/upload disagreement. A concurrent external change during report finalization already implements the same intervention class with 256-output single-token blocks; see the revision note below.

**Evidence discrepancy:** the new JSON contains all 32 finite prefill results, but does not pass every existing prefill gate. Request 16 has contribution PCC below 0.995. The old recurrence-only result demonstrates successful capture/replay, but its overall `passed` field is also false. These accuracy facts do not explain the host trace-buffer bounds exception.

## Scope, revision, and direct observations

Investigation date: 2026-11-12. Checkout: `/home/ttuser/qwen38-flash/tt-metal`, HEAD `d7e788cd88199501a8546ddb7bc381a601af47a9`. The autoport directory is untracked. All source references below are relative to that checkout; model references start at `models/autoports/qwen_qwen3_8_flash_next/` unless stated otherwise.

Only source/log reads and Python-standard-library JSON, hash, and arithmetic checks were performed. No target TTNN import, target execution, hardware reproduction, source modification, or AutoDebug relaunch was performed. This report is the only file written. Proposed device experiments below are follow-ups, not execution evidence.

At initial inspection, the hashes of `tests/probe_batch.py`, `tt/functional_decoder.py`, `tt/decoder_components.py`, `tt/precise_gdn_projection.py`, and `tt/gdn_recurrence.py` matched the failing run's recorded hashes in `doc/functional_decoder/provenance/linear_precise_gdn_batch32.json:32,94-106`. The provenance names the same HEAD at line 115. The older run has matching probe, functional decoder, and recurrence hashes, a different component-file hash, and no precise-GDN-projection entry. These support the stated workload contrast, but do not reconstruct every historical component-file change. The provenance does not independently fingerprint the compiled shared trace library.

**Concurrent-change revision note:** during finalization, an external edit changed `tt/precise_gdn_projection.py` from the failing-run SHA-256 `5a3051f0478e57faafa601402ff30ed6aa6d6c11e580029576fc6baf39f37c19` to `39f0d3092d4804172091813626fb52607dc76a91940427800129dd96ded2f410`. The latter source stores BF16 256-output parent blocks and selects B=256 for Q=1, B=32 otherwise (`:31-35,48-69`). The other four reviewed files still matched the failing provenance at that check. All projection line references and B=32 workload counts below refer to the initially inspected, provenance-matched failing version. The concurrent B=256 implementation follows the graph-reduction workaround class, while retaining B=32 multi-token workspace behavior through device weight slices. Its runtime results were not evaluated in this inspection; this report does not claim that the changed candidate reproduces the old failure or passes acceptance. No external changes were overwritten.

| Direct observation | Evidence | Limit |
|---|---|---|
| Layer 0, batch32, BF16 experts, single-device mesh; trace allocation is dynamic | Failing provenance `:10-13`; `tests/probe_batch.py:29,87,95-102`; failing result `:2-3,707-709` | Batch32 is 32 independent requests, not one fused 32-token decode tensor |
| All prefill results were saved; there are no decode results | `linear_precise_gdn_batch32.json:38-706,709` | Results were collected before the final acceptance check |
| Capture body returns; failure occurs while ending capture and uploading trace commands | Log `linear_precise_gdn_batch32.log:30-52`; probe `:160-163` | Does not prove the captured decode executed successfully |
| The native backtrace is `populate_mesh_buffer -> enqueue_write_shard_to_sub_grid -> write_shard_to_device -> BufferImpl::view` | Same log `:45-52` | This is the trace-storage upload, not a projection output tensor write |
| Old SFPU-recurrence batch32 reached replay comparisons and recorded determinism | `linear_sfpu_batch32.json:1108-1126`; probe `:167-180` | `deterministic: true`, but `passed: false`; not an accuracy acceptance result |

The warm decode is followed by explicit `ttnn.synchronize_device(mesh)` at probe lines 154-156. The capture records work and its body returning is not a completion boundary: `FDMeshCommandQueue::record_begin` enables bypass recording at `tt_metal/distributed/fd_mesh_command_queue.cpp:1637-1641`; `record_end` assembles commands. `MeshDeviceImpl::end_mesh_trace` calls `record_end` and then populates storage at `tt_metal/distributed/mesh_device.cpp:1772-1790`. The exception prevents the first replay at probe line 170. The upload's bounds validation occurs before the enqueue's final blocking finish (`mesh_command_queue_base.cpp:299-308`). There is no need to posit a delayed device fault to explain this host check.

## Trace sizing: the complete causal path

| Stage | Source | Unit and effective type |
|---|---|---|
| Assemble commands into a bypass vector | `tt_metal/distributed/fd_mesh_command_queue.cpp:1843-1865` | `std::vector<uint32_t>`; `size()` counts words with `size_t` |
| Determine maximum stream length, append end command | Same file `:1678,1860-1865` | `size_t max_trace_size`, in words; vector is retained |
| Publish byte size | Same file `:1875`; `tt_metal/distributed/mesh_trace.hpp:38` | `max_trace_size * sizeof(uint32_t)` narrowed into **`uint32_t total_trace_size`** |
| Allocate replicated trace buffer | `tt_metal/distributed/mesh_trace.cpp:50-53,79-96` | Already-truncated count extended to `uint64_t`, rounded to a page |
| Compute upload extent | Same file `:149-165` | Full vector copied; `size_t` byte size rounded using the same page |
| Carry requested logical extent to device buffer | `tt_metal/distributed/mesh_buffer.cpp:236-243,189-199` | Replicated extent is retained; no per-bank division of the logical size |
| Validate upload view | `tt_metal/distributed/fd_mesh_command_queue.cpp:878-880`; `tt_metal/impl/buffers/buffer.cpp:663-666` | Page-aligned offset and size must satisfy `offset + size <= self.size()` |

`BufferRegion` has wide `DeviceAddr` fields (`tt_metal/api/tt-metalium/buffer.hpp:146-151`); `DeviceAddr` is `uint64_t` (`hal_types.hpp:11`). Consequently, upload does not silently narrow to match the undersized allocation. `ReplicatedBufferConfig::size` is also `DeviceAddr` (`mesh_buffer.hpp:48-50`). The validator is correctly reporting inconsistent sizes; it is not the first divergent calculation.

Let `D` be the actual assembled bytes, `P` the selected page, and `R(x,P)` round-up to a multiple of P:

```text
descriptor T = D modulo 2^32
allocation A = R(T, P)
upload     W = R(D, P)
```

The assignment to the descriptor is the direct allocation/producer contract violation. The later `uint64_t` local at `mesh_trace.cpp:50` cannot recover discarded high bits. For example, using P=1024:

| Illustrative actual D | Stored T | Allocated A | Uploaded W |
|---:|---:|---:|---:|
| 4,100 | 4,100 | 5,120 | 5,120 |
| 4,294,971,396 (4 GiB + 4,100) | 4,100 | 5,120 | 4,294,972,416 |

The second row fails at offset zero, with upload exceeding allocation by exactly 4 GiB. These are checked arithmetic examples, not values extracted from the failing run. Pages are powers of two from 1024 through 8192 (`tt_metal/impl/trace/trace_buffer.hpp:18-19`), so all divide 2^32. With one stream, overflow predicts `W - A` is a positive multiple of 4 GiB.

### The same size ceiling exists earlier in assembly

The descriptor is not the only 32-bit boundary:

- `tt_metal/impl/dispatch/system_memory_manager.hpp:144-146` declares the vector separately from a **`uint32_t bypass_buffer_write_offset`**, in bytes.
- `system_memory_manager.cpp:524-528` grows the vector by the command's word count but returns a pointer using that byte cursor.
- `:594-596` advances that cursor with unchecked 32-bit byte addition; `:460-465` returns the same cursor for queue writes. `:576-577` also uses a byte write pointer when copying bypass data.
- Recording enables and clears bypass state at `fd_mesh_command_queue.cpp:1639-1641`; clearing resets the cursor (`system_memory_manager.cpp:420-424`). There is no per-request reset inside the single batch capture.

Once command bytes cross 4 GiB, the cursor can wrap while the vector continues to grow. Subsequent commands can overwrite its beginning. Therefore widening only `MeshTraceDescriptor::total_trace_size`, or merely allocating a larger region, would not establish correct large-trace replay. The first byte-capacity violation may already occur in assembly, before the final narrowing. An upstream repair would need checked/wide sizing through assembly, allocation, accounting and dispatch, or an explicit supported-size rejection/partition. Shared-source repair is outside this investigation's permitted edit boundary.

### Why ordinary page padding is not the cause here

The probe opens `MeshShape(1,1)` (`tests/probe_batch.py:87`). `record_end` starts with the local device range, partitions it by program ranges, and emits one complete vector for each resulting range (`fd_mesh_command_queue.cpp:1658-1667,1681,1865`). A single-device local range has only one nonempty partition. This is one vector containing all 32 request sequences, not 32 separately padded trace chunks.

Thus the only write starts with offset zero (`mesh_trace.cpp:149-154`). For `D < 2^32`, both allocation and upload calculate `R(D,P)` using the same selected P. `tt_metal/api/tt-metalium/math.hpp:23-35,48-52` uses their common wide type for round-up. A short final page cannot make the upload larger than the allocation. Even an unpadded count just below 4 GiB can round to exactly 4 GiB on both sides; that has separate bookkeeping risks, but does not create this inequality.

For multiple disjoint device partitions below the limit, allocation from the maximum stream length is also correct: each independent stream starts at zero and `R(D_range,P) <= R(max_D,P)`. Summing all mesh streams would overstate a replicated per-device requirement. Neither ordinary bank padding nor a missing sum over 32 requests explains the inspected path.

## Why the new projections expose the ceiling

`tt/decoder_components.py:193,197` installs precise QKV and output projections. The `linear` wrapper explicitly routes these objects to their Python projection implementation (`:21-24`), while the other weights retain the native linear route. The recurrence implementation's recorded hash is unchanged between the two batch32 runs.

Each tuple request is processed independently by `FunctionalDecoder.decode_forward` (`tt/functional_decoder.py:174-184`). Each one-token decode therefore constructs the entire projection graph again inside the same trace; the projection's token-block limit does not divide batch32 into multiple captures.

From `tt/precise_gdn_projection.py:25-31,45-66`:

| Projection | Weight `[N,K]` | Output blocks at B=32 | Widths processed by the pairwise-add loop | Extra saved-tail add |
|---|---|---:|---|---:|
| QKV | `[10240,2560]` | 320 | 2560,1280,640,320,160,80,40,20,10,5,2 | 1, from width 5 |
| Output | `[2560,6144]` | 80 | 6144,3072,1536,768,384,192,96,48,24,12,6,3 | 1, from width 3 |

Per QKV block the source constructs 23 slice expressions and 12 adds; per output block, 25 slices and 13 adds. Including the weight cast, multiply, and result reshape gives 38 and 41 expressions respectively. Across both projections there are another eight expressions for input casts/reshapes and final concat/cast. Static counts are:

| B | QKV/output blocks per request | Tensor-operation expressions per request | Across batch32 |
|---:|---:|---:|---:|
| 32, failing run | 320 / 80 | 15,448 | 494,336 |
| 128, proposed | 80 / 20 | 3,868 | 123,776 |
| 256, secondary candidate | 40 / 10 | 1,938 | 62,016 |

These counts exclude the rest of the decoder and are **not** measured kernel counts or trace byte counts. Some expressions are views; others lower to several programs. In particular, non-tile-aligned slice beginnings select the row-major composite path (`ttnn/cpp/ttnn/operations/data_movement/slice/slice.cpp:224-235,377-383,433-457`). Thus the pairwise tree's many slices cannot be assumed to be free address aliases. The new expansion explains why the unchanged end-capture machinery can fail after a previously capturable batch32 workload. It does not numerically prove a 4-GiB crossing without the assembled stream length.

## Model-local interventions

| Intervention | Effect on this source defect | Recommendation |
|---|---|---|
| Increase output block B from 32 to 128 | Reduces captured graph and expected trace bytes; avoids the limit if sufficient | **First candidate; smallest change and retains one complete batch32 trace** |
| Split 32 independent requests across several trace IDs | Limits each individual stream and its assembly cursor | Valid fallback for fully traced computation, with lifetime/accounting conditions below |
| Increase nonzero trace reservation, or switch dynamic/reserved allocation | Changes storage placement and budget; leaves truncated requested size unchanged | Cannot repair this allocation/upload disagreement |
| Widen only final allocation/descriptor | Leaves assembly cursor wrap and other 32-bit constraints | Incomplete shared-source fix; outside authorized scope |
| Fuse the exact projection tree into model-local kernels | Can greatly reduce dispatch graph while retaining the arithmetic | Larger implementation effort; reserve for insufficient block widening or later optimization |

### Preferred first experiment: B=128, same complete trace

For the failing version, the smallest prospective edit is `self.block = 128` at `tt/precise_gdn_projection.py:28`. Both output widths are divisible by 128 and 256. That version's setup already partitions checkpoint weights using `self.block` (`:29-31`), and result reshape uses the same field (`:65`), so no logical tail block or additional persistent matrix is required. No code was changed here. This proposal does not request replacing the concurrent B=256 candidate without comparison evidence.

For each output, changing B only groups more independent dot products into each operation. The sequence of K-index pairings, saved odd tails, tail-add order, and final FP32-to-BF16 cast remain the same (`:50-66`). Concat preserves output order. This is source-level arithmetic equivalence; changed tiling/program choices still need device comparison. Do not replace the balanced K tree with arbitrary partial-dot accumulation and assume the rounding behavior is unchanged.

The tradeoff is larger temporary tensors. For logical input token count Q, the initial FP32 product contains `Q * B * K * 4` bytes. At B=128, one-token products are 1.25 MiB for QKV and 3 MiB for output; at Q=32 they are 40 and 96 MiB. At B=256 those values double. These are individual logical product sizes, not allocator peaks: old products and both slices coexist until `del products` at line 59; layout conversion, padding, weight casts, and other live decoder tensors add workspaces. Reducing graph size does not guarantee a smaller activation peak.

For a fresh experiment from the failing version, fixed B=128 is the smaller starting change. If a broader prefill capacity measurement requires preserving the old maximum `Q*B` product bound, an optional model-local design is B=128 with projection-only token blocks of at most 8 instead of 32 (`:38-41`). That preserves the old maximum product element count while reducing the one-token decode graph. It is a larger change with extra token slicing/concat paths, so it should follow an actual memory need. The concurrent source instead keeps B=32 for multi-token inputs and uses device slices of BF16 parent blocks. These candidates preserve batch32 and keep decode computation inside capture.

### Splitting trace IDs while retaining all 32 traced decodes

Because rows have independent state and inputs, capture static row groups (or single rows) separately and replay every trace in order on CQ0 for each decode step. Use the same request-to-input/state/output mapping for all replays. If a single trace ID is a required interface, use the block-size intervention instead.

The existing full warmup and state snapshot/restore logic must cover every group (`tests/probe_batch.py:142-159`). Preallocate all persistent inputs, states, and **stable external output buffers for all 32 requests before the first capture**. Each group's capture should include a final device copy from its temporary returned outputs into those external buffers; release temporary returns before later captures, retain every trace ID, and read the stable outputs after all group replays finish. This prevents a later capture's retained output from being allocated in freed scratch space that an earlier trace will overwrite. Merely keeping later return tensors alive does not reserve them retroactively against an earlier trace's recorded scratch addresses.

Do not blanket-acknowledge these retained later outputs as corruptible: `acknowledge_corruptible` only removes allocation accounting (`ttnn/ttnn/tools/trace_allocation_tracker.py:103-106,126`), and does not create protection or move storage. The existing one-trace probe's acknowledgements (`tests/probe_batch.py:164-165`) are not a sufficient multi-trace lifetime design. Replaying must execute all groups, then compare all outputs; no uncaptured eager projection or per-token recapture qualifies as this workaround. Groups of complete requests avoid cross-trace intermediate tensor handoffs. Splitting inside one request would require explicit stable producer/consumer buffers and a more careful ownership design.

Splitting is a per-stream workaround, not necessarily a capacity reduction: it retains approximately the same aggregate command storage plus per-trace overhead. In addition to available DRAM and dynamic overlap checks, account for the existing `uint32_t` cumulative trace-size bookkeeping (`tt_metal/distributed/mesh_device.cpp:1813-1814`; `mesh_trace.cpp:55-56,170-174`). Choose group sizes using measured assembled lengths, keep each stream comfortably below 4 GiB including padding, and keep total retained storage within the accounting limit. If the original stream exceeds 4 GiB, splitting alone may exceed that conservative aggregate constraint as well; graph reduction may still be necessary. Do not promise that 32 large trace allocations become safe merely because each is individually below the limit. This makes block widening preferable as the first intervention.

### Why a larger trace region does not fix the source cause

The current probe uses `trace_region_size=0`. `mesh_trace.cpp:63-66` therefore allocates ordinary DRAM top-down, and lines 99-147 verify that trace storage does not overlap live replay footprints. A nonzero value changes allocation to `BufferType::TRACE` and checks the dedicated budget at lines 68-76. Both modes allocate the same `R(truncated_descriptor,P)` and upload the same `R(full_vector,P)`.

The logged stack reached the upload, after these allocation/overlap checks. A larger reserved region can address an independent placement or capacity error, but it cannot enlarge this particular buffer beyond its requested truncated size. Suppressing the bounds check would instead permit an oversized write and would not restore any commands corrupted by the assembly cursor wrapping.

## Competing hypotheses and falsification

| Candidate | Concrete contradiction / prerequisites | Match to observations | Smallest discriminating check |
|---|---|---|---|
| **32-bit stream-size overflow** | Direct narrow descriptor versus wide upload; actual stream must reach at least 4 GiB | Explains host upload bounds check, late end-capture failure, single-range geometry, and graph expansion | Compare actual vector bytes, descriptor, allocation and failing region |
| Ordinary last-page padding | Would require different rounding or repeated chunks for one range | Refuted for the inspected one-vector path below 4 GiB | Verify there is one entry and zero offset in the live failure |
| Trace-region shortage / DRAM overlap | Would need failed allocation, budget or overlap validation | Those checks have different failures and precede the observed upload | Inspect actual requested logical size and error site |
| Projection/recurrence device corruption | Would require a separate mechanism to corrupt host-side stream/buffer metadata | No such mechanism evidenced; captured decode has not been replayed | First test the concrete host size invariant rather than changing arithmetic |
| Different compiled trace code or corrupted descriptor | Current source path must differ from runtime, or metadata must be altered elsewhere | Not excluded by model-file provenance alone | If actual bytes are below 4 GiB, inspect live descriptor and binary/source correspondence |

### Decisive next runtime experiment, without shared-source edits

In a separate execution of the existing model-local reproducer, use read-only debugger inspection at entry to `MeshTrace::populate_mesh_buffer` / its upload loop and at the failing `BufferImpl::view`, with symbols sufficient to inspect arguments. Record:

1. `ordered_trace_data.size()` and every entry's device range and vector element count.
2. The maximum **64-bit** vector byte count; `desc->total_trace_size`; selected `page_size`; allocated `padded_size`.
3. Failing `region.offset`, `region.size`, `self.size()` and `self.page_size()`.

The overflow diagnosis predicts one entry, offset zero, actual bytes at least 4 GiB, descriptor equal to actual bytes modulo 2^32, and `region.size - self.size()` a positive multiple of 4 GiB. Observing actual bytes below 4 GiB with one range falsifies this specific trigger: investigate descriptor integrity and the source/binary correspondence instead of attributing it to padding. An unexpected repeated range would reopen the chunk-offset hypothesis. No shared-source instrumentation is necessary in principle; variable availability depends on build symbols/optimization.

Then test the B=128 candidate under the same batch32 command, provenance capture, runtime audit and watcher settings. Record trace byte counts again, verify the complete trace ends and replays, compare every row against HF, repeat restored-state replay, and compare old/new precise projection outputs on identical device inputs. Include Q=1 and multi-token counts through 32; if the optional token-block change is used, cover 1/3/7/8/9/31/32. A passing B=128 capture alone is evidence for a size-sensitive workaround, not proof of the exact old byte count or full numerical acceptance.

## Other potential issues and acceptance gaps

1. **Repeated chunks for one device range are not handled consistently.** `mesh_trace.cpp:155-166` page-pads each chunk but increments a 32-bit range offset by the original unpadded size. Hypothetical repeated entries could create an unaligned next offset, padding overlap, offset wrap, or aggregate sizing problems. Current assembly emits one vector per disjoint partition, so this is not the reported single-device cause. An unaligned next offset would normally fail `buffer.cpp:664` before line 666.
2. **Near-limit page-selection arithmetic is also 32-bit.** `tt_metal/impl/trace/dispatch.cpp:211,238-240` accepts a 32-bit size and computes fully banked round-up in 32 bits. That can distort page selection near 4 GiB. Since allocation and upload use the same selected page, it does not independently create their observed size inequality below 4 GiB. Treat it as part of a future complete large-trace audit.
3. **Cumulative accounting can wrap, including padding to exactly 4 GiB.** The getter/setter at `mesh_device.cpp:1813-1814` and updates in `mesh_trace.cpp:55-56,170-174` use 32-bit totals. This matters especially for retained split traces but is not the full-vector upload mismatch.
4. **Assembly reserve counts have a separate, larger limit.** `system_memory_manager.cpp:525-526` narrows the vector's word count and next word count to `uint32_t`, corresponding to approximately 16 GiB of words' byte storage. The earlier byte-cursor and descriptor limits already prevent treating that as supported trace capacity.
5. **The stated prefill pass is not supported by the saved gate metrics.** `linear_precise_gdn_batch32.json:371-386` records request16 aggregate contribution PCC `0.994186282157898` and token0 contribution PCC `0.9906083941459656`. `tests/probe_qsa_decoder.py:61-70` requires both aggregate and every-token output/contribution PCC at least 0.995. The probe records metrics before capture but evaluates `passed` only after replay (`tests/probe_batch.py:145-146,180-182`), so reaching capture does not imply prefill acceptance. The precision change's numerical cause is outside this trace-sizing investigation; it remains a separate acceptance failure after capture is fixed.

## Report review

The first-draft claims were revisited against source, runtime sequence, provenance hashes and saved metrics, including an independent xhigh review. The headline deliberately separates the proven 32-bit source defect from the unlogged runtime threshold. Ordinary padding and repeated-chunk theories were tested against actual single-device assembly and are not promoted as causes. Static operation counts were checked with integer arithmetic and are not converted into fabricated trace byte measurements. The model-local block candidate preserves the source K tree but is explicitly unverified on device. The final hash check caught the concurrent projection change; the original failure and new candidate are kept revision-distinct above. This investigation edited no model or shared-library implementation.
