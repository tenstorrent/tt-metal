# Integer route-plan fusion: source-only candidate

Status: design only. No implementation, TTNN import, device execution, or latency measurement was performed for this investigation. The candidate is feasible within the existing deterministic, local EP4 grouped-decode contract. It can replace the count/prefix graph with integer work in the existing route-mask data-movement kernel while leaving dispatch, grouped expert arithmetic, combine, and post-combine unchanged. Speed and exact device equivalence remain gates.

## Scope and proposed interface

Extend `tt/moe_route_mask.py` with a helper such as:

```python
masked_ids, counts, offsets = plan_inactive_routes(
    ids, active_mask, dispatch_table, slots, mesh
)
```

Inputs retain the existing helper contract: one UINT32 TILE top-eight route tensor with logical shape `(1, 1, 32, 8)` and padded shape `(1, 1, 32, 32)`; persistent UINT32 ROW_MAJOR activity vector of 32 words; rank-local INT32 ROW_MAJOR ownership table `(1, 256)`; immutable mapping from the 2, 4, or 32 logical bucket rows to request slots. All tensors use interleaved DRAM. The optional solo0 graph also has physical bucket 2 and uses the same contract.

Outputs:

| Tensor | Dtype/layout | Per-device shape | Meaning |
| --- | --- | --- | --- |
| `masked_ids` | UINT32 TILE | logical `(1,1,32,8)`, padded `(1,1,32,32)` | Existing route-mask result, including preserved padding words |
| `counts` | UINT32 ROW_MAJOR | `(1,256)` | Unaligned token count for each global expert; off-rank entries zero |
| `offsets` | UINT32 ROW_MAJOR | `(1,256)` | Exclusive prefix of tile-aligned counts, measured in token rows |

An independent constructor switch such as `fuse_moe_route_plan=False`, requiring `suppress_inactive_moe_routes=True`, keeps a route-mask-only differential control. Guard the new branch on masked, deterministic, local grouped decode with exactly 32 internally padded tokens. Keep unmasked decode, B1 sparse decode, prefill, and stochastic grouped paths unchanged. No adapter or plugin interface change is needed.

At `_grouped_moe`, replace only the current mask call plus the count/prefix graph. Preserve the existing UINT32-to-UINT16 cast, ROW_MAJOR conversion and `(1,32,8)` reshape for `dispatch_ids`; both dispatch and post-combine must consume these same masked IDs. Producing dispatch-format IDs directly is a separate candidate and is unnecessary for this discriminator.

## Source-checked contracts

The relevant model source is `tt/multichip_decoder.py`:

- `_moe` pads a multirow input internally to a multiple of 32 before router projection. The original `route_slots` mapping survives padding. Router logits are FP32, with 256 expert columns; the grouped path calls `ttnn.topk(..., k=8, dim=-1, sorted=True)` without caller-supplied indices.
- The EP4 ownership table has `0` for this rank's contiguous 64 global expert IDs and `-1` elsewhere. The dispatch and post-combine ownership tables agree. The local expert mask is exactly the nonnegative ownership predicate.
- `_grouped_moe` scatters ones into a zero tensor, sums over rows in FP32, applies the local mask, aligns each count up to 32, multiplies the BF16 aligned counts by a BF16 upper-triangular matrix of ones with FP32 accumulation/output, subtracts the aligned counts, and casts both results to UINT32 ROW_MAJOR.
- `dispatch_capacity(32,64)` is 2240 rows. Preserve this allocation and all native operation parameters.

TopK uniqueness is a source-level prerequisite, not an empirical assumption about untied logits. `ttnn/cpp/ttnn/kernel_lib/index_tile_dataflow.hpp` generates the original width positions `wt*32..wt*32+31`; the TopK reader kernels carry these indices through sorting. The 256 input columns therefore have distinct identities, and selecting eight entries yields eight distinct expert IDs per row even when values tie. No custom duplicate index tensor is supplied by this model. FP32 input requires UINT32 indices, as documented and checked by `operations/reduction/topk/device/topk_device_operation.cpp`.

Native consumers establish the units:

- `operations/experimental/deepseek_prefill/dispatch/device/kernels/dataflow/reader_worker_dispatch.cpp` checks `expert_dispatch_table[id] == -1` before using an offset. A kept route consumes `page_idx = offsets[id]++`; one page is one token row of the dispatch payload. These mutable counters are the kernel's local copies, not a reason to change the model-owned offset output.
- `.../moe_fused_swiglu/device/kernels/moe_fused_swiglu_reader.cpp` indexes raw counts by global expert ID, derives `m_t = ceil(count/32)`, and reads a region base explicitly measured in token rows. It validates base alignment to 32 and region bounds against row capacity.
- `.../combine/device/kernels/dataflow/reader_untilize.cpp` independently reconstructs aligned expert prefixes from the counts. Its metadata start token is the same aligned row prefix used by dispatch. Thus counts and offsets must agree exactly; a merely adequate capacity bound is insufficient.

The native ACK drain and zero-count synchronization remain untouched. Reducing counts is not authorization to bypass native launches or their completion protocol.

## Exact integer algorithm

Read the full route tile, activity page, and ownership page, then wait for those reads. Clear all 256 scratch counts on every invocation. For each physical row `r` in `0..31`:

1. Determine live activity from `r < len(slots)` and the current device value `activity[slots[r]]`. Padding is inactive independently of any activity entry with the same numerical row index.
2. For an inactive row, overwrite its eight logical IDs with the existing valid off-rank skip ID: 64 on rank 0, or 0 on ranks 1–3. Do not increment any count for this row.
3. For a live row, preserve every ID and increment `counts[id]` only when `owners[id] >= 0`. Read the eight logical IDs using the existing tile-face word offset `(r // 16) * 512 + (r % 16) * 16`. Do not interpret the other 24 padded columns as routes.

Then scan all **256 global expert indices in increasing order**:

```text
running = 0
for expert in 0..255:
    offsets[expert] = running
    running += ((counts[expert] + 31) // 32) * 32
```

Write all three output tensors and complete their writes before the operation ends. Preserve every non-route padding word in the original tile. No floating-point arithmetic, atomic increments, collective, host activity read, or host count calculation is needed. A single core owns the full histogram per rank.

Do not zero all off-rank offsets. Counts before a rank's owned range are zero, but offsets after that range equal its final aligned total. The existing triangular prefix produces this result across all 256 entries, including entries that native dispatch will skip.

## Equivalence and corner cases

**Ordinary live rows.** With distinct TopK IDs, scatter-to-one followed by row reduction counts exactly one occurrence per `(row, expert)` pair. Direct integer increments yield the same result. Rank ownership filtering commutes with the row sum because the ownership predicate depends only on expert ID.

**Duplicate fake IDs in inactive rows.** Eight identical skip IDs are intentional. In the old pipeline, scatter sets that expert's element to one, the row sum may count the inactive row once, and the local mask removes it because the skip ID is off-rank. The fused implementation skips the inactive row altogether. Counting its eight fake IDs would be wrong; the proposed algorithm never does so.

**Duplicate IDs in a live row.** These violate the model's TopK contract. A bare histogram would count duplicates whereas scatter-to-one would deduplicate them. The minimal candidate should state and test the unique-live-ID prerequisite. If the helper is later generalized to arbitrary IDs, deduplicate within each live row before histogramming. That adds at most 28 earlier-ID comparisons per row; it does not make malformed repeated live routes valid for native dispatch, which emits each route independently. Do not silently promise support for arbitrary repeated live IDs.

**All 32 active.** With all 32 mapped slots active, the entire ID tile remains bitwise identical. Counts and offsets must match both the current route-mask pipeline and the current unmasked local count pipeline. An all-active mask with only 2 or 4 logical mapped rows still suppresses the remaining padding rows; it is not the 32-live identity case.

**No active rows or no routes owned by a rank.** All counts and offsets are exactly zero on that rank. Native consumers still run their established zero-count protocol. A zero-count expert after other nonzero experts has a nonzero prefix offset; its own alignment contribution is zero.

**Bounds for internally padded 32-row decode.** Each raw count is in `0..32`; each aligned count is either 0 or 32. At most 64 local experts are nonzero, so the final aligned total is at most 2048 rows, below the unchanged capacity of 2240. For `A` live rows, the tighter total bound is `32 * min(64, 8*A)`. Every offset is a multiple of 32 in `0..2048`. Total raw local routes are at most `8*A <= 256`. UINT32 arithmetic has ample headroom.

**Floating-point reference exactness.** Scatter entries are zeros and ones, row sums are integers up to 32, multiplication by `1/32` is binary-exact, and alignment produces 0 or 32. Both BF16 matrix operands represent those values exactly. Prefixes are sums of at most 64 terms of 32, and the existing FP32 accumulator/output represents all resulting integers exactly. Subtracting one aligned count and casting to UINT32 therefore yields the same integer exclusive prefix. This is the arithmetic argument; direct device comparison with the actual reference graph is still required to exclude implementation or layout errors.

## Kernel, trace and allocation implications

The existing one-core generic operation already reads 4096 route bytes, 128 activity bytes and 1024 ownership bytes. Keeping those regions and appending two 1024-byte scratch arrays gives:

| Scratch region | Byte offset relative to aligned base | Bytes |
| --- | --- | --- |
| Route tile | 0 | 4096 |
| Activity | 4096 | 128 |
| Ownership | 4224 | 1024 |
| Counts | 5248 | 1024 |
| Offsets | 6272 | 1024 |

Reserve `7296 + 64 = 7360` bytes, including the existing alignment allowance, rather than placing these arrays on a RISC-V stack. Counts and offsets each occupy one 1024-byte ROW_MAJOR page. Allocate their outputs with matching mesh placement and explicit UINT32/ROW_MAJOR/interleaved-DRAM metadata; a local shape alone is insufficient if allocation accidentally replicates or shards the wrong dimension.

There are two additional output pages, totaling 2048 bytes per invocation per rank; do not multiply that figure into a retained trace-memory claim without measuring allocator retention. The kernel gains histogram initialization, up to 256 route visits, a 256-element integer scan, and two DRAM writes. It removes the intermediate scatter tensor and the count/prefix TTNN graph. Neither the latency saving nor captured command size is known from source alone.

All output words must be assigned on every replay, including zero-count cases. Reuse the captured output addresses; activity is read from the persistent device vector each replay, never compiled from its capture-time contents. The immutable `slots` mapping can remain a runtime-argument constant per bucket. Use the established dataflow ordering primitives so RISC-V stores are visible to NoC before output writes: the index-tile helper explicitly documents the store-to-NoC visibility issue and uses a blocking readback for its seeded words. A NoC completion barrier alone describes outstanding NoC operations, not an independent proof of prior processor-store visibility.

Warm the new operation for B2/B4/B32 and optional solo0 before capture. Keep all model and sampling traces, their selection rules, persistent scheduler inputs, nonblocking execute boundaries, and deferred reads unchanged. The candidate reduces captured work inside each layer; it requires no new trace signature and no per-token host work. Source-only prediction cannot establish whether trace reservation or mixed-traffic latency improves.

## Reduced discriminator before full serving gates

1. On all four ranks, compare all three helper outputs exactly against the existing mask-plus-count/prefix graph. Cover logical bucket sizes 2, 4 and 32; low/high and noncontiguous slot mappings; zero/one/all activity; changed activity on replay; tied-router-value TopK IDs; maximum per-expert count 32; broad local expert coverage; mixed local/off-rank IDs; and physical tile padding preservation. Include the 32-live bitwise identity case and inactive duplicate skip IDs explicitly.
2. Replay one capture after changing both route IDs and activity, then replay unchanged inputs. Inspect all 256 count and offset entries, including the off-rank prefix tail. This detects stale output pages and capture-time activity mistakes.
3. Use route-mask-only versus route-mask-plus-plan generators with identical selected precision, topology, warmup ordering and bucket preparation. Compare logits and all persistent cache/recurrent state for the existing B2/B4/B32 lifecycle and stale-input cases, plus solo0 if that candidate is enabled. Require exact active behavior and preserved dormant state.
4. Only after correctness, compare warmed native decode and the actual vLLM primary workload with identical server settings, then mixed-shape trace retention and CI burst. Keep benchmark JSON as performance evidence. No profiler is required or proposed.

The first implementation decision should follow the queued route-mask/solo measurements. This design does not claim that route planning dominates remaining decode time or that fusion meets the optimization stop rule.
