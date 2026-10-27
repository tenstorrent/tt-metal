# Sparse KV push (GLM sparse MLA prefill)

## Why

In chunked prefill, every sparse MLA layer needs the KVPE rows its top-k selects, from the whole prefix, on every
chip. The dense path gathers the entire prefix onto every chip with `high_bw_all_gather`: at a 1M-token prefix that
moves the whole prefix per layer (about 8 ms on an 8x4 Blackhole Galaxy), while attention itself takes about 1 ms.

A chip's top-k touches only a few percent of a long prefix (9k-53k distinct rows at 1M, median about 30k). GLM
"shared" layers (`config.indexer_types[layer] == "shared"`) reuse the top-k of the full layer before them, so for
them the needed rows are known before the layer runs. The sparse KV push sends each chip only those rows.

## Where it applies

- Layers fed another layer's top-k indices (GLM shared layers; also GLM MTP levels after the first), on a
  `FABRIC_2D_TORUS_XY` mesh of at least 3x3 (`sparse_kv_push_supported`). Meshes the push cannot route keep the
  dense gather.
- Full (indexer) layers keep the dense gather, overlapped with their top-k on sub-devices.

## Data flow

Once per set of indices (the first shared layer after a full layer, or the full layer itself when it prefetches):

1. **Plan** (`ttnn.experimental.sparse_push_plan`): from this chip's top-k, a plan table (bitmap of needed rows,
   per-owner counts, per-owner counts before the chunk start) and the top-k rewritten to slots of a compact buffer.
   Table layout: `device/kernels/sparse_push_constants.hpp`.
2. **Table exchange**: `high_bw_all_gather` collects every chip's table on every chip.
3. **Push** (`ttnn.experimental.sparse_kv_push`): each owner reads the requester's
   counts and bitmap over the blocks it owns and unicasts the needed rows straight into the requester's compact
   buffer, packed into bank-lane packets. One fabric channel per (link, direction), each with 2 senders behind a
   fabric mux; routes follow the fabric routing tables, with ring-distance ties split between both directions
   (`kernels/sparse_push_route.hpp`).
4. `sparse_sdpa` reads the compact buffer through the rewritten indices (no block-cyclic remap). The result is
   bitwise equal to the dense path.

**Overlap with attention.** A layer whose successor reuses its indices runs `sparse_sdpa` beside the successor's
prefix push (`part=1`), on two sub-devices (`SparseKvPushPlanner.prefetch_manager`, grids from
`SparseKvPushPlanner.grids`); clearing the manager joins them. Attention takes the fewest whole columns that keep its busiest core at the fewest
tokens (2 on 80 cores for GLM's 160 tokens per chip): more cores only add DRAM contention with the push. The push's
24 cores take the remaining columns' rows furthest from the Ethernet row. The prefix part holds rows written by
earlier chunks; the successor then pushes the remainder (`part=2`, its own chunk's rows) before its attention. A
shared layer nothing prefetched for pushes all its rows itself (`part=0`).

**Buffers.** A chip's distinct rows are at most its query rows x top-k, so the push buffer is sized from the chunk
at init (`SparseKvPushPlanner.buffer_rows`; about 220 MiB at a 1M cache instead of 703 MiB), not from the cache.
Layer L reads one buffer while L+1's fills: counting from the last full layer, even layers use the dense gather
buffer (idle outside full layers) and odd layers the push buffer. When the push buffer does not fit in the dense one
(small caches), two push buffers alternate instead.

## Trace safety

On the metadata path (`PREFILL_USE_TRACE=1`) nothing depends on a host value that changes per chunk or request:

- the plan reads the chunk start from `metadata[1]`; it is not part of the planner's cache key, and every layer that
  computes its own indices invalidates the planner;
- the push and the fused push take the cache slot from `metadata[0]` (recomposed on device as
  `user * layer_num + layer`);
- planner buffers and semaphores are created by the warm pass, before capture.

## Tests

- MLA level, push vs dense gather (bitwise; single and multi-chunk, padded chunks, prefetch chains, traced replay over
  user slots and chunk starts): `models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla.py -k kv_push`.
- Plan op vs a host reference: `models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_sparse_push_plan.py`.
- Op level with real 1M top-k (prefix beside attention + remainder, and the full push, vs the dense gather):
  `tests/nightly/blackhole/sdpa/test_sparse_sdpa_kv_push.py`.
