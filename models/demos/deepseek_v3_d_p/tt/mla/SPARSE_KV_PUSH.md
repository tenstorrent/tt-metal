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
  `FABRIC_2D_TORUS_XY` mesh of at least 3x3 (`sparse_kv_push_supported`). `ttMLA(sparse_kv_push=None)` picks it
  automatically; `False` forces the dense gather (the reference in tests). Other fabrics keep the dense gather.
- Full (indexer) layers keep the dense gather, overlapped with their top-k on sub-devices.

## Data flow

Once per set of indices (the first shared layer after a full layer, or the full layer itself when it prefetches):

1. **Plan** (`ttnn.experimental.sparse_push_plan`): from this chip's top-k, a plan table (bitmap of needed rows,
   per-owner counts, per-owner counts before the chunk start) and the top-k rewritten to slots of a compact buffer.
   Table layout: `device/sparse_push_plan_device_operation.hpp`.
2. **Table exchange**: `high_bw_all_gather` collects every chip's table on every chip.
3. **Push** (`high_bw_all_gather(push_table=...)`, `HighBwSparsePushFactory`): each owner reads the requester's
   counts and bitmap over the blocks it owns and unicasts the needed rows straight into the requester's compact
   buffer, packed into bank-lane packets. One fabric channel per (link, direction), each with 2 senders behind a
   fabric mux; routes follow the fabric routing tables, with ring-distance ties split between both directions
   (`kernels/sparse_push_route.hpp`).
4. `sparse_sdpa` reads the compact buffer through the rewritten indices (no block-cyclic remap). The result is
   bitwise equal to the dense path.

**Overlap with attention.** A layer whose successor reuses its indices runs `sparse_sdpa` with the successor's
prefix push fused in (`SparseSDPAPushFusedFactory`): attention on the overlap profile's top-k grid, the push senders
and muxes on its gather grid, in one program. The prefix part holds rows written by earlier chunks; the successor
then pushes the remainder (`push_part=2`, its own chunk's rows) before its attention. Compact buffers alternate by
layer parity, so layer L reads its buffer while L+1's fills. Without an overlap profile, a shared layer pushes all
its rows itself (`push_part=0`).

**L1 front.** `sparse_sdpa` is bound by its row reads, so each compact buffer keeps its first rows in L1: a front of
`L1_FRONT_BYTES_PER_CORE` (160 KB) per core per parity, sharded one push packet per shard and round-robin over all
worker cores, with the remaining rows in a DRAM spill of one packet per page (row `r` at page `r / P`). A packet thus
stays one contiguous write, and with a front the plan lays packets out on consecutive slots (`banks = 1`). Both
parities stay allocated for the whole forward: 2 x 192 KB clashes with the circular buffers of the largest GLM
program on the Galaxy. Scaled FP8 front rows are 704 B, the DRAM-aligned page of a 656 B row.

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
- Op level with real 1M top-k (fused prefix + remainder, and the full push, vs the dense gather; all-DRAM, the
  model's L1 front, and a tiny front that sends most rows to the spill):
  `tests/nightly/blackhole/sdpa/test_sparse_sdpa_kv_push.py`.
