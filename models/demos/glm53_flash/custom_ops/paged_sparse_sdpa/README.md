# paged_sparse_sdpa (model-local generic op)

The sparse DSA prefill attention (`ttnn.transformer.sparse_sdpa`) takes its keys as ONE dense ROW_MAJOR tensor
`kv [1,1,T,512]` indexed by token id, so every DSA layer of every prefill chunk copied the user's whole prefix out
of the paged TILE latent cache into a fresh ROW_MAJOR tensor (1 KiB per position per chip, 1 GiB at 1 M). On the
served 1 M profile (4 resident users per row, ~2 GiB of headroom beside the caches) that transient killed the
per-row waves at 378 K, 477 K and 112 K positions in three different ways (two copies, a growing copy against a
fragmented bank, a constant copy against the scorer's blocks) and a resident scratch left no room for a chunk's MoE
buffers (`doc/mesh/README.md`, 2026-10-30). This op reads the cache in place.

| tensor | shape | dtype / layout | role |
|---|---|---|---|
| `q` | `[1, H, S, K_DIM]` (H % 32 == 0) | bf16 ROW_MAJOR, DRAM interleaved | the absorbed queries (as the stock op) |
| `cache` | `[max_blocks, 1, 32, K_DIM]` | bf16 TILE, DRAM interleaved | the paged latent cache (read in place) |
| `indices` | `[1, 1, S, TOPK]` | uint32 ROW_MAJOR | the index rows: 4-token pools (ids 4p..4p+3 at row positions 4g..4g+3), the tail pool's ids, then 0xFFFFFFFF sentinels |
| `out` | `[1, H, S, v_dim]` | bf16 ROW_MAJOR | the attention output (allocated here) |

Scalars: `slot_block0` = the user's first physical block (the identity page table: block of token `id` =
`slot_block0 + id // 32`), `blocks_per_seq` (ids outside the user's pages read as zero rows), `v_dim`, `scale`,
`k_chunk_size` (128), `pool` (4).

Kernels (`kernels/`): `compute_paged_sparse_sdpa.cpp` is tt-metal's `sparse_sdpa_compute.cpp` with one compile-time
switch (`PAGED` at `sparse_sdpa::compute_ct_arg::END`): the K chunk arrives already tiled in `cb_k_in`, so the per-chunk
K tilize is skipped; QK, masks, online softmax, PV, SALAD and the normalize are the stock code. `writer_...` is the
stock writer without the dual-NoC K-half gather and the attention sink. `reader_...` keeps the stock per-token flow
(Q rows, index row, nv = the first sentinel, the control message, the boundary mask tile) and gathers each 128-row
K chunk out of the cache's tiles: a 4-token pool is 4 consecutive rows inside one 16-row face of the source block and
of the destination tile, so a pool is 32 reads of 128 contiguous bytes (16 tiles x 2 face columns; Blackhole's DRAM
reads need 64-byte alignment, which 1-row reads would break). Rows at or past nv are zeroed in L1 after the gather
(the masked keys must be finite), as the stock reader zero-fills its sentinel suffix. The CB set and the compute's
argument layout mirror the stock program factory (`sparse_sdpa_program_factory.cpp`), with `cb_k_rm` a 64-byte dummy.

Contract: bit-identical to `ttnn.transformer.sparse_sdpa(q, user_rows_rm, indices)` on the same data (same compute
kernel, same tiles); the index rows must have the pool structure above (the indexer's `expand_pools` output).
Cost: the gather issues 8x the NoC transactions of the stock 1 KiB row reads (single NoC in this first version);
the stock op's whole-prefix copy (~13 ms per layer-chunk at 1 M) disappears.

Test: `tests/custom_ops/test_paged_sparse_sdpa.py` (single chip: the op vs the stock op on the ROW_MAJOR copy at the
DSA shapes, pools + a short tail + sentinels, several users' slots, a second call with new indices).
