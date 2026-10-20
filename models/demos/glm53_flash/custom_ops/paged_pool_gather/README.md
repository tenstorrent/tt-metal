# paged_pool_gather (model-local generic op)

Gathers pools of `P` consecutive token rows out of a paged bf16 TILE cache into a compact paged TILE cache, one
contiguous page range per user (identity page table). Used by the traced sparse DSA decode (`tt/dsa.py`,
`TtDSA._decode_unified`): the 512 top-scored 4-token pools plus the incomplete tail pool of every user (513 pools,
65 output blocks = 2080 rows per user), which `paged_flash_multi_latent_attention_decode` then attends with a
per-user `cur_pos`. Design: `doc/mesh/sparse_1m.md` section 2.3.

| tensor | shape | dtype / layout | role |
|---|---|---|---|
| `cache` | `[max_blocks, 1, 32, W]` (W % 32 == 0) | bf16 TILE, DRAM interleaved | the paged latent cache (read) |
| `page_table` | `[B_table, blocks_per_seq]`, B_table >= B | int32 ROW_MAJOR | row b maps user b's logical blocks (any mapping) |
| `pool_ids` | `[B, 1, 1, N]` | uint32 ROW_MAJOR | pool ids per user; 0xFFFFFFFF or `P*id >= blocks_per_seq*32` = zero rows |
| `out` | `[B * ceil(N*P/32), 1, 32, W]` | bf16 TILE, pre-allocated by the caller | output row `P*i + j` of user b = cache row `P*pool_ids[b,i] + j` of user b; padding rows zero |

Semantics: pure data movement, no in-place inputs; `out` is fully written (every row of every output block, zeros
where no pool). A pool never crosses a 16-row tile face (`P` in 1/2/4/8/16), so a pool is two contiguous
`P*32`-byte runs per tile (one per face column): 32 reads of 128 B per 4-token pool, written into an L1 tile
block, then `W/32` whole-tile writes.

Kernel (`kernels/reader_paged_pool_gather.cpp`, RISCV_1 reader config): compile-time args
`[Wt, P, N, out_blocks, blocks_per_seq, pt_row_bytes (aligned page), ids_row_bytes (aligned page), slots_per_block]`
+ `TensorAccessorArgs` of cache, page_table, pool_ids, out; runtime args per core
`[item0, n_items, stride, cache_addr, pt_addr, ids_addr, out_addr]`. Work items are (user, output block) dealt
round-robin over `min(B*out_blocks, grid)` cores (core c = `(c // grid.y, c % grid.y)`); two L1 tile blocks
alternate so the writes of one item drain while the next item's reads are in flight. CBs: 0 = tile blocks
(2*Wt+1 x 2 KB), 1 = the pool-id row, 2 = 64 B page-table fragments.

Under trace: all four tensors must be persistent (the runtime args carry their addresses); `TtDSA` keeps one
compact `out` buffer per batch size. Measured on chip 0 with a 1 M-position cache, 513 pools per user: 0.061 ms
(1 user), 0.183 ms (4), 0.339 ms (8). Test: `tests/custom_ops/test_paged_pool_gather.py` (torch golden, permuted
page tables, sentinels, B = 1/4/32, trace replay).
