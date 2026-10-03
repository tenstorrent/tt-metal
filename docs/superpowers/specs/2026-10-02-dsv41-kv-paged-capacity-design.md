# DeepSeek-V4.1-Flash on the 4x8 Blackhole Galaxy: KV / state capacity model, paged cache and sparse decode attention

Status: design + first measurements (2026-10-02). Owner of this document and of `tt/kv_paged.py`, `tt/indexer.py`, `tests/test_kv_paged_*.py`,
`tests/test_indexer_*.py`. Other agents (attention, prefill, spec decode) consume it; every number marked MEASURED comes from a run on host .44
(logs `/mnt/tt-data/ssinghal/dsv4-logs/kv_agent_*`), ESTIMATE means arithmetic on measured pieces, ASSUMPTION means not measured.

## 0. Summary (read this first)

1. **Free DRAM per chip with the full 40-layer model resident, batch 16, one captured decode trace: 1022 MiB/bank x 8 banks = 7.99 GiB (MEASURED).**
   (31.2 GiB allocatable/chip with the 700 MB trace region carved out; layers 69.1 MiB/bank each = 22.1 GiB; embedding (replicated) 1.23 GiB; head 0.08 GiB.)
   Sharding the embedding over the 8 columns gives back ~1.1 GiB. Plan with 7 GiB for KV + spec + prefill, 8-9 GiB is reachable.
2. **The KV is far smaller than feared once the compressed latents are stored ONCE per kv-source layer instead of per reading layer.**
   Per user and context token: 2.5 KiB of latents (3 ratio-2 sources + 1 ratio-1 source, bf16) + 0.6 KiB of index keys = **3.1 KiB/token bf16, 1.6 KiB/token with fp8 latents + bfp8 index keys**
   (today's code keeps a copy per reading layer: 29 KiB/token, 10x too much). Fixed per user: 5 MiB of window rings (40 layers x 128 x 512 bf16).
3. **Capacity (7 GiB, lean formats, uniform context, no reservations):** replicated over the 8 columns: U=4 users/row (batch 16) -> 1M tokens, U=8 (32) -> 563k, U=16 (64) -> 279k,
   U=32 (128) -> 138k. Page/sequence-sharded over the 8 columns: 1M tokens up to batch 128. 64k context fits at batch 128 even replicated. Tables in section 3.
4. **Decode attention at long context = sparse_sdpa (exists, Blackhole, V4-flash shaped: H multiple of 32, sink, 640 = 128 window + 512 selected rows, bf16/fp8 kv) fed by the indexer
   (`indexer_score_dsa` + `topk_large_indices`, both exist).** sparse_sdpa MEASURED 58-83 us/layer for 1-32 users (PCC 0.9998 vs torch). The cost that matters at long context is the
   indexer's top-k (4.4 ns per entry per row: 0.57 ms at 128k entries) and scoring (0.9 ns/entry/user): section 5.
5. **The one missing primitive** is an in-place write of ONE row at a data-dependent index into a ROW_MAJOR cache (sparse_sdpa reads RM rows; `paged_update_cache` is TILE-only). Design in 4.4 (hot tail
   block in TILE + static pending region + rare host-issued block flush). Needs a decision / a small op: see section 7 (open items).
6. **Indexer layer 2 implemented and checked against the checkpoint's Indexer** (section 6): 0.14 ms (matmul backend) to 0.47 ms (fused + fp4-q simulation) per step for 4 users/chip at N = 2k, 0.2-0.5 ms at 8k, 0.7-0.9 ms at 32k (4-8 users);
   top-512 set agreement 95.5% (2k) / 94.1% (8k) / 88.5% (32k, near-tie limited), score-mass recall >= 98%; bfp8_b index keys are lossless for the selection.

## 1. What is cached (derived from `config.json` + `inference/model.py`)

Layers: 40 backbone (0..39) + 3 MTP layers (40..42, window only). `compress_ratios`: layers 0,1 -> 0 (window only); 2..19 -> 2; 20..39 -> 1.
`kv_source_layers = [2, 8, 14, 20]`: the only layers with a Compressor. Every other compressed layer READS the compressed cache of the last source before it
(`shared_attn.compress_kv` slot): 3-7 read 2, 9-13 read 8, 15-19 read 14, 21-39 read 20.
`index_source_layers = [2, 8, 14, 20, 24, 28, 32, 36]`: layers that run an Indexer; the others reuse the published top-k (`shared_attn.topk_idxs`).
Index KEYS are produced only where the layer owns a compressor (2, 8, 14, 20: `k = k_norm(wk(latent))`, RoPE, fp4 simulation); 24/28/32/36 score against layer 20's keys.
Layer 20 is the candidate source (`candidate_topk_blocks=2048`, `candidate_block_size=8`): layers 24..36 only rank positions inside the 2048 best blocks (block score = max of its 8 position scores).

**Selection groups** (layers whose 512 selected compressed rows are identical, so the rows can be fetched once): (2,2): 2-7, (8,8): 8-13, (14,14): 14-19, (20,20): 20-23, (20,24): 24-27,
(20,28): 28-31, (20,32): 32-35, (20,36): 36-39 -> 8 selections per step (`V41Geometry.selection_groups()`).

| state | where | entries per context token | row | notes |
|---|---|---|---|---|
| window ring | every attention layer (40 + 3 MTP) | 128 slots fixed | 512 | K == V; ring slot = pos % 128; fixed per user |
| compressed latent | kv sources 2, 8, 14 | 1/2 | 512 | shared by all readers (today: copied into every reader) |
| compressed latent | kv source 20 | 1 | 512 | shared by layers 20..39 |
| index key | owners 2, 8, 14 | 1/2 | 128 | reference stores fp4-simulated values (exactly representable in bf16) |
| index key | owner 20 (read by 24, 28, 32, 36) | 1 | 128 | |
| compressor state | 2, 8, 14 | fixed | 2 x [kv, score] fp32 (reference) | step-independent trick keeps only the previous token: 4 KiB/layer |
| Engram | layers 1, 14, host | fixed | n-gram hash history (3 tokens) | tables live on the host (2 x 94.6 GB fp8); nothing per context |

Reference storage precision: window K/V and compressed latents are fake-quantised (fp8 per-32 / fp4 e2m1 per-16 with e4m3 scale), index keys and q with fp4 e2m1 per-32 e8m0. All values are
exactly representable in bf16; fp8_e4m3 (1 B) or bfp8_b (1.06 B) storage adds an error well below the model's own fp4 noise (to be confirmed per format by the attention agent; for the index keys
section 6 measures bf16 vs bfp8).

### 1.1 Bytes (`tt/kv_paged.py`, `tests/test_kv_paged_capacity.py`)

Per user, per context token (L tokens): latents `512 * dtype * (1/2 + 1/2 + 1/2 + 1)` = 2560 B (bf16) / 1280 B (fp8); index keys `128 * dtype * 2.5` = 640 B (bf16) / 340 B (bfp8) / 180 B (bfp4).

| format preset | latents | index keys | ring | per token | per user fixed |
|---|---|---|---|---|---|
| `bf16` (all bf16) | 2560 B | 640 B | 40 x 128 KiB | **3200 B** | 5.0 MiB |
| `lean` (latents fp8_e4m3, index keys bfp8_b, ring bf16) | 1280 B | 340 B | 5.0 MiB | **1620 B** | 5.0 MiB |
| `leaner` (+ ring bfp8, index keys bfp4) | 1280 B | 180 B | 2.7 MiB | 1460 B | 2.7 MiB |
| today (copy per reading layer, bf16) | 29696 B | - | 40 x 256 KiB | 29.7 KiB | 10 MiB (256-slot caches) |

Page rounding adds ~0.1 MiB/user (half a 128-token page of every pool). Spec decode adds ring slack (+32 rows x 40 layers = +1.25 MiB bf16, section 4.6), compressor snapshots (+0.4 MiB) and 3 MTP rings (+0.4 MiB).
Per-user fixed cost with spec decode and MTP: 7.1 MiB (lean).

## 2. Free DRAM per chip (MEASURED, `tests/test_kv_paged_memory.py`, log `kv_agent_mem40.log`)

`ttnn.get_memory_view(md, DRAM)` per bank (8 banks/chip), 4x8 mesh, `trace_region_size=700e6`, batch 16 (4 users/row), every layer built by `tests/test_decode_steps.py`'s path:

| stage | allocated MiB/bank | free MiB/bank | free GiB/chip |
|---|---|---|---|
| device open (trace region already carved out) | 0 | 3991 | 31.2 |
| after 40 layers (69.1 MiB/layer: 450 MB experts bfp8 + attention + shared expert + mHC/router, incl. the small current KV caches) | 2762.8 | 1228.1 | 9.60 |
| + device Engram weights (layers 1, 14) | 2768.1 | 1222.8 | 9.55 |
| + embedding (replicated, 129280 x 5120 bf16 = 1.23 GiB) | 2925.9 | 1065.0 | 8.32 |
| + head (vocab-sharded bfp8) | 2936.5 | 1054.5 | 8.24 |
| + step inputs, + eager step (compile-pass persistent intermediates, sampling buffers) | 2968.5 | 1022.4 | **7.99** |
| after trace capture / replay | 2968.5 | 1022.4 | 7.99 |

The largest contiguous free block per bank is 1020.6 MiB (no fragmentation). The 7.99 GiB still contain today's tiny KV caches (~40 MiB/chip) which the new layout replaces.
Free DRAM shrinks slightly with batch (MoE decode buffers, mHC streams: tens of MiB at 32 users/row, ASSUMPTION; re-measure with `DSV41_USERS_PER_ROW=32`).

Options that free more (not adopted here, user decides): embedding sharded over the 8 columns (-1.1 GiB used, +1.1 GiB free; the lookup becomes an all-gather or masked lookup + all-reduce);
bfp4 routed experts (the memory note says -9 GB/chip but chained accuracy cost: layer-9 PCC 0.969 vs 0.990); dropping the duplicated latent copies (already counted: the new layout never copies);
trace region 700 MB is carved out of the 31.2 GiB shown above (it is not free for KV; a smaller trace region would give it back: 105 MiB/bank were carved out at open, ~0.8 GiB/chip).

Hidden per-context costs of the CURRENT step-state (`tt/step_state.py`): rope tables `[P, 512]` x 3 (C, S, -S) and the attention mask table `[P, 128 + max_comp]` bf16 are built for every position P.
At P = 64k that is 192 MiB of rope tables + a quadratic mask table; at 1M, 3 GiB. They must become `[P, 64]` cos/sin tables (0.5 KiB/position: 0.5 GiB at 1M, or compute the angle on the fly) and the mask table must go
(with sparse_sdpa masking is carried by the indices).

## 3. Capacity tables

Max uniform context per user (tokens) such that `users_per_row` users fit in `F - reserve` per chip (`capacity_table`, model in `tt/kv_paged.py`). U = users per mesh row; batch = 4U;
every chip of a row holds all U users of that row. `replicated`: latents/index keys identical on the 8 columns of a row (what the TP-over-heads attention needs today: each column reads the whole
selected set); `sharded x8`: pages (or sequence blocks) of the compressed pools and index keys spread over the 8 columns (window ring, compressor state stay replicated). "1M" = capped at the model limit.

No reservations:

| F, formats, layout | U=1 (4) | 2 (8) | 4 (16) | 8 (32) | 16 (64) | 32 (128) |
|---|---|---|---|---|---|---|
| 8.0 GiB, bf16, replicated | 1M | 1M | 653k | 326k | 162k | 80k |
| 8.0 GiB, bf16, sharded x8 | 1M | 1M | 1M | 1M | 1M | 642k |
| 8.0 GiB, lean, replicated | 1M | 1M | 1M | 644k | 320k | 158k |
| 8.0 GiB, lean, sharded x8 | 1M | 1M | 1M | 1M | 1M | 1M |
| 7.0 GiB, bf16, replicated | 1M | 1M | 571k | 285k | 141k | 70k |
| 7.0 GiB, bf16, sharded x8 | 1M | 1M | 1M | 1M | 1M | 560k |
| 7.0 GiB, lean, replicated | 1M | 1M | 1M | 563k | 279k | 138k |
| 7.0 GiB, lean, sharded x8 | 1M | 1M | 1M | 1M | 1M | 1M |
| 6.0 GiB, bf16, replicated | 1M | 981k | 489k | 244k | 121k | 59k |
| 6.0 GiB, bf16, sharded x8 | 1M | 1M | 1M | 1M | 970k | 478k |
| 6.0 GiB, lean, replicated | 1M | 1M | 967k | 482k | 239k | 118k |
| 6.0 GiB, lean, sharded x8 | 1M | 1M | 1M | 1M | 1M | 945k |

Reading it: with the shared layout, batch costs memory mostly through the per-token terms, and the trade-off is smooth: **max context ~ F / (U x 1.6 KiB)** (lean, replicated), e.g. 7 GiB: 563k at 32 users, halving with every doubling of the batch.
Today's code (29.7 KiB/token) would give only 31k tokens at 32 users (bf16, 7 GiB).

With reservations (7 GiB, lean). Spec/MTP reservation = MTP layer weights + their caches + ring slack (`spec_k=32` rows, 3 MTP rings are included; the 450 MB/layer MTP weights are the reservation itself); prefill reservation =
`chunk_tokens_per_chip x 0.6 MiB` (ASSUMPTION: 3 fp32 mHC streams of 80 KiB + q/o heads + MoE dispatch/combine/intermediates; to be replaced by the prefill agent's measurement) + the indexer score block of the densest layer
`chunk x ctx x 2 B` (/8 when sharded; an upper bound: a tiled score + running top-k needs far less).

| reservation | layout | U=1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|---|
| none | replicated | 1M | 1M | 1M | 563k | 279k | 138k |
| MTP/spec 1.0 GiB | replicated | 1M | 1M | 966k | 480k | 238k | 116k |
| MTP/spec 1.5 GiB | replicated | 1M | 1M | 885k | 440k | 217k | 106k |
| MTP/spec 1.5 GiB | sharded x8 | 1M | 1M | 1M | 1M | 1M | 854k |
| prefill chunk 512 tok/chip | replicated | 1M | 1M | 933k | 499k | 257k | 129k |
| prefill chunk 2048 tok/chip | replicated | 1M | 827k | 573k | 354k | 199k | 105k |
| prefill chunk 2048 tok/chip | sharded x8 | 1M | 1M | 1M | 1M | 1M | 846k |
| MTP 1.0 GiB + prefill 512 | replicated | 1M | 1M | 792k | 423k | 217k | 108k |
| MTP 1.0 GiB + prefill 512 | sharded x8 | 1M | 1M | 1M | 1M | 1M | 869k |

(Regenerate with `DSV41_FREE_GIB=7 pytest -s tests/test_kv_paged_capacity.py` or call `capacity_table`.)

For 64k contexts, replicated, lean, 7 GiB, MTP 1.0 GiB + prefill 512: every batch up to 128 fits (U=32: 108k). So **the sharded layout is only needed for 1M contexts at batch > 16 (or > 8 with big reservations)**, not for the 128..64k first target.
Heterogeneous contexts: with the page pool (4.1) the constraint is `sum_u (fixed + 1.62 KiB x L_u) <= budget`, i.e. the tables above are for the uniform case and any mix with the same average fits.

## 4. Paged layout

### 4.1 Decisions

* One **token page = 128 tokens**, one **page table per user for all token-indexed pools** (GPT-OSS style: `[users, max_pages]` int32, ROW_MAJOR, replicated over the columns, user rows sharded over the mesh rows).
  `max_pages = ceil(max_ctx / 128)` (1M: 8192 entries = 32 KiB/user; 64k: 512).
* **Compressed pool (paged), per mesh row**: flat ROW_MAJOR tensor `[1, 1, num_pages * 320, 512]` (bf16 or fp8_e4m3). Page `p` holds, in this order, the rows of all four kv sources
  `[src2: 64 | src8: 64 | src14: 64 | src20: 128]` (ratio-2 sources have 128/2 entries per 128-token page, ratio-1 has 128). Logical entry `j` of source `s` (ratio `r`) is in logical page `(j * r) // 128`;
  physical row = `page_table[u, (j*r)//128] * 320 + row_offset(s) + (j*r % 128) / r` (`PageLayout.phys_rows`, tested in `tests/test_kv_paged_alloc.py`). One page id therefore addresses every layer's latents; the allocator and the page table are shared.
* **Window rings are not paged**: fixed `RING_ROWS` per layer and user. They live in TILE layout `[U, 1, RING_ROWS, 512]` per layer (written in place by `paged_update_cache`, absolute position with `cache_position_modulo=RING_ROWS`, same op as today)
  and are mirrored into the RM pool region the layer's sparse_sdpa reads (4.4).
* **Index keys: per-user contiguous slabs, not paged** (v1). `indexer_score_dsa` reads `[B,1,T,128]` tiles of ONE user per call (`cache_batch_idx` selects the slab; k batch must be 1, MEASURED: `TT_FATAL kB == 1` for more) and has no page-table input.
  Slab = `[U, 1, T_alloc, 128]` bfp8_b/bf16 TILE per owner layer (2, 8, 14: `T_alloc >= L/2 + 32`; 20: `L + 32`), `T_alloc` chosen at admission from the request's max length, rounded to 4096 entries. That costs only the rounding (index keys are
  21% of the per-token bytes). If true paging of the keys becomes necessary: the op's ND-sharded / block-cyclic key-cache machinery (`block_cyclic_*`, `ring_indexer_score_dsa`) is the existing hook; not used here.
* **Dtypes**: latents fp8_e4m3 ROW_MAJOR (sparse_sdpa's `FP8_E4M3` kv format: gather bytes and L1 halve, tilised in-op to bfp8_b, PCC 0.99982 vs 0.99983 bf16, MEASURED) or bf16; index keys bfp8_b (section 6 measures set agreement);
  ring bf16 (TILE).
* **Block size**: 128 tokens = 64 ratio-2 entries or 128 ratio-1 entries per source = a 160 KiB (fp8) / 320 KiB (bf16) pool page; waste < 0.1 MiB/user, page table 32 KiB/user at 1M, and 128 equals the window so a ring rewrite
  is one page. Smaller blocks only reduce the (small) rounding waste and enlarge the tables.
* **Allocation owner**: the host (`tt/kv_paged.PageAllocator`): `admit(user, tokens, reserve_tokens)`, `grow`, `rollback`, `release`, `page_table(users)`. Device code never allocates. The page table is uploaded (replicated over columns, sharded over rows)
  only when it changes (a page boundary is crossed: every 128 tokens per user), into a persistent buffer (trace-safe, `copy_host_to_device_tensor`). Pool size per chip = `(F_kv - fixed) / page_bytes` pages shared by the U users of the row
  (users of different rows have separate pools: they live on different chips).

### 4.2 Who does what (consumer contract)

* **Prefill** (writes chunks): for each chunk the host `grow`s the user to `chunk_end`, then the chunk's compressed rows are written page-aligned: a chunk of C tokens (C multiple of 128) produces C/2 rows for each ratio-2 source and C rows for source 20; every page is one contiguous 160 rows..
  `ttnn.experimental.deepseek_prefill.update_padded_kv_cache` (RM bf16/fp8, in place, slot/offset as device tensors, trace safe) or `ttnn.experimental.slice_write` (RM bf16, host offsets) write whole rows blocks into the physical pages
  (consecutive logical pages need not be consecutive physical pages: one call per page, 8192 calls for 1M tokens: acceptable once, batch the allocator to hand out contiguous runs for prefill to cut it). Window rings: write the last 128 (+slack) tokens of the prompt at slots `pos % RING_ROWS`.
  Index keys: the chunk's keys into the user's slab at entry offset `chunk_start / ratio` (tile aligned for C multiple of 64).
* **Decode** (1 token per step per user): ring row via `paged_update_cache` (as today); compressed append and index key append every `ratio` tokens when a group completes (see 4.4); attention via sparse_sdpa.
* **Spec decode (1 + k tokens verified per step)**: sparse_sdpa takes `S` queries with independent index rows, so the 1+k query tokens of a user are simply `S = U (1+k)` queries: query `i`'s indices contain ring rows only up to its own position (sentinel tail for the not-yet-visible ones) and its own
  selected compressed rows: causality is carried by the indices. Needed from the layout: (a) ring capacity `RING_ROWS = 128 + 32` (the speculative writes of positions p+1..p+k overwrite the slots of p-127..p-127+k which a rollback would still need); (b) pages for `pos + 1 + k` allocated BEFORE the step (`admit/grow(reserve_tokens=1+k)`);
  (c) rollback = `PageAllocator.rollback(user, accepted_len, keep_reserve_tokens)`: lengths/positions shrink, no data is cleared (rejected rows are masked by position and overwritten later); a compressed entry completed by a rejected token is rewritten by the next real token that completes it;
  (d) compressor state: keep `1 + k` snapshots of the previous-token `[kv|score]` (4 KiB per ratio-2 source and snapshot) and restore the accepted one; (e) the indexer for the 1+k queries scores them as the Sq rows of one tile (free: Sq is padded to 32 anyway, but the fused op's causal mask is tile-aligned: the last <= k entries need a per-row additive mask: OPEN, section 7).

### 4.3 Page pool arithmetic (lean, fp8 latents)

One page (128 tokens) = 320 rows x 512 B = 160 KiB (fp8) / 320 KiB (bf16). 7 GiB - 5 MiB x U of fixed state = pool; for U = 8 users/row: 7 GiB / 160 KiB = 44k pages = 5.6M tokens total, i.e. 700k tokens/user uniform (matches the 563k of the table after index keys).

### 4.4 The row-write problem and the chosen scheme

sparse_sdpa needs kv ROW_MAJOR (bf16/fp8_e4m3) in DRAM, indices uint32 row ids (MEASURED: works, section 5). The decode caches are updated by `paged_update_cache`, which requires TILE layout (it supports bf16/bfp8_b/bfp4_b caches and accepts the
new row as bf16). No op writes ONE row at a runtime index into an RM cache: `ttnn.scatter` is out of place (it allocates a full copy of the pool: unusable), `slice_write` takes host offsets (frozen in a trace, bf16 only),
`update_padded_kv_cache` writes 32-aligned slabs (single slot per call) and `indexed_fused_update_cache` is TILE-only. Scheme (uses only existing ops):

1. **Window ring**: the ring stays TILE `[U,1,RING_ROWS,512]` per layer, updated in place by `paged_update_cache` (as today, 1 op). Per layer and step it is untilised (`to_layout`, MEASURED 14 us for U=8, 27 us for U=32 at 128 rows)
   and written with `ttnn.experimental.slice_write` into the layer's STATIC region of the RM pool (`ring_base(layer, user)`; the whole ring is rewritten every step, 128 KiB/user/layer, positions are in the indices, not in the data). Rings of all layers that read a source live in that source's pool tensor
   (rows after the `num_pages * 320` page rows): sparse_sdpa's single kv tensor then contains the layer's ring rows and the selected compressed rows. (slice_write is bf16 only: with fp8 latents the ring regions need either a bf16 pool for rings + a second sparse_sdpa kv... see OPEN 7.2.)
2. **Compressed append**: a small TILE "hot tail" `[U,1,32,512]` per kv source takes the new latent with `paged_update_cache` (row `len % 32`); the open block of every user is a STATIC pending page; the tail is untilised and written into the pending page every step (as the ring);
   when a user's open block is complete (host knows: `len % 32 == 0`) the HOST issues one eager `slice_write` copying the 32 pending rows to a freshly allocated cold page and swaps the page-table entry (rare: one per 32/64 steps per user per source).
   Indices of entries in the open block resolve to the pending rows through the same page-table translation.
3. **Index keys**: slab in TILE, `paged_update_cache` appends (the cache is `[U,1,T,128]` TILE, exactly the op's native shape), no extra copy.

Alternative that avoids RM entirely (**gather to staging + the existing flash SDPA decode**) was evaluated and rejected: it still needs an RM pool for the gather (`ttnn.embedding` weights must be ROW_MAJOR, MEASURED 29 us for 8 users x 512 rows, 95 us for 32 users, per selection group, plus a tilise and a
copy of ring+selection into a `[U,1,640,512]` tile cache per layer), so it has the same write problem plus extra copies, and costs ~2x sparse_sdpa per layer.

## 5. Decode attention at long context

Reference semantics (`Attention.forward`): every compressed layer attends to the 128 window rows plus, once the compressed cache exceeds 512 entries, the 512 entries chosen by the indexer (otherwise all), one softmax with the attention sink.

### 5.1 sparse_sdpa (`ttnn.transformer.sparse_sdpa`, Blackhole, single chip op run per device on the mesh)

Constraints (from `sparse_sdpa_device_operation.cpp` and the V4 shaped tests `ATTENTION_SINK_SHAPES: 64, 640, 128 + 512, 512, kc 128`): q `[1, H, S, 512]` bf16/fp8 ROW_MAJOR DRAM, H a multiple of 32 (our 8 local heads + the kv row are padded to 32, as in today's SDPA decode);
kv `[1,1,R,512]` RM bf16 or fp8_e4m3 (any R, optionally `[B,1,R,512]` + `cache_batch_idx`); indices `[1,1,S,TOPK]` uint32, 0xFFFFFFFF sentinels as a contiguous TAIL, at least one valid per row; `TOPK % k_chunk == 0` (640 = 5 x 128); attention sink `[1,1,1,H]` bf16 RM
pre-divided by the scale (as today); output `[1,H,S,512]` RM bf16. One `S` row = one query token of one user (each with its own 640 rows), so a whole decode batch of a mesh row is ONE call with `S = U` (or `U(1+k)` for spec decode) over the flat pool.
MEASURED on the 4x8 mesh (replicated, trace of 10 calls, H=32, TOPK=640, 131072-row pool, `tests/test_kv_paged_ops.py`):

| users per chip (S) | kv bf16, k_chunk 128 | kv fp8, k_chunk 128 | PCC vs torch golden (bf16 / fp8 kv) |
|---|---|---|---|
| 1 | 58 us | 62 us | 0.99983 / 0.99982 |
| 8 | 60 us | 63 us | 0.99983 / 0.99982 |
| 16 | 65 us | 65 us | 0.99983 / 0.99982 |
| 32 | 83 us | 69 us | 0.99982 / 0.99981 |

(k_chunk 64 is 0-30 us slower.) The cost is flat in the context length (indices-driven gather of 640 rows/user) and nearly flat in the batch: 38 compressed layers x ~65 us = 2.5 ms/token of attention kernels for any context and batch <= 32/row.
Per layer around it: q `[1,T,32,512]` tile -> `[1,32,T,512]` RM (permute + `to_layout`) and the output back (4 small ops), ring untilise + slice_write (~20-30 us at U = 8), sink tensor, indices (per selection group, shared by up to 6-20 layers).

### 5.2 Per-step cost of the selection (indexer) at context L: MEASURED pieces, ESTIMATE composition

Per owner layer and user (bfp8 keys, `indexer_score_dsa`, head_group 0, k_chunk 128; row-parallel `topk_large_indices`, k = 512, scores `[rows, N]` bf16; all measured on one chip's worth of work, N = valid entries):

| N | score (bf16 keys / bfp8 keys) | topk k=512 |
|---|---|---|
| 2k | 55 / 44 us | 33 us |
| 8k | 64 / 45 us | 42 us |
| 32k | 82 / 57 us | 152 us |
| 128k | 190 / 120 us | 570 us |

=> scoring ~0.9 ns per entry and user (bfp8 keys, above a ~40 us floor, memory bound: 256 B/entry), **top-k ~4.4 ns per entry per row** (so ~4.5 ms at 1M entries, ~2.3 ms at 512k) but one call ranks all rows in parallel (the 32 rows of the padded score tile were in the call), so top-k is flat in the users per chip as long as the
rows fit the cores (concatenate the users' row 0 into `[U, N]`). A matmul composite (`relu(q K^T)` then `w @ s`) is batched over users but 2-10x slower per user than the fused op at large N (U=8, N=128k: 1.9 ms vs ~1.0 ms) and has no advantage below N = 8k.

Composition per step at context L (ratio-1 layers have N = L, ratio-2 N = L/2; owner layers 2, 8, 14 (N = L/2) and 20, 24, 28, 32, 36 (all against layer 20's N = L keys, dense scoring + candidate mask as in the reference)):

| L | scoring per user | top-k single level (flat in U) | what it means (U = users per row) |
|---|---|---|---|
| 64k | 5 x 85 + 3 x 57 ~ 0.6 ms | 5 x 0.30 + 3 x 0.15 ~ 2.0 ms | U=8: 4.8 + 2.0 = ~6.8 ms/token (+12% on ~56 ms); U=32: 19 + 2 = ~21 ms (+37%) |
| 1M, replicated | 5 x 0.95 + 3 x 0.47 ~ 6.2 ms | 5 x 4.5 + 3 x 2.3 ~ 29 ms single level: **must use two levels** | U=1: 6 + ~3 = ~9 ms; U=4: ~28 ms |
| 1M, sharded x8 | /8 + floors ~ 1.2 ms | /8 | U=8: ~10 ms + merge + gather comm (below) |

**Two-level exact top-k** (needed above ~100k entries, used by the reference's own hierarchy for layers 24-36): block-max of the score row over blocks of 32 entries, top-512 BLOCKS (an exact superset argument: the 512 best entries lie in <= 512 blocks, and every such block has max >= the 512th best score,
so the 512 best blocks contain all of them), gather those 16384 candidate scores, top-512 over them. Cost: reduce over `[N/32, 32]`, top-k over N/32 (1M: 32k -> ~0.15 ms), a gather of 512 x 32 values, top-k over 16384 (~0.08 ms): ~0.35 ms instead of 4.5 ms. Not implemented yet (the first indexer is single level).
Layer 20's reference candidate step (2048 blocks of 8) is the same idea at another granularity; layers 24-36 could score ONLY inside the 16384 candidate entries (a 4 MB key gather instead of 256 MB of keys) but that needs RM-gatherable keys (OPEN 7.4); the baseline above scores densely and masks.

Page-sharded layout (b), what the decode attention needs per step (ESTIMATE, nothing measured): each column scores/ranks its 1/8 of the keys (local top-512 + values, `ttnn.gather` of the scores by the local ids), all-gather of `8 x 512` (id, score) pairs per user (a few KiB), a top-512 over those 4096 (`topk_large_indices`, ~35 us), then every column
still needs ALL 512 selected rows for its 8 heads (TP over heads): each column fetches its local rows (embedding on its local pool; non-owned rows read a zero row) and the 8 partial buffers are summed with an all-reduce/all-gather over the row of columns: `U x 512 rows x 512 B` (fp8) = 256 KiB/user per selection group,
8 groups per step -> 2 MiB/user/step: U=8 -> 16 MiB through the column fabric per token (~0.5 ms at ~30 GB/s effective; ASSUMPTION about CCL bandwidth, to be measured), plus ~10 small ops per group (~0.1 ms x 8). Flash-decode style (each column attends over its local keys with all 64 heads and the partial (o, max, sum) are merged) moves ~64 KiB/user/layer instead but needs the
q of all heads on every column and a log-sum-exp merge op that does not exist for sparse_sdpa (no LSE output); not recommended.

## 6. Indexer (layer 2) implementation and results

Files: `tt/indexer.py` (`DSV41DecodeIndexer`: projections + RoPE, `indexer_score_dsa` per user, `topk_large_indices` with a device-side `valid_length_tensor`, optional `fp4_q` simulation, `backend="fused"|"matmul"`),
`tests/indexer_ref_state.py` (CPU: builds keys/query with the checkpoint's own Compressor/Indexer code and real layer-2 weights, saves to `/mnt/tt-data/ssinghal/dsv4-kv-state`),
`tests/test_indexer_device.py` (device comparison + latency), `tests/test_indexer_ops_probe.py` (op scaling), logs `kv_agent_idx*.log`.

Device flow per step (U users per chip): `w = x @ weights_proj/64`, `q = qr @ wq_b`, heads onto tile rows, RoPE (`x*C + (x@P)*S`), optional fp4 sim, permute to `[U,32 heads,Sq,128]` + pad Sq to 32 (the op requires `Sq % 32 == 0`, `T % 32 == 0`, k batch 1),
one `indexer_score_dsa` per user (slab `cache_batch_idx = u`, `chunk_start_idx = T_alloc - 32`, `kv_len = T_alloc`: the scalar arguments are frozen in a trace, so the op always scores the allocated length; capture one trace per `T_alloc` bucket),
`topk_large_indices(k=512, valid_length_tensor=N)` (the valid length is a device tensor, so the trace serves every step while the cache fills; entries >= N are never ranked; requires valid length >= 512: for N < 512 every entry is selected and the host builds the arange).

Results (real layer-2 weights, random token embeddings through attn_norm, 4 users/row replicated; reference = the checkpoint's `Indexer.forward` module call, which my formula reproduces exactly):

Top-512 SET agreement with the reference (min over the 4 identical users; the reference's own bf16/fp4 arithmetic makes ~6-12% of the boundary entries differ between ANY two implementations, see the ceiling column) and latency of one decode step of one index layer (one captured trace, all stages; users/chip U):

| N entries | U | variant | set agreement vs ref | ceiling: ref without q-fp4 sim | score PCC vs ref | score-mass recall | step | projections+rope | score | top-k |
|---|---|---|---|---|---|---|---|---|---|---|
| 2048 | 4 | fused, bf16 keys, no fp4-q | 0.9355 | 0.9414 | 0.992 | 0.993 | 352 us | 119 | 202 | 48 |
| 2048 | 4 | fused, bf16 keys, fp4-q sim | **0.9551** | | 0.9965 | 0.997 | 466 us | 257 | 205 | 35 |
| 2048 | 4 | fused, bfp8 keys, fp4-q sim | 0.9551 | | 0.9965 | 0.997 | 465 us | 256 | 200 | 28 |
| 2048 | 4 | fused, bf16 keys, fp4-q, wq_b bf16 | 0.9570 | | 0.9966 | 0.997 | 468 us | 260 | 202 | 33 |
| 2048 | 4 | matmul composite, bf16 keys | 0.9395 | | 0.992 | 0.993 | **139 us** | 102 | 41 | 41 |
| 8192 | 4 | fused, bf16 keys, no fp4-q | 0.9043 | 0.9062 | 0.988 | 0.986 | 557 us (before batched top-k) | 116 | 332 | 124 |
| 8192 | 4 | fused, bf16 keys, fp4-q sim | **0.9414** | | 0.9957 | 0.994 | 509 us | 256 | 222 | 49 |
| 8192 | 4 | fused, bfp8 keys, fp4-q sim | 0.9414 | | 0.9957 | 0.994 | 500 us | 255 | 216 | 49 |
| 8192 | 4 | matmul composite, bf16 keys | 0.9043 | | 0.988 | 0.985 | 205 us | 103 | 90 | 48 |
| 8192 | 8 | fused, bf16 keys, no fp4-q | 0.9043 | 0.9062 | 0.988 | 0.986 | 614 us | 138 | 440 | 61 |
| 32768 | 4 | fused, bf16 keys, fp4-q sim | 0.8848 | 0.8809 | 0.9907 | 0.983 | 723 us | 257 | 314 | 172 |
| 32768 | 8 | fused, bf16 keys, no fp4-q | 0.8711 | 0.8809 | 0.988 | 0.976 | 921 us | 134 | 612 | 192 |

Findings:
* The device top-k matches the checkpoint Indexer to the reference noise floor: with the fp4 simulation of q on the device the set agreement rises above the "no fp4-q" ceiling (0.955 vs 0.941 at 2k, 0.941 vs 0.906 at 8k); the remaining 4.5-6% differing entries all sit at the selection boundary (score-mass recall >= 0.994 at 2k/8k, 0.983 at 32k).
  Agreement falls with N because the density of near-ties at the 512th score grows (random tokens: scores are not peaked); real text has a sharper selection, to be checked with chained activations.
* **bfp8_b index keys are lossless for the selection** (identical agreement to bf16 keys) and 25-35% faster to score (44 vs 55 us at 2k): use bfp8 keys. wq_b in bfp8 vs bf16 changes agreement by 0.002.
* The fp4 simulation of q costs ~140 us per index layer per step (24 small ops, 8 index layers = 1.1 ms/token): worth it only if exact set agreement matters; without it agreement is at the ceiling of the unsimulated formula (-2 to -4 points).
* top-k as ONE call for all users (row 0 of every user concatenated) costs 28-49 us at 2k-8k and 172-192 us at 32k for 4-8 users (vs 4x per-user: 579 us at 32k).
* Scoring is per user (one `indexer_score_dsa` call each, ~50 us floor + 0.9 ns/entry): 4 users 200 us at 2k, 612 us for 8 users at 32k. For N <= 8k the batched matmul composite is faster (139 us vs 352-466 us at N = 2k, U = 4; 205 vs 500 at 8k) and equally accurate;
  above ~32k the fused op wins (per-user 120 us at 128k, bfp8, vs 273 us).
* Whole indexer at 64k context (N = 64k for layers 20..36, 32k for 2, 8, 14), U = 8: ~(5 x (0.14 proj + 0.68 score + 0.30 topk) + 3 x (0.14 + 0.46 + 0.15)) ms ~ 7.9 ms/token with the fp4-q simulation (+14% of the 56 ms token), ~6.8 ms without. Section 5.2 has the other contexts.

Latency numbers are per chip and identical on every chip (same work). Timing includes nothing of the attention itself; the selected ids feed sparse_sdpa (5.1).

## 7. Open items / decisions for the other agents

1. **Row-write primitive** (4.4): confirm the hot-tail + pending-page + host flush scheme, or ask for a small op "write rows `[U, 1, D]` at per-row runtime indices into an RM interleaved cache" (the natural sibling of `paged_update_cache` / `indexed_fused_update_cache` for ROW_MAJOR bf16/fp8). With it the ring and the compressed pool become plain writes and the untilise + rewrite per layer disappears.
2. **fp8 pool vs `slice_write`** (bf16 only): either a bf16 pool (the `bf16` rows of the tables: ~2x the per-token bytes) or write rings/pending pages with `update_padded_kv_cache` (supports fp8 RM, 32-row aligned slabs, one slot per call) - needs a probe.
3. **Spec decode** with the fused indexer: the causal mask of `indexer_score_dsa` is tile aligned (`chunk_start_idx` multiple of 32); the 1 + k query rows need per-row visibility of the newest <= k entries (additive mask on the last 32 columns, or `kv_len` bucket + `valid_length`).
4. **Candidate hierarchy (layers 24-36)**: dense scoring + mask is the baseline; restricting the scoring to the candidate blocks needs a key gather (RM mirror of the keys, or candidate blocks of 32 = whole tile rows; the latter changes `candidate_block_size` semantics).
5. **Two-level top-k** for N > ~100k (5.2) and per-bucket traces (`T_alloc`) for the score op.
6. **Step-state tables** must stop being `[P, 512]` / `[P, 128+max_comp]` (section 2).
7. Re-measure free DRAM at batch 128 (MoE decode buffers) and with `trace_region_size` reduced.
8. All users must share the position today (compressor group-complete decision on the host; one `valid_length_tensor`): with the per-user valid length tensors of the per-user indexer calls this restriction can go on the indexer side.
