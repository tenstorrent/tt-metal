# The 1 M-context decode path (OPT_SPARSE_1M)

Design note for the traced multi-user sparse decode of the DSA layers at full ISL (1 048 576 positions per
user) on the 8x4 Galaxy. Design of record: `~/ashai/GLM_5.3_DESIGN.md` sections 5.3, 6, 7, 8. Code:
`tt/dsa.py` (`SparseDecodeMeta`, `TtDSA._decode_unified`), `tt/indexer.py` (`score_decode_row`,
`select_decode_pools`), the new op `ttnn.experimental.paged_pool_gather`
(`ttnn/cpp/ttnn/operations/experimental/paged_pool_gather/`), tests `test_pool_gather.py`,
`test_dsa_sparse_traced.py`, `test_sparse_1m.py`, `test_sparse_perf.py`. Branch `arg/glm53_sparse1m`
(worktree `wt_sparse1m`, needs its own C++ build: the gather op).

## 1. Where the step time went (before)

The 1 M path existed as a single-chip, single-user, EAGER layer test (`test_cache_1m.py`): 26.5 ms per
layer-3 decode step at 1 M (20 ms at 131 K, 25 ms at 16 K), i.e. (per `_logs/cache1m_final.log`):

| stage (1 user, 1 M, eager) | ms | why |
|---|---|---|
| whole-prefix latent copy: `ttnn.slice` of the user's 32 770 pages + `to_layout(ROW_MAJOR)` | 12-21 | `sparse_sdpa` reads a ROW_MAJOR `[1,1,T,512]` kv; the paged cache is TILE; +2 GiB transient |
| index scoring (`indexer_score_dsa`, 32-row query block, 262 176 pools) | 1.6-2.2 | with the `StageTimer` synchronizations; the op alone is 0.34-0.40 ms (section 3) |
| top-512 pools over 262 K scores + id expansion | 3.5 | `score_untilize` slice/untilize of the 32-row score block, `topk_large_indices`, 4x expansion, concat |
| `sparse_sdpa` (64 heads padded from 16, S=1) | ~1 | one query row through a prefill kernel |
| dense flash-MLA for every slot (discarded for sparse users) | 0.4 (bounded) / 94.5 (unbounded) | `dense_positions` bound |
| rest (stems, cache append, index maintenance, absorb, o_proj) | ~2 | |

and it could not be traced (host chunk geometry: `chunk_start_idx`, `_place_row(r = a % 32)`, the tail
ids built per `(T, query_start)`, the per-user Python loop over `host_positions`) nor run on a multi-row
mesh (`select(row=)` refuses `_mesh_rows != 1`).

## 2. The traced multi-user step (one DSA layer, all users of the chip)

Every user of the chip, dense (position <= 2046) or sparse, is attended by ONE graph with no host-
dependent geometry: per-user *metadata tensors* replace the host chunk geometry and the per-user eager
loop, and the whole-prefix copy is replaced by a *pool gather* of the selected rows into a small compact
cache that the existing batched flash-MLA decode consumes. Per layer step (B users, H local heads):

```
stem matmul [B,4096] -> q_a | kv_a | index k | pool gate | index head weights        (existing, 1 + 5 slices)
q_resid = rmsnorm(q_a); q = q_b(q_resid); c = rmsnorm(kv_a)                         (existing)
paged_update_cache(latent, positions); index maintenance (rings, pool a//4)          (existing, traced)
q_i = q_resid @ wq_b -> [1,Hi,B,D]; w = head weights [1,1,B,Hi]                      (existing stems)
q_big = concat(q_i, 31 zero rows); w_big likewise                                    (2 ops, batched)
for u in users:                                                                       (4 ops per user, unrolled)
    q_blk = q_big[:, :, u:u+32]; w_blk = w_big[:, :, u:u+32]      # row 0 = user u's query
    scores[u] = indexer_score_dsa(q_blk, pools[u], w_blk, chunk_start = FIXED, cache_batch_idx = u)[row 0]
rows = concat(scores[u])                                             [B,1,1,W] bf16 RM    (1 op)
top = topk_large_indices(rows, k = 512, valid_length_tensor = meta.valid_len_rows)   [B,1,1,512]   (1 op, per-row bounds)
ids = typecast(concat(top, meta.tail_pool)) * meta.keep + meta.dense_term -> uint32     (5 ops, batched)
compact = paged_pool_gather(latent_cache, page_table, ids, pool_size = 4)   [B*65,1,32,512]   (1 op, NEW)
o = paged_flash_multi_latent_attention_decode(q_lat, compact, identity page table, cur_pos = meta.cur_pos)
absorb (W_kv_b2), o_proj, all_reduce over the TP columns                             (existing)
```

The dense flash-MLA over the real cache and the per-user `sparse_sdpa` rows are not run at all: a dense
user's compact cache holds tokens `0..a` in order, so the same kernel over `[0, cur_pos = a]` IS the exact
dense attention (bit-identical to the previous path, checked in `test_dsa_sparse_traced`), and a sparse
user's holds its 2048 selected tokens then the <= 3 tail tokens (`cur_pos = 2047 + tail`). NoPE attention
is order-independent, so the compact row order (score order) is irrelevant.

### 2.1 Metadata contract (`SparseDecodeMeta`, per step, host-derived like `pool_ids` / `ring_slots`)

For a user at position `a` (the token being decoded sees `[0, a]`), with `P = index_kpool = 4`,
`K = index_select_pools = 512`, `n = (a+1) // P` complete pools, `t = (a+1) % P` tail tokens:

| tensor | shape / dtype | value | consumer |
|---|---|---|---|
| `positions`, `pool_ids`, `ring_slots` | `[B]` int32 | `a`, `a // P`, `a % P` | cache append, index maintenance (unchanged) |
| `valid_len_rows` (and `valid_len[u]`, one `[1]` per user) | `[B, 1]` uint32 | `max(n, K)` | `topk_large_indices(valid_length_tensor=)`: the exact per-user causal bound in POOLS (pools `>= n` -- the incomplete pool and stale rows -- are never ranked); the `[rows, 1]` form is the per-row metadata mode added to the op (one 4-byte page per row, bounds pushed per row), so ONE call ranks every user's row |
| `tail_pool` | `[B,1,1,1]` uint32 | `n` | the incomplete pool, gathered into compact slot 512 |
| `keep`, `dense_term` | `[B,1,1,513]` fp32 | sparse user: 1 / 0; dense user (`n < K`): 0 / `iota(513)` | the compact pool list = `keep * [top-512 | n] + dense_term`: ranked pools, or pools `0..512` in order (tail pool `n` lands at slot `n`) |
| `cur_pos` | `[B]` int32 | dense: `a`; sparse: `P*K - 1 + t` | the flash-MLA decode's last attended compact row |

All five are refreshed in place by `FunctionalDecoder.update_decode_positions` (`copy_host_to_device_tensor`,
row-sharded on a multi-row mesh), so a captured decode graph reads the live values at every replay; a
captured per-user op instance keeps its static `cache_batch_idx = u`. `GLM53_SPARSE_DECODE=eager` restores
the previous path (A/B reference).

### 2.2 Index scoring without host geometry

`indexer_score_dsa`'s causal mask is a function of `chunk_start_idx` (a per-dispatch runtime scalar that
a trace freezes) and of the query ROW; a batched `[B]` k is refused (`k batch must be 1 unless a cache slot
is selected`), and the trace-safe `chunk_start_idx_tensor` exists only on the fused-ring factory. So the
scorer runs once per user with `cache_batch_idx = u` and a FIXED chunk start in the LAST 32-token block of
the pool allocation (`pools_per_user * 4 - 32`): row 0 then sees every pool but the last 8 of the
allocation, i.e. the kernel's causal mask is disabled, the pool cache gets 8 pools of slack
(`DECODE_SCORE_SLACK`, `TtIndexCache`), the query always sits in row 0 (no `r = a % 32` placement), and the
user's exact causal bound is applied by the top-k's per-user `valid_len` instead. The pools between `n`
and the slack score against their current (finite) content and are never ranked.

### 2.3 The latent gather (`paged_pool_gather`, new op)

Candidates considered for reading the paged TILE latent cache without the whole-prefix TILE -> RM copy:

| candidate | verdict |
|---|---|
| `sparse_sdpa` over the cache itself | reads ROW_MAJOR kv only; a ROW_MAJOR cache is refused by `paged_update_cache` / `paged_fill_cache` (TILE only) and by the flash-MLA decode; S=1 prefill kernel (poor core use) |
| `sparse_sdpa_msa` (TILE K/V, block ids) | block granularity >= 32 tokens; the selection is 4-token pools |
| `ttnn.embedding` row gather | ROW_MAJOR table only |
| `ttnn.gather` | TILE but needs a full-width `[2176, 512]` index per user and reads whole tiles |
| the DeepSeek `_sparse_mla` path | `sparse_sdpa` over a replicated RM KVPE table: the same RM contract |
| flash-MLA decode over a gathered compact cache (design 5.3 "composite first") | chosen: one existing batched decode kernel for every user; the gather is a pure data-movement kernel |

A bf16 32x32 tile stores four 16x16 faces; a 4-token pool (rows `4j..4j+3`, never crossing a face) is two
contiguous 128-byte runs per tile (one per face column), so a pool is 32 reads of 128 B = 4 KB (no read
amplification) and the same 32 writes into the compact tile block. The op takes the cache
`[max_blocks,1,32,512]`, the page table (any mapping, looked up per pool as a 64-byte fragment) and
`pool_ids [B,1,1,N]` (`N = 513`: 512 ranked pools + the tail pool), zero-fills sentinel / out-of-range
pools and the padding rows, and writes `[B*65,1,32,512]`; work items are (user, output block), one
data-movement kernel per core with two L1 blocks alternating reads and writes. Measured on chip 0 with a
1 M-position cache (`test_pool_gather.py`): 0.061 ms for 1 user, 0.183 ms for 4, 0.339 ms for 8 (2 / 8 / 16
MB moved); bit-exact vs a torch gather with permuted page tables, sentinels and out-of-range ids.

### 2.4 Index-scoring cost at 8 users x 1 M per chip (the step's bandwidth term)

Measured on chip 0 (`_logs/sparse1m_probe_score*.log`, `indexer_score_dsa`, 32 index heads, 32-row query
block with one real row, `k_chunk_size = 32`, per-user call):

| pools per user (positions) | keys read | ms per user | effective |
|---|---|---|---|
| 1 024 (4 K) | 256 KiB | 0.085-0.092 | latency bound |
| 32 768 (128 K) | 8 MiB | 0.085-0.107 | ~90 GB/s |
| 262 176 (1 M) | 64 MiB | 0.34-0.40 | ~170-190 GB/s |

The head count barely matters (a 1-head call with the 32 heads as rows: 0.36 ms), so the kernel streams
the pooled keys at roughly a third of the chip's DRAM bandwidth; `k_chunk_size` 64-256 does not help
(0.34-0.35 ms) and changes the bf16 accumulation path (kept at 32, the validated numerics). The pooled
keys are bf16 today (`64 MiB` per user-layer at 1 M; the design budgeted bfp8, `32 B` per token). Per
layer step at 8 users x 1 M the scoring term is therefore 8 x 0.36 = 2.9 ms, against a DRAM floor of
512 MiB / 512 GB/s = 1.0 ms (bf16) or 0.5 ms (bfp8). The design budget (~0.7 ms per layer) IS the bfp8
floor of this read: meeting it needs (a) bfp8 pooled keys (the kernel accepts bfp8 k; `paged_update_cache`
writes bfp8 caches; an accuracy gate like the experts'), (b) a scorer that streams at DRAM speed and
batches users in one program (per-user `cache_batch_idx` as a `[B]` tensor and a per-batch `valid_end`,
so one dispatch covers the chip's users and the 8 x 16 MB of 32-row outputs shrink to 8 rows). Both are
kernel work on `indexer_score_dsa`; this note records the measured geometry they have to beat.

`topk_large_indices` over 262 176 scores: 1.14 ms for ONE row and 1.16 ms for 8 rows in one call (the rows
run on different cores; a row is serial over its 512-element windows), i.e. per-user calls cost 8 x 1.14 ms at
8 users. The op therefore got a per-row `valid_length_tensor` (`[rows, 1]`: the reader reads page `row`,
derives that row's bounds and pushes them to the compute mailbox per row; the 1-element form is unchanged)
and the decode path ranks all users in one call (`GLM53_TOPK_BATCHED=0` restores the per-user calls).

## 3. The user-partitioned cache over the row's 4 columns (deliverable 3, design)

Today every column chip of a row holds every user of the row (cache + index replicated 4x). Design 5.3:
column `c` owns the pages of users `[c*B/4, (c+1)*B/4)` of its row; attention runs where the cache is,
with all 64 heads of the owned users.

| item | replicated layout (today) | user-partitioned layout |
|---|---|---|
| latent cache / index per chip | `max_batch` users x `max_seq_len` | `max_batch/4` users (+ 1 scratch slot) x `max_seq_len`: 4x the ISL per chip, or 4x the users |
| `q_lat [1,B,16,512]` before attention | local heads, all users | `all_to_all_async_generic(in_dim=1 (users), out_dim=2 (heads), cluster_axis=1)` -> `[1,B/4,64,512]` on the owner (as DeepSeek-V3 decode); H = 64 on every kernel |
| attention output | `[1,B,16,512]` | inverse all-to-all (`in_dim=2, out_dim=1`) -> `[1,B,16,512]` for `W_kv_b2` / `o_proj` |
| indexer queries `q_i [1,Hi,B,D]`, `w [1,1,B,Hi]` | replicated (computed from the replicated x) | `ttnn.mesh_partition(dim=users, cluster_axis=1)`: the chip keeps its own B/4 rows (no exchange) |
| latent / index rows to append (`c`, `k`, `g` `[1,1,B,*]`) | replicated, all users written on every chip | `mesh_partition` over the columns, then the same paged writes for the B/4 local users |
| metadata (`positions`, `pool_ids`, `ring_slots`, `SparseDecodeMeta`) | row-sharded `[B]` | sharded over BOTH axes: host `[rows*B]` -> `shard_2d(rows, cols)` -> `[B/4]` per chip; `valid_len` lists `B/4` tensors |
| prefill (`user = u`) | writes the cache / index of `u` on every column | the owner column writes `u`'s pages; the other columns write the SCRATCH slot through a per-chip page table (a per-chip-different page-table tensor is an ordinary sharded upload); the attention of the prefill rows is unchanged (TP over heads, K/V from the chunk's own latent + the owner's prefix: the prefix rows must be gathered from the owner for the dense rows `< 2048` -- one column all-gather of the `[start, 512]` prefix per chunk, or chunk prefill keeps the prefix replicated until the owner-only cache lands for decode) |
| page tables | `[B, blocks]` identical on the 4 chips | `[B/4, blocks]` per chip; the compact-cache identity table `[B/4, 65]` |
| collectives per DSA layer | 1 all-reduce | 1 all-reduce + 2 all-to-alls of `B x 64 x 512 x 2 B / 4` = 16 KB per user (~20-40 us each) |

The `sparse_sdpa` H % 32 == 0 padding (`_pad_heads_rm`) disappears for decode with the owner holding all
64 heads (the prefill sparse rows keep it until the prefill path moves to the same compact-gather design).
Memory per chip: 11 layers x (1 GiB latent + 64 MiB index) per user-slot at 1 M; 4 x 1 M-slots per row
(one per column) = the design's "4 users/row, 1 M each" row of section 6; a user-slot's 11.4 GiB is the
reason 8 users x 1 M cannot coexist on one chip -- the 8-users-at-1 M figures below are per-step COSTS at
that batch (cache allocated for one layer only).

Host-side code for this layout (not yet run on the block: rows 4-7 are busy and the 8x4 fabric has a link
fault): `TtLatentCache(users_per_chip, scratch=True)`, `GLMMesh.all_to_all_cols(x, in_dim, out_dim)`,
`SparseDecodeMeta.from_positions(..., shard_cols=True)`, `TtDSA.decode_forward` with the two exchanges
around `_decode_unified`; validation plan = `test_mesh_dsa.py` on the rows-4-5 (2,4) block: per-user
PCC vs the single-chip traced path (same tokens, same positions), then one user per column at 1 M.

## 4. Results (one chip, layer 3, real weights; logs under `~/ashai/glm53/_logs/sparse1m_*.log`)

### 4.1 Correctness

| test | what | result | log |
|---|---|---|---|
| `test_pool_gather.py` | `paged_pool_gather` vs a torch gather: permuted page tables, sentinels, out-of-range ids, B 1/2/3/4, N 513/20/1, a second call with new ids | bit-equal in every case | `sparse1m_test_pool_gather.log` |
| `test_dsa_sparse_traced.py::test_two_users_dense_sparse_traced` | 2 user slots, permuted page table: user 0 = R2040 (positions 2040..2051: dense form, the 2047 switch, the 2050 HF bound), user 1 = R4097 (4097..4108); steps 0-1 eager, trace captured on 2, replayed 3..11; every (user, step) layer output vs its HF dump | 26/26 samples > 0.9998 (worst 0.999847 = R4097 step 4, the step with the lowest id overlap); prefill 0.999883 / 0.999930; the dense user's new-path output is BIT-EQUAL to the eager reference path, the sparse user's 0.999988 vs it | `sparse1m_test_traced_2.log` |
| `test_dsa_sparse_traced.py::test_selected_pools_vs_hf` | the traced path's selected pools (+ tail) vs HF `index_ids`, R4097 steps 0-11 | 97.9-99.0 % of the HF ids per step (the eager path: 98-99 %); layer pcc 0.99985-0.99998 | same |
| `test_sparse_1m.py` | 1 user seeded to 16 K / 131 K / 1 M (synthetic context, host twin = `test_cache_1m`), decode at N, N+1 vs the HF restricted forward | new path 0.999803 / 0.999838 (16 K), 0.999767 / 0.999757 (131 K), 0.999680 / 0.999706 (1 M); the eager reference path at the same steps 0.999792 / 0.999822, 0.999747 / 0.999752, 0.999671 / 0.999695 (new vs eager 0.99992-0.99994); id overlap 0.94 / 0.92 / 0.87 (the synthetic-context regime of `test_cache_1m`: near-ties at the top-512 boundary) | `sparse1m_test_1m_2.log` |

### 4.2 Step time, DSA attention block alone (traced, ms per step; the eager reference path for 1 user)

| positions | 1 user (eager reference) | 4 users | 8 users | 4 / 8 users with per-user top-k calls |
|---|---|---|---|---|
| 16 384 | 1.888 (2.4) | 3.421 | 5.308 | 3.429 / 5.393 |
| 131 072 | 2.006 (3.7-6.7) | 3.540 | 5.429 | 3.924 / 6.356 |
| 1 048 576 | 3.012 (13.6-13.8) | 4.532 | 6.428 | 7.869 / 14.252 |

(`test_sparse_1m.py`, `sparse1m_test_1m_3.log` (batched top-k) and `_2.log` (per-user top-k calls); the eager
reference path is the same attention module through `dense_positions` + the per-user `sparse_sdpa` rows, warm,
no stage timer (wall, host-noise bounds) -- its whole-layer figure in `test_cache_1m` was 26.5 ms.) The stage
probe at 1 M (`sparse1m_probe_stages.log`, eager walls): score op 0.38 ms per user, row slice 0.06, top-k 1.14
per call, pool gather 0.17 / 0.50 ms (1 / 8 users), compact flash-MLA 0.41 / 0.53 ms.

### 4.3 Step time, whole layer 3 (mHC + DSA + the single-chip 288-expert MoE), traced

| positions | 1 user | 4 users | 8 users | DSA block ops (ttnn / device) for 1, 4, 8 users |
|---|---|---|---|---|
| 4 096 | 5.933 | 9.067 | 13.039 | 110 / 60, 142 / 90, 178 / 122 |
| 131 072 | 6.067 | 9.206 | 13.180 | |
| 1 048 576 | 7.053 | 10.194 | 14.163 | |
| 1 048 576, per-user top-k calls | 7.046 | 13.505 | 21.807 | 107 / 60, 145 / 93, 189 / 129 |

(`test_sparse_perf.py`, `sparse1m_test_perf_2.log` / `_perf.log`; the single-chip MoE dominates the short-context
figures: the 2x4-block layer-3 step at 4 users/row is 2.42 ms (mesh README). The ISL-dependent increment
4 K -> 1 M is now ~1.1 ms at every batch; the per-user terms left are the scorer call (0.38 ms, 64 MiB of
bf16 pooled keys) and the q / w / row slices.)

Before / after at 1 M, layer 3: 26.5 ms eager layer step for ONE user (`test_cache_1m`, not traceable) ->
7.05 ms traced layer step for 1 user, 10.2 ms for 4, 14.2 ms for 8 (attention block 3.0 / 4.5 / 6.4 ms), every
user in one graph; against the design's ~0.7 ms per layer at 8 users x 1 M the attention block is 9x off, the
scorer read (section 2.4) being the floor-setting term.

## 5. What remains

1. The per-user scorer: a batched, DRAM-speed `indexer_score_dsa` decode mode (per-batch `cache_batch_idx` /
   `valid_end` tensors, one output row per user) would take the 8-user attention step at 1 M from 14 ms towards
   the ~1 ms (bf16) / ~0.5 ms (bfp8 pooled keys) floor; bfp8 pooled keys halve the bytes today (accuracy gate).
2. The op count: 4 ops per user (slice q, slice w, score, row slice) + ~45 fixed; the batched scorer removes
   the per-user part; the metadata blend (5 ops) can fold into the gather kernel (dense users = pools in order,
   tail slot) once it takes the positions tensor. A multi-core top-k row (the 512-window chain is serial on one
   core: 1.14 ms per 262 K row) is the next per-step term at 1 M.
3. Deliverable 3 on the (2,4) block (`test_mesh_sparse_partitioned.py`, host-side code in place) and the 1 M
   mesh step; then the 8x4 run.
4. Prefill at long ISL still reads the whole prefix for `sparse_sdpa` (the untilize); the same compact-gather
   design applies to the prefill sparse rows (gather per 32-row query block).
