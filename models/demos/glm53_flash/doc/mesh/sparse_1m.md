# The 1 M-context decode path (OPT_SPARSE_1M)

Design note for the traced multi-user sparse decode of the DSA layers at full ISL (1 048 576 positions per
user) on the 8x4 Galaxy. Design of record: `~/ashai/GLM_5.3_DESIGN.md` sections 5.3, 6, 7, 8. Code:
`tt/dsa.py` (`SparseDecodeMeta`, `TtDSA._decode_unified`), `tt/indexer.py` (`score_decode_row`,
`select_decode_pools`), the model-local generic op `custom_ops/paged_pool_gather/` (`op.py`, `kernels/`,
README), tests `tests/custom_ops/test_paged_pool_gather.py`, `test_dsa_sparse_traced.py`, `test_sparse_1m.py`,
`test_sparse_perf.py`. Branch `arg/glm53_sparse1m` (worktree `wt_sparse1m`); runs on the main-tree build
(no `ttnn/cpp` / `tt_metal` changes: the gather is a `ttnn.generic_op` whose kernel is JIT-compiled from the
model directory).

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
compact = PagedPoolGather(latent_cache, page_table, ids, out)   [B*65,1,32,512]   (1 generic op, custom_ops/)
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

### 2.3 The latent gather (`custom_ops/paged_pool_gather`, a model-local generic op)

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
pools and the padding rows, and writes the caller's persistent `[B*65,1,32,512]` output (`TtDSA` keeps one
per batch size: under trace the generic op's runtime args carry the io tensors' addresses); work items are
(user, output block), one data-movement kernel per core with two L1 blocks alternating reads and writes
(`ttnn.generic_op`, one `KernelDescriptor` with a `ReaderConfigDescriptor`, three CBs; contract in the op's
README). Measured on chip 0 with a 1 M-position cache: 0.061 ms for 1 user, 0.183 ms for 4, 0.339 ms for 8
(2 / 8 / 16 MB moved); bit-exact vs the torch golden with permuted page tables, sentinels and out-of-range ids,
including under trace replay with refreshed ids.

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
8 users. A per-row `valid_length_tensor` (`[rows, 1]`: the reader reads page `row`, derives that row's bounds
and pushes them to the compute mailbox per row) was prototyped as a 40-line change to the shared op and
measured (8 rows with per-row bounds: 1.138 ms, every row's ids inside its own bound, score multisets equal to
torch's per-prefix top-k); under the model-local-ops rule it has to be re-homed as a custom op (the shared op's
three kernels copied next to the model with that reader change), so the decode path ranks per user today
(`GLM53_TOPK_BATCHED=1` selects the batched call once that op exists; the shared op is untouched).

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
| `tests/custom_ops/test_paged_pool_gather.py` | the generic op vs its torch golden: permuted page tables, sentinels, out-of-range ids, B 1/4/32, N 513/20/1, a second call with new ids, trace replay with refreshed ids | bit-equal in every case (C++ prototype: `sparse1m_test_pool_gather.log`; generic op on the main build: `sparse1m_main_customop.log`) | |
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

### 4.4 Main-tree build (the rule of 2026-10-20: model-local ops only)

The same tests on the main-tree build (`TT_METAL_HOME=$W`, `GLM53_KDA_FUSED=0`; the gather as the generic op,
per-user top-k calls), all green (`sparse1m_main_{customop,traced,regress,1m,perf}.log`):

| what | result (main build) |
|---|---|
| `tests/custom_ops/test_paged_pool_gather.py` (7 cases) | bit-equal to the golden incl. trace replay; eager call 0.146 / 0.233 / 0.357 ms at 1 / 4 / 8 users x 513 pools on the 1 M cache (the Python descriptor build is per call; inside a traced step the kernel alone runs) |
| `test_dsa_sparse_traced.py` (2 tests) | 26/26 (user, step) samples > 0.9998, worst 0.999847; dense user bit-equal to the eager reference path; ids 97.9-99.0 % of HF |
| regression: `test_decode_trace` l3, `test_decoder_layer` (4), `test_batch8` B=8 l3, `test_dsa_sparse`, `test_dsa_long` (4) | 13/13 passed |
| `test_sparse_1m.py` | PCC vs HF-restricted 0.999803 / 0.999838 (16 K), 0.999767 / 0.999757 (131 K), 0.999680 / 0.999706 (1 M); attention step 1.89 / 3.43 / 5.38 ms (16 K), 2.15 / 3.93 / 6.36 (131 K), 2.99 / 7.88 / 14.25 (1 M) at 1 / 4 / 8 users; eager reference 2.4 / 4.0 / 13.6 ms (1 user) |
| `test_sparse_perf.py` (layer 3, traced) | 5.92 / 9.10 / 12.78 ms (4 K), 6.06 / 9.64 / 14.15 (131 K), 7.05 / 13.59 / 21.86 (1 M) at 1 / 4 / 8 users; DSA block 60 / 93 / 129 device ops |

### 4.5 `custom_ops/topk_rows` (the per-row top-k as a model-local op; default `GLM53_TOPK_BATCHED=1`)

Main-tree build, `sparse1m_topkrows_{op,traced,1m,perf,regress}.log`, all green:

| what | result |
|---|---|
| `tests/custom_ops/test_topk_rows.py` (10 cases) | every row's ids inside its own bound and score multisets equal to torch's per-prefix top-512 (W 1056 / 262 176, B 1/4/8/32), trace replay with refreshed bounds; 262 K rows: 1.33 / 1.33 / 1.26 ms for 1 / 4 / 8 rows in one call vs 1.15 / 4.70 / 9.38 ms of per-user shared-op calls |
| `test_dsa_sparse_traced.py` | identical to 4.4 (worst 0.999847; ids 97.9-99.0 % of HF) |
| `test_sparse_1m.py` | PCC identical to 4.4; attention step 1.89 / 3.55 / 5.31 ms (16 K), 2.00 / 3.55 / 5.43 (131 K), **3.01 / 4.53 / 6.41 ms (1 M)** at 1 / 4 / 8 users |
| `test_sparse_perf.py` | layer-3 step 5.94 / 9.14 / 13.13 ms (4 K), 6.07 / 9.27 / 13.40 (131 K), **7.05 / 10.26 / 14.23 (1 M)**; DSA block 60 / 90 / 122 device ops |
| regression (`test_decode_trace` l3, `test_batch8` B=8 l3, `test_dsa_long` x4) | 7/7 passed |

The per-user term left at 1 M is the scorer call (0.38 ms + the 32-row output and the q / w / row slices per user):
`custom_ops/index_score_decode` (section 5, item 1) scores all users in one program with the 32 heads as the rows
of one query tile row (behind `GLM53_SCORER_BATCHED=1` until validated).

### 4.6 `custom_ops/index_score_decode` (the batched decode scorer; `GLM53_SCORER_BATCHED`)

One generic-op program scores every user's single query against its pooled keys: the user's 32 index heads are
the 32 ROWS of one query tile row, so per 8-k-tile chunk the compute runs Dt = 4 `matmul_block` calls (rt 1 x
ct 8, K transposed, ReLU in the packer) accumulated in DEST and then 8 single-tile matmuls with a "w-row tile"
(row 0 = the head weights) whose row 0 is the head-weighted score of 32 pools; the writer keeps row 0 of each
out tile (two 32-byte face pieces) and writes one 512-byte line per chunk into the user's `[B,1,1,W]` bf16 score
row. Items (user, k-band of whole chunks) are dealt over the grid (~grid / B bands per user). No causal mask (the
per-row top-k bound of `topk_rows` applies it). Versus the shared op's per-user calls: no 32-row query block
with one real row (32x less matmul work per key tile), no 32-row score output (16 MB per user) and no row slice.

`tests/custom_ops/test_index_score_decode.py` vs the fp32 golden (random bf16 q / k / w, W 1280 and 262 400, B 1 /
4 / 8 / 32): per-user PCC 0.99995-0.99997, top-512 overlap 0.97-0.998 (bf16 ranking, as the shared scorer);
trace replay with refreshed queries; streaming 64 MiB of bf16 keys per user in 0.49 / 0.76 / 1.40 ms for 1 / 4 / 8
users (128 / 328 / 359 GB/s; the 1-user call is latency-bound on 130 items of 8 chunks). bfp8_b keys
(`ttnn.typecast` of the same keys): PCC 0.99991 vs the golden, top-512 overlap 0.967-0.971 vs the golden and
0.957-0.959 vs the bf16 run (`sparse1m_scorer_op_6.log`). Pitfalls met: compute kernels use `void kernel_main()`
(not the `NAMESPACE`/`MAIN` form); `matmul_block` multiplies ONE inner tile per call (the caller loops the inner
dimension in DEST); a 32-byte DRAM read at a 32-byte offset lands wrong on Blackhole (64-byte alignment: read the
whole face and copy the row); switching the srcA operand between a bfp8 and a bf16 CB needs an explicit
`reconfig_data_format` (the matmul init alone leaves the old unpack format).

The decode path with it (`GLM53_SCORER_BATCHED=1`, now the default; `sparse1m_scorer_{traced,1m,1m_bfp8}.log`):

| what | bf16 pooled keys | bfp8_b pooled keys (`GLM53_POOL_DTYPE=bfloat8_b`) |
|---|---|---|
| `test_dsa_sparse_traced.py` | 26/26 samples > 0.9995 (worst 0.999596); ids 97.9-99.0 % of HF | (not run) |
| `test_sparse_1m.py`, PCC vs HF-restricted at 16 K / 131 K / 1 M | 0.999834 / 0.999852, 0.999813 / 0.999808, 0.999782 / 0.999796 (the per-user shared scorer: 0.999803 / 0.999838, 0.999767 / 0.999757, 0.999680 / 0.999706: the 32-head reduction now accumulates in DEST) | 0.999839 / 0.999813, 0.999824 / 0.999810, 0.999789 / 0.999772 |
| id overlap vs HF (synthetic context) | 0.941 / 0.943, 0.912 / 0.922, 0.865 / 0.883 | 0.945 / 0.943, 0.912 / 0.906, 0.881 / 0.871 |
| attention step, 1 / 4 / 8 users, 16 K | 1.65 / 2.38 / 3.23 ms | 1.60 / 2.10 / 2.68 ms |
| 131 K | 1.78 / 2.49 / 3.35 ms | 1.72 / 2.23 / 2.80 ms |
| **1 M** | **2.76 / 3.48 / 4.34 ms** (per-user scorer + batched top-k: 3.01 / 4.53 / 6.41; per-user both: 2.99 / 7.88 / 14.25) | **2.71 / 3.21 / 3.79 ms** |

Whole layer 3 (`test_sparse_perf.py`, `sparse1m_scorer_perf.log`): 5.69 / 7.98 / 11.21 ms at 4 K, 5.83 / 8.11 /
11.29 at 131 K, **6.90 / 9.09 / 12.26 ms at 1 M** for 1 / 4 / 8 users (per-user scorer: 7.05 / 10.26 / 14.23; the
first traced path: 7.05 / 13.59 / 21.86); the DSA block is 52 / 55 / 55 device ops (96-103 ttnn ops), no longer
growing with the user count. Regression under the default (`sparse1m_scorer_regress.log`): `test_dsa_sparse_traced`
(2), `test_decode_trace` l3, `test_batch8` B=8 l3, `test_dsa_long` (4): 9/9 passed.

At 8 users x 1 M the attention block is now 4.3 ms (bf16) / 3.8 ms (bfp8) against the 0.7 ms budget: the scorer
read (1.4 / 0.7 ms at 359 GB/s), the top-k (1.26 ms per call, serial per row on one core) and ~45 fixed small ops
(stems, index maintenance, blends, gather 0.5 ms, compact attention 0.5 ms) are what is left. The bfp8 pool cache
is a knob (default bf16): the selected sets move by ~4 % vs bf16 on random keys while the layer output vs HF is
unchanged; the real-prompt accuracy gate (eval set) decides it, as for the experts.

### 4.7 The pool-list blend folded into the gather (`paged_pool_gather` takes the top-k output + the tail pool)

`PagedPoolGather(cache, page_table, top_ids [B,1,1,512], tail_pool [B,1,1,1], out)`: the slot -> pool mapping
(the ranked pools; the pools in order for a dense user, `n < 512`; the tail pool at slot 512) is done in the
kernel, so the 5 blend ops (concat, typecast, multiply, add, typecast) and the host `keep` / `dense_term`
tensors are gone; `SparseDecodeMeta` carries `valid_len_rows`, `tail_pool`, `cur_pos` (+ the per-user
`valid_len` list for the eager fallback). Main build, `sparse1m_fold_{gather,traced,1m,perf}.log`: the op bit-exact vs
its golden (B 1/4/32, K 512/16, sentinels, out-of-range, dense and sparse users, trace replay); the decode path
unchanged in PCC (two-user worst 0.999598; 1 M 0.999782 / 0.999796); DSA block **45 / 48 / 48 device ops** at
1 / 4 / 8 users (was 52 / 55 / 55); attention step at 1 M 2.73 / 3.42 / 4.28 ms. The whole-layer sweep in that run
(8.24 / 10.62 / 13.27 ms at 1 M, 7.12 ms at 4 K for 1 user) came out 1.1-1.5 ms ABOVE the previous run at every
point including 4 K, against a faster traced attention step: host load during that sweep (other agents' jobs);
to be re-measured.

### 4.8 Two-stage top-k (`custom_ops/topk_rows` segmented mode + scorer writer v2; default on wide caches)

The scorer's writer v2 writes -inf at and beyond each user's bound (`valid_rows`) and lays the row out as S
segment pages; stage 1 ranks every segment row on its own core and also emits the ranked VALUES (the survivor's
value tiles, materialized like the index tiles, into user-major `[B,1,1,S*512]` tables); stage 2 ranks the S*512
candidate values per user (a bf16 typecast of the table) and its writer translates the winning slots back to pool
ids through the stage-1 id table. -inf candidates rank last, so no per-row bound is needed anywhere. S =
`segments_for(W)`: the largest S <= 8 dividing W/256 with W/S >= 512, and 1 below W = 65536 (two launches + the
typecast lose on short rows). Main build, `sparse1m_seg_{topk_3,traced,1m}.log`:

| what | result |
|---|---|
| `test_topk_rows.py::test_topk_rows_segmented` | ids inside the bound and score multisets == torch (W 262 400 B 1 / 8 with S = 5, W 2560 B 4); stage-1 values consistent with the scores; 262 400 x 8 rows: 0.64 ms vs 1.21 ms single-stage (B = 1: 1.16 vs 1.24; W = 2560: 0.64 vs 0.35, hence the width threshold) |
| `test_index_score_decode.py::test_index_score_decode_masked_segmented` | masked prefix bit-equal to the plain output, tail -inf; segmented layout == masked |
| `test_dsa_sparse_traced.py` | 26/26 > 0.9995 (worst 0.999598) |
| `test_sparse_1m.py` | PCC vs HF 0.999773 at 1 M (unchanged regime); attention step **1.85 / 2.55 / 3.41 ms at 1 / 4 / 8 users**, the same at 16 K and 131 K (the step no longer depends on the position; the cache is the 1 M one) -- from 2.73 / 3.42 / 4.28 |

Against the 0.7 ms budget at 8 users x 1 M the attention block is 3.4 ms; the scorer's 1.4 ms read (bf16; 0.7 with
bfp8 pools) and ~45 small fixed ops (stems, index maintenance, two top-k stages, gather, compact attention) remain.

### 4.9 First (2,4) block run of the partitioned cache (rows 6-7, 2026-10-20 16:46) -- two wiring bugs, fixed host-side

`_pyfix/run_block_chain.sh` on the merged tree (13d0f501b5b): the CCL probe passed (17 s); `test_mesh_sparse_partitioned.py`
failed twice, both host-side (no device fault); the decoder B=32 and `test_mesh_dsa` steps were cut short by the
board reset (`sparse1m_block_*.log`).

| failure | cause | fix (untested on device: the board was down) |
|---|---|---|
| `test_partitioned_two_rows_four_users`: `NotImplementedError: sparse selection on a multi-row mesh is not wired` (indexer.select, prefill of R4097) | `indexer_score_dsa` derives every device's chunk start from its mesh coordinate (`seq_shard_axes=[0]`: row r scores with start + r*Sq; `[]`: the whole mesh is the ring; the on-device metadata form is fused-ring + block-cyclic + P=1 only) -- there is no "replicated rows" form | prefill scores on multi-row meshes come from `custom_ops/index_score_decode` in its new prefill form: the T queries of the prompt are T users of ONE cache slot (`slot=`), the first `cols` (kv_len rounded to 256) pools are scored, row t's causal bound floor((start + t + 1)/P) is written as -inf by the per-row mask. `GLM53_PREFILL_SCORER` = auto (custom on multi-row meshes) / custom / shared; `queries_and_weights(rows_form=True)` hands q as [1,T,Hi,D] (one reshape fewer than the shared form). Tests: `test_index_score_decode_shared_slot` (B = 4 / 70, cols), `test_index_score_decode_prefill_rows_vs_shared_op` (the form vs `indexer_score_dsa` on one chip: prefix PCC, top-512 overlap, mask pattern); then `GLM53_PREFILL_SCORER=custom` through the HF-referenced single-chip tests (`_pyfix/run_chip0_prefill_scorer.sh`) |
| `test_partitioned_1m_step`: `TT_FATAL dims must be unique` (xtensor/partition.cpp:61) from the metadata upload | `ShardTensor2dMesh(dims=(0, 0))` -- the same host dim over both mesh axes -- is rejected by this build's `chunk_ndim` (the tt_dit / deepseek precedent no longer holds) | `GLMMesh.shard_2d(d, d)` returns `ttnn.ShardTensorToMesh(dim=d)`: chunk i of rows*cols goes to chip (i // cols, i % cols), the row-major order the partition's user list already uses |

The prefill form costs one pass over `cols` pools per query (327 KB at the test's 1280-pool width, ~2-3 ms for 2049
sparse rows; at 262 K pools the whole-prompt prefill belongs to the chunked / SP engine anyway). The eager decode
selection (`select(row=)`) keeps the shared op and its single-row guard.

## 5. What remains

1. Done: the batched scorer (4.6), the bfp8 pool-cache knob, the blend folded into the gather (4.7), the two-stage
   top-k (4.8). Next: the 1-user scorer latency (0.49 ms for 64 MiB: fewer, longer bands), a fused stage-1
   (score + per-segment top-k in one program), the fixed small ops of the step.
2. The op count: 4 ops per user (slice q, slice w, score, row slice) + ~45 fixed; the batched scorer removes
   the per-user part; the metadata blend (5 ops) can fold into the gather kernel (dense users = pools in order,
   tail slot) once it takes the positions tensor. A multi-core top-k row (the 512-window chain is serial on one
   core: 1.14 ms per 262 K row) is the next per-step term at 1 M.
3. Deliverable 3 on the (2,4) block (`test_mesh_sparse_partitioned.py`, host-side code in place) and the 1 M
   mesh step; then the 8x4 run.
4. Prefill at long ISL still reads the whole prefix for `sparse_sdpa` (the untilize); the same compact-gather
   design applies to the prefill sparse rows (gather per 32-row query block).

## 7. Memory of the user-partitioned cache at 1 M and the prefill scratch slot (2026-10-21)

`TtLatentCache(partition_users=True)` allocates `users_per_chip + 1` full-length slots per chip: the owner's
slot(s) plus a prefill scratch slot, because the sequence-parallel prefill (v1) runs every column's 16 heads
over the whole prefix and therefore needs the full prompt cache on every column while prefilling. At 1 M
positions that is 2 x 11 layers x 1.07 GB of latent (+ 1.5 GB of pooled keys) = ~25 GB per chip, which with
12.4 GB of weights does not fit the 32 GB chip. The 1 M single-user run therefore runs without partition (one
slot per chip, replicated over the row's columns; each column attends its own 16 heads).

Fix for multi-user 1 M (not implemented): do the prefill like the decode, with the heads -> users exchange
(`GLMMesh.all_to_all_cols`) before the attention of a chunk: column c receives all 64 heads of its own users'
rows and attends over its own cache only; the inverse exchange returns the attention output head-split for
`o_proj`. The scratch slot disappears (one slot per user per chip, 11.4 GB at 1 M), and the per-chunk
whole-prefix copy and the scorer run once per owner instead of on every column. For one user the attention
of a chunk runs on one column; the MLP / KDA / mHC parts stay sequence-parallel over the rows as today.
