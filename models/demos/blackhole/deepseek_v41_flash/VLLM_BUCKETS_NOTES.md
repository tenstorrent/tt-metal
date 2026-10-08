# vLLM adapter: right-sized decode / prefill steps (bucketed batch) - DESIGN NOTE (pre-implementation)

Status: design from code reading; NO device measurement yet (hosts .45/.34 not yet used; the shell host .44 is not ours). Items marked [M] are numbers from existing notes, [C] from code, [?] to be confirmed on device.

## 1. Problem
The adapter always steps all `max_num_seqs` users (`tt/generator_vllm.py` `decode_forward`: `build_decode_inputs(self.B, ...)`, `admit_idle_users`, `m.decode_forward(tok_B,...)`). Step cost follows the BUILD batch: [M] demo B=4 43 ms, 16 43-44, 32 49, 64 65-68, 128 88-95 (BATCH_SCALING_NOTES: growth is mHC ~+390 us/layer from U=4 to 32, moe_compute +~100 us/layer, attention +44 us/layer). B=4/8/16 are within ~1-2 ms: the buckets that matter for decode speed are 16 (floor), 32, 64, 128. Decode runs at 4/8 would only pay off for spec decode rows (U(1+k) <= 32).

## 2. Facts: what is shaped by U (users per mesh row) [C]
| state | shape / where | bucket story |
|---|---|---|
| KV pool, ring regions | one tensor `[1,1,NP*320 + n_ring*U*ring_rows,512]`; ring of (layer slot s, user u) at `ring_origin + s*U*ring_rows + u*ring_rows` (`PagedKVPool.ring_base`) | kernel gets `ring_base` as a common runtime arg and `rows = pos.shape`; a U' < U decode only needs `ring_base` with the MAX stride -> pure prefix, no copy |
| page table | persistent int32 `[U, maxp]` per mesh row | `paged_kv_step` reads `maxp` from the last dim and indexes by user: pass the full table, rows = U' -> prefix, no copy |
| compressor `prev_cs`, `cs_state[i]` | `[1,1,U,1024]` fp32 TILE, per layer | tile padded to 32 rows for any U <= 32: logical-shape view (`ttnn.reshape(t, new, padded)`) [?] confirm zero-copy + in-place writes visible |
| decode index-key slab `k_cache` | `[U,1,n_alloc,128]` bfp8 TILE per key-owner layer (2,8,14,20; others alias) | user is the OUTER dim: no reshape view. fused backend already addresses users with `cache_batch_idx=u` (fine); matmul backend (`default_backend`: <= 65536 entries, i.e. always here) multiplies `q[U,..] @ k_cache[U,..]` batch-matched -> needs a prefix slice (in-trace `ttnn.slice`, a device copy of U' x n_alloc x 128 per layer per step) or the fused backend; `paged_update_cache(k_cache, k, update_idxs)` batch match [?] |
| step tensors (rope rows, pos, tok_dev, rows_cat) | `[U]` / `[U,1,1,k]`, persistent, refreshed per step | per-bucket buffers, tiny: allocate for every bucket up front |
| attention / mHC / shared expert / moe gate / head | weights batch independent; program configs, sharded memory configs, core grids, `batch_per_device`, mHC plans (`mhc_mixes2._PLANS` keyed by T) are derived from U in the constructors | a U' graph needs U'-specific objects/configs; mHC fast kernels exist for T=4/16/32 rows [M] |
| MoE decode buffers (L1 indices/scores, dispatch/combine, 2 global semaphores), CCL semaphores | L1, sized by `batch_per_device` | L1 first-fit fragmentation risk (RECONFIGURE_NOTES 2.): per-bucket sets must be allocated in the same pass as the max set, before any trace, or be shared at max size [?] |
| Engram host rows / hash cache | host `[B, max_ctx+1024]` | per-slot, prefix for free |
| decode traces | one per shape | one trace per bucket, all captured at build after all compiles |
| prefill (interleaved, `DSV41_PREFILL_UP=1`) | trace is U-independent (Up=1: one prompt per mesh row per replay, 512-token chunk) | ALREADY right-sized: 4 prompts per replay, filler cost gone; "pad to nearest bucket" is Up=1 for <= 4 new prompts, more waves otherwise. Baseline numbers 19 s / 81 s in the task are the old whole-batch path; the branch default (tip a1eb1dc6446) has never run on device: validate first |

## 3. Reference pattern (qwen36_vllm.py / generator_interface.py), what we reuse
1. Adapter slices host inputs to the prefix `[:bucket]` of the plugin batch (active requests sit in low slots; slot_remap step keeps the full width), no output re-pad.
2. Per-bucket trace store (own trace id, own persistent I/O).
3. Warm-up order: compile EVERY width un-traced, synchronize + gc, then capture ALL bucket traces (compile-before-capture). 4. Prefill warm-up at init so requests only replay.

## 4. DSV4 slot layout (the key difference)
Model user b = `r*U + u` (mesh row r, user-in-row u). A plugin prefix of length W is meaningless physically; what matters is the SET of physical users: bucket B' = 4*U' needs every live user to have `u < U'`. Plugin slot -> physical user is the adapter's table (`VS.SlotTable.phys`, free to choose at admission since a free slot holds no state). Therefore:
* admission policy `claim_lowest(row-balanced)`: a new request takes the free physical user with the smallest `u`, ties -> the mesh row with the fewest live users (replaces `claim_balanced`'s only-row criterion; prefill at Up=1 spreads over rows already);
* step bucket U' = 1 + max(u of live and parked users), B' = 4*U' rounded up to a supported size; B' in {4,8,16,32,64,128} <= build. Hole-fragmentation (a long-lived user at u=20 pins U'=21 -> 32): mitigated by the lowest-u policy; optional later: migrate a user to a lower u when idle holes exist (ring rows + `prev_cs` + key slab rows + page ids re-keyed: small copy), bounded and hysteretic.
* the plugin's own condensing (`slot_remap`) only permutes logical slots; physical placement stays ours.
* filler rows for non-live users of a bucket are the deterministic diverse tokens (parked users keep their parked position).

## 5. Options with numbers
A. Prefix views / in-trace prefix slices of the max-build persistent state + one decode graph per bucket (chosen as the end state). Cost: build time (one extra compile + capture per bucket, ~1 min each [M reconfigure rebuild 65-95 s is the upper bound]), DRAM: only the per-bucket step buffers and (matmul indexer backend) slice scratch inside the trace; weights and pool shared. No stalls at run time.
B. `Model.reconfigure` between buckets: 65-98 s per switch at 40 layers [M RECONFIGURE_NOTES], loses all live KV state: only as an idle-time fallback. Rejected as the main path.
C. A full second `Model` per bucket with its own state + migration at switch: duplicates non-expert weights (~125 MiB/bank per set [?]) unless uploads are memoised, free DRAM is ~730 MiB/bank at B=32, ~150-690 at B=128: rejected unless A fails on a specific op.
Chosen: A with, per tensor, the cheapest of (zero-copy pass of the max tensor with a smaller row count) / (logical-shape view) / (in-trace prefix slice + in-place write-back). The U'-specific objects (attention/mHC/MoE configs) are built per bucket by the SAME constructors with `keep` (experts, embedding, head, Engram weights shared) and a weight-upload memo (content hash >= 1 MB, non-constant) so non-expert weights are not duplicated [?].

## 6. Allocation / trace safety rules (from the 40-layer hang)
* Allocation order at load: every bucket's persistent buffers -> all prefill buffers/shapes (chunk 512, S_pad = max) -> compile every decode bucket and the prefill chunk un-traced -> sync + gc -> capture all decode traces -> capture the prefill trace; never allocate afterwards (the hot paths then only copy into persistent tensors). Today's interleave path captures the prefill trace lazily at the first request while the decode trace is live: this is the hang pattern; the fix is the warm-up at init (`warmup_model_prefill/decode` hooks + `DSV41_VLLM_S_PAD` fixed at load).
* Replay chunk time stays < TT_METAL_OPERATION_TIMEOUT_SECONDS=5 (chunk 512 replay 0.6 s [M]).
* One spec/scenario pass per process; spec: rows U'(1+k) <= 32 per mesh row per bucket (kept conceptual).

## 7. Plan / validation
1. Device smoke (4 layers, then 40) of the interleave base: sequential + overlapping requests, 30+ mixed requests; fix what fails.
2. Adapter: bucket choice, lowest-u admission, step sliced to B' (`build_decode_inputs(B')`), per-bucket stats; hardware-free tests in `tests/test_vllm_interface.py`.
3. Model: per-bucket decode graph + trace store + warm-up order; confirm the [?] items with micro device tests (reshape view of `[1,1,U,1024]` tile, `paged_update_cache` batch, matmul-indexer prefix slice).
4. 40-layer: B=128 and B=32 builds; steady ms/step per bucket vs demo; TPOT conc 1/4/8/16/32; TTFT 128/1024/4096; exactness (lone request vs demo B=bucket greedy; sequential same-prompt repeats); no-hang mix (>= 30 requests).
