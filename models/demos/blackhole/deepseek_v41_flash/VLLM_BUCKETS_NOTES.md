# vLLM adapter: right-sized decode / prefill steps (bucketed batch) - DESIGN NOTE (pre-implementation)

Status: IMPLEMENTED and device-validated at 40 layers (see section 8, the results). Sections 1-7 are the design as written before the implementation; items marked [M] are numbers from existing notes, [C] from code, [?] were to be confirmed (resolved in section 8).

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


## 8. Implementation and results (40 layers, hosts .34 / .45, logs under /mnt/tt-data/ssinghal/dsv4-logs/vllm_bench/bkt/)

### What was built
* `tt/decode_buckets.py` `DecodeBucket`: a second decoder graph per bucket U' < U (shallow copies of the layers / attention / indexer / step states, MoE block via `DSV41MoEBlock.for_batch`) over the SAME pool, weights,
  experts, router, mHC, head, Engram tables. State by prefix views: `prev_cs` is a zero-copy `ttnn.reshape` view (device-verified: same buffer address, in-place writes eager and in a trace land in the full tensor; the
  padding rows of the tile are rewritten, hence the rule that a bucket covers EVERY live and parked user), the page table is passed whole, the ring stride stays the full U, the index-key slab keeps its [U,..] shape
  (one prefix slice per step shared by all index layers of a slab, padded key append). `Model.warm_serving`: allocate all, compile all (prefill chunk, every bucket), then capture all traces; `Model.bench_buckets`.
* Adapter: smallest bucket >= 1 + highest in-row index of the live set; admission takes the lowest free in-row index (`SlotTable.claim_packed`); optional straggler compaction (`DSV41_VLLM_COMPACT=1`: re-prefill into a low user, 0.42 s measured).
* Idle rows of a decode step feed token 0 (`DSV41_VLLM_FILLER=diverse` restores diverse ids): diverse ids made every idle row activate its own experts: +9..16 ms per step (bucket B'=16: 55.3 -> 44.1 ms).
* Index-key hand-off rewritten with value-independent shapes (gather indices / masks in persistent tensors; `warm_key_export`): the old version sliced at prompt dependent offsets, creating program-cache buffers under live
  traces (found with `TT_METAL_TRACE_ALLOC_TRACKING=1`; the 262 -> 18 -> 0 live-buffer reports are the evidence of the 40-layer hang cause); a device self-test (`DSV41_VLLM_BUCKET_SELFTEST=1`) checks the export (PCC 0.99997),
  run-to-run determinism (PCC 1.0, 64/64 greedy tokens) and bucket vs full-batch logits.
* Spec decode: the verify round is compiled at load (before any trace exists); spec runners per bucket (`SpecRunner(..., Ub=)`, first drafter chunks, prefix views) when the build has U > 4 users per row.

### Steady decode step (host loop incl. host prep; token-0 filler), 40 layers
| build | bucket B'=16 | B'=32 | B'=64 | full |
|---|---|---|---|---|
| B=32 | 42.6-44.1 ms | (full) 52.5-53.2 | | 52.5-53.2 |
| B=128 | 44.3 | 54.1 | 73.2 | 110.9 |
Demo reference: B=16 44 ms, B=32 49-50 ms; padded baselines before: B=128 build 114-145 ms, B=32 build 57-69 ms.

### Stock `run.py --workflow benchmarks`, B=32 build, commit 18ce46ff02c, host .34 (bench_final_b32.log), 0 failures in 19 points
conc1 TPOT 48-50 ms, TTFT 417 ms (ISL 128) .. 3.3 s (ISL 4096) .. 26.7 s (ISL 32768); old padded sweep: conc1 TPOT 62-83 ms, ISL 4096 TTFT 19.0 s.

### Spec decode through vLLM vs the demo (same hosts, GSM8K, effective tok/s/user)
B=4 39.8 vs 40.0, B=8 43.3 vs 46.5, B=16 41.9 vs 43.7 (decode indexer off, as in the demo build; with the indexer on, 36.4), B=32 28.9 vs 29.05. Decode indexer: a build with max_model_len > 512 pays 13-15 ms per spec round (B=16).

## 9. Update: supported batch sizes and the scope of speculative decoding

* Batch 128 is dropped: the largest supported build is B=64 (`max_num_seqs <= 64`, buckets {4, 8, 16, 32, 64}); every B=128 number above is historical.
* Speculative decoding applies up to 32k ISL (plus generation); beyond that serve plain decode (no `speculative_config`). The adapter refuses a spec launch whose `max_model_len` exceeds `tt/spec_policy.SPEC_MAX_CTX` for the batch (B=32: 40000, B<=16: 70000; the DRAM of the spec runners + traces, measured 11.8 MiB per 1k tokens), with a message that says so. 40000 total context covers a 32k ISL (32768) plus ~7k generated tokens at B=32.
* Largest ISL + generation validated for the spec path: demo (tt-metal) B=32 ISL 30059 + 64 tokens at max_seq_len 40000 (SPEC_MATRIX_NOTES.md, log pf_spec_int_specdef_isl32k_b32); B=8/16 ISL 60453 at 70000. Through vLLM: ISL ~4k prompts at B=32 k=3 with max_model_len 33280 (indexer on, TTFT 3.1-3.5 s) and GSM8K (ISL ~100-200, 384 generated) at B=4/8/16/32. A vLLM run at 32k ISL has not been measured.

## 10. Sampled speculative decoding and the environment of a spec server

* Sampling is done by the IN-TRACE sampler (`tt/device_sampler.py`, `DSV41_INTRACE_SAMPLE`, default on; plain and speculative launches): per row temperature / top-k / top-p over the whole vocabulary (a k-ary threshold search for the top-k count and the top-p mass, then the inverse CDF in vocabulary order at the row's uniform `u`), parameters and `u` uploaded with the step (one [rows, 32] fp32 tile), the token ids returned by the trace; greedy rows (temperature 0 / top_k 1) are the exact argmax. `u` comes from the request's own `torch.Generator` (seeded with the request `seed`: reproducible whatever slot / bucket the request runs in; unseeded requests share the server generator). No candidate read, no full-row fallback, no `DSV41_DEV_CAND_K` / `DSV41_SPEC_CAND_K`. The sampler is its OWN trace (not part of the step trace): a decode step with a sampled row replays the step trace and then the sampler trace over its logits (the sampled tokens replace the greedy ones in the token buffer); an all-greedy step replays the step trace only, so greedy serving pays nothing for it. Cost of the sampler trace (device, replay): 3.8 ms at 8 users per mesh row (plain B=32), 9.5 ms at 32 rows per mesh row (spec verify B=32 k=3); the top-k count search resolves one more level than the top-p mass search (`levels_k` 4 vs 3: without it a flat row keeps k + 1 tokens in ~20% of the cases, a 0.044 Kolmogorov deviation at k = 20: tests/test_device_sampler_cpu.py, tests/test_device_sampler.py).
* Sampled speculation is lossless for the drafter's point-mass drafts: the target's own sample (every block row draws with its own uniform) at every block position replaces the argmax, and the device accept rule (draft j+1 == row j) is unchanged. A spec round with a sampled row replays three traces, verify (`SpecDecoder.forward_verify`) | sampler (`forward_sampler`) | accept / commit / draft (`forward_tail`), with one synchronization (`SpecRunner._round` with `sample_cfg`); an all-greedy round keeps the single monolithic trace. The plugin offers drafts to sampled requests only when the model declares `supports_sampled_verify` (vllm-tt-plugin branch `ssinghal/dsv4p1-plugin-sampled-spec`). The first token of a sampled request (the prefill's) is drawn on the host from its logits row.
* The drafter is seeded when the plugin asks for the first drafts of a request (`propose_draft_tokens`), so a request's first decode step is a real verify; a draftless step (requests beyond DSV41_VLLM_SPEC_ISL_MAX, default 32768) runs as the ordinary decode of the same server.
* Environment (defaults are set by the adapter; only the first group is needed in a catalog entry): DSV41_CKPT, DSV41_WEIGHT_CACHE, EXTRA_MODELS_DIR, HF_MODEL, MESH_DEVICE, DSV41_POOL_DTYPE=fp8, DSV41_ENGRAM_RAM=1, DSV41_TRACE_REGION, DSV41_LAYERS (optional). Adapter defaults: DSV41_SPEC (derived from `--speculative-config`; an env value is ignored), DSV41_IDX_SWITCH=131072 (matmul indexer backend for the verify), MOE_COMPUTE_FP32_ACC / MOE_COMPUTE_BFP8_WEIGHTS. Switches: DSV41_VLLM_SPEC_SAMPLED=0 (greedy-only verify), DSV41_VLLM_SPEC_ISL_MAX, DSV41_VLLM_SPEC_GUARD=0 (skip the SPEC_MAX_CTX plan check), DSV41_INTRACE_SAMPLE=0 (no sampler: sampled requests are refused / host-sampled), DSV41_VLLM_STATS_EVERY, DSV41_INDEXER=0 (decode indexer off). A slow cold build needs TT_SERVER_READY_TIMEOUT_SECONDS above the 3600 s default.
