# Long ISL on the 4x8 Galaxy: KV sequence-sharding over the 8 columns, and what else limits context (design note, 2026-10-15)

Status: design only (read-only code mapping by three reviews + the measured communication cost in `tests/test_kvshard_comm_microbench.py`). Nothing implemented.
Purpose: support higher ISL (262144 works today at B=4 with spec after the stashed mask-chunking fix; 500k and 1M run out of DRAM). Not a latency feature.

## 1. What actually costs DRAM per chip as context grows (estimates from the code; see the reviews' file:line refs in the commit message)
Per token, per user, per chip (replicated over the 8 columns today):
- paged pool (compressed latents of kv-source layers 2/8/14/20): 2560 B bf16 / 1280 B fp8; decode index-key slabs (bfp8): ~340 B. **Shrink 8x with column sharding.**
- window rings (40 layers x ring rows x 1 KiB: ~11 MiB/user at the spec ring of 288), compressor state, weights, traces: independent of ctx, stay replicated.
- **RoPE tables: ~7 KiB per position per chip, independent of users** (3 layer kinds x [P,512] C/S + 128-wide indexer tables): ~0.9 GiB at 131k, 1.75 GiB at 262k, 3.5 GiB at 524k, **7 GiB at 1M** - the same size as the ~6.9 GiB/chip free after the model is built. Does NOT shrink with KV sharding; needs the compact [P,64] form (or on-device generation) - cheap and first.
- prefill: `PrefillKV` ~2560 B/token/user (+ key FIFO 640 B) = ~3.1 GiB/user at 1M; masks ~1.5 KiB/token (C=512) = ~1.5 GiB at 1M; indexer score block C x ctx x 2 B. These are replicated and column-local too: **KV sharding of decode alone does not remove the prefill peak.**
Decode KV per chip (pool + keys, bf16): U=1/2/4 = 0.35/0.71/1.41 GiB at 131k; 0.71/1.41/2.83 at 262k; 1.41/2.83/5.66 at 524k; 2.83/5.66/11.3 at 1M. (fp8 pool: ~0.56x.)
Consequence: for 500k-1M the order of work is (1) compact RoPE tables, (2) prefill masks/transients chunked + PrefillKV and key FIFO sharded or streamed, (3) decode KV sharding. Doing (3) alone does not reach 1M.

## 2. Decode KV sharding design (paged path only; the non-paged path asserts one shared position and is not a target)
- Split dimension: logical page -> column = page % 8 (balanced for the indexer's top-512); index slab uses the same entry->column map; window ring + `prev_cs` + drafter rings stay replicated.
- Per-column page table / allocator (`kv_paged.py`, `paged_ops.py`: table sharded on `dims=(0,1)`, pool pages /8, `maxp`/8, per-column `pool.ensure`).
- Write path: `paged_kv_step.cpp` latent write guarded by "this column owns the page" (index -1 skip, as the even rows already do); `indexer.append_key` writes only on the owner.
- Read path per layer: all-gather q to all 64 heads (47 us measured); local `sparse_sdpa` over the column's compressed rows, ring partial per column for its own 8 heads (sink applied once, in the ring partial); exchange (o, max, sum) with `all_to_all` and merge locally (the reduce-scatter version measured 40 us; max/sum gather 20 us); indexer: local top-512 + scores -> gather ids/scores over columns -> global top-512 (extra top-k per index layer).
- Measured communication: ~107 us per attention layer = ~8.5 ms/token over 40 layers (T=16 rows incl. padding; reduce by sending only real rows / exchanging partials instead of gathering q). Therefore gate with `DSV41_KV_SEQSHARD=1` for long contexts only; <= 131072 keeps the replicated layout.
- **Blocker: no LSE out of the kernels.** `sparse_sdpa` and SDPA decode return only `o` with the sink fused; a merge needs (m, l). Needs a kernel change that emits max and sum (sink excluded) or a second pass; and 64 heads exceeds the kernel's 32-head padding (two calls or kernel support). Ring and compressed rows currently share one softmax in one index row (`paged_kv_step.cpp:213-236`) and must become two partials.
- Spec: verify rows are virtual users at their own positions -> per-column valid lengths must be position-driven, rejected drafts stay harmless only if never read before overwritten; the (1+k) rows multiply the exchange payload; `SPEC_MAX_CTX` (131584) and the matmul-indexer cap (65536 entries) must scale with the per-column entry count (>524288 ctx needs the indexer backend switch revisited).
- vLLM: block ids are inert (the model owns the allocator) -> no vLLM-facing change; refill/re-prefill must rebuild the sharded pool and index slab (costly). Prefill handoff (`prefill_handoff.py`: `paged_scatter_rows` ids built on host, uploaded `dims=(0,None)`) needs per-column ids with SKIP for non-owned latents.

## 3. Milestones (each measured before the next)
M0 (cheap, no kernels): compact RoPE tables -> frees 0.9-7 GiB; chunked prefill masks (the stashed `prefill_dyn.py` / `prefill_sparse.py` experiment, validated for bit-exactness first); re-run 262144 / 524288 at B=1..4 to see what actually fails next.
M1: plain decode, B=1/2/4, `DSV41_KV_SEQSHARD=1`, one ratio-1 window+compressed layer without the indexer, PCC vs the replicated path at 8k-32k ctx (needs the (m,l) kernel output); then one indexer layer; then the full model at 262144.
M2: prefill-side sharding (PrefillKV, key FIFO, score block) and handoff; 524288 / 1M boots.
M3: spec verify, drafter interaction, vLLM refill, buckets. Deferred.

## 4. Small batches waste the idle mesh rows (B = 1, 2 and 4 all cost the same per chip)  [added 2026-10-15]
`padded_batch` rounds B up to a multiple of 4 (one user per mesh row at minimum, U = 1) and every mesh tensor is allocated with the SAME shape on all 32 chips. So at 1M context B = 1 and B = 2 need exactly the
per-chip memory of B = 4: the idle rows' DRAM is allocated (and wasted), and one user can never use more than ONE row's worth of memory (8 chips) however small the batch is.
Options: (a) per-row tensors via four 1x8 sub-meshes - frees the idle rows' DRAM for nothing, because a user's KV still lives only on its own row's 8 chips; (b) **shard a user's KV over the mesh rows as well as the
columns** (B = 1: 32-way, B = 2: 16-way, B = 4: 8-way): this is the real fix, it is the column sequence-sharding of section 2 extended to a second mesh axis (page -> chip by page % (rows_used * 8); the q all-gather,
the (o, max, sum) exchange and the top-k merge run over rows x columns instead of the 8 columns; the idle rows then hold KV of the active users). Cost: the exchange crosses rows (more CCL latency than the measured
~107 us per layer for 8 columns) and the batch -> row mapping must change (a user is no longer pinned to one row). It builds on milestone M1; do it after the column version works.
Order of work: B = 4 at 512k (one user per row, 524288 token-users per chip = what B = 16 at 128k already runs) -> B = 4 at 1M -> row-sharding for B = 1 / 2.

## 5. Column-sharded PREFILL indexer: microbenchmark (tests/test_indexer_shard_microbench.py, host .40, U = 1, C = 512)  [added 2026-10-15]
`select_dyn` (tt/prefill_sparse.py) scores each chunk against ALL L index keys on every column ([C, L] scores, a [C, L] visibility mask, top-512 per row; the key slab is a sliding FIFO). The sharded variant scores L/8 keys
per column, takes a local top-512, gathers the 8 x 512 candidates (scores + ids) over the columns and takes the final top-512 of the 4096 candidates:
| L | replicated | sharded (local + merge) | per-column buffers (scores+masked / vis / keys) replicated -> sharded |
|---|---|---|---|
| 65536 | 4.10 ms | 8.25 ms | 128 / 64 / 16 MiB -> 16 / 8 / 2 MiB |
| 131072 | 7.99 ms | 8.79 ms | 256 / 128 / 32 -> 32 / 16 / 4 |
| 262144 | 15.71 ms | 12.31 ms | 512 / 256 / 64 -> 64 / 32 / 8 |
Replicated time is linear in L; sharded = ~8 ms of merge (two all-gathers + gathers + a 4096-wide top-k, not yet optimised) + L/8 of the scoring. Memory per column is 8x smaller at every L; speed only wins from L ~ 131k
(extrapolation to L ~ 1M: ~63 ms -> ~16 ms per chunk and index layer; NOT measured). The L = S_pad group (ratio 1) is the one with the big buffers (the 250k OOM was exactly its [2048 x 251904] bf16 score buffer).
Design constraints found while reading the code: the key slab is a sliding FIFO (concat + copy per chunk), so ownership must be strided (slot mod 8, needs C/ratio divisible by 8: true for ratio 1 / 2 / 4 at C = 512, not for
the tiny ratio-128 group which stays replicated) so every column shifts its local FIFO by Cc/8 with no cross-column traffic; the staircase mask of the current chunk and the visibility mask become per-column strided
constants; the final ids are global FIFO slots into the REPLICATED kv table (sparse_sdpa still reads all 8 columns' worth of entries: the kv table itself (2.5 KiB / token / user) cannot be sharded without the (max, sum)
kernel output). Selection ties at the 512th score can differ from the replicated path (bf16 scores tie often), so validation is by quality (needle / GSM8K), not bit-exactness.
