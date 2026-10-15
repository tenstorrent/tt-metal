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
