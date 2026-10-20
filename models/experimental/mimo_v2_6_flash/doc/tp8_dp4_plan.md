# MiMo-V2.6-Flash on Blackhole Galaxy: TP=8 x DP=4

Phase 0 target (tenstorrent/tt-metal#60807): four independent replicas on one 4x8 Galaxy,
each replica on one row of 8 chips with tensor parallelism 8.

## Mesh plan

| Mesh axis | Size | Role | Collectives on it |
|---|---|---|---|
| 0 (rows) | 4 | DP: independent replicas, each with its own weights, paged KV cache and users | none |
| 1 (cols) | 8 | TP inside a replica | `all_reduce` after attention and after the MLP/MoE, `cluster_axis=1`, 1D ring |

The 4x8 mesh is split into four 1x8 submeshes (`create_submeshes`, the split vLLM's
data-parallel path uses). Fabric is `FABRIC_1D_RING`. Each row's ring is used only by that row.
A long prefill on one row does not stall the other rows.

Why not flatten to TP=32 or do DP-attention + EP=32: see the comparison in
tenstorrent/tt-metal#60809. TP=8 x DP=4 keeps replicas independent and matches vLLM's standard
DP path. The cost is that experts must be BFP4 to fit.

## Per-chip tensor layout (TP=8)

| Tensor | Full shape | Per chip | Notes |
|---|---|---|---|
| Embedding | 152576 x 4096 bf16 | replicated | 1.25 GB |
| `qkv_proj` (SWA) | 4096 -> 64x192 + 8x192 + 8x128 | 4096 x (8x192 + 192 + 192) | 8 Q heads + 1 KV head; V padded 128 -> 192 |
| `qkv_proj` (global), first prefill chunk | 4096 -> 64x192 + 4x192 + 4x128 | same as SWA | 8 Q heads + KV head `c // 2` on chip `c` |
| `qkv_proj` (global), decode and later prefill chunks | | 4096 x (16x192 + 192 + 192) | all 16 Q heads of KV group `c // 2`; the chip pair splits the users (`tt/kv.py`) |
| `o_proj` | 8192 -> 4096 | (8x192) x 4096; global decode (16x128) x 4096 | padded rows are zero; row-parallel + all-reduce |
| Attention sink (SWA) | 64 | 8 | pre-divided by the softmax scale |
| Dense MLP (layer 0) | 4096 -> 16384 -> 4096 | 4096 x 2048, 2048 x 4096 | column / row parallel + all-reduce |
| Router | 4096 x 256 bf16, bias fp32 | replicated | fp32 sigmoid / topk / gather |
| Routed experts | 256 x (3 x 4096 x 2048) | 256 x (3 x 4096 x 256) | intermediate dim TP-sharded, BFP4; decode streams the selected experts bank-locally (`tt/moe_stream.py`), prefill runs them expert-major (`tt/moe_prefill.py`); all-reduce |
| LM head | 4096 x 152576 | 4096 x 19072 | vocab-parallel, host concatenates |
| KV cache, SWA layer | | K, V `[blocks, 1, 64, 192]` | paged by absolute position, only the blocks a window can still read backed: `kv.swa_blocks_per_user` (9) per user for any chunk size (`kv.SwaBlockTable`), BFP8 |
| KV cache, global layer | | K `[blocks, 1, 64, 192]`, V `[blocks, 1, 64, 128]` | full context, only the users this chip owns, BFP8; decode via `paged_flash_multi_latent_attention_decode`, later prefill chunks via `chunked_flash_mla_prefill` with a separate V cache |

RoPE: HF rotates the first 64 of 192 dims with `rotate_half`. The Q/K rows of each head are
permuted to interleaved order so `rotary_embedding_llama` applies; cos/sin are 192 wide with
cos = 1, sin = 0 on the 128 no-RoPE dims, computed on the host for the positions in use (no
max_seq_len tables on device).

## Memory per chip (48 layers)

| Item | Per chip |
|---|---|
| Routed experts, BFP4 (37.8B params) | 21.3 GB |
| Routed experts, BFP8 (for comparison) | 40.2 GB: does not fit |
| Attention weights, BFP8 (0.68B params) | 0.72 GB |
| Embedding (bf16, replicated) | 1.25 GB |
| Router, dense MLP, LM head, norms | ~0.2 GB |
| KV cache, 4 users x 1M, BFP8: global layers (2 owned users x 1M x 9 layers x 320) | 6.4 GB |
| KV cache, SWA blocks (39 layers x 4 users x 9 blocks x 64 x 384) | 0.03 GB |
| **Measured** (`test_capacity_1m_tp8_dp4.py` on `mimo-v2.6/tp8-dp4`, every replica) | weights 23.69 GB; with 4 x 1M KV 30.14 GB of 34.03 GB (3.89 GB free) |

How B=4 @ 1M fits, with existing ops only (no new kernel):

1. SWA layers only attend to 128 tokens. Their page tables are indexed by absolute position (the vLLM
   SlidingWindowSpec layout), but only the blocks a window can still read are backed: 9 per user, the
   window before a prefill chunk, the window and padding tail after it (`kv.SwaBlockTable`).
   A full-length SWA cache would be 66.8 GB per chip.
2. Global layers have 4 KV heads on 8 chips. Rather than store each head on both chips of a pair,
   the pair splits the users: chip `c` keeps the KV of slots with `slot % 2 == c % 2` and decodes all
   16 Q heads of its group for them (decode reads the minimum bytes; no cross-chip softmax merge).
   A first prefill chunk uses 8 heads per chip on fresh K/V, so a prompt that fits one chunk is balanced
   over the pair; a later chunk needs the cached KV, which only the owner holds, so both chips run the
   group's 16 heads through `chunked_flash_mla_prefill` and the non-owner's result is zeroed before the
   all-reduce.
3. The global V cache is 128 wide, read with `paged_flash_multi_latent_attention_decode`
   (`head_dim_v`) in decode and `chunked_flash_mla_prefill(input_tensor_v=...)` in later prefill chunks,
   so V is not padded to 192 in the cache.

Splitting the sequence across the pair instead would need decode SDPA to return the softmax
statistics (LSE) for a cross-chip merge, which it does not today.

## Validation (tests/, measured 2026-10-15 on one BH Galaxy, branch `mimo-v2.6/tp8-dp4`)

All tests open the 4x8 mesh (`FABRIC_1D_RING`), split it into four 1x8 rows and give every row its
own users. References are fp32 torch (`reference/model.py`, bit-identical to HF on real weights).
The first table is for the MiMo-V2.6-Flash-RL checkpoint. MiMo-V2.6-Flash-MOPD, the target, has the same
`config.json` and tensor layout; its baseline and chunked-prefill numbers follow.
`mimo-v2.6/dev` carries `test_decoder_tp8_dp4.py` and `test_model_tp8_dp4.py`; the MoE, traced-decode
and 1M-capacity tests stay on `mimo-v2.6/tp8-dp4`.

| Test | What | Result |
|---|---|---|
| `test_decoder_tp8_dp4.py` layer 0 (global, dense) | 4 rows x 4 users, prefill 128-384 tokens + 3 decode steps, BFP8 weights and KV | output PCC 0.99999; block-delta PCC prefill 0.9977, decode 0.9963 |
| layer 1 (SWA + sink, MoE) | same | output 0.9995 / 0.9979; delta 0.9916 / 0.9946 |
| layer 5 (global, MoE) | same | output 0.9966 / 0.9992; delta 0.9929 / 0.9998 |
| `test_moe_tp8_dp4.py` layers 1, 5 | real golden MoE inputs over 4 rows | top-8 set match 98%; vs reference with device routing PCC 0.9999+; end to end 0.9997-0.9999 |
| `test_model_tp8_dp4.py` | 48 layers, BFP4 experts, 4 rows x 3 users | 40 logits rows vs fp32 reference: top-1 97.5%, PCC mean 0.9924 (min 0.9686). Shared prompt identical on all 4 rows (PCC 1.0, same greedy text) |
| `test_model_traced_tp8_dp4.py` | traced decode, 4 rows x 2 users | traced == eager tokens (logits PCC 1.000000). Step: 1 row 164 ms, 4 rows 228 ms -> 35.1 tok/s aggregate vs 12.2 tok/s on one row (eager: 690 ms for 4 rows) |
| `test_capacity_1m_tp8_dp4.py` | 4 rows x 4 users, max_seq_len 1M, traced decode | fits (above). 16 users: 261 ms/step at 4K, 278 ms at 128K, 449 ms at ~1M |

MOPD baseline, measured 2026-10-16 on one BH Galaxy with `mimo-v2.6/dev` at 650497c08d3 (before chunked
prefill), the MOPD fp32 golden and a weight cache converted fresh from the MOPD checkpoint (first row
1373 s, the other rows 13-17 s from the cache):

| Test | What | Result |
|---|---|---|
| `test_decoder_tp8_dp4.py` layer 0 (global, dense) | same inputs as the RL rows | output PCC 0.99999 / 0.99999; block-delta PCC prefill 0.9982, decode 0.9977 |
| layer 1 (SWA + sink, MoE) | same | output 0.9996 / 0.9990; delta 0.9925 / 0.9933 |
| layer 5 (global, MoE) | same | output 0.9987 / 0.9965; delta 0.9979 / 0.9999 |
| `test_model_tp8_dp4.py` | 48 layers, BFP4 experts, 4 rows x 3 users | 40 logits rows vs the MOPD fp32 reference: top-1 95.0%, PCC mean 0.9928 (min 0.9615). Shared prompt identical on all 4 rows (PCC 1.0, same greedy text) |

Against this baseline, chunked prefill leaves the single-layer worst cases unchanged and the 48-layer
top-1 at 95.0% (PCC mean 0.9925 vs 0.9928, min 0.9604 vs 0.9615).

Chunked prefill (#60808) as first merged, measured 2026-10-16 on one BH Galaxy with the MOPD checkpoint, branch
`mimo-v2.6/chunked-prefill` (padded V, `SlidingWindowSpec` SWA layout with reused blocks, owner-chip global
prefill). The decode-performance change replaced it with chunked prefill on the decode cache layout (128-wide
global V, SWA blocks only around the window); its numbers are under "Validation" at the end:

| Test | What | Result |
|---|---|---|
| `test_chunked_prefill_tp8_dp4.py` layers 0, 1, 5 | 4 prompts of 200-1333 tokens in chunks of 64 / 128 / 256 / 512 (one size per row), caches pre-filled with random values, SWA tables with vLLM null blocks; then 3 decode steps | chunked vs reference: output PCC >= 0.9990, delta >= 0.9937; chunked vs one shot: >= 0.9993, delta >= 0.9984; written K/V vs reference >= 0.99994, V padding exactly 0; decode delta (pooled) >= 0.9963 |
| `test_decoder_tp8_dp4.py` layers 0, 1, 5 | as above (one-shot prefill + decode) | output >= 0.9987 / 0.9965; delta >= 0.9925 / 0.9933 |
| `test_model_tp8_dp4.py` | 48 layers, MOPD golden | top-1 95.0%, PCC mean 0.9925 (min 0.9604); shared prompt identical on all 4 rows |
| `test_model_chunked_tp8_dp4.py` | 48 layers, 1500-token prompt: one shot vs chunks of 128-1024 vs prefix hit at 512 | same next token everywhere; logits PCC vs one shot >= 0.9948 (chunks), 0.9982 (prefix hit); greedy text identical across the three |

Notes:

- Routing is fp32 end to end. `moe_grouped_topk` resolves MiMo's near-ties at bf16 precision
  (80% top-8 match on exact fp32 logits); the fp32 sigmoid/topk/gather path matches 100% on exact
  logits and 98% on device logits, the rest coming from the router matmul (~2^-12 relative).
- 4 rows cost 1.4x one row per traced step, not 1x: the host part of each step (input copies and
  reading 32 x 152576 logits from 8 shards) is serial over the rows. Device sampling removes most of it.
- Performance is untuned: every op is DRAM-interleaved with default program configs.
- Weight cache: the first replica converts the checkpoint in ~24 min (half host MXFP4 dequant,
  half TTNN BFP4 encoding); every later replica load is ~14 s from the cache.

## Scope

This document covers the TP=8 x DP=4 mesh plan and its validation. The remaining Phase 0 items are
tracked separately and do not change the plan above:

- Chunked prefill (`start_pos > 0`) and prefix caching: #60808.
- Device sampling, speculative decoding, performance tuning: #60809.
- vLLM data-parallel serving over the four rows: vllm-tt-plugin#216.

## Decode performance (traced, one TP=8 row unless noted)

Measured with `test_model_traced_tp8_dp4.py` (2 users per row) and `test_decode_perf_tp8.py` (device
profile of 6 layers). Every step kept the accuracy checks above passing.

| Change | Step (1 row) | 4 rows, aggregate |
|---|---|---|
| First traced decode | 163 ms | 233 ms, 34 tok/s |
| MoE decode computes only the selected (user, top-k) pairs: indexed `sparse_matmul`, fused gate/up weight, `in0_block_w` 64; RMSNorm over 32 cores instead of 1 | 64.5 ms | 131 ms, 61 tok/s |
| Per-chip top-k of the LM head on device; the host reads 8 x 32 candidates instead of full logits | 52.5 ms | 62.5 ms, 128 tok/s |
| DRAM-sharded qkv and router matmuls; `sparse_matmul` indexed mode skips its output zero-fill | 46.5 ms | 57.6 ms, 139 tok/s |
| tt-kernel-agent (3 rounds on the MoE decode layer, 651 -> 479 us): per-pair intermediates in L1; the indexed `sparse_matmul` reuses the broadcast activation across experts and prefetches the next expert's first weight block | 38.4 ms | 47.6 ms, 168 tok/s |

### DRAM bandwidth utilization (`tests/perf/decode_bandwidth.py`)

Useful bytes per chip per step (the selected experts' weight slices, attention, router, dense MLP, LM head and
the KV attended, without padding: 3.58 GB at short context, 4 users per replica) / step time / 512 GB/s. The
device loop (`MiMoGalaxyGenerator.decode_steps`) keeps greedy decoding on device: argmax across the row,
token feedback and position update inside the trace; its tokens match the host loop.

| Change | 1 row, device loop | 4 rows, device loop | 1 row, host loop |
|---|---|---|---|
| Start (MoE layer 479 us) | | | 38.6 ms, 18.0 % |
| Device-resident greedy loop | 33.3 ms, 20.9 % | 35.4 ms, 19.7 % | 38.8 ms |
| Routing after the sigmoid in one program (exact fp32 top-8, normalize, mask; replaces top-k, gather and ~20 glue ops; later folded into the streaming gate\|up); CCL on 2 links (all-reduce 37.7 -> 29.7 us) | 26.8 ms, 26.0 % | 28.9 ms, 24.1 % | 31.8 ms |
| tt-kernel-agent `mimo_moe_stream` round 2: bank-local streaming gate|up with the routing selection inside the program (MoE layer 283 us) | 23.7 ms, 29.5 % | 25.8 ms, 27.1 % | 28.7 ms |
| tt-kernel-agent final (4 rounds): bank-local streaming down with the pair sum in dest (MoE layer 204 us) | 20.1 ms, 34.7 % | 24.1 ms, 29.0 % | 24.6 ms |
| Bank-reader cores from the 4x8 parent: row 2's own answer (x = 6 for banks 4-7) slowed six of its chips | 20.1 ms, 34.7 % | 21.95 ms, 31.8 % | 24.8 ms |
| tt-kernel-agent `mimo_attn_decode` round 2: L1 hand-offs around the projections and the all-reduce; SWA o projection without the V padding rows (attention block 174.7 -> 156.9 us) | 18.5 ms, 37.7 % | 21.1 ms, 33.1 % | 22.3 ms |
| tt-kernel-agent `mimo_moe_stream2`: compact 4-row all-gather + fixed-order fp32 reduce, earlier streaming start (MoE layer 204 -> 181 us) | 18.07 ms, 38.6 % | 20.15 ms, 34.6 % | 22.4 ms |
| tt-kernel-agent `mimo_attn_decode` final: wide-K-block 1D matmuls for qkv / o on unpadded weights (attention block -> 147.4 us) | 17.68 ms, 39.5 % | 19.75 ms, 35.3 % | 22.0 ms |
| tt-kernel-agent `mimo_layer_decode`: compact attention all-reduce through the MoE's fp32 reduce, residual adds folded into it, one-launch head concat (layer 340.5 -> 321.7 us) | 16.70 ms, 41.8 % | 18.64 ms, 37.5 % | 21.0 ms |

Accuracy after the last row: teacher-forced top-1 vs the fp32 reference 0.975, logits PCC min 0.9689 / mean
0.9930 (`test_model_tp8_dp4.py`; before this table: 0.950, 0.9693 / 0.9926). The GDDR6 of this Galaxy runs at 14 Gbps on every chip (tt-smi), so the
attainable peak is ~448 GB/s; the table keeps the 512 GB/s spec peak as the denominator.

**MiMo-V2.6-Flash-MOPD** (2026-10-16, branch `mimo-v2.6/decode-perf`, all of the above): device loop 16.50 ms/step on
one row (42.3 %), 18.47 ms on all four rows (37.8 %); tokens equal to the host loop. `test_model_tp8_dp4.py` vs the
MOPD fp32 golden: top-1 0.925, logits PCC min 0.9608 / mean 0.9926. With the original HiFi2 `sparse_matmul` experts
on the same cache (A/B): top-1 0.950, PCC min 0.9608 / mean 0.9925, so the LoFi streaming experts leave the logits
unchanged and the top-1 difference is one near-tie row of 40.

Weight-cache generation: the full MOPD cache (48 layers, ~186 GB) took 2681 s with this branch's build. With
tt-metal#59764 (host BFP preparation) merged on top, layers 0-2 into a fresh cache took 55.4 s instead of 110.1 s
(`ttnn.as_tensor` 16.2 s instead of 69.7 s; checkpoint reading and dequantization unchanged at ~40 s), and every cache
file was byte-identical.

With B=4 x 1M allocated (`test_capacity_1m_tp8_dp4.py`, 16 users, 4 per row), after the first table's rows:
49.1 ms/step at 4K positions (326 tok/s), 74.5 ms at 128K (215 tok/s), 236.7 ms at ~1M (68 tok/s).

Where one MoE decode layer's ~650 us go (device, per chip): expert `sparse_matmul` gate|up 282 us
(16 cores) and down 118 us, fp32 top-k 44 us and gather 33 us, row all-reduce 34 us, the rest in
small ops. The expert matmuls read ~57 MB of BFP4 weights per call, so they run at about half of the
bandwidth a DRAM-sharded matmul reaches. `tests/perf/kagent_moe_decode_{bench,check}.py` are the
tt-kernel-agent perf and correctness commands for this layer (baseline 653 us, spread 0.04 %).

## Decode DRAM utilization, 2026-10-16 (branch `mimo-v2.6/decode-perf`, MOPD)

Traced device-resident greedy loop, 4 users per replica, median of 5 runs (`tests/perf/decode_bandwidth.py
--steps 32 --repeats 5`). Utilization = useful bytes per chip per step / step time / 512 GB/s.

| Change | 1 row ms/step | 1 row % | 4 rows % |
|---|---|---|---|
| Start of this round (MOPD port) | 16.50 | 42.3 | 37.8 |
| Bank-streaming SWA qkv / global o | 16.28 | 42.9 | 37.9 |
| Both RMSNorms fused into the block reductions | 15.85 | 44.0 | 39.5 |
| RouterScoreStream (router + sigmoid + bias in one launch) | 15.33 | 45.5 | 40.0 |
| Row all-gather over the fabric inside the reduce | 14.93 | 46.8 | 41.4 |
| FPU chip sum in the reduce (median of 5 from here) | 14.55 | 48.0 | 43.3 |
| SwaDecodeQkv + SWA paging by absolute position | 14.21 | 49.1 | 43.4 |
| SWA o streamed into compact rows | 14.15 | 49.3 | 44.5 |
| Down inside the fabric reduce launch | 14.01 | 49.8 | 44.1 |

Accuracy at the end: top-1 0.975, logits PCC min 0.9608 / mean 0.9925 vs the fp32 reference; device loop tokens
match the host loop.

SWA KV is now paged by absolute position (`kv.SwaBlockTable`, the vLLM SlidingWindowSpec layout) instead of a
512-token ring; 300 greedy tokens are identical with the sparse table and with every block backed.

Measured but not kept (the code is removed): qkv prefetch into L1 during the reductions (qkv 23 -> 14.5 us, but
the reductions stretch, net flat); the o projection inside the attention-side reduce launch (its ~10 us dispatch is
not hidden); `all_gather_async` (slower than `ttnn.all_gather` for 32 KB).

Next steps toward 55% (12.7 ms/step), largest first:
- GateUp waits ~9 us per layer for user 0's routing before it streams. The routing normalization runs in soft float
  on the rv32im data-movement cores; integer fixed-point normalization (or routing in the router launch) should
  recover most of it (~0.4 ms/step).
- One attention-core launch per SWA layer: SwaDecodeQkv + windowed SDPA with sinks + head concat (~26 -> ~10 us
  per layer, ~0.6 ms/step).
- Router inside the post-attention reduce launch; GreedyHead scan / merge (~50 us/step); the 20 us single-core
  embedding; layer 0's dense MLP.

## Validation on `mimo-v2.6/decode-perf` (MOPD, 2026-10-19, on `mimo-v2.6/dev` with #60957)

Decode on this cache layout, and chunked prefill on it (it replaces #60957's padded-V / reused-block version).

| Test | Result |
|---|---|
| `test_mla_prefill.py -k separate_v` (ttnn) | 4 passed: chunked MLA prefill with a separate paged V cache, bf16 and bfp8, vs torch |
| `test_decoder_tp8_dp4.py` (layers 0, 1, 5) | 3 passed |
| `test_moe_decode_tp8.py` | passed: streaming decode vs prefill, worst user PCC 0.999984; every decode output bit-identical on the 8 chips; 4 fabric decodes back to back bit-identical to the per-step runs |
| `test_chunked_prefill_tp8_dp4.py` (layers 0, 1, 5; chunks of 64-512 on garbage-filled caches) | chunked vs reference >= 0.99892 (delta >= 0.9937), vs one shot >= 0.99932, cached K / V >= 0.99993, decode after it >= 0.99748 |
| `test_model_tp8_dp4.py` (48 layers) | top-1 0.975, logits PCC min 0.9608 / mean 0.9925 vs the fp32 golden |
| `test_model_chunked_tp8_dp4.py` (48 layers, ~1.5K-token prompt) | chunks of 128-1024 and a prefix hit at 512: next-token logits PCC vs one shot >= 0.9963, same top-1 up to a bf16 tie |
| `test_model_traced_tp8_dp4.py` | passed: traced == eager (logits PCC 1.000000); device greedy loop == host loop for 16 tokens per user |
| `tests/perf/decode_bandwidth.py --steps 32 --repeats 5` | device loop 1 row 13.48 ms/step (51.8%), 4 rows 15.06 ms (46.4%); 128/128 tokens match the host loop (every run starts from a fresh prefill) |

## MoE prefill, 2026-10-19 (branch `mimo-v2.6/moe-prefill`, MOPD)

MoE prefill (`tt/moe_prefill.py`) replaces the per-32-token `sparse_matmul`, which re-read ~40% of the expert weights
for every 32 tokens (~37 GB per 4K-token layer per chip). It was built on `mimo-v2.6/decode-perf-chunked` (#61388,
closed) and ported onto `mimo-v2.6/dev` after #61389 without changing it. `TtMiMoMoE.forward` routes as before (fp32
top-8, the token mask applied to the weights) and hands the indices and weights to `PrefillExperts`:

- `DevicePlan` (16 cores, split by expert) sorts the T x 8 (token, k) pairs by expert and pads each expert to
  32-row groups, entirely on device; it is bitwise identical to the host version of the plan.
- `GroupedExperts` runs every group in one program on 30 teams of 4 cores. A team core keeps its slice of the
  job's expert (442 KB of BFP4, from two DRAM banks) in L1 for all of the job's groups, reads its quarter of the
  group's token rows straight from x and shares the tilized quarter with the team, computes two intermediate
  columns (gate|up, SwiGLU x routing weight), swaps h tiles with the team, and writes its 32 down columns of every
  row straight to row k * T + t of a k-major output. `CombineK` sums the 8 rows of every token and tilizes.

One MoE layer, 4096-token chunk, one row (`tests/perf/kagent_moe_prefill_bench.py`). The steps were measured on
`mimo-v2.6/decode-perf-chunked`; the last row is this branch:

| Change | ms |
|---|---|
| Per-32-token `sparse_matmul` | 337.9 |
| Expert-major with ttnn ops (host plan, embedding gathers, indexed `sparse_matmul` per group) | 34.7 |
| Grouped expert kernel (each expert slice read once per job) | 20.3 |
| Plan on device (no routing read back) | 17.6 |
| Team cores read their token rows directly (no gathered copy of x) | 13.2 |
| Rows written straight to the k-major output + `CombineK` | 11.8 -> 10.8 (1 KB reads) |
| Plan on 16 cores | 9.7 |
| This branch (median of 5) | 9.65 |

`tests/perf/kagent_moe_prefill_check.py` on this branch: output vs the saved `sparse_matmul` outputs PCC
0.999990-0.999993 (worst token 0.999980) on the golden layer-1 inputs and on 32, 1056 and 4096 random tokens; every
chip identical, all finite; a second 4096-token call bitwise identical.

Full model on one row (`tests/perf/prefill_time.py`, random tokens): 4096 tokens in one chunk 13.9 s -> 0.731 s (5601
tokens/s); in chunks of 2048 0.842 s, of 1024 1.062 s; 32K tokens in 4K chunks 6.55 s (5004 tokens/s), from 720 ms
for the first chunk to 902 ms for the last. All four rows at once (`MiMoGalaxyGenerator.prefill`, 4K prompts): 4 users
in 2.13 s (7.7k tokens/s per Galaxy), 16 users in 7.91 s (8.3k tokens/s). The host dispatches every op of all four rows
from one Python thread, so four rows give 1.5x one row, not 4x.

### Validation on `mimo-v2.6/moe-prefill` (MOPD, 2026-10-19)

| Test | Result |
|---|---|
| `test_kv_tables.py`, `test_prefill_helpers.py` | 14 passed |
| `test_decoder_tp8_dp4.py` (layers 0, 1, 5) | 3 passed; worst PCC prefill 0.9987, decode 0.9965 (layer 1: 0.99965 / 0.99898) |
| `test_chunked_prefill_tp8_dp4.py` (layers 0, 1, 5) | 3 passed; chunked vs reference >= 0.99892, vs one shot >= 0.99932, cached K / V >= 0.99993, decode after it >= 0.99748 |
| `test_moe_decode_tp8.py` | passed: streaming decode vs prefill (now expert-major), worst user PCC 0.999986, within 2 ulp |
| `test_model_tp8_dp4.py` (48 layers) | top-1 0.950, logits PCC min 0.9581 / mean 0.9921 vs the fp32 golden (`sparse_matmul`: 0.975, 0.9608 / 0.9925; one of the 40 rows flips) |
| `test_model_chunked_tp8_dp4.py` | chunks of 128-1024 and a prefix hit at 512: next-token logits PCC vs one shot >= 0.9953, same top-1 up to a bf16 tie, same greedy text |
| `test_model_traced_tp8_dp4.py` | passed: traced == eager (logits PCC 1.000000); device greedy loop == host loop for 16 tokens per user |
| `tests/perf/decode_bandwidth.py --steps 32 --repeats 5` | device loop 1 row 13.50 ms/step (51.7%), 4 rows 15.07 ms (46.3%); 128/128 tokens match the host loop |

`test_model_traced_tp8_dp4` first failed here on a tie, not on a wrong value: golden prompt 0 reaches an exact bf16 tie at
its second decode step (" that" and " of" both 17.625), and the host loop's `MiMoGalaxyGenerator.greedy_token` took the
first of the per-chip top-k candidates while the device loop's GreedyHead takes the lowest id. `greedy_token` now takes
the lowest id too, like `torch.argmax` over the full logits.
