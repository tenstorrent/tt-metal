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

## Accuracy, 2026-10-20 (branch `mimo-v2.6/top1`, MOPD)

The 40-row top-1 of `test_model_tp8_dp4.py` went from 0.975 (`mimo-v2.6/dev`) to 0.925 on the merged vLLM branch. Six of
its 40 rows have a reference top-2 margin under 0.2 logits. Rows with such a margin are near coin flips for every
expert format below, so this metric moves by a row or two between equivalent builds. With the final experts, the
measured hit rates give an expected 40-row top-1 of 0.926, and a 13% chance of reaching 0.975. Accuracy is measured
here on many positions instead.

**Larger golden** (`reference/make_golden_large.py`):
- 32 C4 sequences of 96 tokens, from documents past the ones the GPTQ calibration uses.
- The fp32 reference's last hidden states; the scorer applies the final norm and LM head.

**Two tests on that golden:**
- `test_model_error_tp8_dp4.py`: prefill every sequence whole. The device's last hidden states go through the fp32
  head on the host.
- `test_model_decode_eval_tp8_dp4.py`: teacher-forced decode after an 8-token prefill. These are the 40-row test's
  decode rows, at 70x the count.

**What changed:**
- **fp32 logits** (HiFi2, fp32 accumulation). In bf16 the logits step is 0.125 at the usual top logit of 16-32, so
  near-tied rows often tie exactly and the lowest token id wins. 3 of the 40 rows tied on the merged branch.
  GreedyScan reads fp32. Decode step: 13.44 / 15.14 ms (1 / 4 rows), against 13.48 / 15.06.
- **Prefill SDPA in fp32 dest**, opt-in (`MIMO_SDPA_FP32_ACC=1`, positions >= 2 KL -9% with stock experts). It is
  off by default: its larger circular buffers clash with decode's persistent L1 buffers under the vLLM adapter. Decode
  SDPA stays in bf16 dest: paged decode SDPA gave non-finite values with fp32 dest.
- **Expert values prepared for bfloat4_b.** Same format, memory and speed; only the values change.
  - MSE: each 16-value block takes the best shared exponent of {e, e-1, e+1}. This is tt-quant's `mse` policy
    (`common.bfp_mse_prepare`), and it matches tt-quant's values exactly.
  - GPTQ (`tests/perf/gptq_experts.py`): per expert, on routing-weighted input Hessians from 16K C4 tokens, with the
    exponents {e, e+1}. Experts with fewer than 64 routed tokens use MSE.
  - MXFP4 to bfloat4_b is a second quantization. E2M1 values need 4 magnitude bits on a shared grid, and bfloat4_b
    has 3. The stock packer gives a weight SQNR of about 18.3 dB; MSE gives about 20.9 dB.
  - A host fp32 reference with only the MSE expert values swapped in reproduces the device's error (positions >= 2:
    KL 0.0143 vs 0.0146). So the expert values are what is left, not activations, attention or KV.

`MIMO_EXPERT_QUANT=stock|mse|gptq` picks the expert values; each has its own cache files. `mse` is the default. `gptq`
needs `gptq_experts.py` output in `MIMO_EXPERT_PREP_DIR` the first time its cache is built.

| Experts | Prefill: top-1 / top-1 at margin >= 0.5 / KL (3072 positions) | Decode: top-1 / top-1 at margin >= 0.5 / KL (2784 positions) |
|---|---|---|
| stock packer (`mimo-v2.6/dev`, merged branch) | 0.8945 / 0.9739 / 0.0609 | 0.9077 / 0.9755 / 0.0556 |
| MSE | 0.9020 / 0.9780 / 0.0466 | 0.9030 / 0.9830 / 0.0310 |
| **GPTQ** | **0.9160 / 0.9876 / 0.0330** | **0.9242 / 0.9880 / 0.0274** |

Decode top-1 by reference margin (stock -> GPTQ):

| Reference margin | stock | GPTQ |
|---|---|---|
| < 0.1 | 0.53 | 0.51 |
| 0.1-0.2 | 0.65 | 0.73 |
| 0.2-0.5 | 0.84 | 0.86 |
| 0.5-1 | 0.937 | 0.972 |
| > 1 | 0.990 | 0.994 |

On the 8 golden prompts (100 positions, `test_model_error_tp8_dp4.py`), the mean KL goes from 0.1535 (stock, bf16
logits) to 0.068 (GPTQ). The first token's massive activation (about 1090 in dim 2412, cancelled in layers 41-45) had
carried most of it.

**Final configuration on `mimo-v2.6/c16-top1`** (mimo-v2.6/vllm-c16 + this work). Settings: GPTQ experts
(`MIMO_EXPERT_QUANT=gptq`), fp32 logits, bf16-dest SDPA. Measured 2026-10-20 on one BH Galaxy
(`runs/top1/final_validate.sh`):

| Check | Result |
|---|---|
| Prefill, 3072 positions | top-1 0.9215 (stock 0.8945), at margin >= 0.5 0.9872 (0.9739), KL 0.0329 (0.0609) |
| Teacher-forced decode, 2784 positions | top-1 0.9221 (stock 0.9077), at margin >= 0.5 0.9875 (0.9755), KL 0.0280 (0.0556) |
| `test_model_tp8_dp4` | rows with a reference margin >= 0.2: 34/34 (the merged branch missed a 0.61-margin row). Raw top-1 0.900: the 4 misses are reference near-ties (0.068-0.179) |
| `test_model_traced_tp8_dp4` | traced == eager (PCC 1.000000) |
| vLLM, c=16 (64 x 128 / 128, DP=4 x 4) | 1005.4, 1023.4, 1004.7 tok/s; TPOT 13.2-13.5 ms; TTFT median 317-322 ms |
| vLLM, c=1 | TTFT 176 ms, TPOT 13.1 ms |
| vLLM, other checks | HTTP suite 78 passed, 1 skipped; served greedy == standalone 16/16; 0 decode-trace recaptures |

## Long context, 2026-10-21 (#61416, on `mimo-v2.6/dev`, MOPD)

Chunked prefill stays correct at 32K and 128K on the decode cache layout (128-wide global V, SWA paged by absolute
position). Measured on one BH Galaxy with the default MSE experts:

| Test | What it runs | Result |
|---|---|---|
| `test_chunked_prefill_tp8_dp4.py` short (layers 0, 1, 5) | 4 prompts of 200-1333 tokens, chunks of 64-512 | chunked vs reference >= 0.99892 (delta >= 0.9937), vs one shot >= 0.99932, written K / V >= 0.99993, decode >= 0.99978 (delta >= 0.9981) |
| `test_chunked_prefill_tp8_dp4.py` 32k (layers 0, 1, 5) | one 32768-token prompt per row, chunks of 1024 / 2048 / 4096 / 8192, vs the fp32 reference (attention in query blocks); 32 decode steps | chunked vs reference >= 0.99843 (delta >= 0.9969), vs one shot >= 0.99973 (delta >= 0.9993), written K / V >= 0.99993, decode >= 0.99997 (delta >= 0.9990) |
| `test_model_long_tp8_dp4.py` 32k | 48 layers, 4 users per row; per row a passkey prompt one shot and in chunks of 1K-8K, plus a shared prompt | same next token for both cuts on every row, logits PCC 0.9948-0.9972; pass key found at 10 / 35 / 65 / 90% depth by both cuts; shared prompt identical on all 4 rows (PCC 1.000000) |
| `test_model_long_tp8_dp4.py` 128k | as 32k, baseline in 16K chunks | same next token everywhere, logits PCC 0.9930-0.9948; every pass key found; shared prompt identical on all 4 rows |
| vLLM, `--max-model-len 40000` (625 blocks) | DP=4 x 4 seqs, a 35K-token needle at 30% depth and a short request | page tables 640 wide; needle recalled, first token after 8.0 s; short request correct |

| | 32K | 128K |
|---|---|---|
| Prefill, 3 prompts per row on 4 rows at once | 64 s | 355 s |
| DRAM per chip: weights / + KV for 4 users / total | 24.24 / 24.50 / 34.18 GB | 24.24 / 25.10 / 34.18 GB |
| Prefill chunks that run after decode (probe from 4096 up) | every size to 32768, the whole prompt (5 s) | every size to 131072, the whole prompt (38 s) |

Fixes that came out of these runs:

- **MoE prefill in 4K slices.** The router's prefill program blocks a whole chunk per core. A 32K chunk needed about
  4.9 MB of L1, and after decode even 8K did not fit: its static circular buffers end at 1.32 MB, past the L1 buffers
  decode keeps allocated (from 1.21 MB). `TtMiMoMoE.forward` runs chunks longer than 4096 tokens through the router and
  experts in 4K slices and all-reduces the whole chunk once; each token's output is unchanged. vLLM sends 4K chunks and
  was not affected.
- **Page table width.** `SwaDecodeQkv` reads a page table row in 64 B pieces, so `attention.page_table_width` pads the
  width to 16 blocks, for the standalone tables and for vLLM's. A `max_seq_len` or `--max-model-len` that was not a
  multiple of 1024 tokens failed on the first decode step.

## Long context: 8 users x 1M (branch `mimo-v2.6/long-ctx`, 2026-10-20)

Goal: 8 users (2 per lane) with 1M context each under vLLM, prefix caching (APC) correct, and the per-user decode
speed. Two changes on top of `mimo-v2.6/c16-top1`:

1. **Global decode SDPA over the decode users' rows only.** `paged_flash_multi_latent_attention_decode` takes its
   batch from the page table's rows and splits its cores evenly over them. With the 32-row table each of 2 users got
   2 of 64 cores: 21 ms per global layer at 1M context (17 GB/s). The model now slices the page table and the owned
   positions to the decode users (2 rows: 32 cores per user, 256-token K chunks); ConcatHeads zero-fills the rows
   past them. One chip, 1M context, one global layer: 21.0 -> 1.15 ms (311 GB/s; 1.07 ms with the 12 x 10 grid).
2. **128-wide SWA V cache.** vLLM sizes each KV buffer to the lane's whole block pool and shares it between one
   layer of each KV cache group, so the SWA layers' 192-wide (zero-padded) V buffers cost 8 x 192 of the 4692 bytes
   per token per chip. The SWA V cache now keeps V's 128 real dims and shares the global layers' V buffers (9 K + 9
   V buffers): 3060 bytes per token per chip, so 2 x 1M per lane fits in 6.55 GB per chip (33472 blocks per lane).
   SWA decode reads it with `paged_flash_multi_latent_attention_decode` (`head_dim_v` 128, sinks, sliding window);
   SwaDecodeQkv writes 4 V tile columns; a later SWA prefill chunk pads the cached window back to 192. vLLM's block
   tables, prefix caching and eviction are unchanged.

Accuracy (unchanged): `test_decoder_tp8_dp4` 3 passed, `test_chunked_prefill_tp8_dp4` 5 passed,
`test_model_tp8_dp4` top-1 0.900 with 34/34 clear rows, PCC min 0.9372 / mean 0.9941.

Traced decode step, 2 users per row (`tests/perf/long_decode.py`, device-resident greedy loop; only the prompt's
KV is real, the step attends over the whole context):

| Position | Before, 4 rows | After, 1 row | After, 4 rows | 4 rows, `MIMO_SDPA_FULL_GRID=1` |
|---|---|---|---|---|
| 4K | 13.95 ms | 11.57 ms | 13.55 ms (73.8 tok/s per user) | 13.25 ms |
| 128K | 36.99 ms | 12.36 ms | 14.61 ms (68.4) | 14.96 ms |
| 512K | 107.94 ms | 16.63 ms | 18.33 ms (54.6) | 17.96 ms |
| ~1M | 201.11 ms | 21.38 ms | 23.29 ms (42.9) | 22.61 ms |

vLLM, DP=4 x 2 seqs, `--max-model-len 1048576`, `MIMO_KV_GB_PER_CHIP=6.6` (`runs/swa1/exp1m.sh`):

| Check | Result |
|---|---|
| c=8, 64 x 128 / 128 | TPOT 12.28-12.34 ms (81 tok/s per user), median ITL 11.06-11.27 ms, 556-560 tok/s |
| c=1 | TPOT 10.24 ms (98 tok/s), TTFT 184 ms |
| c=8, 8 x 128K / 256 | median ITL 12.22 ms (82 tok/s per user once both users of a lane decode); prefill ~68 s per 128K prompt, so the mean TPOT (190 ms) is the other user's prefill chunks interleaved with decode |
| APC (`runs/swa1/apc_check.py`, 13.7K-token shared document, 8 questions) | answers 24/24 correct; greedy text with prefix hits == without 16/16; 8/8 requests of the second warm round reused 13632-13696 of 13701 prompt tokens, in 2.16 s instead of 3.0 s |
| HTTP suite (`tests/tt`, max_num_seqs 8) | 78 passed, 1 skipped |

Decode at 1M is bandwidth-bound: each chip reads its own user's global KV, 9 layers x 1M x 340 bytes = 3.2 GB per
step, which takes 7.1 ms at this Galaxy's 448 GB/s (GDDR6 at 14 Gbps). The SDPA reads it at ~330 GB/s (9.7 ms).
With the 12.2 ms rest of the step, 70 tok/s per user (14.3 ms) holds up to ~350K context; at 1M the ceiling is
~1000 / (12.2 + 7.1) = 52 tok/s, and the step is ~21-23 ms (43-48 tok/s) today. Prefill of a 1M prompt is the other
limit: the chunked MLA prefill makes a 2048-token chunk at context C cost roughly 0.25 s + 12.5 ms x C / 1K, ~1 hour
per 1M prompt.

## Device sampling, 2026-10-21 (branch `mimo-v2.6/device-sampling`, MOPD)

With greedy rows only, vLLM samples on device (GreedyHead in the decode trace, async steps). Any row with
temperature > 0 sent the whole batch to host sampling: each step reloaded its inputs, read the users' fp32 logit rows
(2.4 MB per lane) and sampled on the CPU, synchronously. Sampled c=16 decode was 7.5x slower than greedy.

`tt/device_sampler.py` draws temperature / top-k / top-p tokens inside a second decode trace, which a lane replays
while it holds a sampled row (greedy batches keep the GreedyHead trace and its speed):

| Step | Cores per chip | What |
|---|---|---|
| SampleScan | 32 | Each core loads its range of the chip's vocab shard (fp32 tiles). The compute kernel sums exp((logit - the range's largest) / T) on the SFPU (fp32 unpacked straight to DST). The reader and the writer keep two users' 32 largest logits each (exact values, integer keys), and the writer adds up the sums |
| SampleChipMerge | 1 per user | The chip's sorted top 32 and its normalizer, written to one page |
| all_gather | | The row's 8 pages |
| SampleMerge | 1 | Walks the merged candidates in descending order and draws, as vLLM's `apply_top_k_top_p`: softmax over the vocabulary (or over the top_k), the nucleus keeps a token while the mass ranked above it is below top_p, inverse CDF with a splitmix64 uniform whose state lives on device |

The data movement cores have no FPU, so steps 2 and 3 run in fixed point: logits in Q24, probabilities relative to
the largest in Q30, `exp2` from a 64-entry table and a cubic. The draw equals vLLM's whenever the nucleus has at
most 32 tokens, or more but inside the row's 256 candidates (the 32 largest of every chip). Measured on GPQA reasoning
at temperature 1 and top_p 0.95 (8848 decode rows, host sampler instrumented): the nucleus is 1 token at p50, 5 at p90,
20 at p99; 99.7% of rows have at most 32, and 99.9% fit in 256.

Routing (vllm-tt-plugin): the model declares `max_device_top_k` 256 and `supports_device_top_p_without_top_k` (a row
with no top_k and top_p < 1 stays on device). Pure temperature sampling (no top_k, top_p 1), penalties, logprobs and
structured output still sample on host. Seeded rows sample on device: a row's stream state at decode position p is
mix(seed) + p * GAMMA (`stream_states`), so its draws depend only on (seed, position), whichever steps reload it.

`tests/test_device_sampler_tp8.py`: 312 rows over 4 parameter sets, 3 logit shapes, eager and traced draws. All 168
rows whose nucleus lies in the 32 most likely tokens match the exact full-vocabulary draw (same uniform, replayed
from the seeds); greedy rows take the lowest id among tied maxima. Traced head: 0.164 ms per step (GreedyHead 0.08 ms).

vLLM, DP=4 x 4, 64 x 128 / 128 (`--temperature 1.0 --top-p 0.95` for the sampled rows), one server per column:

| | Before (host sampling) | ttnn-op sampler (first version) | Fused head (this branch) |
|---|---|---|---|
| Greedy c=16 | 993.9 tok/s, TPOT 13.3 ms | 1018.5 tok/s, TPOT 13.2 ms | 994.9 tok/s, TPOT 13.3 ms |
| Sampled c=16 | 133.1 tok/s, TPOT 117.6 ms | 940.8 tok/s, TPOT 13.8 ms | **956.8 tok/s, TPOT 13.5 ms** |
| Sampled c=1 | 25.6 tok/s, TPOT 37.7 ms | 66.8 tok/s, TPOT 13.3 ms | **68.3 tok/s, TPOT 13.1 ms** |

The sampled rows' first token still samples on host after prefill (`sample_on_device_mode="decode_only"`); their TTFT
is ~90 ms above greedy's at c=16 (423 vs 324 ms median), not yet explained by the host sampler (~3 ms for 16 rows).

Validation in serving:
- `MIMO_SAMPLER_CHECK=1` (debug: each sampled step blocks, reads the logit rows and the rows' stream states, and
  recomputes vLLM's draw on the host): 5 minutes of GPQA reasoning at c=16, 14333 sampled rows on the model's own
  logits, 14278 with the nucleus in the candidates, 0 mismatches.
- GPQA Diamond, first 64 questions, c=16, temperature 1.0 / top_p 0.95, 64K context (`runs/speed/gpqa_device.sh`):
  50/64 (53/64 counting 3 answers that end in the template's literal "Answer: $LETTER" after reasoning to the right
  letter); 6 truncated at 64K; 0 failed requests, 0 decode-trace recaptures; 1046 tok/s while all 16 requests decode
  (host sampling: ~133). On the 62 questions both runs answered, host sampling 56 and device sampling 52 (with the
  reasoning fallback), within T=1 run-to-run spread.

## Speculative acceptance on device, 2026-10-22 (branch `mimo-v2.6/dev-mtp-spec-decode-sampled-device-accept`)

Not yet run on hardware. Sampled MTP speculation in the plugin's `logits` mode, BH Galaxy, c=1 SPEED-Bench coding at
temperature 1.0 / top_p 0.95: 2.60 accepted tokens per step and a 40 ms step (TPOT 14.7 ms): verify on device 16.2 ms
(2.4 ms of it reading back 16 x 4 x 152k fp32 logits), verify host 6.0, the plugin's host accept walk 6.1, MTP draft
5.6 + 2.6, gap 3.2. A greedy MTP step is ~28 ms; plain decode with the device sampler 13.1 ms TPOT.

`fused_sample` (`tt/verify_sampler.py`) accepts on device instead: after each verify pass a small trace runs
SampleScan and SampleChipMerge over the pass's slots with each slot's request parameters, the row all_gather, and
SampleVerify (`kernels/sample_verify.cpp`, one core), which walks each request of the pass as the plugin's host walk
does (point-mass drafts: accept draft j when p_j(d_j) > 0 and p_j(d_j) >= u_j; at the first rejection draw from p_j
without d_j; else the bonus from p_n) with uniforms the plugin draws from the request's own generator, in fixed point
as the decode sampler (Q30 probabilities, Q24 uniforms). Inverse-CDF draws walk the candidates in the plugin's table
order (logit then id for a top_k row, id for a nucleus row), so the same uniform draws the same token as the host.
Greedy requests compare with the verify's GreedyHead tokens, exactly the argmax_ids walk. A request whose nucleus
runs past the candidates is accepted on them and counted.

Expected per step: the 2.4 ms logits readback and the 6.1 ms host walk go, and the part of the verify host time that
fills the plugin's [16, 4, 152k] logits buffer; the accept trace adds the sampler head per pass (the decode head takes
0.164 ms) and a 128-byte read. Roughly 40 -> 30-32 ms per step, TPOT ~11.5-12 ms at 2.6 accepted tokens: about a
greedy MTP step plus the accept head. The MTP draft (8.2 ms) and the gap (3.2 ms) are unchanged.

## Prefill profile, 2026-10-21 (#61418, on `mimo-v2.6/dev`, MOPD)

`tests/perf/profile_prefill.py` under `python -m tracy` profiles the last 4096-token chunk of a prompt on one row,
one user, all 48 layers. The zones (`tt/profiling.py`, on with `MIMO_PROFILE_ZONES=1`) mark every layer and its
QKV, RoPE, KV write, SDPA, o_proj, all-reduces, router, expert plan, grouped expert matmul and combine;
`tests/perf/parse_prefill_zones.py` sums the device-kernel time between the markers (worst of the row's 8 chips).
The profiler is read every 3 layers: one read of a 48-layer chunk (~1700 programs per chip) fails its marker
matching. For 32K and 128K the earlier chunks only take their SWA blocks (`--fill skip`) and the cache keeps zeros:
at 32K on layers 0, 1, 5 this changes the chunk by 0.4% and SDPA by 0.0%, the MoE experts by 4-10% (routing sees
other hidden states).

TTFT, one user on one row, random tokens, chunks of 4096, profiler off (`profile_prefill.py --ttft`, second run):

| Prompt | TTFT | tokens/s |
|---|---|---|
| 4K | 0.58 s | 7.1k |
| 32K | 5.12 s | 6.4k |
| 128K | 29.06 s | 4.5k |

Device time of the last chunk, all 48 layers (ms, summed over the layers):

| Zone | 4K (start 0) | 32K (start 28K) | 128K (start 124K) |
|---|---|---|---|
| MoE grouped expert matmul | 186.9 | 178.4 | 176.4 |
| MoE plan (`DevicePlan`) | 74.6 | 73.4 | 73.3 |
| MoE combine | 38.1 | 38.0 | 38.0 |
| MoE router | 23.2 | 23.1 | 23.2 |
| SDPA, 9 global layers | 7.6 | 168.6 | 704.9 |
| SDPA, 39 SWA layers | 10.5 | 15.3 | 15.3 |
| QKV + o_proj | 54.6 | 62.0 | 62.0 |
| RoPE + KV write | 8.5 | 10.3 | 10.3 |
| All-reduces (2 per layer) | 77.8 | 80.6 | 84.0 |
| Dense MLP (layer 0) | 2.2 | 2.2 | 2.2 |
| Norms, residual adds, other | 46.1 | 45.7 | 45.7 |
| **Chunk** | **530.1** | **697.6** | **1235.2** |

- The MoE is ~311 ms of every chunk at any context: 59% of a 4K chunk. The expert plan (1.56 ms per layer) moves
  almost no data (1.2 GB/s): it is latency, 14% of a 4K chunk.
- A global layer's SDPA grows linearly with the cached context (0.84 ms at 4K, 18.7 ms at 32K, 78.3 ms at 128K), so
  over a whole prompt it grows quadratically: 57% of the 128K chunk, ~11 s of the 29 s 128K TTFT (the MoE ~10 s).
  Summing the chunks this way gives 28 s for 128K, against 29.06 s measured.
- The global SDPA is compute-bound: 256 (head, 256-query) units of 131072 keys each, 4 per core on the 8 x 8 grid,
  ~1.1 TFLOP/s per core at HiFi4. Its other chip of the pair runs the same program on the scratch block (#60957).

### Global prefill SDPA at 128K: program config and math fidelity

One global layer's SDPA, last chunk of 128K (layers 0, 1, 5 profiled, `--fill skip`):

| Change | ms | |
|---|---|---|
| 8 x 8 grid, Q / K chunks 256, HiFi4 (before) | 78.3 | |
| Whole 12 x 10 grid (120 cores) | 78.4 | the 128 causal (q, q') pairs still give some cores 2 |
| Q chunk 128 / 64 (whole grid) | 97.2 / 175.4 | smaller units run slower; halving the units (Q 512) gains 1% |
| Q chunk 128 (8 x 8) | 86.9 | |
| K chunk 128 / with Q chunk 512 | 90.5 / 89.3 | Q 512 with K 256 and Q 1024 do not fit L1 |
| HiFi3 | 62.4 | |
| **HiFi2 (now the default, `MIMO_PREFILL_SDPA_FIDELITY` overrides)** | **46.7** | 1.68x |

Accuracy at HiFi2 against HiFi4 (decode SDPA stays HiFi4):

- `test_chunked_prefill_tp8_dp4.py` (short and 32k, layers 0, 1, 5): 8 passed, every prefill PCC within 0.0008 of
  HiFi4 (worst chunked vs reference 0.99843 -> 0.99860, worst delta 0.99374 -> 0.99369).
- `test_model_tp8_dp4.py`: top-1 0.950 both, 34/34 rows with a clear margin; logits PCC min 0.9004 -> 0.8658 (a row
  whose reference margin is 0.07), mean 0.9929 -> 0.9917.
- `test_model_long_tp8_dp4.py -k 32k`: every pass key found at 10-90% depth by both cuts, chunked vs one shot logits
  PCC 0.9933-0.9966 (HiFi4: 0.9948-0.9972), shared prompt identical on all rows.

TTFT (one user, one row, chunks of 4096): 4K 0.58 -> 0.56 s, 32K 5.12 -> 4.78 s, **128K 29.06 -> 24.35 s**.

### The next three costs, 2026-10-22

**Global SDPA on both chips of the pair.** A later chunk of a global layer ran the group's 16 heads on both chips of
the pair and kept one result (`_prefill_owner_chunk`). When both chips keep the user's K / V, it now runs 8 heads per
chip like a first chunk does, over the whole cached context, and the row all-reduce adds the two o projections
(`_prefill_pair_chunk`, `TtMiMoAttention.keep_kv_on_both_chips`). vLLM's caches keep both copies with
`MIMO_KV_PAIR_SPLIT=0`, so serving then takes this path (with the pair split, the default, each chip keeps half of the
keys and runs all 16 heads over them, the same work: `_prefill_kv_split_chunk`, `doc/vllm_integration/README.md`);
the standalone model takes it with `TtMiMoModel(replicate_global_kv=True)` (twice the global KV: 0.8 -> 1.6 GB per
chip for 4 users at 128K). `test_chunked_prefill_tp8_dp4.py` has `kvboth` cases for layers 0 and 5
(short and 32k, and the program-cache test): all pass, every PCC within 0.0002 of the split-user layout.

**MoE plan in two programs.** `plan.cpp` had every one of its 16 cores read and count all T x 8 pairs, then scan them
again for its experts (latency-bound, 1.2 GB/s). `DevicePlanSplit` (`plan_slice.cpp`, `plan_gather.cpp`): each core
counts and sorts one slice of the tokens by expert, then each core gathers its experts' records from every slice in
slice order. The outputs are bit-identical (`tests/perf/moe_plan_bench.py` checks all four on skewed routing).
Device time per call on one row:

| Tokens | `DevicePlan` | `DevicePlanSplit` |
|---|---|---|
| 32 | 0.135 ms | 0.180 ms |
| 1056 | 0.551 ms | 0.339 ms |
| 4096 | 1.553 ms | 0.601 ms |

`PrefillExperts` uses the split plan from 256 tokens (`SPLIT_PLAN_MIN_TOKENS`). A 4K chunk of all 48 layers:
530.1 -> 482.0 ms (the plan 74.6 -> 29.8 ms).

**Fabric packet size.** 8192-byte packets (`--fabric-packet 8192` in `profile_prefill.py`) take 0.03-0.14 ms off each
all-reduce: 29.5 -> 29.1 ms for layers 0, 1, 5 at 4K, about 7 ms of a 48-layer chunk. The packet size is set where the
mesh is opened (test fixtures, the vLLM plugin) and changes decode's collectives too, so it is left at the default
until decode is measured with it.

TTFT, one user on one row, chunks of 4096, profiler off:

| | 4K | 32K | 128K |
|---|---|---|---|
| Before (HiFi4) | 0.58 s | 5.12 s | 29.06 s |
| Prefill SDPA at HiFi2 | 0.56 s | 4.78 s | 24.35 s |
| + global heads split over the pair (`replicate_global_kv`) | 0.57 s | 4.53 s | 20.62 s |
| + MoE plan in two programs | **0.54 s** | **4.18 s** | **19.20 s** |

### MoE prefill experts: where the time goes, 2026-10-27

`tests/perf/moe_experts_bench.py` times the plan, `GroupedExperts` and `CombineK` of one layer alone (4096 tokens,
the layer's router on random tokens: 1127 groups, 10% padding rows, 192 experts used, up to 35 groups per expert).
Under `python -m tracy -r` with `MIMO_PROFILE_ZONES=1`, `tests/perf/moe_bench_parse.py` prints device time per
piece. `--configs` takes `/`-separated define sets for the kernels' ablation switches (`ABL_NO_X` / `ABL_NO_Y` /
`ABL_NO_W` / `ABL_NO_MM`: skip the x row reads, the y row writes, the weight reads after the first job, the matmuls).

| `GroupedExperts` on `mimo-v2.6/dev` (device ms) | |
|---|---|
| as is | 3.69 |
| LoFi | 3.57 |
| no y writes | 2.59 |
| no x reads / no weight reads / no matmuls | 3.52 / 3.53 / 3.34 |
| no x, y / no x, y, weights / no x, y, weights, matmuls | 2.15 / 1.78 / 1.27 |

The kernel is neither DRAM-bound (238 GB/s of ~450) nor math-bound (17% of HiFi2 peak). The y row writes are 1.1
ms: 256 writes of 256 B per core per group, issued by the writer between the team's x and h swaps, so they sit on the
critical path. With no DRAM traffic and no matmuls the program still takes 1.27 ms, 34 us per group: the team's two
swaps per group, tilize and untilize, SwiGLU, all serial per group.

Changes:

- **LoFi.** Matmul unpacks the weights (in1) to srcA; LoFi takes srcA's hidden bit + 4 mantissa MSBs, all of BFP4's
  3 bits, and srcB (x, h) gets 6 mantissa bits at LoFi and at HiFi2. The output is bit-identical to HiFi2.
- **2 KB y rows, off the critical path.** Each team core owns 32 adjacent output columns (32 q + c, read tile by tile
  from the 8 banks as before) and untilizes each 4-column pass into a 32-tile-wide row-major block
  (`pack_untilize_dest<4, 32>`), so it writes 32 rows of 2 KB per group instead of 256 of 256 B. The writer now swaps
  h of group n, then x of group n + 1, then writes group n's rows, which overlap gate | up of group n + 1; compute
  tilizes group n + 1's x quarter between h and down of group n, and the reader reads a job's first group before its
  weights (else the reader would wait for the slot the previous job's last down frees). cb_y and cb_xrm hold two groups with
  BFP4 weights and one with BFP8 (whose weight slices fill L1). Bit-identical output.
- **Team plan without the cap x E scan.** `plan_gather.cpp`'s core 0 dealt the team plan with a loop over cap x 256
  experts (about 9.7K iterations with divisions at 4K). It now deals the whole chunks, then the remainders bucketed
  by size, in the same order: the same plan (`moe_plan_bench.py`: identical to `DevicePlan` at 32, 1056, 4096).

| One layer, device ms | before | after |
|---|---|---|
| plan (`plan_slice` + `plan_gather`) | 0.599 (0.125 + 0.467) | 0.360 (0.125 + 0.236) |
| `GroupedExperts` | 3.53 | 3.19 (model FLOPs: 58 -> 65 TFLOP/s) |
| `CombineK` | 0.83 | 0.83 |

A 4K chunk of all 48 layers saves about 27 ms (47 MoE layers x (0.34 + 0.24 ms)). TTFT, one user on one row, chunks
of 4096, `--replicate-global-kv`, profiler off (`profile_prefill.py --ttft`):

| | 4K | 32K | 128K |
|---|---|---|---|
| before (2026-10-27 profile) | 0.502 s | 4.18 s | 18.96 s |
| after | 0.477 s | 3.93 s | 17.88 s |

Accuracy (MOPD, BFP4 experts): `test_model_tp8_dp4.py` with `MIMO_MOE_DEDUP=0`: top-1 0.950, 34/34 rows with a
reference margin >= 0.2, logits PCC mean 0.9917 (min 0.8658); `test_model_long_tp8_dp4.py -k 32k`: every pass key
found by both cuts, next-token logits PCC 0.9906-0.9952 with the same top-1, shared prompt identical on all rows.
The grouped output is bit-identical to the dev kernel's (`moe_experts_bench.py --ref`).

Tried and not kept:

- Teams of 8 cores (4 x 2), K-split: each core tilizes and multiplies only its own 16 of the 128 x tiles for all
  8 intermediate columns and the team reduce-scatters fp32 partials (8 KB per peer, unpacked to dest), so there is no
  x swap, and with 221 KB of weights per core the next job's weights load during the current job. 3.25 ms
  (`matmul_block` 3.19 ms), no better than the above: twice the groups per team, with the same per-group swaps.
- Two items in `CombineK`'s input buffer: 0.83 ms either way (it reads 268 MB of y; DRAM-bound).

What is left: the y rows (each pair's full-width bf16 output, 268 MB written, read again by `CombineK`) are 0.75 ms
of the grouped program and all of `CombineK`. Cutting them needs the k sum before DRAM, which expert-major order does
not give.

### Global prefill SDPA past 128K: grouped heads and split-K, 2026-10-23

Up to 1M (one user per row, `--users 1`, so the cache fits), the 9 global layers' SDPA is most of a long prompt.
After the pair split, one global layer's later chunk runs `chunked_flash_mla_prefill` with 8 Q heads over one KV
head per chip: 8 heads x 16 Q chunks = 128 units on the 8 x 8 grid, every unit streaming the whole cached K / V
(340 B per position, BFP8). At 1M that is 45.6 GB of K / V reads per call and 108 TFLOP/s, a third of the chip's
HiFi2 peak (~330 TFLOP/s on 120 cores). `tests/perf/global_sdpa_bench.py` times this one call alone
(`grid:q_chunk:k_chunk[:gG][:sS]` configs, last 4096-token chunk of each context).

Two program options in ttnn, both off by default:

- **Grouped heads, `SDPAProgramConfig.q_heads_per_unit` (G).** G Q heads of one KV head form a unit; each K / V chunk
  read is used for all G heads before it is released, so the reads drop by G. Each head keeps its own accumulators.
- **Split-K, `k_splits` (S), now honoured on the chunked path.** Each unit's causal K range is cut into S slices on
  device from the runtime chunk start (one program for every start), and the slices spread over the whole grid. A
  piece writes its normalized output with its row max and row sum to its own plane of a
  `[pieces, H, Sq, DV + 64]` tensor; empty planes carry sum 0. `chunked_flash_mla_prefill` merges the planes with
  ttnn ops, weights in fp32 (3-5 ms per call). Causal pair distribution is off for these units: with 64 units it put
  them on 32 cores.

G = 1, S = 1 builds the same kernels as before. Validation limits G > 1 / S > 1 to chunked causal prefill on the
streaming kernel with bf16 dest, no sink or sliding window, Q heads divisible by KV heads x G, and (S > 1) bf16 Q and
output.

One global layer, last chunk, one chip (ms):

| Program | 128K | 512K | 1M | TFLOP/s at 1M |
|---|---|---|---|---|
| 8 x 8, Q / K 256 (before) | 24.5 | 100.6 | 202-204 | 108 |
| 12 x 10, Q / K 256 | 24.6 | 100.8 | 203.2 | 108 |
| 8 x 8, G 2 | 24.5 | 100.1 | 199.6 | 110 |
| 12 x 10, S 15 | 21.7 | 77.2 | 155.1 | 142 |
| **12 x 10, G 2, S 15 (MiMo now)** | **19.9** | **71.5** | **139-143** | **153-158** |
| 12 x 10, G 2, S 30 / S 120 | 20.1 / 20.2 | 72.7 / 73.1 | 143.9 / 143.9 | 153 |
| 12 x 10, G 2, S 15, K 512 | | | 130.6 | 168 |
| 12 x 10, G 2, S 15, K 128 / Q 128 | | | 161.5 / 158.3 | |
| 8 x 8, Q 128, K 768, K / V forwarding chains | | | 188.3 | 116 |

- G alone gains nothing: on 64 cores the program is compute-bound, not read-bound. S alone fills the 120 cores, but
  then 128 units' reads hit the K / V read ceiling (~300 GB/s). Together the reads halve and all cores work.
- K 512 is the fastest, but its circular buffers (~1.4 MB, estimated) do not fit beside the L1 buffers vLLM's
  decode keeps allocated (from 1.21 MB); K 256 is ~1.06 MB. G 3 does not fit at Q / K 256.
- The last row is the other design tried: the cores of a KV head forward the K / V prefix along chains instead of
  each reading it from DRAM (#57395 extended to paged MLA prefill). It is capped at ~126-129 TFLOP/s on the grid and
  was not kept.
- Below 64K of context the merge costs more than the split saves (32K: 8.7 ms against 5.9 ms).

MiMo (`TtMiMoAttention._prefill_sdpa_program`): a global layer's later chunk (`_prefill_pair_chunk`,
`_prefill_kv_split_chunk`, `_prefill_owner_chunk`) starting at 64K or later, with Q chunk 256 and bf16 dest, takes G 2,
S 15 on the whole grid (`attention.later_chunk_split`; the vLLM warmup runs it once at 64K).
`MIMO_GLOBAL_SDPA=legacy` keeps the 8 x 8 program everywhere, `MIMO_GLOBAL_SDPA=gGsS` picks G and S, and
`MIMO_GLOBAL_SDPA_MIN_START` moves the threshold.

Accuracy:

- `test_mla_prefill.py::test_chunked_flash_mla_prefill_grouped_heads`: G 2 is bit-identical to G 1 at four chunk
  starts, Q / K 128 and 256, and the program cache does not grow across starts.
- `test_mla_prefill.py::test_chunked_flash_mla_prefill_k_split`: (G, S) in (1, 3), (2, 3), (1, 15), (2, 15), shuffled
  pages: PCC against torch 0.99966-0.99978, max abs error 0.0053-0.0557 (unsplit 0.0060-0.0557).
- SDPA regression (`test_mla_prefill.py`, `test_sdpa_prefill.py`, `test_sdpa_chunked.py`,
  `test_mla_prefill_chunked_vs_not.py`): 102 passed, 2 skipped.
- With the split forced on every eligible chunk (`MIMO_GLOBAL_SDPA_MIN_START=0`): `test_chunked_prefill_tp8_dp4.py`
  13 passed; `test_model_tp8_dp4.py` top-1 0.950, 34/34 rows with a clear margin, logits PCC mean 0.9917 (as before);
  `test_model_long_tp8_dp4.py -k 32k` every pass key found.

TTFT, one user on one row (`profile_prefill.py --ttft --replicate-global-kv --users 1`, chunks of 4096, second run):

| | 4K | 32K | 128K | 512K | 1M |
|---|---|---|---|---|---|
| Before (HiFi2, pair split, MoE plan) | 0.520 s | 4.163 s | 19.22 s | 119.3 s | 353.0 s |
| K / V forwarding chains (not kept) | 0.549 s | 4.171 s | 18.91 s | 114.0 s | 333.1 s |
| **Grouped heads + split-K** | **0.516 s** | **4.163 s** | **18.83 s** | **101.1 s** | **272.5 s** |
| tokens/s | 7.9k | 7.9k | 7.0k | 5.2k | 3.8k |

(`--users 1` sizes the caches for one user; the 4-user table above has 4K 0.54 s, 32K 4.18 s, 128K 19.20 s.)

- The 1M saving, 80.5 s, matches the bench: about 36 ms per layer on an average chunk (half the context) x 9 layers x
  256 chunks = 82 s.
- From the bench, the global SDPA is still ~55% of the 1M TTFT (~150 s). The rest is the ~470 ms every chunk spends
  in the MoE, projections and all-reduces at any context (~120 s over 256 chunks).
- Up to 32K nothing changes: the MoE (grouped expert matmul ~177 ms of a chunk) and the SWA layers dominate.

Left for later: merging the split planes on chip (saves the 3-5 ms and the ~38 MB DRAM tensor per call), K 512 if
decode's L1 buffers shrink, the MoE expert matmul for short prompts, a traced prefill chunk, and batch > 1 on the new
programs (written, not run on device).

## 1M prefill on all 4 rows, 2026-10-26 (#61419, on `mimo-v2.6/dev`, MOPD)

`tests/perf/long_1m.py` runs the standalone model on all 4 rows, built at `max_seq_len` 1M + 512, with chunks of 4096
and the global K / V on both chips of a pair (`--replicate-global-kv`, vLLM's layout) unless noted.

- `--mode capacity` logs DRAM per chip after the weights and with the KV caches, then runs the last 4K chunk of a 1M
  prompt on row 0 (the earlier chunks only take their SWA blocks; the cache keeps zeros) and one decode step of every
  user at position 1M - 1.
- `--mode passkey` gives each user its own 1M passkey prompt (the filler and question of `test_model_long_tp8_dp4.py`).
  The rows run at once, with their chunks enqueued round-robin; the users of a row run one after the other. TTFT is
  from the start of the run and includes the first chunk's compile. Then 12 greedy tokens per user are decoded from
  that cache, untraced, and each key must be in the text.

DRAM per chip at 1M (34.18 GB per chip, weights 24.24 GB):

| Users per row | Global K / V | Used | Free | Job |
|---|---|---|---|---|
| 1 | both chips of the pair | 27.52 GB | 6.66 GB | 714 |
| 2 | both chips of the pair | 30.74 GB | 3.44 GB | 717 |
| 4 | the pair splits the users | 30.71 GB | 3.47 GB | 714 |
| 4 | both chips of the pair | out of memory allocating the global KV, 33.69 GB in use | | 715 |
| 4 | the pair splits every 128-token block (`--kv-pair-split`, vLLM's default layout) | 30.72 GB | 3.46 GB | 2026-10-27 |

Used DRAM does not change after the last 4K chunk at 1M, after prefill or after decode (to 0.01 GB). The last 4K chunk at
1M takes 1.46 s with the K / V on both chips (8 heads per chip) and 2.78 s when the pair splits the users (the owner
runs all 16 heads). One untraced decode step at 1M - 1 on one row: 0.19 s with 1 or 4 users, 0.17 s with 2.

With the KV pair split (`mimo-v2.6/kv-pair-split`, 2026-10-27, `--mode capacity --users 4 --kv-pair-split`) 4 users
per row fit as when the pair splits the users (30.72 GB, 3.46 GB free), and the last 4K chunk at 1M takes 1.57 s: each
chip runs all 16 heads over its half of the keys (G 2 / S 15) and the pair merges by LSE, 7% over the head split's
1.46 s, which only fits 1 user per row. Scaled by the last chunk, about 300 s per 1M user on one row, ~20 minutes
for 4 (against ~28 when the pair splits the users). The decode step: 0.196 s with 4 users.

1M passkey prompts:

| | 1 user (row 0) | 4 users (1 per row) | 8 users (2 per row) |
|---|---|---|---|
| Job | 715 | 716 | 718 |
| TTFT | 277.3 s | 476.5-477.4 s | slot 0 472.9-473.8 s, slot 1 946.3-947.2 s |
| Pass key found (depths 10 / 35 / 65 / 90%) | 1 / 1 | 4 / 4 | 8 / 8, every depth twice |
| Decode from the 1M cache | correct | correct on every row | correct in both slots of every row |
| Decode step, untraced, logits read (not TPOT) | 163 ms | 653 ms | 654 ms |

- One row alone matches the earlier 1M TTFT (272.5 s, second run of random tokens). With 4 rows each takes ~476 s: one
  host thread dispatches every row's chunks, and a round over 4 rows takes longer than one chunk on the device. A
  traced prefill chunk or one host thread per row should bring the 4 rows back to ~277 s.
- 4 users per row at 1M does not fit with the K / V on both chips, so not under vLLM's layout. It fits when the pair
  splits the users, but a later chunk then runs all 16 heads on the owner and discards the other chip's result: from the
  last-chunk times, ~416 s per user, ~28 minutes for 4 users on one row.
- #61419's step 1 is stale: the standalone SWA layers keep 9 blocks per user (`kv.SwaBlockTable`), and #61745 sizes
  vLLM for 2 x 1M per lane (`MIMO_KV_GB_PER_CHIP=6.6`), the 8-user case above.

Next: the 4-row prefill dispatch; prefilling a row's two users together; a 1M prompt through the vLLM server; traced
decode TPOT at 1M from a real 1M cache.
