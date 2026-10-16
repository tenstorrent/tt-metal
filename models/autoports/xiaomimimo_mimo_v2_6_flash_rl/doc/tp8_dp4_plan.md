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
| `qkv_proj` (global), prefill | 4096 -> 64x192 + 4x192 + 4x128 | same as SWA | 8 Q heads + KV head `c // 2` on chip `c` |
| `qkv_proj` (global), decode | | 4096 x (16x192 + 192 + 192) | all 16 Q heads of KV group `c // 2`; the chip pair splits the users (`tt/kv.py`) |
| `o_proj` | 8192 -> 4096 | (8x192) x 4096; global decode (16x128) x 4096 | padded rows are zero; row-parallel + all-reduce |
| Attention sink (SWA) | 64 | 8 | pre-divided by the softmax scale |
| Dense MLP (layer 0) | 4096 -> 16384 -> 4096 | 4096 x 2048, 2048 x 4096 | column / row parallel + all-reduce |
| Router | 4096 x 256 bf16, bias fp32 | replicated | fp32 sigmoid / topk / gather |
| Routed experts | 256 x (3 x 4096 x 2048) | 256 x (3 x 4096 x 256) | intermediate dim TP-sharded; `sparse_matmul` on the union of selected experts per 32-token group; all-reduce |
| LM head | 4096 x 152576 | 4096 x 19072 | vocab-parallel, host concatenates |
| KV cache, SWA layer | | K, V `[blocks, 1, 64, 192]` | 512-token ring per user (`cache_position_modulo`), BFP8 |
| KV cache, global layer | | K `[blocks, 1, 64, 192]`, V `[blocks, 1, 64, 128]` | full context, only the users this chip owns, BFP8; decode via `paged_flash_multi_latent_attention_decode` |

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
| KV cache, SWA rings (39 layers x 4 users x 512 x 384) | 0.03 GB |
| **Measured** (`test_capacity_1m_tp8_dp4.py`, every replica) | weights 23.69 GB; with 4 x 1M KV 30.14 GB of 34.03 GB (3.89 GB free) |

How B=4 @ 1M fits, with existing ops only (no new kernel):

1. SWA layers only attend to 128 tokens, so their cache is a 512-token ring per user
   (`cache_position_modulo` on fill, update and decode SDPA). A full-length SWA cache would be
   66.8 GB per chip.
2. Global layers have 4 KV heads on 8 chips. Rather than store each head on both chips of a pair,
   the pair splits the users: chip `c` keeps the KV of slots with `slot % 2 == c % 2` and decodes all
   16 Q heads of its group for them (decode reads the minimum bytes; no cross-chip softmax merge).
   Prefill still uses 8 heads per chip on fresh K/V, so long-prompt prefill is balanced too, until
   chunked prefill needs the cached KV, which only the owner holds.
3. The global V cache is 128 wide, read with `paged_flash_multi_latent_attention_decode`
   (`head_dim_v`), so V is not padded to 192 in the cache.

Splitting the sequence across the pair instead would need decode SDPA to return the softmax
statistics (LSE) for a cross-chip merge, which it does not today.

## Validation (tests/, measured 2026-10-15 on one BH Galaxy, branch `mimo-v2.6/tp8-dp4`)

All tests open the 4x8 mesh (`FABRIC_1D_RING`), split it into four 1x8 rows and give every row its
own users. References are fp32 torch (`reference/model.py`, bit-identical to HF on real weights).

| Test | What | Result |
|---|---|---|
| `test_decoder_tp8_dp4.py` layer 0 (global, dense) | 4 rows x 4 users, prefill 128-384 tokens + 3 decode steps, BFP8 weights and KV | output PCC 0.99999; block-delta PCC prefill 0.9977, decode 0.9963 |
| layer 1 (SWA + sink, MoE) | same | output 0.9995 / 0.9979; delta 0.9916 / 0.9946 |
| layer 5 (global, MoE) | same | output 0.9966 / 0.9992; delta 0.9929 / 0.9998 |
| `test_moe_tp8_dp4.py` layers 1, 5 | real golden MoE inputs over 4 rows | top-8 set match 98%; vs reference with device routing PCC 0.9999+; end to end 0.9997-0.9999 |
| `test_model_tp8_dp4.py` | 48 layers, BFP4 experts, 4 rows x 3 users | 40 logits rows vs fp32 reference: top-1 97.5%, PCC mean 0.9924 (min 0.9686). Shared prompt identical on all 4 rows (PCC 1.0, same greedy text) |
| `test_model_traced_tp8_dp4.py` | traced decode, 4 rows x 2 users | traced == eager tokens (logits PCC 1.000000). Step: 1 row 164 ms, 4 rows 228 ms -> 35.1 tok/s aggregate vs 12.2 tok/s on one row (eager: 690 ms for 4 rows) |
| `test_capacity_1m_tp8_dp4.py` | 4 rows x 4 users, max_seq_len 1M, traced decode | fits (above). 16 users: 261 ms/step at 4K, 278 ms at 128K, 449 ms at ~1M |

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
- Global-layer KV layout for B=4 @ 1M (replicated KV heads, V padding), device sampling,
  speculative decoding, performance tuning: #60809.
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
| Routing after the sigmoid in one program (`tt/moe_route.py`: exact fp32 top-8, normalize, mask; replaces top-k, gather and ~20 glue ops); CCL on 2 links (all-reduce 37.7 -> 29.7 us) | 26.8 ms, 26.0 % | 28.9 ms, 24.1 % | 31.8 ms |
| tt-kernel-agent `mimo_moe_stream` round 2: bank-local streaming gate|up with the routing selection inside the program (MoE layer 283 us) | 23.7 ms, 29.5 % | 25.8 ms, 27.1 % | 28.7 ms |
| tt-kernel-agent final (4 rounds): bank-local streaming down with the pair sum in dest (MoE layer 204 us) | 20.1 ms, 34.7 % | 24.1 ms, 29.0 % | 24.6 ms |
| Bank-reader cores from the 4x8 parent: row 2's own answer (x = 6 for banks 4-7) slowed six of its chips | 20.1 ms, 34.7 % | 21.95 ms, 31.8 % | 24.8 ms |
| tt-kernel-agent `mimo_attn_decode` round 2: L1 hand-offs around the projections and the all-reduce; SWA o projection without the V padding rows (attention block 174.7 -> 156.9 us) | 18.5 ms, 37.7 % | 21.1 ms, 33.1 % | 22.3 ms |
| tt-kernel-agent `mimo_moe_stream2`: compact 4-row all-gather + fixed-order fp32 reduce, earlier streaming start (MoE layer 204 -> 181 us) | 18.07 ms, 38.6 % | 20.15 ms, 34.6 % | 22.4 ms |
| tt-kernel-agent `mimo_attn_decode` final: wide-K-block 1D matmuls for qkv / o on unpadded weights (attention block -> 147.4 us) | 17.68 ms, 39.5 % | 19.75 ms, 35.3 % | 22.0 ms |

Accuracy after the last row: teacher-forced top-1 vs the fp32 reference 0.950, logits PCC min 0.9642 / mean
0.9927 (`test_model_tp8_dp4.py`; before this table: 0.950, 0.9693 / 0.9926). The GDDR6 of this Galaxy runs at 14 Gbps on every chip (tt-smi), so the
attainable peak is ~448 GB/s; the table keeps the 512 GB/s spec peak as the denominator.

With B=4 x 1M allocated (`test_capacity_1m_tp8_dp4.py`, 16 users, 4 per row), after the first table's rows:
49.1 ms/step at 4K positions (326 tok/s), 74.5 ms at 128K (215 tok/s), 236.7 ms at ~1M (68 tok/s).

Where one MoE decode layer's ~650 us go (device, per chip): expert `sparse_matmul` gate|up 282 us
(16 cores) and down 118 us, fp32 top-k 44 us and gather 33 us, row all-reduce 34 us, the rest in
small ops. The expert matmuls read ~57 MB of BFP4 weights per call, so they run at about half of the
bandwidth a DRAM-sharded matmul reaches. `tests/perf/kagent_moe_decode_{bench,check}.py` are the
tt-kernel-agent perf and correctness commands for this layer (baseline 653 us, spread 0.04 %).
