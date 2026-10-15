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
| `qkv_proj` (global) | 4096 -> 64x192 + 4x192 + 4x128 | same | KV head `c // 2` on chip `c` (stored twice across the row) |
| `o_proj` | 8192 -> 4096 | (8x192) x 4096 | rows for the V padding are zero; row-parallel + all-reduce |
| Attention sink (SWA) | 64 | 8 | pre-divided by the softmax scale |
| Dense MLP (layer 0) | 4096 -> 16384 -> 4096 | 4096 x 2048, 2048 x 4096 | column / row parallel + all-reduce |
| Router | 4096 x 256 bf16, bias fp32 | replicated | `moe_grouped_topk` single-group path |
| Routed experts | 256 x (3 x 4096 x 2048) | 256 x (3 x 4096 x 256) | intermediate dim TP-sharded; `sparse_matmul` on the union of selected experts per 32-token group; all-reduce |
| LM head | 4096 x 152576 | 4096 x 19072 | vocab-parallel, host concatenates |
| KV cache (per layer) | | `[blocks, 1, 64, 192]` K and V | paged, BFP8 |

RoPE: HF rotates the first 64 of 192 dims with `rotate_half`. The Q/K rows of each head are
permuted to interleaved order so `rotary_embedding_llama` applies; the cos/sin tables are 192 wide
with cos = 1, sin = 0 on the 128 no-RoPE dims.

## Memory per chip (48 layers)

| Item | Per chip |
|---|---|
| Routed experts, BFP4 (37.8B params) | 21.3 GB |
| Routed experts, BFP8 (for comparison) | 40.2 GB: does not fit |
| Attention weights, BFP8 (0.68B params) | 0.72 GB |
| Embedding (bf16, replicated) | 1.25 GB |
| Router, dense MLP, LM head, norms | ~0.2 GB |
| KV cache, BFP8, 4 users x 1024 tokens (test config) | ~0.08 GB |
| Total (test config) | ~23.6 GB of 32 GB |

At 1M context the global layers' KV is the open item: replicated KV heads on chip pairs and the
V padding do not fit B=4 @ 1M (see #60809, decision 1 and 2).

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

Notes:

- Routing is fp32 end to end. `moe_grouped_topk` resolves MiMo's near-ties at bf16 precision
  (80% top-8 match on exact fp32 logits); the fp32 sigmoid/topk/gather path matches 100% on exact
  logits and 98% on device logits, the rest coming from the router matmul (~2^-12 relative).
- 4 rows cost 1.4x one row per traced step, not 1x: the host part of each step (input copies and
  reading 32 x 152576 logits from 8 shards) is serial over the rows. Device sampling removes most of it.
- Performance is untuned: every op is DRAM-interleaved with default program configs.
- Weight cache: the first replica converts the checkpoint in ~24 min (half host MXFP4 dequant,
  half TTNN BFP4 encoding); every later replica load is ~14 s from the cache.
