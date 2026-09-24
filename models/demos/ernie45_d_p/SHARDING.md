# ERNIE-4.5-21B-A3B prefill: sharding on 4x Blackhole p150b (mesh 1x4)

The machine-checked numbers come from `bringup/plan.py`, whose gate is P2.2. This page gives the reasoning.

| Component | Placement | Per chip | Collective |
|---|---|---|---|
| Residual stream `[S, 2560]` | replicated | full | none |
| RMSNorm (input, post-attn, final) | replicated | full | none |
| Q proj `[2560 -> 20x128]` | column-parallel by head | 5 Q heads | none |
| K, V proj `[2560 -> 4x128]` | column-parallel by head | 1 KV head | none |
| RoPE (interleaved, theta 5e5) | local | 5 Q + 1 K | none |
| SDPA (causal GQA 5:1) | local | 5 Q heads vs 1 KV head | none |
| KV cache | local, 1 KV head | `[users*layers, 1, seq, 128]` | none |
| O proj `[20x128 -> 2560]` | row-parallel | 640 input rows | **all_reduce** |
| Dense MLP (layer 0, 12288) | gate/up column, down row | 3072 | **all_reduce** |
| Router `[2560 -> 64]` fp32 + bias | replicated, same result on every chip | full | none |
| Routed experts (64 x SwiGLU 1536) | expert-parallel | experts 16c .. 16c+15 | summed with shared |
| Shared experts (2 x 1536 = 3072) | gate/up column, down row | 768 | **all_reduce** (one, shared with routed) |
| Embedding `[103424, 2560]` | replicated | full (0.49 GB) | none |
| LM head (tied) | vocab-sharded | 25856 rows | gather on host (prefill needs only the last-token logits) |

Why this split:
- **TP=4 for attention equals the KV head count (4).** Each chip owns exactly one KV head and its 5 GQA
  query heads, so attention needs no communication. This is also the prefill-server KV contract: one KV
  head per TP column (see `models/demos/common/prefill/docs/ADDING_A_PREFILL_MODEL.md` and the `gpt_oss_d_p` GQA template).
- **EP=4 for routed experts needs no all-to-all.** The residual is replicated, so every chip computes the
  same routing locally. Each chip then runs only its 16 experts on the tokens routed to them. The routed partial
  and the chip's quarter of the shared expert are summed locally, and one all_reduce produces the MoE output.
  The cost is replicated router compute (64x2560 per token), which is negligible.
- **Two all_reduces per layer** over `[S, 2560]` bf16: 26 MB per 5k chunk each.
  Future optimization: reduce_scatter + distributed RMSNorm + all_gather (see `tt_transformers/tt/ccl.py`).
- **Memory:** about 11.4 GB per chip of 32 GB at 56320 tokens, 1 user, bf16 KV. Routed experts dominate at 9.5 GB.
  Plenty of headroom for activations and more users.

Chunking: 5120-token chunks for the 55k target (the prefill-server default `PREFILL_CHUNK_SIZE`), and 2048/8192 for
the dev ladder. The KV write offset is `chunk_start`, which is 32-aligned and tile-aligned.
