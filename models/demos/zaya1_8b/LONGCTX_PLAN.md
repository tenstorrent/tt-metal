# ZAYA1-8B long-context (128K) plan + findings

Target: context 128K (in-spec; `max_position_embeddings=131072`, rope_theta 5e6, no
sliding window), batch 8-32. 256K is 2x the trained limit (RoPE extrapolation, quality
risk), deferred.

## Why long context changes the regime
Short-context decode is **op/dispatch bound** (batching helps throughput; multi-chip does
not cut latency). Long context is **KV-capacity + KV-bandwidth bound**.

KV cache = B x S x 40 KB (40 CCA layers, 2 kv-heads x 128, K+V, bf16):
| B / S | 128K | 256K |
|---|---|---|
| 1 | 5.4 GB | 10.7 GB |
| 8 | **43 GB** | 86 GB |
| 32 | 171 GB | 343 GB |

One P150a = 32 GB. **B=8/128K (43 GB) doesn't fit one chip** -> needs **KV context
parallelism** (shard the KV sequence dim across chips), which is also the latency lever
(shards the per-step KV read N-ways). Different from the EP/TP-for-MoE work (shards
weights). Feasibility on 8xP150a (256 GB): B=8/128K -> 4 chips; B=8/256K -> 8 chips;
B=32/128K -> 8 chips + weight-EP; B=32/256K -> infeasible (343 GB > 256 GB).

## Plan
- **L1 flash/chunked prefill** — replace the [1,1,S,S] causal mask + [.,.,S,S] scores
  (68 GB at 128K) with FlashAttention-2 (`ttnn.transformer.scaled_dot_product_attention`,
  GQA, is_causal). DONE (opt-in `ZAYA_FLASH_PREFILL=1`): runs at S=4K/16K (mask gone).
- **L2 flash-decode** — `scaled_dot_product_attention_decode` / `paged_..._decode` over
  long KV (chunked, KV-bandwidth bound). Not started.
- **L3 KV context-parallel decode** — `ring_distributed_scaled_dot_product_attention`
  shards KV across the mesh (cross-chip online softmax). Target B=8/128K on 4 chips. Not started.
- **L4 distributed/chunked prefill for B>1** — B=8/128K KV (43 GB) can't prefill on one
  chip; chunked prefill scattering KV across the CP layout, or ring-attention prefill. Not started.

## Key finding: token-exactness vs flash (cross-cutting blocker)
ttnn flash ops use bf16 online-softmax; zaya's validated decode uses **manual fp32 softmax**
(adopted specifically to stop top-1 MoE routing drift — divergence first appears at MoE
layer 17). Flash prefill is therefore **NOT token-exact** vs the fp32 golden: at the golden
prompt it generates a different stream ([1924,...] / [25567,...] vs golden
[9079, 236761, 107, 2717, 108, 2717]); a fp32 `compute_kernel_config` does not recover it.
This is inherent precision drift, not a tuning bug, and applies to **all** flash paths
(L1 prefill, L2 decode, L3 ring). Consequences:
- Short context keeps the manual fp32 path (`ZAYA_FLASH_PREFILL=0`, default) -> token-exact.
- Long context **requires** flash (S^2 memory infeasible) -> best-effort, not bit-exact.
  No 128K golden exists anyway; validate via **logits PCC / perplexity / qualitative
  coherence**, not token-exactness.
- Open: whether a higher-precision flash kernel (fp32 softmax stats) can close the gap for
  routing-sensitive MoE.

## Open blockers
- **S=65536 prefill FATAL** (`bank_manager.cpp:462` allocator) even with flash -> needs an
  `SDPAProgramConfig` (Q/K chunk tiles) + memory_config to scale past ~16K-32K.
- L2-L4 are a substantial multi-step build (KV context parallelism + distributed prefill).

## ttnn primitives identified
`scaled_dot_product_attention` (flash prefill), `scaled_dot_product_attention_decode` /
`paged_scaled_dot_product_attention_decode` (flash-decode + paged KV),
`ring_distributed_scaled_dot_product_attention` (context-parallel / ring attention),
`chunked_scaled_dot_product_attention`, `SDPAProgramConfig`.
