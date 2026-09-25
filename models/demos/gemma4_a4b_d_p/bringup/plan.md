# Gemma-4 26B-A4B prefill: sharding on 4x Blackhole p150b (mesh 1x4)

The machine-checked numbers come from `plan.yaml` (gate PL.1, `python -m models.demos.common.bringup.plan.check_plan`,
output in `results/plan_memory.json`). This page gives the reasoning. Shape follows `models/demos/ernie45_d_p/SHARDING.md`.

Target: 56320 tokens in 5120-token chunks, 1 user, bf16. Per-chip DRAM 32 GB, budget 27.2 GiB (15% headroom).

## Scheme

- Residual stream `[S, 2816]` is **replicated**. Every norm, residual add, and the router run the same way on every chip.
- **Attention: TP=4 by head.** Q/K/V are column-parallel, O is row-parallel, then one `all_reduce`.
- **Dense MLP (2112, GeGLU): TP=4.** gate/up are column-parallel, down is row-parallel, then one `all_reduce`.
- **Routed experts: EP=4.** Chip c holds experts 32c..32c+31. Every chip dispatches its tokens locally to its own experts
  (same scheme as ERNIE `moe_unified.py`, dispatch group of 1 chip), then one `all_reduce`.
- **Embedding** is replicated. The **LM head** (tied) is a second copy, transposed and vocab-sharded.

## Sliding layer (25 layers: 0-4, 6-10, 12-16, 18-22, 24-28)

16 Q heads x 256, 8 KV heads x 256, window 1024, RoPE theta 1e4.

| Component | Placement | Each chip holds | Collective after | Per chip |
|---|---|---|---|---|
| attn_norm (input_layernorm) | replicated | full [2816] | none | 6 KB |
| q_proj [2816 -> 16x256] | column-parallel by head | 4 Q heads (1024 rows) | none | 5.8 MB |
| k_proj, v_proj [2816 -> 8x256] | column-parallel by head | 2 KV heads (512 rows each) | none | 5.8 MB |
| q_norm, k_norm, V unscaled rms | replicated weights, local | per head | none | 1 KB |
| RoPE (rotate-half, theta 1e4) | local | 4 Q + 2 K heads | none | 0 |
| KV cache (full length) | local, 2 KV heads | K and V [2, 56320, 256] | none | 115 MB |
| SDPA, causal, window 1024, scale 1.0 | local, GQA 2:1 | 4 Q vs 2 KV heads, prev 1024-token K/V tail + chunk | none | 0 |
| o_proj [16x256 -> 2816] | row-parallel | 1024 input columns | **all_reduce** [S, 2816] | 5.8 MB |
| post_attn_norm, attn_residual | replicated | full | none | 6 KB |
| ffn_norm (pre_feedforward_layernorm) | replicated | full | none | 6 KB |
| dense MLP 2112 (gate, up, down) | gate/up column, down row | 528 (padded to 544) | **all_reduce** [S, 2816] | 8.9 MB |
| post_mlp_norm | replicated | full | none | 6 KB |
| router [2816 -> 128], fp32 | replicated, same result on every chip | full | none | 1.5 MB |
| moe_norm (pre_feedforward_layernorm_2) | replicated | full | none | 6 KB |
| routed experts 128 x GeGLU 704, bf16 | expert-parallel | experts 32c..32c+31 | **all_reduce** [S, 2816] | 381 MB |
| post_moe_norm, ffn_combine, post_ffn_norm, ffn_residual (x layer_scalar) | replicated | full | none | 18 KB |
| **Layer total** | | | **3 all_reduces** | **about 523 MB** |

## Global layer (5 layers: 5, 11, 17, 23, 29)

16 Q heads x 512, 2 KV heads x 512, no v_proj (V from k_proj), partial proportional RoPE theta 1e6.

| Component | Placement | Each chip holds | Collective after | Per chip |
|---|---|---|---|---|
| attn_norm | replicated | full | none | 6 KB |
| q_proj [2816 -> 16x512] | column-parallel by head | 4 Q heads (2048 rows) | none | 11.5 MB |
| k_proj [2816 -> 2x512] | by KV head, each head on 2 chips | KV head c//2 (512 rows) | none | 2.9 MB (gate counts 5.8 MB, replicated) |
| K = RoPE(k_norm(k_raw)), V = rms(k_raw) | local | 1 K head, 1 V head | none | 0 |
| KV cache (full length) | local, 1 KV head | K and V [1, 56320, 512] | none | 115 MB |
| chunked causal SDPA, scale 1.0 | local, GQA 4:1 on chip | 4 Q heads vs 1 KV head | none | 0 |
| o_proj [16x512 -> 2816] | row-parallel | 2048 input columns | **all_reduce** [S, 2816] | 11.5 MB |
| norms, residuals, dense MLP, router, experts | as in sliding | as in sliding | **all_reduce** after MLP and after experts | 391 MB |
| **Layer total** | | | **3 all_reduces** | **about 532 MB** |

## Model level

| Component | Placement | Each chip holds | Collective | Per chip |
|---|---|---|---|---|
| embed_tokens [262144, 2816] bf16, x sqrt(2816) | replicated | full table | none | 1.38 GiB |
| final norm | replicated | full | none | 6 KB |
| LM head (tied), last tokens only, softcap 30 | vocab-sharded copy | 65536 rows | gather logits (host or all_gather) | 0.37 GiB |
| vision tower, embed_vision | skipped | none | none | 0 |

## Per-chip total (from the gate, GiB)

| Group | Per chip |
|---|---|
| routed experts (bf16, 32 per chip x 30 layers) | 10.63 |
| activations (planner estimate, one 5120 chunk) | 3.00 |
| KV state, sliding (25 layers, 2 heads x 256, full length) | 2.69 |
| contract KV copy (bfp8, as ERNIE) | 1.84 |
| embedding (replicated) | 1.38 |
| KV state, global (5 layers, 1 head x 512, full length) | 0.54 |
| attention weights | 0.54 |
| LM head vocab-sharded copy | 0.37 |
| RoPE tables, dispatch tables, misc | 0.25 |
| dense MLP | 0.25 |
| router (fp32) | 0.04 |
| **Total** | **21.52 of 27.20 budget** |

Per layer: 3 `all_reduce` of [5120, 2816] bf16 (28.8 MB each per chunk). There is no all-to-all, because the residual is
replicated and every chip routes on its own. Per chunk the model also runs one embedding lookup and one LM-head matmul on the last tokens.

Activation estimate (3.0 GiB): the MoE dispatch and combine buffers in the worst case (all 8 top-k experts on one chip)
are 8 x 5120 x 2816 bf16, 0.23 GB each. Expert intermediates are at most 8 x 5120 x 1408 bf16, 0.12 GB. Add a few
residual-sized tensors (29 MB each), Q/K/V/SDPA buffers and CCL scratch, then double it for fragmentation.

## Departures from the reference plans (ERNIE SHARDING.md, Kimi 4x4), one reason each

- **Three all_reduces per layer, not two.** Gemma-4 applies a separate RMSNorm to the MLP output and to the MoE output
  before adding them, so their partial sums cannot share a single all_reduce the way ERNIE's routed and shared outputs do.
- **Global KV heads duplicated on 2 chips.** There are only 2 KV heads for 4 chips. Each chip holds KV head c//2 and its 4 Q heads,
  so attention still needs no communication (the cost is 2x the global k_proj weight and KV).
- **Experts use `unified_routed_expert_moe` with a new GeluTanh activation.** The op had Silu, SwiGluOai, SituGlu and
  ClampedSiluGlu only. At plan approval the owner decided to add GeluTanh to the op (enum, binding, compute kernel) rather
  than run per-expert matmuls, so Gemma gets the fused path ERNIE uses.
- **Dense-MLP intermediate padded 528 -> 544 per chip.** 2112/4 is not a multiple of 32. The padding is zeros, as in `gemma4/tt/shared_mlp.py`.
- **Sliding KV kept at full length, not 1024.** The prefill contract hands the whole cache to decode. The window only limits
  what SDPA reads: the previous 1024 positions are read from the cache and concatenated before the windowed SDPA.
- **Experts in bf16, not bfp8.** They fit with 5.7 GiB to spare. Accuracy comes first because the fused path already packs
  activations to bfp8. Switching to bfp8 weights would free 4.6 GiB per chip if a later step needs it.
- **Router in fp32.** Softmax-then-top-8 over 128 experts has near ties. The reference found that routing flips drive the
  chunked-vs-one-shot divergence (R.3). fp32 costs 43 MB per chip.
