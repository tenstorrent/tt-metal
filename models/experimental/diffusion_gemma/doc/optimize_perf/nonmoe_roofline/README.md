# Non-MoE roofline — where the remaining denoise-step time is (#47465)

After confirming the expert matmul is already ~92% of the weight-traffic roofline in production
(`DG_SPARSE_MOE_TUNED`, now default-on), tuned MoE = ~2.90 ms/layer × 30 = **~87 ms = ~34% of the
257.93 ms traced step**. This maps the **other ~66%** and asks the user's question directly: *which
bandwidth is NOT saturated, and is it cheaply recoverable?* All numbers are no-trace eager per-op on
QB2 (P150×4), `(1,4)` mesh, TP=4, real shapes. Probes: `~/dg-agent-runs/{nonmoe_roofline,
ccl_microbench,terminal_roofline,argmax_2stage_concept,fast_terminal_test}.py`.

## Empirical DRAM peak
A large pure-BW `add` (4096², 3× traffic) sustains **~2.0 TB/s aggregate / ~235 GB/s per-chip
achievable** — the same ~235 GB/s the tuned MoE matmul reaches (92% of the 256 GB/s figure OPT-004
was tuned against). Use ~235 GB/s/chip as the practical single-op roofline.

## Attention + norms + RoPE — tiny, latency-bound (NOT bandwidth-bound)
| op (per-chip, S=256) | ms | % of ~235 GB/s |
|---|---:|---:|
| qkv_proj `[256,2816]@[2816,2048]` | 0.091 | 28% |
| o_proj `[256,1024]@[1024,2816]` | 0.044 | 29% |
| SDPA (K=512) | 0.055 | 9% |
| rms_norm `[256,2816]` | 0.035 | 18% |
| per-head norm, RoPE mul, residual, concat, L1 copies | 0.01–0.04 each | 7–26% |

**Every non-MoE compute op is <30% of DRAM BW — because each is too small (≤0.09 ms) to saturate
it (latency/launch-bound, not BW-bound).** Sum of all attention device ops ≈ **0.5 ms/layer**. These
are already cheap; tracing hides their dispatch. No bandwidth headroom to chase — the lever would be
*fewer/larger* ops (fusion), and traced dispatch already amortizes the launch cost.

## TP all-reduce (CCL) — fixed latency, ~40 ms/step
`ttnn.all_reduce([1,1,256,2816], Topology.Linear)` = **0.67 ms, flat** across Topology (Linear/Ring),
`num_links` (1/2; ≥3 unavailable), and the decomposed `reduce_scatter+all_gather` (the code's own
TODO) — all identical. 1.44 MB / 0.67 ms = ~2 GB/s ⇒ **latency-bound, not bandwidth-bound**; no knob
moves it. 2 all-reduces/layer × 30 = **~40 ms/step (~15%)**. Only lever = *fewer* all-reduces (a
structural TP change), not a faster one.

## Terminal argmax + entropy over the 262144 vocab — reduction-op-limited
| op on `[1,1,256,262144]` bf16 (134 MB) | ms | GB/s |
|---|---:|---:|
| 1-pass max reduction | 3.46 | 39 |
| **argmax (ROW_MAJOR, current best)** | **13.56** | **10** |
| to_layout→ROW_MAJOR | 0.73 | 366 (fine) |
| argmax on TILE (single-core) | 1239 | — |
| token_entropy (max + 2 sums + exp/…) | 12.68 | — |

A vocab reduction sustains only **39 GB/s (~6× below BW)** — only S=256 rows to parallelize over a
262144-long reduction. `argmax` is a further ~4× slower than `max` (the op itself, not the shape).

**Two fixes tried, both fail in-repo:**
1. **2-stage reshape** (`[256,512,512]`, reduce the small last dim → 131072 rows → 0.365 ms): the
   reshape that splits the *tiled* last dim is **not** zero-copy — it *is* the expensive tilized
   relayout (the 6-D permute). Reshape cost > reduction saved: fast_token_entropy measured **40 ms
   (0.32×, slower)**. (Value is correct: fast_max Δ=0, entropy PCC 0.99970.)
2. **argmax via max+iota+min** (all fast reductions): the vocab index is 18-bit and **bf16 cannot
   hold indices >256**; carrying it needs int32/fp32 reductions (2× bytes) which erases the win.

## Bottom line
The only bandwidth-saturated op is the MoE matmul (no headroom). Every *un*saturated non-MoE op is
limited by something **not cheaply reconfigurable in Python**: op-primitive parallelism (vocab
reductions), fixed CCL latency, tile-reshape cost, or the bf16 index width. The cheap in-repo levers
for the non-MoE 66% are **exhausted**, mirroring the MoE. The remaining real levers are **custom
kernels** — a fused high-parallelism vocab-reduction / argmax+entropy terminal kernel (targets ~27
ms/step at <40% BW), or a TP restructure to cut all-reduce count — both larger efforts.
