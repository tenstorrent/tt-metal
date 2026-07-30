# Non-MoE roofline — which bandwidth is unsaturated, and is it recoverable (#47465)

Status: current for the bandwidth denominator and the CCL / terminal-reduction refutations; its
framing of the non-MoE share as "~66%" and its "levers exhausted" conclusion are superseded.
Owns: the practical per-chip DRAM bandwidth denominator, and the terminal argmax/entropy
reduction-op limit including the 18-bit-index bf16 wall.
See also: [refuted list](../../REFUTED.md), [optimize_perf hub](../README.md).

All numbers are **no-trace eager per-op** on QB2 (P150x4), `(1,4)` mesh, TP=4, real shapes. Probes:
`~/dg-agent-runs/{nonmoe_roofline,ccl_microbench,terminal_roofline,argmax_2stage_concept,fast_terminal_test}.py`.
Env: see [plan](../../../plan.md).

## The bandwidth denominator (use this for every efficiency claim)

A large pure-bandwidth `add` (4096², 3x traffic) sustains **~2.0 TB/s aggregate / ~235 GB/s per chip
achievable** against a ~256 GB/s nominal figure. **~235 GB/s/chip is the practical single-op roofline.**

The MoE share this document was written to complement has moved twice: the current denoise MoE is
`tt/concat_moe.py`, priced by `DG_SKIP` at **75.5 ms of a 238.1 ms step (31.7%)**, and **52.1% of the
step is in no seam at all** — [winter borrow](../winter_borrow_20260727.md). The weight-byte floor that
sets the denoise cost is owned by the [work log](../work_log.md).

## Attention, norms and RoPE — latency-bound, not bandwidth-bound

| op (per chip, S=256) | ms | % of ~235 GB/s |
|---|---:|---:|
| qkv_proj `[256,2816]@[2816,2048]` | 0.091 | 28% |
| o_proj `[256,1024]@[1024,2816]` | 0.044 | 29% |
| SDPA (K=512) | 0.055 | 9% |
| rms_norm `[256,2816]` | 0.035 | 18% |
| per-head norm, RoPE mul, residual, concat, L1 copies | 0.01–0.04 each | 7–26% |

Every non-MoE compute op is **below 30% of DRAM bandwidth because each is too small (<= 0.09 ms) to
saturate it** — latency/launch-bound, not bandwidth-bound. The sum of all attention device ops is
**~0.5 ms/layer**. Being launch-bound, their only lever is *fewer/larger* ops (fusion), and traced
dispatch already amortizes the launch cost.

## TP all-reduce — fixed latency, no knob

**REFUTED:** `ttnn.all_reduce([1,1,256,2816], Topology.Linear)` is **0.67 ms FLAT** across Topology
(Linear/Ring), `num_links` (1 or 2; >=3 unavailable) and the decomposed `reduce_scatter+all_gather` the
code's own TODO suggests — all identical. 1.44 MB / 0.67 ms is ~2 GB/s, i.e. **latency-bound**; the only
lever is *fewer* all-reduces, a structural TP change.

At the 2 all-reduces/layer x 30 layers this was written against, that is ~40 ms/step (~15%). **That
per-layer count predates the concat MoE, which issues one `ccl_allreduce` per layer at
`concat_moe.py:376` — re-derive the count before quoting ~40 ms/step as current.**

## Terminal argmax + entropy over the 262144 vocab — reduction-op-limited

| op on `[1,1,256,262144]` bf16 (134 MB) | ms | GB/s |
|---|---:|---:|
| 1-pass max reduction | 3.46 | 39 |
| **argmax (ROW_MAJOR, current best)** | **13.56** | **10** |
| `to_layout` → ROW_MAJOR | 0.73 | 366 (fine) |
| argmax on TILE (single-core) | 1239 | — |
| token_entropy | 12.68 | — |

A vocab reduction sustains only **39 GB/s, ~6x below bandwidth**, because there are only S=256 rows to
parallelize over a 262144-long reduction; `argmax` is a further ~4x slower than `max` as an op,
independent of the shape.

**Two rewrites tried, both refuted:**
1. **2-stage reshape** (`[256,512,512]`, reduce the small last dim → 131072 rows at 0.365 ms): the
   reshape that splits the TILED last dim is **not zero-copy — it IS the expensive tilized relayout**
   (the 6-D permute), so `fast_token_entropy` measured **40 ms, 0.32x, i.e. SLOWER**, even though the
   values were right (fast_max delta 0, entropy PCC 0.99970).
2. **argmax via max+iota+min** using only fast reductions: the vocab index is **18-bit and bf16 cannot
   hold indices above 256**, so carrying it needs int32/fp32 reductions at 2x the bytes, which erases
   the win.

## Bottom line

Every unsaturated non-MoE op is limited by something not cheaply reconfigurable in Python:
op-primitive parallelism for vocab reductions, fixed CCL latency, tile-reshape cost, or the bf16 index
width. The remaining real levers named here are **custom kernels** — a fused high-parallelism
vocab-reduction / argmax+entropy terminal kernel targeting ~27 ms/step at under 40% bandwidth, or a TP
restructure that cuts the all-reduce count.

**Do not carry this file's "the cheap in-repo levers are exhausted" conclusion forward:** it was
refuted twice since by in-repo levers of exactly this class — the full-canvas RMSNorm
([l1 residency](../l1_residency.md)) and the concat-experts MoE
([winter borrow](../winter_borrow_20260727.md)).
