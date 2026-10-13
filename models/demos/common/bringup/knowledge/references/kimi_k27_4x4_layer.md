# Reference plan: one Kimi K2.7 layer on a 4x4 mesh (SP=4 x TP=4)

The text version of `kimi_k27_4x4_layer.html` (the interactive page, open it in a browser; it was the owner's
artifact https://claude.ai/artifact/4MC3c1hvkErCYxByJwdPZ9). Numbers come from the tt-moe-nappkin cost model's Kimi
K2.7 preset: `layer_costs(isl=5120, kv_cache=46080)` on a 4x4 mesh, attention at HiFi2 (213 TFLOPS per chip), routed
experts at LoFi (426 TFLOPS), DRAM at 80% of 512 GB/s, links at 80% of 2 x 200 Gb/s per direction. They are model
estimates, not measurements. Bring-ups that adopted this layout: xing40_a4b_d_p (4x2: the same scheme with 2 columns),
xing40_a4b_d_p_8x4.

## The layout
- Model: Kimi K2.7 (MLA attention, hidden 7168, 64 heads; MoE with 384 routed experts, top-8, 1 shared expert).
- Chunk: 5120 tokens, the last chunk of a 51200-token prompt (46080 tokens already in the KV cache).
- Rows (mesh axis 0, SP = 4) split the chunk's tokens: row r holds tokens 1280 r .. 1280 r + 1279.
- Columns (mesh axis 1, TP = 4) split the weights; the resting activation layout is [1280, 1792] per chip (a quarter of
  the hidden width).
- Attention heads are split over columns (16 of 64 per chip). Experts are expert-parallel over all 16 chips (24 whole
  experts per chip, chip k = 4 r + c holds experts 24 k .. 24 k + 23); a token only ever travels within its column.

## Steps (per chip, 15.86 ms total; attention 11.29, expert block 4.46, norms and adds 0.11)
| # | Step | Kind | Traffic | One chip reads -> writes | ms |
|---|---|---|---|---|---|
| 1 | Compress Q (wq_a), weight split on its input side | maths | none | [1280, 1792] x [1792, 1536] bfp8 -> [1280, 1536] partial | 0.033 |
| 2 | Sum and share Q | comm | row: reduce-scatter + all-gather | [1280, 1536] -> [1280, 1536] complete | 0.074 |
| 3 | Expand Q per head (wq_b), 16 heads per chip | maths | none | [1280, 1536] x [1536, 3072] -> [1280, 3072] | 0.057 |
| 4 | Fold Q into the compressed space (MLA absorbed form) | memory | none | [16, 1280, 128] x [16, 128, 512] -> [16, 1280, 512] | 0.015 |
| 5 | Compress K,V (wkv_a) and share | comm | row: all-gather | [1280, 1792] x [1792, 576] -> [1280, 576] per row | 0.069 |
| 6 | Scores q . k^T over 51200 keys (the costed chip is the last row: most keys after the causal mask) | maths | none | [16, 1280, 576] x [16, 576, 51200] | 5.603 |
| 7 | Weighted sum scores . v | maths | none | [16, 1280, 51200] x [16, 51200, 512] -> [16, 1280, 512] | 4.981 |
| 8 | Read the KV cache (512 values per cached token) | memory | none | [46080, 512] bfp8 from DRAM | 0.058 |
| 9 | Expand V per head (wkv_b2) | memory | none | [16, 1280, 512] x [16, 512, 128] -> [16, 1280, 128] | 0.054 |
| 10 | Output projection + reduce-scatter | comm | row: reduce-scatter | [1280, 2048] x [2048, 7168] -> [1280, 1792] | 0.349 |
| 11 | Gather full width | comm | row: all-gather | [1280, 1792] -> [1280, 7168] | 0.086 |
| 12 | Router (gate), weight split over the row | memory | row: reduce-scatter + all-gather | [1280, 1792] x [1792, 384] -> [1280, 384] scores | 0.024 |
| 13 | Shared expert, tensor-parallel over the row (hidden 2048 / 4 = 512) | memory | row: reduce-scatter | [1280, 7168] x [7168, 512] (x2), [1280, 512] x [512, 7168] -> [1280, 1792] | 0.257 |
| 14 | Dispatch tokens to experts | comm | column: all-to-all | [1280, 7168] -> [640, 7168] for this chip's experts | 0.459 |
| 15 | Routed experts, 24 per chip (worst case: one expert gets all 5120 tokens, the rest 212 each) | maths | none | per expert [tokens, 7168] x [7168, 2048] bfp4 (x2), x [2048, 7168] | 2.875 |
| 16 | Combine expert outputs | comm | column: all-to-all | [640, 7168] -> [4, 1280, 7168] | 0.459 |
| 17 | Weighted sum of the 8 experts | memory | none | [4, 1280, 7168] -> [1280, 7168] | 0.179 |
| 18 | Reduce-scatter back to quarter width | comm | row: reduce-scatter | [1280, 7168] -> [1280, 1792] | 0.086 |
| 19 | Add shared + routed | memory | none | [1280, 1792] x 2 -> [1280, 1792] | 0.034 |
| 20 | Two RMSNorms and two residual adds | vector | none | [1280, 1792] | 0.112 |

## What a planner reuses
- The scheme: sequence on rows, weights / heads on columns, experts over the whole mesh with dispatch / combine only
  along columns (a token's copies already sit in every column after the row all-gather).
- Where the collectives go: row reduce-scatter after every input-split matmul, row all-gather before anything that needs
  whole token rows (router, experts), column all-to-all for dispatch and combine, row reduce-scatter back to the resting
  quarter-width layout.
- The cost picture: at long context attention dominates (scores + weighted sum ~ 10.6 of 15.9 ms on the last chunk; on
  the first chunk attention is 1.59 ms and the layer 6.16 ms), and dispatch + combine are the largest communication.
- Two modelling defaults to keep in mind: steps run strictly one after another (nappkin's "parallel" overlap option, the
  shared expert during dispatch and each expert during the previous combine, cuts the expert block from 4.46 to
  3.76 ms), and the expert load is the worst case.
