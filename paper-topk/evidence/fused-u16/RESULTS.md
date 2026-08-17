# Fused-u16 end-to-end merge/rebuild, v1: row-parallel path (2026-08-17, p150a)

Rows whose full-width chunk count fits the 5-bit chunk-id stamp (<= 32 chunks
of the LLK window; FUSED_E2E factory gate, mirrored into the program hash via
fused_e2e_gate()) now stay in the fused [bf16|u16] word through every
merge/rebuild: the runtime chunk id (TT_SFPLOADI one-instruction variant) is
stamped into index bits [15:11] per chunk, and ONE global split per row
recovers chunk_id*K + within-chunk — deleting the per-chunk split, halving
merge bodies and rebuild transposes, and halving the DST footprint.

Measured (Tracy medians, 5 iters):
| cell | before | after |
|---|---|---|
| RP leaf 130x65536 k2048 (wave 1) | 359.7 us | **280.6 (1.28x)** |
| 30x65536 plain (RP) | 356.5 | 277.4 |
| **GLM plain call 160x65536 (hybrid)** | 467.0 | **388.0** (= 280.6 rp + 98.8 rect + 8.8 concat) |
| cumulative vs pre-campaign 712.3 | — | **1.84x** |

Front B's model (leaf ~277, cell ~365-375) verified within 4%; the residual
vs model is the v1-unfused rect remainder (v1.5 = tree-path fusion, modeled
rect 99 -> ~76 and single-row tree 34.3 -> ~26-28).

Gates: battery 42/42 (fused boundaries: 31-chunks+tail 63488, 32-chunk exact
65536 & 16384@512-window, UNFUSED fallback 66560 & 17408, poisoned-tail
valid_length both sides, values arm, cache pairs); nightly 181 pass + 2
IOMMU-env. Tie adjudication (cross-width valid_length_matches_sliced cell,
102400-unfused vs 56320-fused): 2,882 index diffs, 0 non-ties, value
multisets BIT-EXACT — test upgraded to assert exactly that contract
(tie-proof per diff + bitwise sorted-value equality; not a tolerance).

v1 scope notes: chunk skip compiled out under FUSED_E2E (its unfused-layout
scratch/threshold don't apply; at k=2048 the gate never arms below 512 chunks
anyway); production width 66560 (33 chunks) is information-theoretically
excluded forever (u16 must address every global position); 61440 and 65536
run fused. DS-V4 k512@65536 (128 chunks of the 512 window) needs the v2
scattered-bit stamp. Tree kernels (single-row anchor, rect waves, explicit
num_slices) remain unfused in v1.
