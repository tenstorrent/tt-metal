# Stage: 00-baseline

| | |
|---|---|
| commit | `9f27e3a4d05` |
| harness | `tests/perf/test_layer_perf.py`, one encoder layer, N150 |
| profile | **655.6 ms kernel** / 2416.5 ms gap / **3072.1 ms wall**, 131 ops |
| CSV | `generated/profiler/reports/2026_08_26_12_45_28/` |
| PCC | gate 0.997, passed |

## Kernel by op code

| Op | inst | ms | % |
|---|---:|---:|---:|
| ReshapeView | 21 | 154.74 | 23.6 |
| GridSample | 5 | 115.21 | 17.6 |
| Concat | 3 | 113.55 | 17.3 |
| Permute | 23 | 104.95 | 16.0 |
| BinaryNg | 18 | 84.35 | 12.9 |
| UntilizeWithUnpadding | 16 | 26.82 | 4.1 |
| Slice | 12 | 25.57 | 3.9 |
| TilizeWithValPadding | 6 | 13.38 | 2.0 |
| Matmul | 11 | 4.69 | 0.7 |
| Reduce | 2 | 4.59 | 0.7 |
| FillPad | 2 | 4.46 | 0.7 |
| Unary / Softmax / LayerNorm / Transpose / Clone | 12 | 3.34 | 0.5 |

**Matmul is 0.7%.** This layer is bound on data movement and layout churn, not compute — so the
whole matmul-tuning playbook (fidelity, weight dtype, program configs) is irrelevant until the
movement ops shrink.

## The ten most expensive ops

Logical shapes, with the tile-padded extent where it differs.

| # | Op | µs | shape in → out | layout |
|--:|---|---:|---|---|
| 1 | Concat | 113542 | `[32, 2484, 1, 4]` ×4 → `[32, 2484, 4, 4]` | RM |
| 2 | ReshapeView | 74268 | `[1, 1, 15261696, 4]` → `[1, 1, 3815424, 16]` | RM |
| 3–6 | GridSample ×4 | 24694 / 24661 / 24652 / 24366 | value `[48, 200, 113, 32]` … `[48, 25, 15, 32]`, out `[48, 9936, 1, 32]` | RM |
| 7 | BinaryNg | 23012 | `[80000, 1, 4→32, 2→32]` | TILE |
| 8 | ReshapeView | 21215 | `[1, 6, 30125→30144, 256]` → `[1, 180750, 8→32, 32]` | TILE |
| 9–10 | Permute ×2 | 19282 / 19260 | `[6, 30125, 1→32, 256]` → `[1, 6, 30125→30144, 256]` | TILE |

- **The concat is the most expensive op in the layer and it is pure plumbing** — four
  `[32, 2484, 1, 4]` tensors stacked along a length-4 axis, 64 cores, 113 ms. It is the per-level
  `stack` in the MSDA decomposition.
- **GridSample cost does not track input size.** Level 0 has 8× the value elements of level 3 and
  runs 1.3% slower: the op is bound by the output (9936 points × 48 batch), not by the feature map.
  Shrinking levels buys nothing.
- **Tile padding on degenerate axes dominates the TILE ops.** Row 7 computes `[80000, 1, 4, 2]` of
  real data padded to `[80000, 1, 32, 32]` — **128×**, 23 ms for 640K useful elements. Rows 9/10 pad
  a length-1 axis to 32.

## Op-to-op gaps

| gap | on op | region |
|---:|---|---|
| **1916.95 ms** | Unary | first op after the SCA rebatch loop |
| 185.44 ms | Clone | TSA entry |
| 82.27 / 81.45 ms | BinaryNg / ReshapeView | TSA |
| 63.20 ms | ReshapeView | SCA scatter-back |
| 35.92 / 22.81 ms | Matmul ×2 | TSA |
| rest | | < 6 ms each |

The 1.917 s is **one stall**, not an accumulation: the device idles while
`TTSpatialCrossAttention.forward` runs a `bs × num_cams` host loop — `to_torch` of `query`,
`reference_points_cam` and both accumulators, `from_torch` of both, per camera, ~36 transfers of
tensors that are invariant across the loop.

**Two-thirds of the layer's wall clock is one Python loop.** No device-side change comes close.
