# Stage: 00-baseline

- source commit: `9f27e3a4d05`
- harness: `tests/perf/test_layer_perf.py`, one encoder layer, N150
- kernel time: **655.6 ms**
- op-to-op gap: **2416.5 ms**
- wall (kernel + gap): **3072.1 ms**
- device ops in the signposted region: **131**
- PCC gate: 0.997, passed
- CSV: `generated/profiler/reports/2026_08_26_12_45_28/ops_perf_results_2026_08_26_12_45_28.csv`

## Kernel time by op code

| Op | inst | us each | ms | % of kernel |
|---|---:|---:|---:|---:|
| ReshapeViewDeviceOperation | 21 | 7368.4 | 154.74 | 23.6 |
| GridSampleOperation | 5 | 23041.1 | 115.21 | 17.6 |
| ConcatDeviceOperation | 3 | 37848.6 | 113.55 | 17.3 |
| PermuteDeviceOperation | 23 | 4563.0 | 104.95 | 16.0 |
| BinaryNgDeviceOperation | 18 | 4686.2 | 84.35 | 12.9 |
| UntilizeWithUnpaddingDeviceOperation | 16 | 1676.0 | 26.82 | 4.1 |
| SliceDeviceOperation | 12 | 2130.8 | 25.57 | 3.9 |
| TilizeWithValPaddingDeviceOperation | 6 | 2230.0 | 13.38 | 2.0 |
| MatmulDeviceOperation | 11 | 426.7 | 4.69 | 0.7 |
| ReduceDeviceOperation | 2 | 2294.2 | 4.59 | 0.7 |
| FillPadDeviceOperation | 2 | 2231.5 | 4.46 | 0.7 |
| UnaryDeviceOperation | 3 | 537.8 | 1.61 | 0.2 |
| SoftmaxDeviceOperation | 2 | 609.8 | 1.22 | 0.2 |
| LayerNormDeviceOperation | 3 | 88.6 | 0.27 | 0.0 |
| TransposeDeviceOperation | 2 | 69.5 | 0.14 | 0.0 |
| CloneOperation | 2 | 51.6 | 0.10 | 0.0 |

**Matmul is 0.7% of kernel time.** This layer is not compute-bound; it is bound on data movement
and on layout churn. Every optimization that targets matmul fidelity, weight dtype, or program
configs — the whole Janus-Pro playbook — is irrelevant here until the movement ops shrink.

## The ten most expensive individual ops

Shapes are `logical` with the tile-padded extent alongside where they differ.

| # | Op | us | shape in → out | layout |
|--:|---|---:|---|---|
| 1 | Concat | 113542 | `[32, 2484, 1, 4]` ×4 → `[32, 2484, 4, 4]` | ROW_MAJOR |
| 2 | ReshapeView | 74268 | `[1, 1, 15261696, 4]` → `[1, 1, 3815424, 16]` | ROW_MAJOR |
| 3 | GridSample | 24694 | value `[48, 200, 113, 32]`, out `[48, 9936, 1, 32]` | ROW_MAJOR |
| 4 | GridSample | 24661 | value `[48, 100, 57, 32]` | ROW_MAJOR |
| 5 | GridSample | 24652 | value `[48, 50, 29, 32]` | ROW_MAJOR |
| 6 | GridSample | 24366 | value `[48, 25, 15, 32]` | ROW_MAJOR |
| 7 | BinaryNg | 23012 | `[80000, 1, 4→32, 2→32]` | TILE |
| 8 | ReshapeView | 21215 | `[1, 6, 30125→30144, 256]` → `[1, 180750, 8→32, 32]` | TILE |
| 9 | Permute | 19282 | `[6, 30125, 1→32, 256]` → `[1, 6, 30125→30144, 256]` | TILE |
| 10 | Permute | 19260 | same | TILE |

Three things fall out of this table:

- **The concat is the single most expensive op in the layer** and it is pure plumbing: four
  `[32, 2484, 1, 4]` sampling-offset tensors stacked along a length-4 axis, ROW_MAJOR, 64 cores,
  113 ms. This is the per-level `stack` in the MSDA decomposition.
- **GridSample cost does not track its input size.** Level 0 has 8× the value elements of level 3
  (`200×113` vs `25×15`) and runs 1.3% slower. The op is bound by the output — `9936` sampled
  points × 48 batch — not by the feature map it samples from, so shrinking levels buys nothing.
- **Tile padding on degenerate dimensions dominates the TILE ops.** Row 7 computes on
  `[80000, 1, 4, 2]` of real data padded to `[80000, 1, 32, 32]` — a **128× multiplier**, 23 ms of
  kernel time for 640K useful elements. Rows 9/10 pad a length-1 axis to 32. Rows 8 and 2 are
  reshapes over 15M-element ROW_MAJOR tensors.

## Op-to-op gaps — where the host stalls

| gap | on op | region |
|---:|---|---|
| **1916.95 ms** | Unary | first op after the SCA rebatch loop |
| 185.44 ms | Clone | TSA entry |
| 82.27 ms | BinaryNg | TSA |
| 81.45 ms | ReshapeView | TSA |
| 63.20 ms | ReshapeView | SCA scatter-back |
| 35.92 ms | Matmul | TSA |
| 22.81 ms | Matmul | TSA |
| all others | | < 6 ms each |

The 1.917 s gap is one stall, not an accumulation: the device sits idle while
`TTSpatialCrossAttention.forward` runs its `bs × num_cams` host loop —
`to_torch(query)`, `to_torch(reference_points_cam)`, `to_torch` of both accumulators and
`from_torch` of both, once per camera, ~36 transfers of tensors that are invariant across the loop.

**Two-thirds of this layer's wall clock is one Python loop.** No device-side change comes close.
