# Stage: 07-folded-grid-page

- harness: `tests/perf/test_layer_perf.py`, one encoder layer, N150, Release build
- kernel time: **112.4 ms** (−12.9 ms)
- op-to-op gap: **40.3 ms** (−3.8 ms)
- wall: **152.7 ms** (−16.7 ms, **−9.9%**)
- device ops in the signposted region: **111** (−2)
- PCC gate: **0.999609**, unchanged from stage 06
- CSV: `generated/profiler/reports/2026_08_28_10_05_57/ops_perf_results_2026_08_28_10_05_57.csv`

## What this change was

`ttnn.experimental.multi_scale_deformable_attn` now accepts the grid with the
point axis folded into the last dimension — `(N, Q, 1, P*2)` — as well as the
`(N, Q*P, 1, 2)` form it took before. BEVFormer passes the folded form and stops
spelling `sampling_grids` out into six dimensions.

## Why the page size was the whole story

A ROW_MAJOR page is the tensor's last dimension, and the buffer allocates it
rounded up to the DRAM alignment:

```
aligned_page_size(raw, DRAM) = round_up(raw, 32)

(N, Q*P, 1, 2)  page  4 B  ->  32 B allocated    8x
(N, Q, 1, P*2)  page 16 B  ->  32 B allocated    2x, and a quarter of the pages
```

The SCA grid is 3.83 M points. At one point per page it occupies **122 MB** of
DRAM for 7.67 MB of data, and the reader issues P four-byte NoC reads per query.
Folded, it is 30.6 MB and one read.

That also explains the profile that motivated this. Sorting the layout ops by
effective bandwidth sorts them by page size, not by size:

| op | MB | ms | GB/s r+w | page |
|---|---:|---:|---:|---:|
| reshape `180750x256 -> 1446000x32` | 92.5 | 4.87 | 38 | 512 → 64 B |
| permute `6x22600x8x32 -> 6x8x22600x32` | 69.4 | 9.96 | 14 | 64 B |
| reshape `119808x32 -> 1916928x2` | 7.7 | 7.42 | **2.1** | 32 → **4 B** |
| reshape `80000x8 -> 320000x2` | 1.3 | 1.70 | **1.5** | 16 → **4 B** |

Against a 288 GB/s DRAM roof, none of these is bandwidth-bound. The two slowest
move the least data; they are the two that write a 4-byte page.

## What it cost, and what it did not

| op | stage 06 | stage 07 | Δ |
|---|---:|---:|---:|
| ReshapeViewDeviceOperation | 26.18 / 16 | **17.20 / 14** | **−8.98** |
| SliceDeviceOperation | 11.31 / 13 | **8.86 / 13** | **−2.45** |
| MSDAOperation | 29.46 / 5 | **27.66 / 5** | **−1.80** |
| everything else | 58.4 | 58.7 | +0.3 |

Both 4-byte-page reshapes are gone: `119808x32 -> 1916928x2` (7.42 ms, SCA) and
`80000x8 -> 320000x2` (1.67 ms, TSA). The per-level grid slice moved from
`2496x4x4x2 -> 2496x1x4x2` at 1.34 ms to `6x8x2496x32 -> 6x8x2496x8` at 1.09 ms.
`MSDAOperation` gained the rest: a quarter as many NoC reads, four times the
size.

By region:

| region | ops | kernel | Δ kernel |
|---|---:|---:|---:|
| SCA — MSDA | 49 | 75.98 ms | **−11.23** |
| TSA — MSDA | 27 | 13.40 ms | **−1.73** |
| SCA — scatter-back + normalise | 16 | 13.02 ms | 0.00 |
| SCA — rebatch | 11 | 8.51 ms | +0.08 |
| FFN | 5 | 1.29 ms | −0.01 |
| TSA — outside MSDA | 3 | 0.18 ms | 0.00 |

## Why stage 05 measured this as nothing

Stage 05 tried the Python half of this — dropping the 6-D reshape for a
column-range slice — and measured `263.6 → 263.5 ms`. That result was correct.
Without the op change the grid still has to reach the op at width 2, so the
4-byte page is still written; only the op that writes it changes. The measurement
was right and the conclusion drawn from it ("the simpler code was also the faster
one") held for what was tried.

The reason to retry was not that the kernel had since got faster. A layout op
that costs 7.4 ms costs 7.4 ms regardless of what runs next to it, and the
retry rationale written into stage 06 on that basis was wrong. What made this
work was reading the page size, which pointed at the op rather than at Python.

## Backward compatibility

The op takes any divisor of `P` per page. `grid_pts_per_stick` is derived
host-side from the grid's last dimension and passed to the reader as a
compile-time arg, so `(N, Q*P, 1, 2)` still compiles to the previous behaviour.
`models/experimental/vadv2/tt/tt_utils.py` is untouched.

## Correctness

- `tests/ttnn/unit_tests/operations/experimental/test_multi_scale_deformable_attn.py`
  — **138 passed**. `test_msda_grid_point_folding` covers `pts_per_page` 1, 2 and
  4 against `P` 4 and 8; `pts_per_page = 1` is the legacy path.
  `test_msda_rejects_grid_width_not_dividing_p` covers the rejected case.
- Full `models/experimental/bevformer/tests/pcc/` — **33 passed**, exit 0,
  nothing deselected.
- Perf-harness PCC gate 0.999609, unchanged.

## What this changes about the plan

Layout plumbing is **60.5 ms, 54% of kernel**, against `MSDAOperation`'s 24.6%.
The proportion barely moved because the op shrank too.

What remains has no more 4-byte pages in it. The largest items are a 92.5 MB
reshape at 38 GB/s and a 69.4 MB permute at 14 GB/s, both splitting or reordering
`embed_dims` into heads, and neither has a Python expression that helps:
permuting before the level split turns the split's 512-byte pages into 64-byte
ones and measures worse by arithmetic alone. That is
[candidate 7](../perf_optimization_candidates.md#candidate-7--an-msda-head-reshape-op).
