# Stage: 03-camera-fold

- source commit: [`65ae83ff009`](https://github.com/tenstorrent/tt-metal/commit/65ae83ff009)
- harness: `tests/perf/test_layer_perf.py`, one encoder layer, N150, Release build
- kernel time: **450.6 ms** (−36.8 ms)
- op-to-op gap: **33.2 ms** (−4.7 ms)
- wall: **483.8 ms** (−41.5 ms, **−7.9%**)
- device ops in the signposted region: **140** (unchanged)
- PCC gate: **0.999611**, unchanged from stage 02
- CSV: `generated/profiler/reports/2026_08_27_10_40_12/ops_perf_results_2026_08_27_10_40_12.csv`

## What this change was

`key` and `value` reach spatial cross attention as `[num_cams, L, bs, embed_dims]`.
That puts `bs` **second-to-last**, and `TILE_LAYOUT` pads the last two axes to a
32×32 tile. At `bs=1` the profiler reports the input verbatim:

```
INPUT_0   6 x 30125 x 32[1] x 256   TILE   BFLOAT16   DRAM_INTERLEAVED
OUTPUT_0  1 x 6 x 30144[30125] x 256  TILE
```

**A 2.96 GB buffer holding 92.6 MB of data** — 32× — and it was permuted twice,
once for `key` and once for `value`, which every caller passes as the same
tensor.

Two parts:

- **The fold runs in ROW_MAJOR.** `_fold_cameras_into_batch` untilizes, permutes,
  reshapes to `[bs*num_cams, L, embed_dims]`, and tiles *that* — `30125 → 30144`,
  0.06% padding instead of 32×. Converting after the fold is the cheap direction.
- **It runs once when the arguments alias.** `value_reshaped = key_reshaped if
  value is key`.

Every harness now uploads camera features ROW_MAJOR, so the padded buffer is
never built at all. SCA still accepts a tiled input; it just pays the untilize.

## Where the time went

| region | ops | kernel | Δ kernel |
|---|---:|---:|---:|
| SCA — rebatch | 11 | 8.4 ms | **−35.8** |
| SCA — MSDA | 70 | 349.1 ms | −1.0 |
| TSA — MSDA | 35 | 78.7 ms | 0.0 |
| SCA — scatter-back + normalise | 13 | 12.7 ms | 0.0 |
| FFN | 3 | 1.1 ms | 0.0 |
| TSA — outside MSDA | 3 | 0.2 ms | 0.0 |

The whole delta sits in the SCA rebatch region, which is where the two permutes
lived. Nothing downstream moved.

## The trade

| Op | before | after | Δ |
|---|---:|---:|---:|
| PermuteDeviceOperation | 83.7 ms / 24 | **47.0 ms / 23** | **−36.7** |
| TilizeWithValPadding | 0.8 ms / 7 | 1.8 ms / 8 | +0.9 |

The two 19 ms padded permutes are gone; the largest remaining permute is 10.1 ms.
Device time spent on padding-carrying ops: **180.9 ms (37%) → 142.1 ms (32%)**.

One extra tilize is the price of tiling after the fold instead of before it.

## Correctness

- Full `tests/pcc/` at this commit: 16 passed across SCA, layer and encoder.
- The one failure was `200×200`, on the pre-existing OOM in the sampling chain —
  byte-identical allocation request, unrelated to this change. Stage 04 removes it.
- Perf-harness PCC gate 0.999611, unchanged.

## What this changes about the plan

This is the second of the two root causes behind candidate 3, and the simpler
one: a degenerate axis in the wrong position, fixed by choosing when to tile.

The first cause — trailing `(num_points, 2)` padding to 128× — is untouched here
and is still the larger share. It is what stage 04 addresses, and what makes
`200×200` unrunnable.
