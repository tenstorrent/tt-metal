# Stage: 08-attn-prepared-once-per-call

- candidate: [5c](../perf_optimization_candidates.md#5c-untilize-attn-once-not-per-level)
- config: `nuscenes_base`, 100×100, N150
- layer profile: **311.3 ms kernel**, 113 device ops (−18), CSV
  `generated/profiler/reports/2026_09_01_23_48_22/`
- **−44.9 ms kernel (−12.6%)** against [stage 07](07-sampling-grid-in-row-major.md)'s 356.2 ms;
  **−369.5 ms (−54.3%)** cumulative against the stage-03 layer
- PCC: **0.999651**, unchanged from stage 07 — pure layout change
- `tests/pcc/`: **33 passed, 0 failed**

## What this change was

`attention_weights` was reshaped to `(bs, Q, heads, L, P)` in TILE after the softmax, which puts a
`4 × 4` tail into the two tiled dimensions — padded to `32 × 32`, 16× waste — and then
`_fused_msda_level` sliced the level axis out of it once per call. Slicing a tile-padded axis that
is not tile-aligned forces an unpad/pad round trip, so three of the four levels each paid
untilize → slice → **re-tilize**, 9.6 ms apiece. Level 1 took a different path at 2.4 ms, which is
what showed the round trip was avoidable rather than intrinsic.

The whole per-level block — slice, reshape, transpose, untilize, and for three levels the round
trip — was **43.0 ms to feed the op a 0.5 MB tensor**.

It is now prepared once, above the level loop, and a level costs one slice:

```
transpose (bs, Q, heads, L*P) -> (bs, heads, Q, L*P)   TILE
to_layout ROW_MAJOR
reshape -> (N, Q, L, P)                                 view
  per level:  attn_all[:, :, level]                     one slice
```

Two things make it cheap, and both come out of the stage-05 CSV rather than from first principles:

- **The head-major move stays in TILE.** A TILE transpose swaps whole tiles and measured 0.40 ms at
  these shapes (row 494); the ROW_MAJOR permute it replaces measured 5.01 ms (row 491) because its
  innermost contiguous run is a handful of elements. The naive version of this change — untilize
  first, then permute in ROW_MAJOR — would have given most of the win back.
- **The untilize runs on an `L*P`-wide tensor**, 16 wide padding to 32, instead of on a `(4, 4)`
  tail padding to `32 × 32`.

The `(L, P)` split then happens in ROW_MAJOR, where a trailing-axis split is a view.

## Where the time went

| op | stage 07 | stage 08 | Δ |
|---|---:|---:|---:|
| TilizeWithValPaddingDeviceOperation | 17.2 ms (6 inst) | **0.9 ms (3 inst)** | **−16.3** |
| UntilizeWithUnpaddingDeviceOperation | 26.5 ms (18) | 13.2 ms (12) | **−13.3** |
| ReshapeViewDeviceOperation | 55.3 ms (21) | 43.0 ms (15) | **−12.3** |
| SliceDeviceOperation | 13.2 ms | 11.5 ms | −1.7 |
| TransposeDeviceOperation | 7.9 ms | 6.7 ms | −1.2 |
| everything else | 236.1 ms | 236.0 ms | −0.1 |

The `Tilize` row is the re-tilize round trip, and it is now three instances of housekeeping instead
of six. Op count 131 → **113**.

Predicted −35 ms, measured −44.9. The estimate under-counted because it priced the replacement
permute at the ROW_MAJOR rate; keeping it in TILE was worth the difference.

## Layer profile now

| Op | inst | ms | % of kernel |
|---|---:|---:|---:|
| MSDAOperation | 5 | 168.1 | 54.0 |
| PermuteDeviceOperation | 10 | 43.4 | 13.9 |
| ReshapeViewDeviceOperation | 15 | 43.0 | 13.8 |
| UntilizeWithUnpaddingDeviceOperation | 12 | 13.2 | 4.2 |
| SliceDeviceOperation | 13 | 11.5 | 3.7 |
| ScatterDeviceOperation | 1 | 10.5 | 3.4 |
| TransposeDeviceOperation | 9 | 6.7 | 2.2 |
| MatmulDeviceOperation | 11 | 4.7 | 1.5 |
| BinaryNgDeviceOperation | 17 | 2.4 | 0.8 |

**`MSDAOperation` is now 54.0% of the layer** and has not moved in absolute terms since stage 04
(167.8 → 168.1 ms). Everything candidate 5 has done so far is model-side; from here the two largest
remaining model-side items are the `value` permute pair
([5d](../perf_optimization_candidates.md#5d-split-value-into-heads-in-row_major) /
[6](../perf_optimization_candidates.md#candidate-6--permutereshape-by-reformulation), 43.4 ms of
Permute) and the reshapes around them, and neither is as clean as 5a–5c were.
