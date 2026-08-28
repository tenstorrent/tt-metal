# Stage: 05-offset-normalizer-folded

- source commit: working tree on `ctr-mmicic/bev-former` (parent
  [`a90ab665de7`](https://github.com/tenstorrent/tt-metal/commit/a90ab665de7))
- config: `nuscenes_base`, 100×100, N150
- layer profile: **456.8 ms kernel**, 127 device ops (−2), CSVs
  `generated/profiler/reports/2026_08_28_10_23_13/` and `…_10_30_24/`
- **−32.7 ms kernel (−6.7%)** against [stage 04](04-fused-msda.md)'s 489.5 ms;
  **−224.3 ms (−32.9%)** cumulative against the stage-03 layer
- PCC: **0.999611**, byte-identical to stage 04 — the fold is exact, not approximate
- `tests/pcc/`: **32 passed, 1 failed**, the same counts as stage 04 and the same single
  pre-existing 200×200 OOM

## No wall-clock number in this report, deliberately

The layer was profiled twice on identical code:

| run | kernel | gap | wall |
|---|---:|---:|---:|
| `2026_08_28_10_23_13` | 456.8 ms | 93.4 ms | 550.2 ms |
| `2026_08_28_10_30_24` | 456.8 ms | 151.2 ms | 608.0 ms |

**Kernel is identical to 0.1 ms. Gap differs by 57.8 ms.** Stage 04 measured 14.0 ms on the same
harness the previous night. Across three runs the gap column reads 14.0 / 93.4 / 151.2 ms with no
code change explaining any of it.

So `wall` would show this change as a 100 ms *regression* while kernel drops 32.7 ms. It is not a
regression; the gap column is not currently a measurement. `DEVICE KERNEL DURATION` is the only
trustworthy figure in this harness right now, and it is what this stage reports. See
[PERF.md](../PERF.md#the-gap-column-is-not-reliable) for what this invalidates.

## What this change was

`sampling_offsets` is a Linear whose output was divided, every call, by the per-level offset
normalizer `[W, H]` — a broadcast SFPU divide over a `(…, num_levels, num_points, 2)` tensor whose
extent-2 trailing axis tile-pads to 32.

The normalizer is fixed by the feature-pyramid config, so the division is a static per-output-channel
scale and folds into the Linear weight exactly:

```
s · (Wx + b)  ==  (Wx + b) / normalizer
```

The Linear emits `num_heads * num_levels * num_points * 2` channels ordered `(head, level, point,
xy)` with xy innermost, so the scale is one row of `1/W_l` and `1/H_l` per level.
`preprocess_linear_weight` stores the weight transposed as `(in, out)` and the bias as `(1, out)`,
so a single `(1, out)` row broadcasts over both. Computed once per `spatial_shapes` and cached on
the module.

The divide, and the cached `offset_normalizer` tensor that fed it, are both deleted.

This is the fix [`vadv2`](../../../vadv2/tt/tt_utils.py) already had —
`fold_offset_normalizer_into_weight`. That version reads `spatial_shapes[0]` and only handles
`num_levels == 1`; this one builds the per-level scale, so it covers SCA's four levels as well as
TSA's one.

## Where the time went

| op | stage 04 | stage 05 | Δ |
|---|---:|---:|---:|
| BinaryNg | 81.5 ms | **48.2 ms** | **−33.3** |
| everything else | 408.0 ms | 408.6 ms | +0.6 |

| region | stage 04 | stage 05 | Δ |
|---|---:|---:|---:|
| TSA | 78.7 ms | 69.6 ms | −9.1 |
| SCA | 352.0 ms | 328.3 ms | −23.7 |

[Candidate 3](../perf_optimization_candidates.md#candidate-3--tile-padding-waste) was scoped at
~24 ms from the SCA divide alone. TSA has the same divide and gave up another 9 ms.

## Layer profile now

129 → 127 ops; the two removed ops are the two divides.

| Op | inst | ms | % of kernel |
|---|---:|---:|---:|
| MSDAOperation | 5 | 167.8 | 36.7 |
| ReshapeViewDeviceOperation | 20 | 77.1 | 16.9 |
| PermuteDeviceOperation | 11 | 62.7 | 13.7 |
| BinaryNgDeviceOperation | 17 | 48.2 | 10.6 |
| UntilizeWithUnpaddingDeviceOperation | 17 | 42.4 | 9.3 |
| TilizeWithValPaddingDeviceOperation | 6 | 17.1 | 3.7 |
| SliceDeviceOperation | 13 | 13.3 | 2.9 |
| ScatterDeviceOperation | 1 | 10.5 | 2.3 |
| TransposeDeviceOperation | 12 | 8.2 | 1.8 |
| MatmulDeviceOperation | 11 | 4.7 | 1.0 |

`MSDAOperation` is now **36.7% of the layer** and the gap to the next op has widened.
[Candidate 6](../perf_optimization_candidates.md#candidate-6--msdaoperation-itself) is the only
remaining item of comparable size, and it is an upstream question rather than a model-side one.

## Note for stage 03

Stage 03 cached the `offset_normalizer` tensor to stop rebuilding it per call. That cache is now
removed along with the divide it served — superseded, not reverted. The saving stage 03 measured
was real; this change removes the consumer entirely.
