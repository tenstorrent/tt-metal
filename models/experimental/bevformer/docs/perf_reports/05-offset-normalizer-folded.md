# Stage: 05-offset-normalizer-folded

| | |
|---|---|
| commit | [`d86d2f722fb`](https://github.com/tenstorrent/tt-metal/commit/d86d2f722fbecdd210e96c2927a9b9648211ebf5) |
| candidate | [3](../perf_optimization_candidates.md#candidate-3--tile-padding-waste) |
| config | `nuscenes_base`, 100×100, N150 |
| profile | **456.8 ms kernel**, 127 ops (−2), CSVs `generated/profiler/reports/2026_08_28_10_23_13/` and `…_10_30_24/` |
| delta | **−32.7 ms kernel (−6.7%)** vs [stage 04](04-fused-msda.md)'s 489.5 ms; **−224.3 ms (−32.9%)** cumulative from stage 03 |
| PCC | **0.999611**, byte-identical to stage 04 — the fold is exact |
| suite | `tests/pcc/` **32 passed, 1 failed** — same counts as stage 04, same pre-existing 200×200 OOM |

## No wall-clock number, deliberately

The layer was profiled twice on identical code:

| run | kernel | gap | wall |
|---|---:|---:|---:|
| `2026_08_28_10_23_13` | 456.8 ms | 93.4 ms | 550.2 ms |
| `2026_08_28_10_30_24` | 456.8 ms | 151.2 ms | 608.0 ms |

**Kernel is identical to 0.1 ms; gap differs by 57.8 ms.** Stage 04 measured 14.0 ms on the same
harness the night before — 14.0 / 93.4 / 151.2 ms across three runs, with no code change explaining
any of it. Taken at face value the wall column would show this change as a ~100 ms *regression* while
kernel drops 32.7 ms. `DEVICE KERNEL DURATION` is the only trustworthy figure in this harness; see
[PERF.md](../PERF.md#the-gap-column-is-not-reliable).

## What changed

`sampling_offsets` is a Linear whose output was divided, every call, by the per-level offset
normalizer `[W, H]` — a broadcast SFPU divide over a `(…, num_levels, num_points, 2)` tensor whose
extent-2 trailing axis tile-pads to 32.

The normalizer is fixed by the feature-pyramid config, so the division is a static per-output-channel
scale and folds into the weight exactly: `s · (Wx + b) == (Wx + b) / normalizer`. The Linear emits
`num_heads * num_levels * num_points * 2` channels ordered `(head, level, point, xy)` with xy
innermost, so the scale is one row of `1/W_l` and `1/H_l` per level;
`preprocess_linear_weight` stores the weight transposed as `(in, out)` and the bias as `(1, out)`, so
a single `(1, out)` row broadcasts over both. Computed once per `spatial_shapes`, cached on the
module. The divide and the cached `offset_normalizer` that fed it are both deleted.

This is the fix [`vadv2`](../../../vadv2/tt/tt_utils.py) already had —
`fold_offset_normalizer_into_weight`. That version reads `spatial_shapes[0]` and handles
`num_levels == 1` only; this one builds the per-level scale, covering SCA's four levels as well as
TSA's one.

## Where the time went

| op | stage 04 | stage 05 | Δ | | region | stage 04 | stage 05 | Δ |
|---|---:|---:|---:|---|---|---:|---:|---:|
| BinaryNg | 81.5 ms | **48.2 ms** | **−33.3** | | TSA | 78.7 ms | 69.6 ms | −9.1 |
| everything else | 408.0 ms | 408.6 ms | +0.6 | | SCA | 352.0 ms | 328.3 ms | −23.7 |

Candidate 3 was scoped at ~24 ms from the SCA divide alone; TSA has the same divide and gave up
another 9 ms.

## Layer profile now

129 → 127 ops; the two removed ops are the two divides.

| Op | inst | ms | % | | Op | inst | ms | % |
|---|---:|---:|---:|---|---|---:|---:|---:|
| MSDAOperation | 5 | 167.8 | 36.7 | | TilizeWithValPadding | 6 | 17.1 | 3.7 |
| ReshapeView | 20 | 77.1 | 16.9 | | Slice | 13 | 13.3 | 2.9 |
| Permute | 11 | 62.7 | 13.7 | | Scatter | 1 | 10.5 | 2.3 |
| BinaryNg | 17 | 48.2 | 10.6 | | Transpose | 12 | 8.2 | 1.8 |
| UntilizeWithUnpadding | 17 | 42.4 | 9.3 | | Matmul | 11 | 4.7 | 1.0 |

`MSDAOperation` is **36.7% of the layer** and the gap to the next op has widened.
[Candidate 10](../perf_optimization_candidates.md#candidate-10--msdaoperation-itself) is the only
remaining item of comparable size, and it is upstream rather than model-side.

**Note for stage 03:** its `offset_normalizer` cache is removed along with the divide it served —
superseded, not reverted. The saving it measured was real; this change removes the consumer entirely.
