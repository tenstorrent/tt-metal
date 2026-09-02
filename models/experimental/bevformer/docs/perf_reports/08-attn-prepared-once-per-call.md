# Stage: 08-attn-prepared-once-per-call

| | |
|---|---|
| commit | [`c649b46fee2`](https://github.com/tenstorrent/tt-metal/commit/c649b46fee237ebc5b17e838b80354aa2c337144) |
| candidate | [5c](../perf_optimization_candidates.md#5c-untilize-attn-once-not-per-level) |
| config | `nuscenes_base`, 100×100, N150 |
| profile | **311.3 ms kernel**, 113 ops (−18), CSV `generated/profiler/reports/2026_09_01_23_48_22/` |
| delta | **−44.9 ms kernel (−12.6%)** vs [stage 07](07-sampling-grid-in-row-major.md)'s 356.2 ms; **−369.5 ms (−54.3%)** cumulative from stage 03 |
| PCC | **0.999651**, unchanged — pure layout change |
| suite | `tests/pcc/` **33 passed, 0 failed** |

## What changed

`attention_weights` was reshaped to `(bs, Q, heads, L, P)` in TILE after the softmax, putting a
`4 × 4` tail into the two tiled dimensions — padded to `32 × 32`, 16× waste — and
`_fused_msda_level` then sliced the level axis out of it per call. Slicing a tile-padded axis that is
not tile-aligned forces an unpad/pad round trip, so three of four levels each paid
untilize → slice → **re-tilize**, 9.6 ms apiece:

| rows | per level | ms |
|---|---|---:|
| 501–503 / 513–515 / 525–527 | untilize `14976×8×32×32` → slice → **re-tilize back**, levels 2–4 | 9.60 / 9.63 / 9.73 |
| 492 | level 1 takes a different path (TILE slice) | 2.41 |
| 493–495, 504–506, 516–518, 528–530 | reshape + transpose + untilize, ×4 | 11.6 |

**43.0 ms to feed the op a 0.5 MB tensor**, and 29 of it is the level-1-vs-rest asymmetry — which is
the proof the round trip was avoidable rather than intrinsic.

It is now prepared once, above the level loop, and a level costs one slice:

```
transpose (bs, Q, heads, L*P) -> (bs, heads, Q, L*P)   TILE
to_layout ROW_MAJOR
reshape -> (N, Q, L, P)                                 view
  per level:  attn_all[:, :, level]                     one slice
```

Two things make it cheap, both read off the stage-05 CSV rather than reasoned:

- **The head-major move stays in TILE.** A TILE transpose swaps whole tiles and measured 0.40 ms at
  these shapes (row 494); the ROW_MAJOR permute it replaces measured 5.01 ms (row 491), because its
  innermost contiguous run is a handful of elements. The naive version of this change — untilize
  first, then permute in ROW_MAJOR — would have given most of the win back.
- **The untilize runs on an `L*P`-wide tensor**, 16 wide padding to 32, instead of on a `(4, 4)` tail
  padding to `32 × 32`.

The `(L, P)` split then happens in ROW_MAJOR.

## Where the time went

| op | stage 07 | stage 08 | Δ |
|---|---:|---:|---:|
| TilizeWithValPadding | 17.2 ms (6 inst) | **0.9 ms (3)** | **−16.3** |
| UntilizeWithUnpadding | 26.5 ms (18) | 13.2 ms (12) | **−13.3** |
| ReshapeView | 55.3 ms (21) | 43.0 ms (15) | **−12.3** |
| Slice | 13.2 ms | 11.5 ms | −1.7 |
| Transpose | 7.9 ms | 6.7 ms | −1.2 |
| everything else | 236.1 ms | 236.0 ms | −0.1 |

The `Tilize` row is the re-tilize round trip, now three instances of housekeeping instead of six. Op
count 131 → **113**.

**Predicted −35 ms, measured −44.9.** The estimate under-counted because it priced the replacement
permute at the ROW_MAJOR rate; keeping it in TILE was worth the difference.

## Layer profile now

| Op | inst | ms | % | | Op | inst | ms | % |
|---|---:|---:|---:|---|---|---:|---:|---:|
| MSDAOperation | 5 | 168.1 | 54.0 | | Scatter | 1 | 10.5 | 3.4 |
| Permute | 10 | 43.4 | 13.9 | | Transpose | 9 | 6.7 | 2.2 |
| ReshapeView | 15 | 43.0 | 13.8 | | Matmul | 11 | 4.7 | 1.5 |
| UntilizeWithUnpadding | 12 | 13.2 | 4.2 | | BinaryNg | 17 | 2.4 | 0.8 |
| Slice | 13 | 11.5 | 3.7 | | | | | |

**`MSDAOperation` is 54.0% of the layer** and has not moved in absolute terms since stage 04
(167.8 → 168.1 ms). The two largest remaining model-side items are the `value` permute pair
([5d](../perf_optimization_candidates.md#5d-split-value-into-heads-in-row_major) /
[6](../perf_optimization_candidates.md#candidate-6--permutereshape-by-reformulation), 43.4 ms of
Permute) and the reshapes around them — neither as clean as 5a–5c were.
