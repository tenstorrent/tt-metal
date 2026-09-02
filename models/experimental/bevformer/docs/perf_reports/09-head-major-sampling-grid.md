# Stage: 09-head-major-sampling-grid

- source commit: [`e6b4ce53fe1`](https://github.com/tenstorrent/tt-metal/commit/e6b4ce53fe163df02964a3655d55222a0a3ed5e0)
- candidate: [5e](../perf_optimization_candidates.md#5e-permute-grid-once-slice-after)
- config: `nuscenes_base`, 100×100, N150
- layer profile: **286.8 ms kernel**, 112 device ops (−1), CSV
  `generated/profiler/reports/2026_09_02_11_34_14/`
- **−24.5 ms kernel (−7.9%)** against [stage 08](08-attn-prepared-once-per-call.md)'s 311.3 ms;
  **−394.0 ms (−57.8%)** cumulative against the stage-03 layer
- PCC: **0.999651**, unchanged — pure layout change. MSDA module PCC 0.999934
- `tests/pcc/`: **33 passed, 0 failed**

## The entry as written was wrong, twice

Worth recording, because both errors came from the same source: assuming ROW_MAJOR is where layout
work is cheap.

1. **"Permute once above the loop, slice after" was worth ~4 ms, not 13.** The per-level permute
   runs on an *already sliced* tensor. Permuting the full tensor once is 4× the data, so four
   permutes of one level ≈ one permute of four levels. There was no hoist win to collect.
2. **"A trailing-axis split is a free ROW_MAJOR view" is false in ttnn.** A ROW_MAJOR tensor's row
   width is its last dimension, so a reshape that changes that extent re-lays the tensor out.
   Stage 07's `(bs, Q, 256) → (bs, Q, heads, L, P, 2)` was **7.09 ms**, not zero
   (`2026_09_01_23_48_22` row 466), and stage 08's `(N, Q, L*P) → (N, Q, L, P)` was another 2.59 ms.
   Both were booked as views.

What actually pays is [stage 08](08-attn-prepared-once-per-call.md)'s lesson, applied to the grid:
**move the head axis while the tensor is still tiled.** A TILE transpose swaps whole tiles — 0.30 to
0.40 ms at these shapes — where the ROW_MAJOR permute of the same axes cost 5.00 ms per level.

## What this change was

The grid was built query-major, `(bs, Q, heads, L, P, 2)`, so every level had to move the head axis
in front of the query axis before the fused op would take it:

```
per level:  slice L        1.28 ms
            permute        5.00 ms      ← (bs, Q, heads, P, 2) -> (bs, heads, Q, P, 2)
            reshape        (view)
                        ×4 levels at SCA + 1 at TSA = 25.1 ms + 3.3 ms
```

It is now built head-major, once, immediately after the Linear and before leaving TILE:

```
Linear    -> (bs, Q, 256)                     TILE
reshape   -> (bs, Q, heads, L*P*2)            TILE
transpose -> (bs, heads, Q, L*P*2)            TILE      ← the head move, 0.30–0.40 ms
to_layout -> ROW_MAJOR                                   (Q × 32, nothing to pad)
add grid_bias                                            broadcast over heads
reshape   -> (bs, heads, Q, L, P, 2)          the one re-layout, paid once
  per level:  slice L, then a row regroup at constant row width — free
```

Two secondary wins fell out:

- **`grid_bias` shrank 8×.** Head-major means the bias no longer varies with head, so it is built
  with a length-1 head axis and broadcast: the `repeat` went from 32 copies to 4
  (`RepeatCodegen` 3.0 → 1.2 ms across the layer), and the bias tensor is an eighth of the size.
- **The per-level grid reshape became genuinely free.** `(bs, heads, Q, P, 2) → (N, Q*P, 1, 2)`
  keeps the trailing 2, so it only regroups rows at constant row width — the one ROW_MAJOR reshape
  ttnn does not re-lay out.

## Where the time went

| op | stage 08 | stage 09 | Δ |
|---|---:|---:|---:|
| PermuteDeviceOperation | 43.4 ms (10 inst) | **19.9 ms (5 inst)** | **−23.5** |
| RepeatCodegenDeviceOperation | 3.1 ms | 1.2 ms | −1.9 |
| ReshapeViewDeviceOperation | 43.0 ms (15) | 42.3 ms (17) | −0.7 |
| TransposeDeviceOperation | 6.7 ms (9) | 7.5 ms (11) | +0.8 |
| UntilizeWithUnpaddingDeviceOperation | 13.2 ms | 13.8 ms | +0.6 |
| BinaryNgDeviceOperation | 2.4 ms | 3.0 ms | +0.6 |
| everything else | 195.5 ms | 195.1 ms | −0.4 |

Five permutes deleted — four SCA levels and TSA's one — and the two new TILE transposes cost 0.8 ms
between them. **The remaining 19.9 ms of Permute is essentially one op**: the 19.3 ms `value`
head permute, which is [5d](../perf_optimization_candidates.md#5d-split-value-into-heads-in-row_major)
and [candidate 6](../perf_optimization_candidates.md#what-a-weight-reorder-cannot-reach)'s known
irreducible-by-weight-reorder case.

## Layer profile now

| Op | inst | ms | % of kernel |
|---|---:|---:|---:|
| MSDAOperation | 5 | 167.9 | 58.5 |
| ReshapeViewDeviceOperation | 17 | 42.3 | 14.7 |
| PermuteDeviceOperation | 5 | 19.9 | 6.9 |
| UntilizeWithUnpaddingDeviceOperation | 12 | 13.8 | 4.8 |
| SliceDeviceOperation | 13 | 11.4 | 4.0 |
| ScatterDeviceOperation | 1 | 10.5 | 3.7 |
| TransposeDeviceOperation | 11 | 7.5 | 2.6 |
| MatmulDeviceOperation | 11 | 4.7 | 1.6 |
| BinaryNgDeviceOperation | 17 | 3.0 | 1.0 |

**`MSDAOperation` is 58.5% of the layer.** The two largest model-side items left are the 21.2 ms
`value` head-split reshape and the 19.3 ms `value` permute — both the same operand, both
[5d](../perf_optimization_candidates.md#5d-split-value-into-heads-in-row_major).

## The rule this stage establishes

Two stages in a row have been won by the same move, so state it as a rule for the rest of this
backlog:

**Do axis moves in TILE; do only elementwise work and constant-row-width regrouping in ROW_MAJOR.**

TILE transposes swap tiles and cost ~0.4 ms at these shapes. ROW_MAJOR permutes with a short
innermost run cost 3–5 ms for the same bytes. And a ROW_MAJOR reshape is free only when the last
dimension does not change; otherwise it is a full re-layout and must be budgeted like one.
