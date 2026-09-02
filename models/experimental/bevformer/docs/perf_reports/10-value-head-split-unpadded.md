# Stage: 10-value-head-split-unpadded

- source commit: [`7820f325bd8`](https://github.com/tenstorrent/tt-metal/commit/7820f325bd86cbca8dfcbe27cf460204c8d9773c)
- candidate: [5d](../perf_optimization_candidates.md#5d-split-value-into-heads-in-row_major)
- config: `nuscenes_base`, 100×100, N150
- layer profile: **280.2 ms kernel**, 106 device ops (−6), CSV
  `generated/profiler/reports/2026_09_02_11_49_26/`
- **−6.6 ms kernel (−2.3%)** against [stage 09](09-head-major-sampling-grid.md)'s 286.8 ms;
  **−400.6 ms (−58.8%)** cumulative against the stage-03 layer
- PCC: **0.999651**, unchanged. The two routes are **bit-identical**, verified on device before
  the change was written
- `tests/pcc/`: **33 passed, 0 failed**

## Measured before it was written

This entry was the one candidate 5 flagged as able to lose, so it was priced with a standalone
benchmark of both whole chains at the real pyramid shapes (`bs=6`, `HW=[22600, 5700, 1450, 375]`,
`heads=8`, `head_dim=32`) rather than from the op table:

| route | wall | ratio | output |
|---|---:|---:|---|
| A — today: padded TILE reshape, then per level TILE slice + transpose + untilize | 39.98 ms | 1.00 | — |
| B — 5d: unpadded reshape, one untilize, free regroup, one ROW_MAJOR permute, per level a row slice | **33.64 ms** | **0.84** | `torch.equal` to A |

Wall clock including dispatch, so read the ratio, not the absolute. 0.84 × the 35.2 ms the profile
attributed to this group predicted about **−5.6 ms**; the profile came back at −6.6 ms. The estimate
in the candidate entry was −10 to −20 ms, so it was roughly 2–3× optimistic — the benchmark is what
kept that from becoming a wasted afternoon on a hoped-for 20 ms.

## What this change was

`value_proj` emits `(bs, H*W, 256)` in TILE; the op wants `(bs*heads, H, W, 32)` in ROW_MAJOR per
level. Splitting the 256 channels into `(heads, head_dim)` while still tiled puts `heads = 8` into a
tiled dimension, where it pads to 32 — so the reshape re-laid out the whole ~92 MB tensor at 4× its
real volume for **21.19 ms**, and the per-level transpose and untilize then ran on the padded shape.

Folding `heads` into the *row* axis instead leaves the padding untouched:

```
(bs, HW, 256)        TILE      value_proj output
(bs, HW*heads, 32)   TILE      same element order, same padding — the row axis absorbs heads
                     ROW_MAJOR one untilize, nothing to pad
(bs, HW, heads, 32)  ROW_MAJOR constant row width (32) — free
(bs, heads, HW, 32)  ROW_MAJOR one permute, and the head axis is now out of the row axis
  per level:  row-range slice on dim 2
  in the op:  reshape to (bs*heads, H, W, 32) — constant row width, free
```

`(bs, HW, 256) → (bs, HW*heads, 32)` is worth being precise about: it does not move an element.
Element `(b, hw, h, d)` sits at flat channel `h*32 + d`, and row `hw*heads + h` column `d` is the
same position in memory. In TILE it is still a re-layout (tiles are assigned differently) but at
1/24 the padded volume of the 4D reshape it replaces.

## Where the time went

| op | stage 09 | stage 10 | Δ |
|---|---:|---:|---:|
| ReshapeViewDeviceOperation | 42.3 ms | **28.0 ms** | **−14.3** |
| TransposeDeviceOperation | 7.5 ms (11 inst) | 2.1 ms (6) | **−5.4** |
| SliceDeviceOperation | 11.4 ms | 11.0 ms | −0.4 |
| UntilizeWithUnpaddingDeviceOperation | 13.8 ms (12) | 13.1 ms (8) | −0.7 |
| PermuteDeviceOperation | 19.9 ms (5) | **33.8 ms (7)** | **+13.9** |
| everything else | 191.9 ms | 192.2 ms | +0.3 |

**The trade is right there in two rows.** The padded reshape and the per-level TILE transposes are
gone; a large ROW_MAJOR permute takes their place. Net −6.6 ms on a 35 ms group — a 19% improvement
to it, not an elimination.

This is the first stage in candidate 5 where the ROW_MAJOR permute won, and it does not contradict
[stage 09](09-head-major-sampling-grid.md)'s rule. The rule says do axis moves in TILE *when the
tensor is already in the shape you need*. Here getting into that shape costs a 4× padded re-layout
of 92 MB, and no TILE transpose is cheap enough to pay for that.

## What is left on this operand

The 33.8 ms of `Permute` is now essentially two ops: the ~19 ms SCA camera permute in
[`tt_spatial_cross_attention.py`](../../tt/tt_spatial_cross_attention.py#L348-L352) and this ~14 ms
head permute. Neither is reachable by a weight reorder —
[candidate 6](../perf_optimization_candidates.md#what-a-weight-reorder-cannot-reach) established
that for the head permute, and the camera permute reorders an input the encoder hands over.

**So candidate 5 is finished on this operand and
[candidate 11](../perf_optimization_candidates.md#candidate-11--absorb-msda-layout-prep) route 1
owns what remains**: an op that accepts TILE input would delete the untilize and the permute
together, which is the whole 47 ms rather than the 6.6 ms a reordering could reach.

## Layer profile now

| Op | inst | ms | % of kernel |
|---|---:|---:|---:|
| MSDAOperation | 5 | 167.9 | 59.9 |
| PermuteDeviceOperation | 7 | 33.8 | 12.1 |
| ReshapeViewDeviceOperation | 17 | 28.0 | 10.0 |
| UntilizeWithUnpaddingDeviceOperation | 8 | 13.1 | 4.7 |
| SliceDeviceOperation | 13 | 11.0 | 3.9 |
| ScatterDeviceOperation | 1 | 10.5 | 3.7 |
| MatmulDeviceOperation | 11 | 4.7 | 1.7 |
| BinaryNgDeviceOperation | 17 | 2.9 | 1.0 |
| TransposeDeviceOperation | 6 | 2.1 | 0.7 |

**`MSDAOperation` is 59.9% of the layer** and has not moved once since stage 04.
