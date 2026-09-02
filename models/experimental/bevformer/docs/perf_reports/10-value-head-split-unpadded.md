# Stage: 10-value-head-split-unpadded

| | |
|---|---|
| commit | [`7820f325bd8`](https://github.com/tenstorrent/tt-metal/commit/7820f325bd86cbca8dfcbe27cf460204c8d9773c) |
| candidate | [5d](../perf_optimization_candidates.md#5d-value-head-split-without-the-padding) |
| config | `nuscenes_base`, 100×100, N150 |
| profile | **280.2 ms kernel**, 106 ops (−6), CSV `generated/profiler/reports/2026_09_02_11_49_26/` |
| delta | **−6.6 ms kernel (−2.3%)** vs [stage 09](09-head-major-sampling-grid.md)'s 286.8 ms; **−400.6 ms (−58.8%)** cumulative from stage 03 |
| PCC | **0.999651**, unchanged — the two routes are **bit-identical**, verified on device before the change was written |
| suite | `tests/pcc/` **33 passed, 0 failed** |

## Measured before it was written

Candidate 5 flagged this as the one item that could lose, so both whole chains were benchmarked at
the real pyramid shapes (`bs=6`, `HW=[22600, 5700, 1450, 375]`, `heads=8`, `head_dim=32`) instead of
priced off the op table:

| route | wall | ratio | output |
|---|---:|---:|---|
| A — today: padded TILE reshape, then per level TILE slice + transpose + untilize | 39.98 ms | 1.00 | — |
| B — 5d: unpadded reshape, one untilize, free regroup, one ROW_MAJOR permute, per level a row slice | **33.64 ms** | **0.84** | `torch.equal` to A |

Wall clock including dispatch, so read the ratio, not the absolute. 0.84 × the 35.2 ms the profile
attributed to this group predicted **−5.6 ms**; the profile came back at −6.6. The candidate entry's
own guess was −10 to −20 ms, so it was 2–3× optimistic — the benchmark is what kept that from becoming
a wasted afternoon.

## What changed

`value_proj` emits `(bs, H*W, 256)` in TILE; the op wants `(bs*heads, H, W, 32)` in ROW_MAJOR per
level. Splitting the 256 channels into `(heads, head_dim)` while still tiled puts `heads = 8` into a
tiled dimension where it pads to 32 — so the reshape re-laid out the whole ~92 MB tensor at 4× its
real volume for **21.19 ms**, and the per-level transpose and untilize then ran on the padded shape.

Folding `heads` into the *row* axis leaves the padding untouched:

```
(bs, HW, 256)        TILE      value_proj output
(bs, HW*heads, 32)   TILE      same element order, same padding — the row axis absorbs heads
                     ROW_MAJOR one untilize, nothing to pad
(bs, HW, heads, 32)  ROW_MAJOR constant row width (32) — free
(bs, heads, HW, 32)  ROW_MAJOR one permute, head axis now out of the row axis
  per level:  row-range slice on dim 2
  in the op:  reshape to (bs*heads, H, W, 32) — constant row width, free
```

`(bs, HW, 256) → (bs, HW*heads, 32)` does not move an element: element `(b, hw, h, d)` sits at flat
channel `h*32 + d`, and row `hw*heads + h` column `d` is the same position in memory. In TILE it is
still a re-layout (tiles are assigned differently) but at 1/24 the padded volume of the 4D reshape it
replaces.

## Where the time went

| op | stage 09 | stage 10 | Δ |
|---|---:|---:|---:|
| ReshapeView | 42.3 ms | **28.0 ms** | **−14.3** |
| Transpose | 7.5 ms (11 inst) | 2.1 ms (6) | **−5.4** |
| UntilizeWithUnpadding | 13.8 ms (12) | 13.1 ms (8) | −0.7 |
| Slice | 11.4 ms | 11.0 ms | −0.4 |
| Permute | 19.9 ms (5) | **33.8 ms (7)** | **+13.9** |
| everything else | 191.9 ms | 192.2 ms | +0.3 |

**The trade is in two rows.** The padded reshape and the per-level TILE transposes are gone; a large
ROW_MAJOR permute takes their place. Net −6.6 ms on a 35 ms group — a 19% improvement to it, not an
elimination.

This is the first stage where the ROW_MAJOR permute won, and it does not contradict
[stage 09](09-head-major-sampling-grid.md)'s rule. The rule says do axis moves in TILE *when the
tensor is already in the shape you need*; here reaching that shape costs a 4×-padded re-layout of
92 MB, and no TILE transpose is cheap enough to pay for it.

## What is left on this operand

The 33.8 ms of `Permute` is two ops: the ~19 ms SCA camera permute
([sca:348-352](../../tt/tt_spatial_cross_attention.py#L348-L352)) and this ~14 ms head permute.
Neither is reachable by a weight reorder —
[candidate 6](../perf_optimization_candidates.md#why-the-weight-reorder-is-dead) establishes that, and
the camera permute reorders an input the encoder hands over (it is
[6a](../perf_optimization_candidates.md#6a-hoist-the-sca-camera-permute-out-of-the-layer-loop), a
hoist, since the tensor is layer-invariant).

**So candidate 5 is finished on this operand and
[candidate 11](../perf_optimization_candidates.md#candidate-11--absorb-msda-layout-prep) route 1 owns
what remains**: an op accepting TILE input would delete the untilize and the permute together — ~47 ms
against the 6.6 ms reordering could reach.

## Layer profile now

| Op | inst | ms | % | | Op | inst | ms | % |
|---|---:|---:|---:|---|---|---:|---:|---:|
| MSDAOperation | 5 | 167.9 | 59.9 | | Scatter | 1 | 10.5 | 3.7 |
| Permute | 7 | 33.8 | 12.1 | | Matmul | 11 | 4.7 | 1.7 |
| ReshapeView | 17 | 28.0 | 10.0 | | BinaryNg | 17 | 2.9 | 1.0 |
| UntilizeWithUnpadding | 8 | 13.1 | 4.7 | | Transpose | 6 | 2.1 | 0.7 |
| Slice | 13 | 11.0 | 3.9 | | | | | |

**`MSDAOperation` is 59.9% of the layer** and has not moved once since stage 04.
