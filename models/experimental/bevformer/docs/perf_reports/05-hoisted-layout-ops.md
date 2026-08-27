# Stage: 05-hoisted-layout-ops

- source commit: [`3ebd77d25c0`](https://github.com/tenstorrent/tt-metal/commit/3ebd77d25c0)
- harness: `tests/perf/test_layer_perf.py`, one encoder layer, N150, Release build
- kernel time: **263.6 ms** (−46.5 ms)
- op-to-op gap: **41.8 ms** (+4.2 ms)
- wall: **305.4 ms** (−42.3 ms, **−12.2%**)
- device ops in the signposted region: **113** (−4)
- PCC gate: **0.999611**, unchanged since stage 02
- CSV: `generated/profiler/reports/2026_08_27_11_19_45/ops_perf_results_2026_08_27_11_19_45.csv`

## What this change was

Two layout ops that ran once per level and cost more than the work they did.

### The grid permute was overhead, not bandwidth

`multi_scale_deformable_attn_ttnn` permuted heads ahead of queries inside the
level loop. Per call:

```
2496 x 8 x 4 x 2  bf16  =  320 KB   in  5.1 ms   =  0.37 GB/s
```

Four of them, plus one for the attention weights. For scale, the permute that
folds the camera axis in the same layer moves **92.6 MB in 1.80 ms — 51 GB/s**.
These were running two orders of magnitude below that, so the cost was
per-call overhead, not the bytes.

`sampling_grids` now leaves `TTMSDeformableAttention.forward` head-major, via one
4-D permute over the whole tensor. The level slice is then the fused op's grid up
to a reshape, and four calls become one.

### The head split ran on a tiled tensor

`value.reshape(bs, num_keys, num_heads, head_dim)` splits `embed_dims` 256 into
`(8, 32)`. Tiled, `num_heads = 8` pads to a full tile:

```
1 x 6 x 30144 x 256  ->  1 x 180750 x 8->32 x 32     4x     21.1 ms
```

And the core attention untilized every level afterwards regardless. Untilizing
once before the split makes the split a view and deletes four per-level
conversions.

## Where the time went

| region | ops | kernel | Δ kernel |
|---|---:|---:|---:|
| SCA — MSDA | 50 | 205.6 ms | **−43.0** |
| TSA — MSDA | 28 | 35.2 ms | **−3.5** |
| SCA — rebatch | 11 | 8.4 ms | +0.1 |
| SCA — scatter-back + normalise | 13 | 12.7 ms | 0.0 |
| FFN | 3 | 1.1 ms | 0.0 |
| TSA — outside MSDA | 3 | 0.2 ms | 0.0 |

## The trade

| Op | before | after | Δ |
|---|---:|---:|---:|
| PermuteDeviceOperation | 47.5 ms / 19 | **25.5 ms / 16** | **−22.0** |
| ReshapeViewDeviceOperation | 41.8 ms / 14 | **26.2 ms / 16** | **−15.6** |
| UntilizeWithUnpadding | 17.5 ms / 12 | **8.5 ms / 9** | **−8.9** |

Device time on padding-carrying ops: **30.9 ms (10%) → 16.5 ms (6%)**, over 23
ops down to 14.

The gap rose 4.2 ms with no new host work in the path; treating it as run-to-run
spread until a measurement says otherwise.

## An approach measured and reverted

The same hoist applied to the `value` head permute, together with dropping the
6-D `sampling_grids` reshape in favour of a column-range slice, measured
**263.6 → 263.5 ms**: nothing, and +14.9 ms of gap.

The reasoning that worked for the grids does not transfer:

```
grid permute:    320 KB / 5.1 ms  =  0.37 GB/s    per-call overhead
value permute:  69.4 MB / 9.8 ms  =  7.1 GB/s     bandwidth
```

The value permute genuinely moves 92.6 MB, and that costs the same in one call
as in four. Dropping the 6-D reshape traded 3.1 ms of `Slice` for 2.7 ms of
`ReshapeView`. Both reverted — the simpler code was also the faster one.

Worth recording because the shape of the two ops looks identical in the source.
Only the byte count separates them, and only a measurement shows which regime an
op is in.

## Kernel time by op code

| Op | inst | ms | % |
|---|---:|---:|---:|
| MSDAOperation | 5 | 167.8 | 63.7 |
| ReshapeViewDeviceOperation | 16 | 26.2 | 9.9 |
| PermuteDeviceOperation | 16 | 25.5 | 9.7 |
| SliceDeviceOperation | 13 | 11.6 | 4.4 |
| ScatterDeviceOperation | 1 | 10.5 | 4.0 |
| UntilizeWithUnpaddingDeviceOperation | 9 | 8.5 | 3.2 |
| MatmulDeviceOperation | 11 | 4.7 | 1.8 |
| BinaryNgDeviceOperation | 19 | 2.1 | 0.8 |
| TilizeWithValPaddingDeviceOperation | 5 | 1.8 | 0.7 |
| UnaryDeviceOperation | 2 | 1.5 | 0.6 |
| RepeatCodegenDeviceOperation | 3 | 1.2 | 0.5 |
| SoftmaxDeviceOperation | 2 | 1.2 | 0.5 |
| everything else | 11 | 1.0 | 0.4 |

## Correctness

- Full `tests/pcc/` — **33 passed**, exit 0, nothing deselected.
- Perf-harness PCC gate 0.999611, identical to stages 02 through 04.

## What this changes about the plan

**`MSDAOperation` is 64% of kernel time.** Everything else in the layer together
is 96 ms.

The residual layout work is down to ~52 ms across `Permute`, `ReshapeView`,
`Slice` and `Untilize`, and it no longer has a common cause — the ops that
remain are different shapes for different reasons, the largest being a
bandwidth-bound 92.6 MB permute at 9.8 ms. Reading the profile op by op turned
up no further target of the kind stages 03–05 fixed.

That closes the Python-side layout work at roughly the point predicted from
stage 04 (~220 ms estimated, 263.6 ms reached; the estimate assumed the value
permute would fold, which it does not). The next order of magnitude is
[candidate 6](../perf_optimization_candidates.md#candidate-6--the-fused-msda-op-itself),
and it is a C++ kernel, not a Python restructure.
