# Stage: 04-flat-sampling-chain

- source commit: [`87e2e9f2de0`](https://github.com/tenstorrent/tt-metal/commit/87e2e9f2de0)
- harness: `tests/perf/test_layer_perf.py`, one encoder layer, N150, Release build
- kernel time: **310.1 ms** (−140.5 ms)
- op-to-op gap: **37.6 ms** (+4.4 ms)
- wall: **347.7 ms** (−136.1 ms, **−28.1%**)
- device ops in the signposted region: **117** (−23)
- PCC gate: **0.999611**, unchanged from stages 02 and 03
- CSV: `generated/profiler/reports/2026_08_27_10_59_09/ops_perf_results_2026_08_27_10_59_09.csv`

**`200×200` spatial cross attention now runs.** See *The OOM* below.

## What this change was

The offsets, the reference-point add and the `[0,1] → [-1,1]` rescale are all
elementwise. None of them needs the `(num_levels, num_points, 2)` axes spelled
out — but they ran on a tensor shaped `(bs*Q*num_heads, num_levels, num_points, 2)`,
whose trailing `(4, 2)` pads to a full 32×32 tile:

```
logical  4 × 2  =    8 elements
tiled   32 × 32 = 1024 elements     128×
```

Every op in the chain carried 128× its own data.

Held flat as `[bs*num_queries, num_heads*num_levels*num_points*2]` the shape
divides the tile exactly — at 100×100 that is `(14976, 256)`, `14976/32 = 468`
and `256/32 = 8`, **zero padding**. The axes are spelled out only afterwards, in
ROW_MAJOR, where degenerate trailing dims cost nothing.

Three supporting pieces:

- **The normalizer becomes one flat row.** In the linear's output order —
  num_heads outermost, then num_levels, num_points, `(x, y)` — the per-level
  scale repeats with period `num_levels*num_points*2`. A single `(1, 256)` row
  broadcasts to the same result as the old `(1, L, 1, 2)` tensor. Built once per
  module, not uploaded per call.
- **The reference points become a repeated row.** Each query's `(D, 2)` block
  applies to every `(head, level, point//D)` triple, so the flat row is that
  block repeated `num_heads*num_levels*(num_points//D)` times.
- **The core attention takes grids already rescaled and ROW_MAJOR.** The `*2 − 1`
  moved into the flat pass, so `multi_scale_deformable_attn_ttnn` no longer does
  it, and its per-level slice no longer needs a layout conversion.

### An approach considered and rejected

Making `num_levels` the outer axis would make the per-level slice contiguous, and
it could be done for free by permuting the `sampling_offsets` weight rows once at
preprocessing. It does not work: the softmax reduces over `num_levels*num_points`
**per head**, which needs head-major grouping, while a contiguous per-level slice
needs level-major. Reconciling them costs a permute — the thing being removed.
Slicing in ROW_MAJOR sidesteps the conflict: a strided read there is address
arithmetic, not 128× the bytes.

## Where the time went

| region | ops | kernel | Δ kernel |
|---|---:|---:|---:|
| SCA — MSDA | 55 | 248.6 ms | **−100.5** |
| TSA — MSDA | 27 | 38.7 ms | **−40.0** |
| SCA — rebatch | 11 | 8.3 ms | −0.1 |
| SCA — scatter-back + normalise | 13 | 12.7 ms | 0.0 |
| FFN | 3 | 1.1 ms | 0.0 |
| TSA — outside MSDA | 3 | 0.2 ms | 0.0 |

TSA loses proportionally more: it runs `num_levels=1` with `Q=10000`, so its
padded chain was the single largest tensor of this shape in the layer.

## The trade

| Op | before | after | Δ |
|---|---:|---:|---:|
| BinaryNgDeviceOperation | 81.1 ms / 19 | **2.3 ms / 19** | **−78.7** |
| ReshapeViewDeviceOperation | 67.7 ms / 20 | 41.8 ms / 14 | −25.9 |
| UntilizeWithUnpadding | 33.6 ms / 20 | 17.5 ms / 12 | −16.2 |
| SliceDeviceOperation | 31.9 ms / 13 | 11.6 ms / 13 | −20.3 |
| RepeatCodegenDeviceOperation | 1.0 ms / 1 | 1.2 ms / 3 | +0.3 |
| PermuteDeviceOperation | 47.0 ms / 19 | 47.5 ms / 19 | +0.5 |
| MSDAOperation | 167.9 ms / 5 | 167.6 ms / 5 | −0.2 |

Same op count for `BinaryNg`, 35× less time — the ops were always cheap, the
padding was not.

Device time spent on padding-carrying ops: **142.1 ms (32%) → 30.9 ms (10%)**,
and the count of such ops drops 49 → 23.

The gap rose 4.4 ms. A first version rebuilt the normalizer row on every call and
cost 23 ms of gap; caching it on the module recovered that, and the remainder is
within run-to-run spread.

## The OOM

`200×200` spatial cross attention used to fail with:

```
TT_FATAL: Out of Memory: Not enough space to allocate 2969567232 B DRAM buffer
```

2.97 GB for a tensor holding **23.2 MB** — the same `(…, 4, 2)` shape at
`Q=40000`, and four of them live at once against 12.85 GB of DRAM. It was
recorded as pre-existing and deselected, and it was: the allocation request was
byte-identical before stages 02 and 03.

Flat and tile-clean, it allocates its 23.2 MB and the test passes. **The whole
`tests/pcc/` directory is 33 passed with nothing deselected.**

## Kernel time by op code

| Op | inst | ms | % |
|---|---:|---:|---:|
| MSDAOperation | 5 | 167.6 | 54.1 |
| PermuteDeviceOperation | 19 | 47.5 | 15.3 |
| ReshapeViewDeviceOperation | 14 | 41.8 | 13.5 |
| UntilizeWithUnpaddingDeviceOperation | 12 | 17.5 | 5.6 |
| SliceDeviceOperation | 13 | 11.6 | 3.7 |
| ScatterDeviceOperation | 1 | 10.5 | 3.4 |
| MatmulDeviceOperation | 11 | 4.7 | 1.5 |
| BinaryNgDeviceOperation | 19 | 2.3 | 0.7 |
| TilizeWithValPaddingDeviceOperation | 5 | 1.8 | 0.6 |
| UnaryDeviceOperation | 2 | 1.5 | 0.5 |
| RepeatCodegenDeviceOperation | 3 | 1.2 | 0.4 |
| SoftmaxDeviceOperation | 2 | 1.2 | 0.4 |
| everything else | 11 | 1.0 | 0.3 |

## Correctness

- Full `tests/pcc/` — **33 passed**, exit 0, no deselection, no OOM.
- Measured PCC ranges 0.994895 to 1.000000 across the suite, every value above
  its own threshold.
- Perf-harness PCC gate 0.999611, identical to stages 02 and 03. The arithmetic
  is unchanged; only the shape it runs on is.

## What this changes about the plan

**Candidate 3 is closed.** Both its root causes are fixed: the degenerate batch
axis in stage 03, the `(num_points, 2)` padding here. What padding remains is
10% of kernel and spread across 23 small ops with no common cause.

**MSDAOperation is now 54% of kernel time and out of reach from Python.** A
bandwidth estimate for one SCA level call — 48 × 2496 × 4 points × 4 bilinear
taps × 32 channels × 2 B ≈ 123 MB, plus ~11 MB of grid, weights and output —
puts the DRAM roof at 0.46 ms against a measured 36 ms. TSA's call gives the same
ratio independently: 88 MB, 0.31 ms roof, 24.35 ms measured.

**The op runs at roughly 1.3% of the memory roof**, or ~4800 cycles per sampled
point per core. The cost being flat across levels whose feature maps differ 64×
in size says the same thing from another angle: nothing about it is bound by the
data it reads.

That is worth raising upstream rather than designing around. The remaining
Python-side work is the 89 ms still in `Permute` and `ReshapeView`, which should
bring the layer to roughly 220 ms; past that the ceiling belongs to the op.
