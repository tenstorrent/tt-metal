# Stage: 06-sfpu-geometry

- source commits: [`86e55413065`](https://github.com/tenstorrent/tt-metal/commit/86e55413065),
  [`192cdf21916`](https://github.com/tenstorrent/tt-metal/commit/192cdf21916)
- harness: `tests/perf/test_layer_perf.py`, one encoder layer, N150, Release build
- kernel time: **125.3 ms** (−138.3 ms)
- op-to-op gap: **44.1 ms** (+2.3 ms)
- wall: **169.4 ms** (−136.0 ms, **−44.5%**)
- device ops in the signposted region: **113** (unchanged)
- PCC gate: **0.999609**, unchanged from stage 05's 0.999611
- CSV: `generated/profiler/reports/2026_08_28_09_16_.../ops_perf_results_*.csv`

## What this change was

**The sampling geometry moved from the reader to the SFPU.** Per point and per
block of 32 queries, the compute kernel now derives

```
px   = (gx + 1) * scale - shift        scale/shift carry the align_corners
x0   = floor(px)                       variant, so the kernel has no branch
dx   = px - x0
w_c  = corner(dx, dy) * attn           for the four corners
```

and hands the reader `x0`/`y0` as bf16. The reader decodes them with integer
shifts, bounds-tests them and turns them into page indices — all integer. Not one
float operation is left in the reader.

Queries occupy tile rows and only column 0 carries meaning. That looks wasteful
until you notice the reduction's scalar tiles were already 32 useful values out
of 1024: `mul_tiles_bcast<COL>` consumes one scalar per row by construction.

## Why the op was never what it looked like

Stage 05 recorded `MSDAOperation` at 167.6 ms, 78× above its DRAM roof, and
concluded the fused kernel needed optimising. That was the wrong read.

`CB-COMPUTE-WAIT-FRONT` measured **36.1 ms on a 36.0 ms call** — the compute
kernel was idle for the entire op, waiting on the reader. Zones inside the reader
put 20.3 ms of that in the coordinate maths and 5.1 ms in the scalar tile, at a
uniform **~140 cycles per float operation**: soft-float emulation on a dataflow
core with no FPU, while the SFPU sat unused.

The op's measured duration was the reader's arithmetic. Moving that arithmetic
onto the vector unit collapsed it:

| call | before | after |
|---|---:|---:|
| TSA | 24.35 ms | **4.25 ms** |
| SCA level 0 | 36.04 ms | **6.27 ms** |
| SCA level 1 | 35.94 ms | **6.26 ms** |
| SCA level 2 | 35.72 ms | **6.26 ms** |
| SCA level 3 | 35.59 ms | **6.43 ms** |
| **total** | **167.6 ms** | **29.5 ms** |

Nothing about the sampling kernel changed. It was never bound by memory or by
grid_sample.

## Two defects this cost, and what they teach

**Out-of-bounds corners lost their mask.** The reader used to fold the bounds
test into the scalar it built, so an out-of-bounds corner contributed zero and
the stale input row behind it never mattered. The compute kernel builds that
scalar now and knows nothing about bounds, so stale rows were multiplied by live
weights. Tiny configs returned values around 1e36 at PCC 0; every other config
carried a high-error ratio near 0.55 that the PCC gate was loose enough to pass.

That ratio was visible from the first run and read as a secondary metric. It was
the defect, in every configuration, the whole time.

**The destination register defaulted to 16 bits.** `px` reaches the feature
map's extent, and bf16's ulp at 200 is 1.0 — so on a 200×200 map `floor(px)`
rounded to the wrong integer and the fraction collapsed to zero, degrading
bilinear sampling to nearest-neighbour. The damage scaled with map size, which is
exactly what the data showed:

| feature map | before | after |
|---|---:|---:|
| 50×50 | 0.999787 | 0.999974 |
| 100×100 | 0.999414 | 0.999951 |
| 200×200 | **0.996156** | 0.999823 |

`fp32_dest_acc_en` fixes it and is *faster*: `MSDAOperation` 35.9 → 29.5 ms and
gap 65.1 → 44.1 ms. The geometry holds three tiles at once, inside the four a
32-bit destination allows.

## Where the time went

| region | ops | kernel | Δ kernel |
|---|---:|---:|---:|
| SCA — MSDA | 50 | 87.2 ms | **−118.4** |
| TSA — MSDA | 28 | 15.1 ms | **−20.1** |
| SCA — scatter-back + normalise | 13 | 12.8 ms | +0.1 |
| SCA — rebatch | 11 | 8.4 ms | 0.0 |
| FFN | 3 | 1.1 ms | 0.0 |
| TSA — outside MSDA | 3 | 0.2 ms | 0.0 |

## Kernel time by op code

| Op | inst | ms | % |
|---|---:|---:|---:|
| MSDAOperation | 5 | 29.5 | 23.5 |
| ReshapeViewDeviceOperation | 16 | 26.2 | 20.9 |
| PermuteDeviceOperation | 16 | 25.6 | 20.5 |
| SliceDeviceOperation | 13 | 11.3 | 9.0 |
| ScatterDeviceOperation | 1 | 10.5 | 8.4 |
| UntilizeWithUnpaddingDeviceOperation | 9 | 8.6 | 6.8 |
| MatmulDeviceOperation | 11 | 4.8 | 3.9 |
| BinaryNgDeviceOperation | 19 | 2.1 | 1.7 |
| everything else | 31 | 6.7 | 5.3 |

## Correctness

- Full `tests/pcc/` — **33 passed**, exit 0, nothing deselected.
- Perf-harness PCC gate 0.999609.
- Constraint the geometry adds: `x0`/`y0` cross as bf16, exact for integers up to
  256, so `h_in` and `w_in` must stay at or below that. Asserted in the reader.
  BEVFormer's largest feature map is 200×113.

## What this changes about the plan

**Layout plumbing is now the largest item.** `Reshape` + `Permute` + `Slice` +
`Untilize` is **71.7 ms, 57% of kernel**, against the fused op's 23.5%. It was in
the op's shadow for five stages.

It is not shapeless. Read against the profile it falls into two groups:

- **~24 ms turning `value` into per-level heads** — untilize, split embed_dims
  into (num_heads, head_dim), permute heads ahead of the spatial axis, slice the
  level. This is the multi-scale deformable analogue of `nlp_create_qkv_heads`.
- **~31 ms preparing grid and attn** — reshape, untilize, head-major permute,
  per-level slice.

`ttnn/cpp/ttnn/operations/experimental/transformer/` already carries **19**
head-reshaping ops, several of them model-specific (`_vit`, `_segformer`,
`_falcon7b`, `_decode`). The precedent for fusing exactly this kind of plumbing
is established; what is missing is the deformable-attention member of that
family. See
[candidate 7](../perf_optimization_candidates.md#candidate-7--an-msda-head-reshape-op).

Two things worth trying first, both cheap and both Python-side: the largest
single reshape (`1x1x119808x32 → 1x1x1916928x2`, 7.4 ms) is a pure ROW_MAJOR view
that ttnn materialises, and the largest permute (9.96 ms on 69.4 MB) is
bandwidth-bound rather than overhead-bound. Stage 05 measured both as a wash, but
it measured them against a kernel twice as slow.
