# Stage: 04-fused-msda

| | |
|---|---|
| commit | [`2bc69b90e3c`](https://github.com/tenstorrent/tt-metal/commit/2bc69b90e3c1472e29428f40542bac47847cbf36) |
| candidate | [2](../perf_optimization_candidates.md#candidate-2--fused-msda) |
| config | `nuscenes_base`, 100×100, N150 |
| profile | **489.5 ms kernel**, 129 ops (+8), CSV `generated/profiler/reports/2026_08_27_23_24_32/` (**deleted**, see below) |
| delta | **−191.6 ms kernel (−28.1%)** vs [stage 03](03-constant-uploads-cached.md) re-measured in the same session (`2026_08_27_23_03_17`: 681.1 ms, 121 ops) |
| PCC | **0.999611** (baseline 0.999608, gate 0.997) |
| suite | `tests/pcc/` **32 passed, 1 failed** in 242.8 s — the failure is pre-existing, see below |

Three caveats on the numbers:

- **No wall-clock figure.** This report originally led with "−190.6 ms layer wall (−27.5%)", from
  14.0 ms of gap against the stage-03 re-measure's 13.0 ms. [Stage 05](05-offset-normalizer-folded.md)
  then profiled identical code twice and got 93.4 and 151.2 ms of gap while kernel held to 0.1 ms,
  which retires the gap column as a measurement ([PERF.md](../PERF.md#the-gap-column-is-not-reliable)).
  The wall figures were removed rather than restated; the kernel figures reproduce.
- **No encoder figure.** Stages 02/03 quote a median over 11 timed iterations, but the encoder
  harness in the tree runs `DEVICE_PERF_ITERS = 1` with no timing loop, so that methodology is not
  reproducible from the repo. Only the layer harness was run.
- **The cited CSVs no longer exist.** Both `2026_08_27_23_24_32/` and the stage-03 re-measure
  `2026_08_27_23_03_17/` were removed from `generated/` between sessions, so neither can be
  re-audited — the same problem [PERF.md](../PERF.md) records for stage 01. The tables below are what
  survives. **Copy cited CSVs out of `generated/` at run time.**

## What changed

The hand-rolled `multi_scale_deformable_attn_ttnn` decomposition is replaced by
`ttnn.experimental.multi_scale_deformable_attn`
([PR #52380](https://github.com/tenstorrent/tt-metal/pull/52380)), which fuses `grid_sample` with the
weighted sum over sampling points into one kernel.

A new `_fused_msda_level()` shapes one level for the op — value → `(N, H, W, D)`, grid →
`(N, Q*P, 1, 2)`, attn → `(N, Q, P)`, all ROW_MAJOR bfloat16 INTERLEAVED, which the op enforces with
`TT_FATAL` rather than converting. The 47-line per-level `grid_sample` chain, the `ttnn.stack`, the
`mul` and the `sum` are deleted outright.

**Multi-level is exact and stays on device.** The candidates doc assumed SCA would need a host-side
weighted sum. It does not: `attention_weights` is softmaxed jointly over `L*P` and thereafter only
summed, so `sum_{l,p} w·v == sum_l ( sum_p w·v )`. Each fused call computes one inner sum and the
levels combine with an L-way `ttnn.add`. No renormalization, no round-trip, no approximation — PCC
moved by 3e-6.

## Where the time went

| op | stage 03 | stage 04 | Δ | | region | stage 03 | stage 04 | Δ |
|---|---:|---:|---:|---|---|---:|---:|---:|
| GridSample | 116.0 | **0.0** | −116.0 | | TSA (`L=1`) | 90.7 | 78.7 | −12.0 |
| Concat | 115.5 | **0.0** | −115.5 | | SCA (`L=4`) | 531.9 | 352.0 | −179.9 |
| ReshapeView | 157.0 | 77.1 | −79.9 | | | | | |
| Permute | 105.4 | 62.5 | −43.0 | | | | | |
| Slice | 29.1 | 13.2 | −15.9 | | | | | |
| FillPad | 4.7 | 0.0 | −4.7 | | | | | |
| Reduce | 4.5 | 0.0 | −4.5 | | | | | |
| BinaryNg | 85.7 | 81.5 | −4.2 | | | | | |
| TilizeWithValPadding | 14.6 | 17.1 | +2.6 | | | | | |
| Transpose | 0.1 | 7.9 | +7.8 | | | | | |
| UntilizeWithUnpadding | 28.9 | 42.4 | +13.5 | | | | | |
| **MSDAOperation** | — | **167.6** | +167.6 | | | | | |

## The fused op is not a faster sampler

| | old `GridSample` | new `MSDAOperation` |
|---|---:|---:|
| TSA | 16.8 ms | **24.4 ms** |
| SCA (4 levels) | 99.2 ms | **143.3 ms** |

**The op is 45% more expensive than the sampling it subsumes.** Every bit of the −191.6 ms comes from
deleting the tail it makes unnecessary — the `stack` (115.5 ms), the reshape after it (74.9 ms), and
the `mul`/`sum`/`FillPad` reduction. VADv2 reaches the same conclusion from the other side:
`models/experimental/vadv2/tt/tt_utils.py` gates its fused path behind `N*Q >= 1024` because below
that the decomposition wins. No such threshold was added here — copying an unmeasured constant from
another model would be worse than omitting it, and BEVFormer's smallest tested shape (30×30 TSA,
`N*Q = 7200`) clears it 7×.

## The 200×200 SCA failure is pre-existing

`test_spatial_cross_attention_forward[…-1-200-200-…]` fails with
`TT_FATAL Out of Memory: Not enough space to allocate 2969567232 B DRAM buffer across 12 banks`.
**Verified not a regression:** stashing this change reproduces the identical failure, byte-for-byte
the same allocation, on the stage-03 tree (jobs `012` vs `013`).

Same DRAM ceiling that [DEAD_ENDS 3](DEAD_ENDS.md#3-a-static-bound-on-max_len) hit when bounding
`max_len`. The candidates doc hoped candidate 2 would relieve it — **it does not**; the allocation is
in the sampling-location math upstream of the op. (It was eventually cleared by
[stage 07](07-sampling-grid-in-row-major.md), which is what did touch that math.)

The encoder tests matter most here: 5/5 pass across `nuscenes_base`/`_fast`/`_tiny` and
`carla_base`/`_tiny`, exercising all six layers at PCC ≥ 0.995–0.997, so the per-level fused path and
the cross-level `ttnn.add` hold over the full stack, not just in the single-layer harness.

## What is left

SCA is 352.0 ms: **105.9 ms pre-core** (linears, softmax, the 23.9 ms `div` by `offset_normalizer`,
the reference-point add) and **246.1 ms core** (4 × `MSDAOperation` = 143.3 ms, plus ~103 ms of
per-level layout prep).

1. **`MSDAOperation`, 143.3 ms** — the largest op in the layer, and slower per sample than the
   `GridSample` it replaced. Upstream question: worth a standalone microbenchmark and, if it
   reproduces, an issue against the op.
2. **Per-level layout prep, ~103 ms** — `Untilize`/`Transpose`/`Slice`/`Permute`/`Tilize` ×4. The
   tilize↔untilize churn around each call is not obviously necessary.
3. **The `div`, 23.9 ms** — [candidate 3](../perf_optimization_candidates.md#candidate-3--tile-padding-waste),
   removable outright by folding the normalizer into the `sampling_offsets` weight.

## Effect on the backlog

- **[Candidate 4](../perf_optimization_candidates.md#candidate-4--the-msda-concat) is deleted, not
  deferred.** The 113.5 ms `Concat` is gone from the profile. Sequencing 4 before 2, as the ordering
  section had it, would have been wasted work.
- **[Candidate 3](../perf_optimization_candidates.md#candidate-3--tile-padding-waste) rescopes from
  ~60 ms to ~24 ms** — its two `Permute` sites were inside the decomposition; only the `div` survives.
- **1b and [candidate 9](../perf_optimization_candidates.md#candidate-9--trace-capture) are
  unchanged.** The DRAM ceiling that sank 1b is still there, as the 200×200 failure shows.
