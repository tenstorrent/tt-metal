# Stage: 02-fused-msda

- source commit: [`828c3315149`](https://github.com/tenstorrent/tt-metal/commit/828c3315149)
- harness: `tests/perf/test_layer_perf.py`, one encoder layer, N150, Release build
- kernel time: **487.4 ms** (−194.1 ms)
- op-to-op gap: **37.9 ms** (−2.9 ms)
- wall: **525.3 ms** (−197.0 ms, **−27.3%**)
- device ops in the signposted region: **140** (−6)
- PCC gate: **0.999611** vs the baseline's 0.999608
- CSV: `generated/profiler/reports/2026_08_27_10_06_35/ops_perf_results_2026_08_27_10_06_35.csv`

## Baseline this is measured against

Not stage 01's numbers. Stage 01 was measured in another workspace on a Debug
build; its op-to-op gap of 218.3 ms is host dispatch cost that a Release build
does not pay. Both figures below come from the same tree, same Release build,
same test, with this stage's commit stashed for the baseline run:

| | baseline | this stage |
|---|---:|---:|
| kernel | 681.5 ms | 487.4 ms |
| gap | 40.8 ms | 37.9 ms |
| wall | 722.3 ms | 525.3 ms |
| device ops | 146 | 140 |

The baseline reproduces stage 01 on both counts that should not move: 146 device
ops, and 681.5 ms of kernel against its 682.0 ms. **Only the gap differs, 40.8 ms
against 218.3 ms** — which is what a Debug build costs in host dispatch, and it
does not touch device time.

**The remaining candidate-1 work is worth far less than stage 01 implies.** At
40.8 ms of total gap, candidates 1b, 1c and 1d together cannot recover more
than that. Kernel is 93% of wall clock.

## What this change was

**`ttnn.experimental.multi_scale_deformable_attn` replaces the `grid_sample`
decomposition.** The op implements the `num_levels == 1` case, so the caller
sums the levels.

The split is exact, not an approximation: `attention_weights` is softmaxed over
`num_levels * num_points` before any sampling happens, so the reduction over
(level, point) is the plain sum of the per-level reductions.

Multi-level was never the blocker stage 01's candidate list treated it as. The
decomposition already looped per level and already built the op's exact input
layouts — `(N, h_in, w_in, D)` and `(N, Q*P, 1, 2)`, both ROW_MAJOR bfloat16.
Only `attn` needed a new target shape, `(N, Q, P)` instead of `(N, 1, Q, L*P)`.

## Where the time went

Per-region, measured iteration only:

| region | ops | kernel | Δ kernel |
|---|---:|---:|---:|
| SCA — MSDA | 70 | 350.1 ms | **−181.5** |
| TSA — MSDA | 35 | 78.7 ms | **−12.6** |
| SCA — rebatch | 11 | 44.2 ms | +0.1 |
| SCA — scatter-back + normalise | 13 | 12.7 ms | 0.0 |
| FFN | 3 | 1.1 ms | 0.0 |
| TSA — outside MSDA | 3 | 0.2 ms | 0.0 |

The entire delta is inside the two MSDA calls. Nothing else moved, which is
what a drop-in replacement should look like.

SCA's MSDA lost 181.5 ms but only 2 device ops. The ops that remain there are
the padded elementwise and layout ops around the sampling, not the sampling
itself — see *What this changes about the plan*.

## The trade

| Op | Δ ms | Δ inst | note |
|---|---:|---:|---|
| GridSampleOperation | **−116.1** | 5 → 0 | absorbed |
| ConcatDeviceOperation | **−115.4** | 3 → 2 | the per-level `stack`; candidate 4, deleted as a side effect |
| ReshapeViewDeviceOperation | −89.1 | 23 → 20 | |
| PermuteDeviceOperation | −22.0 | 27 → 24 | |
| TilizeWithValPadding | −13.5 | 8 → 7 | |
| ReduceDeviceOperation | −4.5 | 2 → 0 | the `sum` over (level, point) |
| FillPadDeviceOperation | −4.3 | 2 → 0 | |
| BinaryNgDeviceOperation | −4.0 | 18 → 19 | one op more, less time: 3 level-sum adds replace a padded `mul` |
| MSDAOperation | **+167.9** | 0 → 5 | |
| UntilizeWithUnpadding | +4.5 | 18 → 20 | `attention_weights` to ROW_MAJOR |
| SliceDeviceOperation | +2.6 | 13 → 17 | per-level `attn` slice |

167.9 ms of fused op replaces 369.2 ms of decomposition. The 115.4 ms concat —
the single most expensive op in the layer at baseline — drops to 0.00 ms.
Candidate 4 needs no separate work.

**MSDAOperation per call**, all on 64 cores:

| call | ms | shape |
|---|---:|---|
| TSA | 24.35 | `num_levels=1`, Q=10000 |
| SCA level 0 | 36.27 | Q=2496 |
| SCA level 1 | 35.94 | Q=2496 |
| SCA level 2 | 35.72 | Q=2496 |
| SCA level 3 | 35.59 | Q=2496 |

The four SCA levels agree to within 0.7 ms despite their `value` tensors
differing 64-fold in size. The op's cost tracks the sample-point count, not the
feature map it samples from.

## Kernel time by op code

| Op | inst | ms | % |
|---|---:|---:|---:|
| MSDAOperation | 5 | 167.9 | 34.4 |
| PermuteDeviceOperation | 24 | 83.7 | 17.2 |
| BinaryNgDeviceOperation | 19 | 81.7 | 16.8 |
| ReshapeViewDeviceOperation | 20 | 67.8 | 13.9 |
| UntilizeWithUnpaddingDeviceOperation | 20 | 33.7 | 6.9 |
| SliceDeviceOperation | 17 | 32.1 | 6.6 |
| ScatterDeviceOperation | 1 | 10.5 | 2.2 |
| MatmulDeviceOperation | 11 | 4.7 | 1.0 |
| UnaryDeviceOperation | 2 | 1.6 | 0.3 |
| SoftmaxDeviceOperation | 2 | 1.2 | 0.3 |
| RepeatCodegenDeviceOperation | 1 | 1.0 | 0.2 |
| TilizeWithValPaddingDeviceOperation | 7 | 0.8 | 0.2 |
| LayerNormDeviceOperation | 3 | 0.3 | 0.1 |
| EmbeddingsDeviceOperation | 2 | 0.3 | 0.1 |
| CloneOperation | 2 | 0.1 | 0.0 |
| TilizeDeviceOperation | 1 | 0.1 | 0.0 |
| UntilizeCodegenDeviceOperation | 1 | 0.1 | 0.0 |
| ConcatDeviceOperation | 2 | 0.0 | 0.0 |

`tt-perf-report` marks all 11 matmuls `SLOW` with "place input 0 in L1". Ignore
it — 4.7 ms total, 1.0% of kernel.

## What this changes about the plan

**Candidate 3 is now the second lever, and it is the same defect as the 200×200
OOM.** `BinaryNg` at 81.7 ms and much of the `Permute` / `ReshapeView` time is
spent on tensors whose trailing dims are `(num_points, 2)` = `(4, 2)`, tile-padded
to `(32, 32)` — 128× the logical element count.

The same padding is what makes 200×200 unrunnable. At that BEV size the padded
form of `(bs·Q·nh, L, P, 2)` is 2,969,567,232 B against a 23.2 MB logical
footprint, and four of them are live at once — the whole 12.85 GB of DRAM. The
allocation fails at `tt_ms_deformable_attention.py:74`. Verified pre-existing:
identical byte count, identical allocator state, with and without this stage.

Fixing the layout should recover kernel time and unblock 200×200 in one change.
Candidate 3's "~60 ms" estimate was made before the concat and grid_sample
dominated ops were removed; the padded ops are now a larger share.

**Revised ordering:** 3 → 2 (further MSDA work, if the op leaves anything on the
table) → the rest of 1 as cleanup → 5. Candidate 4 is closed.

## Correctness

- SCA PCC suite: **7 passed**. `200×200` fails on the pre-existing OOM above,
  confirmed byte-identical on the baseline.
- Layer + encoder PCC suites: **9 passed**, exit 0.
- Perf-harness PCC gate: 0.999611, against the baseline's 0.999608 on the same
  tree.

Measured PCC values across the SCA suite range 0.999927 to 1.000000, all above
their thresholds.
