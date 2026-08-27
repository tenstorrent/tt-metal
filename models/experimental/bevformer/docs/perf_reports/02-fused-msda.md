# Stage: 02-fused-msda

- source commit: [`828c3315149`](https://github.com/tenstorrent/tt-metal/commit/828c3315149)
- harness: `tests/perf/test_layer_perf.py`, one encoder layer, N150, Release build
- kernel time: **557.6 ms** (−191.2 ms)
- op-to-op gap: **37.9 ms** (−2.9 ms)
- wall: **595.5 ms** (−194.1 ms, **−24.6%**)
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
| kernel | 748.8 ms | 557.6 ms |
| gap | 40.8 ms | 37.9 ms |
| wall | 789.6 ms | 595.5 ms |
| device ops | 146 | 140 |

Op count matches stage 01's 146 exactly, so the baseline is the same code path.
Kernel time does not (682.0 vs 748.8 ms), and that gap is unexplained — device
FW duration should not depend on the host build type.

**The remaining candidate-1 work is worth far less than stage 01 implies.** At
40.8 ms of total gap, candidates 1b, 1c and 1d together cannot recover more
than that. Kernel is 94% of wall clock.

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
| SCA — MSDA | 70 | 410.3 ms | **−177.5** |
| TSA — MSDA | 35 | 86.1 ms | **−13.7** |
| SCA — rebatch | 11 | 45.4 ms | +0.1 |
| SCA — scatter-back + normalise | 13 | 13.2 ms | 0.0 |
| FFN | 3 | 1.7 ms | 0.0 |
| TSA — outside MSDA | 3 | 0.2 ms | 0.0 |

The entire delta is inside the two MSDA calls. Nothing else moved, which is
what a drop-in replacement should look like.

SCA's MSDA lost 177.5 ms but only 2 device ops. The ops that remain there are
the padded elementwise and layout ops around the sampling, not the sampling
itself — see *What this changes about the plan*.

## The trade

| Op | Δ ms | Δ inst | note |
|---|---:|---:|---|
| GridSampleOperation | **−118.3** | 5 → 0 | absorbed |
| ConcatDeviceOperation | **−115.5** | 3 → 2 | the per-level `stack`; candidate 4, deleted as a side effect |
| ReshapeViewDeviceOperation | −93.6 | 23 → 20 | |
| PermuteDeviceOperation | −38.7 | 27 → 24 | |
| TilizeWithValPadding | −16.8 | 8 → 7 | |
| ReduceDeviceOperation | −5.2 | 2 → 0 | the `sum` over (level, point) |
| FillPadDeviceOperation | −4.8 | 2 → 0 | |
| TransposeDeviceOperation | −0.5 | 2 → 0 | |
| MSDAOperation | **+169.1** | 0 → 5 | |
| BinaryNgDeviceOperation | +17.7 | 18 → 19 | the 3 level-sum adds |
| UntilizeWithUnpadding | +10.3 | 18 → 20 | `attention_weights` to ROW_MAJOR |
| SliceDeviceOperation | +4.8 | 13 → 17 | per-level `attn` slice |

169.1 ms of fused op replaces 393.4 ms of decomposition. The 115.5 ms concat —
the single most expensive op in the layer at baseline — drops to 0.01 ms.
Candidate 4 needs no separate work.

**MSDAOperation per call**, all on 64 cores:

| call | ms | shape |
|---|---:|---|
| TSA | 24.64 | `num_levels=1`, Q=10000 |
| SCA level 0 | 36.48 | Q=2496 |
| SCA level 1 | 36.19 | Q=2496 |
| SCA level 2 | 35.92 | Q=2496 |
| SCA level 3 | 35.82 | Q=2496 |

The four SCA levels agree to within 0.7 ms. No pathological level.

## Kernel time by op code

| Op | inst | ms | % |
|---|---:|---:|---:|
| MSDAOperation | 5 | 167.9 | 34.4 |
| PermuteDeviceOperation | 24 | 83.6 | 17.2 |
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
