# Removing external special rounding

Plot: `sdpa_pareto_no_rounding.png` (3230 × 1900 pixels).

## Scope

Only the current E_bf16/E_bfp8/E_bfp4 recipes require external Q/K/V rounding.
The canonical `device_attention.py::prepare` passes A/B/C/D inputs through
unchanged. Those four therefore have no distinct no-preprocessing datapoint.
Internal rounding, exponentials, reciprocal, compensation, and all other
compute arithmetic remain untouched here, including internal arithmetic in C/D.

| Recipe | Previous external preparation | Plain-input counterpart |
| --- | --- | --- |
| E_bf16 | Q RNE7, K/V RNE5, all stored BF16 | Original BF16 Q/K/V, zero preparation ops |
| E_bfp8 | Q RNE7; K/V RNE5 then BFP8 pack | Original BF16 Q; ordinary device `ttnn.typecast` of K/V to BFP8 |
| E_bfp4 | Q RNE7; K/V custom final-grid RNE with saturation | Original BF16 Q; ordinary device `ttnn.typecast` of K/V to BFP4 |

RNE bit counts include the leading significand bit. Plain means no custom
rounding, not that the hardware performs no rounding or format conversion.
Packed KV still needs to exist in the specified format: either supplied by an
upstream producer or converted with a standard cast. This tests the device
typecast route specifically, not every possible producer/host packing route.
The current typecast implementation selects precise BFP8 packing automatically
for BFP8 output, and its default packing mode for BFP4. We did not override
these standard settings. BF16-to-packed casts use BF16 destination internally
in this standard path. See `ttnn/cpp/ttnn/operations/copy/typecast/typecast.cpp`.

## Results

All broad-suite errors worsen without special rounding: 18/18 cases for each
of the three variants. Medians below are medians across cases, not per-token
errors or confidence estimates.

| Recipe | Broad median L2, prepared → plain | Relative increase in median | Broad plain L2 range | Normal 256K L2, prepared → plain |
| --- | ---: | ---: | ---: | ---: |
| E_bf16 | 3.0213% → 4.9775% | +64.7% | 2.6432–13.7621% | 3.5853% → 5.5550% |
| E_bfp8 | 3.0868% → 3.8790% | +25.7% | 1.8853–13.4612% | 3.6462% → 4.3188% |
| E_bfp4 | 16.2753% → 31.6663% | +94.6% | 20.1363–59.3051% | 16.4053% → 32.1033% |

For normal inputs, by KV length:

| Recipe | 4K prepared → plain L2 | 32K prepared → plain L2 | 256K prepared → plain L2 |
| --- | ---: | ---: | ---: |
| E_bf16 | 2.9684% → 5.4163% | 3.0742% → 5.3957% | 3.5853% → 5.5550% |
| E_bfp8 | 3.0297% → 4.0615% | 3.1438% → 4.0488% | 3.6462% → 4.3188% |
| E_bfp4 | 16.6924% → 33.2214% | 16.7677% → 33.1078% | 16.4053% → 32.1033% |

Separate common-mode stress at 32K KV:

| Recipe | Q +32 prepared → plain L2 | K +32 prepared → plain L2 | V +32 prepared → plain L2 |
| --- | ---: | ---: | ---: |
| E_bf16 | 1.7491% → 3.2624% | 46.9110% → 41.1104% | 0.6856% → 1.9838% |
| E_bfp8 | 1.7805% → 2.8909% | 46.9229% → 43.4073% | 0.6357% → 1.3982% |
| E_bfp4 | 11.6646% → 32.3681% | 79.8916% → 2252.0255% | 0.5368% → 8.3194% |

The extreme BFP4 common-K output is finite and matches on two actual trace
replays; input/source integrity checks pass. This is not an overflow/NaN metric
artifact. This ablation does not isolate whether its cause is K representation,
Q rounding, downstream arithmetic, or their interaction. Do not generalize this
percentage to ordinary model inputs. It is a strong reason not to treat the
plain BFP4 route as numerically equivalent to our selected recipe.

Two subtleties:

- Plain BFP8 is more accurate here than plain BF16 under the same LoFi kernel.
  More storage bits do not ensure better results under truncated arithmetic.
  This experiment does not separate packing bias from Q/K/V contributions;
  it should not be described as proving one particular causal mechanism.
- Removing rounding slightly improves common-K error for BF16/BFP8, while
  worsening all broad cases. The benefit is not universally monotonic.

PCC alone still understates ordinary-input differences: at normal 256K,
plain BF16 has 5.5550% L2 with PCC 0.999393, and plain BFP8 has 4.3188% L2
with PCC 0.999488. Full metrics, including PCC, remain in `accuracy-v1.json`.

## Performance

| Recipe | Prepared TFLOP/s/core | Plain TFLOP/s/core | Plain median ms |
| --- | ---: | ---: | ---: |
| E_bf16 | 2.122001 | 2.121776 | 259.1017 |
| E_bfp8 | 2.090987 | 2.090914 | 262.9261 |
| E_bfp4 | 2.090106 | 2.089871 | 263.0573 |

Compute-only throughput is unchanged within 0.012%. Preprocessing/casts are
outside all timings, so this does not measure end-to-end savings from deleting
preprocessing. Plain timings were remeasured; prepared timings and unchanged
D/C/B/A points are carried over from the preceding experiments on the same
host. Both timing sets use one core, resident Q256/K512/D128, 16 Q repeats and
512 KV chunks, 549755813888 useful QK+PV FLOPs, nine warmups and 12 measured
blocking trace replays per variant, with interleaved/rotating variant order.
There is no recurring input DM. These are not chip-throughput estimates.

## Qualification and reproducibility

Device: bh-lb-08, reservation 227098, device 0, 2026-09-21. All device calls
used `compute-sprint-v1/run_locked.sh`. Existing host libraries and compiled
attention kernels were reused; native typecast kernels compiled/ran through
the normal JIT as needed. No C++ or production source edits were made.

- `smoke-v1.json`: 15 completed runs, testing first/second/odd-final chunks,
  multi-query blocks, normal/uniform/constant/zero V.
- `accuracy-v1.json`: 63 completed runs, 21 per variant, identical original
  input hashes to the preceding rounded suite. One head, 256 query rows,
  D128, noncausal; seed 20260919 and FP64 reference. Rectangular query samples,
  not full square attention timings.
- Broad cases: six distributions × KV 4096/32768/262144; normal, clipped ±2,
  Q/K ×0.25, Q/K ×2, sparse outliers, uniform attention. Stress: Q/K/V +32,
  each at KV32768. Stress is excluded from broad box statistics.
- Kernel metadata matches the corresponding prepared recipe exactly in all
  63 cases. No chunk, CB format/size, buffering, fidelity or define changes.
- Two actual trace replays match eager output bits in every accuracy/smoke
  run. Original/prepared input hashes and selected source hashes remain
  unchanged. The shared device lock is free and dirty marker absent at end.

Run `bench.py --mode smoke|accuracy|perf --output NEW_FILENAME.json` through
the shared lock wrapper only. Each run rejects existing output files and marks
completion only after all checks pass. Render final evidence locally with:

```sh
/tmp/sdpa-pareto-plot-venv/bin/python experiments/sdpa-l2/compute-sprint-v3/no-rounding-v1/plot.py
```

The plot validates completion, matching inputs/metadata and integrity flags.
Broad-panel x coordinates are actual throughput with no jitter. Stress uses
explicit categorical paired positions to make the effect readable. Boxes
show case quartiles and min–max, not uncertainty. The plot was visually
inspected; both Python files pass local compilation. Frozen evidence and the
preexisting tracked worktree diff are unchanged.

These are operator results, not new FLUX/Wan image/video evaluations. For
integration without custom input ops, B/C/D retain their measured contracts;
plain E_bfp8 is the most defensible of the new LoFi ordinary-input options,
but should not inherit the rounded recipe's model-quality qualification.
