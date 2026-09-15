# Isolated adaptive native-BFP4 preprocessor

Status: implemented and CPU-validated; **not JIT-compiled or device-qualified** by this agent. Existing quantizers and attention kernels are untouched.

`adaptive_bfp4_round.py` exports:

`build(device, src, ncores=1, search="pm", output_format="b4") -> (out, invoke, actual_cores)`

The source must contain finite BF16 normal values or zeros, in tiled matrices whose last two dimensions are multiples of32. Native group16 maxima must have unbiased exponents in[-123,106]. The stricter lower bound permits E−1 outputs without FP32-subnormal quanta. An E+1 output may have exponent107: this is valid for native packing, although outside the old preprocessor's magic-constant *input* contract.

Search modes:

- `baseline`: original exponent E, but the same private normalized arithmetic path and batch1 harness.
- `minus`: E versus E−1.
- `pm` (default): E, E−1, E+1, in that order.

Ties retain the earlier candidate, so baseline wins a baseline tie. E+1 is not universally negligible: the model selected it for approximately0.032% of normal groups but4.12% of channel-outlier groups. Keeping both controls allows an explicit cost/benefit measurement.

## Arithmetic and storage contract

The kernel finds each native group maximum using the qualified eight-lane cyclic-max construction. It saves its power-of-two magnitude scale, normalizes the input magnitudes, then constructs each fixed-grid candidate by two separate FP32 magic additions and saturation. Squared errors use separate FP32 subtraction and multiply operations, with denormals flushed as documented for the SFPU.

The group score is not an unspecified `sum`: add squared even/odd columns into eight lanes, then add cyclic right rotations1,2,4, each rounded to FP32. **Keep only lane0 and broadcast it** before selection. The other lanes have different addition orders and could otherwise disagree near ties. Each candidate replaces the current best only when its broadcast score is strictly lower. Finally rescale the selected exact code and restore the original sign.

One input tile is processed per acquisition. FP32 DST tile0 holds input/result, tile1 holds current best normalized values, and tile2 holds broadcast score and scale in its separate column parities. Three of four tiles in a DST half are used. Scratch is never packed. Raw LREG0…7 are used exclusively; no live SFPI compiler-managed vectors, replay templates, or programmable constants are involved. Conditional flags are push/popped. There is no additional L1 scratch CB.

CB0 input has two2048-byte pages. CB16 output has two576-byte B4 pages (or two2048-byte BF16 pages). Payload is5,248B/core for B4,8,192B/core for BF16. The reader/writer are reused unchanged from `bfp4_round/`; all custom/reused files are pinned before/after measurement.

The device check compares decoded FP32 bits exactly after canonicalizing signed zeros only. It also stores matching SHA256 hashes under that explicit contract; these are **not raw packed-DRAM hashes**. Native BFP's zero representation differs from IEEE signed zero. A BF16 output control separates SFPU selection/rounding from native B4 packing.

## Validation completed

Python syntax and CPU arithmetic/oracle checks passed. Eighteen controls used524,288 values each: normal, thresholds, exact ties, wide exponents, zeros, and one randomly positioned ×32 group outlier, crossed with all three search modes. All induced-exponent and native-grid roundtrip assertions passed. No FP32-tree selection differed from an independent FP64 reconstructed-MSE selector in these cases; that equality is **not** promised for all inputs.

The wide case exercises exponent changes across the supported range, not only fixed-exponent anchors. Out of32,768 groups, full search selected10,726 E−1 groups and10,810 E+1 groups. The exact-ties case has11,148 groups tied between baseline and E+1, all retaining baseline. Normal full-search counts were6,061 E−1 and8 E+1; group-outlier counts were2,316 and1,374 respectively.

No build or JIT was run because this assignment explicitly reserves device/JIT work for the parent. No performance claim is made. This deliberately simple batch1/scratch implementation is a qualification prototype, not an optimized replacement for the existing batch4 preprocessor.

## Suggested parent-run checks

From the repo root, under the existing configured Python environment:

```sh
python experiments/sdpa-l2/bfp4-lofi-v2/adaptive_bfp4_round.py --label adaptive-base-bf16-v1 --search baseline --output-format bf16
python experiments/sdpa-l2/bfp4-lofi-v2/adaptive_bfp4_round.py --label adaptive-minus-bf16-v1 --search minus --output-format bf16
python experiments/sdpa-l2/bfp4-lofi-v2/adaptive_bfp4_round.py --label adaptive-pm-bf16-v1 --search pm --output-format bf16
python experiments/sdpa-l2/bfp4-lofi-v2/adaptive_bfp4_round.py --label adaptive-pm-b4-v1 --search pm
python experiments/sdpa-l2/bfp4-lofi-v2/adaptive_bfp4_round.py --label adaptive-wide-v1 --search pm --distribution wide
python experiments/sdpa-l2/bfp4-lofi-v2/adaptive_bfp4_round.py --label adaptive-ties-v1 --search pm --distribution ties
python experiments/sdpa-l2/bfp4-lofi-v2/adaptive_bfp4_round.py --label adaptive-zeros-v1 --search pm --distribution zeros
python experiments/sdpa-l2/bfp4-lofi-v2/adaptive_bfp4_round.py --label adaptive-outliers-v1 --search pm --distribution group_outliers
python experiments/sdpa-l2/bfp4-lofi-v2/adaptive_bfp4_round.py --label adaptive-pm-perf-v1 --search pm --length 32768 --cores 110 --iters 30
```

Add `--host-only` for CPU-only oracles; no device is imported/opened on that route. `--iters` measures trace-replayed read+preprocess+write time and payload GB/s, not attention FLOPs. Repeat performance with `--search minus` and compare both to the original qualified preprocessor, including the batch-size difference.
