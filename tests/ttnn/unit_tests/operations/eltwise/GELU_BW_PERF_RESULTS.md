# GELU Backward Performance Results

## 1. Refactoring Impact (-02 vs -03)

Branch `-03` vs `-02`: Review feedback refactoring (PolynomialEvaluator, recip_init, BF16 rounding).

No performance regression from the review feedback changes.

**Test: `GeluBwPolyTest.ComprehensiveULPAnalysis`** (65,026 BF16 values through polynomial gelu_derivative)

| Run   | -02 (before) | -03 (after) |
|-------|-------------|-------------|
| Cold  | 3245 ms     | 3182 ms     |
| Warm 1| 639 ms      | 624 ms      |
| Warm 2| 640 ms      | 644 ms      |
| Warm 3| (reconfirm) | 640 ms      |

**Test: `GeluBwPolyTest.CompareWithStandard`** (runs both standard and polynomial paths)

| Run   | -02 (before) | -03 (after) |
|-------|-------------|-------------|
| Cold  | 2196 ms     | 2190 ms     |
| Warm 1| 506 ms      | 505 ms      |
| Warm 2| 506 ms      | 507 ms      |
| Warm 3| (reconfirm) | 515 ms      |

**Conclusion:** Warm-cache timings are identical within noise. The `-03` refactoring
(PolynomialEvaluator::eval, recip_init, float_to_fp16b) is **performance-neutral**.

## 2. Polynomial vs Formula: Apples-to-Apples Comparison

Both implementations are compared through the same dispatch path:
`ttnn::experimental::gelu_bw(grad, input, approximate=mode)` which routes
through `GeluBackwardDeviceOperation` to a single fused compute kernel.

- `approximate="none"` dispatches `eltwise_bw_gelu_approx_none.cpp` (erf+exp formula)
- `approximate="poly"` dispatches `eltwise_bw_gelu_poly.cpp` (polynomial)

Tensor shape: (1, 1, 2048, 32) = 65,536 BF16 values = 64 tiles.

### 2a. Wall-Clock Timing (host-side, 10 runs, warm cache)

| Implementation           | Avg    | Min    | Max    |
|-------------------------|--------|--------|--------|
| experimental "none"     | 0.25ms | 0.24ms | 0.30ms |
| experimental "poly"     | 0.27ms | 0.23ms | 0.36ms |
| composite gelu_bw "none"| 2.08ms | 1.98ms | 2.59ms |

The polynomial kernel and the formula kernel are **the same speed** through
the experimental (fused) dispatch path. The composite `ttnn::gelu_bw` is ~8x
slower because it decomposes into ~130 sub-kernel dispatches per tile.

### 2b. Device Profiler (cycle-level, TRISC_1 = math kernel)

`TT_METAL_DEVICE_PROFILER=1`, profiled run after 2 warmup iterations:

| Implementation        | Kernel Calls | Avg Cycles/Call | Total Kernel Cycles | Total Time  |
|----------------------|-------------|-----------------|---------------------|-------------|
| experimental "none"  | 64          | 11,728          | 750,595             | 0.556 ms    |
| experimental "poly"  | 64          | 11,726          | 750,455             | 0.556 ms    |

Both kernels dispatch exactly 64 calls (one per 32x32 tile) with virtually
identical cycle counts. The polynomial evaluation is **cycle-neutral** compared
to the erf+exp formula.

## 3. Summary

| Metric                          | Formula ("none") | Polynomial ("poly") | Delta       |
|--------------------------------|------------------|---------------------|-------------|
| Max ULP error                   | 32,460           | 1                   | 32,460x     |
| Wall-clock (64 tiles, warm)    | 0.25 ms          | 0.27 ms             | ~same       |
| Total TRISC_1 cycles (profiler)| 750,595          | 750,455             | ~same       |
| Cycles per tile                 | 11,728           | 11,726              | ~same       |

The polynomial implementation achieves **Max ULP = 1** (vs 32,460 for the
formula) with **no measurable performance cost**. The compute kernel runs
in the same number of cycles regardless of the approximation method.

## Environment

- Arch: Blackhole
- Chip freq: 1350 MHz
- Build: Debug
- Date: 2026-02-24
- Kernel cache cleared before cold runs
