# GELU Backward Performance Results

## 1. Three-State Performance Comparison

Three code states measured through the same dispatch path:
`ttnn::experimental::gelu_bw(grad, input, approximate=mode)` which routes
through `GeluBackwardDeviceOperation` to a single fused compute kernel.

| State | `approximate=` | Compute kernel | SFPU math |
|-------|---------------|----------------|-----------|
| Original (main) | `"none"` | `eltwise_bw_gelu_approx_none.cpp` | inline erf+exp formula |
| -02 (pre-review) | `"poly"` | `eltwise_bw_gelu_poly.cpp` | POLYVAL macros |
| -03 (post-review) | `"poly"` | `eltwise_bw_gelu_poly.cpp` | PolynomialEvaluator, recip_init, float_to_fp16b |

Tensor shape: (1, 1, 2048, 32) = 65,536 BF16 values = 64 tiles.

### 1a. Wall-Clock Timing (host-side, 10 runs after 3 warmup)

| State              | Avg    | Min    | Max    | Median |
|--------------------|--------|--------|--------|--------|
| Original (erf+exp) | 0.25ms | 0.23ms | 0.29ms | 0.25ms |
| -02 (polynomial)   | 0.24ms | 0.23ms | 0.26ms | 0.24ms |
| -03 (polynomial)   | 0.25ms | 0.23ms | 0.27ms | 0.25ms |

Wall-clock times are dominated by dispatch overhead and are within noise
across all three states.

### 1b. Device Profiler (TRISC_1 kernel cycles, 2 warmup + 1 profiled)

`TT_METAL_DEVICE_PROFILER=1`, last run (run_host_id 3072):

| State              | Kernel Calls | Avg Cycles/Tile | Total Kernel Cycles | Time @ 1350MHz |
|--------------------|-------------|-----------------|---------------------|----------------|
| Original (erf+exp) | 64          | 11,743          | 751,551             | 0.557 ms       |
| -02 (polynomial)   | 64          | 7,799           | 499,131             | 0.370 ms       |
| -03 (polynomial)   | 64          | 7,843           | 501,926             | 0.372 ms       |

The polynomial kernel is **33% faster** in compute cycles than the original
erf+exp formula (7,843 vs 11,743 cycles/tile). This is expected: the
polynomial uses a compact Horner evaluation while the formula requires
separate erf() and exp() SFPU calls.

### 1c. Validation

The cycle count difference (11,743 vs 7,843) proves that the "poly" and
"none" paths dispatch genuinely different kernels. Our previous measurement
(before host rebuild) showed 11,728 vs 11,726 — essentially identical —
because both paths ran the same "none" kernel due to a stale host binary.

## 2. Refactoring Impact (-02 vs -03)

Branch `-03` vs `-02`: review feedback refactoring (PolynomialEvaluator,
recip_init, BF16 rounding).

| Metric              | -02 (before) | -03 (after) | Delta |
|---------------------|-------------|-------------|-------|
| Avg cycles/tile     | 7,799       | 7,843       | +0.6% |
| Total kernel cycles | 499,131     | 501,926     | +0.6% |

Within measurement noise. The `-03` refactoring is **performance-neutral**.

## 3. Summary

| Metric                          | Original (erf+exp) | Polynomial (-03) | Improvement  |
|--------------------------------|---------------------|-------------------|--------------|
| Max ULP error                   | 32,460              | 1                 | 32,460x      |
| Kernel cycles/tile (TRISC_1)   | 11,743              | 7,843             | 33% faster   |
| Total kernel cycles (64 tiles) | 751,551             | 501,926           | 33% faster   |
| Wall-clock (warm, 64 tiles)    | 0.25 ms             | 0.25 ms           | ~same        |

The polynomial implementation achieves **Max ULP = 1** (vs 32,460 for the
formula) while being **33% faster** in kernel compute cycles. Wall-clock
times are equivalent because they are dominated by dispatch overhead at
this tensor size.

## Environment

- Arch: Blackhole
- Chip freq: 1350 MHz
- Build: Debug
- Date: 2026-02-24
- Host rebuilt with `build_metal.sh --debug --build-all --enable-ccache`
- Kernel cache cleared before each state measurement
