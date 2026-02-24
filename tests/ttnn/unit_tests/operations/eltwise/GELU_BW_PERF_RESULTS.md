# GELU Backward Performance Results

Branch `-03` vs `-02`: Review feedback refactoring (PolynomialEvaluator, recip_init, BF16 rounding)

## 1. Refactoring Impact (-02 vs -03)

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

## 2. Polynomial vs Original Implementation

Comparison of the new polynomial `calculate_gelu_derivative_polynomial` against
the original formula-based `calculate_gelu_derivative`.

### 2a. Wall-Clock Timing (warm cache, 3 runs)

**65,026 BF16 values, ComprehensiveULP test variant for each implementation:**

| Implementation | Run 1  | Run 2  | Run 3  | Avg    |
|---------------|--------|--------|--------|--------|
| Original      | 809 ms | 809 ms | 806 ms | 808 ms |
| Polynomial    | 326 ms | 329 ms | 327 ms | 327 ms |

**Wall-clock speedup: ~2.5x**

Note: Tests exercise different TTNN code paths. The original goes through
composite `ttnn::gelu_bw` (backward with gradient multiply, decomposes into
~130 sub-kernels per tile). The polynomial test calls `gelu_derivative_tile`
directly (single SFPU kernel per tile). Host-side ULP analysis overhead is
similar but not identical.

### 2b. Device Profiler (cycle-level, TRISC_1 = math kernel)

**Standalone profiler runs** (`TT_METAL_DEVICE_PROFILER=1`):

| Implementation | Kernel Calls | Avg Cycles/Call | Total Cycles | Total Time  |
|---------------|-------------|-----------------|-------------|-------------|
| Original      | 1168        | 1,362           | 1,591,152   | 1.18 ms     |
| Polynomial    | 64          | 7,820           | 500,494     | 0.37 ms     |

**Total math kernel cycles: 3.2x fewer for polynomial.**

The original decomposes into many small kernels (1168 calls averaging 1,362
cycles each). The polynomial uses 64 calls (one per 32x32 tile) averaging
7,820 cycles each. The polynomial does more work per call but avoids
kernel dispatch overhead.

### 2c. CompareWithStandard Profiler (same test, both paths)

From `GeluBwPolyTest.CompareWithStandard` (9 test values, both paths):

**Standard gelu_bw path:**
- 72 tile operations, 130 sub-kernel dispatches each
- Avg 18,948 total TRISC_1 cycles per tile (14.04 us)

**Polynomial gelu_derivative path:**
- 3 kernel dispatches per test point:
  - SFPU compute: ~7,491 cycles (5.55 us) - the polynomial evaluation
  - Eltwise op:   ~4,268 cycles (3.16 us)
  - Pack/copy:    ~1,874 cycles (1.39 us)
- Total per tile: ~13,633 cycles (10.10 us)

**Per-tile speedup: ~1.4x** (13,633 vs 18,948 cycles)

## 3. Summary

| Metric                         | Original     | Polynomial   | Delta      |
|-------------------------------|-------------|-------------|------------|
| Max ULP                        | 32,460      | 1           | 32,460x    |
| Wall-clock (65K values, warm) | 808 ms      | 327 ms      | 2.5x       |
| Total math cycles (profiler)  | 1,591,152   | 500,494     | 3.2x       |
| Per-tile cycles (same test)   | 18,948      | 13,633      | 1.4x       |

The polynomial implementation is both more accurate (Max ULP 32,460 -> 1)
and faster (1.4-3.2x depending on measurement method).

## Environment

- Arch: Blackhole
- Chip freq: 1350 MHz
- Build: Debug
- Date: 2026-02-24
- Kernel cache cleared before cold runs
