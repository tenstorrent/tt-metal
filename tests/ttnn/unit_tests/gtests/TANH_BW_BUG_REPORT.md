# Bug Report: ttnn::tanh_bw has catastrophic ULP errors

### Component / Area
TTNN / Eltwise Backward Operations / tanh_bw

### Issue Type (optional)
Bad Outputs

### Observed
`ttnn::tanh_bw` produces Max ULP = 15,139 with mean ULP = 155.59.

The `ttnn::tanh_bw` operation produces catastrophically incorrect results in the transition and saturation regions, with **Max ULP = 15,139** compared to the mathematically correct reference. This is an implementation bug, not a BF16 format limitation.

**Proof**: The forward `ttnn::tanh` operation achieves **Max ULP = 1** across the entire BF16 range using the same testing methodology, demonstrating that correct implementation is achievable.

Per-segment analysis:

```
TANH_BW PER-SEGMENT ULP ANALYSIS
================================

Segment                  Count   Max ULP    Mean ULP     ULP=0    ULP<=1    ULP<=2   Worst Input
------------------------------------------------------------------------------------------------
x < -10                  15967     12666      136.48     98.3%     98.3%     98.3%      -10.0625
-10 <= x < -5              128     14515    13738.42      0.0%      0.0%      0.0%       -5.0312
-5 <= x < -4                32     14886    14709.56      0.0%      0.0%      0.0%       -4.0312
-4 <= x < -3.5              32     15080    14988.47      0.0%      0.0%      0.0%       -3.5156
-3.5 <= x < -3              32     15139     5225.31      0.0%      0.0%      3.1%       -3.3438
-3 <= x < -2                64        95       22.30      4.7%      4.7%      7.8%       -3.0000
-2 <= x < -1               128        17        3.67     12.5%     39.8%     54.7%       -1.7734
-1 <= x < -0.5             128         3        1.04     16.4%     81.2%     98.4%       -0.9336
-0.5 <= x < 0            16001         1        0.08     91.9%    100.0%    100.0%       -0.5000
x == 0                       1         0        0.00    100.0%    100.0%    100.0%        0.0000
0 < x < 0.5              16000         1        0.08     91.9%    100.0%    100.0%        0.0002
0.5 <= x < 1               128         3        1.04     16.4%     81.2%     98.4%        0.8984
1 <= x < 2                 128        17        3.68     11.7%     39.8%     54.7%        1.7734
2 <= x < 3                  64        90       20.81      6.2%      6.2%      9.4%        2.9844
3 <= x < 3.5                32     15139     4756.81      0.0%      0.0%      3.1%        3.3438
3.5 <= x < 4                32     15087    14994.44      0.0%      0.0%      0.0%        3.5000
4 <= x < 5                  32     14896    14721.09      0.0%      0.0%      0.0%        4.0000
5 <= x < 10                128     14527    13752.80      0.0%      0.0%      0.0%        5.0000
x >= 10                  15968     12686      137.27     98.3%     98.3%     98.3%       10.0000
------------------------------------------------------------------------------------------------
```

Specific failure examples:

| Input x | Expected Output | Actual Output | ULP Error |
|---------|-----------------|---------------|-----------|
| -3.3438 | 0.0049 | 0.0000 | 15,139 |
| -5.0312 | 0.0002 | 0.0000 | 14,515 |
| 3.3438 | 0.0049 | 0.0000 | 15,139 |
| 4.0000 | 0.0013 | 0.0000 | 14,896 |

**Note**: Values like 0.0049, 0.0013, 0.0002 are perfectly representable in BF16. The implementation is producing incorrect zeros.

### Expected
`ttnn::tanh_bw` should achieve precision comparable to `ttnn::tanh` (Max ULP <= 2).

The forward `ttnn::tanh` achieves Max ULP = 1:

```
TANH (FORWARD) PER-SEGMENT ULP ANALYSIS
=======================================

Segment                  Count   Max ULP    Mean ULP     ULP=0    ULP<=1    ULP<=2   Worst Input
------------------------------------------------------------------------------------------------
x < -10                  15967         0        0.00    100.0%    100.0%    100.0%        0.0000
-10 <= x < -5              128         1        0.88     12.5%    100.0%    100.0%       -9.0000
-5 <= x < -4                32         1        1.00      0.0%    100.0%    100.0%       -5.0000
-4 <= x < -3                64         1        0.75     25.0%    100.0%    100.0%       -4.0000
-3 <= x < -2                64         1        0.45     54.7%    100.0%    100.0%       -2.7656
-2 <= x < -1               128         1        0.52     47.7%    100.0%    100.0%       -2.0000
-1 <= x < -0.5             128         1        0.41     58.6%    100.0%    100.0%       -1.0000
-0.5 <= x < 0            16001         1        0.07     92.5%    100.0%    100.0%       -0.4980
x == 0                       1         0        0.00    100.0%    100.0%    100.0%        0.0000
0 < x < 0.5              16000         1        0.07     92.5%    100.0%    100.0%        0.0000
0.5 <= x < 1               128         1        0.41     59.4%    100.0%    100.0%        0.5078
1 <= x < 2                 128         1        0.52     47.7%    100.0%    100.0%        1.0000
2 <= x < 3                  64         1        0.47     53.1%    100.0%    100.0%        2.0000
3 <= x < 4                  64         1        0.73     26.6%    100.0%    100.0%        3.0469
4 <= x < 5                  32         1        1.00      0.0%    100.0%    100.0%        4.0000
5 <= x < 10                128         1        0.88     11.7%    100.0%    100.0%        5.0000
x >= 10                  15968         0        0.00    100.0%    100.0%    100.0%        0.0000
------------------------------------------------------------------------------------------------
```

| Operation | Max ULP | Mean ULP | % Within 1 ULP | % Within 2 ULP |
|-----------|---------|----------|----------------|----------------|
| tanh (forward) | **1** | **0.047** | **100%** | **100%** |
| tanh_bw (backward) | 15,139 | 155.59 | 97.97% | 98.11% |

This demonstrates that correct BF16 implementation IS achievable for tanh-family functions.

Mathematical background:
```
Forward:  tanh(x)
Backward: d/dx tanh(x) = 1 - tanh(x)^2 = sech^2(x)
```

**Root cause** (verified in source code):

The TTNN implementation (`ttnn/cpp/ttnn/operations/eltwise/unary_backward/unary_backward.cpp:287-301`) chains high-level ops:

```cpp
Tensor tanh_res = ttnn::tanh(input, output_mem_config);      // BF16 tanh saturates to ±1.0 for |x| > ~3.4
tanh_res = ttnn::square(tanh_res, output_mem_config);        // 1.0² = 1.0
tanh_res = ttnn::rsub(tanh_res, 1.0f, ...);                  // 1.0 - 1.0 = 0.0 ← PRECISION LOSS
ttnn::multiply(grad, tanh_res, ...);                         // grad * 0 = 0
```

Note: An SFPU kernel for tanh derivative exists (`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_tanh_derivative.h`) but has the same bug at line 29:
```cpp
val = val * (-val) + vConst1;  // 1 - tanh(x)² - same precision loss
```

When `tanh(x)` saturates to exactly ±1.0 in BF16, the `1 - tanh²` computation yields exactly 0, even though the true derivative (e.g., 0.0013 at x=4) is representable in BF16.

**Why forward tanh works**: The forward tanh (`ckernel_sfpu_tanh.h:47-77`) uses a **polynomial approximation** (Sollya-generated coefficients) that directly computes tanh(x) without intermediate saturation, achieving Max ULP = 1.

**Suggested fix**: Create a similar polynomial or continued fraction approximation for sech²(x) that directly computes the derivative without relying on `1 - tanh(x)²`.

### Steps (exact commands)
```bash
# Prerequisites: tt-metal already cloned and built from https://github.com/tenstorrent/tt-metal

cd <your-tt-metal-directory>

# Fetch the test branch and apply changes without committing
git remote add ivoitovych https://github.com/ivoitovych/tt-metal.git
git fetch ivoitovych ivoitovych/tanh-bf16-ulp-diagnostic-tests
git merge --no-commit --squash ivoitovych/ivoitovych/tanh-bf16-ulp-diagnostic-tests

# Build only the affected target
# (if build fails with "mpfr not found", run: sudo apt-get install libmpfr-dev libgmp-dev)
cmake --build build_Debug --target unit_tests_ttnn -j$(nproc)

# Run tanh ULP diagnostic tests
./build_Debug/test/ttnn/unit_tests_ttnn --gtest_filter="*TanhUlp*:*TanhBwUlp*"
```

Test files (included in the merge):
- C++ tests: `tests/ttnn/unit_tests/gtests/test_tanh_bw_ulp_diagnostic.cpp`
- Python tests: `tests/ttnn/unit_tests/operations/eltwise/backward/test_tanh_bw_ulp_diagnostic.py`

### Frequency
100% reproducible - affects all values in the transition/saturation region (|x| > 3).

### Software Versions
- tt-metal base: https://github.com/tenstorrent/tt-metal commit `78fc90f44b`
- Test branch: https://github.com/ivoitovych/tt-metal/tree/ivoitovych/tanh-bf16-ulp-diagnostic-tests
- OS: Ubuntu 22.04.5 LTS, Kernel 5.15.0-164-generic
- Python: 3.10.12

### Hardware Details
- Device: Blackhole P150a
- Driver: TT-KMD 2.6.1-pre
- Firmware: 19.1.0

### Is this a regression?
Unknown

Note: Prior tanh_bw tests (`tests/ttnn/nightly/unit_tests/operations/eltwise/backward/test_backward_tanh.py`) use PCC comparison and only test the narrow range `[-1.45, 1.45]`, avoiding the saturation region where this bug manifests.

### Priority
P1

### Impact
Incorrect gradients during backpropagation for any model using tanh activation. This can cause:
- Slower or failed convergence during training
- Incorrect weight updates
- Degraded model quality
