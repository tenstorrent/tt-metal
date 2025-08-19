# TT-Metal Operator Test Comprehensive Analysis Report

**Generated:** 2025-08-18 23:13:42
**Analysis Scope:** All operator tests in ssinghal/tests/

## 📊 EXECUTIVE SUMMARY

| Metric | Count | Percentage |
|--------|-------|------------|
| **Total Operators Tested** | 46 | 100.0% |
| **Total Individual Tests** | 3361 | 100.0% |
| **Tests Passed** | 622 | 18.5% |
| **Tests Failed** | 536 | 15.9% |
| **Tests Skipped** | 2202 | 65.5% |
| **OOM Failures** | 2000 | 59.5% |

## 🎯 OPERATOR STATUS BREAKDOWN

### PASSED Operators (22)

- **add** | Tests: 80 | Pass Rate: 25.0%
- **cat** | Tests: 80 | Pass Rate: 30.0%
- **clone** | Tests: 80 | Pass Rate: 30.0%
- **concat** | Tests: 80 | Pass Rate: 30.0%
- **div** | Tests: 100 | Pass Rate: 44.0%
- **gelu** | Tests: 90 | Pass Rate: 55.6%
- **geluactivation** | Tests: 56 | Pass Rate: 39.3%
- **hardtanh** | Tests: 80 | Pass Rate: 25.0%
- **log** | Tests: 24 | Pass Rate: 16.7%
- **mean** | Tests: 70 | Pass Rate: 54.3%
- **mish** | Tests: 80 | Pass Rate: 30.0%
- **mul** | Tests: 80 | Pass Rate: 25.0%
- **ones** | Tests: 80 | Pass Rate: 40.0%
- **permute** | Tests: 80 | Pass Rate: 30.0%
- **relu** | Tests: 80 | Pass Rate: 30.0%
- **sigmoid** | Tests: 88 | Pass Rate: 36.4%
- **silu** | Tests: 80 | Pass Rate: 30.0%
- **softplus** | Tests: 80 | Pass Rate: 30.0%
- **sub** | Tests: 80 | Pass Rate: 25.0%
- **tanh** | Tests: 80 | Pass Rate: 30.0%
- **transpose** | Tests: 106 | Pass Rate: 41.5%
- **unsqueeze** | Tests: 78 | Pass Rate: 35.9%

### FAILED Operators (17)

- **addmm** | Tests: 74 | Pass Rate: 0.0%
- **bmm** | Tests: 110 | Pass Rate: 16.4%
- **clamp** | Tests: 70 | Pass Rate: 0.0%
- **clampmin** | Tests: 80 | Pass Rate: 0.0%
- **copy** | Tests: 74 | Pass Rate: 0.0%
- **dropout** | Tests: 74 | Pass Rate: 0.0%
- **identity** | Tests: 80 | Pass Rate: 0.0%
- **linalgvectornorm** | Tests: 80 | Pass Rate: 0.0%
- **mm** | Tests: 28 | Pass Rate: 0.0%
- **silu_inplace** | Tests: 80 | Pass Rate: 0.0%
- **softmax** | Tests: 60 | Pass Rate: 0.0%
- **split** | Tests: 39 | Pass Rate: 15.4%
- **split_with_sizes** | Tests: 44 | Pass Rate: 0.0%
- **topk** | Tests: 70 | Pass Rate: 0.0%
- **unsafeview** | Tests: 80 | Pass Rate: 5.0%
- **upsample_nearest2d** | Tests: 48 | Pass Rate: 0.0%
- **view** | Tests: 80 | Pass Rate: 5.0%

### SKIPPED Operators (6)

- **expand** | Tests: 78 | Pass Rate: 0.0%
- **leakyrelu** | Tests: 78 | Pass Rate: 0.0%
- **maxpool2d** | Tests: 78 | Pass Rate: 0.0%
- **native_batch_norm** | Tests: 75 | Pass Rate: 0.0%
- **splitwithsizes** | Tests: 78 | Pass Rate: 0.0%
- **stack** | Tests: 70 | Pass Rate: 0.0%

### UNKNOWN Operators (1)

- **linear** | Tests: 1 | Pass Rate: 0.0%

## 📋 DETAILED TEST RESULTS

| Operator | Status | Total Tests | Passed | Failed | Skipped | Pass Rate | OOM Count |
|----------|--------|-------------|--------|--------|---------|-----------|-----------||
| add | ✅ PASSED | 80 | 20 | 0 | 60 | 25.0% | 60 |
| addmm | ❌ FAILED | 74 | 0 | 68 | 6 | 0.0% | 6 |
| bmm | ❌ FAILED | 110 | 18 | 80 | 12 | 16.4% | 12 |
| cat | ✅ PASSED | 80 | 24 | 0 | 56 | 30.0% | 56 |
| clamp | ❌ FAILED | 70 | 0 | 38 | 32 | 0.0% | 32 |
| clampmin | ❌ FAILED | 80 | 0 | 32 | 48 | 0.0% | 48 |
| clone | ✅ PASSED | 80 | 24 | 0 | 56 | 30.0% | 56 |
| concat | ✅ PASSED | 80 | 24 | 0 | 56 | 30.0% | 56 |
| copy | ❌ FAILED | 74 | 0 | 34 | 40 | 0.0% | 40 |
| div | ✅ PASSED | 100 | 44 | 0 | 56 | 44.0% | 56 |
| dropout | ❌ FAILED | 74 | 0 | 36 | 38 | 0.0% | 38 |
| expand | ⏭️ SKIPPED | 78 | 0 | 0 | 78 | 0.0% | 48 |
| gelu | ✅ PASSED | 90 | 50 | 0 | 40 | 55.6% | 40 |
| geluactivation | ✅ PASSED | 56 | 22 | 0 | 34 | 39.3% | 34 |
| hardtanh | ✅ PASSED | 80 | 20 | 0 | 60 | 25.0% | 60 |
| identity | ❌ FAILED | 80 | 0 | 24 | 56 | 0.0% | 56 |
| leakyrelu | ⏭️ SKIPPED | 78 | 0 | 0 | 78 | 0.0% | 48 |
| linalgvectornorm | ❌ FAILED | 80 | 0 | 32 | 48 | 0.0% | 48 |
| linear | ❓ UNKNOWN | 1 | 0 | 0 | 0 | 0.0% | 0 |
| log | ✅ PASSED | 24 | 4 | 0 | 20 | 16.7% | 20 |
| maxpool2d | ⏭️ SKIPPED | 78 | 0 | 0 | 78 | 0.0% | 48 |
| mean | ✅ PASSED | 70 | 38 | 0 | 32 | 54.3% | 32 |
| mish | ✅ PASSED | 80 | 24 | 0 | 56 | 30.0% | 56 |
| mm | ❌ FAILED | 28 | 0 | 28 | 0 | 0.0% | 0 |
| mul | ✅ PASSED | 80 | 20 | 0 | 60 | 25.0% | 60 |
| native_batch_norm | ⏭️ SKIPPED | 75 | 0 | 0 | 75 | 0.0% | 48 |
| ones | ✅ PASSED | 80 | 32 | 0 | 48 | 40.0% | 48 |
| permute | ✅ PASSED | 80 | 24 | 0 | 56 | 30.0% | 56 |
| relu | ✅ PASSED | 80 | 24 | 0 | 56 | 30.0% | 56 |
| sigmoid | ✅ PASSED | 88 | 32 | 0 | 56 | 36.4% | 56 |
| silu | ✅ PASSED | 80 | 24 | 0 | 56 | 30.0% | 56 |
| silu_inplace | ❌ FAILED | 80 | 0 | 32 | 48 | 0.0% | 48 |
| softmax | ❌ FAILED | 60 | 0 | 18 | 42 | 0.0% | 42 |
| softplus | ✅ PASSED | 80 | 24 | 0 | 56 | 30.0% | 56 |
| split | ❌ FAILED | 39 | 6 | 8 | 25 | 15.4% | 0 |
| split_with_sizes | ❌ FAILED | 44 | 0 | 14 | 30 | 0.0% | 30 |
| splitwithsizes | ⏭️ SKIPPED | 78 | 0 | 0 | 78 | 0.0% | 48 |
| stack | ⏭️ SKIPPED | 70 | 0 | 0 | 70 | 0.0% | 40 |
| sub | ✅ PASSED | 80 | 20 | 0 | 60 | 25.0% | 60 |
| tanh | ✅ PASSED | 80 | 24 | 0 | 56 | 30.0% | 56 |
| topk | ❌ FAILED | 70 | 0 | 38 | 32 | 0.0% | 32 |
| transpose | ✅ PASSED | 106 | 44 | 0 | 62 | 41.5% | 62 |
| unsafeview | ❌ FAILED | 80 | 4 | 20 | 56 | 5.0% | 56 |
| unsqueeze | ✅ PASSED | 78 | 28 | 0 | 50 | 35.9% | 50 |
| upsample_nearest2d | ❌ FAILED | 48 | 0 | 14 | 34 | 0.0% | 34 |
| view | ❌ FAILED | 80 | 4 | 20 | 56 | 5.0% | 56 |

## 🧠 OUT OF MEMORY (OOM) ANALYSIS

**Total OOM Failures:** 2000

### OOM Failures by Operator

| Operator | OOM Count | Largest Shape | Max Memory (MB) | Max Memory (GB) |
|----------|-----------|---------------|-----------------|------------------|
| add | 60 | [1, 96, 2160, 3840] | 1518.8 | 1.48 |
| addmm | 6 | [64000, 512] | 62.5 | 0.06 |
| bmm | 12 | [12, 8160, 8160] | 1524.0 | 1.49 |
| cat | 56 | [1, 96, 2160, 3840] | 1518.8 | 1.48 |
| clamp | 32 | [1, 96, 2160, 3840] | 1518.8 | 1.48 |
| clampmin | 48 | [1, 96, 2160, 3840] | 1518.8 | 1.48 |
| clone | 56 | [1, 96, 2160, 3840] | 1518.8 | 1.48 |
| concat | 56 | [1, 96, 2160, 3840] | 1518.8 | 1.48 |
| copy | 40 | [1, 96, 2160, 3840] | 1518.8 | 1.48 |
| div | 56 | [1, 96, 2160, 3840] | 1518.8 | 1.48 |
| dropout | 38 | [1, 96, 2160, 3840] | 1518.8 | 1.48 |
| expand | 48 | [1, 96, 2160, 3840] | 1518.8 | 1.48 |
| gelu | 40 | [1, 96, 2160, 3840] | 1518.8 | 1.48 |
| geluactivation | 34 | [1, 96, 2160, 3840] | 1518.8 | 1.48 |
| hardtanh | 60 | [1, 96, 2160, 3840] | 1518.8 | 1.48 |
| identity | 56 | [1, 96, 2160, 3840] | 1518.8 | 1.48 |
| leakyrelu | 48 | [1, 96, 2160, 3840] | 1518.8 | 1.48 |
| linalgvectornorm | 48 | [1, 96, 2160, 3840] | 1518.8 | 1.48 |
| log | 20 | [1, 96, 2160, 3840] | 1518.8 | 1.48 |
| maxpool2d | 48 | [1, 96, 2160, 3840] | 1518.8 | 1.48 |
| mean | 32 | [1, 96, 2160, 3840] | 1518.8 | 1.48 |
| mish | 56 | [1, 96, 2160, 3840] | 1518.8 | 1.48 |
| mul | 60 | [1, 96, 2160, 3840] | 1518.8 | 1.48 |
| native_batch_norm | 48 | [1, 96, 2160, 3840] | 1518.8 | 1.48 |
| ones | 48 | [1, 96, 2160, 3840] | 1518.8 | 1.48 |
| permute | 56 | [1, 96, 2160, 3840] | 1518.8 | 1.48 |
| relu | 56 | [1, 96, 2160, 3840] | 1518.8 | 1.48 |
| sigmoid | 56 | [1, 96, 2160, 3840] | 1518.8 | 1.48 |
| silu | 56 | [1, 96, 2160, 3840] | 1518.8 | 1.48 |
| silu_inplace | 48 | [1, 96, 2160, 3840] | 1518.8 | 1.48 |
| softmax | 42 | [1, 12, 8160, 8160] | 1524.0 | 1.49 |
| softplus | 56 | [1, 96, 2160, 3840] | 1518.8 | 1.48 |
| split_with_sizes | 30 | [1, 192, 2176, 1920] | 1530.0 | 1.49 |
| splitwithsizes | 48 | [1, 96, 2160, 3840] | 1518.8 | 1.48 |
| stack | 40 | [1, 96, 2160, 3840] | 1518.8 | 1.48 |
| sub | 60 | [1, 96, 2160, 3840] | 1518.8 | 1.48 |
| tanh | 56 | [1, 96, 2160, 3840] | 1518.8 | 1.48 |
| topk | 32 | [1, 96, 2160, 3840] | 1518.8 | 1.48 |
| transpose | 62 | [1, 96, 2160, 3840] | 1518.8 | 1.48 |
| unsafeview | 56 | [1, 96, 2160, 3840] | 1518.8 | 1.48 |
| unsqueeze | 50 | [1, 96, 2160, 3840] | 1518.8 | 1.48 |
| upsample_nearest2d | 34 | [1, 96, 1088, 1920] | 382.5 | 0.37 |
| view | 56 | [1, 96, 2160, 3840] | 1518.8 | 1.48 |

### Critical OOM Shapes (Top 20)

| Operator | Shape | Memory (MB) | Memory (GB) |
|----------|-------|-------------|-------------|
| split_with_sizes | [1, 192, 2176, 1920] | 1530.0 | 1.494 |
| split_with_sizes | [1, 96, 2176, 3840] | 1530.0 | 1.494 |
| split_with_sizes | [1, 192, 2176, 1920] | 1530.0 | 1.494 |
| split_with_sizes | [1, 96, 2176, 3840] | 1530.0 | 1.494 |
| bmm | [12, 8160, 8160] | 1524.0 | 1.488 |
| bmm | [12, 8160, 8160] | 1524.0 | 1.488 |
| softmax | [1, 12, 8160, 8160] | 1524.0 | 1.488 |
| softmax | [12, 8160, 8160] | 1524.0 | 1.488 |
| softmax | [1, 12, 8160, 8160] | 1524.0 | 1.488 |
| softmax | [12, 8160, 8160] | 1524.0 | 1.488 |
| add | [1, 96, 2160, 3840] | 1518.8 | 1.483 |
| add | [1, 96, 2160, 3840] | 1518.8 | 1.483 |
| cat | [1, 96, 2160, 3840] | 1518.8 | 1.483 |
| cat | [1, 96, 2160, 3840] | 1518.8 | 1.483 |
| clamp | [1, 96, 2160, 3840] | 1518.8 | 1.483 |
| clamp | [1, 96, 4320, 1920] | 1518.8 | 1.483 |
| clamp | [1, 96, 2160, 3840] | 1518.8 | 1.483 |
| clamp | [1, 96, 4320, 1920] | 1518.8 | 1.483 |
| clampmin | [1, 96, 2160, 3840] | 1518.8 | 1.483 |
| clampmin | [1, 96, 2160, 3840] | 1518.8 | 1.483 |

*... and 1980 more OOM failures*

## ❌ FAILURE REASON ANALYSIS

| Failure Reason | Count |
|----------------|-------|
| Runtime error | 10 |
| Matrix dimension mismatch | 3 |
| Missing ttnn operator implementation | 2 |

## 💡 RECOMMENDATIONS

### Immediate Actions:

1. **Memory Optimization Priority:**
   1. **transpose** operator (62 OOM failures)
   2. **add** operator (60 OOM failures)
   3. **hardtanh** operator (60 OOM failures)
   4. **mul** operator (60 OOM failures)
   5. **sub** operator (60 OOM failures)

2. **Implementation Strategies:**
   - **Tensor Chunking:** Split large tensors into smaller manageable pieces
   - **Memory Pooling:** Implement efficient memory reuse patterns
   - **Streaming Processing:** Process data in streams rather than loading entire tensors
   - **Precision Reduction:** Consider using smaller data types where appropriate

---

*Report generated by TT-Metal Operator Test Analyzer*
