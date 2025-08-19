# TT-Metal Operator Non-OOM Failure Analysis Report

**Generated:** 2025-08-18 23:22:05
**Focus:** All failure causes excluding Out of Memory (OOM) issues
**Scope:** Analysis of 46 operators with 282 non-OOM failures

## 🚨 NON-OOM FAILURE EXECUTIVE SUMMARY

| **Metric** | **Value** | **Impact** |
|------------|-----------|------------|
| **Total Non-OOM Failures** | 282 | Implementation Issues |
| **Operators Affected** | 46/46 | 100.0% |
| **Unique Failure Types** | 10 | Error Categories |
| **Critical Failures** | 3 | Blocking Issues |
| **High Priority Failures** | 279 | Major Issues |
| **Medium Priority Failures** | 0 | Minor Issues |

## 📊 FAILURE TYPE BREAKDOWN

| **Failure Type** | **Count** | **Percentage** | **Severity** | **Description** |
|------------------|-----------|----------------|--------------|------------------|
| **Timeout Error** | 92 | 32.6% | ⚠️ High | Test execution timeouts |
| **Runtime Error** | 83 | 29.4% | ⚠️ High | Runtime execution failures |
| **Tt Metal Error** | 53 | 18.8% | ⚠️ High | TT-Metal framework errors |
| **Test Failed** | 29 | 10.3% | ⚠️ High | General test failures |
| **Type Error** | 8 | 2.8% | ⚠️ High | Data type incompatibility |
| **Attribute Error** | 7 | 2.5% | ⚠️ High | Missing ttnn operator implementations |
| **Assertion Error** | 4 | 1.4% | ⚠️ High | Test assertion failures |
| **Dimension Mismatch** | 3 | 1.1% | ⚠️ High | Tensor dimension compatibility issues |
| **Collection Error** | 2 | 0.7% | 🚨 Critical | Test collection/discovery failures |
| **Syntax Error** | 1 | 0.4% | 🚨 Critical | Other failure type |

## 🎯 OPERATOR-SPECIFIC FAILURE ANALYSIS

| **Operator** | **Failure Count** | **Primary Issue** | **Status** |
|--------------|-------------------|-------------------|------------|
| **addmm** | 41 | Runtime Error | ⚠️ HIGH |
| **bmm** | 38 | Runtime Error | ⚠️ HIGH |
| **mm** | 22 | Runtime Error | ⚠️ HIGH |
| **split** | 12 | Assertion Error | ⚠️ HIGH |
| **topk** | 8 | Type Error | ⚠️ HIGH |
| **view** | 8 | Runtime Error | ⚠️ HIGH |
| **clamp** | 7 | Test Failed | ⚠️ HIGH |
| **clampmin** | 7 | Test Failed | ⚠️ HIGH |
| **dropout** | 7 | Test Failed | ⚠️ HIGH |
| **copy** | 6 | Test Failed | ⚠️ HIGH |
| **identity** | 6 | Test Failed | ⚠️ HIGH |
| **softmax** | 6 | Test Failed | ⚠️ HIGH |
| **unsafeview** | 6 | Test Failed | ⚠️ HIGH |
| **linalgvectornorm** | 5 | Attribute Error | ⚠️ HIGH |
| **linear** | 5 | Collection Error | 🚨 CRITICAL |
| **split_with_sizes** | 5 | Tt Metal Error | ⚠️ HIGH |
| **upsample_nearest2d** | 5 | Tt Metal Error | ⚠️ HIGH |
| **silu_inplace** | 4 | Timeout Error | ⚠️ HIGH |
| **add** | 3 | Timeout Error | ⚠️ HIGH |
| **cat** | 3 | Timeout Error | ⚠️ HIGH |
| **clone** | 3 | Timeout Error | ⚠️ HIGH |
| **concat** | 3 | Timeout Error | ⚠️ HIGH |
| **div** | 3 | Timeout Error | ⚠️ HIGH |
| **expand** | 3 | Timeout Error | ⚠️ HIGH |
| **gelu** | 3 | Timeout Error | ⚠️ HIGH |
| **geluactivation** | 3 | Timeout Error | ⚠️ HIGH |
| **hardtanh** | 3 | Timeout Error | ⚠️ HIGH |
| **leakyrelu** | 3 | Timeout Error | ⚠️ HIGH |
| **log** | 3 | Timeout Error | ⚠️ HIGH |
| **maxpool2d** | 3 | Timeout Error | ⚠️ HIGH |
| **mean** | 3 | Timeout Error | ⚠️ HIGH |
| **mish** | 3 | Timeout Error | ⚠️ HIGH |
| **mul** | 3 | Timeout Error | ⚠️ HIGH |
| **native_batch_norm** | 3 | Timeout Error | ⚠️ HIGH |
| **ones** | 3 | Timeout Error | ⚠️ HIGH |
| **permute** | 3 | Timeout Error | ⚠️ HIGH |
| **relu** | 3 | Timeout Error | ⚠️ HIGH |
| **sigmoid** | 3 | Timeout Error | ⚠️ HIGH |
| **silu** | 3 | Timeout Error | ⚠️ HIGH |
| **softplus** | 3 | Timeout Error | ⚠️ HIGH |
| **splitwithsizes** | 3 | Timeout Error | ⚠️ HIGH |
| **stack** | 3 | Timeout Error | ⚠️ HIGH |
| **sub** | 3 | Timeout Error | ⚠️ HIGH |
| **tanh** | 3 | Timeout Error | ⚠️ HIGH |
| **transpose** | 3 | Timeout Error | ⚠️ HIGH |
| **unsqueeze** | 3 | Timeout Error | ⚠️ HIGH |

## 🔍 DETAILED FAILURE ANALYSIS BY TYPE

### Timeout Error (92 failures)

**Operators Affected:** 46

**Examples:**
1. **add**: method: signal
2. **add**: func_only: False
3. **addmm**: method: signal
4. **addmm**: func_only: False
5. **bmm**: method: signal
... and 87 more similar failures

### Runtime Error (83 failures)

**Operators Affected:** 10

**Examples:**
1. **addmm**: TT_THROW @ /home/ubuntu/work/yolov12-high-res-tests/tt-metal/ttnn/cpp/ttnn/operations/matmul/matmul....
2. **addmm**: matmul The width of the first tensor must be equal to the height of the second tensor (256 != 27). T...
3. **addmm**: matmul The width of the first tensor must be equal to the height of the second tensor (512 != 80). T...
4. **addmm**: matmul The width of the first tensor must be equal to the height of the second tensor (256 != 1000)....
5. **addmm**: matmul The width of the first tensor must be equal to the height of the second tensor (1024 != 1000)...
... and 78 more similar failures

### Tt Metal Error (53 failures)

**Operators Affected:** 45

**Examples:**
1. **add**: /home/ubuntu/work/yolov12-high-res-tests/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:141: tt::...
2. **addmm**: /home/ubuntu/work/yolov12-high-res-tests/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:141: tt::...
3. **addmm**: /home/ubuntu/work/yolov12-high-res-tests/tt-metal/ttnn/cpp/ttnn/operations/matmul/matmul.cpp:127: tt...
4. **bmm**: /home/ubuntu/work/yolov12-high-res-tests/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:141: tt::...
5. **bmm**: /home/ubuntu/work/yolov12-high-res-tests/tt-metal/ttnn/cpp/ttnn/operations/matmul/matmul.cpp:127: tt...
... and 48 more similar failures

### Test Failed (29 failures)

**Operators Affected:** 13

**Examples:**
1. **addmm**: RuntimeError:...
2. **addmm**: RuntimeError...
3. **bmm**: RuntimeError: TT_...
4. **bmm**: RuntimeError: TT...
5. **bmm**: TypeError: rand(...
... and 24 more similar failures

### Type Error (8 failures)

**Operators Affected:** 4

**Examples:**
1. **bmm**: rand() received an invalid combination of arguments - got (tuple, dtype=torch.dtype), but expected o...
2. **bmm**: rand(...
3. **mm**: rand() received an invalid combination of arguments - got (tuple, dtype=torch.dtype), but expected o...
4. **mm**: rand() r...
5. **softmax**: softmax() received an invalid combination of arguments - got (Tensor), but expected one of:
... and 3 more similar failures

### Attribute Error (7 failures)

**Operators Affected:** 5

**Examples:**
1. **copy**: module 'torch' has no attribute 'copy_'
2. **dropout**: dropout
3. **dropout**: module 'ttnn' has no attribute 'dropout'
4. **identity**: 'Identity' object has no attribute 'shape'
5. **linalgvectornorm**: norm
... and 2 more similar failures

### Assertion Error (4 failures)

**Operators Affected:** 1

**Examples:**
1. **split**: PCC check failed for split output: list(expected_pytorch_result.shape)=[1, 768, 136, 240] vs list(ac...
2. **split**: PCC check failed for split output: list(expected_pytorch_result.shape)=[1, 1152, 136, 120] vs list(a...
3. **split**: PCC check failed for split output: list(expected_pytorch_result.shape)=[1, 768, 68, 120] vs list(act...
4. **split**: passed, f"PCC check

### Dimension Mismatch (3 failures)

**Operators Affected:** 3

**Examples:**
1. **addmm**: ttnn.matmul: The width of the first tensor must be equal to the height of the second tensor
2. **bmm**: ttnn.matmul: The width of the first tensor must be equal to the height of the second tensor
3. **mm**: ttnn.matmul: The width of the first tensor must be equal to the height of the second tensor

### Collection Error (2 failures)

**Operators Affected:** 1

**Examples:**
1. **linear**: ssinghal/tests/test_linear.py ________________
2. **linear**: ====================================

### Syntax Error (1 failures)

**Operators Affected:** 1

**Examples:**
1. **linear**: unexpected indent

## �� CRITICAL ISSUES REQUIRING IMMEDIATE ATTENTION

**3 Critical Failures Found:**

### Collection Error
**Affected Operators:** linear
**Impact:** Blocks basic functionality

### Syntax Error
**Affected Operators:** linear
**Impact:** Blocks basic functionality

## 🔧 TTNN IMPLEMENTATION GAPS

**Missing TTNN Operators:**
- **copy_**: Required by copy
- **dropout**: Required by dropout
- **norm**: Required by linalgvectornorm
- **shape**: Required by identity
- **silu_**: Required by silu_inplace

## 💡 RECOMMENDATIONS FOR NON-OOM FAILURES

### 🚨 IMMEDIATE ACTIONS (Critical Priority)

2. **Implement Missing TTNN Operators:**
   - Implement `ttnn.copy_` operator
   - Implement `ttnn.dropout` operator
   - Implement `ttnn.norm` operator
   - Implement `ttnn.shape` operator
   - Implement `ttnn.silu_` operator

3. **Fix Dimension Compatibility Issues:**
   - Review tensor shape requirements for operations
   - Add input validation and shape checking
   - Implement automatic shape broadcasting where appropriate

### ⚠️ HIGH PRIORITY ACTIONS

4. **Runtime Error Resolution:**
   - Debug and fix TT-Metal framework errors
   - Improve error handling and reporting
   - Add graceful fallbacks for unsupported operations

5. **Memory Layout Compatibility:**
   - Ensure consistent memory layout handling
   - Add automatic layout conversion where needed
   - Document memory layout requirements

### 🟡 MEDIUM PRIORITY ACTIONS

6. **Test Infrastructure Improvements:**
   - Fix test collection and discovery issues
   - Improve test timeout handling
   - Add better error reporting and logging

7. **Device Management:**
   - Improve device availability checking
   - Add device resource management
   - Implement device fallback strategies

## 📈 FAILURE STATISTICS SUMMARY

### By Severity
- **Critical**: 3 failures (1.1%)
- **High**: 279 failures (98.9%)
- **Medium**: 0 failures (0.0%)

### By Category
- **Implementation Issues**: 7 failures (2.5%)
- **Runtime Issues**: 136 failures (48.2%)
- **Configuration Issues**: 0 failures (0.0%)
- **Test Issues**: 123 failures (43.6%)
- **Compatibility Issues**: 11 failures (3.9%)

---

*Non-OOM Failure Analysis Report generated by TT-Metal Test Analyzer*
*Timestamp: 2025-08-18 23:22:05*
