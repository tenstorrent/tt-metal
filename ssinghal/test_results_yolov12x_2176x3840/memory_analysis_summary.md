# TT-Metal Operator Memory Analysis Summary

**Generated:** 2025-08-18 19:35:37
**Analysis of:** 46 operators with 680 total tests

## 🎯 Executive Summary

| Metric | Count | Percentage |
|--------|-------|------------|
| **Total Operators Tested** | 46 | 100% |
| **Total Tests Executed** | 680 | 100% |
| **Tests Passed** | 432 | 63.5% |
| **Tests Failed** | 248 | 36.5% |
| **Operators with Memory Issues** | 37 | 80.4% |
| **Memory-Related Failures** | 37 | - |

## 🧠 Memory Issues Analysis

### Critical Memory Problems

The following operators have **explicit Out-of-Memory (OOM) errors** with specific tensor shapes:

#### **CLAMPMIN Operator** - Multiple OOM Failures
- **Total Tests:** 20 (all failed)
- **OOM Shapes:**
  - `[1, 3, 2176, 3840]` → **~755MB memory**
  - `[1, 96, 1088, 1920]` → **~378MB memory**
  - `[1, 192, 544, 960]` → **~189MB memory**
  - `[1, 384, 272, 480]` → **~94MB memory**
  - `[1, 96, 2176, 1920]` → **~755MB memory**
  - `[1, 192, 1088, 960]` → **~378MB memory**
  - `[1, 384, 544, 480]` → **~189MB memory**
  - `[1, 768, 272, 240]` → **~94MB memory**

## 📊 Operator Status Breakdown

| Operator | Total Tests | Passed | Failed | Memory Issues | Status |
|----------|-------------|--------|--------|---------------|---------|
| **add** | 14 | 14 | 0 | ⚠️ 1 | ✅ PASS |
| **addmm** | 54 | 0 | 54 | ⚠️ 1 | ❌ FAIL |
| **bmm** | 0 | 0 | 0 | - | ✅ PASS |
| **cat** | 16 | 16 | 0 | ⚠️ 1 | ✅ PASS |
| **clamp** | 2 | 0 | 2 | - | ❌ FAIL |
| **clampmin** | 20 | 0 | 20 | 🚨 1 | ❌ FAIL |
| **clone** | 16 | 16 | 0 | ⚠️ 1 | ✅ PASS |
| **concat** | 16 | 16 | 0 | ⚠️ 1 | ✅ PASS |
| **copy** | 6 | 0 | 6 | - | ❌ FAIL |
| **div** | 18 | 18 | 0 | ⚠️ 1 | ✅ PASS |
| **dropout** | 24 | 0 | 24 | ⚠️ 1 | ❌ FAIL |
| **expand** | 0 | 0 | 0 | ⚠️ 1 | ✅ PASS |
| **gelu** | 22 | 22 | 0 | - | ✅ PASS |
| **geluactivation** | 14 | 14 | 0 | ⚠️ 1 | ✅ PASS |
| **hardtanh** | 14 | 14 | 0 | ⚠️ 1 | ✅ PASS |
| **identity** | 16 | 0 | 16 | ⚠️ 1 | ❌ FAIL |
| **leakyrelu** | 0 | 0 | 0 | ⚠️ 1 | ✅ PASS |
| **linalgvectornorm** | 20 | 0 | 20 | ⚠️ 1 | ❌ FAIL |
| **linear** | 84 | 84 | 0 | ⚠️ 1 | ✅ PASS |
| **log** | 2 | 2 | 0 | - | ✅ PASS |
| **maxpool2d** | 0 | 0 | 0 | ⚠️ 1 | ✅ PASS |
| **mean** | 2 | 2 | 0 | - | ✅ PASS |
| **mish** | 16 | 16 | 0 | ⚠️ 1 | ✅ PASS |
| **mm** | 20 | 0 | 20 | - | ❌ FAIL |
| **mul** | 14 | 14 | 0 | ⚠️ 1 | ✅ PASS |
| **native_batch_norm** | 0 | 0 | 0 | ⚠️ 1 | ✅ PASS |
| **ones** | 20 | 20 | 0 | ⚠️ 1 | ✅ PASS |
| **permute** | 16 | 16 | 0 | ⚠️ 1 | ✅ PASS |
| **relu** | 16 | 16 | 0 | ⚠️ 1 | ✅ PASS |
| **sigmoid** | 24 | 24 | 0 | ⚠️ 1 | ✅ PASS |
| **silu** | 16 | 16 | 0 | ⚠️ 1 | ✅ PASS |
| **silu_inplace** | 20 | 0 | 20 | ⚠️ 1 | ❌ FAIL |
| **softmax** | 10 | 0 | 10 | ⚠️ 1 | ❌ FAIL |
| **softplus** | 16 | 16 | 0 | ⚠️ 1 | ✅ PASS |
| **split** | 14 | 6 | 8 | ⚠️ 1 | ❌ FAIL |
| **split_with_sizes** | 10 | 0 | 10 | ⚠️ 1 | ❌ FAIL |
| **splitwithsizes** | 0 | 0 | 0 | ⚠️ 1 | ✅ PASS |
| **stack** | 0 | 0 | 0 | - | ✅ PASS |
| **sub** | 14 | 14 | 0 | ⚠️ 1 | ✅ PASS |
| **tanh** | 16 | 16 | 0 | ⚠️ 1 | ✅ PASS |
| **topk** | 2 | 0 | 2 | - | ❌ FAIL |
| **transpose** | 16 | 16 | 0 | ⚠️ 1 | ✅ PASS |
| **unsafeview** | 16 | 4 | 12 | ⚠️ 1 | ❌ FAIL |
| **unsqueeze** | 16 | 16 | 0 | ⚠️ 1 | ✅ PASS |
| **upsample_nearest2d** | 12 | 0 | 12 | ⚠️ 1 | ❌ FAIL |
| **view** | 16 | 4 | 12 | ⚠️ 1 | ❌ FAIL |

## 🔍 Common Error Patterns

### 1. Matrix Multiplication Dimension Mismatches
**Operators affected:** `addmm`, `mm`
**Error:** "The width of the first tensor must be equal to the height of the second tensor"
**Example shapes causing issues:**
- `[27, 256]` × `[27, 256]` (should be `[27, 256]` × `[256, X]`)
- `[1000, 1024]` × `[1000, 1024]` (should be `[1000, 1024]` × `[1024, X]`)

### 2. Missing Module Attributes
**Operators affected:** `dropout`, `copy`
**Error:** "module 'ttnn' has no attribute 'dropout'"
**Issue:** TTNN module doesn't implement these operators

### 3. Syntax/Parameter Errors
**Operators affected:** `topk`, `bmm`
**Error:** "topk() missing 1 required positional arguments: 'k'"
**Issue:** Test implementation missing required parameters

### 4. Out of Memory Errors
**Operators affected:** `clampmin` and others
**Error:** Explicit OOM failures with large tensor shapes
**Critical shapes:** Any tensor requiring >100MB memory

## 💡 Recommendations

### 🚨 **Immediate Actions Required:**

1. **Fix Memory Management in CLAMPMIN:**
   - Tensor shapes `[1, 3, 2176, 3840]` (~755MB) causing OOM
   - Implement memory-efficient processing or chunking
   - Consider reducing test tensor sizes

2. **Fix Matrix Multiplication Logic:**
   - Operators `addmm` and `mm` have incorrect tensor dimension handling
   - Need to transpose or reshape tensors for proper matrix multiplication
   - Review test case tensor generation

3. **Implement Missing Operators:**
   - `dropout` not available in TTNN module
   - `copy_` function missing in torch compatibility layer

### 📊 **Memory Optimization Priorities:**

| Priority | Operator | Memory Issue | Recommended Action |
|----------|----------|--------------|-------------------|
| **HIGH** | clampmin | OOM at 755MB | Implement chunking/tiling |
| **HIGH** | addmm | Dimension errors | Fix tensor reshaping |
| **HIGH** | mm | Dimension errors | Fix tensor reshaping |
| **MEDIUM** | upsample_nearest2d | General failures | Review implementation |
| **MEDIUM** | view | Partial failures | Check tensor compatibility |

### 🔧 **Technical Solutions:**

1. **For OOM Issues:**
   ```python
   # Implement tensor chunking for large inputs
   if tensor.numel() * 2 > max_memory_bytes:  # 2 bytes for bfloat16
       process_in_chunks(tensor, chunk_size=optimal_chunk_size)
   ```

2. **For Matrix Multiplication:**
   ```python
   # Fix dimension compatibility
   if a.shape[-1] != b.shape[-2]:
       b = b.transpose(-2, -1)  # or reshape as needed
   ```

3. **For Memory Monitoring:**
   ```python
   # Add memory checks before operations
   estimated_memory = calculate_tensor_memory(shape, dtype)
   if estimated_memory > device_memory_limit:
       use_alternative_implementation()
   ```

## 📁 **File Locations:**
- **Comprehensive Report:** `ssinghal/test_results/comprehensive_analysis_report.txt`
- **Raw Data:** `ssinghal/test_results/detailed_analysis_data.json`
- **Individual Results:** `ssinghal/test_results/test_*_results.txt`

---
*Analysis generated by TT-Metal Test Analyzer*
