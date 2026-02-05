# TT-Metal Codebase Improvement Opportunities

This document catalogs potential improvements identified through a comprehensive codebase scan. Each item is suitable for a separate PR.

---

## 🔴 CRITICAL - Deadline Approaching

### 1. Remove Deprecated MeshDevice APIs
- **Priority:** CRITICAL (Deadline: Feb 28, 2026)
- **Files:**
  - `tt_metal/api/tt-metalium/mesh_device.hpp`
  - `tt_metal/api/tt-metalium/mesh_device_view.hpp`
- **Issue:** `get_devices()` methods marked `[[deprecated]]` with removal date 28-02-2026
- **Error message:** "Deprecated, retrieving physical devices can fail in distributed contexts"
- **Action:** Find all callers and migrate to new API, then remove deprecated methods
- **Tracking:** Check if there's an existing issue for this migration

---

## 🟠 HIGH PRIORITY - Performance Improvements

### 2. FP32 SFPU Untilize Support (COMPLEX - FUTURE WORK)
- **Priority:** HIGH (but complex)
- **Status:** 🔬 **INVESTIGATED** (2026-02-05) - Requires deep LLK expertise
- **GitHub Issues:** #30400 (closed), #33795 (open, assigned to ntarafdar)
- **Files (~11 affected):**
  - `ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_*_program_factory.cpp`
  - `ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/factories/*`
- **Problem:** FP32 with width > 8 tiles can't use `pack_untilize` (MAX_PACK_UNTILIZE_WIDTH=8). Falls back to slow `untilize.cpp` which explicitly overrides `UnpackToDestFp32` → `Default`, causing TF32 truncation and precision loss.
- **TODO marker:** `// TODO: We need SFPU untilize for FP32 (#30400, #33795)`
- **Why it's hard:** The override to `Default` is intentional - simply using `UnpackToDestFp32` with the slow kernel apparently doesn't work (otherwise they'd have done it). Requires writing a new SFPU-based untilize kernel that preserves FP32 precision.
- **Impact:** Would fix FP32 precision loss during untilize operations
- **Complexity:** HIGH - requires LLK/compute kernel expertise, new kernel implementation
- **Plan:** Coordinate with ntarafdar (issue assignee) for future work

### ~~3. LayerNorm Circular Buffer Wait Optimization~~ (NO MEASURABLE IMPACT)
- **Priority:** ~~HIGH~~ → **SKIP**
- **Status:** ⚠️ **TESTED - NO PERF GAIN** (2026-02-05)
- **Files:**
  - `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm.cpp` (line 265)
  - `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm_welford.cpp` (line 297)
- **Problem:** Currently waits for full block size on beta CB when only first height token is needed
- **TODO marker:** `// TODO: optimization - only wait on first ht`
- **Finding:** `cb_wait_front()` when data is already present returns immediately (just a counter check). The redundant calls add ~10-50ns overhead each, which is unmeasurable in ms-scale operations. Tested with NCHt up to 512 - baseline and "optimized" versions showed identical performance within noise.
- **Recommendation:** Not worth a PR. The TODO comment is technically correct but the optimization is negligible on current hardware.

### 4. Untilize Block Size Tuning
- **Priority:** HIGH
- **File:** `ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_parallelize_column_program_factory.cpp` (line 55)
- **Problem:** Each untilize block is a single tile, limiting performance
- **TODO marker:** `// TODO increase block size to increase untilize performance, currently each untilize block is a single tile`
- **Impact:** Direct untilize performance improvement
- **Tradeoff:** Uses more L1 memory
- **Complexity:** LOW-MEDIUM - parameter tuning with benchmarking

### 5. Double Map Lookup Elimination
- **Priority:** HIGH
- **Files (30+ instances):**
  - `tt_metal/impl/lightmetal/lightmetal_capture.cpp` (lines 96, 124, 150, 176)
  - `tt_metal/impl/lightmetal/lightmetal_replay_impl.cpp` (lines 171, 186, 200, 215, 229)
  - `tt_metal/impl/kernels/kernel.cpp` (lines 299, 573, 579)
  - `tt_metal/impl/dispatch/dispatch_core_manager.cpp` (lines 205, 252, 258, 259)
  - `tt_metal/impl/dispatch/debug_tools.cpp` (lines 57, 60, 61, 65)
  - `tt_metal/impl/dispatch/data_collector.cpp` (lines 110, 165)
  - `tt_metal/impl/dispatch/topology.cpp` (lines 603, 613, 812)
  - `tt_metal/impl/program/dispatch.cpp` (lines 165, 179, 247, 372, 2703)
  - `tt_metal/impl/program/program_impl.hpp` (line 241)
- **Problem:** Code performs `.find()` then `.at()` or multiple `.at()` calls on same key
- **Example:**
  ```cpp
  // Current - TWO lookups
  CoreCoord core = available_dispatch_cores_by_device.at(device_id).front();
  available_dispatch_cores_by_device.at(device_id).pop_front();

  // Fixed - ONE lookup
  auto& cores = available_dispatch_cores_by_device.at(device_id);
  CoreCoord core = cores.front();
  cores.pop_front();
  ```
- **Impact:** High - these are in device init, dispatch, kernel execution hot paths
- **Complexity:** LOW - mechanical refactoring

### 6. Allocator Statistics Caching
- **Priority:** HIGH
- **File:** `tt_metal/impl/allocator/algorithms/free_list_opt.cpp` (line 394)
- **Problem:** Statistics are recomputed on every `get_statistics()` call
- **TODO marker:** `// TODO: Cache the statistics`
- **Impact:** Reduces overhead when statistics are queried frequently
- **Complexity:** LOW - add caching with invalidation on allocation/deallocation

---

## 🟡 MEDIUM PRIORITY - Architecture & Operations

### 7. Wormhole Prefetch Barrier Optimization
- **Priority:** MEDIUM
- **File:** `tt_metal/impl/dispatch/kernels/cq_prefetch.cpp`
- **Lines:** 617, 660, 750, 776, 849, 896, 1009, 1030
- **Problem:** Generic barrier calls instead of tagged barriers on Wormhole
- **TODO marker:** `// TODO(pgk); we can do better on WH w/ tagging`
- **Impact:** Architecture-specific performance gains on Wormhole chips
- **Complexity:** MEDIUM - requires WH-specific knowledge and testing

### 8. Group Attention Matmul Core Over-allocation
- **Priority:** MEDIUM
- **File:** `ttnn/cpp/ttnn/operations/experimental/matmul/group_attn_matmul/device/group_attn_matmul_program_factory.cpp` (line 40)
- **Problem:** Always multicasts to 32 cores even when Q_HEADS < 32
- **TODO marker:** `// TODO: Currently, we always mcast to at least 32 cores even when Q_HEADS < 32; we can optimize if we pass in`
- **Impact:** Wasted multicast bandwidth for smaller attention configurations
- **Complexity:** MEDIUM - needs parameter plumbing

### 9. CCL Packing Efficiency
- **Priority:** MEDIUM
- **Files:**
  - `ttnn/cpp/ttnn/operations/ccl/common/kernels/ccl_send.cpp` (line 334)
  - `ttnn/cpp/ttnn/operations/ccl/common/kernels/ccl_send_reader_two_input.cpp` (line 237)
  - `ttnn/cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp` (lines 169, 233)
- **Problem:** Inefficient packing implementation acknowledged in comments
- **Comment:** "Implemented really inefficiently for now - in the future we can do more efficient packing"
- **Impact:** Collective communication operations performance
- **Complexity:** MEDIUM-HIGH

### 10. Pool Index Storage Optimization
- **Priority:** MEDIUM
- **Files:**
  - `ttnn/cpp/ttnn/operations/pool/generic/device/kernels/dataflow/reader_pool_2d.cpp` (line 134)
  - `ttnn/cpp/ttnn/operations/pool/generic/device/kernels/dataflow/reader_mpwi.cpp` (line 149)
- **Problem:** Sub-optimal index CB packing for multi-channel pooling
- **TODO marker:** `// TODO for c > 1 we could optimize by storing two values per write`
- **Impact:** Reduces CB overhead for multi-channel pooling operations
- **Complexity:** LOW-MEDIUM

### 11. Circular Buffer Allocation Per Sub-device
- **Priority:** MEDIUM
- **File:** `tt_metal/impl/program/program.cpp` (line 919)
- **Problem:** CB allocation/validation not optimized for sub-device usage patterns
- **TODO marker:** `// TODO: Circular buffer allocation and validation could be better optimized by determining usage per sub-device`
- **Impact:** Memory efficiency for multi-sub-device programs
- **Complexity:** MEDIUM

### 12. Halo L1 Tradeoff Tuning
- **Priority:** MEDIUM
- **GitHub Issue:** #19980
- **Files:**
  - `ttnn/cpp/ttnn/operations/sliding_window/halo/device/untilize_with_halo_program_factory.cpp` (line 24)
  - `ttnn/cpp/ttnn/operations/sliding_window/halo/device/halo_device_operation.cpp` (line 19)
- **TODO marker:** `// TODO: Look into increasing this to tradeoff some L1 for performance (#19980)`
- **Impact:** Sliding window / halo operations performance
- **Complexity:** LOW-MEDIUM - parameter tuning

### 13. Conv Read Coalescing Cleanup
- **Priority:** MEDIUM
- **Files:**
  - `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/reader_conv_activations_padded_with_halo_3x3_weights_v2.cpp` (line 59)
  - `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/reader_depthwise_conv1d.cpp` (line 52)
  - `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2.cpp` (line 203)
- **Problem:** Read coalescing optimization exists but implementation is messy
- **TODO marker:** `// TODO: need to make the read coalescing optimization cleaner`
- **Impact:** Code quality + potential perf through better compiler optimization
- **Complexity:** MEDIUM

### 14. Binary Op Define Generation Optimization
- **Priority:** MEDIUM
- **File:** `ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_utils.cpp`
- **Lines:** 65-130+ (50+ instances of `.merge()`)
- **Problem:** Sequential `.merge()` calls on maps in switch statements
- **Impact:** Kernel compilation time
- **Complexity:** LOW-MEDIUM - consolidate merge operations

---

## 🟢 LOWER PRIORITY - Technical Debt Cleanup

### 15. Deprecated Dataflow APIs Removal
- **Priority:** LOW
- **Files:**
  - `tt_metal/hw/inc/internal/dataflow/dataflow_api_addrgen.h`
  - `tt_metal/hw/inc/api/dataflow/dataflow_api.h`
- **Problem:** Multiple APIs marked "DEPRECATED AND WILL BE REMOVED SOON"
- **Action:** Migrate callers to `<typename AddrGen> get_noc_addr` template versions
- **Complexity:** MEDIUM - need to find and update all callers

### 16. Legacy Binary Operation Path Consolidation
- **Priority:** LOW
- **Files:**
  - `ttnn/cpp/ttnn/operations/eltwise/binary/binary.cpp`
  - `ttnn/cpp/ttnn/operations/eltwise/binary/binary.hpp`
- **Problem:** `is_legacy_only()` function routes some ops to legacy path instead of `binary_ng`
- **Impact:** Code maintenance burden
- **Complexity:** HIGH - needs careful validation of all binary ops

### 17. UMD Compatibility Shim Removal
- **Priority:** LOW
- **Files:**
  - `tt_metal/third_party/umd/device/api/umd/device/types/tensix_soft_reset_options.hpp`
  - `tt_metal/third_party/umd/device/api/umd/device/types/xy_pair.hpp`
  - `tt_metal/third_party/umd/device/api/umd/device/types/core_coordinates.hpp`
- **Problem:** Compatibility shims with TODOs to remove once clients migrate
- **Complexity:** LOW-MEDIUM

### 18. shared_ptr<bool> Pattern Fix
- **Priority:** LOW
- **File:** `ttnn/cpp/ttnn/operations/data_movement/common/common.hpp` (lines 118-120)
- **Problem:** Uses `std::shared_ptr<bool>` for lambda capture when mutable capture would work
- **Impact:** Minor - unnecessary heap allocation and atomic operations
- **Complexity:** LOW

### 19. Deprecated blockfloat Packing Function
- **Priority:** LOW
- **File:** `tt_metal/impl/data_format/blockfloat_common.hpp` (line 27)
- **Problem:** `pack_fp32_vec_as_bfp_tiles()` marked `[[deprecated]]`
- **Action:** Migrate to `pack_as_bfp_tiles()`
- **Complexity:** LOW

### 20. SubDevice Class Review
- **Priority:** LOW
- **File:** `tt_metal/api/tt-metalium/sub_device.hpp` (line 20)
- **Problem:** TODO to revisit class - may be redundant
- **TODO marker:** `// TODO: Revisit this class and either remove it or bring sub-device APIs over`
- **Complexity:** MEDIUM - needs architectural review

---

## 📊 Summary by Complexity

| Complexity | Issues |
|------------|--------|
| LOW (hours) | #3, #4, #5, #6, #10, #18, #19 |
| LOW-MEDIUM | #12, #14, #17 |
| MEDIUM | #7, #8, #11, #13, #15, #20 |
| MEDIUM-HIGH | #2, #9 |
| HIGH | #1 (urgent), #16 |

---

## 🎯 Recommended PR Order for Maximum Impact

### Phase 1: Quick Wins (1-2 days each)
1. **#3 LayerNorm CB Wait** - Easy, measurable perf gain
2. **#5 Double Map Lookups** - Systematic, shows attention to detail
3. **#6 Allocator Stats Caching** - Clean fix with clear benefit

### Phase 2: Tracked Issues (visibility)
4. **#2 FP32 SFPU Untilize** - Has GitHub issues, high visibility
5. **#12 Halo L1 Tuning** - Has GitHub issue #19980

### Phase 3: Medium Effort
6. **#4 Untilize Block Size** - Perf improvement with benchmarks
7. **#10 Pool Index Storage** - Targeted optimization

### Phase 4: Cleanup
8. **#1 MeshDevice Deprecation** - Must do before deadline
9. **#14 Binary Op Defines** - Compilation time improvement

---

## Notes

- All line numbers are approximate and may shift as codebase evolves
- Each item should be a separate PR for clean review
- Performance claims should be validated with benchmarks before/after
- Some items may have dependencies on each other

*Generated: 2026-02-05*
