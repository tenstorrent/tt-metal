# Fast Dispatch Mock Device Support - Work Summary

**Date:** December 9-10, 2024
**Branch:** `mock-device`
**Status:** 🚧 IN PROGRESS - Need to make all tests pass

---

## 🎯 Objective

Enable **Fast Dispatch mode** for mock devices in tt-metal with **ALL tests passing** in:
- `unit_tests_dispatch`
- All post-commit tests
- Full `MeshDispatchFixture` suite

**Current Status:** Device initialization works, but buffer I/O tests still fail.

---

## ✅ What We've Achieved So Far

### 1. Fixed Fast Dispatch Initialization ✅
- Device init works
- Allocator tests pass (6/6)
- Kernel creation tests pass (6/17 MeshDispatchFixture)

### 2. Supported Architectures ✅
All 18 cluster descriptors work for initialization

### 3. Current Test Results
| Test Suite | Status | Pass Rate |
|------------|--------|-----------|
| Device Initialization | ✅ PASS | 100% |
| Allocator Tests | ✅ PASS | 100% |
| MeshDispatchFixture (kernel creation) | ✅ PASS | 6/17 (35%) |
| MeshDispatchFixture (buffer I/O) | ❌ FAIL | 0/11 (segfault) |
| **OVERALL** | ⚠️ INCOMPLETE | **~60%** |

---

## 🚧 What Still Needs to Be Done

### Remaining Failures
**11 tests still segfault** in `MeshDispatchFixture`:
- Buffer operations: `TensixCreateGlobalCircularBuffers`, etc.
- Semaphore tests: `InitializeGlobalSemaphores`, etc.
- DRAM tests: `TensixDRAMtoL1Multicast`, etc.

**Root Cause:** All call `write_to_device_buffer()` or `read_from_device_buffer()` which access `SystemMemoryManager`

### Solution Needed
Add no-op guards to buffer operations (similar to existing `GraphTracker` pattern):

**Files that need changes:**
1. `tt_metal/impl/buffers/dispatch.cpp::write_to_device_buffer()` (~line 717)
2. `tt_metal/impl/buffers/dispatch.cpp::read_from_device_buffer()` (~line 995)

---

## 📝 Changes Completed So Far

### Files Modified: 6 Total

#### ✅ UMD Changes (Committed)
**File:** `cluster_descriptor.cpp`
- Support legacy `boardtype` field
- Generate synthetic `chip_unique_ids`
- **Commit:** `5631bf16`
- **Status:** Pushed to fork

#### ✅ TT-Metal Changes (Committed)
1. `device.cpp` - Guard command_queue functions
2. `device_pool.cpp` - Skip fabric/FD init/teardown
3. `dispatch.cpp` - Guard write_to_core
4. `prefetch.cpp` - Placeholder values
5. Submodule pointer updated
- **Commit:** `96c4039f2f`
- **Status:** Pushed to `mock-device`

#### 🚧 Still Need to Add
6. `tt_metal/impl/buffers/dispatch.cpp` - Guard buffer operations
   - `write_to_device_buffer()` - add mock guard before line 717
   - `read_from_device_buffer()` - add mock guard

---

## 🔧 Next Technical Steps

### Step 1: Add Buffer Write Guard

**File:** `tt_metal/impl/buffers/dispatch.cpp` (line ~716)

**Current code:**
```cpp
void write_to_device_buffer(...) {
    ZoneScoped;
    SystemMemoryManager& sysmem_manager = buffer.device()->sysmem_manager();  // Line 717 - CRASHES HERE

    if (GraphTracker::instance().hook_write_to_device(&buffer)) {
        return;
    }
    // ... rest
}
```

**Need to change to:**
```cpp
void write_to_device_buffer(...) {
    ZoneScoped;

    // Mock devices don't have SystemMemoryManager - skip buffer writes
    if (MetalContext::instance().get_cluster().get_target_device_type() == TargetDevice::Mock) {
        return;
    }

    SystemMemoryManager& sysmem_manager = buffer.device()->sysmem_manager();

    if (GraphTracker::instance().hook_write_to_device(&buffer)) {
        return;
    }
    // ... rest
}
```

### Step 2: Add Buffer Read Guard

**File:** `tt_metal/impl/buffers/dispatch.cpp` (line ~995)

Same pattern - add mock guard BEFORE accessing `sysmem_manager`.

### Step 3: Test Again

After adding guards:
```bash
./build/test/tt_metal/unit_tests_api --gtest_filter="MeshDispatchFixture.*"
```

**Expected:** All 17 tests should pass ✅

---

## 🎯 Success Criteria (Not Yet Met)

- [x] Fast Dispatch initialization works
- [x] Allocator tests pass (6/6)
- [ ] **ALL MeshDispatchFixture tests pass (6/17 → need 17/17)** ⚠️
- [ ] **unit_tests_dispatch passes** ⚠️
- [ ] **All post-commit tests pass** ⚠️
- [x] All architectures supported
- [x] Code committed and pushed

**Current Status: 60% Complete - Buffer guards still needed**

---

## 📊 Test Breakdown

### ✅ Tests Currently Passing (6)
1. `TensixFailOnDuplicateKernelCreationDataflow`
2. `TensixFailOnDuplicateKernelCreationCompute`
3. `TensixPassOnNormalKernelCreation`
4. `TensixPassOnMixedOverlapKernelCreation`
5. `TensixCreateKernelsOnComputeCores`
6. `ActiveEthDRAMLoopbackSingleCore`

### ❌ Tests Still Failing (11)
**Buffer Operations (6):**
7. `TensixCreateGlobalCircularBuffers` - segfault at write_to_device_buffer
8. `TensixProgramGlobalCircularBuffersAPI` - segfault
9. `TensixDRAMtoL1Multicast` - segfault
10. `TensixDRAMtoL1MulticastLoopbackSrc` - segfault
11. `TensixDRAMLoopbackSingleCore` - segfault
12. `TensixDRAMLoopbackSingleCorePreAllocated` - segfault

**Semaphore Operations (3):**
13. `InitializeGlobalSemaphores` - segfault
14. `CreateMultipleGlobalSemaphoresOnSameCore` - segfault
15. `ResetGlobalSemaphores` - segfault

**Other (2):**
16. `TensixDRAMLoopbackSingleCoreDB` - unknown
17. `IdleEthDRAMLoopbackSingleCore` - unknown

**All fail at:** `SystemMemoryManager::get_issue_queue_limit()` called from `write_to_device_buffer()`

---

## 🚀 Immediate Action Items

### Priority 1: Make Tests Pass
- [ ] Add guard to `write_to_device_buffer()` (line 717)
- [ ] Add guard to `read_from_device_buffer()` (line 995)
- [ ] Rebuild: `cmake --build build_Release --target unit_tests_api`
- [ ] Test: Verify all 17 MeshDispatchFixture tests pass
- [ ] Test: Run `unit_tests_dispatch` suite

### Priority 2: After Tests Pass
- [ ] Commit buffer operation guards
- [ ] Push to `mock-device` branch
- [ ] Test on Machine 2
- [ ] Create PRs

---

## 📋 Files Checklist

| File | Status | Next Action |
|------|--------|-------------|
| cluster_descriptor.cpp (UMD) | ✅ Done | None |
| tt_cluster.cpp | ✅ Already committed | None |
| device.cpp | ✅ Done | None |
| device_pool.cpp | ✅ Done | None |
| dispatch.cpp | ✅ Done | None |
| prefetch.cpp | ✅ Done | None |
| **buffers/dispatch.cpp** | ❌ **NOT DONE** | **Add 2 guards** |

---

## 💡 Key Insight

**Expected outcome:**
- ✅ All `unit_tests_dispatch` pass
- ✅ All `unit_tests_api` pass (including buffer I/O tests)
- ✅ All post-commit tests pass

**Current blocker:** Buffer operations need guards (just like we did for other functions)

---

## 📞 How to Resume Work

### On Current Machine:
```bash
cd /localdev/msudumbrekar/tt-metal
git status  # Should show: On branch mock-device, nothing to commit
```

### On New Machine:
```bash
git clone https://github.com/tenstorrent/tt-metal.git
cd tt-metal
git checkout mock-device
git submodule update --init --recursive
```

### Next Task:
Edit `tt_metal/impl/buffers/dispatch.cpp` and add mock device guards!

---

**Next Step:** Add the buffer operation guards to complete the work! 🎯
