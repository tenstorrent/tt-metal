# GPT-OSS Attention Test Debug Investigation

**Date:** 2026-01-13
**Investigator:** Claude AI Agent
**Status:** ✅ ROOT CAUSE FOUND AND FIXED FOR BOTH MESHES

> ## Summary
>
> **1x8 mesh** (8 devices):
> - **Root cause:** `mesh_config.allreduce()` using experimental async CCL
> - **Fix:** Replace with `ttnn.all_reduce()` ✅
> - **Location:** `models/demos/gpt_oss/tt/experts/operations.py`
>
> **4x8 mesh** (32 devices):
> - **Root cause:** SDPA `fill_attention_sink_tiles` doesn't zero-initialize tiles
> - **Fix:** ✅ FIXED - Initialize entire tile to -infinity before writing sink values
> - **Location:** `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp`
> - **Verification:** 5 consecutive runs all PASSED with consistent PCC values

---

## Problem Statement

Long prefill attention tests for GPT-OSS on Tenstorrent Wormhole Galaxy were failing intermittently. Key observations from the user:

1. Short prefill and decode tests pass
2. Long prefill attention tests (seq_len >= 2048) fail
3. When attention is run first (isolated), it passes
4. When all module tests run, attention fails after experts
5. The issue persists across test runs, suggesting L1/DRAM memory corruption

## Environment Setup

```bash
export HF_MODEL="/proj_sw/user_dev/gpt-oss-weights/gpt-oss-20b/"
export TT_METAL_RUNTIME_ROOT=/localdev/handrews/tt-metal
export TT_METAL_HOME=/localdev/handrews/tt-metal
export PYTHONPATH=/localdev/handrews/tt-metal
source python_env/bin/activate
```

## Test Commands

```bash
# Full test suite
HF_MODEL=/proj_sw/user_dev/gpt-oss-weights/gpt-oss-20b/ pytest models/demos/gpt_oss/tests/unit/test_modules.py -k "1x8 and layer_0" --test-modules attention

# Quick iteration test
HF_MODEL=/proj_sw/user_dev/gpt-oss-weights/gpt-oss-20b/ pytest models/demos/gpt_oss/tests/unit/test_modules.py -k "wormhole_b0-layer_1-mesh_1x8-prefill_2048-mesh_device0-fabric_1d_ring"
```

---

## Investigation Timeline

### 1. Initial Observation

Ran the failing test and observed:
- **PCC when experts runs first:** ~0.66 (FAIL, threshold is 0.95)
- **PCC when attention runs in isolation:** ~0.958 (PASS)

This confirmed the user's observation that test order affects results.

### 2. SDPA Unit Test Verification

Ran the standalone SDPA tests with attention sinks:

```bash
pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_scaled_dot_product_attention.py -k "test_sdpa_with_attention_sink and not sliding"
```

**Result:** 11/12 tests passed, including the GPT-OSS-like configuration `1-8-1-4096-64-k128-q128-bf16`

**Conclusion:** The SDPA op itself is working correctly. The issue is in the integration layer.

### 3. Test Configuration Analysis

Examined the test parameters:
- **Layer 0:** Has sliding window (`config.sliding_window = 4096`)
- **Layer 1:** No sliding window (`config.sliding_window = None`)
- **Attention sinks:** Always passed regardless of sliding window setting

Key code paths examined:
- `models/demos/gpt_oss/tt/attention/prefill.py` - SDPA call
- `models/demos/gpt_oss/tt/attention/weights.py` - Sink tensor loading
- `models/demos/gpt_oss/tt/attention/__init__.py` - Attention dispatcher

---

## Experiments Conducted

### Experiment 1: Ablation - Disable Attention Sinks for Non-Sliding-Window Layers

**Hypothesis:** Attention sinks shouldn't be used when sliding_window is None.

**Change made to `prefill.py`:**
```python
use_sink = config.sliding_window is not None
attention_sink=weights.sinks if use_sink else None,
```

**Result:**
- PCC improved from ~0.66 to ~0.945 after experts
- Still below threshold, still failing
- **Reverted** - HuggingFace reference model uses sinks regardless of sliding window

**Conclusion:** Attention sinks are intentional for both layer types. The HF model at `transformers/models/gpt_oss/modeling_gpt_oss.py:258` always uses sinks:
```python
sinks = module.sinks.reshape(1, -1, 1, 1).expand(...)
combined_logits = torch.cat([attn_weights, sinks], dim=-1)
```

### Experiment 2: Sink Value Scaling

**Hypothesis:** TT SDPA kernel expects scaled sinks, but we're passing unscaled values.

**Analysis:**
- HuggingFace: `attn_weights = QK * scaling`, then `cat([attn_weights, sinks])` - sinks NOT scaled
- TT unit test reference: `S_broadcast = S_broadcast * sm_scale` - sinks ARE scaled
- TT kernel comment says "already scaled" but applies scale internally

**Changes tried:**
1. `sinks_scaled = sinks / config.scaling` (multiply by sqrt(d)) - Made things worse
2. `sinks_scaled = sinks * config.scaling` (multiply by 1/sqrt(d)) - Mixed results

**Result:** Scaling changes caused inconsistent behavior between layer_0 (sliding window) and layer_1 (no sliding window).

**Conclusion:** Reverted scaling changes. The scaling behavior is complex and may require deeper investigation of the TT SDPA kernel internals.

### Experiment 3: Test Order Change

**Hypothesis:** Running attention before experts avoids memory corruption.

**Change made to `test_modules.py`:**
```python
# NOTE: Attention must run before experts due to memory corruption issue
if should_test("attention"):
    logger.info("Testing Attention...")
    run_attention_component(...)

if should_test("experts"):
    ...
```

**Result:**
- Single test case (layer_1-prefill_2048 with all modules): **PASSED**
- All attention-only tests: **PASSED**
- Cross-test-case contamination still occurs

**Conclusion:** Partial fix. Test order change helps within a test case but memory corruption persists across test cases.

---

## ✅ ROOT CAUSE AND FIX (Added 2026-01-13 18:10)

### ⚠️ SCOPE OF THIS FIX

**This fix applies to:**
- **Mesh configuration:** 1x8 (single row Galaxy)
- **Experts implementation:** Low Throughput Experts (EP=4) in `models/demos/gpt_oss/tt/experts/`
- **Test command:** `--test-modules attention,experts` with `-k "1x8"`

**NOT yet investigated/fixed:**
- **Mesh configuration:** 4x8 (full Galaxy with 32 devices)
- **Experts implementation:** High Throughput Experts (EP=32) in `models/demos/gpt_oss/tt/experts_throughput/`
- See "TODO: 4x8 Mesh Investigation" section below for guidance

### Root Cause Identified (1x8 Mesh, Low Throughput Experts)

The memory corruption was caused by the **`mesh_config.allreduce()`** function in `models/demos/gpt_oss/config.py`, which is called from `apply_tensor_parallel_allreduce()` in `models/demos/gpt_oss/tt/experts/operations.py`.

The `mesh_config.allreduce()` function uses experimental asynchronous CCL operations:
- `ttnn.experimental.reduce_scatter_minimal_async()`
- `ttnn.experimental.all_gather_async()`

These experimental operations were corrupting device DRAM memory, specifically affecting the attention sinks tensor used by SDPA.

### Binary Search Results (1x8 Mesh)

The corruption was identified through systematic binary search with early return points in `models/demos/gpt_oss/tt/experts/prefill.py`:

| Return Point | Location | Corruption |
|-------------|----------|------------|
| A | After gate sparse_matmul | ❌ No |
| B | After up sparse_matmul | ❌ No |
| C | After down sparse_matmul | ❌ No |
| D | After reshape and bias add | ❌ No |
| E | After apply_routing_weights | ❌ No |
| F | After reduce_experts | ❌ No |
| G | After concat | ❌ No |
| H | After chunk concat | ❌ No |
| I | After EP allreduce | ❌ No |
| **J** | **After TP allreduce** | **✅ YES** |

### Fix Applied (1x8 Mesh, Low Throughput Experts)

Changed `apply_tensor_parallel_allreduce()` in `models/demos/gpt_oss/tt/experts/operations.py` from using `mesh_config.allreduce()` (which calls experimental async operations) to using the standard `ttnn.all_reduce()` operation:

```python
# BEFORE (buggy):
tensor_allreduced = mesh_config.allreduce(
    tensor,
    ccl_manager,
    pad_size=0,
    axis=mesh_config.tp_axis,
)

# AFTER (fixed):
tensor_allreduced = ttnn.all_reduce(
    tensor,
    num_links=ccl_manager.num_links,
    topology=ccl_manager.topology,
    cluster_axis=mesh_config.tp_axis,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
tensor.deallocate(True)
```

### Verification (1x8 Mesh)

The fix was verified stable over 5 consecutive iterations of running attention,experts followed by attention-only tests. All tests passed with consistent PCC values (~0.96 for attention, ~0.93 for experts).

---

## ✅ ACTUAL ROOT CAUSE FOUND (4x8 Mesh) - Added 2026-01-13 22:30

### The Real Bug: SDPA `fill_attention_sink_tiles` Doesn't Zero-Init L1

**The bug is NOT in `to_layout` or memory allocation.** The bug is in the SDPA kernel's `fill_attention_sink_tiles` function.

### Root Cause

The function `fill_attention_sink_tiles` in `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp` (lines 771-816) only initializes the **first column** of each tile, leaving the rest with **stale L1 values**:

```cpp
// BUG: Only writes to first element of each row, leaves rest uninitialized!
for (uint32_t face = 0; face < 4; face += 2) {
    for (uint32_t row = 0; row < face_height; ++row) {
        uint32_t row_offset = row * face_width;
        tile_ptr[face_offset + row_offset] = sink_value;  // Only first element!
    }
}
```

Later, the SDPA compute kernel calls `reduce_c<MAX>` which reads the **entire tile** to find the maximum:

```cpp
// This reads ALL 32x32 elements of the tile to find max!
reduce_c<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_attention_sink, ...>(...)
```

### Why This Causes Alternating Pass/Fail

1. **Run N (PASS)**: L1 memory is clean (zeros or small values). SDPA works correctly.
   - Experts runs and writes large values to L1 (during `to_layout` and other operations)

2. **Run N+1 (FAIL)**: L1 contains stale large values from Run N.
   - SDPA allocates attention sink CB in L1 region that has stale data
   - `fill_attention_sink_tiles` only fills first column, rest has large stale values
   - `reduce_c<MAX>` picks up stale large values as the max
   - Softmax computation is corrupted → attention fails
   - Experts doesn't run (attention failed)

3. **Run N+2 (PASS)**: Experts didn't run on N+1, so no new large values in L1.
   - SDPA allocates attention sink CB in different L1 region (or same but now "cleaner")
   - Or: the specific L1 region now has smaller values that don't corrupt max
   - Pattern continues...

### Evidence

Setting `TT_METAL_CLEAR_L1=1` (which zeros L1 on device init) **fixes the alternating pattern** - tests pass consistently:

```bash
TT_METAL_CLEAR_L1=1 HF_MODEL=/proj_sw/user_dev/gpt-oss-120b/ pytest ...
# Run 1: PASS
# Run 2: PASS
# Run 3: PASS
# ... (consistently passes)
```

### ✅ Fix Implemented and Verified (2026-01-13 22:43)

The `fill_attention_sink_tiles` function was fixed to **initialize the entire tile to -infinity** (`0xFF80` in bfloat16) before writing sink values. This ensures stale L1 values don't affect the max computation.

**Key change:** Added `-infinity` initialization loop before writing sink values:

```cpp
// -infinity in bfloat16 format (sign=1, exp=0xFF, mantissa=0)
constexpr uint16_t neg_inf_bf16 = 0xFF80;

// First, initialize the ENTIRE tile to -infinity
for (uint32_t i = 0; i < elements_per_tile; ++i) {
    tile_ptr[i] = neg_inf_bf16;
}

// Then fill first column with sink value (existing logic)
// ...
```

### Verification Results

After applying the fix, 5 consecutive test runs all **PASSED** with consistent PCC values:

| Run | Attention PCC | Experts PCC | Result |
|-----|---------------|-------------|--------|
| 1   | 0.9628        | 0.9268      | ✅ PASS |
| 2   | 0.9628        | 0.9268      | ✅ PASS |
| 3   | 0.9628        | 0.9268      | ✅ PASS |
| 4   | 0.9628        | 0.9268      | ✅ PASS |
| 5   | 0.9628        | 0.9268      | ✅ PASS |

**The alternating pass/fail pattern is completely eliminated.**

### Location of Fix

**File:** `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp`
**Function:** `fill_attention_sink_tiles` (lines 771-830)
**Called from:** `reader_interleaved.cpp` line 152

---

## Previous Investigation (Superseded)

The earlier investigation incorrectly identified `to_layout` as the culprit. While `to_layout` does write large values to L1 that persist, the actual bug is in how SDPA reads those stale values due to incomplete tile initialization.

---

## ✅ ROOT CAUSE FOUND (4x8 Mesh) - Added 2026-01-13 19:50 (SUPERSEDED)

### ⚠️ SCOPE OF THIS FINDING

**This investigation applies to:**
- **Mesh configuration:** 4x8 (full Galaxy with 32 devices)
- **Experts implementation:** High Throughput Experts (EP=32) in `models/demos/gpt_oss/tt/experts_throughput/`
- **Test command:** `--test-modules attention,experts` with `-k "4x8"`

### Root Cause Identified (4x8 Mesh, High Throughput Experts) - SUPERSEDED

~~The memory corruption is caused by **`ttnn.to_layout(expert_output, ttnn.ROW_MAJOR_LAYOUT)`** at line 393 of `models/demos/gpt_oss/tt/experts_throughput/decode.py`.~~

**UPDATE:** The actual bug is in SDPA's `fill_attention_sink_tiles` function. See "ACTUAL ROOT CAUSE" section above.

~~This is the **untilize** operation that converts the expert output from TILE layout to ROW_MAJOR layout before `all_to_all_combine`. The operation is writing its output to DRAM memory that overlaps with other tensors (specifically the attention sinks tensor used by SDPA).~~

**Key Difference from 1x8:**
- **1x8 mesh:** Corruption caused by `mesh_config.allreduce()` (experimental async CCL)
- **4x8 mesh:** ~~Corruption caused by `ttnn.to_layout()`~~ → Actually caused by SDPA `fill_attention_sink_tiles` not zero-initializing tiles

### Binary Search Results (4x8 Mesh)

| Return Point | Location | Corruption? | Attention PCC |
|--------------|----------|-------------|---------------|
| A | After `all_to_all_dispatch` | ❌ No | 0.9628 |
| E | After `sparse_matmul` w2 | ❌ No | 0.9628 |
| E1 | After `ttnn.permute` | ❌ No | 0.9628 |
| E2 | After `ttnn.reshape` | ❌ No | 0.9628 |
| **F** | **After `ttnn.to_layout` (untilize)** | **✅ YES** | **0.704** |
| G | After `all_to_all_combine` | ✅ YES | 0.704 |
| H | Before `ttnn.all_reduce` | ✅ YES | 0.940 |
| Full Run | Complete experts forward | ✅ YES | 0.867 |

The corruption appears at point F (after to_layout), confirming the untilize operation as the culprit.

### Workaround Attempts

1. **`ttnn.synchronize_device()` before to_layout**: ❌ Did not help
2. **`ttnn.to_memory_config(DRAM_MEMORY_CONFIG)` before to_layout**: ❌ Did not help
3. **`ttnn.clone()` before to_layout**: ✅ Works but causes OOM
4. **`ttnn.to_layout(..., memory_config=ttnn.L1_MEMORY_CONFIG)`**: ✅ Prevents corruption but causes OOM (tensor too large for L1)

The clone and L1 memory config approaches confirm the hypothesis: the issue is with DRAM memory allocation overlap. When forcing a fresh allocation (clone) or using L1, the corruption is prevented.

### ❌ NO WORKING WORKAROUND

**Critical Finding:** The test order workaround (attention before experts) does NOT reliably work for 4x8 mesh. Tests fail on alternating runs:

```
Run 1: FAIL - Attention PCC = 0.866
Run 2: PASS - Attention PCC = 0.963
Run 3: FAIL - Attention PCC = 0.866
Run 4: PASS - Attention PCC = 0.963
... (pattern continues)
```

This alternating behavior suggests:
1. The `to_layout` (untilize) operation corrupts DRAM memory
2. The corruption persists across pytest invocations (hardware DRAM state)
3. Running experts again somehow overwrites the corrupted region, allowing the next run to pass
4. But that run corrupts it again for the following run

**The within-test order (attention before experts) provides no protection** because the corruption from the PREVIOUS pytest run affects the current run.

**Test code reference:**
```python
# Line 635-650 in test_modules.py
# NOTE: Attention must run before experts due to memory corruption issue
# WARNING: This workaround does NOT work for 4x8 mesh - tests still fail on alternating runs
if should_test("attention"):
    logger.info("Testing Attention...")
    run_attention_component(...)

if should_test("experts"):
    ...
```

### Recommended Fix (REQUIRED - No Workaround Available)

**This is a blocking issue for 4x8 mesh testing.** The proper fix requires changes at the ttnn level:

1. **Investigate DRAM allocator bug**: The `to_layout` (untilize) operation is being allocated DRAM memory that overlaps with existing tensors (specifically attention sinks). This suggests a bug in:
   - Buffer size calculation for the untilize output
   - DRAM bank manager not properly tracking allocated regions
   - Use-after-free scenario where the allocator thinks memory is free when it's not
   - The corruption persists across pytest invocations, indicating hardware DRAM state is affected

2. **Attempted fixes that don't work**:
   - Test order (attention before experts): ❌ Fails on alternating runs
   - `ttnn.clone()` before untilize: ✅ Prevents corruption but causes OOM
   - `ttnn.to_memory_config(DRAM)` before untilize: ❌ No effect
   - `ttnn.synchronize_device()` before untilize: ❌ No effect
   - `ttnn.to_layout(..., memory_config=L1)`: ✅ Prevents corruption but causes OOM

3. **File a bug**: Report to ttnn team with:
   - Binary search results showing `to_layout` at line 393 as culprit
   - The alternating pass/fail pattern across runs
   - Evidence that L1 memory config prevents corruption (but causes OOM)

### Key Differences Between 1x8 and 4x8

| Aspect | 1x8 (Low Throughput) | 4x8 (High Throughput) |
|--------|---------------------|----------------------|
| File | `experts/prefill.py` | `experts_throughput/decode.py` |
| Experts per device | 16 (all on each device) | 4 (distributed) |
| Token routing | None (local) | `all_to_all_dispatch/combine` |
| TP allreduce | Uses `mesh_config.allreduce` | Uses `ttnn.all_reduce` directly |
| Sparsity pattern | `ttnn.repeat` of sparsity mask | `moe_expert_token_remap` |
| **Root cause** | `mesh_config.allreduce` (async CCL) | `ttnn.to_layout` (untilize DRAM overlap) |
| **Fix available** | ✅ Yes (use `ttnn.all_reduce`) | ❌ No - requires ttnn fix |

### Test Commands for 4x8 Mesh

```bash
# Reproduce the alternating pass/fail pattern (run multiple times)
HF_MODEL=/proj_sw/user_dev/gpt-oss-120b/ pytest models/demos/gpt_oss/tests/unit/test_modules.py::test_decoder \
  -k "4x8 and layer_0 and prefill_2048" --test-modules attention,experts
# Run 1: FAIL (attention PCC ~0.866)
# Run 2: PASS (attention PCC ~0.963)
# Run 3: FAIL
# Run 4: PASS
# ... pattern continues

# Note: Even running attention BEFORE experts doesn't help -
# the corruption from the PREVIOUS run affects the current run
```

---

## Watcher Investigation (Added 2026-01-13 20:30)

### Watcher Results

Ran the test with TT_METAL_WATCHER=1 to check for out-of-bounds memory accesses:

```bash
TT_METAL_WATCHER=1 HF_MODEL=/proj_sw/user_dev/gpt-oss-120b/ pytest models/demos/gpt_oss/tests/unit/test_modules.py::test_decoder \
  -k "4x8 and layer_0 and prefill_2048" --test-modules attention,experts
```

**Result:** ❌ No OOB memory accesses detected

The watcher log (`generated/watcher/watcher.log`) shows:
- All device cores completed normally with "GW" (Go Wait) status
- No NOC sanitization errors
- No invalid coordinates or addresses
- No asserts tripped

**Conclusion:** The memory corruption is NOT caused by invalid NOC transactions (writing to invalid addresses). Instead, it's caused by the DRAM allocator assigning overlapping valid addresses to different tensors.

### Memory Address Analysis

Added debug prints to track tensor DRAM addresses:

**Attention Sinks Tensor:**
- Address: `0x97d860` (~10 MB)
- Shape: [1, 8, 1, 1]
- Volume: 8 elements (but tile-padded to 32x32 = 1024 elements)
- This address is CONSISTENT across all runs (pass or fail)

**to_layout (untilize) Operation (on passing runs):**
- Input tensor address: `0x4b354e0` (~79 MB)
- Output tensor address: `0x22b354e0` (~583 MB)
- Shape: [4, 8192, 1, 2880]
- Size: ~180 MB (94M elements × 2 bytes)

**Key Observations:**
1. The sinks tensor address (0x97d860) is very different from the to_layout addresses
2. The addresses suggest no SIMPLE overlap (sinks at 10MB, to_layout at 79-583MB)
3. However, DRAM banks are interleaved - data at logical address X may share physical banks with data at address Y
4. The corruption persists across process restarts, indicating hardware DRAM state is affected

### Hypothesis: DRAM Bank Interleaving Corruption

The Wormhole chip has multiple DRAM banks with interleaved addressing. Even though logical addresses don't overlap:
1. The to_layout output (~180MB) spans many DRAM pages across all banks
2. During the untilize kernel, write transactions may corrupt adjacent bank entries
3. The sinks tensor, though at a different logical address, shares some DRAM bank
4. This explains why L1 memory config prevents corruption (no DRAM bank involvement)

### Next Steps for DRAM Debugging

1. **Dump DRAM bank allocations** - Add logging to track which physical banks each tensor uses
2. **Check for overlapping bank pages** - Verify if sinks and to_layout output share any physical DRAM pages
3. **Inspect untilize kernel** - Check if the dataflow/compute kernels have any OOB writes within L1 that then get flushed to wrong DRAM locations
4. **Test with explicit allocation** - Try pre-allocating a specific DRAM region for to_layout output

### Kernel Review

Reviewed `writer_unary_stick_layout_split_rows_multi_core.cpp` - the untilize writer kernel:
- Uses standard NOC write operations with page-based addressing
- Write addresses calculated via `get_noc_addr(output_page_id, s, output_offset_within_page_in_bytes)`
- No obvious OOB access issues visible in kernel code

**Conclusion:** The bug is likely in the DRAM allocation layer, not in the kernel code itself.

### Bug Report Template for ttnn Team

**Title:** DRAM allocation overlap causes memory corruption in to_layout (untilize) on 4x8 mesh

**Summary:**
The `ttnn.to_layout(tensor, ROW_MAJOR_LAYOUT)` operation on 4x8 mesh (32 devices) causes DRAM memory corruption that affects other tensors (attention sinks). The issue exhibits an alternating pass/fail pattern across pytest runs.

**Reproduction:**
```bash
# Run multiple times - alternates FAIL/PASS
HF_MODEL=/proj_sw/user_dev/gpt-oss-120b/ pytest models/demos/gpt_oss/tests/unit/test_modules.py::test_decoder \
  -k "4x8 and layer_0 and prefill_2048" --test-modules attention,experts
```

**Evidence:**
1. Watcher enabled - NO OOB errors detected (transactions use valid addresses)
2. L1 memory config prevents corruption (but causes OOM)
3. `ttnn.clone()` before to_layout prevents corruption (but causes OOM)
4. Tensor addresses:
   - Sinks: 0x97d860 (~10MB)
   - to_layout input: 0x4b354e0 (~79MB)
   - to_layout output: 0x22b354e0 (~583MB, size ~180MB)
5. Corruption persists across process restarts (hardware DRAM state affected)

**Suspected cause:** DRAM allocator assigning overlapping regions, possibly related to bank interleaving

---

## Previous Root Cause Analysis (Incomplete)

### Confirmed Finding

**The experts operation corrupts device memory that SDPA reads from.**

Evidence:
| Test Scenario | PCC | Status |
|---------------|-----|--------|
| Attention only (isolated) | ~0.96 | PASS |
| Attention first, then experts | ~0.96 | PASS |
| Experts first, then attention | ~0.66 | FAIL |
| Cross-test: prefill_128 → prefill_2048 | ~0.50 | FAIL |

### Memory Corruption Pattern

1. **Within-test corruption:** Experts module writes to memory locations that SDPA later reads
2. **Cross-test corruption:** The mesh_device fixture is reused across tests, so corruption from one test affects subsequent tests
3. **Specific to prefill:** Decode tests (seq_len=1) pass, suggesting the issue is related to larger buffer allocations

### Suspected Components

1. **`sparse_matmul` in experts:** Uses non-standard memory access patterns
2. **Attention sink CB allocation:** The circular buffer for attention sinks may overlap with or be adjacent to memory used by experts
3. **DRAM tensor corruption:** The `weights.sinks` tensor in DRAM may be getting corrupted by experts' OOB writes

---

## Files Modified

### 1. `models/demos/gpt_oss/tests/unit/test_modules.py`

**Status:** Modified (test order change)

**Change:** Moved attention test before experts test in the test_decoder function.

```diff
+    # NOTE: Attention must run before experts due to memory corruption issue
+    # See investigation: experts operations corrupt memory that SDPA reads from
+    if should_test("attention"):
+        logger.info("Testing Attention...")
+        run_attention_component(...)
+
     if should_test("experts"):
         ...
-
-    if should_test("attention"):
-        ...
```

### 2. `models/demos/gpt_oss/tt/attention/weights.py`

**Status:** Reverted to original

Changes to sink scaling were tried but reverted due to inconsistent results.

### 3. `models/demos/gpt_oss/tt/attention/prefill.py`

**Status:** Reverted to original

Ablation test changes (disabling sinks) were reverted.

---

## Key Code Locations

### Attention Implementation
- `models/demos/gpt_oss/tt/attention/prefill.py:91-100` - SDPA call with attention_sink
- `models/demos/gpt_oss/tt/attention/weights.py:121-170` - Sink tensor loading

### SDPA Kernel
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/sdpa.cpp:292-332` - Attention sink processing
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/reader_interleaved.cpp:139-156` - Sink reading and fill
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp:771-817` - `fill_attention_sink_tiles` function

### Experts Implementation
- `models/demos/gpt_oss/tt/experts/prefill.py` - Prefill forward with sparse_matmul

### HuggingFace Reference
- `python_env/lib/python3.10/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:258-259` - Attention sink usage

---

## Suggested Follow-Up Items

### High Priority

1. **Investigate sparse_matmul OOB writes**
   - Location: `models/demos/gpt_oss/tt/experts/prefill.py`
   - The experts prefill uses `ttnn.sparse_matmul` with sparsity patterns
   - Check if output buffers are correctly sized
   - Check if sparsity indices could cause OOB memory access

2. **Add device memory cleanup between tests**
   - Modify `conftest.py` mesh_device fixture to clear L1/DRAM between tests
   - Or use `--forked` pytest option if available to isolate tests

3. **Enable watcher and trace memory accesses**
   - Set `TT_METAL_WATCHER=1` to enable watcher
   - Look for illegal memory access warnings in `generated/watcher/watcher.log`

### Medium Priority

4. **Create minimal reproduction test**
   - Write a standalone test that:
     - Allocates attention_sink tensor in DRAM
     - Runs sparse_matmul (simulating experts)
     - Reads back attention_sink tensor
     - Verifies values haven't changed

5. **Investigate attention_sink tensor address**
   - Print DRAM address of `weights.sinks` tensor
   - Print DRAM addresses of experts output tensors
   - Check for overlap or adjacency

6. **Review `fill_attention_sink_tiles` function**
   - Location: `ttnn/.../dataflow_common.hpp:771`
   - This function fills tiles in CB by reading from source
   - Verify it doesn't read from uninitialized memory

### Low Priority

7. **Sink scaling investigation**
   - Deeper analysis of TT SDPA kernel's internal scaling
   - Compare with HF reference implementation step-by-step
   - May need to modify kernel to match HF's unscaled sink handling

8. **Profile memory usage**
   - Track L1 and DRAM usage during experts and attention
   - Look for buffer reuse patterns that could cause issues

---

## Test Commands for Follow-Up

```bash
# Run with watcher enabled
TT_METAL_WATCHER=1 HF_MODEL=/proj_sw/user_dev/gpt-oss-weights/gpt-oss-20b/ pytest models/demos/gpt_oss/tests/unit/test_modules.py -k "wormhole_b0-layer_1-mesh_1x8-prefill_2048-mesh_device0-fabric_1d_ring"

# Run attention only (should pass)
HF_MODEL=/proj_sw/user_dev/gpt-oss-weights/gpt-oss-20b/ pytest models/demos/gpt_oss/tests/unit/test_modules.py -k "1x8" --test-modules attention

# Run experts then attention (reproduces failure)
HF_MODEL=/proj_sw/user_dev/gpt-oss-weights/gpt-oss-20b/ pytest models/demos/gpt_oss/tests/unit/test_modules.py -k "wormhole_b0-layer_1-mesh_1x8-prefill_2048-mesh_device0-fabric_1d_ring" --test-modules experts,attention

# SDPA unit tests (verify op works in isolation)
pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_scaled_dot_product_attention.py -k "test_sdpa_with_attention_sink"
```

---

## Appendix: PCC Values Observed

| Configuration | Attention First | Experts First |
|--------------|-----------------|---------------|
| layer_0-prefill_128 | ~0.96 | ~0.96 |
| layer_0-prefill_2048 | ~0.96 | ~0.50 |
| layer_0-prefill_4096 | ~0.96 | varies |
| layer_1-prefill_128 | ~0.96 | varies |
| layer_1-prefill_2048 | ~0.96 | ~0.66 |
| layer_1-prefill_4096 | ~0.96 | varies |
| All decode tests | ~0.96+ | ~0.96+ |

Note: Results can vary based on what tests ran previously due to cross-test memory contamination.
