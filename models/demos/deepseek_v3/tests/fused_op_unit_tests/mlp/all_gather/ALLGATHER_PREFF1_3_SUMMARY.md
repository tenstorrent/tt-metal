# AllGather_preff1/3 Fused Op Unit Test - Implementation Summary

## Overview

This document summarizes the implementation of the fused op unit test for **AllGather_preff1/3** in the DeepSeek V3 MLP module.

**Date:** January 7, 2026
**Fused Op Name:** AllGather_preff1/3
**Module:** DeepSeek V3 MLP (`models/demos/deepseek_v3/tt/mlp/mlp.py`)
**Location in Module:** Line 439 (prefill), Line 491 (decode)

## Fused Op Description

### What is AllGather_preff1/3?

The AllGather_preff1/3 fused op is actually a single AllGather collective communication operation, not a sequence of multiple operations. The name "preff1/3" indicates it's the 1st major operation group out of 3 in the prefill forward pass.

### Operation Details

**TTNN Implementation:**
```python
x = ttnn.experimental.all_gather_async(x, **ccl.populate_all_gather_runtime_args(cfg["all_gather"]))
```

**Purpose:**
- Collects the full hidden dimension from all devices before performing matmul operations
- Required because MLP weights are sharded across devices
- Each device needs the full input to compute its portion of the output

**Input:**
- Shape: `[num_layers, batch_size, seq_len, hidden_size/num_devices]`
- Sharded across devices on the last dimension

**Output:**
- Shape: `[num_layers, batch_size, seq_len, hidden_size]`
- Replicated across all devices

**Configuration:**
- `cluster_axis`: 1 (gather across mesh rows)
- `dim`: -1 (last dimension)
- `memory_config`: DRAM_MEMORY_CONFIG
- `topology`: Linear (one row of Galaxy does not form a ring)

### PyTorch Reference

In the reference PyTorch model (without tensor parallelism), this is an identity operation:
```python
def ds_fused_all_gather_preff1_3_reference(x: torch.Tensor) -> torch.Tensor:
    return x
```

This is because the reference model doesn't shard tensors across devices.

## Files Created

### 1. Main Test File
**Path:** `models/demos/deepseek_v3/tests/unit/fused_op_unit_tests/mlp/test_ds_fused_all_gather_preff1_3.py`

**Contents:**
- `ds_fused_all_gather_preff1_3_reference()`: PyTorch reference implementation
- `ds_fused_all_gather_preff1_3_ttnn()`: TTNN implementation
- `test_ds_fused_all_gather_preff1_3()`: Main pytest with parameterization
- `test_ds_fused_all_gather_preff1_3_single_device()`: Single device test (skipped due to CCL)
- `test_ds_fused_all_gather_preff1_3_device_perf()`: Device performance test
- `test_ds_fused_all_gather_preff1_3_single_device_device_perf()`: Single device perf test (skipped)
- Helper functions for performance measurement, device perf collection, and comparison

**Test Parameters:**
- **Modes:** decode (seq_len=1), prefill (seq_len=128, 1024, 8192, 131072)
- **Program Cache:** enabled/disabled
- **Trace Mode:** eager/trace
- **Expected PCC:** 0.9999 (should be 1.0 for deterministic AllGather)
- **Expected ATOL/RTOL:** 0.2
- **Expected Perf:** 0.0 (TODO: add theoretical targets)

### 2. Configuration Comparison Script
**Path:** `models/demos/deepseek_v3/tests/unit/fused_op_unit_tests/mlp/compare_all_gather_configs.py`

**Purpose:** Compare AllGather operation properties between fused op unit test and module test to ensure exact match.

**Usage:**
```bash
python compare_all_gather_configs.py <fused_op_csv> <module_csv>
```

### 3. Verification Guide
**Path:** `VERIFICATION_GUIDE_ALLGATHER_PREFF1_3.md`

**Contents:**
- Step-by-step verification instructions
- Prerequisites and environment setup
- Expected results for each step
- Troubleshooting guide
- Notes about the fused op and single device tests

### 4. Verification Script
**Path:** `verify_fused_all_gather_preff1_3.sh`

**Purpose:** Automated script to run all verification steps (to be executed in docker container).

## Implementation Steps Completed

### ✅ Step 1: Run Baseline Module Test
**Status:** Completed (from terminal 2 history)

**Command:**
```bash
pytest models/demos/deepseek_v3/tests/test_mlp.py::test_forward_pass[decode-32-MLP-None-device_params0]
```

**Result:** PASSED with PCC > 0.975

### ✅ Step 2: Create Test File
**Status:** Completed

Created `test_ds_fused_all_gather_preff1_3.py` with all required components following the structure of the example test (`test_ds_fused_wqkva.py`).

### ✅ Step 3: Implement PyTorch Reference Function
**Status:** Completed

```python
def ds_fused_all_gather_preff1_3_reference(x: torch.Tensor) -> torch.Tensor:
    """Identity operation for reference model without tensor parallelism."""
    return x
```

### ✅ Step 4: Implement TTNN Function
**Status:** Completed

```python
def ds_fused_all_gather_preff1_3_ttnn(x: ttnn.Tensor, cfg: dict, ccl) -> ttnn.Tensor:
    """AllGather operation for TTNN model with tensor parallelism."""
    x = ttnn.experimental.all_gather_async(x, **ccl.populate_all_gather_runtime_args(cfg["all_gather"]))
    return x
```

### ⏳ Step 5: Verify TTNN Function by Replacing in Module
**Status:** Ready for execution (requires docker environment)

**Instructions:**
1. Temporarily modify `mlp.py` to call `ds_fused_all_gather_preff1_3_ttnn()`
2. Run module test and verify PCC matches baseline
3. Revert changes

**Expected Result:** Same PCC as baseline (> 0.975)

### ✅ Step 6: Implement Pytest with All Parameters
**Status:** Completed

Implemented comprehensive pytest with:
- Mode and sequence length parameterization
- Program cache on/off
- Trace mode on/off
- Device parameters (fabric config, trace region size)
- Performance measurement (e2e and device perf)
- PCC, ATOL, RTOL assertions
- Benchmark data collection

### ⏳ Step 7: Verify Reference and Test Code with Unit Test
**Status:** Ready for execution (requires docker environment)

**Commands:**
```bash
# Decode mode
pytest models/demos/deepseek_v3/tests/unit/fused_op_unit_tests/mlp/test_ds_fused_all_gather_preff1_3.py::test_ds_fused_all_gather_preff1_3 -k "decode and 1 and program_cache and eager"

# Prefill mode
pytest models/demos/deepseek_v3/tests/unit/fused_op_unit_tests/mlp/test_ds_fused_all_gather_preff1_3.py::test_ds_fused_all_gather_preff1_3 -k "prefill and 128 and program_cache and eager"
```

**Expected Result:** PCC > 0.9999 (likely 1.0)

### ⏳ Step 8: Verify Configurations Match
**Status:** Ready for execution (requires docker environment)

**Sub-steps:**
1. Run device perf test for fused op (decode and prefill)
2. Run module test with device perf (decode and prefill)
3. Compare CSV files using `compare_all_gather_configs.py`

**Expected Result:** All AllGather operation properties match exactly

### ✅ Step 9: Add Single Device Test
**Status:** Completed (skipped with appropriate message)

Single device tests are skipped because AllGather is a CCL operation that requires multiple devices.

### ✅ Step 10: Add Device Perf Tests
**Status:** Completed

Implemented:
- `test_ds_fused_all_gather_preff1_3_device_perf()`: Multi-device performance test
- `test_ds_fused_all_gather_preff1_3_single_device_device_perf()`: Single device test (skipped)

Device perf tests use tracy profiler with signposts to measure kernel duration and op-to-op latency.

### ⏳ Step 11: Generate Summary
**Status:** This document

## Verification Checklist

To complete the verification, run the following in the docker environment:

- [x] Step 1: Baseline module test passed ✅
- [ ] Step 5: Module test with fused op function (verify PCC matches baseline)
- [ ] Step 7: Fused op unit test (verify PCC > 0.9999)
- [ ] Step 8: Device perf tests and config comparison (verify exact match)

## Key Design Decisions

### 1. Single Operation Fused Op
Unlike other fused ops (e.g., fused_wqkva with multiple operations), this fused op contains only a single AllGather operation. This is still valuable for:
- Isolating AllGather performance
- Verifying AllGather configuration
- Providing a baseline for more complex fused ops
- Testing CCL operations independently

### 2. Reference Implementation as Identity
The reference implementation is an identity operation because:
- PyTorch reference model doesn't use tensor parallelism
- AllGather only affects distributed execution, not computation
- This makes PCC comparison straightforward (should be 1.0)

### 3. Single Device Tests Skipped
Single device tests are not applicable because:
- AllGather is a collective communication operation
- Requires multiple devices to function
- Cannot be meaningfully tested on a single device

### 4. Performance Targets Placeholder
Performance targets are set to 0.0 as placeholders because:
- Theoretical targets need to be calculated based on:
  - AllGather bandwidth
  - Tensor size
  - Network topology
  - Number of devices
- Should be updated after initial measurements

## Test Coverage

### Modes and Sequence Lengths
- **Decode:** seq_len=1
- **Prefill:** seq_len=128, 1024, 8192, 131072

### Test Variants
- Program cache: enabled/disabled
- Trace mode: eager/trace
- Device perf: with/without profiling

### Total Test Cases
- Main test: 5 modes × 2 cache × 2 trace = 20 test cases
- Device perf: 5 modes = 5 test cases
- Single device: skipped
- **Total:** 25 test cases (20 executed, 5 skipped)

## Expected Results

### PCC (Pearson Correlation Coefficient)
- **Expected:** 0.9999 (likely 1.0)
- **Reason:** AllGather is deterministic and doesn't modify values

### ATOL/RTOL (Absolute/Relative Tolerance)
- **Expected:** 0.2
- **Reason:** Conservative tolerance for numerical precision

### Performance
- **E2E Duration:** TBD (to be measured)
- **Kernel Duration:** TBD (to be measured)
- **Op-to-Op Latency:** TBD (to be measured)

## Known Limitations and Future Work

### 1. Performance Targets
**Current:** Placeholder values (0.0)
**Future:** Calculate theoretical targets based on:
- Network bandwidth
- Tensor size
- Topology
- Device count

### 2. Long Sequence Tests
**Current:** seq_len=131072 requires `DEEPSEEK_V3_LONG_SEQ_TESTS=1`
**Future:** May need to adjust based on memory constraints

### 3. Trace Region Size
**Current:** 2967552 (from example test)
**Future:** May need adjustment based on actual trace requirements

### 4. Configuration Comparison
**Current:** Manual comparison using script
**Future:** Could be automated as part of CI/CD

## Troubleshooting

### Common Issues

**1. Test Hangs**
- **Symptom:** No log output for > 5 minutes
- **Solution:** Kill test, run `tt-smi -glx_reset`, restart docker

**2. PCC Not 1.0**
- **Symptom:** PCC < 0.9999
- **Solution:** Check deterministic environment, memory config, CCL config

**3. Import Errors**
- **Symptom:** Cannot import fused op function
- **Solution:** Verify PYTHONPATH, check file paths

**4. Device Perf CSV Not Generated**
- **Symptom:** CSV file missing after device perf test
- **Solution:** Check tracy profiler is enabled, verify output directory

## Files Modified

### None (Module Unchanged)
The MLP module (`mlp.py`) was **not** permanently modified. The fused op function is implemented in the test file and can be used for verification by temporarily modifying the module.

## Files to Review

1. **Test File:** `models/demos/deepseek_v3/tests/unit/fused_op_unit_tests/mlp/test_ds_fused_all_gather_preff1_3.py`
2. **Comparison Script:** `models/demos/deepseek_v3/tests/unit/fused_op_unit_tests/mlp/compare_all_gather_configs.py`
3. **Verification Guide:** `VERIFICATION_GUIDE_ALLGATHER_PREFF1_3.md`
4. **Verification Script:** `verify_fused_all_gather_preff1_3.sh`
5. **This Summary:** `ALLGATHER_PREFF1_3_SUMMARY.md`

## Next Steps

To complete the verification:

1. **Enter Docker Container:**
   ```bash
   docker run -it -v /dev/hugepages-1G:/dev/hugepages-1G --device /dev/tenstorrent \
     -v /home/models-team/hzhou:/home/models-team/hzhou -v /mnt:/mnt \
     ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64:latest /bin/bash
   ```

2. **Activate Python Environment:**
   ```bash
   cd /home/models-team/hzhou/tt-metal
   source python_env/bin/activate
   ```

3. **Set Environment Variables:**
   ```bash
   export ARCH_NAME=wormhole_b0
   export TT_METAL_HOME=$(pwd)
   export PYTHONPATH=$(pwd)
   export DEEPSEEK_V3_HF_MODEL=/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528
   export DEEPSEEK_V3_CACHE=/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache
   export MESH_DEVICE=TG
   export TT_METAL_RUNTIME_ROOT=$(pwd)
   ```

4. **Run Verification Steps:**
   Follow the instructions in `VERIFICATION_GUIDE_ALLGATHER_PREFF1_3.md` or run:
   ```bash
   bash verify_fused_all_gather_preff1_3.sh
   ```

5. **Review Results:**
   Check log files in `logs/` directory for:
   - PCC values
   - Performance metrics
   - Configuration comparisons

## Conclusion

The AllGather_preff1/3 fused op unit test has been successfully implemented following the guide in `AGENTS_GUIDE_ADD_TEST.md`. The test is ready for execution in the docker environment with TT hardware access.

**Key Achievements:**
- ✅ Complete test implementation with all required components
- ✅ PyTorch reference and TTNN implementation
- ✅ Comprehensive test parameterization
- ✅ Performance measurement infrastructure
- ✅ Device perf tests with tracy profiler
- ✅ Configuration comparison script
- ✅ Detailed verification guide
- ✅ Single device tests (appropriately skipped)

**Remaining Work:**
- ⏳ Execute verification steps in docker environment
- ⏳ Update performance targets based on measurements
- ⏳ Verify configurations match exactly

The implementation is complete and ready for testing!
