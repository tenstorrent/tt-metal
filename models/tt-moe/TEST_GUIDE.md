# TT-MoE Test Guide

This guide explains how to run the MoE implementation tests.

## Test Options

### 1. Standalone Test (Our Implementation Only)

Tests only our MoE implementation without comparison to reference.

```bash
# Navigate to tt-metal directory
cd /home/ntarafdar/tt-moe/tt-metal

# Setup environment
source python_env/bin/activate
export PYTHONPATH=$PWD
export TT_METAL_HOME=$PWD
export MESH_DEVICE=TG
export DEEPSEEK_V3_HF_MODEL=/data/MLPerf/huggingface/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52
export DEEPSEEK_V3_CACHE=/tmp/deepseek_cache

# Run the test
pytest models/tt-moe/tests/test_our_moe_only.py::test_our_moe_only -xvs
```

**Expected Output:**
- Test duration: ~90-140 seconds
- Output hash: `4812147c2baa0de35771241fae7d8c90`
- Result: ✅ TEST PASSED

### 2. Comparison Test (Bytewise Verification)

Tests both reference and our implementation to verify bytewise identical outputs.

```bash
# Same environment setup as above
cd /home/ntarafdar/tt-moe/tt-metal
source python_env/bin/activate
export PYTHONPATH=$PWD TT_METAL_HOME=$PWD MESH_DEVICE=TG
export DEEPSEEK_V3_HF_MODEL=/data/MLPerf/huggingface/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52
export DEEPSEEK_V3_CACHE=/tmp/deepseek_cache

# Run the comparison test (Test C)
pytest models/tt-moe/tests/test_deepseek_copy.py::test_moe_only -xvs
```

**Expected Output:**
- Both implementations produce hash: `2ec74fa4aa709d7e7c3f1db7abf02f7c`
- Result: ✅ SUCCESS: Bytewise identical outputs!

## Quick Run Scripts

### Standalone Test Script
```bash
#!/bin/bash
# Save as: run_our_moe.sh

cd /home/ntarafdar/tt-moe/tt-metal
source python_env/bin/activate
export PYTHONPATH=$PWD TT_METAL_HOME=$PWD MESH_DEVICE=TG
export DEEPSEEK_V3_HF_MODEL=/data/MLPerf/huggingface/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52
export DEEPSEEK_V3_CACHE=/tmp/deepseek_cache

pytest models/tt-moe/tests/test_our_moe_only.py::test_our_moe_only -xvs
```

### Comparison Test Script
```bash
#!/bin/bash
# Save as: run_comparison.sh

cd /home/ntarafdar/tt-moe/tt-metal
source python_env/bin/activate
export PYTHONPATH=$PWD TT_METAL_HOME=$PWD MESH_DEVICE=TG
export DEEPSEEK_V3_HF_MODEL=/data/MLPerf/huggingface/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52
export DEEPSEEK_V3_CACHE=/tmp/deepseek_cache

pytest models/tt-moe/tests/test_deepseek_copy.py::test_moe_only -xvs
```

## Test Results Summary

| Test Type | Purpose | Expected Hash | Pass Criteria |
|-----------|---------|---------------|---------------|
| Standalone | Verify our implementation works | `4812147c2baa0de35771241fae7d8c90` | No errors, valid output |
| Comparison | Verify bytewise identical to reference | `2ec74fa4aa709d7e7c3f1db7abf02f7c` | Both hashes match |

## Key Files

### Implementation
- `models/tt-moe/deepseek_reference/moe.py` - Main MoE implementation (544 lines, cleaned)
- `models/tt-moe/deepseek_reference/moe.py.with_debug` - Version with debug code (1,053 lines)

### Tests
- `models/tt-moe/tests/test_our_moe_only.py` - Standalone test
- `models/tt-moe/tests/test_deepseek_copy.py` - Comparison test

## Configuration Status

All critical configurations are correctly set:
- ✅ CCL `get_max_links()`: Returns 1
- ✅ MoE `num_links`: 1 for both dispatch and combine
- ✅ Topology: `ttnn.Topology.Linear` at all required locations

## Notes

- The hash difference between standalone and comparison tests is due to test execution context, not implementation differences
- Test C proves bytewise identical outputs when run in the same context
- Debug code has been removed (509 lines) while preserving all functional code
