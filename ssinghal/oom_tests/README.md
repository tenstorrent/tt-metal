# OOM Test Suite

This directory contains pytest files specifically designed to test Out-of-Memory (OOM) failure scenarios.

## Files Generated

### Individual Operator Tests
- `test_oom_view.py` - OOM tests for view operator
- `test_oom_cat.py` - OOM tests for cat operator
- `test_oom_silu.py` - OOM tests for silu operator
- `test_oom_add.py` - OOM tests for add operator
- `test_oom_geluactivation.py` - OOM tests for geluactivation operator
- `test_oom_tanh.py` - OOM tests for tanh operator
- `test_oom_mish.py` - OOM tests for mish operator
- `test_oom_hardtanh.py` - OOM tests for hardtanh operator
- `test_oom_linear.py` - OOM tests for linear operator
- `test_oom_mul.py` - OOM tests for mul operator
- `test_oom_softplus.py` - OOM tests for softplus operator
- `test_oom_sigmoid.py` - OOM tests for sigmoid operator
- `test_oom_permute.py` - OOM tests for permute operator
- `test_oom_bmm.py` - OOM tests for bmm operator
- `test_oom_unsafeview.py` - OOM tests for unsafeview operator

### Master Test
- `test_master_oom.py` - Comprehensive test covering top OOM scenarios across all operators

## Purpose

These tests serve to:

1. **Document** problematic input shapes that cause OOM failures
2. **Verify** that OOM handling works correctly (tests should be SKIPPED)
3. **Track** memory requirements for optimization efforts
4. **Benchmark** memory usage patterns across different operators

## Usage

Run all OOM tests:
```bash
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
export ARCH_NAME=wormhole_b0
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
source python_env/bin/activate

# Run all OOM tests (should mostly be skipped)
pytest ssinghal/oom_tests/ -v

# Run specific operator OOM tests
pytest ssinghal/oom_tests/test_oom_view.py -v

# Run master OOM analysis
pytest ssinghal/oom_tests/test_master_oom.py -v
```

## Expected Behavior

- Most tests should be **SKIPPED** due to OOM conditions
- Tests that **FAIL** indicate unexpected behavior
- Tests that **PASS** indicate the operation succeeded (unexpected for OOM scenarios)

## Statistics

- **Total OOM failures documented**: 118
- **Operators with OOM issues**: 15
- **Memory range**: 21.6 MB - 16000.0 MB

## Most Problematic Operators

- **view**: 42 failures
- **permute**: 22 failures
- **unsafeview**: 14 failures
- **add**: 10 failures
- **silu**: 6 failures
