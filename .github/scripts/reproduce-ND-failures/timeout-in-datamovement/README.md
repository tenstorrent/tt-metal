# Timeout in Data Movement - Device Timeout in Gather Operation

## Original Failure

- **Job**: tt-metal-l2-tests (wormhole_b0, N300) / ttnn nightly data_movement tests wormhole_b0 N300
- **Date**: 2026-01-30
- **Frequency**: Non-deterministic
- **Error**: `RuntimeError: TT_THROW @ system_memory_manager.cpp:627: TIMEOUT: device timeout, potential hang detected`

## Root Cause

The failure occurs during `ttnn.to_torch()` when reading data back from the device after a `ttnn.gather()` operation with a large tensor shape `[1, 151936]`.

The internal device timeout (5 seconds, set by `TT_METAL_OPERATION_TIMEOUT_SECONDS`) triggers during `copy_completion_queue_data_into_user_space` in the `FDMeshCommandQueue::read_completion_queue()` function.

**Stack trace points to:**
- `tt::tt_metal::buffer_dispatch::copy_completion_queue_data_into_user_space()`
- `tt::tt_metal::distributed::FDMeshCommandQueue::read_completion_queue()`

## Reproduction

The stress test in `tests/test_gather_timeout_stress.py` reproduces this failure by:
- Running the exact failing gather operation with shape `[1, 151936]`
- Setting `TT_METAL_OPERATION_TIMEOUT_SECONDS=5` to match CI's hang detection timeout
- Using multiple iterations and test variants

### Run Test

```bash
# Activate virtual environment first
source /opt/venv/bin/activate

# Set required environment variables (CRITICAL!)
export TT_METAL_OPERATION_TIMEOUT_SECONDS=5
export ARCH_NAME=wormhole_b0
export TT_METAL_HOME=/tt-metal
export PYTHONPATH=/tt-metal

# Reset device before testing
tt-smi -r 0

# Run a single test to verify reproduction
pytest "tests/test_gather_timeout_stress.py::test_gather_long_tensor_stress[0]" -x -v --timeout=120 2>&1 | tee logs/run_test.log

# Run full stress test (50 iterations of the main test case)
pytest tests/test_gather_timeout_stress.py::test_gather_long_tensor_stress -x -v --timeout=300 2>&1 | tee logs/full_stress.log
```

### Expected Behavior

- **Success**: Test passes, `ttnn.to_torch()` returns the result tensor
- **Failure (Reproduced)**:
  ```
  RuntimeError: TT_THROW @ /tt-metal/tt_metal/impl/dispatch/system_memory_manager.cpp:627: tt::exception
  info:
  TIMEOUT: device timeout, potential hang detected, the device is unrecoverable
  ```

## Results

**Reproduction successful on first attempt (2026-02-03):**

The test reproduced the exact error immediately with:
- Test: `test_gather_long_tensor_stress[0]`
- Time to failure: ~5 seconds (matches the 5-second timeout)
- Error: Exact match with CI failure

See `logs/run_4_single.log` for the successful reproduction log.

## Key Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `TT_METAL_OPERATION_TIMEOUT_SECONDS` | `5` | **CRITICAL** - Device operation timeout |
| `ARCH_NAME` | `wormhole_b0` | Architecture |
| `TT_METAL_HOME` | `/tt-metal` | TT-Metal installation path |
| `PYTHONPATH` | `/tt-metal` | Python path |

## Notes

- This issue is 100% reproducible with `TT_METAL_OPERATION_TIMEOUT_SECONDS=5` on this hardware
- The gather operation with shape `[1, 151936]` consistently triggers the timeout
- Without the timeout environment variable, the operation may hang indefinitely
