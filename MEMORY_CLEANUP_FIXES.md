# Memory Cleanup Fixes - Summary

## Problems Identified

### Problem 1: Memory Not Deallocating After Tests
**Symptom**: L1 and DRAM buffers remain allocated after running pytest tests

**Root Causes**:
1. Program cache (compiled kernels) not being cleared after tests
2. `conftest.py` fixture runs AFTER test teardown, potentially too late
3. No explicit cleanup at the end of test functions

### Problem 2: Allocation Server Not Displaying Statistics
**Symptom**: Server doesn't show remaining buffers at the end of program

**Root Cause**:
- Server only responds to explicit `DUMP_REMAINING` messages
- No automatic dump was happening after tests

## Solutions Implemented

### Fix 1: Enhanced conftest.py Fixture

**File**: `/home/tt-metal-apv/models/tt_transformers/demo/conftest.py`

**Changes**:
```python
@pytest.fixture(scope="function", autouse=True)
def clear_program_cache_after_test(request, mesh_device):
    """Automatically clear program cache after each test."""
    import gc, socket, struct, time

    yield  # Run the test

    if isinstance(mesh_device, ttnn.MeshDevice):
        # 1. Force garbage collection
        gc.collect()

        # 2. Clear program cache
        mesh_device.disable_and_clear_program_cache()

        # 3. Force another GC
        gc.collect()

        # 4. Wait for deallocations to propagate
        time.sleep(1)

        # 5. Request buffer dump from allocation server
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect("/tmp/tt_allocation_server.sock")
            msg = struct.pack('B3xiQB3xiQQ4Q', 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            sock.send(msg)
            sock.close()
        except Exception as e:
            logger.warning(f"Could not request buffer dump: {e}")
```

**Benefits**:
- ✅ Automatically clears program cache after EVERY test
- ✅ Forces garbage collection to free Python objects
- ✅ Automatically requests buffer dump from allocation server
- ✅ Provides clear logging of cleanup steps

### Fix 2: Explicit Cleanup in Test Function

**File**: `/home/tt-metal-apv/models/tt_transformers/demo/simple_text_demo.py`

**Changes**:
Added at the end of `test_demo_text()` function (line 1252):
```python
# Explicit cleanup before test ends
logger.info("=" * 80)
logger.info("TEST CLEANUP: Clearing program cache before test completion...")
if isinstance(mesh_device, ttnn.MeshDevice):
    mesh_device.disable_and_clear_program_cache()
    logger.info("✓ Program cache cleared in test")
logger.info("=" * 80)
```

**Benefits**:
- ✅ Clears cache BEFORE test teardown
- ✅ Ensures cleanup happens even if fixture fails
- ✅ Provides clear logging in test output

### Fix 3: Helper Script for Memory Monitoring

**File**: `/home/tt-metal-apv/models/tt_transformers/demo/run_test_with_memory_check.sh`

**Purpose**: Automated test runner that:
1. Checks if allocation server is running
2. Runs the pytest test
3. Waits for cleanup to complete
4. Automatically dumps remaining buffers

**Usage**:
```bash
cd /home/tt-metal-apv
./models/tt_transformers/demo/run_test_with_memory_check.sh
```

**Benefits**:
- ✅ One-command test execution with memory monitoring
- ✅ Automatic buffer dump after test
- ✅ Clear error messages if server not running

### Fix 4: Comprehensive Documentation

**File**: `/home/tt-metal-apv/models/tt_transformers/demo/MEMORY_DEBUGGING_GUIDE.md`

**Contents**:
- Explanation of expected vs. unexpected memory
- Step-by-step debugging guide
- Common issues and solutions
- Best practices for memory monitoring

## Expected Behavior After Fixes

### What You Should See Now:

1. **During Test Execution**:
   ```
   TEST CLEANUP: Clearing program cache before test completion...
   ✓ Program cache cleared in test
   ```

2. **After Test Completion** (from conftest fixture):
   ```
   CLEANUP: Starting post-test cleanup...
   CLEANUP: Running garbage collection...
   CLEANUP: Clearing program cache...
   ✓ Program cache cleared
   CLEANUP: Requesting buffer dump from allocation server...
   ✓ Buffer dump requested (check allocation server output)
   ✓ Post-test cleanup complete
   ```

3. **In Allocation Server Output**:
   ```
   ╔══════════════════════════════════════════════════════════════╗
   ║           REMAINING ALLOCATED BUFFERS                       ║
   ╚══════════════════════════════════════════════════════════════╝

   Device 0:
     L1: 1 buffers, 0.012 MB total
       - Buffer 0x30000: 12.0 KB (PID 12345, ref_count=1)

   Total remaining buffers: 1
   ```

### What's Normal vs. What's a Leak:

#### ✅ **Normal (Expected)**:
- **~12KB L1 per device**: Circular buffers (infrastructure)
- **Small DRAM (<1MB)**: System buffers
- **Consistent across runs**: Same buffers remain each time

#### ❌ **Potential Leak**:
- **36KB DRAM per device**: Program cache (should be cleared now)
- **Large DRAM (>100MB)**: Model weights or KV cache
- **Growing memory**: Increases with each test run
- **High ref_count (>2)**: Multiple references preventing cleanup

## Testing the Fixes

### Step 1: Start Allocation Server
```bash
cd /home/tt-metal-apv/tt_metal/programming_examples/alexp_examples/memory_utilization_monitor
./allocation_server_poc &
```

### Step 2: Run Test
```bash
cd /home/tt-metal-apv
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-1" -v
```

### Step 3: Verify Cleanup
Check the allocation server output for:
1. Deallocation messages during cleanup
2. Final buffer dump showing minimal remaining memory
3. No "unknown buffer" warnings (these are now silently handled)

### Step 4: Run Multiple Times
```bash
# Run 3 times to check for accumulation
for i in {1..3}; do
    echo "=== Run $i ==="
    pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-1" -v
    sleep 2
done
```

Memory should NOT grow across runs.

## Troubleshooting

### Issue: "Could not request buffer dump"
**Cause**: Allocation server not running
**Solution**: Start the server first (see Step 1 above)

### Issue: Still seeing 36KB DRAM
**Cause**: Program cache from previous run before fixes
**Solution**:
```python
# Manually clear once:
import ttnn
mesh_device.disable_and_clear_program_cache()
```

### Issue: Memory still growing
**Cause**: Potential leak in model code (not cleanup code)
**Solution**: Use the debugging guide to identify the source

## Files Modified

1. `/home/tt-metal-apv/models/tt_transformers/demo/conftest.py`
   - Enhanced fixture with automatic cleanup and buffer dump

2. `/home/tt-metal-apv/models/tt_transformers/demo/simple_text_demo.py`
   - Added explicit cleanup at end of test function

3. `/home/tt-metal-apv/models/tt_transformers/demo/run_test_with_memory_check.sh` (NEW)
   - Helper script for automated testing with memory monitoring

4. `/home/tt-metal-apv/models/tt_transformers/demo/MEMORY_DEBUGGING_GUIDE.md` (NEW)
   - Comprehensive guide for memory debugging

## Summary

**Before Fixes**:
- ❌ Program cache not cleared (36KB DRAM leak per device)
- ❌ No automatic buffer dump
- ❌ Manual cleanup required

**After Fixes**:
- ✅ Automatic program cache clearing (2 places: test + fixture)
- ✅ Automatic buffer dump after each test
- ✅ Clear logging of cleanup steps
- ✅ Helper script for easy testing
- ✅ Comprehensive documentation

**Result**: Memory should now be properly cleaned up after each test, with automatic reporting of any remaining buffers.
