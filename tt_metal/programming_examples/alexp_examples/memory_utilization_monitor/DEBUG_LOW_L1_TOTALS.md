# Debugging: Why Am I Only Seeing KB Instead of MB?

## Problem
You're seeing **total L1 usage of only KBs** when you expect **~171MB** (or at least 10s-100s of MB for a real workload).

---

## Possible Causes & Fixes

### 1. **AllocationClient Not Enabled in graph_tracking.cpp**

The existing buffer tracking checks `AllocationClient::is_enabled()` before reporting.

**Check**: Add debug logging to see if this is the issue:

```bash
# Check if the check is failing
grep -n "AllocationClient::is_enabled()" tt_metal/graph/graph_tracking.cpp

# Output should show line 148:
# if (AllocationClient::is_enabled()) {
```

**Fix**: Verify the environment variable is set **before** creating any buffers:

```bash
export TT_ALLOC_TRACKING_ENABLED=1
echo $TT_ALLOC_TRACKING_ENABLED  # Should print "1"

# Then run your workload
```

---

### 2. **Buffers Using DRAM Instead of L1**

Many workloads default to DRAM for large buffers.

**Check**: Look at the server output for buffer types:
- `buffer_type=0` = DRAM
- `buffer_type=1` = L1
- `buffer_type=2` = SYSTEM_MEMORY
- `buffer_type=3` = L1_SMALL
- `buffer_type=4` = TRACE

**Server Output Example**:
```
✓ Allocated 16777216 bytes of DRAM ...  ← This is DRAM, not L1!
✓ Allocated 8388608 bytes of L1 ...     ← This is L1!
```

**Fix**: Force L1 allocation in your workload:
```python
import ttnn
tensor = ttnn.from_torch(
    torch_tensor,
    device=device,
    memory_config=ttnn.L1_MEMORY_CONFIG,  # ← Force L1
    dtype=ttnn.bfloat16
)
```

---

### 3. **Server Only Showing Per-Type Totals, Not Grand Total**

The server shows totals per buffer type. You might need to sum them manually.

**Check**: Send SIGUSR1 to the server to get a summary:
```bash
# Find server PID
ps aux | grep allocation_server_poc

# Send signal
kill -USR1 <PID>
```

**Server Output**:
```
Device 0:
  DRAM: 45 buffers, 234.56 MB total  ← DRAM total
  L1: 143 buffers, 127.45 MB total   ← L1 total
  TRACE: 2 buffers, 32.00 MB total   ← Trace total
```

**Total = DRAM + L1 + TRACE = 234.56 + 127.45 + 32.00 = 394.01 MB**

---

### 4. **Graph Tracking Not Compiled In**

The tracking code might not be compiled or linked properly.

**Check**: Verify the tracking functions exist:
```bash
cd /home/ttuser/aperezvicente/tt-metal

# Should find the tracking code
grep -n "AllocationClient::report_allocation" tt_metal/graph/graph_tracking.cpp

# Should see lines 149-150:
#     AllocationClient::report_allocation(
#         buffer->device()->id(), buffer->size(), ...
```

**Fix**: Rebuild from scratch:
```bash
rm -rf build
source ./env_vars_setup.sh
./build_metal_with_flags.sh
```

---

### 5. **Workload Not Actually Using Much L1**

Some workloads are DRAM-heavy or don't allocate much memory at all.

**Check**: Run a known L1-heavy workload:
```bash
# Simple matmul that uses L1
pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_matmul.py -s
```

**Or use the debug script**:
```bash
cd tt_metal/programming_examples/alexp_examples/memory_utilization_monitor
python3 debug_tracking.py
```

This explicitly creates L1 buffers and tells you what to expect.

---

### 6. **Multiple Processes / Race Conditions**

If multiple processes are allocating/deallocating rapidly, you might catch it between allocations.

**Check**: Use the dump command to see persistent allocations:
```bash
kill -USR1 <server_pid>
```

---

## Debug Script

Run this to explicitly test L1 tracking:

```bash
cd /home/ttuser/aperezvicente/tt-metal

# Terminal 1: Start server
export TT_ALLOC_TRACKING_ENABLED=1
./build/install/bin/allocation_server_poc

# Terminal 2: Run debug script
export TT_ALLOC_TRACKING_ENABLED=1
python3 tt_metal/programming_examples/alexp_examples/memory_utilization_monitor/debug_tracking.py
```

**Expected server output**:
```
✓ [PID 12345] Allocated 524288 bytes of L1 on device 0 (buffer_id=0x1a0000)
✓ [PID 12345] Allocated 2097152 bytes of L1 on device 0 (buffer_id=0x1b0000)
✓ [PID 12345] Allocated 8388608 bytes of L1 on device 0 (buffer_id=0x1c0000)
```

If you **don't** see these, the buffer tracking is broken.

---

## Add Debug Logging

If still not working, add explicit debug output to `graph_tracking.cpp`:

```cpp
// In graph_tracking.cpp, line 148
if (AllocationClient::is_enabled()) {
    // ADD THIS:
    std::cerr << "[DEBUG] Tracking buffer: device=" << buffer->device()->id()
              << ", size=" << buffer->size()
              << ", type=" << (int)buffer->buffer_type()
              << ", addr=0x" << std::hex << buffer->address() << std::dec
              << std::endl;

    AllocationClient::report_allocation(
        buffer->device()->id(), buffer->size(),
        static_cast<uint8_t>(buffer->buffer_type()), buffer->address());
} else {
    // ADD THIS:
    std::cerr << "[DEBUG] AllocationClient NOT enabled, skipping tracking" << std::endl;
}
```

Then rebuild and run. You should see `[DEBUG]` lines for every allocation.

---

## Quick Checklist

- [ ] `TT_ALLOC_TRACKING_ENABLED=1` is set before running workload
- [ ] Allocation server is running
- [ ] Workload creates L1 buffers (not just DRAM)
- [ ] Server output shows allocations with `buffer_type=1` (L1)
- [ ] Individual allocations are MB-sized, not just KB
- [ ] Sent SIGUSR1 to see total summary

---

## Still Not Working?

**Check raw allocation flow:**

1. Verify buffer allocation calls tracking:
```bash
# Check that graph tracking is hooked into buffer allocation
grep -A5 "GraphTracker::instance().track_allocate" tt_metal/impl/buffers/buffer.cpp
```

2. Verify AllocationClient is compiled:
```bash
# Should exist
ls -la build/lib/libtt_metal.so
nm build/lib/libtt_metal.so | grep AllocationClient
```

3. Check server is receiving messages:
```bash
# Add logging to server
# In allocation_server_poc.cpp, handle_client() function, add:
std::cout << "[SERVER] Received message type: " << (int)msg.type << std::endl;
```

If the server never prints this, messages aren't reaching it.

---

## What Total Should I Expect?

**Minimal workload**: 10-50MB total (kernel code + small buffers)
**Typical workload**: 50-200MB total (kernels + data buffers + CBs)
**Large workload**: 200-500MB+ (across multiple devices, DRAM + L1)

If you're seeing < 1MB total, something is definitely wrong with buffer tracking.
