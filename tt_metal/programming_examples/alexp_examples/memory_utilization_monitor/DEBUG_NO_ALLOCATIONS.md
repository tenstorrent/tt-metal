# Debug: No Allocations Showing

## Symptoms
- `allocation_server_poc` is running
- `TT_ALLOC_TRACKING_ENABLED=1` is set
- C++ test or Python script runs
- **But NO allocations appear in the server**

## Checklist

### 1. Is the Server Actually Running?
```bash
ps aux | grep allocation_server_poc | grep -v grep
```
Should show a running process.

### 2. Does the Socket Exist?
```bash
ls -l /tmp/tt_allocation_server.sock
```
Should show: `srwxr-xr-x ... /tmp/tt_allocation_server.sock`

### 3. Is the Environment Variable Set?
```bash
echo $TT_ALLOC_TRACKING_ENABLED
```
Should output: `1`

**CRITICAL**: The environment variable must be set in the SAME terminal/shell where you run the test!

### 4. Was the Library Rebuilt?
```bash
# Check when impl was last built
ls -lh /home/tt-metal-apv/build/lib/libtt_metal.so
```

If this file is older than your code changes, rebuild:
```bash
cd /home/tt-metal-apv
cmake --build build --target impl -j8
```

### 5. Connection Test
Run this to see detailed connection info:
```bash
cd /home/tt-metal-apv/tt_metal/programming_examples/alexp_examples/memory_utilization_monitor
export TT_ALLOC_TRACKING_ENABLED=1
./test_connection.sh
```

## Common Issues

### Issue 1: Environment Variable Not Set in Test Shell

**Problem**: You set `TT_ALLOC_TRACKING_ENABLED=1` in one terminal, but run the test in another.

**Solution**: Set it in the SAME terminal where you run the test:
```bash
# Terminal where you run the test
export TT_ALLOC_TRACKING_ENABLED=1
./build/programming_examples/test_tracking_cpp
```

### Issue 2: Server Started After Test

**Problem**: The test connects to the server during initialization. If the server isn't running yet, the connection fails silently.

**Solution**: Always start the server FIRST, then run the test:
```bash
# Terminal 1: Start server FIRST
./allocation_server_poc

# Terminal 2: Then run test
export TT_ALLOC_TRACKING_ENABLED=1
./test_tracking_cpp
```

### Issue 3: Old Library Loaded

**Problem**: Python or C++ is loading an old version of the library without the tracking code.

**For C++**:
```bash
# Rebuild impl
cd /home/tt-metal-apv
cmake --build build --target impl -j8

# Rebuild your test
cmake --build build --target test_tracking_cpp -j4
```

**For Python**:
```bash
# Full rebuild required
cd /home/tt-metal-apv
./build_metal.sh
```

### Issue 4: Socket Permissions

**Problem**: Socket file has wrong permissions.

**Solution**:
```bash
# Remove old socket
rm -f /tmp/tt_allocation_server.sock

# Restart server
./allocation_server_poc
```

### Issue 5: Connection Silently Fails

**Problem**: The `AllocationClient` tries to connect but fails silently (only warns once).

**Debug**: Look for this warning when running your test:
```
[TT-Metal] Warning: Allocation tracking enabled but server not available at /tmp/tt_allocation_server.sock
```

If you see this, the server isn't running or the socket doesn't exist.

## Verification Steps

### Step 1: Verify Server is Ready
```bash
# Terminal 1
cd /home/tt-metal-apv/tt_metal/programming_examples/alexp_examples/memory_utilization_monitor
./allocation_server_poc

# You should see:
# 🚀 Allocation Server started
#    Listening on: /tmp/tt_allocation_server.sock
#    Press Ctrl+C to stop
```

### Step 2: Verify Environment
```bash
# Terminal 2
export TT_ALLOC_TRACKING_ENABLED=1
echo "Tracking enabled: $TT_ALLOC_TRACKING_ENABLED"
```

### Step 3: Run Test
```bash
# Same terminal (Terminal 2)
cd /home/tt-metal-apv
./build/programming_examples/test_tracking_cpp
```

### Step 4: Check Server Output
Go back to Terminal 1. You should see:
```
✓ [PID 12345] Allocated 1073741824 bytes of DRAM on device 0 (buffer_id=...)
```

## Still Not Working?

### Add Debug Output

Edit `/home/tt-metal-apv/tt_metal/impl/allocator/allocator.cpp` and add debug prints:

```cpp
DeviceAddr Allocator::allocate_buffer(Buffer* buffer) {
    // ... existing code ...

    // NEW: Report allocation to tracking server
    std::cerr << "[DEBUG] AllocationClient::is_enabled() = "
              << AllocationClient::is_enabled() << std::endl;
    std::cerr << "[DEBUG] device_id_ = " << device_id_ << std::endl;

    if (AllocationClient::is_enabled() && device_id_ >= 0) {
        std::cerr << "[DEBUG] Reporting allocation: device=" << device_id_
                  << " size=" << size << " type=" << (int)buffer_type
                  << " addr=" << address << std::endl;
        AllocationClient::report_allocation(
            device_id_,
            size,
            static_cast<uint8_t>(buffer_type),
            address
        );
    }
    return address;
}
```

Then rebuild and run again to see what's happening.

## Quick Test Script

Use this to test everything at once:
```bash
cd /home/tt-metal-apv/tt_metal/programming_examples/alexp_examples/memory_utilization_monitor
./test_connection.sh
```

This will check all prerequisites and run the test.
