# Testing the Allocation Tracking Integration

## Quick Test with TTNN

We've created a simple test script that performs basic TTNN operations to verify the allocation tracking works end-to-end.

### Prerequisites

- ✅ TT-Metal rebuilt with integration
- ✅ Allocation server POC built
- ✅ Python environment with ttnn available
- ✅ TT device available (device 0)

### Test Setup (3 Terminals)

#### Terminal 1: Start the Allocation Server

```bash
cd /home/tt-metal-apv/tt_metal/programming_examples/alexp_examples/memory_utilization_monitor
./allocation_server_poc
```

Expected output:
```
🚀 Allocation Server started
   Listening on: /tmp/tt_allocation_server.sock
   Press Ctrl+C to stop
```

#### Terminal 2: Start the Monitor

```bash
cd /home/tt-metal-apv/tt_metal/programming_examples/alexp_examples/memory_utilization_monitor
./allocation_monitor_client -r 500
```

Expected output:
```
📊 Allocation Server Monitor
   Monitoring device 0 via server
   Refresh: 500ms

Device 0 Statistics:
------------------------------------------------------------
  DRAM:           0.00 B /        12.00 GB  [░░░] 0.0%
  L1:             0.00 B /        75.00 MB  [░░░] 0.0%
```

#### Terminal 3: Run the Test

```bash
cd /home/tt-metal-apv/tt_metal/programming_examples/alexp_examples/memory_utilization_monitor

# Enable allocation tracking
export TT_ALLOC_TRACKING_ENABLED=1

# Run the test
python test_ttnn_allocations.py
```

### What the Test Does

The script performs these operations in sequence:

1. **Opens TT Device** → You'll see initial device allocations
2. **Allocates Small Tensor (4MB DRAM)** → Monitor shows +4MB
3. **Allocates Medium Tensor (16MB DRAM)** → Monitor shows +16MB
4. **Allocates L1 Tensor (~1MB L1)** → Monitor shows +1MB in L1
5. **Performs Operations** → May see temporary buffers
6. **Deallocates Tensors** → Monitor shows memory decreasing
7. **Closes Device** → Memory returns to baseline

### Expected Output

#### In Terminal 3 (Test Script)

```
════════════════════════════════════════════════════════════
  TTNN Allocation Tracking Test
════════════════════════════════════════════════════════════
✓ Allocation tracking is ENABLED
ℹ PID: 12345
ℹ Watch the allocation monitor to see real-time memory changes!

[Step 1] Opening TT Device
────────────────────────────────────────────────────────────
ℹ This will allocate initial device structures...
✓ Device opened: <Device object>

[Step 2] Allocating Small Tensor (4MB DRAM)
────────────────────────────────────────────────────────────
ℹ Creating 32x32x32x32 tensor (~4MB)...
✓ Small tensor created on device
...
```

#### In Terminal 1 (Server)

```
✓ [PID 12345] Allocated 1048576 bytes of DRAM on device 0 (buffer_id=...)
✓ [PID 12345] Allocated 4194304 bytes of DRAM on device 0 (buffer_id=...)
✓ [PID 12345] Allocated 16777216 bytes of DRAM on device 0 (buffer_id=...)
✓ [PID 12345] Allocated 1048576 bytes of L1 on device 0 (buffer_id=...)
✗ [PID 12345] Freed buffer ... (4194304 bytes)
✗ [PID 12345] Freed buffer ... (16777216 bytes)
...
```

#### In Terminal 2 (Monitor)

```
Device 0 Statistics:
------------------------------------------------------------
  DRAM:       20.00 MB /        12.00 GB  [█░░░░░] 0.2%
  L1:          1.00 MB /        75.00 MB  [█░░░░░] 1.3%

[Memory usage should increase as tensors are allocated]
[Memory usage should decrease as tensors are deallocated]
```

### Success Criteria

✅ **Test PASSED if you see:**
- Allocation messages in server log matching tensor creation
- Memory usage increasing in monitor when tensors allocated
- Memory usage decreasing in monitor when tensors deallocated
- Final memory close to starting baseline

❌ **Test FAILED if you see:**
- No allocation messages (tracking not working)
- Monitor shows all zeros (not connected)
- Server crashes (bug in integration)
- Python errors (ttnn issues)

### Troubleshooting

#### Issue: No allocations visible

**Check:**
```bash
# 1. Is tracking enabled?
echo $TT_ALLOC_TRACKING_ENABLED  # Should be "1"

# 2. Is server running?
ps aux | grep allocation_server

# 3. Does socket exist?
ls -l /tmp/tt_allocation_server.sock

# 4. Is TT-Metal rebuilt?
ls -lh /home/tt-metal-apv/build/lib/libtt_metal.so
# Should have recent timestamp
```

#### Issue: Python errors

```bash
# Make sure you're using the right Python environment
which python
python -c "import ttnn; print(ttnn.__version__)"

# Make sure device is available
python -c "import ttnn; ttnn.open_device(0)"
```

#### Issue: Server warnings

If you see:
```
[TT-Metal] Warning: Allocation tracking enabled but server not available
```

This means:
- Tracking is enabled in TT-Metal ✅
- But server isn't running ❌

**Solution:** Start the server in Terminal 1

### Advanced Testing

#### Test Multiple Processes

Run multiple instances simultaneously:

```bash
# Terminal 3
export TT_ALLOC_TRACKING_ENABLED=1
python test_ttnn_allocations.py &

# Terminal 4
export TT_ALLOC_TRACKING_ENABLED=1
python test_ttnn_allocations.py &
```

Monitor should show **combined** memory from both processes! 🎉

#### Test Without Tracking

```bash
# Disable tracking
export TT_ALLOC_TRACKING_ENABLED=0

# Run test
python test_ttnn_allocations.py

# Should run normally but no allocations in monitor
# This verifies zero overhead when disabled
```

#### Stress Test

```bash
# Run many allocations rapidly
export TT_ALLOC_TRACKING_ENABLED=1
for i in {1..10}; do
    python test_ttnn_allocations.py &
done

# Monitor should handle all processes smoothly
```

## Alternative: Simple Python One-Liner

If you don't want to run the full script:

```bash
export TT_ALLOC_TRACKING_ENABLED=1

python -c "
import ttnn
import time

print('Opening device...')
device = ttnn.open_device(0)
print('Device opened - check monitor!')
time.sleep(5)

print('Closing device...')
ttnn.close_device(device)
print('Done - memory should return to baseline')
"
```

## Next Steps After Testing

### If Test Passes ✅

Congratulations! Your integration works! Now you can:

1. **Enable tracking for development**
   ```bash
   echo "export TT_ALLOC_TRACKING_ENABLED=1" >> ~/.bashrc
   ```

2. **Set up as systemd service**
   - Deploy allocation server as daemon
   - Auto-start on boot

3. **Create monitoring dashboards**
   - Real-time memory visualization
   - Alerting on high usage

4. **Integrate into CI/CD**
   - Memory leak detection tests
   - Performance regression detection

### If Test Fails ❌

1. Review [INTEGRATION_FIX.md](/home/tt-metal-apv/tt_metal/impl/allocator/INTEGRATION_FIX.md)
2. Check [INTEGRATION_STATUS.md](INTEGRATION_STATUS.md) troubleshooting section
3. Verify all files are in place and built correctly
4. Check server logs for errors

## Summary

This test verifies the complete end-to-end flow:

```
TTNN Operations → TT-Metal Allocator → AllocationClient →
Unix Socket → Allocation Server → Monitor Client → Your Eyes! 👀
```

**Good luck with testing!** 🚀

---

*Quick reference: For more details, see [INTEGRATION_STATUS.md](INTEGRATION_STATUS.md)*
