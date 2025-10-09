# 8-Device Mesh Allocation Test

## Overview

`test_mesh_allocation.py` validates allocation tracking across an 8-device mesh (2x4 topology) by:
- Creating distributed tensors across all 8 devices
- Performing parallel computations (add, matmul)
- Tracking memory allocations on each device
- Deallocating tensors and verifying memory is freed

## What It Tests

✅ **Multi-device allocation tracking** - All 8 devices report allocations
✅ **Distributed tensors** - Memory spread across mesh
✅ **Parallel computation** - Operations on all devices simultaneously
✅ **Deallocation tracking** - Memory freed on all devices
✅ **Correct device IDs** - No corruption (0-7, not random numbers)

## Prerequisites

1. **8 Tenstorrent devices available** (2x4 mesh)
2. **Allocation server running**
3. **Python bindings rebuilt** (after device ID fixes)
4. **Tracking enabled** via environment variable

## How to Run

### Step 1: Rebuild Python Bindings (If Not Done)

```bash
cd /home/tt-metal-apv
./build_metal.sh
```

This is required for Python to have the device ID fixes.

### Step 2: Start Allocation Server

```bash
cd /home/tt-metal-apv/tt_metal/programming_examples/alexp_examples/memory_utilization_monitor
./allocation_server_poc
```

### Step 3: Start Multi-Device Monitor (Optional)

In another terminal:
```bash
cd /home/tt-metal-apv/tt_metal/programming_examples/alexp_examples/memory_utilization_monitor

# Monitor all 8 devices
./allocation_monitor_client -a -r 500

# Or specific devices
./allocation_monitor_client -d 0 -d 1 -d 2 -d 3 -d 4 -d 5 -d 6 -d 7 -r 500
```

### Step 4: Run the Test

In another terminal:
```bash
cd /home/tt-metal-apv/tt_metal/programming_examples/alexp_examples/memory_utilization_monitor

# Enable tracking and run
export TT_ALLOC_TRACKING_ENABLED=1
python test_mesh_allocation.py
```

## What You Should See

### In the Test Output:
```
[Step 1] Opening 8-Device Mesh (2x4 Topology)
✓ Mesh device opened: MeshDevice(2x4 grid, 8 devices)

Mesh topology shows devices 0-7 in 2x4 grid:
┌──────────────┬─────────────┬──────────────┬─────────────┐
│  Dev. ID: 4  │ Dev. ID: 0  │  Dev. ID: 3  │ Dev. ID: 7  │
├──────────────┼─────────────┼──────────────┼─────────────┤
│  Dev. ID: 5  │ Dev. ID: 1  │  Dev. ID: 2  │ Dev. ID: 6  │
└──────────────┴─────────────┴──────────────┴─────────────┘

[Step 2] Creating Distributed Tensor (100MB per device)
✓ Tensor distributed across mesh!
📊 Each device should show ~100MB DRAM allocation

[Step 3] Creating Second Distributed Tensor (100MB per device)
✓ Second tensor distributed!
📊 Each device should now show ~200MB DRAM allocation

[Step 4] Performing Distributed Computation (Add)
✓ Distributed addition completed!

[Step 5] Performing Distributed MatMul
✓ Distributed matmul completed!

[Step 6] Deallocating Tensors - WATCH MEMORY DROP!
✓ All tensors deallocated - memory should be back to baseline!
```

### In the Allocation Server:
```
✓ [PID 12345] Allocated 104857600 bytes of DRAM on device 0 (buffer_id=...)
✓ [PID 12345] Allocated 104857600 bytes of DRAM on device 1 (buffer_id=...)
✓ [PID 12345] Allocated 104857600 bytes of DRAM on device 2 (buffer_id=...)
... (allocations on all 8 devices)

📊 Current Statistics:
  Device 0:
    Buffers: 2
    DRAM: 209715200 bytes
    Total: 209715200 bytes
  Device 1:
    Buffers: 2
    DRAM: 209715200 bytes
    Total: 209715200 bytes
  ... (stats for all 8 devices)
  Active allocations: 16

✗ [PID 12345] Freed buffer ... (deallocation messages)
```

### In the Monitor Client:
You should see memory bars for all 8 devices going up and down as tensors are allocated and deallocated.

## Test Sequence

1. **Open Mesh** - Initializes 8 devices in 2x4 topology
2. **Allocate Tensor 1** - ~100MB DRAM per device (800MB total)
3. **Allocate Tensor 2** - ~100MB DRAM per device (800MB total)
4. **Compute Add** - Parallel addition on all devices
5. **Compute MatMul** - Parallel matrix multiplication
6. **Deallocate** - Free all tensors, memory returns to baseline

## Mesh Topology

The 2x4 mesh distributes tensors like this:

```
Row 0: [Device 4] [Device 0] [Device 3] [Device 7]
Row 1: [Device 5] [Device 1] [Device 2] [Device 6]
```

Each device gets a shard of the tensor based on the sharding configuration.

## Troubleshooting

### "Not enough devices"
You need 8 Tenstorrent devices. Check with:
```bash
tt-smi
```

### "No allocations showing"
1. Check `TT_ALLOC_TRACKING_ENABLED=1` is set
2. Verify allocation server is running
3. Make sure Python bindings were rebuilt in `build_Release_tracy`

### "Corrupted device IDs"
Python bindings need to be rebuilt with the device ID fixes:
```bash
cd /home/tt-metal-apv
./build_metal.sh
```

## Comparison with Other Tests

| Test | Devices | Purpose |
|------|---------|---------|
| `test_tracking_cpp.cpp` | 1 | C++ verification |
| `test_ttnn_allocations.py` | 1 | Single-device Python test |
| `test_persistent_memory.py` | 1 | Long-running single-device |
| **`test_mesh_allocation.py`** | **8** | **Multi-device mesh test** |

## Success Criteria

✅ All 8 devices show allocations
✅ Device IDs are correct (0-7)
✅ Memory distributed evenly across mesh
✅ Computations complete without errors
✅ Deallocations tracked on all devices
✅ Memory returns to baseline

This test validates that the allocation tracking system works correctly in a real multi-device, distributed computing scenario!
