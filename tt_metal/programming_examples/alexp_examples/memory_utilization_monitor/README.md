# TT-Metal Memory Monitoring Tools

A comprehensive memory tracking and monitoring system for Tenstorrent devices, similar to `nvidia-smi` but with detailed per-buffer tracking and real-time telemetry.

## Overview

This toolset provides:
- **Automatic Memory Tracking**: Tracks all DRAM, L1, L1_SMALL, TRACE buffers, Circular Buffers (CBs), and Kernel code memory across all TT-Metal workloads
- **Real-Time Telemetry**: Device temperature, power, clock frequencies, and more via UMD firmware access
- **Process-Level Tracking**: See which processes are using device memory
- **Historical Metrics**: Charts showing memory usage over time
- **Automatic Cleanup**: Detects dead processes and cleans up orphaned allocations

## Components

### 1. `allocation_server_poc` - Allocation Tracking Server

A centralized server that tracks all memory allocations across multiple processes using Unix domain sockets.

**Features:**
- Tracks DRAM, L1, L1_SMALL, TRACE, CB, and Kernel allocations
- Automatic device detection (local PCIe and remote devices via cluster topology)
- Process lifecycle tracking with automatic cleanup of dead processes
- Reference counting for cached program buffers
- Background cleanup thread (runs every 3 seconds)

**Usage:**
```bash
# Start the server (keeps running in background)
./allocation_server_poc

# Or start in quiet mode (no output)
./allocation_server_poc --quiet
```

The server will:
1. Auto-detect all Tenstorrent devices (local and remote)
2. Listen on `/tmp/tt_allocation_server.sock`
3. Print allocation/deallocation events as they happen
4. Clean up orphaned allocations from dead processes

**Output Example:**
```
🔍 Device detection (using UMD Cluster for accurate specs):
   Device 0: Blackhole (34GB DRAM, 1024MB L1) [Local]
   Device 1: Blackhole (34GB DRAM, 1024MB L1) [Remote]
   Total: 2 device(s) detected
🚀 Allocation Server started
   Listening on: /tmp/tt_allocation_server.sock
   Press Ctrl+C to stop

✓ [PID 12345] Allocated 4194304 bytes of DRAM on device 0 (buffer_id=0x7f8a2c000000)
✓ [KERNEL_LOAD] Application kernel on Device 0: +2.5 MB (Total: 382.2 MB)
✗ [PID 12345] Freed buffer 0x7f8a2c000000 on device 0 (4194304 bytes, FINAL)
✗ [KERNEL_UNLOAD] Application kernel on Device 0: -2.5 MB (Total: 379.7 MB)
```

### 2. `tt_smi_umd` - System Management Interface

An interactive monitoring tool similar to `nvidia-smi` or `nvtop` that displays real-time device status, memory usage, and telemetry.

**Features:**
- Real-time device telemetry (temperature, power, clocks) via UMD firmware access
- Memory breakdown by type (DRAM, L1, CBs, Kernels)
- Historical charts showing usage over the last 60 seconds
- Multiple view modes: Main, Charts, Detailed Telemetry
- Process list showing which PIDs are using devices
- Works with local AND remote devices

**Usage:**
```bash
# Single snapshot
./tt_smi_umd

# Watch mode (refreshes every 500ms)
./tt_smi_umd -w

# Custom refresh rate (e.g., 1000ms)
./tt_smi_umd -w -r 1000

# Use sysfs instead of UMD (comparison mode)
./tt_smi_umd --sysfs
```

**Interactive Controls (Watch Mode):**
- `1` - Switch to Main View (default)
- `2` - Switch to Charts View (60-second history)
- `3` - Switch to Detailed Telemetry View
- `q` - Quit

**Output Example:**
```
┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ tt-smi-umd v1.0 (with UMD telemetry)                                Tue Nov 18 15:30:45 2025 │
├────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ GPU  Name            Temp      Power     AICLK       Memory-Usage                                  │
├────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 0    Blackhole      45°C      25W       1000 MHz    2.5GB/34.1GB                                  │
│ 1    Blackhole      43°C      23W       1000 MHz    N/A                                            │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘

Memory Breakdown:
Device 0 (Blackhole):
----------------------------------------------------------------------
  DRAM:       2.5GB     / 34.1GB      [████████░░░░░░░░░░░░░░░] 7.3%
  L1 Memory:  382.2MB   / 1024.0MB    [████████░░░░░░░░░░░░░░░] 37.3%
    Buffers:  127.4MB
    CBs:      127.4MB
    Kernels:  127.4MB
```

## Complete Workflow

### Step 1: Enable Tracking in Your Application

Set the environment variable to enable automatic tracking:

```bash
export TT_ALLOC_TRACKING_ENABLED=1
```

This tells TT-Metal to report all allocations to the tracking server.

### Step 2: Start the Allocation Server

```bash
# In Terminal 1
cd $TT_METAL_HOME$
./build_Release/tt_metal/programming_examples/alexp_examples/memory_utilization_monitor/allocation_server_poc
```

Leave this running in the background. It will track all memory events.

### Step 3: Start the Monitoring Tool

```bash
# In Terminal 2
cd $TT_METAL_HOME$
./build_Release/tt_metal/programming_examples/alexp_examples/memory_utilization_monitor/tt_smi_umd -w
```

This will show real-time memory usage and telemetry in a continuously updating display.

### Step 4: Run Your Workload

```bash
# In Terminal 3
cd $TT_METAL_HOME$
export TT_ALLOC_TRACKING_ENABLED=1

# Run any TT-Metal program
python your_model.py
# or
./your_cpp_program
```

Watch Terminal 1 to see allocation events and Terminal 2 to see real-time memory usage.

### Step 5: Verify Cleanup

After your workload finishes:
- Terminal 1 (server) should show deallocations
- Terminal 2 (tt_smi_umd) should show memory usage drop to near 0 MB
- Any remaining memory is likely persistent system kernels (dispatch, fabric)

## Building from Source

The tools are built automatically as part of the TT-Metal build:

```bash
cd $TT_METAL_HOME$
cmake -B build_Release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build_Release -j $(nproc)
```

Executables will be in:
- `build_Release/tt_metal/programming_examples/alexp_examples/memory_utilization_monitor/allocation_server_poc`
- `build_Release/tt_metal/programming_examples/alexp_examples/memory_utilization_monitor/tt_smi_umd`

## Architecture

### Message Protocol

All communication uses a packed struct (`AllocMessage`) sent over Unix domain sockets:

```cpp
struct __attribute__((packed)) AllocMessage {
    Type type;              // ALLOC, FREE, QUERY, CB_ALLOC, KERNEL_LOAD, etc.
    int32_t device_id;      // Device ID
    uint64_t size;          // Size in bytes
    uint8_t buffer_type;    // Buffer type or kernel type
    int32_t process_id;     // Owner PID
    uint64_t buffer_id;     // Buffer address or kernel ID
    // ... response fields ...
};
```

### Tracking Integration

The tracking is integrated into TT-Metal's core allocator via `GraphTracker`:

1. **Buffer Tracking**: `Buffer::allocate_impl()` and `Buffer::deallocate()` call `GraphTracker::track_allocate()` / `track_deallocate()`
2. **CB Tracking**: `track_allocate_cb()` / `track_deallocate_cb()` report circular buffer events
3. **Kernel Tracking**: `track_kernel_load()` / `track_kernel_unload()` report kernel lifecycle events
4. **Client Library**: `AllocationClient` sends messages to the server via Unix domain sockets

### Device Detection

The server uses UMD's `Cluster` API to detect devices:
- **Local devices**: PCIe-attached devices that can be accessed directly
- **Remote devices**: Devices accessed via ethernet (memory tracking only, no telemetry)
- **SocDescriptor**: Provides accurate specs including harvesting info

## Troubleshooting

### Server not receiving events

**Problem:** `tt_smi_umd` shows "N/A" for memory usage

**Solution:**
1. Ensure `allocation_server_poc` is running
2. Set `export TT_ALLOC_TRACKING_ENABLED=1` before running your workload
3. Check that `/tmp/tt_allocation_server.sock` exists

### Telemetry shows N/A

**Problem:** Temperature, power, clocks show "N/A"

**Possible causes:**
- Device is remote (no direct firmware access)
- Device is in use by another process
- UMD initialization failed

**Solution:**
- Close other tools using the devices
- For remote devices, telemetry is not available (by design)

### Memory doesn't drop to 0 after workload

**Expected behavior:** Some memory (dispatch, fabric kernels) is persistent and stays loaded for the entire device session. This is normal.

**Unexpected behavior:** If application kernels don't deallocate, check:
1. Is `TT_ALLOC_TRACKING_ENABLED=1` set?
2. Did the program exit cleanly (not crashed)?
3. Check server logs for deallocation events

### Build errors

**Problem:** `allocation_client.hpp` or `graph_tracking.hpp` not found

**Solution:** Ensure you've built the full TT-Metal project:
```bash
cmake --build build_Release -j $(nproc)
```

## Advanced Usage

### Dump Remaining Buffers

```bash
# Connect to server and dump all remaining allocations
echo "" | nc -U /tmp/tt_allocation_server.sock
```

### Query Device Info

```python
import socket
import struct

sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect('/tmp/tt_allocation_server.sock')

# Send DEVICE_INFO_QUERY (type=6, device_id=-1 for count)
msg = struct.pack('<BxxxiQBxxxiQQ' + 'Q'*6 + 'Q'*2 + 'I'*6 + 'I'*4, 6, -1, 0, 0, 0, 0, 0, ...)
sock.send(msg)

response = sock.recv(128)
# Parse response...
```

### Custom Tracking

If you're writing custom TT-Metal code, you can report allocations directly:

```cpp
#include "tt_metal/impl/allocator/allocation_client.hpp"

// Report an allocation
AllocationClient::report_allocation(device_id, size, buffer_type, address);

// Report a deallocation
AllocationClient::report_deallocation(device_id, address);
```

## Performance Impact

The tracking system has minimal overhead:
- **Allocation/Deallocation**: ~1-2 microseconds per event (async socket send)
- **Network**: Unix domain sockets are very fast (local memory copy)
- **Server**: Handles 1000+ events/second easily
- **Memory**: ~100 bytes per tracked buffer in server

## Related Files

- `allocation_server_poc.cpp` - Server implementation
- `tt_smi_umd.cpp` - Monitoring tool implementation
- `tt_metal/impl/allocator/allocation_client.cpp` - Client library
- `tt_metal/graph/graph_tracking.cpp` - Integration into TT-Metal
- `tt_metal/impl/buffers/buffer.cpp` - Buffer tracking hooks
- `tt_metal/impl/program/program.cpp` - Kernel tracking hooks

## License

SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
SPDX-License-Identifier: Apache-2.0
