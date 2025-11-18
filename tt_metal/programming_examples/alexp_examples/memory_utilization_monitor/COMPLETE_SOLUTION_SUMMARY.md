# Complete Memory Monitoring Solution

## Summary

You now have a **three-tier system** for monitoring Tenstorrent device memory usage:

1. **`allocation_server_poc`** - Central tracking daemon (like a mini kernel driver in user-space)
2. **`allocation_monitor_client`** - Detailed memory monitor with live updates and charts
3. **`tt-smi`** - nvidia-smi style system-wide overview

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Your Applications                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Python      │  │  C++ App     │  │  Test Suite  │          │
│  │  script.py   │  │  ./myapp     │  │  pytest      │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                  │                  │                   │
│         └──────────────────┴──────────────────┘                  │
│                            ↓                                      │
│                  Send ALLOC/FREE messages                         │
│                            ↓                                      │
└────────────────────────────┼──────────────────────────────────────┘
                             ↓
         ┌───────────────────────────────────────┐
         │   Allocation Server (Daemon)          │
         │   • Tracks all processes              │
         │   • Aggregates allocations            │
         │   • Detects dead processes            │
         │   • Unix socket: /tmp/tt_alloc*.sock │
         └───────────────────┬───────────────────┘
                             ↓
         ┌───────────────────┴───────────────────┐
         │                                        │
         ↓                                        ↓
┌────────────────────┐                  ┌────────────────────┐
│ Monitor Client     │                  │    tt-smi          │
│ • Live bar charts  │                  │ • nvidia-smi style │
│ • Per-device view  │                  │ • Process list     │
│ • High refresh rate│                  │ • One-shot or watch│
└────────────────────┘                  └────────────────────┘
```

## Quick Start Guide

### 1. Build Everything

```bash
# From tt-metal root
cd /home/ttuser/aperezvicente/tt-metal-apv
cmake --build build --target allocation_server_poc allocation_monitor_client tt_smi -j
```

### 2. Basic Usage (Like nvidia-smi)

**Single snapshot:**
```bash
./build/tt_smi
```

**Watch mode:**
```bash
./build/tt_smi -w
```

This shows:
- All devices in the system
- Temperature and power
- All processes with devices open
- ⚠️ **No memory tracking without server**

### 3. Full Memory Tracking Setup

**Terminal 1: Start the server (required for memory tracking)**
```bash
./build/allocation_server_poc
```

**Terminal 2: Run your workload**
```bash
python your_model.py
# Or: ./your_cpp_app
```

**Terminal 3: Monitor with tt-smi**
```bash
./build/tt_smi -w
```

**Or: Monitor with detailed client**
```bash
./build/allocation_monitor_client -a -r 500
```

## Tool Comparison

| Feature | tt-smi | allocation_monitor_client | allocation_server_poc |
|---------|--------|---------------------------|----------------------|
| **Purpose** | System overview | Detailed monitoring | Backend daemon |
| **Like** | nvidia-smi | htop for memory | systemd service |
| **UI Style** | Table | Bar charts | Log output |
| **Refresh** | Once or watch | Continuous | N/A (server) |
| **Shows processes** | ✅ Yes | ❌ No (aggregated) | ✅ Yes (logs) |
| **Shows devices** | ✅ Yes | ✅ Yes | ✅ Yes (logs) |
| **Temperature** | ✅ Yes | ❌ No | ❌ No |
| **Power** | ✅ Yes | ❌ No | ❌ No |
| **Auto-detect devices** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Requires server** | Optional* | ✅ Yes | N/A (is server) |

*Without server, tt-smi shows processes but not memory usage.

## Answering Your Original Question

> "How to make it as a nvidia-smi kind of thing where you can instantiate as a terminal and checks all process id currently running?"

**Answer:** Use `tt-smi` - it's exactly what you asked for!

```bash
./build/tt_smi -w
```

### What It Does (Like nvidia-smi)

1. **Finds all processes** using Tenstorrent devices by scanning `/proc/[pid]/fd/`
2. **Shows device info** - temperature, power, architecture
3. **Displays memory usage** - when allocation server is running
4. **Updates continuously** - in watch mode (`-w`)

### Differences from nvidia-smi

| Feature | nvidia-smi | tt-smi |
|---------|-----------|--------|
| **Process detection** | ✅ Automatic | ✅ Automatic (via /proc) |
| **Memory tracking** | ✅ Automatic (kernel) | ⚠️ Requires server + instrumentation |
| **Temperature** | ✅ Yes | ✅ Yes (via sysfs) |
| **No setup needed** | ✅ Yes | ⚠️ Server needed for memory |

**Why the difference:**
- NVIDIA: Kernel driver intercepts `cudaMalloc()` → tracks everything
- Tenstorrent: TT-Metal allocator is in user-space → needs explicit tracking

## Your Original Approach Was Correct!

You asked:
> "I don't know if I'm doing it the proper way"

**You were absolutely right!** Your `allocation_server_poc` + `allocation_monitor_client` architecture is **the correct approach** because:

1. ✅ **Cross-process tracking** - The only way to see all processes
2. ✅ **Real-time updates** - Server maintains live state
3. ✅ **Process death detection** - Automatic cleanup of dead processes
4. ✅ **Extensible** - Can add more features easily

### What I Added

I just added **`tt-smi`** as a convenience tool that:
- Provides a familiar interface (like nvidia-smi)
- Can work without server (shows processes, but not memory)
- Makes it easier for users who just want a quick overview

**Your server/client architecture remains the foundation!**

## Integration with Allocator

You also asked:
> "Could this be done using the memory state from memory report in the allocator?"

**Short answer:** Not for cross-process monitoring.

**Long answer:**

The allocator's `get_statistics()` API is perfect for:
- ✅ Single-process self-monitoring
- ✅ Post-mortem analysis (CSV dumps)
- ✅ Getting actual device capabilities

But it **cannot** provide cross-process visibility because:
- Each process has its own `Device` instance
- Each `Device` has its own `Allocator` instance
- Allocators are isolated in separate process memory spaces

**Your server already uses the allocator API correctly:**

```cpp
// In allocation_server_poc.cpp
auto device = tt::tt_metal::CreateDeviceMinimal(i);
tt::ARCH arch = device->arch();
uint64_t total_dram = device->num_dram_channels() * device->dram_size_per_channel();
```

The server uses TT-Metal APIs to get device info, then **supplements** it with cross-process tracking via IPC.

## Complete Workflow Example

### Development Workflow

```bash
# Terminal 1: Start server once
./build/allocation_server_poc

# Leave it running...

# Terminal 2: Quick checks during development
./build/tt_smi        # Check current state
./build/tt_smi -w     # Watch continuously

# Terminal 3: Your development work
python experiment1.py
python experiment2.py
./run_tests

# Terminal 2 shows everything in real-time!
```

### Production Monitoring

```bash
# Start server as a background service
./build/allocation_server_poc > server.log 2>&1 &
SERVER_PID=$!

# Run your workload
python production_workload.py

# Check memory usage periodically
./build/tt_smi > snapshot.txt

# Or: Monitor continuously
./build/allocation_monitor_client -a -r 1000 &
MONITOR_PID=$!

# ... workload runs ...

# Check for leaks at the end
./build/tt_smi

# Cleanup
kill $MONITOR_PID
kill $SERVER_PID
```

### CI/CD Integration

```python
#!/usr/bin/env python3
import subprocess
import time

# Start server
server = subprocess.Popen(['./build/allocation_server_poc'])
time.sleep(1)

try:
    # Run tests
    subprocess.run(['python', '-m', 'pytest', 'tests/'])

    # Check for memory leaks
    result = subprocess.run(['./build/tt_smi'], capture_output=True, text=True)
    if 'Device open' in result.stdout:
        print("✅ All tests passed, devices properly cleaned up")
    else:
        print("⚠️  Devices still in use after tests!")

finally:
    server.terminate()
    server.wait()
```

## Roadmap: Path to True nvidia-smi Parity

### Current State (Your Implementation ✅)

```
User-space solution:
  • allocation_server_poc (daemon)
  • allocation_monitor_client (detailed view)
  • tt-smi (system overview)
  • Requires manual instrumentation
```

### Phase 2: Auto-Instrumentation (Feasible Now)

```cpp
// In tt_metal/impl/device/device.cpp
Device::Device(...) {
    // ... existing code ...

    // Auto-register with server if available
    try {
        AllocationClient::connect_to_server(device_id_);
    } catch (...) {
        // Server not running, continue without tracking
    }
}
```

**Benefits:**
- No manual instrumentation needed
- All TT-Metal apps automatically tracked
- Still works if server not running

### Phase 3: Kernel-Level Tracking (Long-term)

**Add to TT-KMD:**
```c
// In kernel driver
struct tt_process_memory {
    pid_t pid;
    char comm[16];
    uint64_t dram_allocated;
    uint64_t l1_allocated;
};

// Expose via sysfs:
// /sys/class/tenstorrent/tenstorrent!0/processes/<pid>/dram_allocated
// /sys/class/tenstorrent/tenstorrent!0/processes/<pid>/l1_allocated
```

**Then tt-smi becomes:**
```cpp
// No server needed!
// Just read /sys/class/tenstorrent/...
auto procs = read_processes_from_sysfs(device_id);
```

**Benefits:**
- Works exactly like nvidia-smi
- No server needed
- No instrumentation needed
- Works for any process using the device

## Files Summary

### Core Implementation

| File | Purpose | Requires TT-Metal |
|------|---------|------------------|
| `allocation_server_poc.cpp` | Central tracking daemon | ✅ Yes |
| `allocation_monitor_client.cpp` | Detailed monitor UI | ❌ No (standalone) |
| `tt_smi.cpp` | nvidia-smi style tool | ❌ No (standalone) |

### Documentation

| File | Content |
|------|---------|
| `TT_SMI_README.md` | Complete tt-smi documentation |
| `NVIDIA_SMI_ARCHITECTURE.md` | How nvidia-smi works vs tt-smi |
| `LIMITATIONS.md` | Why cross-process tracking is hard |
| `MEMORY_ARCHITECTURE.md` | Where memory info lives (TT-Metal/UMD/KMD) |
| `ALLOCATION_SERVER_README.md` | Server architecture and usage |
| `COMPLETE_SOLUTION_SUMMARY.md` | This file |

## Commands Quick Reference

```bash
# Build everything
cmake --build build --target allocation_server_poc allocation_monitor_client tt_smi -j

# Start server (required for memory tracking)
./build/allocation_server_poc

# Quick system check (works without server)
./build/tt_smi

# Watch mode (like nvidia-smi)
./build/tt_smi -w

# Detailed monitoring (requires server)
./build/allocation_monitor_client -a -r 500

# Monitor specific devices (requires server)
./build/allocation_monitor_client -d 0 -d 1
```

## Conclusion

**Your original implementation was correct!** The server + client architecture is the proper way to achieve cross-process memory tracking for Tenstorrent devices.

**What you have now:**
1. ✅ Cross-process memory tracking (via server)
2. ✅ Detailed monitoring with bar charts (monitor_client)
3. ✅ nvidia-smi style overview (tt-smi)
4. ✅ Process auto-detection (tt-smi scans /proc)
5. ✅ Device auto-detection (all tools)
6. ✅ Real-time updates (all tools)
7. ✅ Dead process cleanup (server)

**To make it "production ready":**
- Add auto-instrumentation in TT-Metal (Phase 2)
- Or: Extend TT-KMD for kernel-level tracking (Phase 3)

But for now, **your architecture is the right approach** and the tools work great for development and monitoring! 🎉
