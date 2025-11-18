# tt-smi: Tenstorrent System Management Interface

A `nvidia-smi` style tool for monitoring Tenstorrent devices and processes.

## Overview

`tt-smi` provides a system-wide view of:
- All Tenstorrent devices in the system
- Device temperature and power consumption
- Memory utilization (when allocation server is running)
- All processes using Tenstorrent devices
- Per-process memory usage (when processes are instrumented)

## Quick Start

### 1. Build the Tools

```bash
# From tt-metal root
cmake --build build --target tt_smi allocation_server_poc -j
```

### 2. Basic Usage (No Server Needed)

Show current state once:
```bash
./build/tt_smi
```

Watch mode (like `watch nvidia-smi`):
```bash
./build/tt_smi -w
```

Fast refresh (500ms):
```bash
./build/tt_smi -w -r 500
```

### 3. With Memory Tracking

For full memory tracking, start the allocation server in a separate terminal:

**Terminal 1:**
```bash
./build/allocation_server_poc
```

**Terminal 2:**
```bash
./build/tt_smi -w
```

## Sample Output

### Without Allocation Server

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ tt-smi v1.0                                                          Mon Nov  3 12:34:56 2025      │
├────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ GPU Name           Temp      Power     Memory-Usage        Utilization                             │
├────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 0   Wormhole_B0    65°C      150W      N/A                                                         │
│ 1   Wormhole_B0    62°C      145W      N/A                                                         │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘

Processes:
┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ PID     Name                Device  DRAM        L1          Status                                │
├────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 12345   python3             0       N/A         N/A         Device open (no tracking)              │
│ 12346   test_app            0       N/A         N/A         Device open (no tracking)              │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘

💡 TIP: For memory tracking, start the allocation server:
   ./allocation_server_poc
```

### With Allocation Server Running

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ tt-smi v1.0                                                          Mon Nov  3 12:34:56 2025      │
├────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ GPU Name           Temp      Power     Memory-Usage        Utilization                             │
├────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 0   Wormhole_B0    65°C      150W      2.4GB/12.0GB        [███████████████░░░░░] 75%             │
│ 1   Wormhole_B0    62°C      145W      800.0MB/12.0GB      [███░░░░░░░░░░░░░░░░░] 25%             │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘

Processes:
┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ PID     Name                Device  DRAM        L1          Status                                │
├────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 12345   python3             0       1.2GB       45.0MB      Connected to server                    │
│ 12346   test_app            0       800.0MB     30.0MB      Connected to server                    │
│ 12347   python3             1       400.0MB     15.0MB      Connected to server                    │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘

💡 TIP: Processes must be instrumented to report memory to the server
```

## How It Works

### Three Levels of Monitoring

#### Level 1: Device Detection (Always Works)

`tt-smi` scans `/proc/[pid]/fd/` to find all processes with Tenstorrent device files open:

```bash
# Internally does something like:
for pid in /proc/[0-9]*; do
    if ls -l $pid/fd | grep -q /dev/tenstorrent/; then
        echo "Found process: $pid"
    fi
done
```

**What you get:**
- ✅ List of all PIDs using devices
- ✅ Process names
- ❌ No memory usage info

#### Level 2: Device Info (Requires Allocation Server)

When the allocation server is running, `tt-smi` queries it for:
- Device architecture (Grayskull, Wormhole_B0, etc.)
- Total memory capacity (DRAM, L1)
- Aggregate memory usage across all processes

**What you get:**
- ✅ Device specs
- ✅ Total memory used (all processes combined)
- ✅ Memory utilization percentage
- ❌ No per-process breakdown (yet)

#### Level 3: Per-Process Tracking (Requires Instrumentation)

**Future enhancement:** When processes are instrumented to report allocations:

```cpp
// In your application code
#include <tt-metalium/allocation_client.hpp>

// Allocations automatically reported to server
auto buffer = Buffer::create(...);
```

**What you get:**
- ✅ Per-process DRAM usage
- ✅ Per-process L1 usage
- ✅ Process status (connected/not connected)

## Comparison with nvidia-smi

### What nvidia-smi Does

```bash
nvidia-smi
```

Shows:
- ✅ All GPUs
- ✅ Temperature, power, utilization
- ✅ All processes using GPUs
- ✅ Per-process memory usage **without any instrumentation**

**How:** NVIDIA kernel driver tracks ALL allocations at kernel level.

### What tt-smi Does

```bash
./tt_smi
```

**Current capabilities:**
- ✅ All Tenstorrent devices
- ✅ Temperature, power (from sysfs)
- ✅ All processes with devices open (via /proc)
- ⚠️ Memory tracking requires allocation server + instrumentation

**Why the difference:**
- NVIDIA: Kernel driver intercepts `cudaMalloc()` and tracks everything
- Tenstorrent: TT-KMD doesn't track allocations (they happen in TT-Metal, user-space)

### Architecture Comparison

**NVIDIA:**
```
Application → libcuda.so → NVIDIA KMD (tracks here) → /proc/driver/nvidia
                                      ↓
                               nvidia-smi reads /proc
```

**Tenstorrent (Current):**
```
Application → TT-Metal → Allocator (in-process, isolated)
                              ↓
                      No system-wide tracking
                              ↓
                      tt-smi can only see PIDs via /proc/fd
```

**Tenstorrent (With Allocation Server):**
```
Application → TT-Metal → Allocation Server (user-space daemon)
                              ↓
                      tt-smi queries server
```

## Use Cases

### 1. Quick System Check

```bash
./tt_smi
```

**See:**
- Which devices are present
- Current temperature and power
- Which processes are using devices

### 2. Monitoring During Development

```bash
# Terminal 1: Start server
./allocation_server_poc

# Terminal 2: Run your workload
python my_model.py

# Terminal 3: Monitor in real-time
./tt_smi -w -r 500
```

**See:**
- Real-time memory usage
- Memory leaks (if any)
- Peak memory consumption

### 3. Multi-Process Debugging

When running multiple processes:

```bash
# Terminal 1: Server
./allocation_server_poc

# Terminal 2: Process 1
python process1.py

# Terminal 3: Process 2
python process2.py

# Terminal 4: Monitor all
./tt_smi -w
```

**See:**
- Which process is using which device
- Per-device memory usage (aggregate)
- All active PIDs

### 4. CI/CD Integration

```bash
# In your test script
./allocation_server_poc &
SERVER_PID=$!

# Run tests
python run_tests.py

# Check if any processes still hold devices
./tt_smi > device_usage.txt

# Cleanup
kill $SERVER_PID
```

## Limitations

### 1. No Per-Process Memory (Without Instrumentation)

Unlike `nvidia-smi`, `tt-smi` cannot show per-process memory usage unless processes report to the allocation server.

**Why:**
- NVIDIA: Kernel driver sees all allocations
- Tenstorrent: Allocations happen in user-space, each process isolated

**Solution:**
- Use allocation server + instrument your code
- Or: Future kernel-level tracking (requires TT-KMD changes)

### 2. Temperature/Power May Not Be Available

`tt-smi` reads from `/sys/class/tenstorrent/tenstorrent!N/`:
- May require elevated permissions
- Not all hardware exposes these sensors
- Some remote devices don't report telemetry

### 3. Process-Device Mapping Not Precise

Currently, `tt-smi` associates all processes with device 0 by default.

**Reason:** Can't reliably determine which device a process opened just from `/proc/fd`.

**Future enhancement:** Processes could report their device ID to the server.

## Advanced Usage

### Custom Refresh Rate

```bash
# Very fast (100ms)
./tt_smi -w -r 100

# Slow (5 seconds)
./tt_smi -w -r 5000
```

### Script Integration

```python
import subprocess
import json

# Get device info
output = subprocess.check_output(['./tt_smi']).decode('utf-8')
print(output)

# Parse for automation
# (Future: add --json flag for machine-readable output)
```

### Multiple Terminals

```bash
# Terminal 1: Device 0 only
./allocation_monitor_client -d 0 -r 500

# Terminal 2: All devices
./tt_smi -w

# Terminal 3: Run workload
python my_script.py
```

## Future Enhancements

### 1. Kernel-Level Tracking

**Goal:** Make `tt-smi` work exactly like `nvidia-smi` without requiring a separate server.

**Implementation:**
- Extend TT-KMD to track allocations
- Expose via `/sys/class/tenstorrent/tenstorrent!N/processes`
- `tt-smi` reads directly from sysfs

**Benefits:**
- No allocation server needed
- No instrumentation needed
- Works for all processes automatically

### 2. Auto-Instrumentation

**Goal:** Make TT-Metal automatically connect to allocation server when device is created.

**Implementation:**
```cpp
// In Device constructor
Device::Device(...) {
    // ... existing code ...
    AllocationClient::auto_connect();
}
```

**Benefits:**
- No manual instrumentation
- All TT-Metal applications tracked automatically
- Server sees all processes

### 3. GPU Utilization Metrics

**Goal:** Show compute utilization, not just memory.

**Data sources:**
- NOC (Network on Chip) activity counters
- Kernel execution time
- PCIe bandwidth usage

**Display:**
```
│ GPU  Compute  Memory   PCIe      Processes │
│  0   85%      75%      2.5GB/s   3         │
```

### 4. JSON Output Mode

```bash
./tt_smi --json
```

Output:
```json
{
  "devices": [
    {
      "id": 0,
      "name": "Wormhole_B0",
      "temperature": 65.0,
      "power": 150.0,
      "memory": {
        "total": 12884901888,
        "used": 2576980377,
        "free": 10307921511
      },
      "processes": [
        {"pid": 12345, "name": "python3", "memory": 1288490189}
      ]
    }
  ]
}
```

## Troubleshooting

### "No processes using Tenstorrent devices"

**Causes:**
1. No processes have devices open
2. Devices are in use but not `/dev/tenstorrent/*` files

**Check:**
```bash
# List device files
ls -la /dev/tenstorrent/

# Check if devices exist
ls /sys/class/tenstorrent/
```

### "Allocation server not running"

**Solution:**
```bash
# Terminal 1: Start server
./allocation_server_poc

# Terminal 2: Run tt-smi
./tt_smi
```

### "Temperature/Power shows N/A"

**Causes:**
1. No sysfs telemetry available
2. Permission denied reading sysfs

**Check:**
```bash
# Try reading manually
cat /sys/class/tenstorrent/tenstorrent\!0/asic_temp
cat /sys/class/tenstorrent/tenstorrent\!0/power

# If permission denied, run with sudo
sudo ./tt_smi
```

### "Process shows but no memory usage"

**Cause:** Process is not instrumented to report to allocation server.

**Solution:** The process needs to send `ALLOC`/`FREE` messages to the server, or use the automatic allocation client.

## Command Reference

```bash
./tt_smi              # Show current state once
./tt_smi -w           # Watch mode (continuous refresh)
./tt_smi -w -r 1000   # Watch mode with 1000ms refresh
./tt_smi -h           # Show help
```

## See Also

- `allocation_server_poc` - Central allocation tracking daemon
- `allocation_monitor_client` - Detailed memory monitor with bar charts
- `NVIDIA_SMI_ARCHITECTURE.md` - Detailed comparison with nvidia-smi
- `LIMITATIONS.md` - Why cross-process tracking is challenging
