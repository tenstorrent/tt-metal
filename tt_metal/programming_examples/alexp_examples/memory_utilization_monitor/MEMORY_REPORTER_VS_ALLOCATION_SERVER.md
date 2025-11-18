# MemoryReporter vs Allocation Server: Why We Built Both

## TL;DR

**MemoryReporter** = **Per-device, in-process** memory tracking (like `device.print_memory()`)
**Allocation Server** = **Cross-process, real-time** memory tracking (like `nvidia-smi`)

They solve **different problems** and are **complementary**, not competing!

---

## What MemoryReporter Provides

### API Overview
```cpp
#include <tt-metalium/memory_reporter.hpp>

// Get memory statistics for ONE device in the CURRENT process
auto stats = device->allocator()->get_statistics(BufferType::L1);

// Or use MemoryReporter wrapper:
MemoryView view = GetMemoryView(device, BufferType::L1);

// View contains:
view.num_banks                                  // Number of L1 banks
view.total_bytes_per_bank                       // Total allocatable per bank
view.total_bytes_allocated_per_bank             // Currently allocated per bank
view.total_bytes_free_per_bank                  // Free per bank
view.largest_contiguous_bytes_free_per_bank     // Largest free block
view.block_table                                // Detailed block info
```

### What It Tracks

**✅ Tracks (from Allocator):**
- Explicit buffer allocations (DRAM, L1, L1_SMALL, TRACE)
- Per-bank statistics
- Fragmentation info (largest free block)
- Memory block tables (address, size, allocated/free)

**❌ Does NOT Track:**
- Circular buffers (allocated at kernel setup)
- Kernel code (loaded at compile time)
- Firmware/runtime overhead
- **Memory from OTHER processes!**

### Key Limitations

#### 1. **In-Process Only** 🚨 (The Big One!)

```cpp
// Process A:
Device* device_a = CreateDevice(0);
auto buffer_a = CreateBuffer(device_a, 1024*1024, BufferType::L1);

// Process B (SEPARATE PROCESS):
Device* device_b = CreateDevice(0);  // Same device!
auto stats = device_b->allocator()->get_statistics(BufferType::L1);
// ❌ Does NOT see buffer_a allocated by Process A!
```

**Why?** Each process has its own `Device` object with its own `Allocator` instance. They don't share state!

#### 2. **Only Tracks Allocator-Managed Memory**

```cpp
// Tracked:
auto buffer = CreateBuffer(device, size, BufferType::L1);  // ✅ Visible to MemoryReporter

// NOT Tracked:
CircularBufferConfig cb_config(1024*1024, ...);  // ❌ NOT visible to MemoryReporter
CreateKernel(..., cb_config);                     // CBs are allocated outside the allocator
```

#### 3. **Requires Device Handle**

You need a `Device*` pointer to query memory:
```cpp
Device* device = CreateDevice(0);  // Must create/own the device
auto view = GetMemoryView(device, BufferType::L1);
```

This means:
- ❌ Can't query devices you don't own
- ❌ Can't query devices used by other processes
- ❌ Can't do system-wide monitoring

#### 4. **Generates Files, Not Real-Time API**

Primary interface is CSV file generation:
```cpp
DumpDeviceMemoryState(device);  // Writes to .reports/tt_metal/*.csv
```

Not ideal for real-time monitoring tools!

---

## What Allocation Server Provides

### Architecture Overview

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Process A   │     │ Process B   │     │ Process C   │
│  Device(0)  │────▶│  Device(0)  │────▶│  Device(1)  │
│  1GB DRAM   │     │  512MB L1   │     │  2GB DRAM   │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                    │
       │    Unix Domain Socket (IPC)            │
       └───────────────────┼────────────────────┘
                           ▼
                ┌──────────────────────┐
                │ Allocation Server    │
                │  - Aggregates all    │
                │  - Cross-process     │
                │  - Real-time queries │
                └──────────────────────┘
                           ▲
                           │
                  ┌────────┴────────┐
                  │   tt_smi_umd    │
                  │  (monitoring)   │
                  └─────────────────┘
```

### What It Tracks

**✅ Tracks:**
- All allocations from **ALL processes** (cross-process)
- Real-time updates (allocation + deallocation events)
- Per-device aggregation across processes
- Process ownership (which PID owns which buffer)
- Dead process cleanup (orphaned buffers)

**❌ Does NOT Track (yet):**
- Circular buffers
- Kernel code
- Firmware overhead
- Per-bank statistics (only device-wide totals)

### Key Advantages

#### 1. **Cross-Process Tracking** ✨ (The Killer Feature!)

```cpp
// Process A:
Device* dev_a = CreateDevice(0);
CreateBuffer(dev_a, 1GB, BufferType::DRAM);
// Server sees: Device 0, DRAM: 1GB (PID 1234)

// Process B (separate process, same device):
Device* dev_b = CreateDevice(0);
CreateBuffer(dev_b, 512MB, BufferType::L1);
// Server sees: Device 0, DRAM: 1GB (PID 1234), L1: 512MB (PID 5678)

// tt_smi_umd (monitoring tool):
query_server(device_id=0);
// Gets: DRAM: 1GB, L1: 512MB (TOTAL across both processes!)
```

#### 2. **No Device Handle Required**

```bash
# Monitor all devices without creating Device objects:
./tt_smi_umd

# No need to "own" the device!
# Server already knows about all devices from processes that ARE using them.
```

#### 3. **Real-Time IPC**

```cpp
// Instant notification on allocation:
CreateBuffer(...);
// → Sends message to server immediately
// → tt_smi_umd sees update on next refresh (500ms)
```

#### 4. **Process Lifecycle Management**

```cpp
// Process crashes without cleanup:
// (Process PID 1234 dies unexpectedly)

// Server detects (via background thread):
// "PID 1234 is dead, cleaning up 5 buffers, 2GB memory"
```

### How It Works

#### Instrumentation in `buffer.cpp`:

```cpp
// In Buffer::Buffer() constructor:
void Buffer::Buffer(...) {
    // ... normal allocation ...

    // NEW: Notify allocation server
    if (server_available()) {
        send_alloc_message(device_id, buffer_id, size, buffer_type);
    }
}

// In Buffer::~Buffer() destructor:
Buffer::~Buffer() {
    // NEW: Notify allocation server
    if (server_available()) {
        send_free_message(device_id, buffer_id);
    }

    // ... normal deallocation ...
}
```

#### Server Protocol:

```cpp
struct AllocMessage {
    enum Type : uint8_t {
        ALLOC = 1,              // Buffer allocated
        FREE = 2,               // Buffer freed
        QUERY = 3,              // Query current stats
        RESPONSE = 4,           // Response to query
        DEVICE_INFO_QUERY = 6,  // Query device capabilities
        DEVICE_INFO_RESPONSE = 7
    };

    int32_t device_id;
    uint64_t size;
    uint8_t buffer_type;  // DRAM, L1, L1_SMALL, TRACE
    int32_t process_id;
    uint64_t buffer_id;

    // Response fields:
    uint64_t dram_allocated;
    uint64_t l1_allocated;
    uint64_t l1_small_allocated;
    uint64_t trace_allocated;
};
```

---

## Why We Built Allocation Server (Instead of Just Using MemoryReporter)

### Problem: Multi-Process GPU Monitoring

**Goal:** Build `nvidia-smi` for Tenstorrent chips

```bash
$ nvidia-smi
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.54       Driver Version: 535.54       CUDA Version: 12.2   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA H100        Off  | 00000000:01:00.0 Off |                    0 |
| N/A   45C    P0    60W / 700W |  12345MiB / 81920MiB |      15%      Default|
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA H100        Off  | 00000000:02:00.0 Off |                    0 |
| N/A   42C    P0    50W / 700W |   5678MiB / 81920MiB |       8%      Default|
+-------------------------------+----------------------+----------------------+
```

**Notice:** Shows memory usage **across ALL processes!**

### Why MemoryReporter Can't Do This:

#### Attempt 1: Create Device in tt_smi
```cpp
// tt_smi.cpp
int main() {
    Device* device = CreateDevice(0);  // ❌ Conflicts with user's process!
    auto stats = device->allocator()->get_statistics(BufferType::L1);
    // Shows ONLY allocations made by tt_smi itself (probably zero!)
}
```

**Problem:** Each process has its own allocator state. Creating a device in `tt_smi` doesn't let you see allocations from other processes.

#### Attempt 2: Share Device Pointer Across Processes
```cpp
// ❌ IMPOSSIBLE! Device* is in-process memory!
// Can't share pointers across process boundaries!
```

**Problem:** C++ objects like `Device` exist in process-local memory. You can't access them from another process.

#### Attempt 3: Use Kernel Driver (KMD)

The Linux kernel driver **does** track total device memory usage at the hardware level, but:

```cpp
// KMD tracks physical memory pages, NOT:
// - Which process owns which allocation
// - Buffer types (DRAM vs L1 vs L1_SMALL)
// - Allocation timestamps
// - Buffer IDs
```

**Problem:** KMD has low-level info but not application-level semantics.

### Solution: Allocation Server (IPC)

**Key Insight:** We need a **separate process** that receives allocation events from ALL processes via **inter-process communication (IPC)**.

```
┌────────────────────────────────────────────────────────────┐
│  Why MemoryReporter Can't Work for Multi-Process Tracking │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Process A                    Process B                   │
│  ┌────────────┐               ┌────────────┐             │
│  │ Device(0)  │               │ Device(0)  │             │
│  │ Allocator  │               │ Allocator  │             │
│  │  - 1GB     │               │  - 512MB   │             │
│  └────────────┘               └────────────┘             │
│       ▲                             ▲                     │
│       │                             │                     │
│       └─────────────────────────────┘                     │
│                    │                                       │
│           Can't access each other!                        │
│           (different memory spaces)                       │
│                                                            │
│  Solution: Allocation Server                              │
│  ┌────────────────────────────────────────────┐          │
│  │  Receives IPC messages from both:          │          │
│  │    Process A: "allocated 1GB DRAM"         │          │
│  │    Process B: "allocated 512MB L1"         │          │
│  │                                             │          │
│  │  Aggregates: Device 0 total = 1GB + 512MB  │          │
│  └────────────────────────────────────────────┘          │
└────────────────────────────────────────────────────────────┘
```

---

## When to Use Each

### Use MemoryReporter When:

✅ **In-process debugging**
```python
# In your Python model:
device = ttl.device.GetDefaultDevice()
ttl.device.DumpDeviceMemoryState(device)  # Debug YOUR allocations
```

✅ **Detailed per-bank analysis**
```cpp
auto view = GetMemoryView(device, BufferType::L1);
for (int bank = 0; bank < view.num_banks; bank++) {
    std::cout << "Bank " << bank << ": "
              << view.total_bytes_allocated_per_bank << " bytes\n";
}
```

✅ **Fragmentation analysis**
```cpp
auto stats = device->allocator()->get_statistics(BufferType::L1);
std::cout << "Largest free block: " << stats.largest_free_block_bytes << "\n";
// Useful for deciding if a new buffer can fit
```

✅ **CSV report generation**
```cpp
EnableMemoryReports();
// Automatically generates CSV files during program execution
```

### Use Allocation Server When:

✅ **System-wide monitoring** (like `nvidia-smi`)
```bash
./tt_smi_umd -w  # Watch mode, shows ALL processes
```

✅ **Multi-process tracking**
```bash
# Terminal 1:
python run_model_A.py  # Uses Device 0

# Terminal 2:
python run_model_B.py  # Uses Device 0

# Terminal 3:
./tt_smi_umd  # Shows BOTH models' memory usage!
```

✅ **Process identification**
```bash
$ ./tt_smi_umd
Device 0:
  PID 1234 (python): 2.5 GB DRAM, 100 MB L1
  PID 5678 (python): 1.8 GB DRAM, 50 MB L1
```

✅ **Real-time alerts**
```cpp
// Monitor and alert on high usage:
while (true) {
    auto stats = query_server(device_id);
    if (stats.dram_allocated > threshold) {
        send_alert("High DRAM usage!");
    }
}
```

✅ **Dead process cleanup**
```
Server automatically detects:
"PID 1234 died, cleaning up 5GB of orphaned buffers"
```

---

## The Combined Solution (Best of Both Worlds!)

### Integrate MemoryReporter into tt_smi_umd

We can **combine** both approaches:

```cpp
// In tt_smi_umd.cpp:

void display_full_memory_tracking(int device_id) {
    // 1. Query allocation server (cross-process, allocator-tracked)
    auto server_stats = query_allocation_server(device_id);
    uint64_t allocator_dram = server_stats.dram_allocated;
    uint64_t allocator_l1 = server_stats.l1_allocated;

    // 2. Query MemoryReporter (in-process, includes CBs if we own device)
    if (we_own_device) {
        auto view_l1 = GetMemoryView(device, BufferType::L1);
        uint64_t total_l1 = view_l1.total_bytes_allocated_per_bank * view_l1.num_banks;

        // 3. Infer non-allocator usage
        uint64_t cb_and_kernel = total_l1 - allocator_l1;

        std::cout << "L1 Memory Breakdown:\n";
        std::cout << "  Allocator-tracked: " << allocator_l1 / 1024 / 1024 << " MB (from server)\n";
        std::cout << "  CB + Kernels:      " << cb_and_kernel / 1024 / 1024 << " MB (inferred)\n";
        std::cout << "  Total:             " << total_l1 / 1024 / 1024 << " MB\n";
    } else {
        // Can't query MemoryReporter (don't own device), show server data only
        std::cout << "L1 Memory (allocator-tracked): " << allocator_l1 / 1024 / 1024 << " MB\n";
        std::cout << "  (CB + Kernel usage not available when device owned by other process)\n";
    }
}
```

### Why Both Are Needed:

| Feature | Allocation Server | MemoryReporter | Combined |
|---------|-------------------|----------------|----------|
| Cross-process | ✅ | ❌ | ✅ |
| Real-time | ✅ | ❌ | ✅ |
| Process PIDs | ✅ | ❌ | ✅ |
| Per-bank stats | ❌ | ✅ | ✅ |
| Fragmentation info | ❌ | ✅ | ✅ |
| CB + Kernel (inferred) | ❌ | ✅ | ✅ |
| No device ownership needed | ✅ | ❌ | ✅ |

---

## Why We Didn't Know About MemoryReporter

### 1. **Documentation**
- MemoryReporter is in `detail::` namespace → considered "internal" API
- Primary docs focus on CSV file generation, not programmatic queries
- Not advertised as a "monitoring API"

### 2. **Design Intent**
MemoryReporter was designed for:
- **Development/debugging** (CSV reports for analysis)
- **Per-program profiling** (track memory during program compilation)
- **Post-mortem analysis** (dump state to files)

NOT designed for:
- Real-time system monitoring
- Cross-process tracking
- Production dashboards

### 3. **Naming**
"Reporter" suggests **reporting/logging** (passive, files), not **querying** (active, API).

Compare:
- `nvidia-smi` → clearly a system monitoring tool
- `MemoryReporter` → sounds like a debug logger

---

## Going Forward: Integration Plan

### Phase 1: Add MemoryReporter to tt_smi_umd ✅ (Easy)

Add inferred CB + Kernel tracking:
```cpp
// Show allocator-tracked (from server) + total (from MemoryReporter)
uint64_t allocator_tracked = query_server();
uint64_t total = query_memory_reporter();
uint64_t cb_kernel = total - allocator_tracked;
```

### Phase 2: Full L1 Tracking (Hard)

Hook into Program/CB creation to track everything:
- Requires modifying TT-Metal core
- ~500-1000 lines of code
- See `FULL_L1_TRACKING_GUIDE.md`

### Phase 3: Unified API (Future)

Create a new public API that combines both:
```cpp
namespace tt::tt_metal {
    class SystemMemoryMonitor {
    public:
        // Cross-process stats (from allocation server)
        SystemMemoryStats get_system_stats(int device_id);

        // Per-process stats (from MemoryReporter)
        ProcessMemoryStats get_process_stats(Device* device);

        // Combined view
        DetailedMemoryStats get_detailed_stats(int device_id);
    };
}
```

---

## Summary

**Why we built the Allocation Server:**
- MemoryReporter is **per-process** and can't see allocations from other processes
- We needed **cross-process tracking** like `nvidia-smi`
- IPC (Unix sockets) was the only way to aggregate across processes

**Why MemoryReporter is still valuable:**
- Provides **per-bank** and **fragmentation** info that the server doesn't
- Can be used to **infer** CB + Kernel usage (total - allocator-tracked)
- Already exists and works today!

**Best solution:**
Use **BOTH**:
- **Allocation Server** → Cross-process, real-time, allocator-tracked memory
- **MemoryReporter** → In-process, detailed stats, infer non-allocator memory

This gives you the full picture! 🎯
