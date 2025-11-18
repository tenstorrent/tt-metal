# How to Make a `tt-smi` Like `nvidia-smi`

## How `nvidia-smi` Works

### NVIDIA Architecture
```
┌─────────────────────────────────────────────────────┐
│  Process 1 (python)          Process 2 (./app)     │
│       ↓                             ↓               │
│  libcuda.so                    libcuda.so          │
│       ↓                             ↓               │
└───────┼─────────────────────────────┼───────────────┘
        │                             │
        └─────────────┬───────────────┘
                      ↓
        ┌─────────────────────────────┐
        │  NVIDIA Kernel Driver       │
        │  • Tracks ALL allocations   │
        │  • Knows all PIDs          │
        │  • Maintains global state   │
        └─────────────┬───────────────┘
                      ↓
        ┌─────────────────────────────┐
        │  /proc/driver/nvidia/       │
        │  • gpus/0/information       │
        │  • gpus/0/processes         │  ← nvidia-smi reads this
        └─────────────────────────────┘
```

**Key Point:** The kernel driver intercepts EVERY `cudaMalloc()` call and records:
- Process ID (PID)
- Allocated size
- GPU memory address
- Process name

## Current Tenstorrent Architecture

```
┌─────────────────────────────────────────────────────┐
│  Process 1 (python)          Process 2 (./app)     │
│       ↓                             ↓               │
│  TT-Metal                      TT-Metal            │
│  Allocator A                   Allocator B         │
│  (isolated)                    (isolated)          │
└───────┼─────────────────────────────┼───────────────┘
        │                             │
        └─────────────┬───────────────┘
                      ↓
        ┌─────────────────────────────┐
        │  TT-KMD (Kernel Driver)     │
        │  • NO allocation tracking   │  ← Missing!
        │  • Only telemetry           │
        └─────────────┬───────────────┘
                      ↓
        ┌─────────────────────────────┐
        │  /sys/class/tenstorrent/    │
        │  • asic_temp                │
        │  • power                     │
        │  ❌ NO process info         │  ← Can't read this!
        └─────────────────────────────┘
```

**Problem:** The kernel driver doesn't track allocations, so there's no centralized place to query all processes.

---

## Solution: Three Approaches

### Approach 1: Process Discovery via `/proc` (Implemented Below)

Since the kernel doesn't track allocations, we can discover processes that have Tenstorrent devices open:

```bash
# Find all processes with /dev/tenstorrent/* open
lsof /dev/tenstorrent/0 | awk 'NR>1 {print $2}' | sort -u
```

or

```bash
# Alternative using fuser
fuser /dev/tenstorrent/0 2>/dev/null
```

or

```bash
# Walk /proc manually
for pid in /proc/[0-9]*; do
    if ls -l $pid/fd 2>/dev/null | grep -q tenstorrent; then
        echo $pid | cut -d/ -f3
    fi
done
```

**Strategy:**
1. Allocation server discovers which PIDs have devices open
2. For each PID, check if it's connected to the server
3. Query stats from connected processes
4. Display a table like `nvidia-smi`

### Approach 2: Auto-Registration (Easier, Recommended)

Make the **TT-Metal allocator automatically connect** to the server when a device is created:

```cpp
// In tt_metal/impl/device/device.cpp
Device::Device(...) {
    // ... existing code ...

    // Auto-register with allocation server if available
    AllocationClient::try_connect_to_server(device_id_);
}
```

This way:
- No manual instrumentation needed
- Every process automatically reports allocations
- Server knows about all active processes
- Can implement process death detection

### Approach 3: Kernel-Level Tracking (Future)

**Long-term solution:** Extend TT-KMD to track allocations like NVIDIA does.

Add to kernel driver:
```c
// In TT-KMD
struct tt_process_alloc_info {
    pid_t pid;
    char comm[16];  // process name
    uint64_t dram_allocated;
    uint64_t l1_allocated;
};

// Expose via:
// /sys/class/tenstorrent/tenstorrent!0/processes
```

**Pros:**
- System-wide view without user-space cooperation
- Works even if process doesn't use TT-Metal
- Can track kernel-level DMA allocations

**Cons:**
- Requires kernel driver changes
- Needs upstreaming to TT-KMD
- More complex implementation

---

## Implementation: `tt-smi` with Process Discovery

I'll create an enhanced version of your monitor that:
1. Discovers all processes with devices open
2. Queries the allocation server for each
3. Displays in nvidia-smi style format

### Enhanced Monitor Features

**Display:**
```
┌─────────────────────────────────────────────────────────────────────┐
│ tt-smi v1.0                          Mon Nov  3 12:34:56 2025       │
├─────────────────────────────────────────────────────────────────────┤
│ GPU  Name           Temp   Power   Mem-Usage    Utilization         │
├─────────────────────────────────────────────────────────────────────┤
│  0   Wormhole_B0    65°C   150W    2.4GB/12GB   [████████░░] 75%   │
└─────────────────────────────────────────────────────────────────────┘

Processes:
┌─────────────────────────────────────────────────────────────────────┐
│ PID    Name         Device  DRAM     L1       Type                  │
├─────────────────────────────────────────────────────────────────────┤
│ 12345  python3      0       1.2GB    45MB     TTNN Workload         │
│ 12346  ./test_app   0       800MB    30MB     Direct API            │
│ 12347  python3      1       400MB    15MB     TTNN Workload         │
└─────────────────────────────────────────────────────────────────────┘
```

Let me implement this now.
