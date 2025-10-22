# L1 Memory Tracking: Complete Explanation

## Why You're Seeing KBs Instead of MBs

### TL;DR:
The **kernel/firmware tracking we just added** tracks small per-core writes (KBs). The **large buffer allocations (MBs)** are tracked separately via existing hooks in `graph_tracking.cpp`. You should see **both** types of tracking.

---

## L1 Memory Architecture

### Total L1 Memory (~171MB for Wormhole):
- **64-110 worker cores** × **~1.5MB per core** = ~171MB total
- Each core has its own separate L1 SRAM

### L1 Memory Layout Per Core (~1.5MB):

```
┌─────────────────────────────────────┐
│  Firmware (~30KB)                   │ ← Tracked by llrt.cpp (NEW!)
├─────────────────────────────────────┤
│  Kernel Binaries (~20-50KB)         │ ← Tracked by llrt.cpp (NEW!)
├─────────────────────────────────────┤
│  Launch Messages & Args (~1KB)      │ ← Tracked by llrt.cpp (NEW!)
├─────────────────────────────────────┤
│  Circular Buffers (varies, KB-MB)   │ ← Tracked by graph_tracking.cpp
├─────────────────────────────────────┤
│  User Buffers (BULK, MB range)      │ ← Tracked by graph_tracking.cpp
│  - Tensor data                      │
│  - Activations                      │
│  - Intermediate results             │
└─────────────────────────────────────┘
```

---

## What Each Tracking Hook Captures

### 1. **Existing Tracking (graph_tracking.cpp)** ✅
**Location**: `tt_metal/graph/graph_tracking.cpp:148-150`

```cpp
AllocationClient::report_allocation(
    buffer->device()->id(),
    buffer->size(),                           // ← LARGE sizes (MBs)
    static_cast<uint8_t>(buffer->buffer_type()),
    buffer->address()                         // ← Buffer address as ID
);
```

**Tracks**:
- User-allocated L1 buffers (tensor data, activations) - **MBs per buffer**
- Circular buffers - **KB-MB per CB**
- DRAM buffers
- Trace buffers

**Buffer IDs**: Use the actual buffer address from the allocator

---

### 2. **New Kernel/Firmware Tracking (llrt.cpp)** ✨ NEW
**Location**: `tt_metal/llrt/llrt.cpp:68-95`

```cpp
// Track per-core kernel/firmware writes
uint64_t buffer_id = (core.x << 48) | (core.y << 32) | (address & 0xFFFFFFFF);
AllocationClient::report_allocation(device_id, size, 1 /* L1 */, buffer_id);
```

**Tracks**:
- Kernel binaries written to each core - **~20-50KB per core**
- Firmware loaded to each core - **~30KB per core**
- Launch messages - **~1KB per core**

**Buffer IDs**: Unique per-core using `(core_x, core_y, address)` encoding

---

## Why Per-Core Tracking Uses Unique Buffer IDs

### The Problem:
- Multiple cores can have the **same L1 address** (e.g., 0x10000)
- Core (1,2) at address 0x10000 is **different physical memory** than Core (3,4) at 0x10000
- Need unique IDs to track them separately

### The Solution:
```
buffer_id = (core.x << 48) | (core.y << 32) | (address & 0xFFFFFFFF)

Example:
  Core (5, 3) at address 0x20000:
  buffer_id = 0x0005000300020000
              ^^^^core_x
                  ^^^^core_y
                      ^^^^^^^^address
```

This ensures each core's memory is tracked independently.

---

## Expected Allocation Server Output

### When Running a Workload:

```
✓ [PID 12345] Allocated 16777216 bytes of L1 on device 0 (buffer_id=0x1a0000)
  ↑ Large buffer allocation (~16MB) - from graph_tracking.cpp

✓ [PID 12345] Allocated 2097152 bytes of L1 on device 0 (buffer_id=0x1b0000)
  ↑ Another large buffer (~2MB) - from graph_tracking.cpp

✓ [PID 12345] Allocated 65536 bytes of L1 on device 0 (buffer_id=0x2c0000)
  ↑ Circular buffer (64KB) - from graph_tracking.cpp

✓ [PID 12345] Allocated 24576 bytes of L1 on device 0 (buffer_id=0x50003000a0000)
  ↑ Kernel binary on core (5,3) (~24KB) - from llrt.cpp (NEW!)

✓ [PID 12345] Allocated 32768 bytes of L1 on device 0 (buffer_id=0x5000300010000)
  ↑ Firmware on core (5,3) (~32KB) - from llrt.cpp (NEW!)

... (repeats for each active core)
```

### Memory Dump Summary:

```
╔══════════════════════════════════════════════════════════════╗
║           REMAINING ALLOCATED BUFFERS                       ║
╚══════════════════════════════════════════════════════════════╝

Device 0:
  L1: 143 buffers, 127.45 MB total
    ↑ This should be in the MB range for any real workload

  Breakdown:
  - ~10-20 large buffers (1-50MB each) = most of the memory
  - ~50-100 small buffers (KB each) = kernel/firmware per core
```

---

## Debugging: Why Am I Only Seeing KBs?

### Possible Causes:

#### 1. **Workload Doesn't Allocate Much L1**
Some tests use mostly DRAM. Check buffer type in server output.

**Fix**: Run a workload that uses L1:
```bash
pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_matmul.py
```

#### 2. **Looking at Individual Allocations, Not Total**
The server shows each allocation as it happens. Total is in the dump.

**Fix**: Send `SIGUSR1` to see total:
```bash
kill -USR1 <server_pid>
```

#### 3. **Buffer Tracking Not Enabled**
The existing buffer tracking in `graph_tracking.cpp` requires `TT_ALLOC_TRACKING_ENABLED=1`.

**Fix**: Ensure env var is set:
```bash
export TT_ALLOC_TRACKING_ENABLED=1
```

#### 4. **Only Seeing New Tracking**
If you only see buffer_ids like `0x50003000a0000` (with high bits set), you're only seeing the new per-core tracking.

**Fix**: Check that `AllocationClient::is_enabled()` returns true in `graph_tracking.cpp`.

---

## Verification Commands

### 1. Build with tracking:
```bash
cd /home/ttuser/aperezvicente/tt-metal
source ./env_vars_setup.sh
./build_metal_with_flags.sh
```

### 2. Start server:
```bash
export TT_ALLOC_TRACKING_ENABLED=1
./build/install/bin/allocation_server_poc
```

### 3. Run workload (in another terminal):
```bash
export TT_ALLOC_TRACKING_ENABLED=1
pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_matmul.py::test_matmul_1d -s
```

### 4. Check server output:
- Should see allocations as they happen
- Should see mix of large (MB) and small (KB) allocations

### 5. Get summary:
```bash
# In another terminal:
pkill -USR1 allocation_server_poc
```

---

## Summary

| What | Size | Where Tracked | Buffer ID Format |
|------|------|---------------|------------------|
| User Buffers | **MBs** | `graph_tracking.cpp` | `buffer->address()` |
| Circular Buffers | KB-MB | `graph_tracking.cpp` | CB address |
| Kernel Binaries | 20-50KB/core | `llrt.cpp` ✨ NEW | `(core_x << 48) \| ...` |
| Firmware | ~30KB/core | `llrt.cpp` ✨ NEW | `(core_x << 48) \| ...` |
| Launch Messages | ~1KB/core | `llrt.cpp` ✨ NEW | `(core_x << 48) \| ...` |

**Expected Total**: 100-200MB for typical workloads with many active cores

The KB-sized allocations are correct—they're **per-core** code/firmware. The MB allocations are the data buffers and should appear separately!
