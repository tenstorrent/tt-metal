# Buffer Types in TT-Metal - Complete Explanation

## Overview

TT-Metal tracks several types of memory allocations separately. Here's what each one means:

---

## 1. **"Buffers"** (Generic Buffer Allocations)
### What are they?
- **Generic data buffers** created by user programs
- Can be in **DRAM** or **L1** depending on `BufferType`
- Created via `Buffer::create()` or `CreateBuffer()` API
- Used for: weights, activations, inputs/outputs, intermediate results

### Examples:
```cpp
// DRAM buffer for model weights
auto weight_buffer = Buffer::create(device, size, BufferType::DRAM);

// L1 buffer for intermediate data
auto scratch_buffer = Buffer::create(device, size, BufferType::L1);
```

### In `tt_smi_umd`:
- Shown as **"DRAM: X.XX GB used"** or **"L1: X.XX MB used"**
- Most flexible type - user controls where it goes
- Tracked via `report_allocation()` with `buffer_type` parameter

---

## 2. **CBs (Circular Buffers)**
### What are they?
- **Specialized L1 buffers** for streaming data between kernels
- Act as **producer-consumer queues** (FIFO)
- **Always in L1** (never DRAM)
- Used for: kernel-to-kernel data streaming, tile processing pipelines

### Why separate tracking?
- Have specific size/alignment requirements
- Allocated in **CB region** of L1 (separate from regular L1 buffers)
- Critical for understanding pipeline efficiency

### Examples:
```cpp
// CB for streaming data from DRAM reader to compute kernel
CircularBufferConfig cb_config = CircularBufferConfig(size)
    .set_data_format(DataFormat::Float16_b);
CreateCircularBuffer(program, core, cb_config);
```

### In `tt_smi_umd`:
- Shown as **"CBs: X.XX MB"** under L1 breakdown
- Helps identify circular buffer pressure
- Tracked via `report_cb_allocation()` with `CB_ALLOC` type

---

## 3. **Kernels (Compiled Kernel Code)**
### What are they?
- **Compiled binary code** for RISC-V processors
- Stored in **DRAM**, loaded to **L1 ring buffer** when needed
- Includes: BRISC, NCRISC, TRISC kernels (compute, data movement)
- Automatically managed by dispatch system

### Why separate tracking?
- Shows **code footprint** vs data footprint
- Kernels are **evicted/reloaded** from L1 ring buffer as needed
- Helps understand program complexity

### Examples:
```cpp
// Kernels are automatically tracked when you add them to a program
auto reader_kernel = CreateKernel(
    program, "tt_metal/kernels/reader.cpp",
    core, DataMovementConfig{...});

auto compute_kernel = CreateKernel(
    program, "tt_metal/kernels/compute.cpp",
    core, ComputeConfig{...});
```

### In `tt_smi_umd`:
- Shown as **"Kernels: X.XX MB"** under L1 breakdown
- Represents **maximum L1 footprint** when loaded (not always resident)
- Tracked via `report_kernel_load()` with `KERNEL_LOAD` type

---

## 4. **L1_SMALL**
### What are they?
- **Small allocations** from L1 allocator
- Usually for: semaphores, runtime args, small scratch buffers
- Uses a different allocation strategy (small block allocator)

### In `tt_smi_umd`:
- Shown as **"L1 Small: X.XX MB"**
- Usually very small (few KB)

---

## 5. **TRACE**
### What are they?
- **Trace buffers** for recording/replaying command sequences
- Used by Fast Dispatch for optimizing repeated operations
- Can be in DRAM or L1 depending on trace type

### In `tt_smi_umd`:
- Shown as **"Trace: X.XX MB"**
- Helps identify trace capture overhead

---

## Memory Hierarchy Summary

```
┌─────────────────────────────────────────────────────┐
│                    DRAM (31 GB)                     │
│  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │
│  │   Buffers   │  │   Kernels   │  │   Trace    │ │
│  │  (weights,  │  │  (binaries  │  │  (command  │ │
│  │   data, I/O)│  │   stored)   │  │   capture) │ │
│  └─────────────┘  └─────────────┘  └────────────┘ │
└─────────────────────────────────────────────────────┘
                        ↓ (loaded as needed)
┌─────────────────────────────────────────────────────┐
│                   L1 (306 MB)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │
│  │   Buffers   │  │     CBs     │  │  Kernels   │ │
│  │  (scratch,  │  │  (producer/ │  │ (ring buf, │ │
│  │intermediate)│  │   consumer) │  │  evictable)│ │
│  └─────────────┘  └─────────────┘  └────────────┘ │
│  ┌─────────────┐  ┌─────────────┐                 │
│  │  L1_SMALL   │  │    TRACE    │                 │
│  │ (semaphores,│  │  (capture)  │                 │
│  │    args)    │  │             │                 │
│  └─────────────┘  └─────────────┘                 │
└─────────────────────────────────────────────────────┘
```

---

## Why Track Separately?

### 1. **Performance Debugging**
- High CB usage → pipeline bottleneck
- High Kernel usage → too many programs loaded
- High Buffer usage → data movement issues

### 2. **Resource Planning**
- Know which programs use most L1
- Identify optimization opportunities
- Plan model partitioning

### 3. **Memory Leak Detection**
- Each type has different lifecycle
- Separate tracking helps find which type leaks
- Easier to trace back to source

---

## In tt_smi_umd Display

```
View 1 (Detailed):
┌────────────────────────────────────────────────────┐
│ Device 0: Blackhole                                │
│ DRAM: 1.23 GB / 31.00 GB [██░░░░░░░░░░░░░░] 3.97% │
│ L1:   45.67 MB / 306.00 MB [████░░░░░░░░░░] 14.92%│
│   Memory Breakdown:                                │
│   ├─ L1 Buffers:  20.00 MB [████░░░░░░] 43.8%     │
│   ├─ CBs:         22.47 MB [████░░░░░░] 49.2%     │
│   ├─ Kernels:      0.20 MB [░░░░░░░░░░]  0.4%     │
│   └─ Total L1:    42.67 MB                         │
└────────────────────────────────────────────────────┘
```

---

## Common Scenarios

### Scenario 1: Data-Parallel Training
```
DRAM Buffers: HIGH (model weights replicated)
L1 Buffers:   MEDIUM (activations)
CBs:          HIGH (streaming activations)
Kernels:      LOW (same kernels reused)
```

### Scenario 2: Large Model Inference
```
DRAM Buffers: VERY HIGH (model weights)
L1 Buffers:   LOW (limited intermediate storage)
CBs:          MEDIUM (tile streaming)
Kernels:      MEDIUM (many kernel types)
```

### Scenario 3: Device Initialization
```
DRAM Buffers: LOW
L1 Buffers:   LOW
CBs:          NONE
Kernels:      ~0.2 MB (dispatch system kernels - persistent!)
```

---

## Persistent Kernel Allocations

Some kernels are **never deallocated** during device lifetime:
1. **Fast Dispatch system kernels** (prefetch, dispatch, completion queue)
2. **Command queue infrastructure**
3. **Device management kernels**

These show up as **~0.2 MB per device** of kernel memory that stays allocated until device close. This is **NORMAL** and **expected**!
