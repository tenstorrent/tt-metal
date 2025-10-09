# Memory Utilization Architecture on Tenstorrent Hardware

This document explains where and how memory utilization information is tracked and accessed across the different layers of the Tenstorrent software stack.

## Overview: The Three-Layer Architecture

Memory utilization for Tenstorrent hardware is managed and accessible at three distinct layers:

1. **TT-Metal** (User-space application layer) - High-level memory allocation tracking
2. **TT-UMD** (User-Mode Driver) - Device communication and low-level memory management
3. **TT-KMD** (Kernel-Mode Driver) - Hardware telemetry and PCIe performance counters

---

## 1. TT-Metal Layer (Highest Level)

**Location**: `/home/tt-metal-apv/tt_metal/`

### Where Memory Utilization is Tracked

The TT-Metal layer provides the **most detailed application-level memory tracking** through its allocator system.

#### Key Components:

**A. Memory Allocator** (`tt_metal/impl/allocator/`)
- **File**: `allocator.cpp`
- **Function**: `Allocator::get_statistics(BufferType buffer_type)`
- **Returns**: `Statistics` struct with:
  - `total_allocatable_size_bytes`
  - `total_allocated_bytes`
  - `total_free_bytes`
  - `largest_free_block_bytes`
  - `largest_free_block_addrs`

**B. Buffer Types Tracked**:
```cpp
enum class BufferType {
    DRAM,        // Device DRAM memory
    L1,          // Level 1 on-chip memory
    L1_SMALL,    // Small L1 region
    TRACE,       // Trace buffer region
};
```

**C. Memory Reporter** (`tt_metal/detail/reports/memory_reporter.cpp`)
- **Function**: `GetMemoryView(const IDevice* device, const BufferType& buffer_type)`
- **API**: `device->allocator()->get_statistics(buffer_type)`
- **Returns**: `MemoryView` struct with per-bank and aggregate statistics

#### Access Methods:

**C++ API:**
```cpp
#include <tt-metalium/memory_reporter.hpp>

// Get detailed memory view
auto memory_view = detail::GetMemoryView(device, BufferType::L1);

// Access statistics
size_t total_allocated = memory_view.total_bytes_allocated_per_bank * memory_view.num_banks;
size_t total_free = memory_view.total_bytes_free_per_bank * memory_view.num_banks;
```

**Python API:**
```python
import ttnn

# Dump memory state to CSV files
ttnn.device.dump_device_memory_state(device, prefix="my_snapshot")

# Enable automatic memory reports during program compilation
ttnn.device.EnableMemoryReports()
```

#### Implementation Details:

**Allocation Algorithm** (`tt_metal/impl/allocator/algorithms/free_list_opt.cpp`)
- Uses free list data structure to track allocated/free blocks
- Real-time calculation of statistics by iterating through blocks:
```cpp
Statistics FreeListOpt::get_statistics() const {
    size_t total_allocated_bytes = 0;
    size_t total_free_bytes = 0;

    for (size_t i = 0; i < block_address_.size(); i++) {
        if (block_is_allocated_[i]) {
            total_allocated_bytes += block_size_[i];
        } else {
            total_free_bytes += block_size_[i];
        }
    }
    // ... returns Statistics struct
}
```

**Bank Manager** (`tt_metal/impl/allocator/bank_manager.cpp`)
- Manages memory across multiple banks (L1 cores, DRAM channels)
- Aggregates statistics across all banks
- Handles bank-specific allocation strategies

#### Output Files Generated:

When you call `dump_device_memory_state()` or enable memory reports, TT-Metal generates:

1. **`l1_usage_summary.csv`**
   - Per-program L1 memory usage
   - Minimum largest free L1 block
   - Maximum buffer size that can be allocated

2. **`memory_usage_summary.csv`**
   - Total allocatable, allocated, free bytes
   - Per buffer type (DRAM, L1, L1_SMALL, TRACE)
   - Largest free block size

3. **`detailed_memory_usage.csv`**
   - Every memory block with address, size, allocation status
   - Complete memory map visualization

---

## 2. TT-UMD Layer (User-Mode Driver)

**Location**: `/home/tt-umd/`

### Where Memory is Managed

The UMD layer focuses on **low-level device memory management** but doesn't track application-level utilization statistics.

#### Key Components:

**A. Device Memory Access**
- **File**: `device/api/umd/device/chip/chip.hpp`
- **Functions**:
  - `write_to_device()` - Write to L1 memory
  - `read_from_device()` - Read from L1 memory
  - `write_to_sysmem()` - Write to system memory (DRAM)
  - `read_from_sysmem()` - Read from system memory

**B. Memory Mapping**
- **TLB Manager**: Maps device memory to host address space
- **Sysmem Manager**: Manages system memory allocations
- **DMA Operations**: Direct memory access for large transfers

#### What UMD Provides:

- **Raw memory access** to device L1 and DRAM
- **Memory-mapped I/O** for register access
- **DMA transfer capabilities** for bulk data movement
- **PCIe BAR management** for memory-mapped regions

#### NOT Provided by UMD:

- ❌ Memory utilization statistics
- ❌ Allocation tracking
- ❌ Memory fragmentation analysis

The UMD is primarily a **transport layer** - it moves data but doesn't track how memory is being used at the application level.

---

## 3. TT-KMD Layer (Kernel-Mode Driver)

**Location**: `/home/tt-kmd/`

### Where Hardware Telemetry is Accessible

The KMD layer provides **hardware-level telemetry** through sysfs, but this is primarily for physical device monitoring, not memory allocation tracking.

#### Key Components:

**A. Telemetry System** (`telemetry.h`)
- **Exposed via**: `/sys/class/tenstorrent/tenstorrent!<N>/`
- **Telemetry Tags**:
  - `TELEMETRY_VCORE` - Core voltage
  - `TELEMETRY_POWER` - Power consumption
  - `TELEMETRY_CURRENT` - Current draw
  - `TELEMETRY_ASIC_TEMP` - Temperature
  - `TELEMETRY_AICLK` - AI clock frequency
  - `TELEMETRY_FAN_RPM` - Fan speed

**B. PCIe Performance Counters** (`sysfs-attributes.md`)
- **Location**: `/sys/class/tenstorrent/tenstorrent!<N>/pcie_perf_counters/`
- **Counters Available**:
  - PCIe transaction counts
  - NOC (Network on Chip) activity
  - Data movement statistics
  - PCIe bandwidth utilization

#### What KMD Provides:

**System-Level Monitoring:**
```bash
# Read device temperature
cat /sys/class/tenstorrent/tenstorrent!0/asic_temp

# Read power consumption
cat /sys/class/tenstorrent/tenstorrent!0/power

# Read PCIe performance counters
cat /sys/class/tenstorrent/tenstorrent!0/pcie_perf_counters/*
```

#### NOT Provided by KMD:

- ❌ Application memory allocation tracking
- ❌ L1/DRAM buffer utilization
- ❌ Memory fragmentation information
- ❌ Per-program memory usage

The KMD focuses on **physical hardware health and PCIe performance**, not logical memory allocation.

---

## Summary: Where to Get Memory Utilization

| What You Want | Where to Get It | Layer | API |
|---------------|----------------|-------|-----|
| **L1 memory allocated/free** | TT-Metal Allocator | TT-Metal | `device->allocator()->get_statistics(BufferType::L1)` |
| **DRAM memory allocated/free** | TT-Metal Allocator | TT-Metal | `device->allocator()->get_statistics(BufferType::DRAM)` |
| **Per-buffer memory usage** | Memory Reporter | TT-Metal | `detail::GetMemoryView(device, buffer_type)` |
| **Memory block details** | Memory Reporter | TT-Metal | `dump_device_memory_state(device)` |
| **Total device memory size** | Device Info | TT-Metal | `device->l1_size_per_core()`, `device->dram_size_per_channel()` |
| **PCIe bandwidth usage** | sysfs | TT-KMD | `/sys/class/tenstorrent/tenstorrent!N/pcie_perf_counters/` |
| **Hardware telemetry** | sysfs | TT-KMD | `/sys/class/tenstorrent/tenstorrent!N/` |
| **Raw memory R/W** | Chip API | TT-UMD | `chip->read_from_device()`, `chip->write_to_device()` |

---

## Recommended Approach for Memory Monitoring

### For Application-Level Memory Tracking:

**Use TT-Metal APIs** - This is the ONLY layer that tracks logical memory allocation:

```cpp
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/memory_reporter.hpp>

// Get memory statistics
auto stats = device->allocator()->get_statistics(BufferType::L1);

std::cout << "L1 Allocated: " << stats.total_allocated_bytes << " bytes\n";
std::cout << "L1 Free: " << stats.total_free_bytes << " bytes\n";
std::cout << "Largest Free Block: " << stats.largest_free_block_bytes << " bytes\n";
```

### For Hardware Health Monitoring:

**Use TT-KMD sysfs** - For physical device monitoring:

```bash
#!/bin/bash
# Monitor device health
watch -n 1 "cat /sys/class/tenstorrent/tenstorrent!0/asic_temp && \
            cat /sys/class/tenstorrent/tenstorrent!0/power"
```

### For PCIe Performance Analysis:

**Use TT-KMD PCIe counters** - For bandwidth and transfer analysis:

```bash
# Read all PCIe performance counters
ls /sys/class/tenstorrent/tenstorrent!0/pcie_perf_counters/
```

---

## Why Memory Utilization is Only in TT-Metal

The **allocation tracking happens at the TT-Metal layer** because:

1. **Semantic Understanding**: TT-Metal knows what buffers are for (tensors, CBs, etc.)
2. **Application Context**: Only TT-Metal understands program memory requirements
3. **Optimization Layer**: Memory allocation strategies are implemented here
4. **Bank Management**: TT-Metal manages multi-bank allocation across cores

The lower layers (UMD, KMD) only provide:
- Raw memory access (UMD)
- Hardware monitoring (KMD)

They don't track "how much memory is allocated" because that's an application-layer concern.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  Application Layer (Python/C++)                             │
│  • Uses ttnn.allocate(), ttnn.create_device()              │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│  TT-Metal (Memory Tracking Layer)                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Allocator::get_statistics()                          │  │
│  │ • Tracks L1, DRAM, L1_SMALL, TRACE buffers          │  │
│  │ • Returns allocated/free/largest_free statistics     │  │
│  │ • Manages bank allocation across cores               │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ← THIS IS WHERE YOU GET MEMORY UTILIZATION                 │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│  TT-UMD (Device Communication Layer)                        │
│  • write_to_device() / read_from_device()                  │
│  • DMA transfers                                            │
│  • TLB/Memory mapping                                       │
│  • NO memory utilization tracking                           │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│  TT-KMD (Kernel Driver)                                     │
│  • Hardware telemetry (temp, power, voltage)               │
│  • PCIe performance counters                                │
│  • NO memory utilization tracking                           │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│  Hardware (Tenstorrent Device)                              │
│  • L1 SRAM, DRAM, Compute Cores                            │
└─────────────────────────────────────────────────────────────┘
```

---

## Conclusion

**For your memory monitoring tool**, you should:

1. ✅ **Use TT-Metal APIs** - `device->allocator()->get_statistics()` or `GetMemoryView()`
2. ✅ **Read from TT-Metal layer** - This is where allocation tracking happens
3. ❌ **Don't use TT-UMD** - It doesn't track utilization
4. ❌ **Don't use TT-KMD** - It only has hardware health telemetry

The tool I created uses the correct approach by accessing the TT-Metal allocator system directly through the device APIs.
