# Shared Memory (SHM) Allocation Tracking - Implementation Guide

## Overview

The SHM tracking system provides real-time memory allocation monitoring for Tenstorrent devices through persistent shared memory regions. Each physical chip gets its own unique SHM file that persists across process lifetimes, enabling tools like `tt_smi` to monitor memory usage without interfering with running workloads.

## Architecture

### Key Design: Composite ASIC ID

Each physical chip is uniquely identified by a **composite asic_id**:

```cpp
asic_id = (board_id << 8) | asic_location
```

**Examples (N300 boards):**
- Board 1834, chip 0 (MMIO): `asic_id = 469504` (0x1834 << 8 | 0)
- Board 1834, chip 1 (Remote): `asic_id = 469505` (0x1834 << 8 | 1)
- Board 1919, chip 0 (MMIO): `asic_id = 491264` (0x1919 << 8 | 0)
- Board 1919, chip 1 (Remote): `asic_id = 491265` (0x1919 << 8 | 1)

**Examples (Galaxy/UBB systems with 4 trays, 8 chips each):**

For Galaxy systems, `asic_location` is encoded as: `(tray_id << 4) | chip_in_tray`
- Tray 1, Chip 5: `asic_id = (board_id << 8) | 0x15` → board_id shifted, tray=1, chip=5
- Tray 4, Chip 8: `asic_id = (board_id << 8) | 0x48` → board_id shifted, tray=4, chip=8

The tray_id and chip_in_tray are extracted from PCI bus ID:
```cpp
// Wormhole Galaxy: tray_bus_ids = {0xC0, 0x80, 0x00, 0x40} maps to trays 1-4
uint16_t bus_upper = pci_bus & 0xF0;  // Identifies tray
uint32_t chip_in_tray = pci_bus & 0x0F;  // Identifies chip within tray (0-15)
```

**Why this works:**
- Globally unique across all boards, trays, and chips
- Stable - same physical chip always has the same ID
- Independent of Metal's logical device enumeration
- Works across any `TT_VISIBLE_DEVICES` configuration
- Scales to large Galaxy systems (32+ chips)

### SHM File Naming

Each chip's SHM file is named:
```
/dev/shm/tt_device_<asic_id>_memory
```

**Examples (N300):**
- `/dev/shm/tt_device_469504_memory` - Board 1834, chip 0
- `/dev/shm/tt_device_469505_memory` - Board 1834, chip 1
- `/dev/shm/tt_device_491264_memory` - Board 1919, chip 0
- `/dev/shm/tt_device_491265_memory` - Board 1919, chip 1

**Examples (Galaxy with board_id=0xABCD):**
- `/dev/shm/tt_device_11259149_memory` - Tray 1, Chip 5 (0xABCD00 | 0x15)
- `/dev/shm/tt_device_11259208_memory` - Tray 4, Chip 8 (0xABCD00 | 0x48)

## Components

### 1. SharedMemoryStatsProvider (Metal Side)

**Location:** `tt_metal/impl/profiler/memory_stats_shm.{hpp,cpp}`

**Responsibilities:**
- Creates/attaches to SHM region for a device
- Records allocations and deallocations
- Maintains aggregated and per-chip statistics
- Optional per-PID tracking

**Key Methods:**
```cpp
// Constructor - creates SHM file based on asic_id
SharedMemoryStatsProvider(uint64_t asic_id, int device_id);

// Record allocation (called from graph_tracking.cpp)
void record_allocation(pid_t pid, uint64_t size, ShmBufferType type, uint32_t chip_id);

// Record deallocation (called from graph_tracking.cpp)
void record_deallocation(pid_t pid, uint64_t size, ShmBufferType type, uint32_t chip_id);
```

**Initialization (device.cpp):**
```cpp
// Extract physical chip identifiers
uint64_t board_id = tt_device->get_board_id();
uint32_t asic_location = this->is_mmio_capable() ? 0 : 1;

// Compute composite asic_id
uint64_t asic_id = (board_id << 8) | asic_location;

// Create SHM provider
shm_stats_provider_ = std::make_unique<SharedMemoryStatsProvider>(asic_id, this->id_);
```

### 2. Buffer Allocation Tracking (Graph Tracking)

**Location:** `tt_metal/graph/graph_tracking.cpp`

**Integration Points:**

**Allocation:**
```cpp
device->get_shm_stats_provider()->record_allocation(
    getpid(),
    buffer->size(),
    to_shm_buffer_type(buffer->buffer_type()),
    buffer->device()->id()  // chip_id for per-chip tracking
);
```

**Deallocation:**
```cpp
device->get_shm_stats_provider()->record_deallocation(
    getpid(),
    buffer->size(),
    to_shm_buffer_type(buffer->buffer_type()),
    buffer->device()->id()  // chip_id for per-chip tracking
);
```

### 3. tt_smi (Monitoring Tool)

**Location:** `tt_metal/programming_examples/alexp_examples/memory_utilization_monitor/tt_smi.cpp`

**How it Works:**

1. **Discover physical chips via UMD TopologyDiscovery:**
   ```cpp
   for (const auto& [chip_id_composite, chip] : g_topology_cache.chips) {
       // Get board_id and asic_location for each chip
       uint64_t board_id = tt_device->get_board_id();
       uint32_t asic_location = chip_info.asic_location;
   }
   ```

2. **Scan SHM files:**
   ```cpp
   DIR* dir = opendir("/dev/shm");
   // Parse: tt_device_<asic_id>_memory
   uint64_t asic_id = parse_from_filename();
   ```

3. **Match chips to SHM files:**
   ```cpp
   uint64_t asic_id = (dev.board_serial << 8) | dev.asic_location;
   if (chip_shm_map.count(asic_id)) {
       read_memory_from_shm(dev);
   }
   ```

4. **Display per-chip memory usage:**

   **N300 Format:**
   ```
   ID          DRAM Usage          L1 Usage
   -----------------------------------------------
   1834:0      2.8 GB / 12.0 GB    11.0 KB / 1.5 GB
   1834:1R     2.8 GB / 12.0 GB    11.0 KB / 1.5 GB
   ```

   **Galaxy Format:**
   ```
   ID          DRAM Usage          L1 Usage
   -----------------------------------------------
   T1:N0       2.8 GB / 12.0 GB    11.0 KB / 1.5 GB
   T1:N5       1.2 GB / 12.0 GB    5.0 KB / 1.5 GB
   T4:N8       0 B / 12.0 GB       0 B / 1.5 GB
   ```

## SHM Data Structure

```cpp
struct DeviceMemoryRegion {
    // Header
    uint32_t version;
    uint32_t num_active_processes;
    uint64_t last_update_timestamp;
    std::atomic<uint32_t> reference_count;

    // Physical chip identification
    uint64_t board_serial;  // board_id (extracted from asic_id)
    uint64_t asic_id;       // asic_location (extracted from asic_id)
    int32_t device_id;      // Logical Metal device ID

    // Aggregated statistics (all buffers on this chip)
    std::atomic<uint64_t> total_dram_allocated;
    std::atomic<uint64_t> total_l1_allocated;
    std::atomic<uint64_t> total_l1_small_allocated;
    std::atomic<uint64_t> total_trace_allocated;
    std::atomic<uint64_t> total_cb_allocated;
    std::atomic<uint64_t> total_kernel_allocated;

    // Per-chip statistics (for remote device tracking)
    ChipStats chip_stats[MAX_CHIPS_PER_DEVICE];

    // Per-process statistics (optional)
    ProcessStats processes[MAX_PROCESSES];
};
```

## Buffer Types Tracked

| Buffer Type | Description | Typical Location |
|------------|-------------|------------------|
| `DRAM` | Main device memory | DRAM banks |
| `L1` | Fast on-chip memory | Tensix cores |
| `L1_SMALL` | Reserved L1 region | Tensix cores |
| `TRACE` | Trace buffer for recording | DRAM |
| `CB` | Circular buffers | L1 |
| `KERNEL` | Kernel code | L1 |

## Usage

### Enable SHM Tracking

```bash
export TT_METAL_SHM_STATS_ENABLED=1
```

### Optional: Enable Per-PID Tracking

```bash
export TT_METAL_SHM_STATS_PER_PID=1
```

This enables detailed per-process memory breakdown but adds slight overhead (~50ns per allocation).

### Run Workload

```bash
# Your Metal application
python my_workload.py
```

SHM files are automatically created in `/dev/shm/`:
```bash
ls -lh /dev/shm/tt_device_*
# /dev/shm/tt_device_469504_memory
# /dev/shm/tt_device_491264_memory
```

### Monitor with tt_smi

```bash
# Real-time monitoring
./build/programming_examples/tt_smi -w

# One-time snapshot
./build/programming_examples/tt_smi

# Detailed per-process view
./build/programming_examples/tt_smi -d
```

### Cleanup

SHM files persist across runs (like UMD lock files) for continuous monitoring:

```bash
# Manual cleanup if needed
rm /dev/shm/tt_device_*_memory
```

## Key Features

### 1. Physical Chip Stability

✅ **Problem Solved:** Logical device IDs change based on `TT_VISIBLE_DEVICES`

**Example:**
```bash
# Run 1: Only board 1919
TT_VISIBLE_DEVICES=1919 python workload.py
# Creates: /dev/shm/tt_device_491264_memory (1919:0)

# Run 2: All boards
TT_VISIBLE_DEVICES=1834,1919 python workload.py
# Creates: /dev/shm/tt_device_469504_memory (1834:0)
#          /dev/shm/tt_device_491264_memory (1919:0)  ← Same file!
```

### 2. Galaxy (UBB) System Support

Galaxy systems have unique architecture:
- **4 trays** per board
- **8 chips per tray** (32 chips total)
- All chips on same `board_id`
- Each chip gets unique SHM file via encoded `asic_id`

**PCI Bus Encoding:**
```cpp
// Tray identification from PCI bus upper nibble
tray_bus_ids = {0xC0, 0x80, 0x00, 0x40};  // Maps to trays 1-4
tray_id = position in array + 1;
chip_in_tray = pci_bus & 0x0F;  // 0-15
```

**SHM File Creation:**
```cpp
// device.cpp computes asic_id for each chip
if (board_type == BoardType::UBB) {
    uint32_t asic_location_composite = (tray_id << 4) | chip_in_tray;
    asic_id = (board_id << 8) | asic_location_composite;
}
// Creates: /dev/shm/tt_device_<asic_id>_memory
```

**tt_smi Display Format:**
```
ID          Arch          Temp      Power     AICLK       DRAM Usage          L1 Usage
------------------------------------------------------------------------------------------
T1:N0       Wormhole_B0   36°C     21W       500 MHz     2.8 GB / 12.0 GB    11 KB / 1.5 GB
T1:N1       Wormhole_B0   35°C     22W       500 MHz     2.8 GB / 12.0 GB    11 KB / 1.5 GB
T2:N0       Wormhole_B0   36°C     21W       500 MHz     0 B / 12.0 GB       0 B / 1.5 GB
...
T4:N7       Wormhole_B0   35°C     21W       500 MHz     0 B / 12.0 GB       0 B / 1.5 GB
```

### 3. Remote Device Support (N300)

For N300 boards with MMIO and remote chips:
- MMIO chips (asic_location=0) have their own SHM
- Remote chips (asic_location=1) have their own SHM
- Each creates a separate Device object in Metal

### 4. Multi-Process Safe

- Atomic operations for all counters
- Reference counting for lifecycle management
- Process cleanup on exit (when refcount → 0)

### 5. Persistent Monitoring

SHM files persist between runs, allowing:
- Monitoring idle state (0 allocations is meaningful)
- Historical view of last known state
- Non-invasive monitoring by tools

## Performance

- **Allocation tracking:** ~20-50ns overhead per allocation
- **SHM access:** Lock-free atomic operations
- **tt_smi:** Read-only, non-blocking, safe during device reset

## Debugging

### Check if SHM tracking is enabled

```bash
# Look for debug logs
TT_METAL_LOGGER_LEVEL=DEBUG ./my_app 2>&1 | grep "SHM tracking"
# Output: "Device 0: board_id=0x727, asic_location=0 -> asic_id=0x72700 for SHM tracking"
```

### Inspect SHM file

```bash
# List SHM files
ls -lh /dev/shm/tt_device_*

# Check if process is attached
cat /proc/$(pgrep python)/maps | grep tt_device
```

### Verify board_id and asic_location

```bash
# From UMD
./build/programming_examples/tt_smi  # Shows board:location format
```

## Architecture Decisions

### Why Composite asic_id?

**Alternatives considered:**
1. ❌ `(board_id, asic_location)` tuple - requires parsing two fields
2. ❌ Logical `device_id` - changes with `TT_VISIBLE_DEVICES`
3. ✅ Composite `asic_id` - single globally unique ID

### Why Separate SHM per Chip?

**Benefits:**
- Isolated tracking per physical chip
- Scales to any number of boards
- No coordination between boards needed
- Simple cleanup (per-chip refcount)

### Why Persistent SHM?

**Benefits:**
- Continuous monitoring (like nvidia-smi)
- View allocations even when process exits
- Non-invasive - monitoring doesn't affect workload
- Matches UMD's persistent lock files pattern

## Troubleshooting

### Issue: "No tracking" in tt_smi

**Causes:**
1. SHM tracking not enabled: `export TT_METAL_SHM_STATS_ENABLED=1`
2. SHM files not created: Check `/dev/shm/tt_device_*`
3. Board ID mismatch: Verify with debug logs

### Issue: Stale allocations after process exit

**Solution:**
- SHM is reset when last process detaches (refcount → 0)
- If stale data persists: `rm /dev/shm/tt_device_*_memory`

### Issue: Different allocations on different runs

**Likely cause:** Different `TT_VISIBLE_DEVICES` configuration

**Solution:** Composite asic_id ensures same chip → same file

## Summary

The SHM tracking system provides robust, persistent, per-chip memory monitoring using a simple composite asic_id approach. It's designed to be:

- **Stable:** Physical chip IDs never change
- **Scalable:** Works with any number of boards
- **Non-invasive:** Monitoring doesn't affect workloads
- **Performant:** Lock-free atomic operations
- **Persistent:** SHM survives process restarts

This enables real-time memory profiling and debugging for Tenstorrent multi-chip systems.
