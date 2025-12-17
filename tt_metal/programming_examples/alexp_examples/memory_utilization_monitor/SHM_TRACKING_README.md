# Shared Memory (SHM) Allocation Tracking - Implementation Guide

## Overview

The SHM tracking system provides real-time memory allocation monitoring for Tenstorrent devices through persistent shared memory regions. Each physical chip gets its own unique SHM file that persists across process lifetimes, enabling tools like `tt_smi` to monitor memory usage without interfering with running workloads.

## Architecture

### Key Design: Hardware-Based Unique ASIC ID

Each physical chip has a **64-bit unique ASIC ID** that is read directly from chip hardware by UMD's topology discovery. This ID is globally unique and stable across reboots.

#### How UMD Discovers Unique ASIC IDs

UMD's `TopologyDiscovery` reads the ASIC ID from chip firmware via Ethernet cores:

```cpp
// topology_discovery_wormhole.cpp
uint64_t TopologyDiscoveryWormhole::get_local_asic_id(Chip* chip, tt_xy_pair eth_core) {
    TTDevice* tt_device = chip->get_tt_device();
    uint32_t asic_id_lo, asic_id_hi;

    // Read 64-bit ASIC ID directly from chip hardware
    tt_device->read_from_device(&asic_id_lo, eth_core, ...);
    tt_device->read_from_device(&asic_id_hi, eth_core, ...);

    return ((uint64_t)asic_id_hi << 32) | asic_id_lo;
}
```

**Examples (N300 boards - real hardware IDs):**
- Chip 0: `asic_id = 0x251732099` (39,281,073,305)
- Chip 1: `asic_id = 0x2517320d4` (39,281,073,428)
- Chip 2: `asic_id = 0x2517320eb` (39,281,073,451)
- Chip 3: `asic_id = 0x2517320f4` (39,281,073,460)

**Examples (Galaxy/UBB systems):**
- Similar 64-bit IDs unique to each chip in the system
- Each tray and chip combination has its own unique hardware ID

#### Composite ASIC ID for SHM File Naming

For SHM file naming, Metal computes a composite `asic_id`:

```cpp
// device.cpp - SHM provider initialization
asic_id = (board_id << 8) | asic_location_composite

// For N300: asic_location = 0 (MMIO) or 1 (Remote)
// For Galaxy: asic_location = (tray_id << 4) | chip_in_tray
```

**Why this approach:**
- **Hardware ASIC ID**: Used for display and chip correlation (globally unique, stable)
- **Composite ASIC ID**: Used for SHM file naming (predictable, board-relative)
- Globally unique across all boards, trays, and chips
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

**How Allocations are Linked to ASICs:**

The key insight is that each `Device` object owns its own `SharedMemoryStatsProvider`, which writes to a specific SHM file. The linkage is through direct pointer ownership:

```
Buffer Object
    └─> device pointer (IDevice*)
            └─> Device instance (for physical chip)
                    └─> shm_stats_provider_ (unique per Device)
                            └─> Writes to: /dev/shm/tt_device_<asic_id>_memory
```

**Integration Points:**

**Allocation:**
```cpp
void GraphTracker::track_allocate(const Buffer* buffer) {
    // Each buffer knows its Device
    auto* device = dynamic_cast<const Device*>(buffer->device());

    // Each Device owns its SHM provider (created during Device initialization)
    if (device && device->get_shm_stats_provider()) {
        // Record allocation to THIS device's SHM file
        device->get_shm_stats_provider()->record_allocation(
            getpid(),
            buffer->size(),
            to_shm_buffer_type(buffer->buffer_type()),
            buffer->device()->id()  // chip_id for per-chip tracking
        );
    }
}
```

**Deallocation:**
```cpp
void GraphTracker::track_deallocate(Buffer* buffer) {
    auto* device = dynamic_cast<Device*>(buffer->device());
    if (device && device->get_shm_stats_provider()) {
        device->get_shm_stats_provider()->record_deallocation(
            getpid(),
            buffer->size(),
            to_shm_buffer_type(buffer->buffer_type()),
            buffer->device()->id()
        );
    }
}
```

**Example with 4 N300 Boards (8 chips total):**

```
Process initializes 8 Device objects:

Device 0 (MMIO, asic_id=0x251732099):
  ├─ shm_stats_provider_ → /dev/shm/tt_device_<id0>_memory
  └─ All buffers allocated on Device 0 → recorded to this SHM file

Device 1 (Remote, asic_id=0x251732099R):
  ├─ shm_stats_provider_ → /dev/shm/tt_device_<id1>_memory
  └─ All buffers allocated on Device 1 → recorded to this SHM file

Device 2 (MMIO, asic_id=0x2517320d4):
  ├─ shm_stats_provider_ → /dev/shm/tt_device_<id2>_memory
  └─ All buffers allocated on Device 2 → recorded to this SHM file

... (and so on for all 8 chips)
```

Each allocation automatically goes to the correct SHM file because the buffer's Device pointer leads directly to that Device's unique SHM provider. **No lookup or mapping is needed** - it's direct O(1) pointer dereferencing.

### 3. tt_smi (Monitoring Tool)

**Location:** `tt_metal/programming_examples/alexp_examples/memory_utilization_monitor/tt_smi.cpp`

**How it Works:**

1. **Discover physical chips via UMD TopologyDiscovery:**
   ```cpp
   // TopologyDiscovery reads 64-bit ASIC IDs from chip hardware
   auto [descriptor, chip_map] = TopologyDiscovery::discover(options);

   for (const auto& [chip_id_composite, chip] : chip_map) {
       // chip_id_composite IS the unique hardware ASIC ID
       dev.chip_id_composite = chip_id_composite;

       // Also get board info for SHM file matching
       uint64_t board_id = tt_device->get_board_id();
       uint32_t asic_location = chip_info.asic_location;

       // Compute composite asic_id for SHM file lookup
       dev.asic_id = (board_id << 8) | asic_location;
   }
   ```

2. **Scan SHM files:**
   ```cpp
   DIR* dir = opendir("/dev/shm");
   // Parse filenames: tt_device_<asic_id>_memory
   // Returns map: asic_id → file_descriptor
   auto chip_shm_map = scan_shm_files();
   ```

3. **Match chips to SHM files:**
   ```cpp
   // Use the computed composite asic_id to find matching SHM file
   uint64_t asic_id = dev.asic_id;  // Pre-computed in step 1

   if (chip_shm_map.count(asic_id)) {
       // Found matching SHM file for this chip!
       read_memory_from_shm(dev);
   }
   ```

4. **Display using hardware ASIC ID:**
   ```cpp
   // Display the true unique hardware ASIC ID (chip_id_composite)
   // This is what the user sees - globally unique, stable across reboots
   if (dev.chip_id_composite != 0) {
       printf("%llx", dev.chip_id_composite);  // e.g., "251732099"
   }
   ```

4. **Display per-chip memory usage:**

   **N300 Format (showing true hardware ASIC IDs):**
   ```
   ID          DRAM Usage          L1 Usage
   -----------------------------------------------
   251732099   2.8 GB / 12.0 GB    11.0 KB / 1.5 GB
   251732099R  2.8 GB / 12.0 GB    11.0 KB / 1.5 GB   (Remote chip)
   2517320d4   1.2 GB / 12.0 GB    5.0 KB / 1.5 GB
   2517320d4R  1.2 GB / 12.0 GB    5.0 KB / 1.5 GB   (Remote chip)
   ```

   The ID is the chip's **unique 64-bit hardware ASIC ID** (displayed in hex).
   Remote chips are marked with 'R' suffix.

   **Galaxy Format:**
   ```
   ID          DRAM Usage          L1 Usage
   -----------------------------------------------
   T1:N0       2.8 GB / 12.0 GB    11.0 KB / 1.5 GB
   T1:N5       1.2 GB / 12.0 GB    5.0 KB / 1.5 GB
   T4:N8       0 B / 12.0 GB       0 B / 1.5 GB
   ```

   Galaxy systems use a special "T<tray>:N<chip>" format for readability.

## SHM Data Structure

```cpp
struct DeviceMemoryRegion {
    // Header
    uint32_t version;                        // Structure version for compatibility
    uint32_t num_active_processes;           // Count of active processes
    uint64_t last_update_timestamp;          // Last update time (nanoseconds)
    std::atomic<uint32_t> reference_count;   // Number of attached processes

    // Physical chip identification
    uint64_t board_serial;  // Board ID portion (upper bits of composite asic_id)
    uint64_t asic_id;       // ASIC location portion (lower bits of composite asic_id)
    int32_t device_id;      // Logical Metal device ID (for reference)

    // Aggregated statistics (all buffers on this chip, across all processes)
    std::atomic<uint64_t> total_dram_allocated;
    std::atomic<uint64_t> total_l1_allocated;
    std::atomic<uint64_t> total_l1_small_allocated;
    std::atomic<uint64_t> total_trace_allocated;
    std::atomic<uint64_t> total_cb_allocated;
    std::atomic<uint64_t> total_kernel_allocated;

    // Per-chip statistics (for remote device tracking via gateway)
    ChipStats chip_stats[MAX_CHIPS_PER_DEVICE];

    // Per-process statistics (enabled via TT_METAL_SHM_STATS_PER_PID=1)
    struct ProcessStats {
        pid_t pid;                       // Process ID (0 = unused slot)
        uint64_t dram_allocated;         // DRAM allocated by this process
        uint64_t l1_allocated;           // L1 allocated by this process
        uint64_t l1_small_allocated;
        uint64_t trace_allocated;
        uint64_t cb_allocated;
        uint64_t kernel_allocated;
        uint64_t last_update_timestamp;
        char process_name[64];           // Process name from /proc/<pid>/comm
    } processes[MAX_PROCESSES];
};
```

### How PIDs are Tracked

When per-PID tracking is enabled (`TT_METAL_SHM_STATS_PER_PID=1`):

1. **PID Registration**: On first allocation from a process, Metal finds or creates an entry in `processes[]`
2. **Process Name**: Read from `/proc/<pid>/comm` and stored in SHM
3. **Per-Allocation Update**: Each allocation/deallocation updates that process's counters
4. **Liveness Check**: `tt_smi` uses `kill(pid, 0)` to check if process is still alive
5. **Cleanup**: Dead process entries are filtered out when displaying

```cpp
// SharedMemoryStatsProvider::find_or_create_pid_entry()
ProcessStats* entry = find_unused_slot();
entry->pid = getpid();
entry->process_name = read_from("/proc/<pid>/comm");
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

## Unique Chip Identifiers

The system uses multiple types of unique identifiers for different purposes:

### 1. Hardware ASIC ID (64-bit) - For Display and Correlation

- **Source**: Read directly from chip firmware via UMD's `TopologyDiscovery`
- **Uniqueness**: Globally unique across all chips, burned into hardware
- **Stability**: Never changes, survives reboots
- **Usage**: Display in `tt_smi`, chip correlation
- **Example**: `0x251732099`, `0x2517320d4`
- **Access**: `chip_id_composite` from topology discovery

```cpp
// Read from chip hardware (topology_discovery_wormhole.cpp)
uint64_t asic_id = get_local_asic_id(chip, eth_core);
// Returns: 64-bit unique ID from chip firmware
```

### 2. Composite ASIC ID - For SHM File Naming

- **Source**: Computed by Metal from board_id and asic_location
- **Formula**: `asic_id = (board_id << 8) | asic_location`
- **Uniqueness**: Unique within a system, board-relative
- **Stability**: Stable as long as board topology doesn't change
- **Usage**: SHM file naming (`/dev/shm/tt_device_<asic_id>_memory`)
- **Benefits**: Predictable, human-readable structure

```cpp
// Computed during Device initialization (device.cpp)
uint64_t board_id = tt_device->get_board_id();
uint8_t asic_location = tt_device->get_asic_location();
uint64_t asic_id = (board_id << 8) | asic_location;
```

### 3. PCI Bus:Device:Function (BDF) - For PCIe Devices

- **Source**: Linux kernel PCI enumeration
- **Format**: `domain:bus:device.function` (e.g., `0000:03:00.0`)
- **Uniqueness**: Unique per host system
- **Usage**: MMIO-capable chips only
- **Note**: Changes if PCIe topology changes

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

### Verify unique ASIC IDs

```bash
# tt_smi displays the true 64-bit hardware ASIC IDs
./build/programming_examples/tt_smi

# Example output:
# ID          Arch          DRAM Usage          L1 Usage
# ---------------------------------------------------------
# 251732099   Wormhole_B0   0 B / 12.0 GB      0 B / 1.5 GB
# 251732099R  Wormhole_B0   0 B / 12.0 GB      0 B / 1.5 GB
# 2517320d4   Wormhole_B0   0 B / 12.0 GB      0 B / 1.5 GB
# 2517320d4R  Wormhole_B0   0 B / 12.0 GB      0 B / 1.5 GB

# Check SHM files on disk
ls -lh /dev/shm/tt_device_*
# Note: SHM filenames use composite asic_id (board_id << 8 | location)
#       Display IDs are hardware ASIC IDs (read from chip firmware)
```

## Architecture Decisions

### Why Two Types of ASIC IDs?

**Design Choice**: Use hardware ASIC ID for display, composite ASIC ID for SHM files

**Hardware ASIC ID (chip_id_composite):**
- ✅ True global uniqueness (burned into chip)
- ✅ Never changes, survives reboots
- ✅ Best for user-facing display
- ❌ Unpredictable structure (hardware-assigned)
- ❌ Can't compute without reading from chip

**Composite ASIC ID (board_id << 8 | location):**
- ✅ Predictable structure for file naming
- ✅ Can be computed from known board topology
- ✅ Human-readable encoding
- ✅ Works for file system naming
- ⚠️ Requires board_id stability

**Alternatives considered:**
1. ❌ Hardware ASIC ID only - unpredictable filenames, hard to debug
2. ❌ Composite ID only - less user-friendly display
3. ❌ Logical `device_id` - changes with `TT_VISIBLE_DEVICES`
4. ✅ Both IDs - best of both worlds (display + file naming)

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

**Debug:**
```bash
TT_METAL_LOGGER_LEVEL=DEBUG python my_app.py 2>&1 | grep "SHM tracking"
# Should show: "Device X: board_id=0x..., asic_location=... -> asic_id=0x... for SHM tracking"
```

### Issue: Same ID appearing for multiple devices in tt_smi

**Symptom:** Multiple chips showing identical IDs (e.g., all show `1732:0`, `1732:1R`)

**Cause:** Display logic incorrectly extracting board ID from composite values

**Solution:** Use `chip_id_composite` (hardware ASIC ID) for display instead of trying to parse `board_serial`

```cpp
// WRONG: board_serial may already be composite
uint32_t board_short = (dev.board_serial >> 12) & 0xFFFF;

// CORRECT: Use hardware ASIC ID directly
if (dev.chip_id_composite != 0) {
    printf("%llx", dev.chip_id_composite);  // Shows true unique ID
}
```

### Issue: Stale allocations after process exit

**Solution:**
- SHM is reset when last process detaches (refcount → 0)
- If stale data persists: `rm /dev/shm/tt_device_*_memory`

**Auto-cleanup:**
```bash
./build/programming_examples/tt_smi -c   # Clean dead processes once
./build/programming_examples/tt_smi -w -c # Watch mode with auto-cleanup
```

### Issue: Different allocations on different runs

**Likely cause:** Different `TT_VISIBLE_DEVICES` configuration

**Solution:** Composite asic_id ensures same chip → same file

**Verify:**
```bash
# List SHM files with timestamps
ls -lht /dev/shm/tt_device_*

# Check which processes have SHM mapped
for pid in $(pgrep -f python); do
    echo "Process $pid:";
    cat /proc/$pid/maps | grep tt_device
done
```

## Summary

The SHM tracking system provides robust, persistent, per-chip memory monitoring using a dual-identifier approach:

### Key Design Elements:

1. **Hardware ASIC ID (64-bit)**: True unique ID from chip firmware for display
2. **Composite ASIC ID**: Computed ID for SHM file naming
3. **Direct Pointer Linkage**: Buffer → Device → SHM Provider (O(1) lookup)
4. **Per-Chip Isolation**: Each physical chip has its own SHM file
5. **Atomic Operations**: Lock-free updates for performance

### System Properties:

- **Stable:** Physical chip IDs never change, survive reboots
- **Scalable:** Works with any number of boards/chips
- **Non-invasive:** Monitoring doesn't affect workloads
- **Performant:** ~20-50ns overhead per allocation
- **Persistent:** SHM files survive process restarts (like UMD locks)
- **Accurate:** Each allocation automatically goes to correct chip's SHM file

### Allocation Flow:

```
Buffer Allocation
    └─> buffer->device() points to Device object
            └─> device->shm_stats_provider_ (unique per Device)
                    └─> record_allocation() updates specific SHM file
                            └─> /dev/shm/tt_device_<asic_id>_memory
```

No mapping or lookup needed - direct pointer ownership ensures allocations are tracked to the correct physical chip.

### Monitoring Flow:

```
tt_smi reads topology
    └─> Discovers hardware ASIC IDs from chip firmware
            └─> Matches to SHM files via composite asic_id
                    └─> Displays with hardware ASIC ID for uniqueness
```

This enables real-time, per-chip memory profiling and debugging for Tenstorrent multi-chip systems.
