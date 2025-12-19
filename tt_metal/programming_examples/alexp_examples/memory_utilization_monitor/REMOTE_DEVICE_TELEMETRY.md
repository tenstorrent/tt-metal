# Remote Device Telemetry in tt_smi

## Overview

`tt_smi` can monitor **both local (PCIe) and remote (Ethernet-connected) Tenstorrent devices** in multi-chip systems like Galaxy (N300, T3000, etc.). This document explains how remote device telemetry works.

---

## Architecture

### Device Discovery

```
┌─────────────────────────────────────────────────────────────┐
│                    tt_smi Startup                           │
│  TopologyDiscovery::discover(no_remote_discovery = false)   │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────┐
         │  Scans PCIe bus for gateway chips   │
         │  - Reads PCI device descriptors     │
         │  - Discovers local devices          │
         └─────────────────────────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────┐
         │  Ethernet topology discovery         │
         │  - Configures NOC-to-Ethernet routing│
         │  - Maps remote chips via gateway     │
         │  - Assigns unique chip IDs           │
         └─────────────────────────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────┐
         │  Returns:                            │
         │  - ClusterDescriptor (topology)      │
         │  - chip_map (Chip objects)           │
         └─────────────────────────────────────┘
```

### Key Components

#### 1. **TopologyDiscovery** (from UMD)
- **Location**: `umd/device/topology/topology_discovery.hpp`
- **Purpose**: Discovers all chips in the system (local + remote)
- **Output**:
  - `ClusterDescriptor`: Physical topology (which chips are connected)
  - `chip_map`: Map of `chip_id_composite` → `Chip*` objects

#### 2. **Chip** (from UMD)
- **Location**: `umd/device/chip/chip.hpp`
- **Purpose**: Represents a single Tenstorrent chip
- **Key method**: `get_tt_device()` - returns the low-level device interface
- **Important**: Works for **both local and remote** chips transparently

#### 3. **TTDevice** (from UMD)
- **Location**: `umd/device/tt_device.hpp`
- **Purpose**: Low-level device interface for register access
- **Remote access**: For remote chips, automatically routes requests through:
  - Gateway chip's PCIe interface
  - NOC (Network-on-Chip) to Ethernet cores
  - Ethernet link to remote chip
  - Remote chip's ARC firmware

#### 4. **FirmwareInfoProvider** (from UMD)
- **Location**: `umd/device/firmware/firmware_info_provider.hpp`
- **Purpose**: High-level API for reading telemetry from ARC firmware
- **Methods**:
  - `get_asic_temperature()` - Die temperature (°C)
  - `get_tdp()` - Thermal Design Power (W)
  - `get_aiclk()` - AI Clock frequency (MHz)

---

## How Remote Telemetry Works

### Step-by-Step Flow

```
User runs: ./tt_smi -w

1. INITIALIZATION (once at startup)
   ↓
   TopologyDiscovery::discover()
   ├─ Scans PCIe: Finds gateway chips (e.g., chip 0, chip 2)
   ├─ Ethernet discovery: Routes through gateways to find remote chips
   └─ Builds chip_map:
      {
        0x12345678 → Chip* (local, PCIe-attached)
        0x12345679 → Chip* (remote, via chip 0's Ethernet)
        0x1234567A → Chip* (local, PCIe-attached)
        0x1234567B → Chip* (remote, via chip 2's Ethernet)
      }

2. TELEMETRY QUERY (every refresh interval, e.g., 1 second)
   ↓
   For each device in chip_map:
   ├─ chip->get_tt_device()  // Get device interface
   ├─ tt_device->get_firmware_info_provider()  // Access firmware API
   └─ firmware_info->get_asic_temperature()    // Read telemetry
      │
      └─ FOR REMOTE CHIPS, this internally does:
         ├─ TTDevice routes request through gateway chip
         ├─ Gateway's Ethernet core forwards to remote chip
         ├─ Remote chip's ARC firmware responds with telemetry
         └─ Response is routed back to host via same path

3. DISPLAY
   ↓
   tt_smi prints:
   Device 0      (local)    - 45°C, 35W, 1000 MHz
   Device 0R     (remote)   - 42°C, 33W, 1000 MHz  ← Accessed via Device 0
   Device 1      (local)    - 38°C, 28W, 1000 MHz
   Device 1R     (remote)   - 36°C, 25W, 1000 MHz  ← Accessed via Device 1
```

### Ethernet Routing Details

For remote chips, all communication goes through **Ethernet cores** on the gateway chip:

```
┌──────────────────────────────────────────────────────────────┐
│                        Host (x86 CPU)                         │
└──────────────────────────────────────────────────────────────┘
                           │ PCIe
                           ▼
┌──────────────────────────────────────────────────────────────┐
│               Gateway Chip (e.g., Device 0)                   │
│  ┌──────────┐    ┌─────────┐    ┌──────────────────┐        │
│  │  Tensix  │◄──►│   NOC   │◄──►│  Ethernet Core   │─┐      │
│  │  Cores   │    │ Routing │    │  (eth_dispatch)  │ │      │
│  └──────────┘    └─────────┘    └──────────────────┘ │      │
└────────────────────────────────────────────────────────┼──────┘
                                                         │ Ethernet Link
                                                         │ (100 Gbps)
┌────────────────────────────────────────────────────────┼──────┐
│               Remote Chip (e.g., Device 0R)            │      │
│  ┌──────────────────┐    ┌─────────┐    ┌──────────┐ │      │
│  │  Ethernet Core   │◄──►│   NOC   │◄──►│  Tensix  │ │      │
│  │  (eth_dispatch)  │    │ Routing │    │  Cores   │ │      │
│  └──────────────────┘    └─────────┘    └──────────┘ │      │
│           │                                            │      │
│           ▼                                            │      │
│  ┌──────────────────┐                                 │      │
│  │  ARC Firmware    │  ← Telemetry data source        │      │
│  │  (Temperature,   │                                 │      │
│  │   Power, AICLK)  │                                 │      │
│  └──────────────────┘                                 │      │
└────────────────────────────────────────────────────────┴──────┘
```

**Key Points:**
- **Transparent**: `tt_smi` doesn't need to know which chips are remote
- **Automatic routing**: UMD's `TopologyDiscovery` configures all routing tables
- **Same API**: Local and remote chips use identical `FirmwareInfoProvider` API
- **Resilient**: Built-in error handling and retry logic for flaky Ethernet links

---

## Code Example

```cpp
// Initialize topology discovery
TopologyDiscoveryOptions options;
options.no_remote_discovery = false;  // Enable remote device discovery
auto [descriptor, chip_map] = TopologyDiscovery::discover(options);

// Iterate over all chips (local + remote)
for (auto& [chip_id, chip_ptr] : chip_map) {
    Chip* chip = chip_ptr.get();

    // Get device interface (works for both local and remote!)
    TTDevice* tt_device = chip->get_tt_device();

    // Get firmware info provider
    auto firmware_info = tt_device->get_firmware_info_provider();

    // Read telemetry (automatic Ethernet routing for remote chips)
    double temp = firmware_info->get_asic_temperature();
    auto tdp = firmware_info->get_tdp();
    auto aiclk = firmware_info->get_aiclk();

    std::cout << "Chip " << chip_id << ": "
              << temp << "°C, "
              << tdp.value() << "W, "
              << aiclk.value() << " MHz\n";
}
```

---

## Error Handling

`tt_smi` implements **resilient telemetry queries** with adaptive retry intervals:

### Error Types

1. **`chip_offline`** (2s retry interval)
   - Chip is being reset or powered down
   - Fast retry to quickly detect recovery

2. **`eth_busy`** (15s retry interval)
   - Ethernet cores are busy with inference workload
   - Longer retry to avoid interfering with real work

3. **`other`** (10s retry interval)
   - Generic errors (firmware not responding, etc.)
   - Medium retry interval

### Cached Telemetry

When a query fails, `tt_smi` shows:
- **Last known good values** with a "stale" indicator
- **Time since last successful read**

Example:
```
Device 0R   Wormhole_B0   42°C* (15s old)   N/A   N/A   OK (ETH busy)
```

This prevents the UI from going blank during transient errors.

---

## Memory Tracking for Remote Devices

Memory statistics (DRAM, L1) work differently from telemetry:

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Metal Application                         │
│  (allocates buffers on local and remote devices)            │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   SHM Tracking Layer                         │
│  Records allocations to /dev/shm/tt_device_<asic_id>_memory │
│                                                              │
│  Per-chip stats:                                             │
│  ├─ chip_stats[0]: Gateway chip (local)                     │
│  ├─ chip_stats[1]: Remote chip 1                            │
│  └─ chip_stats[2]: Remote chip 2                            │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────┐
         │           tt_smi                     │
         │  Reads per-chip stats from SHM       │
         │  - chip_stats[0] → Display as "0"    │
         │  - chip_stats[1] → Display as "0R"   │
         └─────────────────────────────────────┘
```

**Key Difference from Telemetry:**
- **Telemetry**: Actively queried from remote chips via Ethernet
- **Memory**: Tracked in shared memory by Metal runtime, read locally by `tt_smi`

---

## Device Naming Convention

`tt_smi` uses a naming scheme to show remote devices:

| Display ID    | Meaning                              |
|---------------|--------------------------------------|
| `261834045`   | Local chip (PCIe-attached)           |
| `361834045R`  | Remote chip (via gateway 261834045)  |
| `26191901e`   | Local chip (PCIe-attached)           |
| `36191901eR`  | Remote chip (via gateway 26191901e)  |

**Format**: `<gateway_asic_id>R` for remote chips

---

## Performance Considerations

### Telemetry Query Overhead

- **Local chips**: ~100 μs (direct PCIe read)
- **Remote chips**: ~5-10 ms (PCIe → NOC → Ethernet → NOC → ARC firmware)

### Refresh Rate

`tt_smi` default refresh interval: **1 second**

For 4-chip systems (2 local + 2 remote):
- Total telemetry time: ~20 ms per refresh
- CPU overhead: Negligible (<1%)
- **Does not interfere with inference workloads**

### Avoiding Contention

When Ethernet cores are busy (inference running):
- UMD may return `ETH_BUSY` error
- `tt_smi` automatically backs off (15s retry interval)
- Shows cached telemetry with age indicator

---

## Limitations

1. **No Direct PCIe Access to Remote Chips**
   - Remote chips are not directly visible on the PCIe bus
   - All access is routed through gateway chip's Ethernet

2. **Ethernet Link Required**
   - If Ethernet link is down, remote chip telemetry is unavailable
   - Memory tracking still works (via SHM on local host)

3. **ARC Firmware Dependency**
   - Telemetry requires ARC firmware to be running on the chip
   - If firmware hangs, telemetry will fail

4. **Latency**
   - Remote telemetry is ~50-100x slower than local
   - Not suitable for high-frequency monitoring (<100ms intervals)

---

## Summary

`tt_smi` leverages UMD's **TopologyDiscovery** to:
1. **Discover** all chips (local + remote) at startup
2. **Configure** Ethernet routing tables automatically
3. **Query** telemetry through a unified API (transparent remote access)
4. **Handle** errors gracefully with cached values and adaptive retry

**Result**: Users see a single unified view of all devices, regardless of whether they're connected via PCIe or Ethernet.
