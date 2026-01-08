# TT-SMI - Tenstorrent System Management Interface

A comprehensive monitoring tool for Tenstorrent devices with real-time telemetry, memory tracking, and beautiful terminal UI.

## Quick Start

```bash
# Pure Python (30-second install, no compilation)
cd $TT_METAL_HOME/tt_metal/programming_examples/alexp_examples/memory_utilization_monitor

pip install -e .

# Run
tt-smi-ui -w            # Watch mode (live updates)
tt-smi-ui -w -g         # Graph mode (nvtop-style)
tt-smi-ui -w -r 500     # Fast refresh (500ms)
```

---

## Features

### Beautiful UI
- **Rich library** with colors, tables, live updates
- **Graph mode** with nvtop-style charts (temperature, power, DRAM, L1)
- **Dynamic resizing** - charts adjust to terminal size
- **Matrix layout** - 2x2, 2x4, 4x4 device grids

### Telemetry
- Temperature, Power, AICLK via UMD
- Voltage and current monitoring
- Local and remote device support (Ethernet)
- Resilient error handling with cached values

### Memory Tracking
- DRAM, L1, L1_SMALL, Trace, CB allocations
- Per-process breakdown (PID, name)
- Per-chip tracking (local + remote)
- Automatic dead process cleanup

---

## Installation

**Requires TT-Metal to be built first:**

```bash
# Step 1: Build TT-Metal (if not already done)
cd $TT_METAL_HOME$
./build_metal_with_flags.sh

# Step 2: Install Python UI with C++ support
cd tt_metal/programming_examples/alexp_examples/memory_utilization_monitor
pip install pybind11
pip install -e .
```

**Features:**
- Memory monitoring
- Per-process tracking
- Graph visualization
- Telemetry (temp/power/clock via UMD)
- Remote device support


## Usage

### Command Line

```bash
# Single snapshot
tt-smi-ui

# Watch mode (live updates)
tt-smi-ui -w

# Graph mode (nvtop-style)
tt-smi-ui -w -g

# Fast refresh (500ms)
tt-smi-ui -w -r 500


# JSON output
tt-smi-ui --json
```

### Python API

```python
from tt_smi_ui import get_devices, update_telemetry_parallel, cleanup_dead_processes

# Get all devices
devices = get_devices()

# Update telemetry (parallel, fast)
update_telemetry_parallel(devices, timeout=1.0)

for dev in devices:
    print(f"Device {dev.display_id}:")
    print(f"  Arch: {dev.arch_name}")
    print(f"  Temp: {dev.temperature}°C")
    print(f"  Power: {dev.power}W")
    print(f"  DRAM: {dev.used_dram}/{dev.total_dram}")
    print(f"  L1: {dev.used_l1}/{dev.total_l1}")

    for proc in dev.processes:
        print(f"    PID {proc['pid']}: {proc['name']}")

# Clean up dead processes
cleaned = cleanup_dead_processes()
print(f"Cleaned {cleaned} dead processes")
```

### Dashboard UI

```python
from tt_smi_ui import get_devices, update_telemetry_parallel, update_memory
from tt_smi_ui.ui.dashboard import Dashboard
from tt_smi_ui.ui.graphs import GraphWindow
from rich.console import Console

console = Console()
dashboard = Dashboard(console)

# Table view
dashboard.watch(
    get_devices_func=get_devices,
    refresh_ms=500,
    update_telemetry_parallel_func=update_telemetry_parallel,
    update_memory_func=update_memory
)

# Graph view
graph_window = GraphWindow(console, history_size=100)
dashboard.watch(
    get_devices_func=get_devices,
    refresh_ms=500,
    update_telemetry_parallel_func=update_telemetry_parallel,
    update_memory_func=update_memory,
    graph_window=graph_window
)
```

---

## Architecture

```
tt-metal-apv/tt_metal/programming_examples/alexp_examples/memory_utilization_monitor/
│
├── tt_smi.cpp                  # Standalone C++ tool (CMake-built)
│
├── tt_smi/                     # Python package
│   ├── __init__.py             # Package exports
│   ├── core.py                 # Python API wrapper
│   ├── cli.py                  # Command-line interface
│   │
│   ├── bindings/
│   │   ├── native.cpp          # pybind11 C++ bindings
│   │   └── shm_reader.py       # Pure Python fallback
│   │
│   └── ui/
│       ├── dashboard.py        # Rich-based TUI (table view)
│       └── graphs.py           # nvtop-style graphs
│
├── tt_smi_backend.hpp          # C++ API header (shared with Python)
├── tt_smi_backend.cpp          # C++ implementation
├── setup.py                    # Python installer
└── README.md                   # This file
```

### Key Components

#### 1. C++ Backend (`tt_smi_backend.cpp/hpp`)

Reusable library extracted from `tt_smi.cpp`:
- `enumerate_devices()` - Device discovery via TopologyDiscovery
- `query_telemetry()` - UMD telemetry (temp/power/clock)
- `read_memory_from_shm()` - SHM memory stats
- `cleanup_dead_processes()` - Automatic dead PID cleanup
- `reset_devices()` - Device warm reset

#### 2. Python Bindings (`bindings/native.cpp`)

pybind11 wrapper exposing C++ structs:
- `Device` - Full device info
- `TelemetryData` - Temp/power/clock/voltage/current
- `ProcessMemory` - Per-process stats

#### 3. Fallback SHM Reader (`bindings/shm_reader.py`)

Pure Python implementation:
- Direct `/dev/shm/tt_device_*_memory` parsing
- No UMD dependency (no telemetry)
- Works if C++ compilation fails

#### 4. Core API (`core.py`)

Unified Python interface:
- `get_devices()` - Enumerate devices
- `update_telemetry_parallel()` - Fast parallel telemetry
- `update_memory()` - Refresh memory stats
- `reset_devices()` - Device reset
- Automatically uses native or fallback backend

#### 5. Dashboard (`ui/dashboard.py`)

Rich library UI:
- Color-coded tables
- Live refresh mode
- Temperature/power/clock display
- DRAM/L1 usage bars
- Per-process breakdown
- Status indicators (OK/Offline/ETH busy)

#### 6. Graph Window (`ui/graphs.py`)

nvtop-style visualization:
- Combined charts (Temperature, Power, DRAM%, L1%)
- Unicode box-drawing characters for smooth lines
- Smart intersection detection (`┼├┤┬┴`)
- Matrix layout (2x2, 2x4, 4x4)
- Dynamic sizing based on terminal
- Different colors for each metric
- Legend with current values

---

## SHM (Shared Memory) Tracking

### Overview

Real-time memory allocation monitoring through persistent shared memory regions. Each physical chip gets its own unique SHM file that persists across process lifetimes.

### Unique Chip Identifiers

The system uses two types of IDs:

#### 1. Hardware ASIC ID (64-bit) - For Display
- **Source**: Read directly from chip firmware via UMD
- **Uniqueness**: Globally unique, burned into hardware
- **Example**: `0x251732099`
- **Usage**: Display in tt-smi, chip correlation

#### 2. Composite ASIC ID - For SHM File Naming
- **Formula**: `asic_id = (board_id << 8) | asic_location`
- **Example**: `/dev/shm/tt_device_469504_memory`
- **Usage**: SHM file naming

### SHM File Structure

Each chip's SHM file contains:
- **Aggregated statistics**: Total allocations across all processes
- **Per-chip statistics**: For remote device tracking
- **Per-process statistics**: PID, name, allocations by type

### Allocation Flow

```
Buffer Allocation
    └─> buffer->device() points to Device object
            └─> device->shm_stats_provider_ (unique per Device)
                    └─> record_allocation() updates specific SHM file
                            └─> /dev/shm/tt_device_<asic_id>_memory
```

### Buffer Types Tracked

| Buffer Type | Description | Location |
|------------|-------------|----------|
| `DRAM` | Main device memory | DRAM banks |
| `L1` | Fast on-chip memory | Tensix cores |
| `L1_SMALL` | Reserved L1 region | Tensix cores |
| `TRACE` | Trace buffer | DRAM |
| `CB` | Circular buffers | L1 |

**Note**: Kernel binaries are NOT tracked - they reside in reserved L1 region (KERNEL_CONFIG).

### Usage

#### Enable Tracking (Default)

```bash
# SHM tracking is enabled by default
python my_workload.py

# Check SHM files
ls -lh /dev/shm/tt_device_*
```

#### Disable Tracking (For Benchmarking)

```bash
export TT_METAL_SHM_TRACKING_DISABLED=1
python my_workload.py
```

### Automatic Dead Process Cleanup

- `tt-smi-ui` automatically detects and removes dead PIDs from all SHM files
- Uses `kill(pid, 0)` to check process liveness
- Scans `/dev/shm/tt_device_*_memory` on every refresh
- Enabled by default for accurate memory reporting

### Performance

- **Per allocation**: ~110-140ns overhead
- **First allocation per chip**: ~20-50μs (one-time setup)
- **Total for 500K allocations**: ~70ms (negligible)

---

## Remote Device Telemetry

### Architecture

`tt-smi-ui` monitors **both local (PCIe) and remote (Ethernet-connected)** devices:

```
Host (x86 CPU)
     │ PCIe
     ▼
Gateway Chip (Device 0)
     │ NOC → Ethernet Core
     │ Ethernet Link (100 Gbps)
     ▼
Remote Chip (Device 0R)
     │ Ethernet Core → NOC → ARC Firmware
     └─> Telemetry (Temperature, Power, AICLK)
```

### How It Works

1. **Initialization**: `TopologyDiscovery::discover()` finds all chips
2. **Telemetry Query**: `firmware_info->get_asic_temperature()` (automatic routing)
3. **Error Handling**: Adaptive retry (2s for offline, 15s for ETH busy)
4. **Display**: Unified view with 'R' suffix for remote chips

### Device Naming

| Display ID    | Meaning                              |
|---------------|--------------------------------------|
| `251732099`   | Local chip (PCIe-attached)           |
| `251732099R`  | Remote chip (via gateway 251732099)  |
| `T1:N5`       | Galaxy: Tray 1, Chip 5               |

### Performance

- **Local chips**: ~100 μs (direct PCIe)
- **Remote chips**: ~5-10 ms (PCIe → Ethernet → ARC)
- **Default refresh**: 500ms (no interference with inference)

---

## Testing

### Prerequisites

```bash
cd /path/to/tt-metal
./create_env.sh
source ./env_vars_setup.sh
./build_metal_with_flags.sh
pip install -e .
```

### Test 1: Monitor Idle Devices

```bash
# Terminal 1: Start monitoring
tt-smi-ui -w
```

Expected: All devices show 0 memory usage, telemetry working.

### Test 2: Monitor Workload

```bash
# Terminal 1: Start fast monitoring
tt-smi-ui -w -r 150

# Terminal 2: Run workload
export TT_VISIBLE_DEVICES=0
python my_workload.py
```

Expected: Memory allocations appear, AICLK increases, temperature rises.

### Test 3: Multi-Process

```bash
# Terminal 1: Monitor
tt-smi-ui -w

# Terminal 2: Run on Device 0
export TT_VISIBLE_DEVICES=0
python workload.py

# Terminal 3: Run on Device 1
export TT_VISIBLE_DEVICES=1
python workload.py
```

Expected: Two different PIDs visible, isolated to their devices.

### Test 4: Dead Process Cleanup

```bash
# Terminal 1: Monitor
tt-smi-ui -w

# Terminal 2: Start workload, then kill (Ctrl+C)
python workload.py
```

Expected: Within 500ms, dead PID detected, memory counters decrease to 0.

### Test 5: Graph Mode

```bash
# Terminal 1: Graph mode
tt-smi-ui -w -g

# Terminal 2: Run workload
python workload.py

# Resize terminal window
```

Expected: Graphs show temperature/power/DRAM/L1 history, adjust to window size.

---

## Example Output

### Table View

```
+================================================================================+
| tt-smi - Tenstorrent System Management Interface              Mon Jan 05 2026 |
+================================================================================+

ID          Arch          Temp    Power   AICLK     DRAM Usage          L1 Usage         Status
-----------------------------------------------------------------------------------------------------
251732099   Wormhole_B0   67°C    85W     1000MHz   2.3GiB / 12.0GiB    45.2MiB / 93MiB  OK
251732099R  Wormhole_B0   65°C    82W     1000MHz   2.3GiB / 12.0GiB    45.2MiB / 93MiB  OK
2517320d4   Wormhole_B0   45°C    22W     500MHz    0B / 12.0GiB        0B / 93MiB       OK
2517320d4R  Wormhole_B0   42°C    18W     500MHz    0B / 12.0GiB        0B / 93MiB       OK

Per-Process Memory Usage:
Dev          PID      Process       DRAM        L1         L1 Small    Trace       CB
------------------------------------------------------------------------------------------
251732099    12345    python        2.1GiB      40.0MiB    2.0MiB      200.0MiB    3.2MiB
251732099R   12345    python        2.1GiB      40.0MiB    2.0MiB      200.0MiB    3.2MiB
```

### Graph View

```
┌─ Device 251732099 (Wormhole_B0) ─────────────────────────────────────┐
│ 67°C  85W  1000MHz  DRAM [████████░░] 38%  L1 [██████░░░░] 23%      │
│                                                                       │
│ 100 ┤                                        ┌──── Temp C: 67 (62-67)│
│  75 ┤           ┌────┐                 ┌────┤ ── Power W: 85 (21-85)│
│  50 ┤      ┌────┘    └─────┐      ┌───┘    │ ── DRAM %: 38 (0-100) │
│  25 ┤ ┌────┘                └──────┘        │ ── L1 %: 23 (0-100)   │
│   0 └─┘                                     │                        │
└───────────────────────────────────────────────────────────────────────┘
```
