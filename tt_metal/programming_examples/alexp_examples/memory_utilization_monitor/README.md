# TT-SMI - Tenstorrent System Management Interface

Real-time monitoring tool for Tenstorrent devices with telemetry, memory tracking, and beautiful terminal UI.

## Quick Start

```bash
cd tt_metal/programming_examples/alexp_examples/memory_utilization_monitor
pip install -e .

# Run
tt-smi-ui -w            # Watch mode (live updates)
tt-smi-ui -w -g         # Graph mode (nvtop-style)
tt-smi-ui -w -r 500     # Fast refresh (500ms)
```

## Features

- **Telemetry**: Temperature, Power, AICLK, Voltage, Current
- **Memory Tracking**: DRAM, L1, L1_SMALL, Trace, CB allocations
- **Per-Process Breakdown**: See which PID is using memory
- **Graph Mode**: nvtop-style visualization with temperature, power, and memory charts
- **Multi-Device Support**: Local (PCIe) and remote (Ethernet) devices
- **Auto Cleanup**: Removes dead process entries automatically

## Installation

### Requirements

TT-Metal must be built first:

```bash
# Build TT-Metal
cd /path/to/tt-metal
./build_metal_with_flags.sh

# Install tt-smi-ui
cd tt_metal/programming_examples/alexp_examples/memory_utilization_monitor
pip install -e .
```

## Usage

```bash
# Single snapshot
tt-smi-ui

# Watch mode (updates every 500ms)
tt-smi-ui -w

# Graph mode (nvtop-style)
tt-smi-ui -w -g

# Fast refresh (150ms)
tt-smi-ui -w -r 150

# SHM-only (non-invasive, no telemetry)
tt-smi-ui --shm-only

# JSON output
tt-smi-ui --json

# Device reset
tt-smi-ui --reset
```

## Python API

```python
from tt_smi_ui import get_devices, update_telemetry_parallel

# Get all devices
devices = get_devices()
update_telemetry_parallel(devices, timeout=1.0)

for dev in devices:
    print(f"Device {dev.display_id}:")
    print(f"  Temp: {dev.temperature}°C")
    print(f"  Power: {dev.power}W")
    print(f"  DRAM: {dev.used_dram}/{dev.total_dram}")
    print(f"  L1: {dev.used_l1}/{dev.total_l1}")
```

## Memory Tracking (SHM)

Memory allocations are automatically tracked via shared memory files (`/dev/shm/tt_device_*_memory`).

### Tracked Buffer Types

- **DRAM**: Main device memory
- **L1**: Fast on-chip memory
- **L1_SMALL**: Reserved L1 region
- **TRACE**: Trace buffers
- **CB**: Circular buffers

### Disable Tracking

```bash
export TT_METAL_SHM_TRACKING_DISABLED=1
python my_workload.py
```

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

Per-Process Memory Usage:
Dev          PID      Process       DRAM        L1         L1 Small    Trace       CB
------------------------------------------------------------------------------------------
251732099    12345    python        2.1GiB      40.0MiB    2.0MiB      200.0MiB    3.2MiB
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

## Troubleshooting

### No devices found

```bash
# Check SHM files
ls -lh /dev/shm/tt_device_*

# Run simple Metal operation to create SHM files
python -c "import ttnn; device = ttnn.open_device(0); ttnn.close_device(device)"

# Try SHM-only mode
tt-smi-ui --shm-only
```

### Clear stale memory data

```bash
rm /dev/shm/tt_device_*_memory
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TT_METAL_SHM_TRACKING_DISABLED` | `0` | Set to `1` to disable memory tracking |
| `TT_VISIBLE_DEVICES` | all | Comma-separated list of device IDs |

## Performance

- **SHM tracking overhead**: ~110-140ns per allocation
- **Telemetry query**: 100μs (local), 5-10ms (remote)
- **Default refresh**: 500ms (no interference with workloads)
