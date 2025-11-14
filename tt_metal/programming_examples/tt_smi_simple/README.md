# tt_smi_simple - Tenstorrent Chip Telemetry Monitor

A lightweight telemetry monitoring tool for Tenstorrent chips that displays real-time temperature, power, current, voltage, and clock frequency information directly from device firmware.

## Quick Start

### Build
```bash
cd tt-metal-apv
make tt_smi_simple
```

### Run
```bash
# Show current telemetry
./build/programming_examples/alexp_examples/memory_utilization_monitor/tt_smi_simple

# Watch mode with continuous updates
./build/programming_examples/alexp_examples/memory_utilization_monitor/tt_smi_simple -w

# Watch mode with 500ms refresh
./build/programming_examples/alexp_examples/memory_utilization_monitor/tt_smi_simple -w -r 500
```

## Output Format

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ tt-smi-simple v1.0                                                                    HH:MM:SS │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Chip  Name           Temp        Power    Curr    Volt    AICLK       HB    Status       │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 0     Wormhole_B0    38°C        12W      5A      850mV   500M        ●     OK           │
│ 1     Wormhole_B0    39°C        13W      6A      845mV   500M        ●     OK           │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Columns
- **Chip**: Device ID (0, 1, 2, etc.)
- **Name**: Chip architecture (Wormhole_B0, Blackhole, etc.)
- **Temp**: ASIC temperature in Celsius
- **Power**: Power consumption in Watts
- **Curr**: Current draw in Amperes
- **Volt**: Core voltage in millivolts
- **AICLK**: AI clock frequency in MHz
- **HB**: Heartbeat/status indicator
- **Status**: Device status

## Status Indicators

### Heartbeat (HB) Column
- **● Green**: Device responding normally
- **◐ Yellow**: Device busy (some telemetry may be N/A)
- **○ Yellow**: Limited telemetry available
- **● Blue**: Remote device detected

### Status Column
- **OK**: Device operating normally
- **Busy**: Device under load, telemetry may be incomplete
- **Remote**: Device is remote (limited monitoring)
- **Failed**: Device initialization failed

## Features

- **Direct UMD Access**: Reads telemetry straight from device firmware
- **No Memory Tracking**: Lightweight, doesn't monitor DRAM allocation
- **Retry Logic**: Automatically retries failed telemetry reads
- **Busy Detection**: Shows when device is under load and telemetry is incomplete
- **Watch Mode**: Continuous monitoring with configurable refresh rate
- **Local + Remote**: Works with both local PCIe and remote devices

## Troubleshooting

### "N/A" Values
- **During device operation**: Normal when device is busy processing
- **Shows "Busy" status**: Indicates device is under load
- **Telemetry temporarily unavailable**: Firmware prioritizes compute over monitoring

### Device Not Found
- Ensure devices are properly initialized
- Check if another process is using the device
- Try closing other monitoring tools

### Build Issues
- Ensure TT-Metal dependencies are installed
- Check CMake configuration
- Verify UMD libraries are available

## Command Line Options

```
Usage: tt_smi_simple [OPTIONS]

Options:
  -w, --watch    Watch mode (continuous updates)
  -r <ms>        Refresh interval in milliseconds (default: 1000)
  -h, --help     Show help message
```

## Examples

```bash
# One-time snapshot
./build/programming_examples/metal_example_tt_smi_simple

# Monitor every 2 seconds
./build/programming_examples/metal_example_tt_smi_simple -w -r 2000

# Fast monitoring for performance analysis
./build/programming_examples/metal_example_tt_smi_simple -w -r 50
```

## Technical Notes

- Telemetry validation ranges:
  - Temperature: -50°C to 100°C
  - Power: 0W to 300W
  - Current: 0A to 350A
  - Voltage: 0mV to 950mV
  - Clock: 0MHz to 1100MHz

- Uses TT-UMD (Universal Metal Device) APIs for direct firmware access
- Implements retry logic for reliable telemetry reading
- Detects device busy states to avoid false error reporting
