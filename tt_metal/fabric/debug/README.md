# Fabric Debug Tools

This directory contains debugging utilities for the fabric EDM (Ethernet Data Movement) system on Tenstorrent devices. The tools help analyze fabric router binaries, inspect ERISC core registers, and debug fabric flow control issues.

## dump_erisc_values.py

A comprehensive tool for reading and monitoring values from ERISC (Ethernet RISC) cores across Tenstorrent devices. Designed specifically for debugging fabric EDM flow control and routing issues.

### Features

- **Multiple Operation Modes**:
  - Single snapshot: Read register values once
  - Polling mode: Monitor register changes over time with delta analysis
  - Buffer mode: Read and analyze circular buffer contents
  - Fabric stream register mode: Display fabric flow control registers in matrix format

- **Device and Core Filtering**:
  - Filter by device IDs and core coordinates
  - Automatic detection of active vs idle cores
  - Reset status checking for core accessibility

- **Multiple Output Formats**: JSON, CSV, and formatted console output
- **Architecture Support**: Wormhole and Blackhole devices with automatic detection
- **Flow Control Debugging**: Specialized fabric stream register monitoring

### Usage Examples

```bash
# Basic snapshot of default ERISC registers (reset status, wall clocks)
python3 tt_metal/fabric/debug/dump_erisc_values.py

# Monitor fabric stream registers for flow control debugging
python3 tt_metal/fabric/debug/dump_erisc_values.py --fabric-streams --poll --duration 10

# Check specific addresses on device 0, cores (0,5) and (0,7)
python3 tt_metal/fabric/debug/dump_erisc_values.py --addresses 0xFFB121B0,0xFFB121F0 --devices 0 --cores 0,5,0,7

# Continuous monitoring with CSV export for analysis
python3 tt_metal/fabric/debug/dump_erisc_values.py --poll --csv --output fabric_debug.csv --duration 30 --interval 0.5

# Analyze buffer contents (useful for EDM circular buffers)
python3 tt_metal/fabric/debug/dump_erisc_values.py --buffer-mode --addresses 0x12345678 --num-elements 4 --slot-size 64

# Monitor only sender channels with changes-only output
python3 tt_metal/fabric/debug/dump_erisc_values.py --fabric-streams --stream-group sender_free_slots --poll --changes-only
```

### Fabric Stream Register Modes

The tool provides specialized fabric stream register modes for flow control debugging:

- **`sender_free_slots`**: Monitor sender channel buffer space (streams 17-21)
- **`receiver_free_slots`**: Monitor receiver channel buffer space (streams 12-16)
- **`all_fabric_free_slots`**: Complete view of all fabric flow control registers

Example matrix output:
```
=== FABRIC STREAM REGISTERS ===
ALL FABRIC STREAM FREE SLOTS:
(Complete view of all fabric EDM sender/receiver buffer space for flow control debugging)

                    Stream12   Stream13   Stream14   Stream15   Stream16   Stream17   Stream18   Stream19   Stream20   Stream21
Dev0 Core(0,5):     0x00001234 0x00001235 0x00001236 0x00001237 0x00001238 0x00001239 0x0000123a 0x0000123b 0x0000123c 0x0000123d
Dev0 Core(0,7):     0x00001244 0x00001245 0x00001246 0x00001247 0x00001248 0x00001249 0x0000124a 0x0000124b 0x0000124c 0x0000124d
```

### Requirements

- Python 3.7+
- ttexalens library (Tenstorrent debugging framework)
- Access to Tenstorrent devices

### Command Line Options

#### Basic Options
- `--addresses`: Comma-separated hex addresses to read
- `--output`: Output file path (default: stdout)
- `--json`: Output in JSON format
- `--decimal`: Show values in decimal instead of hex

#### Filtering Options
- `--devices`: Comma-separated device IDs (default: all devices)
- `--cores`: Core coordinates as x1,y1,x2,y2,... (default: all cores)
- `--include-idle`: Include idle ethernet cores (default: active only)
- `--skip-reset-check`: Skip core reset status checking

#### Polling Mode
- `--poll`: Enable continuous monitoring
- `--interval`: Polling interval in seconds (default: 0.1)
- `--duration`: Total polling duration (default: 10.0)
- `--csv`: Output in CSV format for analysis
- `--changes-only`: Only show values that changed
- `--no-delta-summary`: Skip final delta summary

#### Buffer Mode
- `--buffer-mode`: Treat addresses as buffer starts
- `--num-elements`: Buffer elements (default: 4)
- `--slot-size`: Slot size in bytes (default: 64)
- `--data-per-slot`: Data per slot in bytes (default: 16)

#### Fabric Stream Mode
- `--fabric-streams`: Enable fabric register matrix mode
- `--stream-group`: Register group to display

## fabric_binary_analyzer.py

A utility to analyze binary sizes of `fabric_erisc_router` kernels from the tt-metal cache.

### Features

- **Focused Analysis**: Analyzes only `fabric_erisc_router` binaries
- **Comprehensive Statistics**: Min, max, mean, median, and 95th percentile for text/data sections
- **ELF Analysis**: Extracts section sizes using `readelf` from load segments
- **Build Tracking**: Groups binaries by build hash with configuration counts
- **Clean Interface**: Simple command-line interface with clear output

### Usage

```bash
# Basic analysis with statistics summary
python3 tt_metal/fabric/debug/fabric_binary_analyzer.py

# Include detailed per-binary breakdown
python3 tt_metal/fabric/debug/fabric_binary_analyzer.py --detailed

# Use custom cache directory
python3 tt_metal/fabric/debug/fabric_binary_analyzer.py --cache-dir /path/to/cache
```

### Example Output

```
Found 100 fabric_erisc_router ELF binaries, analyzing...

FABRIC ERISC ROUTER BINARY SIZE ANALYSIS
Total fabric_erisc_router binaries analyzed: 100

SIZE STATISTICS:
Section      Minimum    Maximum    Mean       Median     95th %ile
Text Size    7.4 KB     10.3 KB    8.8 KB     9.0 KB     10.0 KB
Data Size    68 B       152 B      104 B      88.0 B     152 B
Total Size   7.4 KB     10.3 KB    8.9 KB     9.1 KB     10.1 KB

BINARIES BY BUILD:
Build Hash: 1a6f17ff97 (100 binaries, 100 unique configs, avg: 8.9 KB)
```

### Requirements

- Python 3.6+
- `readelf` utility (from binutils package)
- Compiled fabric router kernels in cache directory

### Technical Details

- Analyzes `fabric_erisc_router` ELF binaries in `~/.cache/tt-metal-cache/`
- Text section: executable RISC-V instructions
- Data section: combined `.data` and `.bss` sections
- Statistics: min/max/mean/median/95th percentile across all binaries
- Each kernel hash represents a unique router configuration

## erisc_debug_constants.py

Hardware constants and configuration data for ERISC debugging, including:

- **ERISC Register Addresses**: Common registers for core status and wall clocks
- **Stream Register Constants**: Architecture-specific indices and masking
- **Fabric Stream Groups**: Logical groupings of fabric flow control registers
- **Architecture Mapping**: Device architecture detection and normalization

### Key Constants

```python
# Default ERISC registers for basic core health checks
DEFAULT_ADDRESSES = [
    0xFFB121B0,  # ETH_RISC_RESET - Check if core is out of reset
    0xFFB121F0,  # ETH_RISC_WALL_CLOCK_0 - Monitor core activity
    0xFFB121F4,  # ETH_RISC_WALL_CLOCK_1 - High bits of wall clock
]

# Fabric stream groups for flow control debugging
FABRIC_STREAM_GROUPS = {
    "sender_free_slots": {"stream_ids": [17, 18, 19, 20, 21], ...},
    "receiver_free_slots": {"stream_ids": [12, 13, 14, 15, 16], ...},
    "all_fabric_free_slots": {"stream_ids": [12, 13, 14, 15, 16, 17, 18, 19, 20, 21], ...},
}
```

## erisc_debug_utils.py

Utility functions providing a clean interface for ERISC debugging operations:

- **`get_stream_reg_address()`**: Calculate physical addresses for stream registers
- **`normalize_architecture()`**: Device architecture detection and normalization
- **`get_fabric_stream_addresses()`**: Get all addresses for a fabric register group
- **`detect_device_architecture()`**: Auto-detect device architecture from ttexalens
- **`parse_core_key()`**: Parse core coordinate strings

### Usage Example

```python
from erisc_debug_utils import get_fabric_stream_addresses, detect_device_architecture

# Get all fabric sender channel addresses for wormhole
addresses = get_fabric_stream_addresses("sender_free_slots", "BUF_SPACE_AVAILABLE", "wormhole")

# Auto-detect device architecture
arch = detect_device_architecture(device)
```

## Troubleshooting

### Common Issues

1. **No devices found**: Ensure ttexalens is properly initialized and devices are accessible
2. **No ethernet cores found**: Check device configuration and core filtering options
3. **Read failures**: Cores may be in reset or idle - try `--include-idle` and `--skip-reset-check`
4. **Architecture detection fails**: Manually specify architecture or check device type support

### Debugging Flow Control Issues

1. **Start with fabric streams**: Use `--fabric-streams` to get overview of all channels
2. **Monitor over time**: Add `--poll` to see register changes during operations
3. **Focus on specific channels**: Use `--stream-group` to narrow down problematic channels
4. **Export for analysis**: Use `--csv` output for offline analysis and visualization
5. **Check for blockages**: Look for zero or very low values in free slot registers

### Performance Tips

- Use `--devices` and `--cores` to limit scope and improve performance
- Set reasonable `--interval` values (0.1s default is usually sufficient)
- Consider `--changes-only` for long-running monitoring to reduce output noise
