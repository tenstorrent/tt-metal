# TT Device Memory Utilization Monitor

A real-time C++ tool for monitoring memory utilization status on Tenstorrent (TT) devices. This tool provides a comprehensive view of device information and memory usage with color-coded visualizations and real-time updates.

## Available Versions

This package includes three different versions of the memory monitor:

1. **memory_monitor_minimal** - Basic device detection and information display (recommended for initial testing)
2. **memory_monitor_simple** - Enhanced version with device initialization and detailed memory information
3. **memory_monitor** - Full-featured version with detailed memory statistics (may have compilation issues due to internal API dependencies)

## Features

- **Real-time Monitoring**: Live updates of memory utilization across all available TT devices
- **Multi-Buffer Support**: Monitors DRAM, L1, L1_SMALL, and TRACE memory types
- **Visual Indicators**: Color-coded memory bars and utilization percentages
- **Multi-Device Support**: Automatically detects and monitors all available TT devices
- **Configurable Refresh Rate**: Adjustable update interval (minimum 100ms)
- **Clean Terminal Interface**: ANSI color-coded output with clear formatting
- **Graceful Shutdown**: Ctrl+C handling for clean exit

## Memory Types Monitored

- **DRAM**: Main device memory for large data storage
- **L1**: Level 1 cache memory for compute operations
- **L1_SMALL**: Small L1 memory region for specific operations
- **TRACE**: Memory region for debugging and tracing

## Build Instructions

### Prerequisites

- TT-Metal build environment
- CMake 3.22 or later
- C++17 compatible compiler

### Building

The tool is automatically built as part of the TT-Metal programming examples:

```bash
# Build all programming examples (including memory monitor)
cmake -S . -B build-cmake -DTT_METAL_BUILD_PROGRAMMING_EXAMPLES=ON
cmake --build build-cmake --target programming_examples -j

# Or build just the memory monitor
cmake --build build-cmake --target memory_monitor -j
```

### Build Output

The executables will be located at:
```
build-cmake/programming_examples/alexp_examples/memory_utilization_monitor/
├── memory_monitor_minimal    # Recommended for basic monitoring
├── memory_monitor_simple     # Enhanced version with device details
└── memory_monitor           # Full-featured version (may have issues)
```

## Usage

### Basic Usage

```bash
# Use the minimal version (recommended)
./memory_monitor_minimal

# Use the simple version for more details
./memory_monitor_simple

# Use the full version (may have compilation issues)
./memory_monitor

# All versions support the same command line options:
# Monitor with custom refresh rate (500ms)
./memory_monitor_minimal -r 500

# Monitor with 2-second refresh rate
./memory_monitor_minimal --refresh 2000
```

### Command Line Options

- `-r, --refresh <ms>`: Set refresh interval in milliseconds (default: 1000, minimum: 100)
- `-h, --help`: Show help message and exit

### Examples

```bash
# Quick monitoring with 500ms updates
./memory_monitor_minimal -r 500

# Slower monitoring for detailed analysis
./memory_monitor_simple --refresh 5000

# Show help
./memory_monitor_minimal --help
```

## Output Format

The tool displays information in the following format:

```
================================================================================
|                    TT Device Memory Utilization Monitor                     |
================================================================================
Press Ctrl+C to exit

System Info:
  Time: 2025-01-27 14:30:15
  Refresh: 1000ms
  Devices: 2

Device 0 (ID: 0)
--------------------------------------------------
  DRAM Memory:
    Banks: 8
    Total:         2.00 GB
    Allocated:     1.25 GB (62.5%)
    Free:          750.00 MB
    Largest:       500.00 MB
    Usage:         [████████████████████░░░░░░░░░░] 62.5%

  L1 Memory:
    Banks: 64
    Total:         32.00 MB
    Allocated:     20.50 MB (64.1%)
    Free:          11.50 MB
    Largest:       8.00 MB
    Usage:         [████████████████████░░░░░░░░░░] 64.1%

  L1_SMALL Memory:
    Banks: 64
    Total:         2.00 MB
    Allocated:     1.20 MB (60.0%)
    Free:          800.00 KB
    Largest:       400.00 KB
    Usage:         [████████████████████░░░░░░░░░░] 60.0%

  TRACE Memory:
    Banks: 1
    Total:         16.00 MB
    Allocated:     4.50 MB (28.1%)
    Free:          11.50 MB
    Largest:       11.50 MB
    Usage:         [████████░░░░░░░░░░░░░░░░░░░░░░] 28.1%
```

## Color Coding

- **Green**: Low memory utilization (< 75%)
- **Yellow**: Medium memory utilization (75-89%)
- **Red**: High memory utilization (≥ 90%)

## Error Handling

The tool includes comprehensive error handling for:

- Device initialization failures
- Memory access errors
- Invalid command line arguments
- Device disconnection

## Technical Details

### Memory Statistics

For each buffer type, the tool reports:

- **Total**: Total allocatable memory across all banks
- **Allocated**: Currently allocated memory
- **Free**: Available free memory
- **Largest**: Size of the largest contiguous free block
- **Utilization**: Percentage of memory currently in use

### Performance Considerations

- Minimum refresh rate is 100ms to prevent excessive CPU usage
- Memory statistics are retrieved directly from device allocators
- No memory overhead for monitoring (read-only operations)

## Troubleshooting

### Common Issues

1. **No devices found**: Ensure TT devices are properly connected and drivers are loaded
2. **Permission denied**: Run with appropriate permissions to access TT devices
3. **Build errors**: Ensure TT-Metal is properly built with programming examples enabled

### Debug Information

The tool provides detailed error messages for common issues:

```
Error: No TT devices available
Error initializing devices: [specific error message]
Error reading DRAM memory: [specific error message]
```

## Integration

This tool can be integrated into:

- Performance monitoring scripts
- Automated testing frameworks
- Development workflows
- Production monitoring systems

## License

SPDX-License-Identifier: Apache-2.0

## Contributing

This tool is part of the TT-Metal programming examples. Contributions should follow the TT-Metal contribution guidelines.
