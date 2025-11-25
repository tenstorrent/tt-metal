# Topology Visualizer - Project Summary

## Overview

A terminal-based C++ tool for visualizing TT-Metal physical topology and checking ethernet link connectivity from the cluster descriptor. The tool provides a matrix-based visualization showing which links are defined in the topology between chips.

## Project Structure

```
topology_viz/
├── topology_viz.cpp         # Main C++ source file
├── CMakeLists.txt           # Build configuration
├── README.md                # User documentation
├── EXAMPLE_OUTPUT.md        # Example outputs
├── PROJECT_SUMMARY.md       # This file
└── build_and_run.sh         # Build and run script
```

## Features Implemented

### 1. Chip Discovery
- Automatically detects all chips in the cluster
- Retrieves chip architecture (Wormhole B0, Blackhole, etc.)
- Identifies board type (N300, P150, etc.)
- Counts available ethernet channels per chip

### 2. Topology Matrix Visualization
- Matrix display showing chip-to-chip connections
- Color-coded link definition status:
  - 🟢 Green: All links defined in topology
  - 🟡 Yellow: Partial links defined
  - 🔴 Red: No links defined
  - Gray: No connection
- Numbers indicate link count between chips

### 3. Ethernet Link Connectivity
- Reads link definitions from cluster descriptor
- Uses `ethernet_core_has_active_ethernet_link()` API
- Per-channel connectivity checking
- Shows CONNECTED/NOT_CONNECTED based on topology

### 4. Detailed Link Reporting
- Lists every ethernet channel connection
- Shows source and destination chip/channel pairs
- Individual link status for troubleshooting
- Organized by source chip

### 5. Summary Statistics
- Total chip count
- Total ethernet link count
- Links defined vs not defined counts
- Overall topology connectivity percentage
- Guidance to use `system_health` for runtime status

## Technical Implementation

### Key Components

1. **ChipTopology Structure**
   - Stores discovered chips and their properties
   - Maintains outgoing link map
   - Tracks chip architectures and board types

2. **Link Status Detection**
   - Uses ClusterDescriptor API to check topology definitions
   - No device firmware reads required (no initialization needed)
   - Checks `ethernet_core_has_active_ethernet_link()` for each channel

3. **Topology Discovery**
   - Queries ClusterDescriptor for ethernet connections
   - Checks each channel for active links
   - Builds connectivity graph

4. **Visualization**
   - ANSI color codes for terminal output
   - Box-drawing characters for clean UI
   - Formatted tables with alignment

### UMD APIs Used

```cpp
// Get cluster descriptor
cluster->get_cluster_description()

// Get ethernet connections map
cluster_desc->get_ethernet_connections()

// Check if link is defined in topology
cluster_desc->ethernet_core_has_active_ethernet_link(chip_id, channel)

// Get remote endpoint
cluster_desc->get_chip_and_channel_of_remote_ethernet_core(chip_id, channel)

// Get SoC descriptor for chip information
cluster->get_soc_descriptor(chip_id)
```

## Build System Integration

### CMakeLists.txt
- Targets TT::Metalium package
- C++17 standard
- Links against pthread and stdc++fs
- Installs to bin/ directory

### Parent CMakeLists.txt
Updated to include topology_viz subdirectory:
```cmake
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/topology_viz)
```

## Usage

### Quick Start
```bash
cd /path/to/tt-metal
./tt_metal/programming_examples/alexp_examples/topology_viz/build_and_run.sh
```

### Manual Build
```bash
cd /path/to/tt-metal
cmake --build build --target topology_viz
./build/bin/topology_viz
```

### Prerequisites
- TT-Metal environment set up
- Valid cluster descriptor
- TT devices detected (no initialization required)
- Sufficient permissions for device access

## Code Quality

### Design Principles
- Clear function names (snake_case)
- Minimal nesting through early returns
- Const-correctness where applicable
- Exception handling for device access
- No external dependencies beyond TT-Metal/UMD

### Error Handling
- Try-catch blocks around cluster access
- Graceful handling of missing topology definitions
- Clear error messages with color coding
- Returns LinkStatus::NOT_CONNECTED when topology check fails

### Performance
- Single-pass topology discovery
- No device firmware reads (fast execution)
- Efficient data structures (std::map, std::vector)
- No repeated lookups or device access

## Future Enhancements

### Planned Features
1. **Real-time Monitoring Mode**
   - Continuous refresh with ncurses
   - Historical link status tracking
   - Flapping detection

2. **Export Capabilities**
   - DOT format for Graphviz
   - JSON for programmatic parsing
   - HTML report generation

3. **Advanced Metrics**
   - Link retrain counts
   - CRC error rates
   - Bandwidth utilization
   - Latency measurements

4. **Interactive Mode**
   - Select chip for detailed view
   - Filter by link status
   - Sort options
   - Search functionality

5. **Configuration Options**
   - Command-line arguments for output format
   - Specify target chips
   - Export options
   - Verbosity levels

## Known Limitations

1. **Topology Only**: Shows link definitions from cluster descriptor, not runtime UP/DOWN status. Use `system_health` for runtime status.

2. **Architecture Support**: Tested primarily on Wormhole B0 and Blackhole. Other architectures should work but may need validation.

3. **Snapshot Mode**: Shows topology at one point in time. No continuous monitoring yet.

4. **Single Cluster**: Designed for local clusters. Remote clusters via telemetry not yet supported.

5. **No Runtime Status**: Does not initialize devices or check firmware link status. This is by design for fast, initialization-free topology visualization.

## Testing Recommendations

### Unit Tests
- Test topology discovery with mock cluster
- Verify link status interpretation per architecture
- Test matrix rendering with various chip counts
- Validate summary statistics calculation

### Integration Tests
- Test on single N300 board (2 chips)
- Test on 4-chip mesh configuration
- Test with failed links
- Test on multi-board systems

### Edge Cases
- Single chip (no links)
- All links down
- Mixed architectures
- Large clusters (16+ chips)

## Documentation

- **README.md**: User-facing documentation
- **EXAMPLE_OUTPUT.md**: Visual examples of tool output
- **PROJECT_SUMMARY.md**: This technical overview
- **Code Comments**: Inline documentation in source

## Integration with Other Tools

### Complementary Tools
- **tt_smi_umd**: Device memory and telemetry monitoring
- **system_health**: UMD-level health checking
- **allocation_monitor**: Memory tracking

### Use in Workflows
- Pre-deployment verification
- Continuous health monitoring
- Troubleshooting guides
- System documentation

## Conclusion

The Topology Visualizer provides a clean, terminal-based interface for understanding TT-Metal cluster topology and ethernet link connectivity from the cluster descriptor. It leverages UMD APIs to read topology definitions and presents the information in an intuitive matrix format with color-coded connectivity indicators.

The tool is designed to be simple, fast, and informative - suitable for quick topology verification, documentation, and understanding cluster layout without requiring device initialization. For runtime link status monitoring (UP/DOWN), use the complementary `system_health` tool.
