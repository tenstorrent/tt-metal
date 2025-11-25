# TT-Metal Physical Topology & Ethernet Link Visualizer

Terminal-based tool for visualizing the physical topology of TT-Metal chips and checking ethernet link connectivity in a matrix format.

## Features

- **Chip Discovery**: Automatically discovers all chips in the cluster
- **Topology Matrix**: Displays a matrix showing ethernet connections between chips
- **Link Status**: Shows topology-defined connections (CONNECTED/NOT_CONNECTED)
- **Color-Coded**: Uses color coding for easy identification
  - 🟢 Green: All links defined in topology
  - 🟡 Yellow: Partial links defined
  - 🔴 Red: No links defined
- **Detailed View**: Lists each ethernet channel and its connection status
- **Summary Statistics**: Provides overview of total chips, links, and connectivity percentage

## Building

From the tt-metal build directory:

```bash
cmake --build . --target topology_viz
```

Or build all alexp_examples:

```bash
cmake --build . --target alexp_examples
```

## Running

```bash
./build/programming_examples/topology_viz
```

The tool will:
1. Initialize the cluster
2. Discover all chips
3. Check topology-defined ethernet links
4. Display multiple visualizations

## Output Format

### 1. Chip Details
Shows each chip's ID, architecture, board type, and number of ethernet channels.

### 2. Topology Matrix
A matrix where:
- Rows represent source chips
- Columns represent destination chips
- Numbers indicate the count of ethernet links
- Colors indicate if links are defined in topology

### 3. Detailed Link Status
Lists each ethernet channel with:
- Source chip and channel
- Destination chip and channel
- Link status (CONNECTED/NOT_CONNECTED based on topology)

### 4. Summary
- Total chip count
- Total ethernet links
- Links defined/not defined
- Overall topology connectivity percentage

## Example Output

```
╔═══════════════════════════════════════════════════════════╗
║  TT-Metal Physical Topology & Ethernet Link Visualizer  ║
╚═══════════════════════════════════════════════════════════╝

Initializing cluster...
Discovering topology...

╔══════════════════════════════════════════════════════════════╗
║              CHIP DETAILS                                    ║
╚══════════════════════════════════════════════════════════════╝

 Chip ID   Architecture     Board Type       Eth Channels
----------------------------------------------------------
       0    Wormhole B0           N300                  2
       1    Wormhole B0           N300                  2

╔══════════════════════════════════════════════════════════════╗
║              ETHERNET LINK TOPOLOGY MATRIX                   ║
╚══════════════════════════════════════════════════════════════╝

          0     1
    -------------
  0 |     -     2
  1 |     2     -

Legend: ■ All links defined  ■ Partial  ■ Not defined  . No connection
Numbers indicate count of ethernet links between chips
Note: Shows topology from cluster descriptor. Use 'system_health' for runtime link status.
```

## Topology vs Runtime Status

This tool shows **topology-defined connections** from the cluster descriptor:
- **CONNECTED**: Link is defined in the cluster topology
- **NOT_CONNECTED**: No link defined for this channel

For **runtime link status** (UP/DOWN after device initialization), use:
```bash
./build/tools/umd/system_health
```

Runtime status requires devices to be initialized by running a workload first.

## Architecture Support

- **Wormhole B0**: Full support
- **Blackhole**: Full support
- **Grayskull**: Limited ethernet support

## Requirements

- TT-Metal environment
- TT-UMD access
- Valid cluster descriptor
- **No device initialization required** (unlike runtime status tools)

## Troubleshooting

### No chips found
- Ensure devices are properly detected
- Check that TT-SMI can see the devices
- Verify cluster descriptor is valid

### All links show NOT_CONNECTED
- Check cluster descriptor file path
- Verify ethernet connections are defined in YAML
- Ensure cluster descriptor matches hardware

### Compilation errors
- Ensure TT-Metalium is properly installed
- Check that CMake can find TT::Metalium package
- Verify C++17 compiler is available

## Technical Details

### Link Status Detection

The tool reads link definitions from the cluster descriptor:
- Uses `ClusterDescriptor::ethernet_core_has_active_ethernet_link()` to check if links are defined
- Does not read device firmware (no initialization needed)
- Shows topology connectivity, not runtime status

### UMD APIs Used

- `Cluster::get_cluster_description()`: Get cluster topology
- `ClusterDescriptor::get_ethernet_connections()`: Get ethernet links
- `ClusterDescriptor::ethernet_core_has_active_ethernet_link()`: Check if link is defined
- `Cluster::get_soc_descriptor()`: Get SoC information

## Future Enhancements

- Real-time monitoring mode with refresh (requires runtime status)
- Export topology to file (DOT, JSON)
- Integration with system_health for runtime status
- Bandwidth/throughput monitoring
- Interactive mode with ncurses

## Related Tools

- `system_health`: Runtime link status (UP/DOWN) after initialization
- `tt_smi_umd`: Device monitoring with memory/telemetry
- `allocation_monitor_client`: Memory allocation tracking
