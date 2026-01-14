# Fabric Router Tests

## Overview

This directory contains tests for the TT-Metal fabric router functionality, including routing table configuration, topology management, connection mapping, and router control features.

## Test Categories

### Router Configuration Tests

Tests for routing table setup and configuration:

- `test_routing_tables.cpp` - Basic routing table functionality
- `test_custom_routing_tables.cpp` - Custom routing table configurations
- `test_router_archetypes.cpp` - Router archetype definitions
- `test_router_channel_mapping.cpp` - Channel mapping logic

### Topology and Connection Tests

Tests for fabric topology discovery and connection management:

- `test_topology_mapper.cpp` - Topology mapping functionality
- `test_topology_mapper_utils.cpp` - Topology utility functions
- `test_mesh_graph_descriptor.cpp` - Mesh graph descriptors
- `test_connection_registry.cpp` - Connection registry management
- `test_router_connections.cpp` - Router connection setup
- `test_router_connection_mapping.cpp` - Connection mapping logic
- `test_connection_mapping_logic.cpp` - Advanced connection mapping
- `test_fabric_builder_local_connections.cpp` - Local connection builder
- `test_fabric_topology_helpers.cpp` - Topology helper functions

### Multi-Host Tests

Tests for multi-host fabric configurations:

- `test_multi_host.cpp` - Multi-host fabric scenarios

### Integration Tests

End-to-end integration tests:

- `test_z_router_integration.cpp` - Router integration scenarios
- `test_z_router_device_detection.cpp` - Device detection integration

### Router Control Tests

Tests for runtime router control and state management:

- **`test_fabric_router_pause_control.cpp`** - **Router pause/resume functionality**

## Pause/Resume Feature Documentation

The fabric router pause/resume feature provides runtime control over packet forwarding through the fabric network. This is a critical capability for safe fabric state management, debugging, and testing.

### Quick Links

- **[Pause/Resume Main README](README_PAUSE_RESUME.md)** - Feature overview, API guide, and usage instructions
- **[Architecture Documentation](ARCHITECTURE.md)** - System design, component interactions, and diagrams
- **[API Reference](API_REFERENCE.md)** - Complete API documentation for all components
- **[Usage Guide](USAGE_GUIDE.md)** - Practical examples, patterns, and best practices

### Key Features

- **Runtime Traffic Control**: Pause and resume packet forwarding on demand
- **State Management**: Query router state across the fabric
- **Traffic Validation**: Telemetry-based detection of traffic flow
- **Worker Management**: Utilities for launching and managing traffic generators
- **Observability**: Comprehensive logging and state inspection

### Quick Example

```cpp
#include "tests/tt_metal/tt_fabric/common/fabric_command_interface.hpp"

auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
test_utils::FabricCommandInterface cmd_interface(control_plane);

// Pause all routers
cmd_interface.pause_routers();
bool success = cmd_interface.wait_for_pause();

// Resume routers
cmd_interface.resume_routers();
cmd_interface.wait_for_state(RouterStateCommon::RUNNING);
```

### Test Files

The pause/resume test suite is located at:
- `test_fabric_router_pause_control.cpp` - Main test file with 9 test cases

### Utility Components

Supporting utilities in `tests/tt_metal/tt_fabric/common/`:
- `fabric_command_interface.{hpp,cpp}` - High-level router control API
- `fabric_traffic_validation.{hpp,cpp}` - Traffic detection via telemetry
- `fabric_router_state_utils.{hpp,cpp}` - State query and logging
- `fabric_worker_kernel_helpers.{hpp,cpp}` - Worker kernel management
- `fabric_traffic_generator_defs.hpp` - Shared constants and types

### Running Pause/Resume Tests

```bash
# Run all fabric tests including pause/resume
./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=*Pause*

# Run specific pause/resume test
./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=*PauseStopsTraffic*

# Run with slow dispatch for consistent timing
TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=*Pause*
```

## Build Integration

All tests in this directory are built as part of the `fabric_unit_tests` executable. The test is registered in `tests/tt_metal/tt_fabric/CMakeLists.txt`.

### CMake Configuration

```cmake
set(UNIT_TESTS_FABRIC_SRC
    # ... other tests ...
    ${CMAKE_CURRENT_SOURCE_DIR}/fabric_router/test_fabric_router_pause_control.cpp
    # ... utility sources ...
)

add_executable(fabric_unit_tests ${UNIT_TESTS_FABRIC_SRC})
```

## Hardware Requirements

Most tests require:
- At least 2 devices in fabric topology
- Devices must support fabric protocol
- Telemetry enabled in firmware (for pause/resume tests)

## Test Execution

### Run All Fabric Router Tests

```bash
# Build tests
./build_metal.sh

# Run all fabric router tests
./build/test/tt_metal/tt_fabric/fabric_unit_tests
```

### Run Specific Test Categories

```bash
# Routing tests
./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=*Routing*

# Topology tests
./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=*Topology*

# Connection tests
./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=*Connection*

# Pause/resume tests
./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=*Pause*:*Resume*
```

### Skip Conditions

Tests automatically skip if:
- Insufficient devices available
- Required hardware features not present
- Telemetry not available (for pause/resume tests)

## Contributing

When adding new fabric router tests:

1. Add test source file to this directory
2. Update `CMakeLists.txt` to include new test
3. Follow existing naming conventions (`test_*.cpp`)
4. Use appropriate fixture base class (`Fabric1DFixture`, etc.)
5. Add clear test documentation
6. Include skip conditions for missing hardware
7. Update this README if adding a new test category

## Related Documentation

- Fabric Control Plane: `tt_metal/fabric/control_plane.hpp`
- Fabric Test Fixtures: `tests/tt_metal/tt_fabric/common/fabric_fixture.hpp`
- Fabric Data Movement Tests: `tests/tt_metal/tt_fabric/fabric_data_movement/`
- Router State Manager: Core control plane component

## Support

For questions or issues:

- Check the pause/resume documentation (links above)
- Review existing test code for examples
- Consult the architecture documentation for system design
- Check the troubleshooting section in README_PAUSE_RESUME.md
