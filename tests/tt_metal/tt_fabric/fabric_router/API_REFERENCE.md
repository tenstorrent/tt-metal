# Fabric Router Pause/Resume API Reference

## Table of Contents

- [Overview](#overview)
- [FabricCommandInterface](#fabriccommandinterface)
- [Traffic Validation API](#traffic-validation-api)
- [Router State Utilities](#router-state-utilities)
- [Worker Kernel Helpers](#worker-kernel-helpers)
- [Constants and Types](#constants-and-types)
- [Error Handling](#error-handling)

## Overview

This document provides detailed API reference for all components of the fabric router pause/resume system. All APIs are in the `tt::tt_fabric::test_utils` namespace unless otherwise noted.

### Include Files

```cpp
#include "tests/tt_metal/tt_fabric/common/fabric_command_interface.hpp"
#include "tests/tt_metal/tt_fabric/common/fabric_traffic_validation.hpp"
#include "tests/tt_metal/tt_fabric/common/fabric_router_state_utils.hpp"
#include "tests/tt_metal/tt_fabric/common/fabric_worker_kernel_helpers.hpp"
#include "tests/tt_metal/tt_fabric/common/fabric_traffic_generator_defs.hpp"
```

## FabricCommandInterface

High-level API for controlling fabric routers.

### Class Declaration

```cpp
namespace tt::tt_fabric::test_utils {

class FabricCommandInterface {
public:
    explicit FabricCommandInterface(ControlPlane& control_plane);

    void pause_routers();
    void resume_routers();
    bool all_routers_in_state(RouterStateCommon expected_state);
    bool wait_for_pause(std::chrono::milliseconds timeout = DEFAULT_PAUSE_TIMEOUT);
    bool wait_for_state(
        RouterStateCommon target_state,
        std::chrono::milliseconds timeout = DEFAULT_PAUSE_TIMEOUT,
        std::chrono::milliseconds poll_interval = DEFAULT_POLL_INTERVAL);
    RouterStateCommon get_router_state(
        const FabricNodeId& fabric_node_id,
        chan_id_t channel_id);

protected:
    std::vector<std::pair<FabricNodeId, chan_id_t>> get_all_router_cores() const;

private:
    ControlPlane& control_plane_;
};

}  // namespace tt::tt_fabric::test_utils
```

### Constructor

#### FabricCommandInterface()

```cpp
explicit FabricCommandInterface(ControlPlane& control_plane);
```

Creates a command interface connected to the specified control plane.

**Parameters:**
- `control_plane` - Reference to the fabric control plane instance. Must remain valid for the lifetime of the FabricCommandInterface object.

**Example:**
```cpp
auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
test_utils::FabricCommandInterface cmd_interface(control_plane);
```

**Preconditions:**
- Control plane must be initialized
- Fabric topology must be discovered

**Complexity:** O(1)

---

### Public Methods

#### pause_routers()

```cpp
void pause_routers();
```

Issues pause commands to all active routers in the fabric.

**Behavior:**
1. Queries control plane for all active router cores
2. Issues PAUSE command to each router via FabricRouterStateManager
3. Returns immediately without waiting for state transition

**Postconditions:**
- Pause commands are queued to all routers
- Actual state transition occurs asynchronously
- Use `wait_for_pause()` to confirm completion

**Exceptions:**
- May throw if control plane communication fails

**Thread Safety:** Not thread-safe

**Example:**
```cpp
cmd_interface.pause_routers();
bool success = cmd_interface.wait_for_pause();
```

**See Also:** `wait_for_pause()`, `resume_routers()`

---

#### resume_routers()

```cpp
void resume_routers();
```

Issues resume (RUN) commands to all routers, returning them to normal operation.

**Behavior:**
1. Queries control plane for all active router cores
2. Issues RUN command to each router
3. Returns immediately without waiting for state transition

**Postconditions:**
- Resume commands are queued to all routers
- Routers asynchronously transition to RUNNING state
- Use `wait_for_state(RUNNING)` to confirm completion

**Exceptions:**
- May throw if control plane communication fails

**Thread Safety:** Not thread-safe

**Example:**
```cpp
cmd_interface.resume_routers();
bool success = cmd_interface.wait_for_state(RouterStateCommon::RUNNING);
```

**See Also:** `pause_routers()`, `wait_for_state()`

---

#### all_routers_in_state()

```cpp
bool all_routers_in_state(RouterStateCommon expected_state);
```

Queries current state of all routers and checks if they match the expected state.

**Parameters:**
- `expected_state` - The state to check for (RUNNING or PAUSED)

**Returns:**
- `true` - All routers are in the expected state
- `false` - One or more routers are in a different state

**Behavior:**
- Queries state of all active routers
- Compares each router's state to expected_state
- Returns false immediately if any router doesn't match

**Performance:**
- O(N) where N = number of routers
- Blocking call - reads state from all routers
- Typical latency: 1-10ms per router

**Thread Safety:** Not thread-safe

**Example:**
```cpp
cmd_interface.pause_routers();
std::this_thread::sleep_for(std::chrono::milliseconds(500));

if (cmd_interface.all_routers_in_state(RouterStateCommon::PAUSED)) {
    log_info(LogTest, "All routers paused");
} else {
    log_warning(LogTest, "Some routers still transitioning");
}
```

**See Also:** `get_router_state()`, `wait_for_state()`

---

#### wait_for_pause()

```cpp
bool wait_for_pause(
    std::chrono::milliseconds timeout = DEFAULT_PAUSE_TIMEOUT);
```

Waits for all routers to enter the PAUSED state within the specified timeout.

**Parameters:**
- `timeout` - Maximum time to wait (default: 5000ms)

**Returns:**
- `true` - All routers entered PAUSED state before timeout
- `false` - Timeout occurred before all routers paused

**Behavior:**
1. Polls router state at regular intervals (DEFAULT_POLL_INTERVAL = 100ms)
2. Returns true immediately when all routers reach PAUSED state
3. Returns false if timeout expires

**Performance:**
- Best case: One poll iteration (~100ms)
- Typical case: 200-500ms
- Worst case: Full timeout (5000ms)

**Thread Safety:** Not thread-safe

**Side Effects:**
- Blocks calling thread during wait
- Performs multiple control plane queries

**Example:**
```cpp
cmd_interface.pause_routers();
auto start = std::chrono::steady_clock::now();

if (cmd_interface.wait_for_pause()) {
    auto duration = std::chrono::steady_clock::now() - start;
    log_info(LogTest, "Pause completed in {} ms",
        std::chrono::duration_cast<std::chrono::milliseconds>(duration).count());
} else {
    log_error(LogTest, "Pause timeout after 5000ms");
}
```

**See Also:** `wait_for_state()`, `all_routers_in_state()`

---

#### wait_for_state()

```cpp
bool wait_for_state(
    RouterStateCommon target_state,
    std::chrono::milliseconds timeout = DEFAULT_PAUSE_TIMEOUT,
    std::chrono::milliseconds poll_interval = DEFAULT_POLL_INTERVAL);
```

Generic method to wait for all routers to enter a specific state.

**Parameters:**
- `target_state` - The desired router state (RUNNING or PAUSED)
- `timeout` - Maximum wait time (default: 5000ms)
- `poll_interval` - Time between state checks (default: 100ms)

**Returns:**
- `true` - All routers reached target state before timeout
- `false` - Timeout occurred

**Behavior:**
1. Polls router state every `poll_interval` milliseconds
2. Returns true when all routers reach `target_state`
3. Returns false if `timeout` expires

**Performance:**
- Iterations: timeout / poll_interval (e.g., 50 iterations for 5s timeout)
- Cost per iteration: O(N) where N = number of routers
- Sleep time per iteration: poll_interval

**Thread Safety:** Not thread-safe

**Example:**
```cpp
// Wait for resume with custom timeout
cmd_interface.resume_routers();
bool resumed = cmd_interface.wait_for_state(
    RouterStateCommon::RUNNING,
    std::chrono::milliseconds(3000),  // 3 second timeout
    std::chrono::milliseconds(50));   // Poll every 50ms

if (resumed) {
    log_info(LogTest, "Resume successful");
}
```

**See Also:** `wait_for_pause()`, `all_routers_in_state()`

---

#### get_router_state()

```cpp
RouterStateCommon get_router_state(
    const FabricNodeId& fabric_node_id,
    chan_id_t channel_id);
```

Queries the state of a specific router core.

**Parameters:**
- `fabric_node_id` - Fabric node identifier
  - `mesh_id` - Mesh identifier
  - `logical_x` - Logical X coordinate
  - `logical_y` - Logical Y coordinate
- `channel_id` - Router channel identifier

**Returns:**
- Current state of the specified router (RUNNING or PAUSED)

**Behavior:**
- Queries FabricRouterStateManager for router state
- Returns current state at time of query
- State may change immediately after return

**Performance:**
- O(1) operation
- Single control plane query
- Typical latency: 1-5ms

**Thread Safety:** Not thread-safe

**Example:**
```cpp
FabricNodeId node{.mesh_id = 0, .logical_x = 0, .logical_y = 0};
chan_id_t channel = 0;

RouterStateCommon state = cmd_interface.get_router_state(node, channel);
log_info(LogTest, "Router state: {}",
    test_utils::router_state_to_string(state));
```

**See Also:** `all_routers_in_state()`, `router_state_to_string()`

---

### Protected Methods

#### get_all_router_cores()

```cpp
std::vector<std::pair<FabricNodeId, chan_id_t>> get_all_router_cores() const;
```

Queries control plane for all active router cores.

**Returns:**
- Vector of (FabricNodeId, channel_id) pairs representing all active routers

**Behavior:**
- Queries control plane for fabric topology
- Enumerates all active router cores across all meshes
- Returns in arbitrary order

**Performance:** O(N) where N = number of routers

**Usage:** Internal helper method, used by pause_routers() and resume_routers()

---

## Traffic Validation API

Functions for detecting traffic flow through telemetry counters.

### Types

#### TelemetrySnapshot

```cpp
struct TelemetrySnapshot {
    std::map<FabricNodeId, std::map<chan_id_t, uint64_t>> words_sent_per_channel;
};
```

Represents a snapshot of telemetry counters at a specific point in time.

**Fields:**
- `words_sent_per_channel` - Nested map of node -> channel -> words_sent count

**Usage:**
```cpp
TelemetrySnapshot snap1 = capture_telemetry_snapshot(cp, mesh_id, num_devices);
std::this_thread::sleep_for(std::chrono::milliseconds(100));
TelemetrySnapshot snap2 = capture_telemetry_snapshot(cp, mesh_id, num_devices);

if (telemetry_changed(snap1, snap2)) {
    log_info(LogTest, "Traffic detected");
}
```

---

### Functions

#### capture_telemetry_snapshot()

```cpp
TelemetrySnapshot capture_telemetry_snapshot(
    ControlPlane& control_plane,
    MeshId mesh_id,
    size_t num_devices);
```

Captures current telemetry counter values across all devices.

**Parameters:**
- `control_plane` - Control plane instance
- `mesh_id` - Mesh identifier to capture
- `num_devices` - Number of devices in the mesh

**Returns:**
- TelemetrySnapshot containing current counter values

**Behavior:**
- Reads `words_sent` telemetry counter for each router channel
- Creates a point-in-time snapshot of all counters
- Does not wait or aggregate over time

**Performance:**
- O(N * C) where N = devices, C = channels per device
- Blocking operation
- Typical latency: 10-50ms depending on device count

**Thread Safety:** Not thread-safe

**Example:**
```cpp
auto snap = capture_telemetry_snapshot(control_plane, 0, 4);
log_info(LogTest, "Captured telemetry from {} devices", snap.words_sent_per_channel.size());
```

---

#### telemetry_changed()

```cpp
bool telemetry_changed(
    const TelemetrySnapshot& before,
    const TelemetrySnapshot& after);
```

Compares two snapshots to detect counter changes.

**Parameters:**
- `before` - Earlier telemetry snapshot
- `after` - Later telemetry snapshot

**Returns:**
- `true` - At least one counter increased
- `false` - No counters changed (or decreased)

**Behavior:**
- Iterates through all nodes and channels present in both snapshots
- Returns true if any `words_sent` counter increased
- Ignores counters that decreased (shouldn't happen in practice)

**Performance:** O(N * C) where N = nodes, C = channels

**Thread Safety:** Thread-safe (read-only)

**Example:**
```cpp
auto snap1 = capture_telemetry_snapshot(cp, mesh_id, num_devices);
std::this_thread::sleep_for(std::chrono::milliseconds(100));
auto snap2 = capture_telemetry_snapshot(cp, mesh_id, num_devices);

if (telemetry_changed(snap1, snap2)) {
    log_info(LogTest, "Traffic active");
} else {
    log_info(LogTest, "No traffic");
}
```

---

#### validate_traffic_flowing()

```cpp
bool validate_traffic_flowing(
    ControlPlane& control_plane,
    MeshId mesh_id,
    size_t num_devices,
    std::chrono::milliseconds sample_interval = DEFAULT_TRAFFIC_SAMPLE_INTERVAL);
```

Validates that traffic is actively flowing on at least one channel.

**Parameters:**
- `control_plane` - Control plane instance
- `mesh_id` - Mesh identifier to check
- `num_devices` - Number of devices in the mesh
- `sample_interval` - Time between telemetry samples (default: 100ms)

**Returns:**
- `true` - Traffic detected (at least one counter increased)
- `false` - No traffic detected in sample interval

**Behavior:**
1. Captures telemetry snapshot (before)
2. Sleeps for sample_interval
3. Captures telemetry snapshot (after)
4. Compares snapshots using telemetry_changed()

**Performance:**
- Duration: sample_interval + 2 * snapshot_cost
- Typical: 120-150ms
- Blocking operation

**Thread Safety:** Not thread-safe

**Use Cases:**
- Validate traffic before issuing pause
- Confirm worker kernels are generating traffic
- Check traffic resumes after resume command

**Example:**
```cpp
// Validate traffic is flowing before pause
bool flowing = test_utils::validate_traffic_flowing(control_plane, 0, 4);
ASSERT_TRUE(flowing) << "No traffic detected before pause";

// Issue pause
cmd_interface.pause_routers();
cmd_interface.wait_for_pause();

// Validate traffic stopped
bool stopped = test_utils::validate_traffic_stopped(control_plane, 0, 4);
ASSERT_TRUE(stopped) << "Traffic still flowing after pause";
```

---

#### validate_traffic_stopped()

```cpp
bool validate_traffic_stopped(
    ControlPlane& control_plane,
    MeshId mesh_id,
    size_t num_devices,
    std::chrono::milliseconds sample_interval = DEFAULT_TRAFFIC_SAMPLE_INTERVAL);
```

Validates that traffic has stopped on all channels.

**Parameters:**
- `control_plane` - Control plane instance
- `mesh_id` - Mesh identifier to check
- `num_devices` - Number of devices in the mesh
- `sample_interval` - Time between telemetry samples (default: 100ms)

**Returns:**
- `true` - No traffic detected (all counters unchanged)
- `false` - Traffic still flowing (at least one counter increased)

**Behavior:**
1. Captures telemetry snapshot (before)
2. Sleeps for sample_interval
3. Captures telemetry snapshot (after)
4. Returns inverse of telemetry_changed()

**Performance:**
- Duration: sample_interval + 2 * snapshot_cost
- Typical: 120-150ms
- Blocking operation

**Thread Safety:** Not thread-safe

**Use Cases:**
- Validate traffic stops after pause
- Confirm routers are not forwarding packets
- Check pause effectiveness

**Example:**
```cpp
// After pausing routers
cmd_interface.pause_routers();
bool paused = cmd_interface.wait_for_pause();
ASSERT_TRUE(paused);

// Validate traffic actually stopped
bool stopped = test_utils::validate_traffic_stopped(
    control_plane, mesh_id, num_devices,
    std::chrono::milliseconds(200));  // Longer sample for confidence

ASSERT_TRUE(stopped) << "Traffic still detected after pause";
```

---

## Router State Utilities

Observability and debugging utilities for router state inspection.

### Functions

#### router_state_to_string()

```cpp
const char* router_state_to_string(RouterStateCommon state);
```

Converts router state enum to human-readable string.

**Parameters:**
- `state` - Router state enum value

**Returns:**
- String representation:
  - `RouterStateCommon::RUNNING` -> `"RUNNING"`
  - `RouterStateCommon::PAUSED` -> `"PAUSED"`
  - Unknown values -> `"UNKNOWN"`

**Performance:** O(1)

**Thread Safety:** Thread-safe

**Example:**
```cpp
RouterStateCommon state = cmd_interface.get_router_state(node, channel);
log_info(LogTest, "Router state: {}", router_state_to_string(state));
```

---

#### log_all_router_states()

```cpp
void log_all_router_states(
    ControlPlane& control_plane,
    const std::vector<MeshId>& mesh_ids);
```

Logs the current state of all routers across specified meshes.

**Parameters:**
- `control_plane` - Control plane instance
- `mesh_ids` - Vector of mesh IDs to inspect

**Output Format:**
```
[INFO] === Router State Summary ===
[INFO] Mesh ID: 0
[INFO]   Device 0 Channel 0: RUNNING
[INFO]   Device 0 Channel 1: RUNNING
[INFO]   Device 1 Channel 0: PAUSED
[INFO]   Device 1 Channel 1: PAUSED
...
```

**Behavior:**
- Iterates through all meshes in mesh_ids
- Queries state of all routers in each mesh
- Logs state using log_info(LogTest, ...)

**Performance:**
- O(N * C) where N = devices, C = channels per device
- Blocking operation
- Typical latency: 50-200ms depending on device count

**Thread Safety:** Not thread-safe

**Use Cases:**
- Debugging test failures
- Verifying expected fabric state
- Observability during pause operations (NFR-5)

**Example:**
```cpp
auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
auto mesh_ids = control_plane.get_user_physical_mesh_ids();

// Log state before pause
log_info(LogTest, "Router state before pause:");
test_utils::log_all_router_states(control_plane, mesh_ids);

cmd_interface.pause_routers();
cmd_interface.wait_for_pause();

// Log state after pause
log_info(LogTest, "Router state after pause:");
test_utils::log_all_router_states(control_plane, mesh_ids);
```

---

#### count_routers_by_state()

```cpp
std::map<RouterStateCommon, uint32_t> count_routers_by_state(
    ControlPlane& control_plane,
    const std::vector<MeshId>& mesh_ids);
```

Aggregates router state counts across the fabric.

**Parameters:**
- `control_plane` - Control plane instance
- `mesh_ids` - Vector of mesh IDs to count

**Returns:**
- Map of state to count, e.g.:
  ```cpp
  {
      {RouterStateCommon::RUNNING, 4},
      {RouterStateCommon::PAUSED, 0}
  }
  ```

**Behavior:**
- Queries state of all routers across all specified meshes
- Aggregates counts by state
- Returns empty map if no routers found

**Performance:**
- O(N * C) where N = devices, C = channels per device
- Blocking operation

**Thread Safety:** Not thread-safe

**Use Cases:**
- Quick summary of fabric state
- Validate expected state distribution
- Metrics collection

**Example:**
```cpp
auto state_counts = test_utils::count_routers_by_state(control_plane, mesh_ids);

log_info(LogTest, "Router state summary:");
for (const auto& [state, count] : state_counts) {
    log_info(LogTest, "  {}: {}",
        test_utils::router_state_to_string(state), count);
}

// Verify all routers are paused
ASSERT_EQ(state_counts[RouterStateCommon::RUNNING], 0);
ASSERT_GT(state_counts[RouterStateCommon::PAUSED], 0);
```

---

## Worker Kernel Helpers

Utilities for managing traffic generator kernels on devices.

### Types

#### WorkerMemoryLayout

```cpp
struct WorkerMemoryLayout {
    uint32_t source_buffer_address;
    uint32_t teardown_signal_address;
    uint32_t packet_payload_size_bytes;
};
```

Describes L1 memory layout for worker kernel.

**Fields:**
- `source_buffer_address` - L1 address of packet source buffer
- `teardown_signal_address` - L1 address of teardown mailbox
- `packet_payload_size_bytes` - Size of packets to generate

---

### Functions

#### allocate_worker_memory()

```cpp
WorkerMemoryLayout allocate_worker_memory(
    const std::shared_ptr<tt_metal::distributed::MeshDevice>& device);
```

Allocates L1 memory addresses for worker kernel buffers.

**Parameters:**
- `device` - Target device for allocation

**Returns:**
- WorkerMemoryLayout struct containing allocated addresses

**Behavior:**
- Allocates non-overlapping L1 regions for:
  - Source buffer (packet data)
  - Teardown signal mailbox
- Returns addresses suitable for kernel configuration

**Performance:** O(1), no device communication required

**Thread Safety:** Not thread-safe

**Example:**
```cpp
auto device = get_devices()[0];
auto mem_layout = test_utils::allocate_worker_memory(device);

log_info(LogTest, "Allocated memory:");
log_info(LogTest, "  Source buffer: 0x{:x}", mem_layout.source_buffer_address);
log_info(LogTest, "  Teardown signal: 0x{:x}", mem_layout.teardown_signal_address);
log_info(LogTest, "  Packet size: {} bytes", mem_layout.packet_payload_size_bytes);
```

---

#### create_traffic_generator_program()

```cpp
std::shared_ptr<tt_metal::Program> create_traffic_generator_program(
    const std::shared_ptr<tt_metal::distributed::MeshDevice>& device,
    const CoreCoord& logical_core,
    const FabricNodeId& dest_fabric_node,
    const WorkerMemoryLayout& mem_layout);
```

Creates a program containing the traffic generator kernel.

**Parameters:**
- `device` - Target device
- `logical_core` - Logical core coordinates for kernel placement
- `dest_fabric_node` - Destination node for generated packets
- `mem_layout` - Memory layout for buffers

**Returns:**
- Shared pointer to configured Program ready for execution

**Behavior:**
1. Creates new Program
2. Adds traffic generator kernel to specified core
3. Configures compile-time args (addresses, encoding)
4. Configures runtime args (destination, seed)
5. Returns program (not yet launched)

**Performance:** O(1), program creation only

**Thread Safety:** Not thread-safe

**Example:**
```cpp
auto device = get_devices()[0];
auto mem_layout = test_utils::allocate_worker_memory(device);
CoreCoord worker_core{0, 0};

FabricNodeId dest_node = test_utils::get_fabric_node_id(get_devices()[1]);

auto program = test_utils::create_traffic_generator_program(
    device, worker_core, dest_node, mem_layout);

// Launch the program
RunProgramNonblocking(device, *program);
```

---

#### signal_worker_teardown()

```cpp
void signal_worker_teardown(
    const std::shared_ptr<tt_metal::distributed::MeshDevice>& device,
    const CoreCoord& logical_core,
    uint32_t teardown_signal_address);
```

Signals a running worker kernel to terminate gracefully.

**Parameters:**
- `device` - Device running the worker
- `logical_core` - Core coordinates of the worker
- `teardown_signal_address` - L1 address to write teardown signal

**Behavior:**
- Writes `WORKER_TEARDOWN` (1) to the specified L1 address
- Worker kernel polls this address and exits when it reads 1
- Non-blocking: returns immediately after write

**Postconditions:**
- Teardown signal written to L1
- Kernel will exit on next poll iteration
- Use wait_for_worker_complete() to wait for exit

**Performance:** O(1), single L1 write

**Thread Safety:** Not thread-safe

**Example:**
```cpp
// Signal teardown to worker
test_utils::signal_worker_teardown(
    device, worker_core, mem_layout.teardown_signal_address);

// Wait for worker to complete
test_utils::wait_for_worker_complete(
    this, device, *program, std::chrono::milliseconds(1000));
```

---

#### wait_for_worker_complete()

```cpp
void wait_for_worker_complete(
    BaseFabricFixture* fixture,
    const std::shared_ptr<tt_metal::distributed::MeshDevice>& device,
    tt_metal::Program& program,
    std::chrono::milliseconds timeout = std::chrono::milliseconds(1000));
```

Waits for a worker program to complete after teardown is signaled.

**Parameters:**
- `fixture` - Test fixture reference (for WaitProgramDone)
- `device` - Device running the worker
- `program` - Worker program
- `timeout` - Maximum wait time (default: 1000ms)

**Behavior:**
1. Waits for program to complete
2. Returns when program exits
3. Throws exception if timeout expires

**Exceptions:**
- Throws std::runtime_error if worker does not complete within timeout

**Performance:**
- Best case: Immediate (kernel already complete)
- Typical case: < 100ms (kernel exits on next poll)
- Worst case: timeout duration

**Thread Safety:** Not thread-safe

**Example:**
```cpp
try {
    test_utils::signal_worker_teardown(device, core, teardown_address);
    test_utils::wait_for_worker_complete(
        this, device, *program, std::chrono::milliseconds(2000));
    log_info(LogTest, "Worker completed successfully");
} catch (const std::exception& e) {
    log_error(LogTest, "Worker timeout: {}", e.what());
}
```

---

## Constants and Types

### Constants

```cpp
namespace tt::tt_fabric::test_utils {

// Worker kernel teardown protocol
constexpr uint32_t WORKER_KEEP_RUNNING = 0;
constexpr uint32_t WORKER_TEARDOWN = 1;

// Timing constants
constexpr std::chrono::milliseconds DEFAULT_TRAFFIC_SAMPLE_INTERVAL{100};
constexpr std::chrono::milliseconds DEFAULT_PAUSE_TIMEOUT{5000};
constexpr std::chrono::milliseconds DEFAULT_POLL_INTERVAL{100};

}  // namespace
```

### Enums

#### RouterStateCommon

```cpp
namespace tt::tt_fabric {

enum class RouterStateCommon {
    RUNNING = 0,
    PAUSED = 1,
};

}  // namespace
```

Router state values:
- `RUNNING`: Router actively forwards packets
- `PAUSED`: Router stops forwarding new packets

---

#### RouterCommand

```cpp
namespace tt::tt_fabric {

enum class RouterCommand {
    PAUSE = 0,
    RUN = 1,
};

}  // namespace
```

Commands for router control:
- `PAUSE`: Stop packet forwarding
- `RUN`: Resume packet forwarding

---

### Structures

#### FabricNodeId

```cpp
namespace tt::tt_fabric {

struct FabricNodeId {
    uint32_t mesh_id;
    uint32_t logical_x;
    uint32_t logical_y;

    bool operator==(const FabricNodeId& other) const;
};

}  // namespace
```

Identifies a fabric node by mesh and logical coordinates.

**Fields:**
- `mesh_id` - Mesh identifier
- `logical_x` - Logical X coordinate in fabric
- `logical_y` - Logical Y coordinate in fabric

**Example:**
```cpp
FabricNodeId node{
    .mesh_id = 0,
    .logical_x = 1,
    .logical_y = 0
};
```

---

#### TrafficGeneratorCompileArgs

```cpp
namespace tt::tt_fabric::test_utils {

struct TrafficGeneratorCompileArgs {
    uint32_t source_buffer_address;
    uint32_t packet_payload_size_bytes;
    uint32_t target_noc_encoding;
    uint32_t teardown_signal_address;
    uint32_t is_2d_fabric;
};

}  // namespace
```

Compile-time configuration for traffic generator kernel.

---

#### TrafficGeneratorRuntimeArgs

```cpp
namespace tt::tt_fabric::test_utils {

struct TrafficGeneratorRuntimeArgs {
    uint32_t dest_chip_id;
    uint32_t dest_mesh_id;
    uint32_t random_seed;
};

}  // namespace
```

Runtime configuration for traffic generator kernel.

---

## Error Handling

### Exception Behavior

Most APIs do not throw exceptions under normal operation. Exceptions may be thrown for:

1. **Control Plane Communication Failures**
   - Functions: All FabricCommandInterface methods
   - Cause: Device communication error
   - Recovery: Let exception propagate, test will fail

2. **Worker Timeout**
   - Function: `wait_for_worker_complete()`
   - Cause: Worker kernel did not exit within timeout
   - Recovery: Catch exception, log error, continue cleanup

3. **Invalid Parameters**
   - All functions may throw for invalid parameters
   - Cause: Null pointers, invalid mesh IDs, etc.
   - Recovery: Fix caller

### Return Value Conventions

- **Boolean returns**: Use bool for success/failure or presence/absence
  - `true` = success, condition met, data present
  - `false` = failure, timeout, no data

- **Pointer returns**: Return shared_ptr or raw pointer
  - Valid pointer = success
  - nullptr = not used in current API

- **Void returns**: Operation always succeeds or throws

### Error Checking Examples

```cpp
// Check boolean returns
if (!cmd_interface.wait_for_pause()) {
    // Handle timeout
    test_utils::log_all_router_states(control_plane, mesh_ids);
    FAIL() << "Pause timeout";
}

// Catch worker exceptions
for (size_t i = 0; i < workers.size(); ++i) {
    try {
        wait_for_worker_complete(this, device, *workers[i]);
    } catch (const std::exception& e) {
        log_error(LogTest, "Worker {} failed: {}", i, e.what());
        // Continue cleanup of other workers
    }
}

// Validate state
ASSERT_TRUE(cmd_interface.all_routers_in_state(RouterStateCommon::PAUSED))
    << "Not all routers paused";
```

---

## Thread Safety

**None of the APIs are thread-safe.** All functions must be called from a single thread.

**Rationale:**
- Test code is single-threaded
- Control plane is not designed for concurrent access
- Simplifies implementation and reasoning

**Usage:**
```cpp
// CORRECT: Single-threaded usage
void TestMethod() {
    cmd_interface.pause_routers();
    cmd_interface.wait_for_pause();
}

// INCORRECT: Multi-threaded usage
void ThreadA() {
    cmd_interface.pause_routers();  // Race condition!
}
void ThreadB() {
    cmd_interface.resume_routers();  // Race condition!
}
```

---

## Performance Summary

| Operation | Complexity | Typical Latency |
|-----------|-----------|-----------------|
| `pause_routers()` | O(N) | 10-50ms |
| `resume_routers()` | O(N) | 10-50ms |
| `wait_for_pause()` | O(N * M) | 200-500ms |
| `all_routers_in_state()` | O(N) | 10-50ms |
| `get_router_state()` | O(1) | 1-5ms |
| `validate_traffic_flowing()` | O(D * C) | 120-150ms |
| `validate_traffic_stopped()` | O(D * C) | 120-150ms |
| `log_all_router_states()` | O(N * C) | 50-200ms |
| `signal_worker_teardown()` | O(1) | < 1ms |
| `wait_for_worker_complete()` | - | < 100ms |

Where:
- N = number of routers
- M = poll iterations (timeout / poll_interval)
- D = number of devices
- C = channels per router

---

## See Also

- [Main README](README_PAUSE_RESUME.md) - Feature overview and usage guide
- [Architecture Documentation](ARCHITECTURE.md) - System design and diagrams
- [Usage Guide](USAGE_GUIDE.md) - Code examples and best practices
