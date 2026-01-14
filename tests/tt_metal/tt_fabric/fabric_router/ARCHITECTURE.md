# Fabric Router Pause/Resume Architecture

## System Overview

The fabric router pause/resume feature provides runtime control over packet forwarding in the TT-Metal fabric network. This document describes the architecture, component interactions, and implementation details.

## High-Level Architecture

```mermaid
flowchart TB
    subgraph "Test Layer"
        TestHarness[Test Harness<br/>test_fabric_router_pause_control.cpp]
        TrafficGen[Traffic Generator Kernels<br/>Device-Side]
    end

    subgraph "API Layer"
        CmdIF[FabricCommandInterface<br/>High-Level Control API]
        TrafficVal[Traffic Validation<br/>Telemetry-Based Detection]
        RouterState[Router State Utils<br/>Query & Logging]
        WorkerHelpers[Worker Kernel Helpers<br/>Launch & Teardown]
    end

    subgraph "Control Plane"
        CP[ControlPlane<br/>Fabric Management]
        RSM[FabricRouterStateManager<br/>State Management]
    end

    subgraph "Hardware"
        Router[Fabric Routers<br/>Multiple Devices]
        Telemetry[Telemetry Counters<br/>words_sent, etc.]
    end

    TestHarness --> CmdIF
    TestHarness --> TrafficVal
    TestHarness --> RouterState
    TestHarness --> WorkerHelpers
    TestHarness --> TrafficGen

    CmdIF --> CP
    CmdIF --> RSM
    TrafficVal --> CP
    TrafficVal --> Telemetry
    RouterState --> CP
    WorkerHelpers --> TrafficGen

    CP --> Router
    RSM --> Router
    Router --> Telemetry
```

## Component Architecture

### Layer 1: Test Harness

The test harness orchestrates the complete test lifecycle:

```mermaid
stateDiagram-v2
    [*] --> Initialize
    Initialize --> LaunchWorkers: Setup Complete
    LaunchWorkers --> ValidateTraffic: Workers Started
    ValidateTraffic --> PauseRouters: Traffic Confirmed
    PauseRouters --> WaitForPause: Commands Issued
    WaitForPause --> ValidateStop: All Paused
    WaitForPause --> Timeout: Timeout
    ValidateStop --> ResumeRouters: Traffic Stopped
    ResumeRouters --> CleanupWorkers: Resume Complete
    CleanupWorkers --> [*]
    Timeout --> Error
    Error --> CleanupWorkers: Cleanup
```

### Layer 2: API Components

#### FabricCommandInterface

High-level abstraction for router control operations.

**Responsibilities:**
- Query active routers from control plane
- Issue pause/resume commands to all routers
- Poll router state until target state reached
- Provide timeout detection

**Key Methods:**
- `pause_routers()`: Issue PAUSE command to all routers
- `resume_routers()`: Issue RUN command to all routers
- `wait_for_pause()`: Poll until all routers PAUSED or timeout
- `all_routers_in_state()`: Check if all routers in expected state

#### Traffic Validation

Telemetry-based traffic detection for validation.

**Responsibilities:**
- Capture telemetry snapshots at different times
- Compare snapshots to detect traffic changes
- Provide validation predicates for test assertions

**Key Methods:**
- `capture_telemetry_snapshot()`: Read all telemetry counters
- `telemetry_changed()`: Compare two snapshots
- `validate_traffic_flowing()`: Assert traffic is active
- `validate_traffic_stopped()`: Assert traffic is stopped

#### Router State Utils

Observability and debugging utilities.

**Responsibilities:**
- Query router state across fabric
- Log detailed state information
- Aggregate state statistics

**Key Methods:**
- `log_all_router_states()`: Print all router states
- `count_routers_by_state()`: Aggregate counts by state
- `router_state_to_string()`: Convert enum to string

#### Worker Kernel Helpers

Management of device-side traffic generator kernels.

**Responsibilities:**
- Allocate L1 memory for worker kernels
- Create and configure kernel programs
- Launch kernels on device cores
- Signal graceful teardown

**Key Methods:**
- `allocate_worker_memory()`: Allocate L1 addresses
- `create_traffic_generator_program()`: Build kernel program
- `signal_worker_teardown()`: Write to teardown mailbox
- `wait_for_worker_complete()`: Block until kernel exits

### Layer 3: Control Plane

The control plane provides the core infrastructure for fabric management.

**Provided by TT-Metal Core:**
- Fabric topology discovery
- Router enumeration
- Telemetry reader interface
- FabricRouterStateManager integration

### Layer 4: Hardware

Physical fabric routers and telemetry infrastructure.

## Data Flow Diagrams

### Pause Command Flow

```mermaid
sequenceDiagram
    participant Test as Test Harness
    participant CmdIF as FabricCommandInterface
    participant CP as ControlPlane
    participant RSM as RouterStateManager
    participant R0 as Router 0
    participant R1 as Router 1
    participant Rn as Router N

    Test->>CmdIF: pause_routers()
    CmdIF->>CP: get_all_router_cores()
    CP-->>CmdIF: [(node0, ch0), (node1, ch0), ...]

    loop For each router
        CmdIF->>RSM: send_command(PAUSE, node, channel)
        RSM->>R0: Write PAUSE to control register
        RSM->>R1: Write PAUSE to control register
        RSM->>Rn: Write PAUSE to control register
    end

    CmdIF-->>Test: Commands issued

    Test->>CmdIF: wait_for_pause(5000ms)

    loop Poll every 100ms until timeout
        CmdIF->>RSM: get_state(node, channel)
        RSM->>R0: Read state register
        R0-->>RSM: PAUSED
        RSM-->>CmdIF: PAUSED

        CmdIF->>RSM: get_state(node, channel)
        RSM->>R1: Read state register
        R1-->>RSM: PAUSED
        RSM-->>CmdIF: PAUSED

        alt All routers PAUSED
            CmdIF-->>Test: true (success)
        else Timeout reached
            CmdIF-->>Test: false (timeout)
        end
    end
```

### Traffic Validation Flow

```mermaid
sequenceDiagram
    participant Test as Test Harness
    participant TV as TrafficValidation
    participant CP as ControlPlane
    participant TR as TelemetryReader
    participant R0 as Router 0
    participant R1 as Router 1

    Test->>TV: validate_traffic_flowing()

    TV->>TV: capture_telemetry_snapshot()
    TV->>CP: get_fabric_nodes()
    CP-->>TV: [node0, node1, ...]

    loop For each node
        TV->>TR: read_telemetry(node)
        TR->>R0: Read words_sent counter
        R0-->>TR: 1000
        TR-->>TV: {words_sent: 1000}
    end

    TV->>TV: Store snapshot1
    TV->>TV: sleep(100ms)
    TV->>TV: capture_telemetry_snapshot()

    loop For each node
        TV->>TR: read_telemetry(node)
        TR->>R1: Read words_sent counter
        R1-->>TR: 1250
        TR-->>TV: {words_sent: 1250}
    end

    TV->>TV: Store snapshot2
    TV->>TV: telemetry_changed(snapshot1, snapshot2)

    alt Counter increased
        TV-->>Test: true (traffic flowing)
    else Counter unchanged
        TV-->>Test: false (no traffic)
    end
```

### Worker Kernel Management

```mermaid
sequenceDiagram
    participant Test as Test Harness
    participant WH as WorkerHelpers
    participant Device as Device
    participant Kernel as Traffic Generator Kernel
    participant L1 as L1 Memory

    Test->>WH: allocate_worker_memory(device)
    WH->>Device: Allocate L1 buffers
    Device-->>WH: WorkerMemoryLayout

    Test->>WH: create_traffic_generator_program(device, core, dest, mem_layout)
    WH->>WH: Configure compile args
    WH->>WH: Configure runtime args
    WH-->>Test: Program

    Test->>Device: RunProgramNonblocking(program)
    Device->>Kernel: Launch on core

    Note over Kernel: Kernel runs continuously<br/>Polls teardown address

    loop Continuous operation
        Kernel->>L1: Read teardown_address
        L1-->>Kernel: WORKER_KEEP_RUNNING (0)
        Kernel->>Kernel: Generate & send packet
    end

    Test->>WH: signal_worker_teardown(device, core, teardown_address)
    WH->>L1: Write WORKER_TEARDOWN (1) to teardown_address

    Kernel->>L1: Read teardown_address
    L1-->>Kernel: WORKER_TEARDOWN (1)
    Kernel->>Kernel: Exit loop

    Test->>WH: wait_for_worker_complete(device, program, 1000ms)
    WH->>Device: Wait for program completion
    Device-->>WH: Complete
    WH-->>Test: Success
```

## State Machine

### Router State Transitions

```mermaid
stateDiagram-v2
    [*] --> RUNNING: Initial State
    RUNNING --> PAUSED: PAUSE command
    PAUSED --> RUNNING: RUN command

    note right of RUNNING
        Router forwards packets
        Telemetry counters increment
    end note

    note right of PAUSED
        Router stops forwarding
        Counters stable
        In-flight packets may complete
    end note
```

### Test State Machine

```mermaid
stateDiagram-v2
    [*] --> Setup: Test Start
    Setup --> WorkersLaunched: launch_traffic_generators()
    WorkersLaunched --> TrafficFlowing: validate_traffic_flowing()
    TrafficFlowing --> PauseIssued: cmd_interface.pause_routers()
    PauseIssued --> WaitingForPause: cmd_interface.wait_for_pause()

    WaitingForPause --> AllPaused: All routers PAUSED
    WaitingForPause --> PauseTimeout: Timeout

    AllPaused --> TrafficStopped: validate_traffic_stopped()
    TrafficStopped --> Cleanup: signal_worker_teardown()
    PauseTimeout --> Error: Assertion failure

    Cleanup --> Complete: Workers cleaned up
    Complete --> [*]

    Error --> Cleanup: TearDown()
```

## Component Interactions

### Initialization Sequence

```mermaid
sequenceDiagram
    participant GTest as GoogleTest Framework
    participant Fixture as Fabric1DFixture
    participant Test as FabricRouterPauseControlTest
    participant CP as ControlPlane

    GTest->>Fixture: SetUp()
    Fixture->>Fixture: Initialize fabric topology
    Fixture->>CP: Initialize control plane
    Fixture->>Fixture: Discover devices

    GTest->>Test: SetUp()
    Test->>Test: Initialize member variables
    Test->>Test: workers_launched_ = false

    GTest->>Test: TEST_F body execution
    Note over Test: Test runs here

    GTest->>Test: TearDown()
    alt Workers were launched
        Test->>Test: cleanup_workers()
        Test->>Test: Signal teardown
        Test->>Test: Wait for completion
    end

    GTest->>Fixture: TearDown()
    Fixture->>Fixture: Cleanup fabric resources
```

### Error Handling Flow

```mermaid
flowchart TD
    Start[Test Execution] --> Launch[Launch Workers]
    Launch --> |Success| Validate[Validate Traffic]
    Launch --> |Error| Error1[Launch Error]

    Validate --> |Traffic OK| Pause[Issue Pause]
    Validate --> |No Traffic| Error2[Traffic Error]

    Pause --> Wait[Wait for Pause]
    Wait --> |Success| CheckStop[Validate Stopped]
    Wait --> |Timeout| Error3[Pause Timeout]

    CheckStop --> |Stopped| Success[Test Success]
    CheckStop --> |Still Flowing| Error4[Stop Validation Error]

    Error1 --> TearDown[TearDown Called]
    Error2 --> TearDown
    Error3 --> TearDown
    Error4 --> TearDown
    Success --> TearDown

    TearDown --> CheckLaunched{workers_launched_?}
    CheckLaunched --> |true| Cleanup[cleanup_workers]
    CheckLaunched --> |false| End[Test End]
    Cleanup --> End
```

## Memory Layout

### Worker Kernel L1 Memory

```
L1 Address Space:
+-----------------------------------+
| Source Buffer                      |
| (Packet payload data)             |
| Size: packet_payload_size_bytes   |
| Address: source_buffer_address    |
+-----------------------------------+
| Teardown Signal                   |
| (Single uint32_t)                 |
| Values: 0=KEEP_RUNNING, 1=TEARDOWN|
| Address: teardown_signal_address  |
+-----------------------------------+
| Other kernel data...              |
+-----------------------------------+
```

### WorkerMemoryLayout Structure

```cpp
struct WorkerMemoryLayout {
    uint32_t source_buffer_address;      // Base of source buffer
    uint32_t teardown_signal_address;    // Teardown mailbox
    uint32_t packet_payload_size_bytes;  // Size of packets
};
```

## Telemetry Architecture

### Telemetry Snapshot Structure

```cpp
struct TelemetrySnapshot {
    // Map: FabricNodeId -> (channel_id -> words_sent_count)
    std::map<FabricNodeId, std::map<chan_id_t, uint64_t>> words_sent_per_channel;
};
```

### Traffic Detection Algorithm

```
Algorithm: validate_traffic_flowing()
1. snapshot_before = capture_telemetry_snapshot()
2. sleep(sample_interval)  // Default: 100ms
3. snapshot_after = capture_telemetry_snapshot()
4. return telemetry_changed(snapshot_before, snapshot_after)

Algorithm: telemetry_changed()
For each (node, channel) in both snapshots:
    if snapshot_after[node][channel] > snapshot_before[node][channel]:
        return true  // Traffic detected
return false  // No traffic
```

## Performance Considerations

### Pause Latency Breakdown

```mermaid
gantt
    title Pause Operation Timeline
    dateFormat X
    axisFormat %L ms

    section Command Phase
    Issue Commands :milestone, 0, 0
    Command Propagation :10, 50

    section Router Phase
    Router Processing :50, 100
    State Transition :100, 150

    section Verification Phase
    First State Poll :150, 200
    Confirmation :200, 250

    section Total
    Complete :milestone, 250, 250
```

Typical timeline:
- Command issue: < 10ms
- Command propagation to routers: 10-50ms
- Router state transition: 50-150ms
- State polling and confirmation: 50-100ms
- **Total typical latency: 200-400ms**

### Polling Strategy

```
Wait Loop Configuration:
- poll_interval: 100ms (configurable)
- timeout: 5000ms (configurable)
- max_iterations: timeout / poll_interval = 50

Per-iteration cost:
- State query per router: ~1-5ms
- Total per iteration (N routers): N * 5ms
- Sleep: 100ms
- Iteration total: ~100-150ms
```

## Scalability

### Device Scaling

The architecture scales linearly with device count:

```
Operation Complexity:
- pause_routers(): O(N) where N = number of routers
- wait_for_pause(): O(N * M) where M = number of poll iterations
- validate_traffic_flowing(): O(D) where D = number of devices

Memory:
- Telemetry snapshots: O(N * C) where C = channels per router
- Worker programs: O(W) where W = number of workers
```

### Multi-Mesh Support

The implementation supports multiple meshes:

```cpp
// Single mesh
test_utils::log_all_router_states(control_plane, {mesh_id});

// Multiple meshes
auto all_mesh_ids = control_plane.get_user_physical_mesh_ids();
test_utils::log_all_router_states(control_plane, all_mesh_ids);
```

## Design Patterns

### Command Pattern

FabricCommandInterface implements the command pattern:

```
Command: pause_routers()
- Encapsulates pause operation
- Separates invocation from execution
- Supports undo via resume_routers()

Command: resume_routers()
- Reverse operation
- Returns system to original state
```

### Observer Pattern

Telemetry-based validation implements observer pattern:

```
Subject: Router telemetry counters
Observers: Traffic validation utilities
Notification: Counter value changes
Response: Update snapshot, compare, validate
```

### Template Method Pattern

Worker lifecycle follows template method pattern:

```
Template: Worker management
1. allocate_worker_memory()
2. create_traffic_generator_program()
3. RunProgramNonblocking()
4. signal_worker_teardown()
5. wait_for_worker_complete()

Each step is customizable but order is fixed
```

## Threading Model

The implementation is single-threaded with explicit waits:

```
Main Thread:
1. Launch workers (non-blocking kernel launch)
2. Issue pause commands (blocking control plane calls)
3. Poll for pause completion (sleep-based polling)
4. Validate traffic (blocking telemetry reads)
5. Cleanup workers (blocking wait)

No Concurrency:
- No shared mutable state between threads
- No need for locks or atomics
- Simple reasoning about execution order
```

## Error Recovery

### Failure Modes and Recovery

1. **Worker Launch Failure**
   - Detection: Program creation or launch throws exception
   - Recovery: TearDown skips cleanup if workers_launched_ is false
   - Cleanup: No cleanup needed if launch failed

2. **Pause Timeout**
   - Detection: wait_for_pause() returns false
   - Recovery: Test fails with assertion
   - Cleanup: TearDown still attempts worker cleanup

3. **Traffic Validation Failure**
   - Detection: validate_traffic_stopped() returns false during pause
   - Recovery: Test fails with assertion
   - Cleanup: TearDown ensures workers are cleaned up

4. **Worker Teardown Timeout**
   - Detection: wait_for_worker_complete() throws exception
   - Recovery: Exception is caught and logged
   - Cleanup: Continue cleanup of remaining workers

### Cleanup Guarantees

```cpp
void TearDown() override {
    // GUARANTEE: Always attempt cleanup, even if test fails
    if (workers_launched_) {
        cleanup_workers();
    }
    Fabric1DFixture::TearDown();
}
```

This ensures:
- Workers are always signaled to stop
- Device resources are released
- No leaked kernels or memory

## Future Enhancements

Potential improvements to the architecture:

1. **Asynchronous State Polling**
   - Replace sleep-based polling with event-driven notifications
   - Reduce latency and improve responsiveness

2. **Partial Pause Support**
   - Pause specific routers or channels
   - Enable more granular traffic control

3. **State History**
   - Track state transitions over time
   - Enable performance analysis and debugging

4. **Automatic Recovery**
   - Detect and recover from transient pause failures
   - Retry mechanisms with exponential backoff

5. **Multi-Level Pause**
   - PAUSE: Stop new packets
   - DRAIN: Complete in-flight packets
   - FREEZE: Hard stop everything

## References

### Component Files

- Test: `test_fabric_router_pause_control.cpp`
- Command Interface: `fabric_command_interface.{hpp,cpp}`
- Traffic Validation: `fabric_traffic_validation.{hpp,cpp}`
- Router State Utils: `fabric_router_state_utils.{hpp,cpp}`
- Worker Helpers: `fabric_worker_kernel_helpers.{hpp,cpp}`
- Definitions: `fabric_traffic_generator_defs.hpp`

### External Dependencies

- ControlPlane: `tt_metal/fabric/control_plane.hpp`
- FabricRouterStateManager: Core control plane component
- Fabric Telemetry: Hardware telemetry subsystem
- TT-Metal Core: Device management and program execution

### Design Documents

- Main README: `README_PAUSE_RESUME.md`
- API Documentation: See API_REFERENCE.md
- Usage Guide: See USAGE_GUIDE.md
