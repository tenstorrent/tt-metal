# Architecture

**Analysis Date:** 2026-03-16

## Pattern Overview

**Overall:** Layered Hardware Abstraction with Device-Host Asymmetry

The tt-metal codebase follows a strict separation between:
- **Host-side** (C++/Python): Device management, program construction, command dispatch
- **Device-side** (Firmware/Kernels): Compute and dataflow execution on Tenstorrent hardware (Wormhole, Blackhole)
- **Control plane**: Fabric infrastructure for multi-device coordination

**Key Characteristics:**
- Explicit device abstraction layer (`IDevice` interface in `tt_metal/api/tt-metalium/`)
- Command-queue based dispatch model with hardware abstraction via HAL
- Program-kernel execution model where Programs encapsulate kernel configurations
- Multi-device scaling via Fabric and Mesh abstractions for distributed execution
- Allocator-based memory management with support for sharded/interleaved layouts

## Layers

**API Layer (Public Interface):**
- Location: `tt_metal/api/tt-metalium/`
- Contains: Header files defining public interfaces (device.hpp, program.hpp, buffer.hpp, host_api.hpp)
- Depends on: UMD (User Mode Driver), HAL types
- Used by: All downstream applications (TTNN, models, user code)
- Pattern: Stable C++ interfaces with forward declarations; minimal includes

**Implementation Layer (Core Runtime):**
- Location: `tt_metal/impl/`
- Contains: Device initialization, program compilation, buffer allocation, command dispatch
- Key subdirectories:
  - `impl/device/` - Device instantiation and initialization (`device.cpp`, `device.hpp`)
  - `impl/program/` - Program compilation and kernel assembly (`program.cpp`, `program_impl.hpp`)
  - `impl/dispatch/` - Command queue management and device communication
  - `impl/allocator/` - Memory management with support for L1 banking and sharding
  - `impl/buffers/` - Buffer lifecycle and circular buffer management
  - `impl/kernels/` - Kernel metadata and compilation
- Depends on: Common utilities, JIT build system, HAL
- Used by: API layer

**Fabric Layer (Multi-Device Communication):**
- Location: `tt_metal/fabric/`
- Contains: Ethernet routing, device interconnect configuration, channel management
- Key subdirectories:
  - `fabric/impl/` - Fabric context and builder implementation
  - `fabric/builder/` - Static channel allocation and fabric construction
  - `fabric/ccl/` - Collective communication operations
  - `fabric/protobuf/` - Message serialization for control plane
  - `fabric/debug/` - Channel trimming and bandwidth profiling
- Pattern: Builder pattern for fabric construction; runtime context for communication
- Depends on: Device layer, Program layer, networking primitives
- Used by: MeshDevice, distributed execution

**Distributed Layer (Multi-Mesh Scaling):**
- Location: `tt_metal/distributed/`
- Contains: Multi-device command queues, mesh workloads, socket-based communication
- Key subdirectories:
  - `distributed/multihost/` - Cross-host synchronization via sockets
  - `distributed/flatbuffer/` - Serialization for remote execution
- Abstractions: MeshDevice, MeshCommandQueue, MeshWorkload, MeshEvent
- Depends on: Fabric layer, networking (TCP/Unix sockets)
- Used by: TTNN distributed training/inference

**Kernel & Firmware Layer:**
- Location: `tt_metal/hw/` and `tt_metal/kernels/`
- Contains:
  - Device-side firmware and kernel templates
  - `hw/firmware/` - Bootcode and runtime firmware
  - `hw/inc/` - Device-side headers (memory layouts, core constants)
  - `hw/ckernels/` - Compiler kernels (LLK-based)
  - `kernels/compute/` and `kernels/dataflow/` - Kernel templates
- Pattern: Static C headers for device code; JIT compilation to binary at runtime
- Depends on: Hardware architecture definitions (WH, BH specifications)
- Used by: JIT build system, kernel compilation

**Common & Infrastructure:**
- Location: `tt_metal/common/`, `tt_metal/hostdevcommon/`
- Contains: Shared data structures, core assignments, coordinate systems
- Patterns: Coordinate types (CoreCoord, MeshCoord), layouts, constants
- Depends on: None (foundational)
- Used by: All layers

**Hardware Abstraction Layer (HAL):**
- Location: `tt_metal/api/tt-metalium/hal.hpp` (interface) → `impl/` (implementations)
- Purpose: Abstracts hardware differences across WH/BH architectures
- Provides: Memory address calculations, core type mappings, device capabilities
- Depends on: UMD hardware descriptors

## Data Flow

**Program Execution Flow:**

1. **Host Setup** (User code)
   - Device creation via `CreateDevice(device_id)`
   - Buffer allocation via `CreateBuffer(config)` (L1, DRAM, interleaved, or sharded)
   - Program construction via `Program()` with kernels and circular buffers

2. **Program Compilation** (`impl/program/program.cpp`)
   - Kernel JIT compilation: device code → binary via `jit_build/build.hpp`
   - Circular buffer allocation and configuration
   - Runtime args serialization
   - Generates dispatch commands and kernel binaries

3. **Dispatch Phase** (`impl/dispatch/`, `impl/device/dispatch.cpp`)
   - Program enqueue to command queue
   - HAL translates virtual→physical coordinates
   - Dispatch core receives commands and configures kernels
   - Runtime args written to device SRAM

4. **Device Execution**
   - Compute cores execute kernel binary on tiles
   - Data movement cores (ERISC) orchestrate eth/DRAM transfers
   - Circular buffers manage kernel I/O synchronization

5. **Readback** (User code)
   - `Buffer::read()` → PCIe/tunnel back to host

**State Management:**

- **Program State**: Owned by `detail::ProgramImpl` (shared_ptr allows persistence across Program object lifetime)
- **Device State**: Tracked in `Device` object (allocator, kernel cache, dispatch queue state)
- **Buffer State**: Tracked in `Buffer` object (address, size, shard mapping) and allocator
- **Circular Buffer State**: Managed per-program via `CircularBuffer` objects
- **Synchronization**: Command queue tracks completion via events; semaphores for kernel coordination

**Multi-Device Execution:**

1. **Mesh Construction** (User code via TTNN)
   - Create MeshDevice with per-device Programs
   - Fabric builder configures eth routing between devices

2. **Fabric Initialization** (`fabric/fabric_init.cpp`)
   - Build routing tables based on mesh topology
   - Allocate fabric channels for inter-device transfers
   - Spawn router/datamover kernels on ERISC cores

3. **Workload Execution** (MeshDevice)
   - Enqueue per-device workload segments
   - Synchronized via command queue or mesh events
   - Fabric handles data routing transparently

## Key Abstractions

**Device (IDevice):**
- Purpose: Represents a single physical Tenstorrent chip
- Examples: `Device` (single-device), `SubDevice` (logical partition of Device)
- Pattern: Virtual interface for hardware abstraction; implementations handle WH/BH differences
- Methods: allocate buffers, enqueue programs, query hardware state
- Location: `api/tt-metalium/device.hpp`, `impl/device/device.hpp`

**Program:**
- Purpose: Encapsulates a kernel execution graph with I/O buffers
- Contains: Kernels, circular buffers, runtime args, dispatch commands
- Pattern: Builder pattern (add kernels + CBs) → immutable at execution
- Location: `api/tt-metalium/program.hpp`, `impl/program/program_impl.hpp`
- Cached: Per-device LRU cache to avoid recompilation

**Buffer:**
- Purpose: Represents device memory (DRAM or L1)
- Variants: Interleaved (striped across cores), Sharded (contiguous per core), Distributed (across mesh)
- Layouts: Supported layouts (HEIGHT_SHARDED, WIDTH_SHARDED, BLOCK_SHARDED, SINGLE_CORE)
- Location: `api/tt-metalium/buffer.hpp`, `impl/buffers/buffer.cpp`
- Allocator: Multi-strategy allocator (`impl/allocator/`) supporting L1 banking

**CircularBuffer:**
- Purpose: Synchronization primitive for kernel I/O within a program
- Properties: Tied to program, has reader/writer config, format descriptors
- Lifecycle: Created as part of program, deallocated when program ends
- Location: `api/tt-metalium/circular_buffer.hpp`, `impl/buffers/circular_buffer.hpp`

**MeshDevice (Distributed):**
- Purpose: Logical aggregation of physical devices in a mesh topology
- Contains: Per-device workload, mesh command queue for synchronization
- Pattern: Map of (device_id → SubDevice)
- Location: `distributed/mesh_device.hpp`, `distributed/mesh_device.cpp`

**Fabric (Multi-Device Routing):**
- Purpose: Interconnect configuration for automatic data routing
- Responsibilities:
  - Channel allocation for eth links
  - Router kernel generation (on ERISC cores)
  - Bandwidth/latency optimization
- Pattern: Builder constructs fabric; runtime context executes routing
- Location: `fabric/fabric.hpp`, `fabric/fabric_builder.cpp`

## Entry Points

**Device Creation:**
- Location: `api/tt-metalium/host_api.hpp` → `CreateDevice(device_id)`
- Implementation: `impl/device/device.cpp` → `Device::Device()`
- Triggers: User calling GetDevice() or CreateDevice()
- Responsibilities: Hardware initialization, allocator setup, dispatch queue creation

**Program Execution:**
- Location: `api/tt-metalium/host_api.hpp` → `EnqueueProgram(device, program)`
- Implementation: `impl/device/dispatch.cpp` → `Device::enqueue_program()`
- Triggers: User explicitly enqueuing program for device
- Responsibilities: Kernel compilation, CB allocation, command queue population

**Kernel Dispatch:**
- Location: `impl/dispatch/hardware_command_queue.cpp`
- Mechanism: Command queue writes dispatch commands to device SRAM
- Dispatch core (firmware) reads commands and configures worker cores
- Pattern: Ring buffer of commands; dispatch core polls and executes

## Error Handling

**Strategy:** Exceptions for API errors; asserts for internal invariants

**Patterns:**

- **Allocation Failures:** Throw `std::exception` with detailed error message (buffer size exceeds available memory)
- **Invalid Arguments:** Throw with enum to specific error code; include bounds/constraints
- **Device Communication:** Checked returns from UMD calls; propagate error to caller
- **Kernel Compilation:** JIT errors captured, logged, rethrown with backtrace
- **Coordinate/Layout Validation:** Precondition asserts in internal functions; user-facing validation before device ops

**Example:** `buffer.cpp` validates shard spec against core range; throws if core_range > device grid

## Cross-Cutting Concerns

**Logging:**
- Framework: `tt-logger` (custom logger in `tt_metal/logging/`)
- Pattern: Global logger instance; debug/info/error levels
- Usage: Device init, kernel compilation, dispatch events

**Validation:**
- Coordinate validation: CoreCoord/MeshCoord bounds checked against device grid
- Buffer layout validation: Shard parameters must fit available cores
- Kernel configuration: Number of CBs <= max per-core; ring buffer sizes fit L1
- Pattern: Validation in constructors/setters; early error feedback

**Authentication/Authorization:**
- Not applicable: Single-user host process model; no credential management

**Concurrency:**
- Model: Single-threaded host API; async on hardware (dispatch queue runs in background)
- Synchronization: Mutexes on shared state (allocator, program cache, device state)
- Events: Fabric context uses condition variables for inter-device sync
- No locks held during device I/O (async dispatch via UMD)

---

*Architecture analysis: 2026-03-16*
