# Codebase Structure

**Analysis Date:** 2026-03-16

## Directory Layout

```
tt-metal-2/
├── tt_metal/                           # Core low-level runtime & APIs
│   ├── api/
│   │   └── tt-metalium/               # Public C++ API headers (device, program, buffer)
│   ├── impl/                          # Implementation of API + runtime
│   │   ├── device/                    # Device class and initialization
│   │   ├── program/                   # Program compilation and dispatch
│   │   ├── dispatch/                  # Command queues and hardware communication
│   │   ├── allocator/                 # Memory allocation (L1 banking, sharding)
│   │   ├── buffers/                   # Buffer and circular buffer implementations
│   │   ├── kernels/                   # Kernel metadata and types
│   │   ├── context/                   # Global metal context (HAL, cluster state)
│   │   ├── debug/                     # Watcher, inspector, performance tools
│   │   ├── event/                     # Event synchronization primitives
│   │   ├── sub_device/                # SubDevice (logical device partitions)
│   │   └── experimental/              # Experimental features
│   ├── fabric/                        # Multi-device ethernet routing
│   │   ├── builder/                   # Fabric construction (channel allocation)
│   │   ├── impl/                      # Fabric runtime context
│   │   ├── ccl/                       # Collective communication kernels
│   │   ├── protobuf/                  # Message serialization for fabric
│   │   ├── debug/                     # Channel trimming, bandwidth profiling
│   │   └── hw/                        # Fabric hardware configuration
│   ├── distributed/                   # Multi-mesh and multi-host execution
│   │   ├── multihost/                 # Socket-based cross-host communication
│   │   ├── flatbuffer/                # Serialization for remote workloads
│   │   └── mesh_*.cpp/.hpp            # MeshDevice, MeshCommandQueue, MeshWorkload
│   ├── hw/                            # Device-side firmware and kernel templates
│   │   ├── firmware/                  # Bootcode, ETH/DRAM driver firmware
│   │   ├── inc/                       # Device-side C headers (memory, constants)
│   │   ├── ckernels/                  # Compute kernels (LLK based)
│   │   └── toolchain/                 # Cross-compilation tools
│   ├── kernels/                       # Kernel code templates
│   │   ├── compute/                   # Compute core kernels (matrix ops, etc)
│   │   └── dataflow/                  # Data movement kernels (DRAM mover, eth)
│   ├── common/                        # Shared utilities
│   │   ├── core_assignment.hpp        # Core coordinate assignment logic
│   │   └── stable_hash.hpp            # Hash functions for caching
│   ├── hostdevcommon/                 # Shared host-device structures
│   │   ├── kernel_structs.h           # Device-visible runtime arg structures
│   │   └── common_values.hpp          # Shared constants
│   ├── jit_build/                     # JIT compilation system
│   ├── logging/                       # Logging framework
│   ├── llrt/                          # Low-level runtime (cluster, MMIO)
│   ├── graph/                         # Deprecated: Graph mode execution
│   ├── test/                          # Internal unit tests
│   ├── tools/                         # Development tools
│   ├── soc_descriptors/               # Hardware SOC configs (WH, BH specs)
│   └── third_party/                   # UMD submodule (hardware driver)
├── ttnn/                              # High-level neural network operations
│   ├── api/ttnn/                      # TTNN public API
│   ├── tt_lib/                        # Operations library (fallback, fused)
│   ├── core/                          # Tensor abstraction and graph
│   ├── cpp/                           # C++ operation implementations
│   └── test/                          # TTNN tests
├── tests/                             # Test suite
│   ├── tt_metal/
│   │   ├── tt_metal/                  # Device/program/buffer unit tests
│   │   ├── tt_fabric/                 # Fabric topology and routing tests
│   │   ├── distributed/               # Multi-device distributed tests
│   │   ├── microbenchmarks/           # Performance microbenchmarks
│   │   └── test_utils/                # Test utilities and fixtures
│   └── tt_neural_network_model/       # Model reference tests
├── runtime/                           # Binary utilities (firmware loader, etc)
├── scripts/                           # Build and utility scripts
├── cmake/                             # CMake build infrastructure
├── docs/                              # Sphinx documentation
├── models/                            # LLM/transformer model implementations
├── tt-train/                          # Training library
└── CMakeLists.txt                     # Root CMake build file
```

## Directory Purposes

**tt_metal/api/tt-metalium/:**
- Purpose: Public-facing C++ API headers
- Contains: All `.hpp` files that users include (device.hpp, program.hpp, buffer.hpp, etc.)
- Key files: `host_api.hpp` (free functions), `device.hpp` (IDevice interface), `tt_metal.hpp` (convenience wrapper)
- Pattern: Headers only; no implementation; stable ABI

**tt_metal/impl/device/:**
- Purpose: Device lifecycle and core runtime
- Contains: `Device` class implementation, device manager, dispatch entry points
- Key files:
  - `device.cpp` - Device constructor, program enqueue, memory queries
  - `device_manager.cpp` - Device pool and lifecycle
  - `dispatch.cpp` - Command dispatch coordination
  - `device_impl.hpp` - Implementation details (pimpl)

**tt_metal/impl/program/:**
- Purpose: Program compilation and kernel assembly
- Contains: Program execution pipeline from Program object to device commands
- Key files:
  - `program.cpp` - Program construction, kernel addition, CB configuration
  - `program_impl.hpp` - Internal program state
  - `dispatch.cpp` - Program-to-command translation
  - `program_descriptors.cpp` - Descriptor-based program creation

**tt_metal/impl/dispatch/:**
- Purpose: Command queue and hardware communication
- Contains: Ring buffer management, device command generation, event tracking
- Key files:
  - `hardware_command_queue.cpp` - Main CQ implementation
  - `command_queue_common.hpp` - CQ constants and types
  - `device_command.hpp` - Command structure definitions
  - `dispatch_core_common.hpp` - Dispatch core constants

**tt_metal/impl/allocator/:**
- Purpose: Device memory allocation with policy support
- Contains: L1 banking allocator, sharded buffer support, DRAM interleaving
- Key files:
  - `allocator.cpp` - Public allocation interface
  - `l1_banking_allocator.hpp` - L1 partitioning strategy
  - `allocator_state.hpp` - Allocator state for buffer lifecycle

**tt_metal/fabric/:**
- Purpose: Multi-device ethernet interconnect management
- Contains: Topology-aware fabric construction, channel allocation, bandwidth accounting
- Key files:
  - `fabric.cpp` - Main fabric API
  - `fabric_builder.cpp` - Constructs routing fabric
  - `fabric_context.cpp` - Runtime fabric state
  - `control_plane.cpp` - Multi-device coordination

**tt_metal/distributed/:**
- Purpose: Distributed execution across multiple devices/hosts
- Contains: MeshDevice abstraction, mesh command queues, socket communication
- Key files:
  - `mesh_device.cpp` - Multi-device workload container
  - `mesh_command_queue_base.cpp` - Base CQ for mesh
  - `mesh_socket.cpp` - Socket-based inter-device communication
  - `multihost/*.cpp` - Cross-host synchronization

**tt_metal/hw/:**
- Purpose: Device-side code (firmware, kernel templates)
- Contains:
  - `firmware/` - RISC-V firmware (bootcode, ETH driver)
  - `inc/` - Device-visible constants and structures
  - `ckernels/` - LLK compute kernel implementations
  - `toolchain/` - Cross-compilation toolchain
- Note: Code here is compiled to device binaries, not host code

**tt_metal/common/:**
- Purpose: Shared data structures across layers
- Key files:
  - `core_assignment.hpp` - Logic for core grid assignment
  - `stable_hash.hpp` - Deterministic hashing for caching
- Pattern: Headers only; foundational types

**tests/tt_metal/tt_metal/:**
- Purpose: Device/program/buffer unit tests
- Structure:
  - `data_movement/` - Buffer copy and reshard tests (loopback, conv_hardcoded, etc.)
  - `eth/` - Ethernet routing tests
  - `common/` - Common test fixtures
  - `lightmetal/` - Program serialization tests
- Pattern: C++ source + Python fixtures; CMake targets for each test

**tests/tt_metal/tt_fabric/:**
- Purpose: Fabric-specific tests (routing, topology, CCL)
- Structure:
  - `fabric_router/` - Router kernel and topology tests
  - `fabric_data_movement/` - Inter-device data transfer tests
  - `custom_mesh_descriptors/` - Custom topology tests
  - `benchmark/` - Fabric throughput microbenchmarks

## Key File Locations

**Entry Points:**
- `tt_metal/api/tt-metalium/host_api.hpp` - Free function API (CreateDevice, EnqueueProgram, CreateBuffer)
- `tt_metal/api/tt-metalium/device.hpp` - Device interface (IDevice, Device)
- `tt_metal/impl/device/device.cpp` - Device implementation

**Configuration:**
- `tt_metal/soc_descriptors/` - Hardware SOC configuration files (*.yaml for WH, BH)
- `cmake/project_options.cmake` - Build options (ENABLE_UNITY_BUILD, etc.)
- `CMakePresets.json` - CMake presets for build configuration

**Core Logic:**
- `tt_metal/impl/program/program.cpp` - Program execution pipeline
- `tt_metal/impl/allocator/allocator.cpp` - Memory allocation
- `tt_metal/impl/dispatch/hardware_command_queue.cpp` - Command dispatch
- `tt_metal/fabric/fabric_builder.cpp` - Fabric construction

**Testing:**
- `tests/tt_metal/tt_metal/` - Device/program tests
- `tests/tt_metal/tt_fabric/` - Fabric/distributed tests
- `conftest.py` - Pytest configuration and fixtures

## Naming Conventions

**Files:**
- Implementation: `name.cpp` (definitions) paired with `name.hpp` (declarations)
- Public headers: Located in `api/` or `api/tt-metalium/`
- Internal headers: Located in `impl/` subdirectories
- Pattern: CamelCase for public APIs (Device, Program, Buffer), snake_case for internal helpers

**Directories:**
- Layer directories: `impl/`, `fabric/`, `distributed/` (lowercase, module-focused)
- Feature directories: `ccl/`, `debug/`, `builder/` (describe purpose)
- Test directories: Match source structure; prefix with `test_` or suffix `/tests`
- Pattern: Hierarchy follows logical grouping (feature → subsystem → utility)

**Classes:**
- Public: PascalCase (Device, Program, Buffer, IDevice)
- Internal impl: ProgramImpl, DeviceImpl (pimpl pattern)
- Interfaces: Prefix with 'I' (IDevice, ICommandQueue)

**Functions:**
- Public free functions: PascalCase (CreateDevice, EnqueueProgram)
- Member functions: camelCase (enqueue_program, create_buffer)
- Internal: snake_case (get_available_cores, allocate_l1)

**Constants:**
- Enum values: PascalCase (BufferType::DRAM, CoreType::WORKER)
- Macro constants: UPPER_CASE (MAX_CORES_PER_DEVICE)
- Type aliases: PascalCase (ChipId, CoreCoord)

## Where to Add New Code

**New Feature (e.g., new operation):**
- Primary code: `ttnn/` (if high-level) or `tt_metal/impl/` (if low-level hardware)
- Tests: Mirror source structure under `tests/` (e.g., new op → `tests/ttnn/operations/`)
- Build: Add CMakeLists.txt target; register in parent CMakeLists.txt

**New Device Management Feature:**
- Implementation: `tt_metal/impl/device/` if modifying device lifecycle
- Public API: `tt_metal/api/tt-metalium/` (new header or extend device.hpp)
- Tests: `tests/tt_metal/tt_metal/common/` or dedicated test directory

**New Multi-Device Feature:**
- Implementation: `tt_metal/fabric/` (if routing) or `tt_metal/distributed/` (if execution)
- Public API: `tt_metal/api/tt-metalium/mesh_device.hpp` or `distributed.hpp`
- Tests: `tests/tt_metal/tt_fabric/` or `tests/tt_metal/distributed/`

**New Kernel/Compute Feature:**
- Device-side code: `tt_metal/hw/` (firmware or templates)
- Host-side wrapper: `tt_metal/impl/kernels/` or TTNN operation
- Build: Register in `tt_metal/hw/CMakeLists.txt`; wire up JIT compilation in `impl/program/`

**Utilities:**
- Shared helpers: `tt_metal/common/`
- Device-host common: `tt_metal/hostdevcommon/`
- Internal only: Keep in subsystem directory (e.g., `fabric/builder/internal.hpp`)

## Special Directories

**tt_metal/third_party/umd/:**
- Purpose: User Mode Driver (hardware abstraction from Tenstorrent)
- Generated: No (git submodule)
- Committed: No (external dependency)
- Usage: Imported via CMake; provides hardware API (cluster descriptors, device types)

**build/ (symlink to build_Release):**
- Purpose: CMake build artifacts
- Generated: Yes (populated by CMake)
- Committed: No (.gitignored)
- Build output: `build/test/` contains test binaries, `build/lib/` contains libraries

**generated/:**
- Purpose: Generated code from protobuf/flatbuffer schemas
- Generated: Yes (by build system)
- Committed: Yes (so builds work offline)
- Triggers: Changes to `.proto` or `.fbs` files in `fabric/protobuf/`, `distributed/flatbuffer/`

**.cache/tt-metal-cache/:**
- Purpose: JIT compilation kernel cache
- Generated: Yes (at runtime by kernel compilation)
- Committed: No (.gitignored)
- Purge: Remove to force recompilation after `.hpp` changes in kernel headers

---

*Structure analysis: 2026-03-16*
