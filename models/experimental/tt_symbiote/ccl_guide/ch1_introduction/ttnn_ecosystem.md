# 1.3 CCL in the TTNN Ecosystem

## The tt-metal Stack

tt-metal is the full software stack for Tenstorrent hardware. It is layered as follows, from lowest to highest abstraction:

```
┌─────────────────────────────────────────────────────────────┐
│                   User Python Code                          │
│         (model definitions, training loops, etc.)           │
├─────────────────────────────────────────────────────────────┤
│                    TTNN Python API                           │
│        ttnn.all_gather, ttnn.matmul, ttnn.linear, ...        │
├─────────────────────────────────────────────────────────────┤
│              TTNN C++ Operation Layer                        │
│   CCL ops, Matmul ops, Elementwise ops, ...                  │
│   (program factories, device operations, nanobind glue)      │
├─────────────────────────────────────────────────────────────┤
│              tt-metal Runtime                                │
│   Device management, command queue, program compilation      │
├─────────────────────────────────────────────────────────────┤
│              tt-fabric                                       │
│   Ethernet link abstraction, Topology enum, routing tables   │
├─────────────────────────────────────────────────────────────┤
│              Hardware (ERISC, Tensix, NOC, DRAM)             │
└─────────────────────────────────────────────────────────────┘
```

CCL lives primarily in the **TTNN C++ Operation Layer**. The Python API is a thin nanobind wrapper over C++ implementations. The C++ implementations are themselves composed of device operations (program factories + kernel code) that run on Tensix and ERISC cores, using tt-fabric's Ethernet primitives for inter-chip transfers.

---

## Directory Structure

### Core CCL Operations

```
ttnn/cpp/ttnn/operations/ccl/
├── all_gather/
│   ├── all_gather.hpp                  # C++ op declaration
│   ├── all_gather.cpp                  # C++ op implementation
│   ├── all_gather_nanobind.cpp         # Python bindings
│   └── device/
│       ├── all_gather_device_operation.hpp
│       ├── all_gather_device_operation.cpp
│       └── all_gather_program_factory.cpp   # Creates the Metal program
│
├── reduce_scatter/
│   ├── reduce_scatter.hpp/.cpp
│   ├── reduce_scatter_nanobind.cpp
│   └── device/
│       ├── reduce_scatter_device_operation.hpp
│       ├── reduce_scatter_device_operation.cpp
│       └── reduce_scatter_program_factory.cpp
│
├── all_reduce/
│   ├── all_reduce.hpp/.cpp
│   ├── all_reduce_nanobind.cpp
│   └── device/ ...
│
├── broadcast/
├── all_broadcast/
├── reduce_to_root/
├── mesh_partition/
├── all_to_all_dispatch/
├── all_to_all_combine/
│
├── common/
│   ├── types/
│   │   └── ccl_types.hpp               # Core type definitions (see below)
│   ├── uops/
│   │   └── ccl_command.hpp             # Command structures for kernel communication
│   ├── kernels/                        # Shared kernel utilities
│   └── ...
│
├── kernels/
│   └── edm/
│       └── erisc_datamover.cpp         # The EDM kernel itself
│
└── shared_with_host/                   # Structs visible to both host and device code
    ├── hetergeneous_data_structs.hpp   # Heterogeneous data structures
    └── sharded_tensor_addr_gen.hpp     # Sharded tensor address generation
# Note: ccl_common.hpp lives at the top-level ccl/ directory, not in shared_with_host/
```

### Experimental Async Operations

```
ttnn/cpp/ttnn/operations/experimental/ccl/
├── all_gather_async/
│   ├── all_gather_async.hpp/.cpp
│   └── device/ ...
├── all_reduce_async/
├── reduce_scatter_minimal_async/
├── all_to_all_async/
├── all_gather_matmul_async/            # Fused: AllGather + Matmul
├── matmul_reduce_scatter_async/        # Fused: Matmul + ReduceScatter
├── llama_reduce_scatter_matmul/        # Llama-specific fused op
├── ring_attention_all_gather_async/
├── rms_allgather/
└── deepseek_minimal_broadcast/
```

> **Gotcha:** The `experimental/ccl/` directory has no stability guarantees. File names, op signatures, and config structs here are subject to change without notice between tt-metal releases. Treat them as reference implementations rather than stable API surfaces.

---

## Key C++ Types

Understanding the following types is essential when reading CCL source code, writing custom ops, or debugging issues that require looking past the Python API.

### `tt::tt_fabric::Topology`

```cpp
// Defined in tt_metal/api/tt-metalium/experimental/fabric/fabric_edm_types.hpp
// surfaced through CCL (namespace is tt::tt_fabric in that header)
enum class Topology {
    NeighborExchange = 0,  // Single-hop neighbor exchange
    Linear           = 1,  // Chain topology, no wrap-around
    Ring             = 2,  // Ring topology with wrap-around link
    Mesh             = 3,  // 2-D mesh
    Torus            = 4,  // 2-D torus (mesh + wrap-around in both axes)
};
// Only Linear and Ring are currently exposed through the CCL Python API
// as ttnn.Topology.Linear and ttnn.Topology.Ring.
```

This enum is what you pass as `topology=ttnn.Topology.Ring` in Python. The mapping is done in the nanobind layer.

At the kernel level, the EDM uses the topology value to decide whether to configure a "return path" channel. In Ring mode both directions are active; in Linear mode only one direction is established.

### `EriscDatamoverConfig`

Configures how each EDM channel is set up. The definition is in:

```
ttnn/cpp/ttnn/operations/ccl/ccl_host_datastructures.hpp
```

The actual struct (from `namespace ttnn::ccl`) manages L1 layout for the ERISC core rather than exposing simple named fields for buffer count or semaphore IDs. Its real signature is:

```cpp
// Actual definition — ttnn/cpp/ttnn/operations/ccl/ccl_host_datastructures.hpp
struct EriscDatamoverConfig {
    // Total unreserved L1 available on the ERISC core (populated from HAL at construction)
    std::size_t total_l1_buffer_space = tt::tt_metal::hal::get_erisc_l1_unreserved_size();
    // Base address of the unreserved L1 region on the ERISC core
    std::size_t usable_l1_base_address = tt::tt_metal::hal::get_erisc_l1_unreserved_base();

    // Layout constants (all sizes in bytes, all are multiples of eth_word_size_bytes)
    static constexpr std::size_t semaphore_size = 32;
    static constexpr std::size_t handshake_location_size = 16;   // one Ethernet word
    static constexpr std::size_t handshake_padding_multiple = 3;
    static constexpr std::size_t eth_channel_sync_size_bytes = 16;
    static constexpr std::size_t eth_word_size_bytes = 16;

    // Helper methods that compute addresses within the ERISC L1 layout
    uint32_t get_edm_handshake_address() const;
    uint32_t get_semaphores_base_address(std::size_t num_edm_channels) const;
    uint32_t get_buffers_base_address(std::size_t num_edm_channels) const;
    uint32_t compute_buffer_size(
        std::size_t num_edm_channels,
        std::size_t num_buffers_per_channel = 1,
        uint32_t page_size = eth_word_size_bytes);
};
```

The struct is primarily a **L1 address calculator**: given a channel count, it answers "where in ERISC L1 do the semaphores start?" and "how large can each buffer slot be?". Buffer count and per-channel sizes are passed separately to `EriscDatamoverBuilder` (see below), not stored in `EriscDatamoverConfig` itself.

**When you need to touch this:** When writing a custom program factory that manually sets up EDM channels, call `get_buffers_base_address` and `compute_buffer_size` to derive L1 addresses before passing them to `EriscDatamoverBuilder`. For standard TTNN ops the program factory handles this automatically.

### `CCLOpConfig`

A per-operation configuration class that wraps the input/output tensors and topology for a CCL op, providing accessor methods to program factories. Its real interface (from `ccl_host_datastructures.hpp`) is:

```cpp
// Actual definition — ttnn/cpp/ttnn/operations/ccl/ccl_host_datastructures.hpp
// Note: this is a class with private data members, not a plain struct.
// All access is through const accessor methods.
class CCLOpConfig {
public:
    CCLOpConfig(
        std::vector<Tensor>& input_tensors,
        const std::vector<Tensor>& output_tensors,
        Topology topology);

    // Accessors used by program factories
    uint32_t  get_page_size()    const;
    tt::tt_metal::Tile get_tile() const;
    Topology  get_topology()     const;
    bool      is_input_sharded() const;
    bool      is_output_sharded() const;
    Tensor const& get_input_tensor(std::size_t i)  const;
    Tensor const& get_output_tensor(std::size_t i) const;

    // Emit #define strings consumed by kernel compile-time args
    std::map<std::string, std::string> emit_worker_defines() const;
};
```

The class is an **opaque accessor pattern**: program factories query it via getters rather than reading public members directly. This insulates factory code from changes to tensor storage layout.

### `GlobalSemaphore`

A multi-device synchronization primitive. A `GlobalSemaphore` is backed by a semaphore address that is mapped into the L1 of every participating device simultaneously. It allows:

- Tensix cores to signal ERISC cores that data is ready to send
- ERISC cores to signal Tensix cores that received data is available
- Cross-chip barrier synchronization between collectives

In Python:

```python
# Create a semaphore shared across all devices in the mesh.
# core_range_set specifies which cores on each device host the semaphore;
# 0 is the initial semaphore value.
core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
sem = ttnn.create_global_semaphore(mesh_device, core_range_set, 0)

# Pass to an async op
result = ttnn.all_gather_async(
    tensor,
    dim=0,
    multi_device_global_semaphore=sem,
    topology=ttnn.Topology.Ring,
)
```

> **Gotcha:** `GlobalSemaphore` objects are tied to a specific `MeshDevice` instance and cannot be reused across different mesh views. If you reshape or re-slice your mesh, allocate a fresh semaphore.

### `Shape4D<T>`

A typed 4-D shape descriptor used throughout CCL kernel code:

```cpp
template <typename T>
struct Shape4D {
    T w;  // batch dimension
    T z;  // channel / head dimension
    T y;  // height / sequence length
    T x;  // width / hidden dim
};

// Common instantiations
using TensorShape = Shape4D<uint32_t>;
using TileShape   = Shape4D<uint32_t>;
```

CCL kernels use `Shape4D` rather than raw `std::array<uint32_t, 4>` to make dimension semantics explicit and to catch transposition errors at compile time.

### `ttnn::ccl::EriscDatamoverBuilder`

A host-side utility class that constructs the configuration tables sent to ERISC cores before a CCL op runs:

```cpp
// Key public methods (from ccl_host_datastructures.hpp)
class EriscDatamoverBuilder {
public:
    // Returns a ChannelBufferInterface with the L1 buffer/semaphore addresses
    // for the newly registered sender channel.
    [[nodiscard]]
    ChannelBufferInterface add_sender_channel(
        uint32_t worker_semaphore_id,
        uint32_t num_eth_messages_to_forward,
        std::vector<ccl::WorkerXY> const& worker_coords,
        uint32_t expected_message_size_bytes = 0);

    // Analogous to add_sender_channel for the receive side.
    [[nodiscard]]
    ChannelBufferInterface add_receiver_channel(
        uint32_t worker_semaphore_id,
        uint32_t num_eth_messages_to_forward,
        std::vector<ccl::WorkerXY> const& worker_coords,
        uint32_t expected_message_size_bytes = 0);

    // Build the flat uint32_t runtime-args vector passed to the EDM kernel.
    [[nodiscard]]
    std::vector<uint32_t> get_runtime_args() const;

    // Build compile-time args for the EDM kernel (indexed by RISC id).
    [[nodiscard]]
    std::vector<uint32_t> get_compile_time_args(uint32_t risc_id) const;

    // Query the configured per-channel buffer size in bytes.
    [[nodiscard]]
    uint32_t get_eth_buffer_size_bytes() const;
};
```

If you are writing a program factory that needs to set up EDM channels manually, this is the class to use. Standard CCL ops handle this internally.

---

## The Python API Layer

### How Ops Are Registered

Each CCL op has a `*_nanobind.cpp` file that registers it with the TTNN Python module via a `bind_<op>()` function called from the central TTNN nanobind entry point. Users do not interact with this layer directly.

### Python API Quick Reference

```python
import ttnn

# AllGather: replicate sharded tensor across all devices
result = ttnn.all_gather(
    input_tensor,              # TT tensor, sharded across devices
    dim,                       # int: dimension along which tensor is sharded
    cluster_axis=None,         # Optional[int]: mesh axis (None = all devices)
    num_links=None,            # Optional[int]: number of Ethernet links to use
    topology=None,             # Optional[ttnn.Topology]: Ring or Linear
    memory_config=None,        # Optional[ttnn.MemoryConfig]: output memory layout
)

# ReduceScatter: reduce + scatter across devices
result = ttnn.reduce_scatter(
    input_tensor,
    dim,                       # int: dimension to scatter output across
    cluster_axis=None,
    num_links=None,
    topology=None,
    memory_config=None,
)

# AllReduce: reduce and replicate
result = ttnn.all_reduce(
    input_tensor,
    cluster_axis=None,
    num_links=None,
    topology=None,
    memory_config=None,
)

# Broadcast: one sender → all receivers
result = ttnn.broadcast(
    input_tensor,
    sender_coord,              # ttnn.MeshCoordinate: mesh coordinate of the sender
    topology=ttnn.Topology.Linear,
    memory_config=None,
)
```

---

## How a CCL Call Flows Through the Stack

A `ttnn.all_gather(...)` call traverses these layers in order:

1. **Python → nanobind** (`all_gather_nanobind.cpp`): converts Python args to C++ types.
2. **C++ op entry point** (`all_gather.cpp`): validates args, resolves `cluster_axis` and `num_links`, constructs `CCLOpConfig`.
3. **Program factory** (`all_gather_program_factory.cpp`): creates the Metal Program, configures `EriscDatamoverBuilder` channels, compiles Tensix dataflow kernels, sets runtime args.
4. **tt-metal Runtime**: enqueues the Program on the device's `CommandQueue` and dispatches compiled kernel binaries.
5. **Hardware**: Tensix kernels DMA tensor shards to ERISC L1 via NOC; EDM streams data over Ethernet to the remote chip's ERISC L1, then NOC to remote Tensix L1.
6. **Return**: `GlobalSemaphore` signals completion; Python call returns the result tensor.

The program factory (step 3) is where buffer sizes, kernel configurations, and channel assignments are chosen — that is the layer to modify when tuning.

---

## Finding Your Way Around the Codebase

The following table maps common questions to the files that answer them:

| Question | Where to look |
|----------|--------------|
| What arguments does `all_gather` accept in C++? | `ttnn/cpp/ttnn/operations/ccl/all_gather/all_gather.hpp` |
| How are EDM channels and Tensix dataflow kernels configured for AllGather? | `ttnn/cpp/ttnn/operations/ccl/all_gather/device/all_gather_program_factory.cpp` |
| How does the EDM kernel handle flow control? | `ttnn/cpp/ttnn/operations/ccl/kernels/edm/erisc_datamover.cpp` |
| What are the core CCL types (Topology, etc.)? | `ttnn/cpp/ttnn/operations/ccl/common/types/ccl_types.hpp` |
| How are commands structured between host and kernel? | `ttnn/cpp/ttnn/operations/ccl/common/uops/ccl_command.hpp` |
| How is the experimental async AllGather different? | `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/` |
| Where is the MoE AllToAll logic? | `ttnn/cpp/ttnn/operations/ccl/all_to_all_dispatch/` and `all_to_all_combine/` |
| What shared structs are used by both host and device? | `ttnn/cpp/ttnn/operations/ccl/shared_with_host/` |

---

## Memory Layout Considerations

CCL operations impose constraints on input tensor memory layout:

### Sharding and Interleaving

CCL accepts both **interleaved** and **sharded** tensors, but performance differs significantly:

- **Interleaved** (default): tensor pages are distributed across DRAM banks in round-robin. Reading a contiguous slice for transmission requires gathering from multiple DRAM banks — adds latency.
- **Width-sharded / height-sharded**: tensor data for each device is contiguous in L1. CCL can issue large, sequential NOC transfers — much higher effective bandwidth.

```python
# Create an L1-sharded tensor for optimal CCL performance
shard_config = ttnn.create_sharded_memory_config(
    shape=tensor.shape,
    core_grid=ttnn.CoreGrid(r=4, c=8),
    strategy=ttnn.ShardStrategy.WIDTH,
)
tensor_sharded = ttnn.to_memory_config(tensor, shard_config)

# Now AllGather operates on L1-local data — minimal DRAM traffic
result = ttnn.all_gather(tensor_sharded, dim=3, topology=ttnn.Topology.Ring)
```

> **Gotcha:** Not all CCL ops support all shard strategies. Check the program factory source for the specific op to see which `TensorMemoryLayout` values it handles. Passing an unsupported layout will raise a `RuntimeError` with a message like "Unsupported memory layout for all_gather."

### Output Memory Config

By default, CCL ops produce output tensors with the same memory config as the input. Use the `memory_config` parameter to request a specific layout for the output — useful when the result feeds directly into an op that requires a particular layout:

```python
output_mem_config = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.INTERLEAVED,
    ttnn.BufferType.L1,
)
result = ttnn.all_reduce(partial_output, memory_config=output_mem_config)
```

---

## Relationship to `MeshDevice`

All CCL ops operate on tensors that live on a `MeshDevice`, a logical handle to a 2-D grid of physical devices. The `MeshDevice` shape (rows × columns) must match the physical hardware topology — attempting to create a 2×4 mesh on 4 devices wired in a chain will fail at device initialization. Full `MeshDevice` setup examples are in Chapter 2.

---

## Common Error Messages and Their Meanings

| Error | Likely cause |
|-------|-------------|
| `Collective timed out after N seconds` | Topology mismatch (Ring specified but no physical wrap-around), or one device crashed |
| `Unsupported memory layout for all_gather` | Input tensor is in a shard strategy the op does not handle |
| `Invalid cluster_axis: 1, mesh has only 1 dimension` | Passing `cluster_axis=1` to a 1-D mesh (e.g., a 1xN `MeshDevice`) |
| `num_links exceeds available Ethernet ports` | `num_links` is larger than the physical port count between adjacent chips |

---

## Summary

| Concept | Where it lives | What it does |
|---------|---------------|-------------|
| Python CCL API | `ttnn.all_gather`, `ttnn.all_reduce`, etc. | User-facing collective functions |
| Nanobind glue | `*_nanobind.cpp` files | Connects Python calls to C++ implementations |
| C++ op entry point | `[op_name].cpp` | Validates args, constructs config, invokes program factory |
| Program factory | `[op]_program_factory.cpp` | Builds Metal program, configures EDM channels, compiles kernels |
| EDM kernel | `erisc_datamover.cpp` | Runs on ERISC; handles flow-controlled Ethernet transfer |
| Core types | `ccl_types.hpp` | `Topology`, `EriscDatamoverConfig`, `CCLOpConfig`, etc. |
| `GlobalSemaphore` | TTNN runtime | Multi-device synchronization primitive |
| `Shape4D<T>` | Kernel code | 4-D tensor shape with named dimensions |

---

*Back to [Chapter 1 Index](index.md)*

*Next chapter: [Chapter 2: Basic Operations](../ch2_basic_operations/index.md)*
