# MeshDevice API Usage in TTNN Operations

This document catalogs how `MeshDevice` and `IDevice` APIs are used within TTNN operations. The information was gathered by analyzing the codebase with compile commands exported via `./build_metal.sh -c -e --debug --build-all`.

## Overview

TTNN operations interact with devices through two primary interfaces:
1. **`MeshDevice`** - A collection of devices arranged in a mesh topology
2. **`IDevice`** - The base device interface (inherited by `MeshDevice`)

The `MeshDevice` class lives in `tt_metal/api/tt-metalium/mesh_device.hpp` and provides both the `IDevice` interface implementation and mesh-specific functionality.

---

## Table of Contents

- [IDevice Interface APIs](#idevice-interface-apis)
- [MeshDevice-Specific APIs](#meshdevice-specific-apis)
- [MeshDeviceView APIs](#meshdeviceview-apis)
- [Tensor-Device Interaction](#tensor-device-interaction)
- [Buffer and Memory APIs](#buffer-and-memory-apis)
- [Program and Kernel APIs](#program-and-kernel-apis)
- [Common Usage Patterns](#common-usage-patterns)

---

## IDevice Interface APIs

These APIs are defined in `tt_metal/api/tt-metalium/device.hpp` and implemented by `MeshDevice`.

### Architecture and Identification

| API | Description | Usage Count | Example Files |
|-----|-------------|-------------|---------------|
| `device->arch()` | Get device architecture (WORMHOLE_B0, BLACKHOLE, etc.) | ~30+ | matmul_op, groupnorm, sdpa_decode |
| `device->id()` | Get device ID | Common | Most device operations |
| `device->build_id()` | Get device build ID | Rare | Internal use |
| `device->num_hw_cqs()` | Number of hardware command queues | Rare | Queue management |
| `device->is_initialized()` | Check if device is initialized | Rare | Validation |

### Grid and Core Information

| API | Description | Usage Count | Example Files |
|-----|-------------|-------------|---------------|
| `device->compute_with_storage_grid_size()` | Get compute grid with storage | ~50+ | matmul, layernorm, groupnorm |
| `device->grid_size()` | Get device grid size | ~20+ | matmul_op_multi_core |
| `device->logical_grid_size()` | Get logical grid dimensions | Moderate | Grid calculations |
| `device->dram_grid_size()` | Get DRAM grid dimensions | ~5 | reshard_program_factory |
| `device->worker_cores()` | Get worker cores for sub-device | ~15+ | binary_ng, ternary, matmul |
| `device->num_worker_cores()` | Get number of worker cores | Moderate | Work distribution |

### Memory and Allocation

| API | Description | Usage Count | Example Files |
|-----|-------------|-------------|---------------|
| `device->allocator()` | Get device allocator | ~30+ | topk, matmul, conv2d |
| `device->allocator(sub_device_id)` | Get allocator for sub-device | Moderate | Sub-device operations |
| `device->l1_size_per_core()` | Get L1 memory size per core | ~10+ | topk, matmul, untilize |
| `device->dram_size_per_channel()` | Get DRAM size per channel | Rare | Memory validation |
| `device->num_dram_channels()` | Get number of DRAM channels | Rare | DRAM operations |
| `device->lowest_occupied_compute_l1_address()` | Get lowest occupied L1 address | ~3 | matmul_op, common |

### Core Coordinate Conversion

| API | Description | Usage Count | Example Files |
|-----|-------------|-------------|---------------|
| `device->worker_core_from_logical_core()` | Convert logical to worker core | Moderate | Core mapping |
| `device->ethernet_core_from_logical_core()` | Convert logical to ethernet core | Rare | Ethernet operations |
| `device->logical_core_from_ethernet_core()` | Convert ethernet to logical core | Rare | Ethernet operations |
| `device->virtual_core_from_logical_core()` | Convert logical to virtual core | ~3 | reshard operations |
| `device->virtual_noc0_coordinate()` | Get virtual NOC0 coordinate | Rare | NOC operations |
| `device->logical_core_from_dram_channel()` | Get logical core from DRAM channel | Rare | DRAM addressing |
| `device->dram_channel_from_logical_core()` | Get DRAM channel from logical core | Rare | DRAM addressing |

### Ethernet Operations

| API | Description | Usage Count | Example Files |
|-----|-------------|-------------|---------------|
| `device->get_active_ethernet_cores()` | Get active ethernet cores | Rare | CCL operations |
| `device->get_inactive_ethernet_cores()` | Get inactive ethernet cores | Rare | CCL operations |
| `device->is_active_ethernet_core()` | Check if ethernet core is active | Rare | CCL validation |
| `device->ethernet_cores()` | Get set of ethernet cores | Rare | Ethernet setup |
| `device->get_connected_ethernet_core()` | Get connected ethernet core | Rare | Multi-chip communication |
| `device->get_ethernet_sockets()` | Get ethernet sockets for chip | Rare | Multi-chip communication |

### Sub-Device Management

| API | Description | Usage Count | Example Files |
|-----|-------------|-------------|---------------|
| `device->get_sub_device_ids()` | Get sub-device IDs | ~15+ | binary_ng, ternary, matmul |
| `device->create_sub_device_manager()` | Create sub-device manager | Rare | Sub-device setup |
| `device->load_sub_device_manager()` | Load sub-device manager | Rare | Sub-device setup |
| `device->remove_sub_device_manager()` | Remove sub-device manager | Rare | Sub-device cleanup |
| `device->get_active_sub_device_manager_id()` | Get active sub-device manager | Rare | Sub-device queries |
| `device->num_sub_devices()` | Get number of sub-devices | Rare | Sub-device queries |
| `device->set_sub_device_stall_group()` | Set sub-device stall group | Rare | Synchronization |
| `device->reset_sub_device_stall_group()` | Reset sub-device stall group | Rare | Synchronization |

### NOC and Encoding

| API | Description | Usage Count | Example Files |
|-----|-------------|-------------|---------------|
| `device->get_noc_unicast_encoding()` | Get NOC unicast encoding | Rare | NOC operations |
| `device->get_noc_multicast_encoding()` | Get NOC multicast encoding | Rare | NOC operations |
| `device->has_noc_mcast_txns()` | Check for NOC multicast transactions | Rare | NOC configuration |
| `device->num_noc_unicast_txns()` | Get number of NOC unicast transactions | Rare | NOC configuration |
| `device->noc_data_start_index()` | Get NOC data start index | Rare | NOC configuration |

### Program Cache

| API | Description | Usage Count | Example Files |
|-----|-------------|-------------|---------------|
| `device->get_program_cache()` | Get program cache | Moderate | Operation dispatch |
| `device->enable_program_cache()` | Enable program cache | Rare | Initialization |
| `device->disable_and_clear_program_cache()` | Disable and clear cache | Rare | Cleanup |
| `device->num_program_cache_entries()` | Get cache entry count | Rare | Debugging |

### Other IDevice APIs

| API | Description | Usage Count | Example Files |
|-----|-------------|-------------|---------------|
| `device->command_queue()` | Get command queue | Rare | Queue operations |
| `device->sysmem_manager()` | Get system memory manager | Rare | Memory management |
| `device->is_mmio_capable()` | Check MMIO capability | ~1 | moe_utils |
| `device->get_mesh_device()` | Get parent mesh device | Rare | Profiler use |
| `device->get_programmable_core_type()` | Get core type for coord | Rare | Core type queries |
| `device->get_dev_addr()` | Get device address for core | Rare | Address queries |
| `device->get_dev_size()` | Get device size for core | Rare | Size queries |

---

## MeshDevice-Specific APIs

These APIs are unique to `MeshDevice` and not part of the base `IDevice` interface.

### Device Access

| API | Description | Usage Count | Example Files |
|-----|-------------|-------------|---------------|
| `mesh_device->get_devices()` | Get all devices in mesh | Moderate | CCL operations |
| `mesh_device->get_device(coord)` | Get device at mesh coordinate | ~15+ | all_gather, reduce_scatter |
| `mesh_device->get_device(physical_id)` | Get device by physical ID | Rare | Device lookup |
| `mesh_device->get_device(row, col)` | Get device at 2D position | Moderate | 2D mesh operations |
| `mesh_device->num_devices()` | Get total device count | Moderate | Mesh size queries |

### Mesh Shape and Topology

| API | Description | Usage Count | Example Files |
|-----|-------------|-------------|---------------|
| `mesh_device->shape()` | Get mesh shape | ~10+ | CCL operations, reduce_to_root |
| `mesh_device->num_rows()` | Get number of rows (2D) | Moderate | 2D mesh operations |
| `mesh_device->num_cols()` | Get number of columns (2D) | Moderate | 2D mesh operations |
| `mesh_device->reshape()` | Reshape logical mesh | Rare | Mesh reconfiguration |
| `mesh_device->get_view()` | Get MeshDeviceView | ~15+ | CCL operations |
| `mesh_device->is_local()` | Check if coordinate is local | Rare | Multi-host |

### Submesh Operations

| API | Description | Usage Count | Example Files |
|-----|-------------|-------------|---------------|
| `mesh_device->create_submesh()` | Create a submesh | Rare | Mesh partitioning |
| `mesh_device->create_submeshes()` | Create multiple submeshes | Rare | Mesh partitioning |
| `mesh_device->get_parent_mesh()` | Get parent mesh device | Rare | Hierarchy navigation |
| `mesh_device->get_submeshes()` | Get all submeshes | Rare | Submesh queries |
| `mesh_device->is_parent_mesh()` | Check if parent mesh | Rare | Hierarchy queries |
| `mesh_device->quiesce_devices()` | Wait for all submesh work | Rare | Synchronization |

### Mesh Command Queue

| API | Description | Usage Count | Example Files |
|-----|-------------|-------------|---------------|
| `mesh_device->mesh_command_queue()` | Get mesh command queue | Moderate | Workload dispatch |
| `mesh_device->mesh_command_queue(cq_id)` | Get specific command queue | Rare | Multi-queue |

### Mesh Tracing

| API | Description | Usage Count | Example Files |
|-----|-------------|-------------|---------------|
| `mesh_device->begin_mesh_trace()` | Begin mesh trace | Rare | Tracing |
| `mesh_device->end_mesh_trace()` | End mesh trace | Rare | Tracing |
| `mesh_device->replay_mesh_trace()` | Replay mesh trace | Rare | Tracing |
| `mesh_device->release_mesh_trace()` | Release mesh trace | Rare | Tracing |
| `mesh_device->get_mesh_trace()` | Get mesh trace buffer | Rare | Tracing |

### Fabric and Multi-Host

| API | Description | Usage Count | Example Files |
|-----|-------------|-------------|---------------|
| `mesh_device->get_fabric_node_id()` | Get fabric node ID for coord | ~10+ | CCL async operations |
| `mesh_device->compile_fabric()` | Compile fabric | Rare | Fabric setup |
| `mesh_device->configure_fabric()` | Configure fabric | Rare | Fabric setup |
| `mesh_device->init_fabric()` | Initialize fabric | Rare | Fabric setup |

### Thread Pool

| API | Description | Usage Count | Example Files |
|-----|-------------|-------------|---------------|
| `mesh_device->enqueue_to_thread_pool()` | Enqueue work to thread pool | ~1 | mesh_workload |
| `mesh_device->wait_for_thread_pool()` | Wait for thread pool completion | Rare | Synchronization |

### Static Factory Methods

| API | Description | Usage Count | Example Files |
|-----|-------------|-------------|---------------|
| `MeshDevice::create()` | Create mesh device | Common | Device initialization |
| `MeshDevice::create_unit_mesh()` | Create single-device mesh | Common | Single device setup |
| `MeshDevice::create_unit_meshes()` | Create multiple unit meshes | Rare | Multi-device setup |

---

## MeshDeviceView APIs

`MeshDeviceView` provides query interfaces for mesh sub-regions. Located in `tt_metal/api/tt-metalium/mesh_device_view.hpp`.

### Device Queries

| API | Description | Usage Count | Example Files |
|-----|-------------|-------------|---------------|
| `mesh_view.get_devices()` | Get all devices in view | Moderate | CCL operations |
| `mesh_view.get_devices(range)` | Get devices in range | Rare | Range queries |
| `mesh_view.get_device(coord)` | Get device at coordinate | Rare | Point queries |
| `mesh_view.num_devices()` | Get device count | Moderate | Size queries |
| `mesh_view.size()` | Get view size | Moderate | Size queries |
| `mesh_view.empty()` | Check if view is empty | Rare | Validation |

### 2D Mesh Operations

| API | Description | Usage Count | Example Files |
|-----|-------------|-------------|---------------|
| `mesh_view.is_mesh_2d()` | Check if 2D mesh | ~10+ | CCL operations |
| `mesh_view.num_rows()` | Get row count | ~15+ | all_gather, reduce_scatter |
| `mesh_view.num_cols()` | Get column count | ~15+ | all_gather, reduce_scatter |
| `mesh_view.get_devices_on_row()` | Get devices on row | ~5 | ring_attention |
| `mesh_view.get_devices_on_column()` | Get devices on column | ~5 | llama_reduce_scatter |

### Fabric Node IDs

| API | Description | Usage Count | Example Files |
|-----|-------------|-------------|---------------|
| `mesh_view.get_fabric_node_ids()` | Get all fabric node IDs | Rare | Fabric operations |
| `mesh_view.get_fabric_node_id(coord)` | Get fabric node ID at coord | Moderate | CCL async |
| `mesh_view.get_fabric_node_ids_on_row()` | Get node IDs on row | Rare | Row operations |
| `mesh_view.get_fabric_node_ids_on_column()` | Get node IDs on column | Rare | Column operations |

### Shape and Coordinate Queries

| API | Description | Usage Count | Example Files |
|-----|-------------|-------------|---------------|
| `mesh_view.shape()` | Get view shape | Moderate | Shape queries |
| `mesh_view.mesh_id()` | Get mesh ID | Rare | ID queries |
| `mesh_view.contains(coord)` | Check if contains coordinate | ~3 | point_to_point |
| `mesh_view.find_device(device_id)` | Find device coordinate | Rare | Reverse lookup |
| `mesh_view.is_local(coord)` | Check if coordinate is local | Rare | Multi-host |

### Ring and Line Utilities

| API | Description | Usage Count | Example Files |
|-----|-------------|-------------|---------------|
| `mesh_view.get_line_coordinates()` | Get line coordinates | Rare | Line operations |
| `mesh_view.get_ring_coordinates()` | Get ring coordinates | Rare | Ring operations |
| `mesh_view.get_line_devices()` | Get devices in line order | Rare | Line operations |
| `mesh_view.get_ring_devices()` | Get devices in ring order | Rare | Ring operations |
| `mesh_view.get_line_fabric_node_ids()` | Get line fabric node IDs | Rare | Line fabric |
| `mesh_view.get_ring_fabric_node_ids()` | Get ring fabric node IDs | Rare | Ring fabric |

---

## Tensor-Device Interaction

Tensors interact with devices through these primary patterns:

### Device Access from Tensor

```cpp
// Get mesh device from tensor
MeshDevice* mesh_device = tensor.device();

// Get architecture from tensor's device
tt::ARCH arch = tensor.device()->arch();

// Check storage type before device access
if (tensor.storage_type() == StorageType::DEVICE) {
    auto* device = tensor.device();
}
```

### Data Movement

| API | Description | Usage Count |
|-----|-------------|-------------|
| `tensor.to_device(mesh_device, ...)` | Move tensor to device | ~70+ |
| `tensor.cpu(blocking, cq_id)` | Move tensor to host | ~50+ |
| `tensor.buffer()` | Get device buffer | ~250+ |
| `tensor.mesh_buffer()` | Get mesh buffer | Moderate |
| `tensor.device_storage()` | Get device storage | Moderate |

### Buffer Information

```cpp
// Common buffer queries in operations
tensor.buffer()->address()     // Buffer address
tensor.buffer()->page_size()   // Page size
tensor.buffer()->num_pages()   // Number of pages
```

---

## Buffer and Memory APIs

### Allocator APIs

| API | Description | Usage Pattern |
|-----|-------------|---------------|
| `device->allocator()->get_statistics()` | Get allocation statistics | Memory checks |
| `device->allocator()->get_base_allocator_addr()` | Get base allocator address | L1 calculations |
| `device->allocator()->get_alignment()` | Get allocation alignment | Buffer creation |
| `device->allocator()->get_num_banks()` | Get number of memory banks | Bank calculations |
| `device->allocator()->get_bank_ids_from_logical_core()` | Get bank IDs from core | Sharding |

### Common Memory Calculations

```cpp
// Calculate available L1 space
auto lowest_address = device->lowest_occupied_compute_l1_address();
uint32_t max_l1_space = lowest_address.has_value()
    ? lowest_address.value()
    : device->l1_size_per_core();
max_l1_space -= device->allocator()->get_base_allocator_addr(HalMemType::L1);

// Get L1 statistics
auto L1_stats = device->allocator()->get_statistics(BufferType::L1);
auto largest_free = L1_stats.largest_free_block_bytes;
auto total_allocated = L1_stats.total_allocated_bytes;
```

---

## Program and Kernel APIs

Operations create programs using these tt-metalium APIs:

### Program Creation

| API | Description | Usage Count |
|-----|-------------|-------------|
| `Program()` | Create empty program | ~100+ |
| `CreateKernel()` | Create a kernel | ~450+ |
| `CreateCircularBuffer()` | Create circular buffer | ~450+ |
| `tt::tt_metal::create_cb()` | Helper to create CB | ~100+ |

### Runtime Arguments

| API | Description | Usage Count |
|-----|-------------|-------------|
| `SetRuntimeArgs()` | Set kernel runtime args | ~300+ |
| `GetRuntimeArgs()` | Get kernel runtime args | ~300+ |

### NOC Preference Helpers

```cpp
// Get preferred NOC for DRAM operations
tt::tt_metal::NOC writer_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());
tt::tt_metal::NOC reader_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());
```

---

## Common Usage Patterns

### Pattern 1: Getting Compute Grid for Work Distribution

```cpp
auto grid = device->compute_with_storage_grid_size();
auto num_pages = output.buffer()->num_pages();
auto [num_cores, all_cores, core_group_1, core_group_2, ...] =
    tt::tt_metal::split_work_to_cores(grid, num_pages);
```

### Pattern 2: Architecture-Specific Configuration

```cpp
if (device->arch() == tt::ARCH::WORMHOLE_B0) {
    // Wormhole-specific configuration
} else if (device->arch() == tt::ARCH::BLACKHOLE) {
    // Blackhole-specific configuration
}

auto compute_config = get_compute_kernel_config_args(
    device->arch(),
    operation_attributes.compute_kernel_config
);
```

### Pattern 3: Getting Device from Tensor in Operations

```cpp
// In device operation implementations
auto* device = input_tensor.device();
auto arch = device->arch();
auto grid = device->compute_with_storage_grid_size();
```

### Pattern 4: CCL Operations with Mesh Views

```cpp
auto* mesh_device = tensor.device();
const auto& mesh_view = mesh_device->get_view();

TT_FATAL(mesh_view.is_mesh_2d(), "Operation requires 2D mesh");

std::size_t num_devices = (cluster_axis == 0)
    ? mesh_view.num_rows()
    : mesh_view.num_cols();

auto devices = (cluster_axis == 0)
    ? mesh_view.get_devices_on_column(coord[1])
    : mesh_view.get_devices_on_row(coord[0]);
```

### Pattern 5: Sub-Device Worker Core Access

```cpp
for (const auto& sub_device_id : device->get_sub_device_ids()) {
    const auto& sub_device_workers = device->worker_cores(
        HalProgrammableCoreType::TENSIX,
        sub_device_id
    );
    // Use workers...
}
```

### Pattern 6: Mesh Workload Factory Pattern

```cpp
// Creating program at specific mesh coordinate
auto cached_program = WorkloadFactory::create_at(
    operation_attributes,
    mesh_coordinate,  // MeshCoordinate
    tensor_args,
    tensor_return_value
);
```

### Pattern 7: Fabric Node ID for Multi-Chip Communication

```cpp
const auto src_node_id = mesh_device->get_fabric_node_id(sender_coord);
const auto dst_node_id = mesh_device->get_fabric_node_id(receiver_coord);
auto forwarding_links = tt::tt_fabric::get_forwarding_link_indices(
    dst_node_id,
    src_node_id
);
```

---

## File References

Key files where these APIs are heavily used:

### CCL Operations
- `ttnn/cpp/ttnn/operations/ccl/all_gather/device/`
- `ttnn/cpp/ttnn/operations/ccl/reduce_scatter/device/`
- `ttnn/cpp/ttnn/operations/ccl/all_to_all_*/device/`
- `ttnn/cpp/ttnn/operations/experimental/ccl/*/device/`

### Matmul Operations
- `ttnn/cpp/ttnn/operations/matmul/device/matmul_op_multi_core_*.cpp`

### Normalization Operations
- `ttnn/cpp/ttnn/operations/normalization/*/device/`

### Data Movement
- `ttnn/cpp/ttnn/operations/data_movement/*/device/`

### Core Device Operation Framework
- `ttnn/api/ttnn/device_operation.hpp`
- `ttnn/core/tensor/tensor.cpp`
- `ttnn/core/tensor/tensor_ops.cpp`

---

## Header File Locations

| Header | Path |
|--------|------|
| MeshDevice | `tt_metal/api/tt-metalium/mesh_device.hpp` |
| MeshDeviceView | `tt_metal/api/tt-metalium/mesh_device_view.hpp` |
| IDevice | `tt_metal/api/tt-metalium/device.hpp` |
| MeshConfig | `tt_metal/api/tt-metalium/mesh_config.hpp` |
| Device Operation | `ttnn/api/ttnn/device_operation.hpp` |
| TTNN Device | `ttnn/api/ttnn/device.hpp` |

---

*Generated by analyzing tt-metal codebase with compile commands exported via `./build_metal.sh -c -e --debug --build-all`*
