# Detailed Explanation of ReduceScatterSmall_PersistentFabric Test

## Test Overview
This test validates a reduce-scatter operation across multiple devices connected via fabric. It creates a distributed tensor replicated across 4 devices, then performs a reduce-scatter operation along dimension 3.

---

## Line-by-Line Breakdown

### Line 38: Test Declaration
```cpp
TEST(CclAsyncOp, ReduceScatterSmall_PersistentFabric) {
```
- Declares a Google Test test case named `ReduceScatterSmall_PersistentFabric` in the `CclAsyncOp` test suite
- Tests asynchronous collective communication operations (CCL) on persistent fabric

### Line 39: Dimension Setup
```cpp
const size_t dim = 3;
```
- Sets the reduction dimension to 3 (the last dimension in a 4D tensor)
- This is the dimension along which reduce-scatter will operate

### Line 40: Layout Configuration
```cpp
constexpr auto layout = Layout::TILE;
```
- Sets tensor layout to TILE (tiled memory layout optimized for hardware)
- TILE layout organizes data in 32x32 element tiles

### Lines 41-47: Architecture and Device Validation
```cpp
auto arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
constexpr size_t test_expected_num_devices = 4;
if (tt::tt_metal::GetNumAvailableDevices() < test_expected_num_devices) {
    log_info(tt::LogTest, "This test can only be run on T3000 devices");
    return;
}
```

**Function Trace:**
- `tt::get_arch_from_string()`: Converts architecture string to enum (WORMHOLE_B0, BLACKHOLE, etc.)
- `tt::test_utils::get_umd_arch_name()`: Gets the architecture name from UMD (User Mode Driver)
- `tt::tt_metal::GetNumAvailableDevices()`: Returns the number of available physical devices

**What it does:** Validates that at least 4 devices are available, otherwise skips the test

### Lines 48-51: Architecture Type Check
```cpp
if (arch == tt::ARCH::GRAYSKULL) {
    log_info(tt::LogTest, "Test must be run on WH");
    return;
}
```
- Skips test if running on Grayskull architecture (requires Wormhole or Blackhole)

### Line 52: Mesh Device Fixture Initialization
```cpp
MeshFabric1DFixture test_fixture(tt::tt_fabric::FabricConfig::FABRIC_1D);
```

**Function Trace:**

**1. `MeshFabric1DFixture` Constructor** (from `test_fabric_edm_common.hpp:214`):
```cpp
MeshFabric1DFixture(tt::tt_fabric::FabricConfig fabric_config)
    : BaseFabricFixture(fabric_config) {
    this->SetupDevices();
}
```

**2. `BaseFabricFixture` Constructor** (from `test_fabric_edm_common.hpp:117`):
```cpp
BaseFabricFixture(tt::tt_fabric::FabricConfig fabric_config, ...)
    : device_open(false) {
    tt::tt_fabric::SetFabricConfig(fabric_config, reliability_mode);
}
```
- Sets fabric configuration to FABRIC_1D (1D linear topology for device communication)

**3. `SetupDevices()`** (from `test_fabric_edm_common.hpp:199`):
```cpp
void SetupDevices() override {
    ValidateEnvironment();
    mesh_device_ = MeshDevice::create(MeshDeviceConfig(GetDeterminedMeshShape()));
    device_open = true;
}
```

**4. `ValidateEnvironment()`** (from `test_fabric_edm_common.hpp:87`):
- Checks that `TT_METAL_SLOW_DISPATCH_MODE` is not set
- Gets architecture and device count
- Validates device configuration (T3000, TG, or LLMBox)

**5. `GetDeterminedMeshShape()`** (from `test_fabric_edm_common.hpp:76`):
```cpp
MeshShape GetDeterminedMeshShape() const {
    if (num_devices_ == TG_NUM_DEVICES || num_devices_ == GALAXY_6U_NUM_DEVICES) {
        return MeshShape{8, 4};  // 8x4 mesh
    } else if (num_devices_ == 4) {
        return MeshShape{1, 4};  // 1x4 mesh (line of 4 devices)
    } else {
        return MeshShape{2, 4};  // 2x4 mesh
    }
}
```
- Returns mesh shape based on device count (for 4 devices: `{1, 4}`)

**6. `MeshDevice::create()`** (from `mesh_device.cpp:229`):
```cpp
std::shared_ptr<MeshDevice> MeshDevice::create(const MeshDeviceConfig& config, ...) {
    // Gets mapped devices from SystemMesh
    auto mapped_devices = SystemMesh::instance().get_mapped_devices(config.mesh_shape(), config.offset());

    // Creates ScopedDevices wrapper
    auto scoped_devices = std::make_shared<ScopedDevices>(...);

    // Initializes fabric and dispatch firmware
    DevicePool::instance().initialize_fabric_and_dispatch_fw();

    // Creates and returns MeshDevice
    return result;
}
```
- Creates a mesh device representing 4 devices in a 1x4 configuration
- Initializes fabric communication infrastructure
- Sets up device pools and firmware

**What it does:** Creates a fixture that manages a 1D fabric mesh of 4 devices, handling initialization and cleanup

### Lines 54-58: Test Configuration
```cpp
const size_t num_devices = test_expected_num_devices;
const ttnn::Shape input_shape({1, 1, 32, 32 * num_devices});
const MemoryConfig in_memory_config = MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM);
const auto num_elems = input_shape.volume();
```

**Breakdown:**
- `num_devices = 4`
- `input_shape = {1, 1, 32, 128}` (batch=1, channel=1, height=32, width=128)
  - Width is 32 * 4 = 128, so each device will get 32 elements after reduce-scatter
- `in_memory_config`: INTERLEAVED memory layout in DRAM
- `num_elems`: Total elements = 1 * 1 * 32 * 128 = 4096

### Lines 60-72: Input Tensor Distribution Setup
```cpp
const Tensor input_mesh_tensor = ttnn::distributed::distribute_tensor(
    ttnn::experimental::view(ttnn::arange(0, num_elems, 1, DataType::BFLOAT16), input_shape).to_layout(layout),
    *ttnn::distributed::create_mesh_mapper(
        *test_fixture.mesh_device_,
        ttnn::distributed::MeshMapperConfig{
            .placements =
                {ttnn::distributed::MeshMapperConfig::Replicate{},
                 ttnn::distributed::MeshMapperConfig::Replicate{}},
            .mesh_shape_override = MeshShape{1, num_devices}}),
    *test_fixture.mesh_device_);
```

This is a complex nested call. Let's break it down from innermost to outermost:

**1. `ttnn::arange(0, num_elems, 1, DataType::BFLOAT16)`** (from `creation.hpp:331`):

**Function Trace:**
```cpp
Tensor arange_impl(int64_t start, int64_t stop, int64_t step, ...) {
    auto size = std::max<int64_t>(0, tt::div_up(std::abs(stop - start), std::abs(step)));
    auto owned_buffer = std::vector<T>(size);

    auto index = 0;
    for (auto value = start; (step > 0) ? (value < stop) : (value > stop); value += step) {
        owned_buffer[index++] = T(static_cast<float>(value));
    }

    TensorSpec spec{...};
    return Tensor::from_vector(std::move(owned_buffer), spec, ...);
}
```
- Creates a 1D tensor with values [0, 1, 2, ..., 4095] as BFLOAT16
- Returns a host-side tensor

**2. `ttnn::experimental::view(..., input_shape)`** (from `view.cpp:116`):

**Function Trace:**
```cpp
Tensor ViewOperation::invoke(const Tensor& tensor, const Shape& shape) {
    return tensor_reshape(tensor, shape, shape);
}

Tensor tensor_reshape(const Tensor& input_tensor, const Shape& new_logical_shape, const Shape& new_padded_shape) {
    // Creates new TensorSpec with reshaped dimensions
    auto new_spec = TensorSpec(new_logical_shape, TensorLayout::fromPaddedShape(...));

    // Returns tensor with new shape but same underlying data
    return Tensor(input_tensor.storage(), new_spec, ...);
}
```
- Reshapes the 1D tensor [4096] to shape [1, 1, 32, 128]
- Creates a view (no data copy, just metadata change)

**3. `.to_layout(layout)`** (from tensor implementation):
- Converts tensor layout from ROW_MAJOR to TILE
- Reorganizes data into 32x32 tiles

**4. `ttnn::distributed::create_mesh_mapper(...)`** (from `distributed_tensor.cpp:501`):

**Function Trace:**
```cpp
std::unique_ptr<TensorToMesh> create_mesh_mapper(MeshDevice& mesh_device, const MeshMapperConfig& config) {
    return std::make_unique<TensorToMesh>(TensorToMesh::create(mesh_device, config));
}
```

**5. `TensorToMesh::create()`** (from `distributed_tensor.cpp`):
- Creates a mapper that defines how to distribute tensors across the mesh
- Config specifies:
  - `placements = {Replicate{}, Replicate{}}`: Replicate along both mesh dimensions
  - `mesh_shape_override = {1, 4}`: Override mesh shape to 1x4

**What `Replicate{}` means:** Each device gets a full copy of the entire tensor

**6. `ttnn::distributed::distribute_tensor(...)`** (from `distributed_tensor.cpp:514`):

**Function Trace:**
```cpp
Tensor distribute_tensor(
    const Tensor& tensor,
    const TensorToMesh& mapper,
    std::optional<std::reference_wrapper<MeshDevice>> mesh_device,
    std::optional<ttnn::QueueId> cq_id) {

    TT_FATAL(tensor.storage_type() == tt::tt_metal::StorageType::HOST, ...);

    // Apply mapper to create distributed tensor
    Tensor output = mapper(tensor);

    // Move tensor to device if mesh_device is provided
    if (mesh_device.has_value()) {
        return output.to_device(&(mesh_device->get()), output.memory_config(), cq_id);
    }
    return output;
}
```

**What happens:**
1. `mapper(tensor)` applies the replication strategy:
   - Takes the host tensor [1, 1, 32, 128]
   - Creates 4 copies, one for each device in the mesh
   - Each device gets the full tensor [1, 1, 32, 128]
2. `to_device()` uploads each copy to its respective device's DRAM

**Result:** `input_mesh_tensor` is a distributed tensor where:
- Each of the 4 devices has a complete copy of the tensor [1, 1, 32, 128]
- Tensor contains values [0, 1, 2, ..., 4095] arranged in TILE layout

### Line 74: Reduce-Scatter Operation
```cpp
auto output_tensor = ttnn::reduce_scatter(input_mesh_tensor, dim);
```

**Function Trace:**

**1. `ttnn::reduce_scatter()`** (from `reduce_scatter.cpp:21`):

```cpp
ttnn::Tensor ExecuteReduceScatter::invoke(
    const ttnn::Tensor& input_tensor,
    int32_t dim,
    ...) {

    // Check if need to handle 2D mesh (skip in this case)
    if (cluster_axis == std::nullopt) {
        auto mesh_shape = input_tensor.device()->get_view().shape();
        if (!mesh_shape.is_line_topology()) {
            // Would recursively call reduce_scatter for each dimension
        }
    }

    // Normalize dimension index
    uint32_t normalized_dim = input_tensor.logical_shape().get_normalized_index(dim);
    // dim=3 normalizes to 3 (already positive)

    // Determine topology (Linear or Ring)
    tt::tt_fabric::Topology topology_ = topology.value_or(
        ::ttnn::ccl::get_usable_topology(input_tensor, tt::tt_fabric::get_fabric_topology(), cluster_axis));

    // Get number of links for communication
    uint32_t num_links_ = num_links.value_or(common::get_num_links(*mesh_device, cluster_axis));

    // Check if need composite implementation (fallback)
    if (composite_common::use_composite_reduce_scatter(input_tensor, dim, cluster_axis)) {
        return composite_common::composite_reduce_scatter(...);
    }

    // Use primitive reduce-scatter implementation
    return ttnn::prim::reduce_scatter(...).at(1);
}
```

**2. `ttnn::prim::reduce_scatter()`** (registered operation, calls `ReduceScatterDeviceOperation`):

**3. `ReduceScatterDeviceOperation::invoke()`** (from `reduce_scatter_device_operation.hpp`):
- Creates output tensor specs
- Allocates output tensors
- Creates and caches mesh workload (programs)

**4. `ReduceScatterProgram::create_mesh_workload()`** (from `reduce_scatter_program_factory.cpp:21`):

```cpp
cached_mesh_workload_t create_mesh_workload(...) {
    MeshWorkload workload;

    // Create semaphores for synchronization
    std::vector<GlobalSemaphore> multidevice_semaphores = {
        create_global_semaphore(mesh_device, subdevice_core_range_set, 0),
        create_global_semaphore(mesh_device, subdevice_core_range_set, 0),
        create_global_semaphore(mesh_device, subdevice_core_range_set, 0),
    };

    // Create barrier semaphore
    auto barrier_semaphore = create_global_semaphore(...);

    // Synchronize to ensure buffers are allocated
    Synchronize(mesh_device, std::nullopt, subdevice_ids);

    // Create program for each mesh coordinate
    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(...);
        workload.add_program(MeshCoordinateRange(coord), std::move(cached_program.program));
    }

    return cached_mesh_workload_t(std::move(workload), std::move(shared_variables));
}
```

**5. `ReduceScatterProgram::create_at()`** (from `reduce_scatter_program_factory.cpp:62`):

```cpp
CachedProgram create_at(...) {
    Program program{};

    // Get mesh topology information
    uint32_t target_ring_size = get_topological_dimension(input_tensor, cluster_axis);
    // For 1x4 mesh: target_ring_size = 4

    // Find neighbors in the ring/line
    const std::optional<MeshCoordinate> forward_coordinate =
        get_physical_neighbor_from_physical_coord(input_tensor, mesh_coordinate, 1, ...);
    const std::optional<MeshCoordinate> backward_coordinate =
        get_physical_neighbor_from_physical_coord(input_tensor, mesh_coordinate, -1, ...);

    // Get device index in the ring (0, 1, 2, or 3)
    uint32_t device_index = get_linearized_index_from_physical_coord(...);

    // Choose builder based on topology
    auto builder = (topology == Topology::Ring)
        ? build_ring_reduce_scatter_minimal_async_program_artifacts
        : build_line_reduce_scatter_minimal_async_program_artifacts;

    // Build the actual program
    auto reduce_scatter_program_artifacts = builder(
        program,
        input_tensor,
        output_tensor,
        mesh_coordinate,
        forward_coordinate,
        backward_coordinate,
        intermediate_tensor,
        dim,                    // 3
        num_links,
        target_ring_size,       // 4
        device_index,          // 0-3
        topology,
        multidevice_semaphores,
        barrier_semaphore,
        ...
    );

    return {std::move(program), {...}};
}
```

**What Reduce-Scatter Does:**

For a **reduce-scatter** operation along dimension 3 across 4 devices:

1. **Reduce Phase:**
   - Each device reduces (sums) its local data with data from other devices
   - Uses ring/line topology for communication
   - Device 0 receives from device 3, sends to device 1
   - Device 1 receives from device 0, sends to device 2
   - Device 2 receives from device 1, sends to device 3
   - Device 3 receives from device 2, sends to device 0
   - After multiple communication rounds, each device has partial reductions

2. **Scatter Phase:**
   - The reduced tensor is partitioned along dimension 3
   - Each device gets 1/4 of the reduced tensor
   - Device 0 gets elements [0:32] along dim 3
   - Device 1 gets elements [32:64] along dim 3
   - Device 2 gets elements [64:96] along dim 3
   - Device 3 gets elements [96:128] along dim 3

**Result:** `output_tensor` is a distributed tensor where:
- Each device has shape [1, 1, 32, 32] (width reduced from 128 to 32)
- Each device contains the reduced sum of all devices' data for its slice
- For a tensor with values [0, 1, 2, ..., 4095], each device gets:
  - Sum of corresponding elements from all 4 devices
  - Partitioned along dimension 3

### Lines 76-78: Synchronization
```cpp
log_info(tt::LogTest, "Waiting for Op finish");
tt_metal::distributed::Finish(test_fixture.mesh_device_->mesh_command_queue(), {{SubDeviceId(0)}});
```

**Function Trace:**

**1. `mesh_device_->mesh_command_queue()`**:
- Returns reference to the mesh command queue
- Command queue manages asynchronous operations

**2. `tt_metal::distributed::Finish()`** (from `distributed.cpp:67`):

```cpp
void Finish(MeshCommandQueue& mesh_cq, tt::stl::Span<const SubDeviceId> sub_device_ids) {
    mesh_cq.finish(sub_device_ids);
}
```

**3. `FDMeshCommandQueue::finish()`** (from `fd_mesh_command_queue.cpp:513`):

```cpp
void FDMeshCommandQueue::finish(tt::stl::Span<const SubDeviceId> sub_device_ids) {
    auto lock = lock_api_function_();
    this->finish_nolock(sub_device_ids);
}

void FDMeshCommandQueue::finish_nolock(...) {
    // Enqueue a record event
    auto event = this->enqueue_record_event_to_host_nolock(sub_device_ids);

    // Wait for all outstanding reads to complete
    std::unique_lock<std::mutex> lock(reads_processed_cv_mutex_);
    reads_processed_cv_.wait(lock, [this] {
        return num_outstanding_reads_.load() == 0 || thread_exception_state_.load();
    });

    // Mark command queue as finished for each subdevice
    for (auto& sub_device_id : select_sub_device_ids(mesh_device_, sub_device_ids)) {
        sub_device_cq_owner[*sub_device_id].finished(this->id_);
    }

    // Handle exceptions if any
    if (should_handle_exception_.load()) {
        // ... exception handling ...
    }
}
```

**What it does:**
- Blocks until all operations in the command queue complete
- Ensures reduce-scatter has finished on all devices
- Synchronizes across subdevices (waits for SubDeviceId(0))

### Line 80: Test Completion
```cpp
log_info(tt::LogTest, "Finished");
```
- Logs completion message
- Test fixture destructor will clean up devices and fabric config

---

## Summary

This test:

1. **Sets up** a 1D fabric mesh of 4 devices
2. **Creates** a tensor [1, 1, 32, 128] with sequential values [0, 1, 2, ..., 4095]
3. **Replicates** this tensor across all 4 devices (each device gets full copy)
4. **Performs reduce-scatter** along dimension 3:
   - Reduces (sums) data across devices
   - Partitions result along dimension 3
   - Each device gets [1, 1, 32, 32] slice of the reduced tensor
5. **Synchronizes** to ensure operation completes
6. **Cleans up** resources via fixture destructor

The test validates that the reduce-scatter operation correctly:
- Communicates data between devices via fabric
- Performs reduction (summation) across devices
- Partitions the result correctly along the specified dimension
- Maintains data consistency across the distributed system
