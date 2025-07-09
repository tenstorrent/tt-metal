# TTNN Device to MeshDevice Migration Guide

April 23, 2025

# Introduction

There is a major change being merged to TT-Metal & TTNN in relation to working with multiple devices. Currently, TTNN manages multi-device tensors and operations by creating a tensor on each of user exposed device and deploying the same OP on each device with a lot of non-trivial multi-threading and synchronization involved.
This management of multiple devices is being lowered to Metal layer. More specifically:
1.  A TTNN user will always see a MeshDevice.
    - This object has the same APIs as the Device object, currently exposed to users.
	- When interfacing with mutiple devices, the MeshDevice will span the extent of the physical cluster.
	- For cases where users want to control a single physical device a Unit-Mesh (1x1 MeshDevice) Handle will be provided.
	- The user experience should remain unchanged in either case, since the MeshDevice APIs are backwards compatible with the Device APIs.
2.  Tensors will be allocated in lock-step on multiple physical devices while being backed by the newly introduced MeshBuffer. The cluster of physical devices is virtualized by a single MeshDevice and the MeshBuffer resides in the distributed memory space exposed by the MeshDevice. This implies that:
	- MultiDeviceStorage is being removed in favour of always using DeviceStorage
	- Tensor allocations go through the MeshDevice allocator (instead of individual device allocators), so the buffer on each of the devices gets the same address.
	- The behavior of `get_device_tensors` changes to return a single-device view into the multi-device Tensor (backed by a MeshBuffer).
	- To call `aggregate_as_tensor` for Tensors on device, all of them must be backed by the same MeshBuffer.
	- If a user wants to aggregate a collection of individual Tensor shards before sending them to device.`aggregate_as_tensor` can be called on the host tensors: `aggregate_as_tensor(host_tensors).to(mesh_device)`.
3. Each operation is dispatched to multiple devices by Metal, instead of by TTNN.
	- TTNN lowers each user defined operation to a MeshWorkload instead of a Program.
	- TT-Metal handles multi-threaded dispatch of the MeshWorkload to the MeshDevice, making TTNN runtime much simpler.
	- The MeshWorkload structure allows users to natively express Multi-Device operations while providing a clean separation of the tasks between TTNN (OP Lowering) and TT-Metal (Workload Dispatch)

This change results in multiple benefits for both TTNN developers and TTNN users:
1. TTNN host dispatch implementation is drastically simplified 🎉
2.  There are no more waits to acquire Tensor metadata, it is returned immediately, since TTNN is synchronous 🎉
3. Significant performance improvements 🎉
4. There’s no need to create multiple threads in neither TTNN or user code, all of the multi-threading is handled internally by TT-Metal 🎉

For more information, please see the [TT-Distributed Tech Report](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/TT-Distributed/TT-Distributed-Architecture-1219.md).

## Limitations

This major change comes with a few limitations. Specifically:

1.  It is impossible to interleave calls to MeshDevice and a single Device managed by it. The user has to make a choice which one to use and stick to it.
2.  All Tensor buffers are allocated in lock-step on all devices within the Mesh. This means it is impossible to have a multi-device Tensor which has different addresses on different devices, unless one is willing to spawn a collection of MeshDevices.
3.  You can not aggregate arbitrary Tensors from different mesh devices into a multi-device Tensor due to (1) and (2).
4.  All workloads are executed in lock-step fashion across all devices within MeshDevice, one MeshWorkload at a time. The MeshWorkload itself can be homogenous (one program targeting the entire MeshDevice) or heterogenous (different programs running on different physical devices).


## Migration Steps (Applicable to C++ Users Only)

This section provides a set of steps for C++ users to seamlessly port their codebase to TT-Mesh. For Python users, changes introduced by TT-Mesh should be effectively abstracted away, since the TTNN Python APIs have already been ported to support the new backend.

### 1. Update Device Management
For cases where you want to explicitly control each physical device, instead of seeing a Virtualized MeshDevice, please follow the steps below.

Replace all usages of `CreateDevice` as follows:
```cpp
// Old
tt::tt_metal::IDevice* device = tt::tt_metal::CreateDevice(device_id);
// New
#include <tt-metallium/distributed.hpp>

std::shared_ptr<tt::tt_metal::distributed::MeshDevice> device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);
```
Analogously you should replace calls to `ttnn::open_device` with `ttnn::open_mesh_device` and calls to `CreateDevices` with `create_unit_meshes`.

Please note the type of the returned device has changed to a shared_ptr, so there’s no need to explicitly call `CloseDevice`/`CloseDevices`/`ttnn::close_device` anymore if you’re using RAII.
```cpp
// Remove this line
tt::tt_metal::CloseDevice(device);
```

If you need to close the device before destroying the object, you should simply call `device->close();`.

### 2. Remove Device/IDevice
You should replace all mentions of `tt::tt_metal::IDevice` or `tt::tt_metal::Device` in your codebase with `tt::tt_metal::distributed::MeshDevice

### 3. Remove command_queue() calls
All calls to `device->command_queue()` are fundamentally incompatible with MeshDevice, and an exception will be thrown on any such call. You should replace all usages of `command_queue()` with `mesh_command_queue()` in your codebase. Some specific use-cases may require special attention as listed below.

### 4. Event synchronization

With migration to MeshDevice, event synchronization calls should be updated as well:
```cpp
// Old CQ -> CQ and CQ -> Host  (Events are recorded on the device and propagated to host)
auto  event = std::make_shared<Event>();
ttnn::record_event(device->command_queue(*io_cq), event);
ttnn::wait_for_event(device->command_queue(*op_cq), event); // op_cq waits for an event from io_cq
ttnn::event_synchronize(event); // host waits for an event from io_cq

// New MeshCQ -> Host (User Controls whether an Event should be propagated to host or not)
auto event = ttnn::record_event_to_host(device->mesh_command_queue(*io_cq));
ttnn::event_synchronize(event);

// New MeshCQ -> MeshCQ (Event is not propagated to host -> faster)
auto event = ttnn::record_event(device->mesh_command_queue(*io_cq));
ttnn::wait_for_event(device->mesh_command_queue(*op_cq), event);
```
### 5. Manual calls to Metal APIs
Ideally, TTNN users should use Tensors and TT-NN OPs instead of direct access to Metal buffers/programs. However, if you have direct calls to those APIs in your codebase, they would need to be updated as follows:

```cpp
// Old tensor_impl::allocate_buffer_on_device
auto input_buffer = tt::tt_metal::tensor_impl::allocate_buffer_on_device(device,  tensor_spec);
auto  storage = tt::tt_metal::DeviceStorage{input_buffer};

// New allocate_tensor_on_device (preferred replacement)
auto  input_tensor = allocate_tensor_on_device(tensor_spec, device.get());

// New tensor_impl::allocate_mesh_buffer_on_device
auto input_buffer = tt::tt_metal::tensor_impl::allocate_mesh_buffer_on_device(device, tensor_spec);
auto storage = tt::tt_metal::DeviceStorage{input_buffer,  DistributedTensorConfig{},  {{tt::tt_metal::distributed::MeshCoordinate{0,  0},  tensor_spec}}};

// Old EnqueueProgram
tt::tt_metal::EnqueueProgram(device->command_queue(), program, true);

// New EnqueueMeshWorkload
tt::tt_metal::distributed::MeshWorkload workload;
workload.add_program(device->get_view().coord_range(), std::move(program));
tt_metal::distributed::EnqueueMeshWorkload(device->mesh_command_queue(0),  workload,  true);

// Old CreateBuffer
tt_metal::InterleavedBufferConfig  buff_config{.device = device, .size = dram_buffer_size,  .page_size = page_size, .buffer_type = buffType};
auto buffer = CreateBuffer(buff_config);

// New MeshBuffer::create
tt_metal::distributed::DeviceLocalBufferConfig  buff_config{.page_size = page_size,  .buffer_type = buffType};
tt_metal::distributed::MeshBufferConfig  mesh_config = tt_metal::distributed::ReplicatedBufferConfig{.size = dram_buffer_size};
auto buffer = tt_metal::distributed::MeshBuffer::create(mesh_config, buff_config, device.get());

// Old WriteToBuffer
tt_metal::detail::WriteToBuffer(src_buffer, src_vec);

// New WriteShard
tt_metal::distributed::WriteShard(device->mesh_command_queue(0), src_mesh_buffer, src_vec, *device->get_view().coord_range().begin());

// Old ReadFromBuffer
tt_metal::detail::ReadFromBuffer(dst_mesh_buffer, result_vec);

// New ReadShard
tt_metal::distributed::ReadShard(device->mesh_command_queue(0), result_vec, dst_mesh_buffer,
*device->get_view().coord_range().begin());

```
### Possible issues

#### OwnedStorage vs MultiDeviceHostStorage

In some cases, Tensors which used to have OwnedStorage may have MultiDeviceHostStorage with a single owned buffer after the migration. It may break some usages and the user code needs to be updated to handle another storage type.
