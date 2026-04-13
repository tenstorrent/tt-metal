// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/tensor/tensor_apis.hpp>
#include <tt-metalium/experimental/tensor/impl/tensor_impl.hpp>

namespace tt::tt_metal {

// ======================================================================================
//                            Transfer classification
// ======================================================================================

bool is_uniform_write(const HostTensor& host_tensor, const distributed::MeshDevice& device) {
    const auto& device_mesh_shape = device.shape();
    const auto& host_buffer = host_tensor.buffer();

    if (host_buffer.shape() != device_mesh_shape) {
        return false;
    }

    auto all_coords = distributed::MeshCoordinateRange(device_mesh_shape);
    return std::ranges::all_of(
        all_coords, [&](const auto& coord) { return host_buffer.shard_coords().contains(coord); });
}

// ======================================================================================
//                                Uniform Data movement APIs
// ======================================================================================

HostTensor enqueue_read_mesh_tensor(distributed::MeshCommandQueue& cq, const MeshTensor& device_tensor, bool blocking) {
    auto mesh_buffer = device_tensor.mesh_buffer_invariant_breaking();
    auto& device = device_tensor.device();

    auto distributed_host_buffer = DistributedHostBuffer::create(device.get_view());

    distributed::MeshCoordinateRange all_coords(device.shape());
    std::vector<distributed::MeshCoordinate> coords(all_coords.begin(), all_coords.end());
    distributed_host_buffer.emplace_shards(
        coords,
        [&](const distributed::MeshCoordinate&) {
            return tensor_impl::allocate_host_buffer(device_tensor.tensor_spec());
        },
        DistributedHostBuffer::ProcessShardExecutionPolicy::PARALLEL);

    cq.enqueue_read(mesh_buffer, distributed_host_buffer, /*shards=*/std::nullopt, blocking);

    return HostTensor(std::move(distributed_host_buffer), device_tensor.tensor_spec(), device_tensor.tensor_topology());
}

MeshTensor enqueue_write_mesh_tensor(
    distributed::MeshCommandQueue& cq,
    const HostTensor& host_tensor,
    distributed::MeshDevice& mesh_device,
    ttsl::optional_reference<const MemoryConfig> memory_config) {
    TT_FATAL(
        is_uniform_write(host_tensor, mesh_device),
        "Incompatible shape between source host tensor and target MeshDevice. For non-uniform transfers, use the "
        "non-uniform data movement APIs.");
    std::optional<TensorSpec> tensor_spec_overriden_memory_config;
    if (memory_config) {
        tensor_spec_overriden_memory_config = host_tensor.tensor_spec().with_memory_config(*memory_config);
    }

    const auto* tensor_spec = tensor_spec_overriden_memory_config.has_value()
                                  ? &tensor_spec_overriden_memory_config.value()
                                  : &host_tensor.tensor_spec();

    auto result = MeshTensor::allocate_on_device(mesh_device, *tensor_spec, host_tensor.tensor_topology());
    enqueue_write_mesh_tensor(cq, host_tensor, result);
    return result;
}

void enqueue_read_mesh_tensor(
    distributed::MeshCommandQueue& cq, const MeshTensor& device_tensor, HostTensor& host_tensor, bool blocking) {
    TT_FATAL(host_tensor.logical_shape() == device_tensor.logical_shape(), "Host tensor has different shape");
    TT_FATAL(host_tensor.dtype() == device_tensor.dtype(), "Host tensor has different dtype");
    TT_FATAL(
        host_tensor.tensor_spec().page_config() == device_tensor.tensor_spec().page_config(),
        "Host tensor has different page config");

    auto mesh_buffer = device_tensor.mesh_buffer_invariant_breaking();

    cq.enqueue_read(mesh_buffer, host_tensor.buffer(), /*shards=*/std::nullopt, blocking);
    host_tensor.update_tensor_topology(device_tensor.tensor_topology());
}

void enqueue_write_mesh_tensor(
    distributed::MeshCommandQueue& cq, const HostTensor& host_tensor, MeshTensor& device_tensor) {
    TT_FATAL(
        is_uniform_write(host_tensor, device_tensor.device()),
        "Incompatible shape between source host tensor and target MeshDevice. For non-uniform transfers, use the "
        "non-uniform data movement APIs.");
    TT_FATAL(host_tensor.logical_shape() == device_tensor.logical_shape(), "Host tensor has different shape");
    TT_FATAL(host_tensor.dtype() == device_tensor.dtype(), "Host tensor has different dtype");
    TT_FATAL(
        host_tensor.tensor_spec().page_config() == device_tensor.tensor_spec().page_config(),
        "Host tensor has different page config");

    const auto& mesh_buffer = device_tensor.mesh_buffer_invariant_breaking();

    // Uniform H2D copy.
    cq.enqueue_write(mesh_buffer, host_tensor.buffer(), /*blocking=*/false);
    device_tensor = MeshTensor(
        mesh_buffer,
        host_tensor.tensor_spec().with_memory_config(device_tensor.memory_config()),
        host_tensor.tensor_topology());
}

// ======================================================================================
//                    Unit Tensor enqueue_read/write_mesh_tensor
// ======================================================================================

void enqueue_read_mesh_tensor(
    distributed::MeshCommandQueue& queue,
    const MeshTensor& device_tensor,
    std::byte* dst,
    const std::optional<BufferRegion>& region,
    bool blocking) {
    TT_FATAL(queue.device()->num_devices() == 1, "enqueue_read_mesh_tensor only supports single device mesh");
    std::vector<distributed::ShardDataTransfer> shard_data_transfers = {
        distributed::ShardDataTransfer{*distributed::MeshCoordinateRange(queue.device()->shape()).begin()}
            .host_data(dst)
            .region(region)};
    queue.enqueue_read_shards(shard_data_transfers, device_tensor.mesh_buffer_invariant_breaking(), blocking);
}

void enqueue_write_mesh_tensor(
    distributed::MeshCommandQueue& queue,
    const std::byte* src,
    MeshTensor& device_tensor,
    const std::optional<BufferRegion>& region) {
    TT_FATAL(queue.device()->num_devices() == 1, "enqueue_write_mesh_tensor only supports single device mesh");
    std::vector<distributed::ShardDataTransfer> shard_data_transfers = {
        distributed::ShardDataTransfer{*distributed::MeshCoordinateRange(queue.device()->shape()).begin()}
            .host_data(const_cast<std::byte*>(src))
            .region(region)};
    queue.enqueue_write_shards(device_tensor.mesh_buffer_invariant_breaking(), shard_data_transfers, false);
}

}  // namespace tt::tt_metal
