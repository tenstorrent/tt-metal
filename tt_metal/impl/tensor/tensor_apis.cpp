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

// ======================================================================================
//              Non-uniform enqueue_read/write_mesh_tensor
// ======================================================================================

namespace non_uniform_data_movement {

HostTensor enqueue_read_mesh_tensor(
    distributed::MeshCommandQueue& cq,
    const MeshTensor& device_tensor,
    std::span<const distributed::MeshCoordinate> coords,
    bool blocking) {
    auto distributed_host_buffer = DistributedHostBuffer::create(device_tensor.device().get_view());
    distributed_host_buffer.emplace_shards(
        {coords.begin(), coords.end()},
        [&](const distributed::MeshCoordinate&) {
            return tensor_impl::allocate_host_buffer(device_tensor.tensor_spec());
        },
        DistributedHostBuffer::ProcessShardExecutionPolicy::PARALLEL);

    HostTensor result(std::move(distributed_host_buffer), device_tensor.tensor_spec(), device_tensor.tensor_topology());
    enqueue_read_mesh_tensor(cq, device_tensor, result, coords, blocking);
    return result;
}

void enqueue_read_mesh_tensor(
    distributed::MeshCommandQueue& cq,
    const MeshTensor& device_tensor,
    HostTensor& host_tensor,
    std::span<const distributed::MeshCoordinate> coords,
    bool blocking) {
    TT_FATAL(host_tensor.logical_shape() == device_tensor.logical_shape(), "Host tensor has different shape");
    TT_FATAL(host_tensor.dtype() == device_tensor.dtype(), "Host tensor has different dtype");
    TT_FATAL(
        host_tensor.tensor_spec().page_config() == device_tensor.tensor_spec().page_config(),
        "Host tensor has different page config");

    const auto& distributed_host_buffer = host_tensor.buffer();

    std::vector<std::pair<distributed::MeshCoordinate, std::optional<HostBuffer>>> shards;
    shards.reserve(coords.size());
    for (const auto& device_coord : coords) {
        shards.push_back({device_coord, distributed_host_buffer.get_shard(device_coord)});
    }

    DistributedHostBuffer dst_distributed_host_buffer =
        DistributedHostBuffer::create(device_tensor.device().get_view());
    const size_t expected_size_bytes = device_tensor.tensor_spec().compute_packed_buffer_size_bytes();
    for (const auto& [device_coord, host_buffer] : shards) {
        dst_distributed_host_buffer.emplace_shard(device_coord, [&]() {
            TT_FATAL(host_buffer.has_value(), "Host shard for device shard {} is not populated.", device_coord);
            TT_FATAL(
                host_buffer->view_bytes().size() == expected_size_bytes,
                "Host shard for device shard {} has invalid size: {} != {}",
                device_coord,
                host_buffer->view_bytes().size(),
                expected_size_bytes);
            return *host_buffer;
        });
    }

    std::unordered_set<distributed::MeshCoordinate> shard_set(coords.begin(), coords.end());
    cq.enqueue_read(device_tensor.mesh_buffer_invariant_breaking(), dst_distributed_host_buffer, shard_set, blocking);

    host_tensor = HostTensor(
        std::move(dst_distributed_host_buffer), device_tensor.tensor_spec(), device_tensor.tensor_topology());
}

std::pair<MeshTensor, std::vector<distributed::MeshCoordinate>> enqueue_write_mesh_tensor(
    distributed::MeshCommandQueue& cq,
    const HostTensor& host_tensor,
    distributed::MeshDevice& mesh_device,
    ttsl::optional_reference<const MemoryConfig> memory_config) {
    std::optional<TensorSpec> tensor_spec_overriden_memory_config;
    if (memory_config) {
        tensor_spec_overriden_memory_config = host_tensor.tensor_spec().with_memory_config(*memory_config);
    }

    const auto* tensor_spec = tensor_spec_overriden_memory_config.has_value()
                                  ? &tensor_spec_overriden_memory_config.value()
                                  : &host_tensor.tensor_spec();

    auto result = MeshTensor::allocate_on_device(mesh_device, *tensor_spec, host_tensor.tensor_topology());
    auto coords = non_uniform_data_movement::enqueue_write_mesh_tensor(cq, host_tensor, result);
    return {std::move(result), std::move(coords)};
}

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

void h2d_as_replicate_tensor_on_1x1_mesh(
    const HostTensor& host_tensor, MeshTensor& device_tensor, distributed::MeshCommandQueue& command_queue) {
    const auto host_buffer = host_tensor.buffer().get_shard(distributed::MeshCoordinate(0, 0));
    auto data_to_write = host_buffer->view_bytes();
    const auto expected_packed_buffer_size_bytes = device_tensor.tensor_spec().compute_packed_buffer_size_bytes();
    const auto input_size_bytes = data_to_write.size();
    TT_FATAL(
        input_size_bytes == expected_packed_buffer_size_bytes,
        "Host data with total size {}B does not match expected size {}B of device buffer!",
        input_size_bytes,
        expected_packed_buffer_size_bytes);

    auto mesh_buffer = device_tensor.mesh_buffer_invariant_breaking();
    command_queue.enqueue_write_mesh_buffer(mesh_buffer, data_to_write.data(), /*blocking=*/false);

    const auto& mesh_device_shape = mesh_buffer->device()->shape();
    auto topology = TensorTopology::create_fully_replicated_tensor_topology(mesh_device_shape);
    device_tensor =
        MeshTensor(mesh_buffer, host_tensor.tensor_spec().with_memory_config(device_tensor.memory_config()), topology);
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

std::vector<distributed::MeshCoordinate> enqueue_write_mesh_tensor(
    distributed::MeshCommandQueue& cq, const HostTensor& host_tensor, MeshTensor& device_tensor) {
    TT_FATAL(host_tensor.logical_shape() == device_tensor.logical_shape(), "Host tensor has different shape");
    TT_FATAL(host_tensor.dtype() == device_tensor.dtype(), "Host tensor has different dtype");
    TT_FATAL(
        host_tensor.tensor_spec().page_config() == device_tensor.tensor_spec().page_config(),
        "Host tensor has different page config");

    const auto& host_storage_shape = host_tensor.buffer().shape();
    const auto& dst_device_shape = device_tensor.device().shape();

    // Special case of replicating tensors on 1x1 mesh across the entire mesh device.
    if (host_storage_shape.mesh_size() < dst_device_shape.mesh_size() &&
        host_storage_shape == distributed::MeshShape(1, 1)) {
        CMAKE_UNIQUE_NAMESPACE::h2d_as_replicate_tensor_on_1x1_mesh(host_tensor, device_tensor, cq);

        // All coordinates of the MeshDevice
        distributed::MeshCoordinateRange range(device_tensor.device().shape());
        return {range.begin(), range.end()};
    }

    auto mesh_buffer = device_tensor.mesh_buffer_invariant_breaking();
    cq.enqueue_write(mesh_buffer, host_tensor.buffer(), /*blocking=*/false);

    // DistributedHostBuffer may not cover the entire MeshDevice, must preserve coords here.
    // Coordinates here represents the shards that are local to this instance, there maybe other shards that are on
    // another host.
    std::vector<distributed::MeshCoordinate> coords;
    const auto& shard_coords = host_tensor.buffer().shard_coords();
    coords.reserve(shard_coords.size());
    std::copy(shard_coords.begin(), shard_coords.end(), std::back_inserter(coords));

    device_tensor = MeshTensor(
        mesh_buffer,
        host_tensor.tensor_spec().with_memory_config(device_tensor.memory_config()),
        host_tensor.tensor_topology());

    return coords;
}

}  // namespace non_uniform_data_movement

}  // namespace tt::tt_metal
