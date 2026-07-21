// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstring>
#include <functional>
#include <unordered_set>

#include "host_tensor_impl.hpp"
#include "mesh_tensor_impl.hpp"

#include <tt-metalium/experimental/distributed_tensor/distributed_tensor_apis.hpp>
#include <tt-metalium/experimental/tensor/impl/tensor_impl.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>
#include "tt_metal/distributed/pinned_memory_cache.hpp"
#include "tt_metal/distributed/mesh_device_view_impl.hpp"

namespace tt::tt_metal {

// ======================================================================================
//                        Topology accessors
// ======================================================================================

const TensorTopology& tensor_topology(const MeshTensor& tensor) { return tensor.impl().topology(); }

const TensorTopology& tensor_topology(const HostTensor& tensor) { return tensor.impl().topology(); }

void update_tensor_topology(MeshTensor& tensor, TensorTopology topology) {
    tensor.impl().update_topology(std::move(topology));
}

void update_tensor_topology(HostTensor& tensor, TensorTopology topology) {
    tensor.impl().update_topology(std::move(topology));
}

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

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

constexpr size_t k_pin_write_threshold_bytes = 32 * 1024 * 1024;

bool should_use_pinned_write_path(distributed::MeshDevice& mesh_device, size_t size_bytes) {
    if (size_bytes <= k_pin_write_threshold_bytes) {
        return false;
    }
    const auto params = experimental::GetMemoryPinningParameters(mesh_device);
    return params.max_pins > 0 && params.can_map_to_noc;
}

}  // namespace CMAKE_UNIQUE_NAMESPACE

}  // namespace

// ======================================================================================
//              Non-uniform enqueue_read/write_tensor
// ======================================================================================

namespace non_uniform_data_movement {

HostTensor enqueue_read_tensor(
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

    auto result = HostTensor::from_buffer(
        std::move(distributed_host_buffer), device_tensor.tensor_spec(), tensor_topology(device_tensor));
    enqueue_read_tensor(cq, device_tensor, result, coords, blocking);
    return result;
}

void enqueue_read_tensor(
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
    cq.enqueue_read(device_tensor.impl().raw_mesh_buffer(), dst_distributed_host_buffer, shard_set, blocking);

    host_tensor = HostTensor::from_buffer(
        std::move(dst_distributed_host_buffer), device_tensor.tensor_spec(), tensor_topology(device_tensor));
}

std::pair<MeshTensor, std::vector<distributed::MeshCoordinate>> enqueue_write_tensor(
    distributed::MeshCommandQueue& cq,
    const HostTensor& host_tensor,
    distributed::MeshDevice& mesh_device,
    ttsl::optional_reference<const MemoryConfig> memory_config) {
    std::optional<TensorSpec> tensor_spec_overriden_memory_config;
    if (memory_config) {
        const auto& old_spec = host_tensor.tensor_spec();
        tensor_spec_overriden_memory_config = TensorSpec(
            old_spec.logical_shape(),
            TensorLayout(
                old_spec.tensor_layout().get_data_type(),
                old_spec.tensor_layout().get_page_config(),
                *memory_config,
                old_spec.tensor_layout().get_alignment()));
    }

    const auto* tensor_spec = tensor_spec_overriden_memory_config.has_value()
                                  ? &tensor_spec_overriden_memory_config.value()
                                  : &host_tensor.tensor_spec();

    auto result = MeshTensor::allocate_on_device(mesh_device, *tensor_spec, tensor_topology(host_tensor));
    auto coords = non_uniform_data_movement::enqueue_write_tensor(cq, host_tensor, result);
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

    auto mesh_buffer = device_tensor.impl().raw_mesh_buffer();
    auto* mesh_device = mesh_buffer->device();

    const bool use_pinned =
        ::tt::tt_metal::CMAKE_UNIQUE_NAMESPACE::should_use_pinned_write_path(*mesh_device, data_to_write.size());

    if (use_pinned) {
        // Replication fans a single 1x1 host shard out to the whole mesh, but only the chips owned
        // by this host can be pinned/written; restrict the pin and the transfers to those so the
        // remote coordinates are a complete no-op here.
        const auto& view = mesh_device->get_view();
        std::vector<distributed::MeshCoordinate> local_coords;
        distributed::MeshCoordinateRangeSet local_range;
        for (const auto& coord : distributed::MeshCoordinateRange(mesh_device->shape())) {
            if (view.impl().is_local(coord)) {
                local_coords.push_back(coord);
                local_range.merge(distributed::MeshCoordinateRange(coord, coord));
            }
        }

        HostBuffer pinned_buffer(*host_buffer);
        auto pinned_memory = local_coords.empty() ? nullptr
                                                  : experimental::PinnedMemoryCache::instance().try_pin(
                                                        *mesh_device, local_range, pinned_buffer, /*map_to_noc=*/true);

        if (pinned_memory) {
            std::vector<distributed::ShardDataTransfer> transfers;
            transfers.reserve(local_coords.size());
            for (const auto& coord : local_coords) {
                auto xfer = distributed::ShardDataTransfer{coord}
                                .host_data(const_cast<void*>(static_cast<const void*>(data_to_write.data())))
                                .region(BufferRegion(0, data_to_write.size()));
                experimental::ShardDataTransferSetPinnedMemory(xfer, pinned_memory);
                transfers.push_back(std::move(xfer));
            }
            command_queue.enqueue_write_shards(mesh_buffer, transfers, /*blocking=*/true);
        } else {
            command_queue.enqueue_write_mesh_buffer(mesh_buffer, data_to_write.data(), /*blocking=*/false);
        }
    } else {
        command_queue.enqueue_write_mesh_buffer(mesh_buffer, data_to_write.data(), /*blocking=*/false);
    }

    const auto& mesh_device_shape = mesh_buffer->device()->shape();
    auto topology = TensorTopology::create_fully_replicated_tensor_topology(mesh_device_shape);
    const auto& old_spec = host_tensor.tensor_spec();
    device_tensor = MeshTensor::from_buffer(
        std::move(*mesh_buffer),
        TensorSpec(
            old_spec.logical_shape(),
            TensorLayout(
                old_spec.tensor_layout().get_data_type(),
                old_spec.tensor_layout().get_page_config(),
                device_tensor.memory_config(),
                old_spec.tensor_layout().get_alignment())),
        topology);
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

std::vector<distributed::MeshCoordinate> enqueue_write_tensor(
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

    auto mesh_buffer = device_tensor.impl().raw_mesh_buffer();
    const auto& distributed_host_buffer = host_tensor.buffer();

    size_t total_size = 0;
    for (const auto& coord : distributed_host_buffer.shard_coords()) {
        auto buf = distributed_host_buffer.get_shard(coord);
        if (buf) {
            total_size += buf->view_bytes().size();
        }
    }

    const bool use_pinned =
        ::tt::tt_metal::CMAKE_UNIQUE_NAMESPACE::should_use_pinned_write_path(*cq.device(), total_size);

    if (use_pinned) {
        auto* mesh_device = mesh_buffer->device();
        const auto& view = mesh_device->get_view();
        std::vector<distributed::ShardDataTransfer> transfers;
        transfers.reserve(distributed_host_buffer.shard_coords().size());
        bool any_pinned = false;

        for (const auto& coord : distributed_host_buffer.shard_coords()) {
            // get_shard yields a buffer only for shards owned by this host, so remote chips are
            // never pinned or added to the transfer list -- the transfer is a no-op for them here.
            auto buf = distributed_host_buffer.get_shard(coord);
            if (buf) {
                // The host buffer's distribution must agree with the device's: host memory can only
                // be pinned to MMIO devices local to this process, so a populated shard for a coord
                // the device owns on another host must never reach try_pin (which would fault while
                // resolving the remote device).
                TT_FATAL(
                    view.impl().is_local(coord),
                    "Host buffer holds a shard for device coordinate {}, but that device is not local "
                    "to this host; host memory can only be pinned to MMIO devices owned by this process.",
                    coord);
                auto coord_range = distributed::MeshCoordinateRangeSet(distributed::MeshCoordinateRange(coord, coord));
                HostBuffer pinned_buf(*buf);
                auto pinned_memory = experimental::PinnedMemoryCache::instance().try_pin(
                    *mesh_device, coord_range, pinned_buf, /*map_to_noc=*/true);

                auto xfer = distributed::ShardDataTransfer{distributed::MeshCoordinate(coord)}
                                .host_data(buf->view_bytes().data())
                                .region(BufferRegion(0, buf->view_bytes().size()));
                if (pinned_memory) {
                    experimental::ShardDataTransferSetPinnedMemory(xfer, std::move(pinned_memory));
                    any_pinned = true;
                }
                transfers.push_back(std::move(xfer));
            }
        }
        if (any_pinned) {
            cq.enqueue_write_shards(mesh_buffer, transfers, /*blocking=*/true);
        } else {
            cq.enqueue_write(mesh_buffer, distributed_host_buffer, /*blocking=*/false);
        }
    } else {
        cq.enqueue_write(mesh_buffer, distributed_host_buffer, /*blocking=*/false);
    }

    // DistributedHostBuffer may not cover the entire MeshDevice, must preserve coords here.
    // Coordinates here represents the shards that are local to this instance, there maybe other shards that are on
    // another host.
    std::vector<distributed::MeshCoordinate> coords;
    const auto& shard_coords = distributed_host_buffer.shard_coords();
    coords.reserve(shard_coords.size());
    std::copy(shard_coords.begin(), shard_coords.end(), std::back_inserter(coords));

    const auto& old_spec = host_tensor.tensor_spec();
    device_tensor = MeshTensor::from_buffer(
        std::move(*mesh_buffer),
        TensorSpec(
            old_spec.logical_shape(),
            TensorLayout(
                old_spec.tensor_layout().get_data_type(),
                old_spec.tensor_layout().get_page_config(),
                device_tensor.memory_config(),
                old_spec.tensor_layout().get_alignment())),
        tensor_topology(host_tensor));

    return coords;
}

}  // namespace non_uniform_data_movement

}  // namespace tt::tt_metal
