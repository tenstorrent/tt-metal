// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/per_core_allocation/mesh_buffer.hpp>
#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/overloaded.hpp>
#include "distributed/mesh_device_impl.hpp"

namespace tt::tt_metal::experimental::per_core_allocation {

DeviceAddr get_per_core_address(const distributed::MeshBuffer& mesh_buffer, const CoreCoord& core) {
    auto* buffer = mesh_buffer.get_reference_buffer();
    TT_FATAL(is_per_core_allocation(*buffer), "Buffer does not use per-core allocation");
    return get_per_core_address(*buffer, core);
}

DeviceAddr get_per_core_address(
    const distributed::MeshBuffer& mesh_buffer,
    const distributed::MeshCoordinate& device_coord,
    const CoreCoord& core) {
    auto* buffer = mesh_buffer.get_device_buffer(device_coord);
    TT_FATAL(is_per_core_allocation(*buffer), "Buffer does not use per-core allocation");
    return get_per_core_address(*buffer, core);
}

bool is_per_core_allocation(const distributed::MeshBuffer& mesh_buffer) {
    // Check if the reference buffer uses per-core allocation
    auto* buffer = mesh_buffer.get_reference_buffer();
    return is_per_core_allocation(*buffer);
}

std::shared_ptr<distributed::MeshBuffer> create_on_single_device(
    const distributed::MeshBufferConfig& mesh_buffer_config,
    const distributed::DeviceLocalBufferConfig& device_local_config,
    distributed::MeshDevice* mesh_device,
    const distributed::MeshCoordinate& coord) {
    const DeviceAddr device_local_size = std::visit(
        tt::stl::overloaded{
            [](const distributed::ReplicatedBufferConfig& c) { return c.size; },
            [](const distributed::ShardedBufferConfig& config) {
                const auto [shard_height, shard_width] = config.physical_shard_shape();
                return config.compute_datum_size_bytes() * shard_height * shard_width;
            }},
        mesh_buffer_config);

    // Create a non-owning MeshBuffer — each device buffer will own its own allocation.
    auto mesh_buffer = std::shared_ptr<distributed::MeshBuffer>(new distributed::MeshBuffer(
        mesh_buffer_config, device_local_config, /*address=*/0, device_local_size, mesh_device));

    // Only allocate on the target device.
    TT_FATAL(mesh_device->impl().is_local(coord), "Target device coordinate must be local");
    auto* device = mesh_device->impl().get_device(coord);
    auto buffer = Buffer::create(
        device,
        device_local_size,
        device_local_config.page_size,
        device_local_config.buffer_type,
        device_local_config.sharding_args,
        device_local_config.bottom_up,
        device_local_config.sub_device_id);

    mesh_buffer->buffers_.at(coord) = distributed::MaybeRemote<std::shared_ptr<Buffer>>::local(std::move(buffer));
    return mesh_buffer;
}

}  // namespace tt::tt_metal::experimental::per_core_allocation
