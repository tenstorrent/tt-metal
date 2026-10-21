// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/per_core_allocation/mesh_buffer.hpp>
#include "impl/buffers/buffer_impl.hpp"
#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/overloaded.hpp>
#include "distributed/mesh_device_impl.hpp"
#include <algorithm>
#include <limits>
#include <unordered_set>

namespace tt::tt_metal::experimental::per_core_allocation {

std::shared_ptr<distributed::MeshBuffer> create_l1_tensor_view(
    distributed::MeshDevice& mesh,
    const distributed::MeshBufferConfig& mesh_config,
    const distributed::DeviceLocalBufferConfig& local_config,
    const std::vector<SRAMTensorViewAddress>& placements) {
    TT_FATAL(local_config.buffer_type == BufferType::L1, "L1 view requires L1 memory");
    TT_FATAL(is_sharded(local_config.sharding_args.buffer_layout()), "L1 view requires sharded memory");
    TT_FATAL(!placements.empty(), "L1 view requires addresses");
    std::vector<distributed::MeshCoordinate> coords;
    std::unordered_set<distributed::MeshCoordinate> seen;
    for (const auto& [location, address] : placements) {
        TT_FATAL(mesh.impl().is_local(location.device_coord), "L1 view device must be local");
        if (seen.insert(location.device_coord).second) {
            coords.push_back(location.device_coord);
        }
    }
    const auto* replicated = std::get_if<distributed::ReplicatedBufferConfig>(&mesh_config);
    TT_FATAL(replicated != nullptr, "L1 view requires replicated mesh buffer configuration");
    const bool per_core = is_per_core_allocation(local_config.sharding_args);
    std::optional<DeviceAddr> lockstep_address;
    auto result = std::shared_ptr<distributed::MeshBuffer>(
        new distributed::MeshBuffer(mesh_config, local_config, 0, replicated->size, &mesh));
    for (const auto& coord : coords) {
        std::unordered_map<CoreCoord, DeviceAddr> addresses;
        for (const auto& [location, address] : placements) {
            if (location.device_coord != coord) {
                continue;
            }
            TT_FATAL(addresses.emplace(location.core_coord, address).second, "Duplicate L1 view placement");
        }
        const auto first = std::find_if(placements.begin(), placements.end(), [&](const auto& placement) {
            return placement.first.device_coord == coord;
        });
        auto local = BufferImpl::create(
            mesh.impl().get_device(coord),
            first->second,
            replicated->size,
            local_config.page_size,
            BufferType::L1,
            local_config.sharding_args,
            local_config.bottom_up,
            local_config.sub_device_id);
        for (const auto& [location, address] : placements) {
            if (location.device_coord != coord) {
                continue;
            }
            TT_FATAL(address % local->alignment() == 0, "L1 view address must be aligned");
            TT_FATAL(
                address <= std::numeric_limits<DeviceAddr>::max() - local->aligned_size_per_bank(),
                "L1 view address overflows");
            if (!per_core) {
                if (!lockstep_address.has_value()) {
                    lockstep_address = address;
                }
                TT_FATAL(address == *lockstep_address, "Lockstep tensor view requires one address on every shard");
            }
        }
        if (per_core) {
            local->impl().set_per_core_addresses(std::move(addresses));
        }
        result->buffers_.at(coord) = distributed::MaybeRemote<std::shared_ptr<Buffer>>::local(std::move(local));
    }
    result->address_ = per_core ? result->get_reference_buffer()->address() : *lockstep_address;
    return result;
}

DeviceAddr get_per_core_address(
    const distributed::MeshBuffer& mesh_buffer,
    const distributed::MeshCoordinate& device_coord,
    const CoreCoord& core) {
    TT_FATAL(
        mesh_buffer.device()->impl().is_local(device_coord),
        "get_per_core_address: device coordinate ({}, {}) is not local or has no allocated buffer. "
        "create_on_single_device only allocates on one device within the mesh.",
        device_coord[0],
        device_coord[1]);
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
        ttsl::overloaded{
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
    auto buffer = BufferImpl::create(
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
