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
#include "impl/allocator/allocator.hpp"
#include <tt-metalium/experimental/range_lockstep_allocation/buffer.hpp>
#include <algorithm>
#include <limits>
#include <unordered_set>

namespace tt::tt_metal::experimental::per_core_allocation {

namespace {
std::vector<CoreCoord> shard_cores(const BufferShardingArgs& args) {
    if (const auto& distribution = args.buffer_distribution_spec(); distribution.has_value()) {
        return distribution->cores_with_data();
    }
    TT_FATAL(args.shard_spec().has_value(), "L1 tensor view requires a shard specification");
    const auto& shard = args.shard_spec()->tensor_shard_spec;
    return corerange_to_cores(shard.grid, std::nullopt, shard.orientation == ShardOrientation::ROW_MAJOR);
}
}  // namespace

std::vector<L1TensorExtent> get_l1_tensor_owner_extents(
    const distributed::MeshBuffer& buffer, const std::vector<distributed::MeshCoordinate>& coords) {
    TT_FATAL(buffer.is_allocated(), "Cannot inspect deallocated L1 storage");
    auto mesh = buffer.mesh_device_.lock();
    TT_FATAL(mesh != nullptr, "L1 owner MeshDevice is closed");
    const bool owned = std::holds_alternative<distributed::MeshBuffer::OwnedBufferState>(buffer.state_);
    const bool pinned = std::holds_alternative<distributed::MeshBuffer::OwnerPinnedViewState>(buffer.state_);
    std::vector<L1TensorExtent> result;
    for (const auto& coord : coords) {
        TT_FATAL(mesh->impl().is_local(coord), "L1 owner device must be local");
        const auto* local = buffer.get_device_buffer(coord);
        TT_FATAL(local != nullptr && local->is_l1(), "L1 owner requires live L1 storage");
        const auto& allocator = local->impl().allocator_;
        const bool independent = is_per_core_allocation(*local);
        const bool range = range_lockstep_allocation::is_range_lockstep_allocation(*local);
        std::vector<CoreCoord> cores;
        if (owned && !independent && !range) {
            std::unordered_set<CoreCoord> seen;
            for (uint32_t bank = 0; bank < allocator->get_num_banks(BufferType::L1); ++bank) {
                auto core = allocator->get_logical_core_from_bank_id(bank);
                if (seen.insert(core).second) {
                    cores.push_back(core);
                }
            }
        } else {
            cores = shard_cores(buffer.device_local_config_.sharding_args);
        }
        DeviceAddr size = local->aligned_size_per_bank();
        if (!pinned && (owned || local->impl().owns_data_)) {
            size = detail::calculate_bank_size_spread(
                local->aligned_size(),
                local->aligned_page_size(),
                local->num_cores().value_or(allocator->get_num_banks(BufferType::L1)),
                allocator->get_l1_allocation_alignment());
        }
        for (const auto& core : cores) {
            result.push_back({coord, core, get_shard_base_address(*local, core), size});
        }
    }
    return result;
}

std::shared_ptr<distributed::MeshBuffer> create_l1_tensor_view(
    distributed::MeshDevice& mesh,
    const distributed::MeshBufferConfig& mesh_config,
    const distributed::DeviceLocalBufferConfig& local_config,
    const std::vector<std::vector<L1TensorExtent>>& owners,
    const std::vector<L1TensorViewPlacement>& placements,
    std::shared_ptr<void> owner_pin) {
    TT_FATAL(owner_pin != nullptr, "L1 view requires retained owners");
    TT_FATAL(local_config.buffer_type == BufferType::L1, "L1 view requires L1 memory");
    TT_FATAL(is_sharded(local_config.sharding_args.buffer_layout()), "L1 view requires sharded memory");
    const auto cores = shard_cores(local_config.sharding_args);
    TT_FATAL(!cores.empty(), "L1 view shard grid is empty");
    std::vector<distributed::MeshCoordinate> coords;
    std::unordered_set<distributed::MeshCoordinate> seen;
    for (const auto& placement : placements) {
        TT_FATAL(mesh.impl().is_local(placement.device_coord), "L1 view device must be local");
        if (seen.insert(placement.device_coord).second) {
            coords.push_back(placement.device_coord);
        }
    }
    TT_FATAL(
        !coords.empty() && placements.size() == coords.size() * cores.size(),
        "L1 view requires complete shard coverage");
    const auto* replicated = std::get_if<distributed::ReplicatedBufferConfig>(&mesh_config);
    TT_FATAL(replicated != nullptr, "L1 view requires replicated mesh buffer configuration");
    const bool per_core = is_per_core_allocation(local_config.sharding_args);
    std::optional<DeviceAddr> lockstep_address;
    auto result = std::shared_ptr<distributed::MeshBuffer>(
        new distributed::MeshBuffer(mesh_config, local_config, 0, replicated->size, &mesh));
    result->state_ = distributed::MeshBuffer::OwnerPinnedViewState{std::move(owner_pin)};
    for (const auto& coord : coords) {
        std::unordered_map<CoreCoord, DeviceAddr> addresses;
        for (const auto& placement : placements) {
            if (placement.device_coord != coord) {
                continue;
            }
            TT_FATAL(placement.owner_index < owners.size(), "L1 view owner index is out of range");
            const auto& extents = owners[placement.owner_index];
            auto owner = std::find_if(extents.begin(), extents.end(), [&](const auto& extent) {
                return extent.device_coord == coord && extent.core_coord == placement.core_coord;
            });
            TT_FATAL(owner != extents.end(), "L1 view owner has no extent on selected device/core");
            TT_FATAL(placement.offset <= owner->size, "L1 view offset exceeds owner extent");
            TT_FATAL(
                owner->address <= std::numeric_limits<DeviceAddr>::max() - placement.offset,
                "L1 view address overflows");
            TT_FATAL(
                addresses.emplace(placement.core_coord, owner->address + placement.offset).second,
                "Duplicate L1 view placement");
        }
        TT_FATAL(addresses.size() == cores.size(), "L1 view requires complete shard coverage");
        for (const auto& core : cores) {
            TT_FATAL(addresses.contains(core), "L1 view is missing shard core {}", core.str());
        }
        auto local = BufferImpl::create(
            mesh.impl().get_device(coord),
            addresses.at(cores.front()),
            replicated->size,
            local_config.page_size,
            BufferType::L1,
            local_config.sharding_args,
            local_config.bottom_up,
            local_config.sub_device_id);
        for (const auto& placement : placements) {
            if (placement.device_coord != coord) {
                continue;
            }
            const auto& extents = owners[placement.owner_index];
            auto owner = std::find_if(extents.begin(), extents.end(), [&](const auto& extent) {
                return extent.device_coord == coord && extent.core_coord == placement.core_coord;
            });
            TT_FATAL(addresses.at(placement.core_coord) % local->alignment() == 0, "L1 view address must be aligned");
            TT_FATAL(local->aligned_size_per_bank() <= owner->size - placement.offset, "L1 view exceeds owner extent");
            if (!per_core) {
                if (!lockstep_address.has_value()) {
                    lockstep_address = addresses.at(placement.core_coord);
                }
                TT_FATAL(
                    addresses.at(placement.core_coord) == *lockstep_address,
                    "Lockstep tensor view requires one address on every shard");
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
