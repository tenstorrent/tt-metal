// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/per_core_allocation/mesh_buffer.hpp>
#include "impl/buffers/buffer_impl.hpp"
#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>
#include <tt-metalium/experimental/per_core_allocation/allocator_state.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/overloaded.hpp>
#include "distributed/mesh_device_impl.hpp"
#include "impl/allocator/allocator.hpp"
#include <algorithm>
#include <limits>
#include <unordered_set>

namespace tt::tt_metal::experimental::per_core_allocation {

namespace {

class HybridReservationScope {
public:
    HybridReservationScope(AllocatorImpl* mesh_allocator, const std::vector<AllocatorImpl*>& device_allocators) :
        allocator_(mesh_allocator) {
        TT_FATAL(
            allocator_->try_begin_hybrid_allocation(device_allocators),
            "Another HYBRID mesh allocation or pool reservation is in progress");
    }
    ~HybridReservationScope() { allocator_->end_hybrid_allocation(); }

private:
    AllocatorImpl* allocator_;
};

struct RetainedMeshBufferOwners {
    struct MirrorLease {
        IDevice* device = nullptr;
        DeviceAddr address = 0;
    };

    std::vector<std::shared_ptr<Buffer>> buffers;
    std::vector<std::shared_ptr<void>> owner_pins;
    std::vector<MirrorLease> mirror_leases;

    ~RetainedMeshBufferOwners() {
        for (const auto& lease : mirror_leases) {
            if (lease.device->is_initialized()) {
                lease.device->allocator_impl()->unmirror_lockstep_allocation(lease.address);
            }
        }
    }
};

std::vector<CoreCoord> shard_cores(const BufferShardingArgs& args) {
    if (const auto& distribution = args.buffer_distribution_spec(); distribution.has_value()) {
        return distribution->cores_with_data();
    }
    TT_FATAL(args.shard_spec().has_value(), "L1 pool views require a sharded TensorSpec");
    const auto& spec = args.shard_spec()->tensor_shard_spec;
    return corerange_to_cores(spec.grid, std::nullopt, spec.orientation == ShardOrientation::ROW_MAJOR);
}

}  // namespace

std::shared_ptr<L1Pool> reserve_l1_pool(
    distributed::MeshDevice& mesh_device, const std::vector<L1PoolExtent>& extents) {
    TT_FATAL(
        mesh_device.impl().coowner_ranks().empty(),
        "reserve_l1_pool does not support co-owned/multi-host meshes: exact reservation is not cross-rank atomic");
    auto pool = std::make_shared<L1Pool>();
    pool->mesh_device_ = mesh_device.shared_from_this();
    pool->extents_ = extents;
    pool->reserved_owners_.reserve(extents.size());

    std::vector<AllocatorImpl*> device_allocators;
    device_allocators.reserve(mesh_device.get_view().num_devices());
    for (auto* device : mesh_device.get_view().get_devices()) {
        device_allocators.push_back(device->allocator_impl().get());
    }
    HybridReservationScope guard(mesh_device.allocator_impl().get(), device_allocators);

    for (const auto& extent : extents) {
        TT_FATAL(mesh_device.impl().is_local(extent.device_coord), "L1 pool extent device must be local");
        auto* device = mesh_device.impl().get_device(extent.device_coord);
        const DeviceAddr alignment = device->allocator_impl()->get_l1_allocation_alignment();
        TT_FATAL(extent.size != 0 && extent.size % alignment == 0, "L1 pool extent size must be aligned");
        TT_FATAL(extent.address % alignment == 0, "L1 pool extent address must be aligned");

        ShardSpecBuffer shard_spec(
            CoreRangeSet(extent.core_coord),
            {1, static_cast<uint32_t>(extent.size / alignment)},
            ShardOrientation::ROW_MAJOR,
            {1, 1},
            {1, static_cast<uint32_t>(extent.size / alignment)});
        BufferShardingArgs args(shard_spec, TensorMemoryLayout::HEIGHT_SHARDED);
        set_per_core_allocation(args, true);
        pool->reserved_owners_.push_back(BufferImpl::create_reserved(
            device,
            extent.size,
            alignment,
            BufferType::L1,
            args,
            {{extent.core_coord, extent.address}},
            {{extent.core_coord, get_l1_occupied_ranges(mesh_device, extent.device_coord, extent.core_coord)}}));
    }
    return pool;
}

size_t adopt_l1_pool_extent(
    const std::shared_ptr<L1Pool>& pool, const L1PoolExtent& extent, std::shared_ptr<void> owner_pin) {
    TT_FATAL(pool != nullptr && owner_pin != nullptr, "External L1 pool extent requires pool and owner pin");
    auto mesh = pool->mesh_device_;
    TT_FATAL(mesh != nullptr && mesh->impl().is_local(extent.device_coord), "External extent device must be local");
    const DeviceAddr alignment =
        mesh->impl().get_device(extent.device_coord)->allocator_impl()->get_l1_allocation_alignment();
    TT_FATAL(extent.size != 0, "External extent cannot be empty");
    TT_FATAL(extent.address % alignment == 0, "External extent address must be aligned");
    TT_FATAL(extent.size % alignment == 0, "External extent size must be aligned");
    TT_FATAL(extent.address <= std::numeric_limits<DeviceAddr>::max() - extent.size, "External extent overflows");
    const DeviceAddr end = extent.address + extent.size;
    const auto occupied = get_l1_occupied_ranges(*mesh, extent.device_coord, extent.core_coord);
    TT_FATAL(
        std::any_of(occupied.begin(), occupied.end(), [&](const auto& range) {
            return range.first <= extent.address && end <= range.second;
        }),
        "External extent is not covered by a live L1 allocation");
    auto adopted = extent;
    adopted.externally_owned = true;
    pool->extents_.push_back(std::move(adopted));
    pool->external_owner_pins_.push_back(std::move(owner_pin));
    return pool->extents_.size() - 1;
}

std::shared_ptr<void> retain_l1_pool_owner(
    const distributed::MeshBuffer& mesh_buffer,
    const std::vector<distributed::MeshCoordinate>& device_coords) {
    TT_FATAL(mesh_buffer.is_allocated(), "Cannot adopt deallocated MeshBuffer storage");
    auto retained = std::make_shared<RetainedMeshBufferOwners>();
    auto mesh = mesh_buffer.mesh_device_.lock();
    TT_FATAL(mesh != nullptr, "Cannot adopt storage from a closed MeshDevice");

    if (const auto* owned = std::get_if<distributed::MeshBuffer::OwnedBufferState>(&mesh_buffer.state_)) {
        TT_FATAL(mesh_buffer.device_local_config_.buffer_type == BufferType::L1, "External pool owner must be in L1");
        retained->buffers.push_back(owned->backing_buffer);
        const DeviceAddr address = owned->backing_buffer->address();
        const DeviceAddr size = get_shard_allocation_size(*owned->backing_buffer);
        retained->mirror_leases.reserve(device_coords.size());
        for (const auto& coord : device_coords) {
            TT_FATAL(mesh->impl().is_local(coord), "External pool owner device must be local");
            auto* device = mesh->impl().get_device(coord);
            device->allocator_impl()->mirror_lockstep_allocation(address, size);
            retained->mirror_leases.push_back({device, address});
        }
    } else if (const auto* pinned =
                   std::get_if<distributed::MeshBuffer::OwnerPinnedViewState>(&mesh_buffer.state_)) {
        retained->owner_pins.push_back(pinned->owner_pin);
    } else {
        TT_FATAL(
            std::holds_alternative<distributed::MeshBuffer::ExternallyOwnedState>(mesh_buffer.state_) &&
                is_per_core_allocation(mesh_buffer.device_local_config_.sharding_args),
            "External-address MeshBuffer has no retainable allocation owner");
        retained->buffers.reserve(device_coords.size());
        for (const auto& coord : device_coords) {
            TT_FATAL(mesh->impl().is_local(coord), "External pool owner device must be local");
            const auto& owner = mesh_buffer.buffers_.at(coord).value();
            TT_FATAL(owner->impl().owns_data_, "Per-core MeshBuffer device buffer does not own its allocation");
            retained->buffers.push_back(owner);
        }
    }
    return std::static_pointer_cast<void>(retained);
}

std::shared_ptr<distributed::MeshBuffer> create_l1_pool_view(
    const std::shared_ptr<L1Pool>& pool,
    const distributed::MeshBufferConfig& mesh_buffer_config,
    const distributed::DeviceLocalBufferConfig& device_local_config,
    const std::vector<L1PoolPlacement>& placements) {
    TT_FATAL(pool != nullptr, "L1 pool is null");
    auto mesh = pool->mesh_device_;
    TT_FATAL(mesh != nullptr, "L1 pool MeshDevice is closed");
    TT_FATAL(device_local_config.buffer_type == BufferType::L1, "L1 pool views require L1 TensorSpec");
    TT_FATAL(is_sharded(device_local_config.sharding_args.buffer_layout()), "L1 pool views require sharded TensorSpec");

    const DeviceAddr device_local_size = std::visit(
        ttsl::overloaded{
            [](const distributed::ReplicatedBufferConfig& c) { return c.size; },
            [](const distributed::ShardedBufferConfig& c) {
                const auto [h, w] = c.physical_shard_shape();
                return c.compute_datum_size_bytes() * h * w;
            }},
        mesh_buffer_config);
    const auto cores = shard_cores(device_local_config.sharding_args);
    TT_FATAL(!cores.empty(), "L1 pool view shard grid is empty");

    std::vector<distributed::MeshCoordinate> selected_coords;
    std::unordered_set<distributed::MeshCoordinate> selected_set;
    for (const auto& placement : placements) {
        TT_FATAL(mesh->impl().is_local(placement.device_coord), "L1 pool placement device must be local");
        if (selected_set.insert(placement.device_coord).second) {
            selected_coords.push_back(placement.device_coord);
        }
    }
    TT_FATAL(!selected_coords.empty(), "L1 pool view needs at least one selected device");
    TT_FATAL(
        placements.size() == selected_coords.size() * cores.size(),
        "L1 pool placements must cover every shard on each selected device");

    std::optional<DeviceAddr> lockstep_address;
    const bool per_core = is_per_core_allocation(device_local_config.sharding_args);
    auto result = std::shared_ptr<distributed::MeshBuffer>(new distributed::MeshBuffer(
        mesh_buffer_config, device_local_config, 0, device_local_size, mesh.get()));
    result->state_ = distributed::MeshBuffer::OwnerPinnedViewState{std::static_pointer_cast<void>(pool)};

    for (const auto& coord : selected_coords) {
        std::unordered_map<CoreCoord, DeviceAddr> addresses;
        for (const auto& placement : placements) {
            if (placement.device_coord != coord) {
                continue;
            }
            TT_FATAL(placement.owner_index < pool->extents_.size(), "L1 pool owner index is out of range");
            const auto& owner = pool->extents_[placement.owner_index];
            TT_FATAL(owner.device_coord == coord && owner.core_coord == placement.core_coord, "Placement owner device/core mismatch");
            TT_FATAL(placement.offset <= owner.size, "Placement offset is outside owner extent");
            TT_FATAL(owner.address <= std::numeric_limits<DeviceAddr>::max() - placement.offset, "Placement address overflows");
            TT_FATAL(addresses.emplace(placement.core_coord, owner.address + placement.offset).second, "Duplicate L1 pool placement");
        }
        TT_FATAL(addresses.size() == cores.size(), "L1 pool placements do not cover this device's shard cores");
        for (const auto& core : cores) {
            TT_FATAL(addresses.contains(core), "L1 pool placement is missing shard core {}", core.str());
        }

        const DeviceAddr first_address = addresses.at(cores.front());
        auto buffer = BufferImpl::create(
            mesh->impl().get_device(coord),
            first_address,
            device_local_size,
            device_local_config.page_size,
            BufferType::L1,
            device_local_config.sharding_args,
            device_local_config.bottom_up,
            device_local_config.sub_device_id);
        const DeviceAddr required = get_shard_allocation_size(*buffer);
        const DeviceAddr alignment = buffer->alignment();
        for (const auto& placement : placements) {
            if (placement.device_coord != coord) {
                continue;
            }
            const auto& owner = pool->extents_[placement.owner_index];
            TT_FATAL(placement.offset % alignment == 0, "L1 pool placement offset must be aligned");
            TT_FATAL(placement.offset <= owner.size && required <= owner.size - placement.offset, "L1 pool placement exceeds owner extent");
            const DeviceAddr address = owner.address + placement.offset;
            if (!per_core) {
                if (!lockstep_address.has_value()) {
                    lockstep_address = address;
                }
                TT_FATAL(address == lockstep_address.value(), "Lockstep TensorSpec requires one address on every shard");
            }
        }
        if (per_core) {
            buffer->impl().set_per_core_addresses(std::move(addresses));
        }
        result->buffers_.at(coord) = distributed::MaybeRemote<std::shared_ptr<Buffer>>::local(std::move(buffer));
    }
    TT_FATAL(lockstep_address.has_value() || per_core, "L1 pool view has no address");
    result->address_ = per_core ? result->get_reference_buffer()->address() : lockstep_address.value();
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
