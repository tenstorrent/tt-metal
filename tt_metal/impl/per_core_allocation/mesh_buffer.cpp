// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/per_core_allocation/mesh_buffer.hpp>
#include "impl/buffers/buffer_impl.hpp"
#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/overloaded.hpp>
#include "distributed/mesh_device_impl.hpp"
#include <algorithm>
#include <limits>
#include <map>
#include <unordered_set>

namespace tt::tt_metal::experimental::per_core_allocation {

// Validate all addresses before creating non-owning buffer wrappers.
std::shared_ptr<distributed::MeshBuffer> create_l1_buffer_view(
    distributed::MeshDevice& mesh,
    const distributed::MeshBufferConfig& mesh_config,
    const distributed::DeviceLocalBufferConfig& local_config,
    const std::vector<SRAMTensorViewAddress>& placements) {
    TT_FATAL(local_config.buffer_type == BufferType::L1, "L1 view requires L1 memory");
    TT_FATAL(is_sharded(local_config.sharding_args.buffer_layout()), "L1 view requires sharded memory");
    TT_FATAL(!placements.empty(), "L1 view requires addresses");
    std::map<distributed::MeshCoordinate, std::vector<std::pair<CoreCoord, DeviceAddr>>> by_device;
    for (const auto& [location, address] : placements) {
        TT_FATAL(mesh.impl().is_local(location.device_coord), "L1 view device must be local");
        by_device[location.device_coord].emplace_back(location.core_coord, address);
    }
    const auto* replicated = std::get_if<distributed::ReplicatedBufferConfig>(&mesh_config);
    TT_FATAL(replicated != nullptr, "L1 view requires replicated mesh buffer configuration");
    const bool per_core = is_per_core_allocation(local_config.sharding_args);
    const auto lockstep_address = placements.front().second;
    std::vector<CoreCoord> expected_cores;
    uint64_t pages_per_core;
    const auto& args = local_config.sharding_args;
    if (const auto& distribution = args.buffer_distribution_spec(); distribution.has_value()) {
        expected_cores = distribution->cores_with_data();
        pages_per_core = distribution->max_num_dev_pages_per_core();
    } else {
        TT_FATAL(args.shard_spec().has_value(), "L1 view requires a shard specification");
        const auto& spec = *args.shard_spec();
        expected_cores = corerange_to_cores(spec.grid(), std::nullopt, spec.orientation() == ShardOrientation::ROW_MAJOR);
        pages_per_core = spec.num_pages();
    }
    const std::unordered_set<CoreCoord> core_set(expected_cores.begin(), expected_cores.end());
    for (const auto& [coord, entries] : by_device) {
        auto* device = mesh.impl().get_device(coord);
        const uint64_t alignment = device->allocator()->get_alignment(BufferType::L1);
        const uint64_t aligned_page = ((uint64_t(local_config.page_size) + alignment - 1) / alignment) * alignment;
        TT_FATAL(
            pages_per_core == 0 || aligned_page <= std::numeric_limits<DeviceAddr>::max() / pages_per_core,
            "L1 view size overflows");
        const auto bytes_per_core = pages_per_core * aligned_page;
        std::unordered_set<CoreCoord> seen;
        for (const auto& [core, address] : entries) {
            TT_FATAL(seen.insert(core).second, "Duplicate L1 view placement");
            TT_FATAL(core_set.contains(core), "L1 view placement core is not in the buffer configuration");
            TT_FATAL(address % alignment == 0, "L1 view address must be aligned");
            TT_FATAL(address <= std::numeric_limits<DeviceAddr>::max() - bytes_per_core, "L1 view address overflows");
            TT_FATAL(
                address <= device->l1_size_per_core() && bytes_per_core <= device->l1_size_per_core() - address,
                "L1 view exceeds device SRAM");
            TT_FATAL(per_core || address == lockstep_address, "Lockstep tensor view requires one address on every core");
        }
        TT_FATAL(seen.size() == core_set.size(), "L1 view placements must cover every configured data core");
    }
    // The constructor is private; this friend owns the wrapper, not the addressed SRAM.
    auto result = std::shared_ptr<distributed::MeshBuffer>(
        new distributed::MeshBuffer(mesh_config, local_config, 0, replicated->size, &mesh));
    for (const auto& [coord, entries] : by_device) {
        auto local = BufferImpl::create(
            mesh.impl().get_device(coord),
            entries.front().second,
            replicated->size,
            local_config.page_size,
            BufferType::L1,
            local_config.sharding_args,
            local_config.bottom_up,
            local_config.sub_device_id);
        if (per_core) {
            std::unordered_map<CoreCoord, DeviceAddr> addresses(entries.begin(), entries.end());
            local->impl().set_per_core_addresses(std::move(addresses));
        }
        result->buffers_.at(coord) = distributed::MaybeRemote<std::shared_ptr<Buffer>>::local(std::move(local));
    }
    result->address_ = per_core ? result->get_reference_buffer()->address() : lockstep_address;
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
