// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/per_core_allocation/mesh_buffer.hpp>
#include "impl/buffers/buffer_impl.hpp"
#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>
#include "impl/allocator/allocator.hpp"
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/tensor/spec/tensor_spec.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/overloaded.hpp>
#include "distributed/mesh_device_impl.hpp"
#include "impl/debug/inspector/inspector.hpp"
#include <map>
#include <unordered_set>

namespace tt::tt_metal::experimental::per_core_allocation {

// Validate all addresses before creating non-owning buffer wrappers.
distributed::MeshBuffer create_sram_buffer_view(
    distributed::MeshDevice& mesh, const TensorSpec& spec, const PerNodeMeshDeviceMemorySpan& memory) {
    TT_FATAL(spec.memory_config().buffer_type() == BufferType::L1, "SRAM view requires L1 memory");
    const auto packed_size = spec.compute_packed_buffer_size_bytes();
    const auto page_size = spec.compute_page_size_bytes();
    const distributed::ReplicatedBufferConfig mesh_config{.size = packed_size};
    const distributed::DeviceLocalBufferConfig local_config{
        .page_size = page_size, .buffer_type = BufferType::L1, .sharding_args = spec.compute_buffer_sharding_args()};
    TT_FATAL(is_sharded(local_config.sharding_args.buffer_layout()), "SRAM view requires sharded memory");
    TT_FATAL(!memory.per_device_spans.empty(), "SRAM view requires addresses");
    // Ordered maps keep reference addresses stable regardless of the caller's hash-map iteration order.
    std::map<distributed::MeshCoordinate, std::map<CoreCoord, NodeLocalSpan>> by_device;
    for (const auto& [coord, local] : memory.per_device_spans) {
        TT_FATAL(
            distributed::MeshCoordinateRange(mesh.shape()).contains(coord),
            "SRAM view device coordinate is outside the mesh");
        TT_FATAL(mesh.impl().is_local(coord), "SRAM view device must be local");
        TT_FATAL(!local.per_node_spans.empty(), "SRAM view requires addresses on each selected device");
        by_device.emplace(
            coord, std::map<CoreCoord, NodeLocalSpan>(local.per_node_spans.begin(), local.per_node_spans.end()));
    }
    const bool per_core = is_per_core_allocation(local_config.sharding_args);
    const auto lockstep_address = by_device.begin()->second.begin()->second.address;
    // TensorSpec resolves a distribution for every sharded layout, including partially used grids.
    const auto& expected_cores = local_config.sharding_args.buffer_distribution_spec().value().cores_with_data();
    const std::unordered_set<CoreCoord> core_set(expected_cores.begin(), expected_cores.end());
    for (const auto& [coord, entries] : by_device) {
        auto* device = mesh.impl().get_device(coord);
        const uint64_t alignment = device->allocator_impl()->get_alignment(BufferType::L1);
        const auto bytes_per_core = spec.compute_consumed_memory_bytes_per_bank(alignment, expected_cores.size());
        for (const auto& [core, span] : entries) {
            TT_FATAL(core_set.contains(core), "SRAM view placement core is not in the buffer configuration");
            TT_FATAL(bytes_per_core <= span.size, "SRAM view padded tensor footprint exceeds supplied span");
            device->allocator_impl()->validate_l1_range(span.address, span.size);
            TT_FATAL(
                per_core || span.address == lockstep_address,
                "Lockstep tensor view requires one address on every core");
        }
        TT_FATAL(entries.size() == core_set.size(), "SRAM view placements must cover every configured data core");
    }
    // Construct metadata directly; this view never owns the addressed SRAM.
    distributed::MeshBuffer result(mesh_config, local_config, 0, packed_size, &mesh);
    for (const auto& [coord, entries] : by_device) {
        auto local = BufferImpl::create(
            mesh.impl().get_device(coord),
            entries.begin()->second.address,
            packed_size,
            local_config.page_size,
            BufferType::L1,
            local_config.sharding_args,
            local_config.bottom_up,
            local_config.sub_device_id);
        if (per_core) {
            std::unordered_map<CoreCoord, DeviceAddr> addresses;
            for (const auto& [core, span] : entries) {
                addresses.emplace(core, span.address);
            }
            local->impl().set_per_core_addresses(std::move(addresses));
        }
        result.buffers_.at(coord) = distributed::MaybeRemote<std::shared_ptr<Buffer>>::local(std::move(local));
    }
    result.address_ = per_core ? result.get_reference_buffer()->address() : lockstep_address;
    Inspector::mesh_buffer_allocated(&result);
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

    mesh_buffer->address_ = buffer->address();
    mesh_buffer->buffers_.at(coord) = distributed::MaybeRemote<std::shared_ptr<Buffer>>::local(std::move(buffer));
    return mesh_buffer;
}

}  // namespace tt::tt_metal::experimental::per_core_allocation
