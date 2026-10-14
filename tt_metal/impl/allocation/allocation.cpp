// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/allocation/allocation.hpp>

#include <cstdint>
#include <utility>

#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>
#include <tt-metalium/experimental/range_lockstep_allocation/buffer.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/overloaded.hpp>

#include "impl/allocation/allocation_impl.hpp"

namespace tt::tt_metal::experimental {

Allocation::Allocation() : pimpl_(std::make_unique<AllocationImpl>()) {}

Allocation::Allocation(AllocationImpl&& impl) : pimpl_(std::make_unique<AllocationImpl>(std::move(impl))) {}

Allocation::~Allocation() = default;

Allocation::Allocation(Allocation&&) noexcept = default;

Allocation& Allocation::operator=(Allocation&&) noexcept = default;

Allocation Allocation::adopt_from_meshtensor(MeshTensor tensor) {
    const auto& mesh_buffer = tensor.mesh_buffer();
    TT_FATAL(mesh_buffer.is_allocated(), "Cannot adopt a deallocated MeshTensor buffer");
    const auto* buffer = mesh_buffer.get_backing_buffer();
    if (per_core_allocation::is_per_core_allocation(mesh_buffer.device_local_config().sharding_args)) {
        TT_FATAL(
            mesh_buffer.device()->num_devices() == 1,
            "Per-node reservations cannot represent independent per-core allocations across multiple devices");
        // Per-core mesh buffers own their device buffers directly, without a mesh-level backing buffer.
        buffer = mesh_buffer.get_reference_buffer();
    }
    TT_FATAL(buffer != nullptr, "Cannot adopt a MeshTensor that does not own its buffer allocation");
    TT_FATAL(buffer->is_allocated(), "Cannot adopt a deallocated MeshTensor device buffer");
    TT_FATAL(buffer->buffer_type() == BufferType::L1, "Allocation only supports MeshTensors backed by L1 buffers");

    const auto size = static_cast<uint32_t>(buffer->aligned_size_per_bank());

    if (per_core_allocation::is_per_core_allocation(*buffer)) {
        return Allocation(AllocationImpl(
            PerNodeAllocation(per_core_allocation::get_per_core_addresses(*buffer), size, std::move(tensor))));
    } else if (range_lockstep_allocation::is_range_lockstep_allocation(*buffer)) {
        // Match the allocator: the distribution spec takes precedence over the shard grid,
        // which can include cores without data.
        CoreRangeSet allocated_nodes;
        if (const auto& distribution = buffer->buffer_distribution_spec(); distribution.has_value()) {
            allocated_nodes = CoreRangeSet(distribution->cores_with_data());
        } else {
            allocated_nodes = buffer->shard_spec().grid();
        }
        return Allocation(AllocationImpl(
            RangeLockstepAllocation(buffer->address(), size, std::move(allocated_nodes), std::move(tensor))));
    } else {
        return Allocation(AllocationImpl(LockstepAllocation(buffer->address(), size, std::move(tensor))));
    }
}

MeshTensor Allocation::release_as_meshtensor(Allocation allocation) {
    TT_FATAL(allocation.pimpl_ != nullptr, "Cannot release a MeshTensor from a moved-from Allocation");
    return std::visit(
        ttsl::overloaded{
            [](NoAlloc&) -> MeshTensor { TT_THROW("Cannot release a MeshTensor from an empty Allocation"); },
            [](TensorBackedResource& backing) -> MeshTensor { return std::move(backing.resource); }},
        allocation.pimpl_->_detail);
}

AllocationImpl& Allocation::impl() { return *pimpl_; }

const AllocationImpl& Allocation::impl() const { return *pimpl_; }

}  // namespace tt::tt_metal::experimental
