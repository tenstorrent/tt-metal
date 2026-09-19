// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/range_lockstep_allocation/buffer.hpp>
#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include "impl/buffers/buffer_impl.hpp"
#include "impl/buffers/buffer_sharding_args_impl.hpp"
#include <algorithm>
#include <tt_stl/assert.hpp>

namespace tt::tt_metal::experimental::range_lockstep_allocation {

BufferShardingArgs& set_range_lockstep_allocation(BufferShardingArgs& args, bool enable) {
    if (enable) {
        TT_FATAL(
            !per_core_allocation::is_per_core_allocation(args),
            "range_lockstep_allocation and per_core_allocation are mutually exclusive: a buffer either takes one "
            "address across its cores or an independent address on each");
        TT_FATAL(
            args.shard_spec().has_value() || args.buffer_distribution_spec().has_value(),
            "range_lockstep_allocation requires a sharded buffer: an interleaved buffer spans every bank, so there "
            "is no narrower core set to scope the allocation to");
    }
    args.impl().range_lockstep_allocation_ = enable;
    return args;
}

bool is_range_lockstep_allocation(const BufferShardingArgs& args) { return args.impl().range_lockstep_allocation_; }

bool is_range_lockstep_allocation(const Buffer& buffer) { return buffer.impl().range_lockstep_allocation_; }

BufferShardingArgs& set_core_allocation_extents(BufferShardingArgs& args, CoreAllocationExtents extents) {
    TT_FATAL(
        is_range_lockstep_allocation(args),
        "Per-core extents require range_lockstep_allocation so every core receives the same base address");
    TT_FATAL(!extents.empty(), "Per-core extents cannot be empty");
    for (const auto& [core, extent] : extents) {
        TT_FATAL(extent > 0, "Allocation extent for core {} must be positive", core.str());
    }
    args.impl().range_lockstep_allocation_extents_ = std::move(extents);
    return args;
}

const CoreAllocationExtents& core_allocation_extents(const BufferShardingArgs& args) {
    return args.impl().range_lockstep_allocation_extents_;
}

const CoreAllocationExtents& core_allocation_extents(const Buffer& buffer) {
    return buffer.impl().range_lockstep_allocation_extents_;
}

VariableExtentAllocation::VariableExtentAllocation(
    std::shared_ptr<distributed::MeshBuffer> mesh_buffer, CoreAllocationExtents extents) :
    mesh_buffer_(std::move(mesh_buffer)), extents_(std::move(extents)) {}

std::shared_ptr<VariableExtentAllocation> VariableExtentAllocation::create(
    distributed::MeshDevice* mesh_device, CoreAllocationExtents extents, bool bottom_up) {
    TT_FATAL(mesh_device != nullptr, "Variable-extent range-lockstep allocation requires a mesh device");
    TT_FATAL(!extents.empty(), "Variable-extent range-lockstep allocation requires at least one core");

    DeviceAddr maximum_extent = 0;
    std::vector<CoreRange> singleton_ranges;
    singleton_ranges.reserve(extents.size());
    for (const auto& [core, extent] : extents) {
        maximum_extent = std::max(maximum_extent, extent);
        singleton_ranges.emplace_back(core, core);
    }

    auto sharding_args = BufferShardingArgs(
        ShardSpecBuffer(
            CoreRangeSet(std::move(singleton_ranges)),
            {1, 1},
            ShardOrientation::ROW_MAJOR,
            {1, 1},
            {1, 1}),
        TensorMemoryLayout::HEIGHT_SHARDED);
    set_range_lockstep_allocation(sharding_args, true);
    set_core_allocation_extents(sharding_args, extents);

    auto mesh_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = maximum_extent},
        distributed::DeviceLocalBufferConfig{
            .page_size = maximum_extent,
            .buffer_type = BufferType::L1,
            .sharding_args = std::move(sharding_args),
            .bottom_up = bottom_up,
        },
        mesh_device);
    return std::shared_ptr<VariableExtentAllocation>(
        new VariableExtentAllocation(std::move(mesh_buffer), std::move(extents)));
}

DeviceAddr VariableExtentAllocation::address() const {
    TT_FATAL(mesh_buffer_ != nullptr && mesh_buffer_->is_allocated(), "Variable-extent allocation is deallocated");
    return mesh_buffer_->address();
}

const CoreAllocationExtents& VariableExtentAllocation::extents() const { return extents_; }

Buffer* VariableExtentAllocation::backing_buffer() const {
    TT_FATAL(mesh_buffer_ != nullptr && mesh_buffer_->is_allocated(), "Variable-extent allocation is deallocated");
    return mesh_buffer_->get_reference_buffer();
}

void VariableExtentAllocation::deallocate() {
    if (mesh_buffer_ != nullptr) {
        mesh_buffer_->deallocate();
        mesh_buffer_.reset();
    }
}

}  // namespace tt::tt_metal::experimental::range_lockstep_allocation
