// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <unordered_map>
#include <tt-metalium/buffer.hpp>

namespace tt::tt_metal::distributed {
class MeshBuffer;
class MeshDevice;
}  // namespace tt::tt_metal::distributed

namespace tt::tt_metal::experimental::range_lockstep_allocation {

// Experimental and subject to change: this header carries no API-stability guarantee.
//
// A lockstep buffer takes one address across the cores it occupies. By default the allocator
// keeps that address clear of per-core allocations on EVERY core, which is what an op needs when
// it addresses the buffer on cores outside its own shard grid — a multicast, for instance, writes
// to every core in its rectangle whether or not that core is a destination.
//
// Range lockstep narrows that guarantee to the cores the buffer actually occupies. Use it only
// when nothing reaches the buffer on a core it was not allocated on; otherwise the placement may
// land on top of a per-core allocation elsewhere on the grid.
//
// Expected to become the default once ops that address buffers across cores declare the region
// they reach, at which point these functions go away.

BufferShardingArgs& set_range_lockstep_allocation(BufferShardingArgs& args, bool enable);
bool is_range_lockstep_allocation(const BufferShardingArgs& args);
bool is_range_lockstep_allocation(const Buffer& buffer);

using CoreAllocationExtents = std::unordered_map<CoreCoord, DeviceAddr>;

BufferShardingArgs& set_core_allocation_extents(BufferShardingArgs& args, CoreAllocationExtents extents);
const CoreAllocationExtents& core_allocation_extents(const BufferShardingArgs& args);
const CoreAllocationExtents& core_allocation_extents(const Buffer& buffer);

class VariableExtentAllocation {
public:
    static std::shared_ptr<VariableExtentAllocation> create(
        distributed::MeshDevice* mesh_device, CoreAllocationExtents extents, bool bottom_up = false);

    DeviceAddr address() const;
    const CoreAllocationExtents& extents() const;
    Buffer* backing_buffer() const;
    void deallocate();

private:
    VariableExtentAllocation(
        std::shared_ptr<distributed::MeshBuffer> mesh_buffer, CoreAllocationExtents extents);

    std::shared_ptr<distributed::MeshBuffer> mesh_buffer_;
    CoreAllocationExtents extents_;
};

}  // namespace tt::tt_metal::experimental::range_lockstep_allocation
