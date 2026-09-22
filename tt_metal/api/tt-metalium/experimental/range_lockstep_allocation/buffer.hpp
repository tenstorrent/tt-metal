// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/buffer.hpp>

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

}  // namespace tt::tt_metal::experimental::range_lockstep_allocation
