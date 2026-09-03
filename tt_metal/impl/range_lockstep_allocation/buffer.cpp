// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/range_lockstep_allocation/buffer.hpp>
#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>
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
    args.range_lockstep_allocation_ = enable;
    return args;
}

bool is_range_lockstep_allocation(const BufferShardingArgs& args) { return args.range_lockstep_allocation_; }

bool is_range_lockstep_allocation(const Buffer& buffer) { return buffer.range_lockstep_allocation_; }

}  // namespace tt::tt_metal::experimental::range_lockstep_allocation
