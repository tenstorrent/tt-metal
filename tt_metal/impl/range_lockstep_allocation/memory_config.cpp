// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/range_lockstep_allocation/memory_config.hpp>
#include <tt-metalium/experimental/per_core_allocation/memory_config.hpp>
#include <tt_stl/assert.hpp>

#include "impl/tensor/spec/memory_config/memory_config_impl.hpp"

namespace tt::tt_metal::experimental::range_lockstep_allocation {

bool is_range_lockstep_allocation(const MemoryConfig& config) { return config.impl().range_lockstep_allocation_; }

void set_range_lockstep_allocation(MemoryConfig& config, bool enable) {
    if (enable) {
        TT_FATAL(
            !per_core_allocation::is_per_core_allocation(config),
            "range_lockstep_allocation and per_core_allocation are mutually exclusive: a buffer either takes one "
            "address across its cores or an independent address on each");
        TT_FATAL(
            config.is_sharded(),
            "range_lockstep_allocation requires a sharded memory layout: an interleaved buffer spans every bank, so "
            "there is no narrower core set to scope the allocation to");
    }
    config.impl().range_lockstep_allocation_ = enable;
}

}  // namespace tt::tt_metal::experimental::range_lockstep_allocation
