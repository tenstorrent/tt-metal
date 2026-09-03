// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/per_core_allocation/memory_config.hpp>
#include <tt-metalium/experimental/range_lockstep_allocation/memory_config.hpp>
#include <tt_stl/assert.hpp>

#include "impl/tensor/spec/memory_config/memory_config_impl.hpp"

namespace tt::tt_metal::experimental::per_core_allocation {

bool is_per_core_allocation(const MemoryConfig& config) { return config.impl().per_core_allocation_; }

void set_per_core_allocation(MemoryConfig& config, bool enable) {
    if (enable) {
        TT_FATAL(config.buffer_type() == BufferType::L1, "per_core_allocation is only supported for L1 buffers");
        // The reverse of the check in set_range_lockstep_allocation. Without it the two flags can
        // both end up set by ordering the calls the other way round, which serializes but aborts on
        // the way back in and throws when the tensor's sharding args are built.
        TT_FATAL(
            !range_lockstep_allocation::is_range_lockstep_allocation(config),
            "per_core_allocation and range_lockstep_allocation are mutually exclusive: a buffer either takes an "
            "independent address on each core or one address across them");
        TT_FATAL(config.is_sharded(), "per_core_allocation requires a sharded memory layout");
        TT_FATAL(!config.created_with_nd_shard_spec(), "per_core_allocation is not supported with NdShardSpec");
    }
    config.impl().per_core_allocation_ = enable;
}

}  // namespace tt::tt_metal::experimental::per_core_allocation
