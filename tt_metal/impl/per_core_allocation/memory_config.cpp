// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/tensor/spec/memory_config/memory_config.hpp>
#include <tt_stl/assert.hpp>

namespace tt::tt_metal::experimental::per_core_allocation {

bool is_per_core_allocation(const MemoryConfig& config) { return config.per_core_allocation_; }

void set_per_core_allocation(MemoryConfig& config, bool enable) {
    if (enable) {
        TT_FATAL(config.buffer_type_ == BufferType::L1, "per_core_allocation is only supported for L1 buffers");
        TT_FATAL(config.is_sharded(), "per_core_allocation requires a sharded memory layout");
        TT_FATAL(!config.created_with_nd_shard_spec_, "per_core_allocation is not supported with NdShardSpec");
    }
    config.per_core_allocation_ = enable;
}

}  // namespace tt::tt_metal::experimental::per_core_allocation
