// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/tensor_serialization_support.hpp>
#include <tt-metalium/experimental/per_core_allocation/memory_config.hpp>
#include <tt-metalium/experimental/range_lockstep_allocation/memory_config.hpp>
#include <tt-metalium/experimental/tensor_layout_apis_with_custom_alignment.hpp>

namespace tt::tt_metal {

TensorLayout restore_tensor_layout_from_serialized(
    DataType dtype, const PageConfig& page_config, const MemoryConfig& memory_config, const Alignment& alignment) {
    return experimental::tensor_layout_with_custom_alignment(dtype, page_config, memory_config, alignment);
}

MemoryConfig create_memory_config_with_prepopulated_shard_specs(PrepopulatedShardSpecs specs) {
    MemoryConfig config(
        specs.memory_layout,
        specs.buffer_type,
        std::move(specs.shard_spec),
        std::move(specs.nd_shard_spec),
        specs.created_with_nd_shard_spec);
    // Through the setters, so their mode and mutual-exclusion guards still run on the rebuilt config.
    if (specs.per_core_allocation) {
        experimental::per_core_allocation::set_per_core_allocation(config, true);
    }
    if (specs.range_lockstep_allocation) {
        experimental::range_lockstep_allocation::set_range_lockstep_allocation(config, true);
    }
    return config;
}

}  // namespace tt::tt_metal
