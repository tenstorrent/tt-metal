// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/tensor/spec/memory_config/memory_config.hpp>
#include <tt-metalium/tensor/tensor_types.hpp>

namespace tt::tt_metal {

// ======================================================================================
//                    Tensor serialization support APIs
// ======================================================================================
//
// Outside the Runtime Tensor graduation surface.
// Explicitly provided for flatbuffer / round-trip reconstruction of TensorLayout and
// MemoryConfig (e.g. TTNN tensor_spec flatbuffer). These reconstruct objects from
// already-validated serialized fields and intentionally bypass normal construction
// paths that derive or validate shard specs.

TensorLayout restore_tensor_layout_from_serialized(
    DataType dtype, const PageConfig& page_config, const MemoryConfig& memory_config, const Alignment& alignment);

// Fields rather than parameters so callers name them: the last three are adjacent bools, and
// designated initializers make a transposition a compile error instead of a silent one. None are
// defaulted, since a caller that omits an experimental flag silently downgrades an opt-in.
struct PrepopulatedShardSpecs {
    TensorMemoryLayout memory_layout;
    BufferType buffer_type;
    std::optional<ShardSpec> shard_spec;
    std::optional<NdShardSpec> nd_shard_spec;
    bool created_with_nd_shard_spec;
    bool per_core_allocation;
    bool range_lockstep_allocation;
};

// NOLINTNEXTLINE(readability-redundant-declaration)
MemoryConfig create_memory_config_with_prepopulated_shard_specs(PrepopulatedShardSpecs specs);

}  // namespace tt::tt_metal
