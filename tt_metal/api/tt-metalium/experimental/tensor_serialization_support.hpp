// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/experimental/tensor/spec/memory_config/memory_config.hpp>
#include <tt-metalium/experimental/tensor/tensor_types.hpp>

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

// NOLINTNEXTLINE(readability-redundant-declaration)
MemoryConfig create_memory_config_with_prepopulated_shard_specs(
    TensorMemoryLayout memory_layout,
    BufferType buffer_type,
    std::optional<ShardSpec> shard_spec,
    std::optional<NdShardSpec> nd_shard_spec,
    bool created_with_nd_shard_spec);

}  // namespace tt::tt_metal
