// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * Internal page-config layout helpers for tt_metal.
 *
 * Not part of the installed public API. Callers visit via PageConfig::get_config().
 */

#include <optional>

#include <tt-metalium/shape2d.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/alignment.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/page_config.hpp>
#include <tt-metalium/experimental/tensor/spec/memory_config/memory_config.hpp>
#include <tt-metalium/experimental/tensor/tensor_types.hpp>

namespace tt::tt_metal {

Alignment create_default_alignment(const PageConfig& page_config, DataType dtype, const MemoryConfig& memory_config);
void validate_alignment(
    const PageConfig& page_config, const Alignment& alignment, DataType dtype, const MemoryConfig& memory_config);
Shape2D get_page_shape(
    const PageConfig& page_config,
    const Shape2D& physical_size,
    DataType dtype,
    const MemoryConfig& memory_config,
    const std::optional<Shape2D>& physical_shard_size);
size_t get_page_size_bytes(const PageConfig& page_config, const Shape2D& page_shape, DataType dtype);
Alignment get_required_shard_shape_alignment(const PageConfig& page_config);
Alignment get_recommended_shard_shape_alignment(const PageConfig& page_config, DataType dtype);

}  // namespace tt::tt_metal
