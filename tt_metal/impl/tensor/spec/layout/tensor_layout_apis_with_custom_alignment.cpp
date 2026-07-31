// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/fmt.hpp>
#include <tt-metalium/experimental/tensor_layout_apis_with_custom_alignment.hpp>

#include <tt_stl/assert.hpp>

#include "page_config_impl.hpp"
#include "tensor_layout_impl.hpp"

namespace tt::tt_metal {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

size_t round_up(size_t value, size_t multiple) {
    if (multiple == 0) {
        return value;
    }

    return ((value + multiple - 1) / multiple) * multiple;
};

std::tuple<bool, bool> get_inner_hw_overpadded(
    const tt::tt_metal::Shape& logical_shape,
    const tt::tt_metal::Shape& legacy_padded_shape,
    const PageConfig& page_config) {
    if (page_config.get_layout() == Layout::TILE) {
        const auto& tile = page_config.get_tile();
        const uint32_t min_padded_h = round_up(logical_shape[-2], tile.get_height());
        const uint32_t min_padded_w = round_up(logical_shape[-1], tile.get_width());
        return {legacy_padded_shape[-2] > min_padded_h, legacy_padded_shape[-1] > min_padded_w};
    }
    // Always true for non-tile layouts; alignment is the standard padding mechanism for row-major tensors
    return {true, true};
}

Alignment legacyShapeToAlignment(
    const tt::tt_metal::Shape& logical_shape,
    const tt::tt_metal::Shape& legacy_padded_shape,
    const PageConfig& page_config,
    const MemoryConfig& memory_config) {
    if (logical_shape == legacy_padded_shape) {
        return Alignment{};
    }

    const int padded_rank = legacy_padded_shape.rank();
    bool alignment_can_be_2D = true;
    for (int i = -3; i >= -padded_rank; i--) {
        alignment_can_be_2D &= logical_shape[i] == legacy_padded_shape[i];
    }

    // 2D SHARDED
    if (memory_config.shard_spec().has_value()) {
        TT_FATAL(
            alignment_can_be_2D,
            "Tensor with shape {} ({}) cannot be sharded because alignment will have rank greater than 2!",
            logical_shape,
            legacy_padded_shape);
        if (page_config.get_layout() == Layout::ROW_MAJOR) {
            const auto& shard_spec = memory_config.shard_spec().value();
            return Alignment{shard_spec.shape[1]};
        }
        return Alignment{};
    }

    const auto [height_overpadded, width_overpadded] =
        get_inner_hw_overpadded(logical_shape, legacy_padded_shape, page_config);
    // INTERLEAVED with only height/width padding
    if (alignment_can_be_2D) {
        ttsl::SmallVector<uint32_t> values(std::min((int)padded_rank, 2));
        const auto alignment_size = values.size();
        if (alignment_size >= 1) {
            values[alignment_size - 1] =
                width_overpadded ? legacy_padded_shape[-1] : page_config.get_tile().get_width();
        }
        if (alignment_size == 2) {
            values[alignment_size - 2] =
                height_overpadded ? legacy_padded_shape[-2] : page_config.get_tile().get_height();
        }
        Alignment result(std::move(values));
        return result;
    }

    // INTERLEAVED with (deprecated) non-height/width padding
    // NOTE: Rank > 2 is guaranteed in this case
    ttsl::SmallVector<uint32_t> values(padded_rank);

    // When the inner dimensions are not over-padded beyond the logical H/W, use the tile width and height
    // for the innermost alignment; otherwise use the legacy padded H/W.
    values[padded_rank - 1] = width_overpadded ? legacy_padded_shape[-1] : page_config.get_tile().get_width();
    values[padded_rank - 2] = height_overpadded ? legacy_padded_shape[-2] : page_config.get_tile().get_height();

    uint32_t cumulative_padded_volume = legacy_padded_shape[-2];
    for (int dim = padded_rank - 3; dim >= 0; dim--) {
        cumulative_padded_volume *= legacy_padded_shape[dim];
        values[dim] = cumulative_padded_volume;
    }

    for (auto& value : values) {
        if (value == 0) {
            value = 1;
        }
    }

    Alignment result(std::move(values));
    return result;
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

TensorLayout tensor_layout_with_custom_alignment(
    DataType dtype, const PageConfig& page_config, const MemoryConfig& memory_config, const Alignment& alignment) {
    TensorLayout result(dtype, page_config, memory_config);
    result.impl().set_custom_alignment(alignment);
    return result;
}

TensorLayout tensor_layout_from_padded_shape(
    DataType dtype,
    const PageConfig& page_config,
    const MemoryConfig& memory_config,
    const tt::tt_metal::Shape& logical_shape,
    const tt::tt_metal::Shape& padded_shape) {
    return tensor_layout_with_custom_alignment(
        dtype,
        page_config,
        memory_config,
        CMAKE_UNIQUE_NAMESPACE::legacyShapeToAlignment(logical_shape, padded_shape, page_config, memory_config));
}

}  // namespace tt::tt_metal
