// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/fmt.hpp>
#include <tt-metalium/experimental/tensor_layout_apis_with_custom_alignment.hpp>
#include <tt-metalium/math.hpp>

#include <tt_stl/assert.hpp>

#include "page_config_impl.hpp"
#include "tensor_layout_impl.hpp"

namespace tt::tt_metal::experimental {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

std::tuple<bool, bool> get_inner_hw_overpadded(
    const tt::tt_metal::Shape& logical_shape,
    const tt::tt_metal::Shape& legacy_padded_shape,
    const PageConfig& page_config) {
    if (page_config.get_layout() == Layout::TILE) {
        const auto& tile = page_config.get_tile();
        const uint32_t min_padded_h = tt::round_up(logical_shape[-2], tile.get_height());
        const uint32_t min_padded_w = tt::round_up(logical_shape[-1], tile.get_width());
        return {legacy_padded_shape[-2] > min_padded_h, legacy_padded_shape[-1] > min_padded_w};
    }
    // Always true for non-tile layouts; alignment is the standard padding mechanism for row-major tensors
    return {true, true};
}

// A sharded ROW_MAJOR tensor is padded to a multiple of its shard width, its default width alignment. Validation
// requires a 2D shard width to be a multiple of the width alignment, so a 2D-sharded tensor is padded to the shard
// width. An ND-sharded tensor keeps the given padded width, rounded up to a multiple of the shard width: an unrelated
// width alignment would be combined with the shard width by lcm, which can be many times either.
void pad_row_major_width_to_shard(
    ttsl::SmallVector<uint32_t>& values, const PageConfig& page_config, const MemoryConfig& memory_config) {
    if (page_config.get_layout() != Layout::ROW_MAJOR || !memory_config.is_sharded() || values.empty()) {
        return;
    }
    if (memory_config.shard_spec().has_value()) {
        values.back() = memory_config.shard_spec()->shape[1];
    } else {
        values.back() = tt::round_up(values.back(), memory_config.nd_shard_spec()->shard_shape[-1]);
    }
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

    // 2D SHARDED: check, then use the same height/width alignment as interleaved tensors below.
    if (memory_config.shard_spec().has_value()) {
        TT_FATAL(
            alignment_can_be_2D,
            "Tensor with shape {} ({}) cannot be sharded because alignment will have rank greater than 2!",
            logical_shape,
            legacy_padded_shape);
    }

    const auto [height_overpadded, width_overpadded] =
        get_inner_hw_overpadded(logical_shape, legacy_padded_shape, page_config);
    // Only height/width padding: interleaved and 2D-sharded tensors use the same alignment, so a 2D-sharded tensor
    // keeps the given padding.
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
        pad_row_major_width_to_shard(values, page_config, memory_config);
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
    pad_row_major_width_to_shard(values, page_config, memory_config);

    Alignment result(std::move(values));
    return result;
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

TensorLayout tensor_layout_with_custom_alignment(
    DataType dtype, const PageConfig& page_config, const MemoryConfig& memory_config, const Alignment& alignment) {
    return TensorLayout(std::make_unique<TensorLayoutImpl>(dtype, page_config, memory_config, alignment));
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

}  // namespace tt::tt_metal::experimental
