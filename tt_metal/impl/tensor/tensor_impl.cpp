// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/tensor/details/tensor_impl.hpp>

#include <tt-metalium/math.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt_stl/span.hpp>

namespace tt::tt_metal::tensor_impl {

// ======================================================================================
//     Helpers for converting between logical <-> physical data with full tensor spec
// ======================================================================================
namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

// Useful information about how a shard_shape cuts a 2D shape
// - num_shards_height: Number of shards along the height (including partial last shard, if any)
// - last_shard_height: Height of last partial shard (if None, it will be same as full shard shape height)
// - num_shards_width: Number of shards along the width (including partial last shard, if any)
// - last_shard_width: Width of last partial shard (if None, it will be same as full shard shape width)
struct ShardDivisionSpec {
    size_t num_shards_height = 0;
    size_t last_shard_height = 0;
    size_t num_shards_width = 0;
    size_t last_shard_width = 0;
};

ShardDivisionSpec compute_shard_division_spec(const Shape2D& shape, const Shape2D& shard_shape) {
    const auto num_shards_height = tt::div_up(shape.height(), shard_shape.height());
    const auto last_shard_height =
        shape.height() % shard_shape.height() > 0 ? shape.height() % shard_shape.height() : shard_shape.height();
    const auto num_shards_width = tt::div_up(shape.width(), shard_shape.width());
    const auto last_shard_width =
        shape.width() % shard_shape.width() > 0 ? shape.width() % shard_shape.width() : shard_shape.width();

    return ShardDivisionSpec{num_shards_height, last_shard_height, num_shards_width, last_shard_width};
};

// TODO: Remove when we get rid of physical sharding and generalize interleaved and sharded; when we do, directly get
// from TensorLayout
std::array<Shape2D, 2> get_logical_and_physical_shard_shapes(const TensorSpec& tensor_spec) {
    const auto& logical_shape = tensor_spec.logical_shape();
    const auto& padded_shape = tensor_spec.padded_shape();

    Shape2D logical_shard_shape{logical_shape[-2], logical_shape[-1]};
    Shape2D physical_shard_shape = {padded_shape[-2], padded_shape[-1]};
    return {logical_shard_shape, physical_shard_shape};
}

using LogicalPhysicalIdxPairs = std::vector<std::pair<size_t, size_t>>;
using LogicalPhysicalMapping = std::pair<LogicalPhysicalIdxPairs, size_t>;
std::vector<LogicalPhysicalMapping> compute_logical_to_physical_shards_mapping(
    const Shape2D& logical_2d_shape,
    const Shape2D& logical_shard_shape,
    const Shape2D& physical_shard_shape,
    const size_t physical_stride) {
    const auto logical_stride = logical_2d_shape.width();

    const auto [num_shards_height, last_shard_height, num_shards_width, last_shard_width] =
        compute_shard_division_spec(logical_2d_shape, logical_shard_shape);

    std::vector<LogicalPhysicalMapping> logical_physical_mapping{};
    logical_physical_mapping.reserve(num_shards_height * num_shards_width);

    for (size_t shard_height_idx = 0; shard_height_idx < num_shards_height; shard_height_idx++) {
        for (size_t shard_width_idx = 0; shard_width_idx < num_shards_width; shard_width_idx++) {
            const auto num_shard_rows =
                shard_height_idx == num_shards_height - 1 ? last_shard_height : logical_shard_shape.height();
            const auto num_shard_cols =
                shard_width_idx == num_shards_width - 1 ? last_shard_width : logical_shard_shape.width();

            auto indices = LogicalPhysicalIdxPairs(num_shard_rows);
            const auto logical_start_idx = (shard_height_idx * logical_shard_shape.height() * logical_stride) +
                                           (shard_width_idx * logical_shard_shape.width());
            const auto physical_start_idx = (shard_height_idx * physical_shard_shape.height() * physical_stride) +
                                            (shard_width_idx * physical_shard_shape.width());
            for (size_t i = 0; i < num_shard_rows; i++) {
                indices[i] = {(i * logical_stride) + logical_start_idx, (i * physical_stride) + physical_start_idx};
            }

            logical_physical_mapping.emplace_back(indices, num_shard_cols);
        }
    }
    return logical_physical_mapping;
};

// Converts a span of logical data to row major physical data.
template <typename T>
std::vector<T> convert_to_row_major_physical_data(
    tt::stl::Span<const T> logical_data, const TensorSpec& tensor_spec, T pad_value) {
    const auto& physical_shape = tensor_spec.physical_shape();
    const size_t physical_stride = physical_shape.width();
    auto [logical_shard_shape, physical_shard_shape] =
        CMAKE_UNIQUE_NAMESPACE::get_logical_and_physical_shard_shapes(tensor_spec);

    std::vector<T> row_major_physical_data(physical_shape.height() * physical_shape.width(), pad_value);

    const auto logical_physical_mapping = CMAKE_UNIQUE_NAMESPACE::compute_logical_to_physical_shards_mapping(
        tensor_spec.logical_2d_shape(), logical_shard_shape, physical_shard_shape, physical_stride);

    for (const auto& [indices, cols] : logical_physical_mapping) {
        for (const auto& [logical_idx_start, physical_idx_start] : indices) {
            for (size_t col = 0; col < cols; col++) {
                row_major_physical_data[physical_idx_start + col] = logical_data[logical_idx_start + col];
            }
        }
    }
    return row_major_physical_data;
}

// Converts a span of row major physical data to logical data.
template <typename T>
std::vector<T> convert_to_logical_data(tt::stl::Span<const T> row_major_physical_data, const TensorSpec& tensor_spec) {
    const auto& logical_2d_shape = tensor_spec.logical_2d_shape();
    const size_t physical_stride = tensor_spec.physical_shape().width();
    auto [logical_shard_shape, physical_shard_shape] =
        CMAKE_UNIQUE_NAMESPACE::get_logical_and_physical_shard_shapes(tensor_spec);

    std::vector<T> logical_data(logical_2d_shape.height() * logical_2d_shape.width(), 0);

    const auto logical_physical_mapping = CMAKE_UNIQUE_NAMESPACE::compute_logical_to_physical_shards_mapping(
        logical_2d_shape, logical_shard_shape, physical_shard_shape, physical_stride);

    for (const auto& [indices, cols] : logical_physical_mapping) {
        for (const auto& [logical_idx_start, physical_idx_start] : indices) {
            for (size_t col = 0; col < cols; col++) {
                logical_data[logical_idx_start + col] = row_major_physical_data[physical_idx_start + col];
            }
        }
    }
    return logical_data;
}

template <typename T>
std::vector<T> convert_layout_row_major_to_tile(
    const Shape2D& shape, const Tile& tile, tt::stl::Span<const T> data_to_convert) {
    if (shape.width() * shape.height() == 0) {
        return std::vector<T>();
    }
    TT_FATAL(
        (shape.height() % tile.get_tile_shape()[0] == 0 && shape.width() % tile.get_tile_shape()[1] == 0),
        "Unsupported shape for tensor conversion from row-major to tile layout. The tensor shape height and width must "
        "be a multiple of tile height ({}) and width ({}), but the provided shape is {}",
        tile.get_tile_shape()[0],
        tile.get_tile_shape()[1],
        shape);

    auto tile_shape = tile.get_tile_shape();
    auto face_shape = tile.get_face_shape();
    auto transpose_within_face = tile.get_transpose_within_face();
    auto transpose_of_faces = tile.get_transpose_of_faces();

    return convert_layout(
        data_to_convert,
        shape,
        TensorLayoutType::LIN_ROW_MAJOR,
        TensorLayoutType::TILED_NFACES,
        tile_shape,
        face_shape,
        transpose_within_face,
        transpose_of_faces);
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

template <typename T>
std::vector<T> encode_tensor_data(tt::stl::Span<const T> logical_data, const TensorSpec& tensor_spec, T pad_value) {
    if (logical_data.size() == 0) {
        return {};
    }

    const auto& logical_shape = tensor_spec.logical_shape();
    const auto& physical_shape = tensor_spec.physical_shape();

    TT_FATAL(
        logical_data.size() == logical_shape.volume(),
        "Logical data size {} should be same as volume indicated by logical shape {}",
        logical_data.size(),
        logical_shape);

    // If needed, convert logical data to row major physical data.
    // `row_major_physical_data_span` stores span unconditionally (cheap), while `row_major_physical_data` stores the
    // converted vector only when needed (expensive).
    std::vector<T> row_major_physical_data;
    tt::stl::Span<const T> row_major_physical_data_span;
    if (tensor_spec.logical_2d_shape() != physical_shape) {
        row_major_physical_data =
            CMAKE_UNIQUE_NAMESPACE::convert_to_row_major_physical_data(logical_data, tensor_spec, pad_value);
        row_major_physical_data_span = tt::stl::make_const_span(row_major_physical_data);
    } else {
        row_major_physical_data_span = logical_data;
    }

    TT_FATAL(
        row_major_physical_data_span.size() == physical_shape.height() * physical_shape.width(),
        "Physical data size {} should be same as volume indicated by physical shape {}",
        row_major_physical_data_span.size(),
        physical_shape);

    if (tensor_spec.layout() == Layout::TILE) {
        return CMAKE_UNIQUE_NAMESPACE::convert_layout_row_major_to_tile(
            physical_shape, tensor_spec.tile(), row_major_physical_data_span);
    }
    if (!row_major_physical_data.empty()) {
        // If conversion to physical data was performed, return the row major physical data to avoid extra copy.
        return row_major_physical_data;
    }  // Otherwise, copy the `row_major_physical_data_span`.
    return std::vector<T>(row_major_physical_data_span.begin(), row_major_physical_data_span.end());
}

template std::vector<bfloat16> encode_tensor_data<bfloat16>(
    tt::stl::Span<const bfloat16> logical_data, const TensorSpec& tensor_spec, bfloat16 pad_value);
template std::vector<float> encode_tensor_data<float>(
    tt::stl::Span<const float> logical_data, const TensorSpec& tensor_spec, float pad_value);
template std::vector<int32_t> encode_tensor_data<int32_t>(
    tt::stl::Span<const int32_t> logical_data, const TensorSpec& tensor_spec, int32_t pad_value);
template std::vector<uint32_t> encode_tensor_data<uint32_t>(
    tt::stl::Span<const uint32_t> logical_data, const TensorSpec& tensor_spec, uint32_t pad_value);
template std::vector<uint16_t> encode_tensor_data<uint16_t>(
    tt::stl::Span<const uint16_t> logical_data, const TensorSpec& tensor_spec, uint16_t pad_value);
template std::vector<uint8_t> encode_tensor_data<uint8_t>(
    tt::stl::Span<const uint8_t> logical_data, const TensorSpec& tensor_spec, uint8_t pad_value);

template <typename T>
std::vector<T> decode_tensor_data(tt::stl::Span<const T> physical_data, const TensorSpec& tensor_spec) {
    if (physical_data.size() == 0) {
        return {};
    }

    const auto& physical_shape = tensor_spec.physical_shape();
    TT_FATAL(
        physical_data.size() == physical_shape.height() * physical_shape.width(),
        "Physical data size {} should be same as volume indicated by physical shape {}",
        physical_data.size(),
        physical_shape);

    // If needed, convert physical data to row major physical data.
    // `row_major_physical_data_span` stores span unconditionally (cheap), while `row_major_physical_data` stores the
    // converted vector only when needed (expensive).
    std::vector<T> row_major_physical_data;
    tt::stl::Span<const T> row_major_physical_data_span;
    if (tensor_spec.layout() == Layout::TILE) {
        row_major_physical_data =
            tensor_impl::convert_layout_tile_to_row_major(physical_shape, tensor_spec.tile(), physical_data);
        row_major_physical_data_span = tt::stl::make_const_span(row_major_physical_data);
    } else {
        row_major_physical_data_span = physical_data;
    }

    // Same pattern as the above - `logical_data` is non empty only when the conversion to logical data was performed.
    std::vector<T> logical_data;
    tt::stl::Span<const T> logical_data_span;
    if (const auto& logical_2d_shape = tensor_spec.logical_2d_shape(); logical_2d_shape != physical_shape) {
        logical_data = CMAKE_UNIQUE_NAMESPACE::convert_to_logical_data(row_major_physical_data_span, tensor_spec);
        logical_data_span = tt::stl::make_const_span(logical_data);
    } else {
        logical_data_span = row_major_physical_data_span;
    }

    const auto& logical_shape = tensor_spec.logical_shape();
    TT_FATAL(
        logical_data_span.size() == logical_shape.volume(),
        "Logical data size {} should be same as volume indicated by logical shape {}",
        logical_data_span.size(),
        logical_shape);

    // Check if conversion to logical data was performed, to avoid extra copy upon return.
    if (!logical_data.empty()) {
        return logical_data;
    }
    if (!row_major_physical_data.empty()) {
        return row_major_physical_data;
    }
    return std::vector<T>(logical_data_span.begin(), logical_data_span.end());
}

template std::vector<bfloat16> decode_tensor_data<bfloat16>(
    tt::stl::Span<const bfloat16> physical_data, const TensorSpec& tensor_spec);
template std::vector<float> decode_tensor_data<float>(
    tt::stl::Span<const float> physical_data, const TensorSpec& tensor_spec);
template std::vector<int32_t> decode_tensor_data<int32_t>(
    tt::stl::Span<const int32_t> physical_data, const TensorSpec& tensor_spec);
template std::vector<uint32_t> decode_tensor_data<uint32_t>(
    tt::stl::Span<const uint32_t> physical_data, const TensorSpec& tensor_spec);
template std::vector<uint16_t> decode_tensor_data<uint16_t>(
    tt::stl::Span<const uint16_t> physical_data, const TensorSpec& tensor_spec);
template std::vector<uint8_t> decode_tensor_data<uint8_t>(
    tt::stl::Span<const uint8_t> physical_data, const TensorSpec& tensor_spec);

}  // namespace tt::tt_metal::tensor_impl
