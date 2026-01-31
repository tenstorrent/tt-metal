// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt_stl/span.hpp>
#include <tt-metalium/shape2d.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>

namespace tt::tt_metal::tensor_impl {

// ======================================================================================
//                                  Layout converters
// ======================================================================================

template <typename T>
std::vector<T> convert_layout_tile_to_row_major(
    const Shape2D& shape, const Tile& tile, tt::stl::Span<const T> data_to_convert) {
    auto tile_shape = tile.get_tile_shape();
    auto face_shape = tile.get_face_shape();
    auto transpose_within_face = tile.get_transpose_within_face();
    auto transpose_of_faces = tile.get_transpose_of_faces();

    return convert_layout(
        data_to_convert,
        shape,
        TensorLayoutType::TILED_NFACES,
        TensorLayoutType::LIN_ROW_MAJOR,
        tile_shape,
        face_shape,
        transpose_within_face,
        transpose_of_faces);
}

// Converts logical data into physical data based on tensor spec
// - Logical data: Flat container of row major data corresponding to some ND logical shape
// - Physical data: Flat container of physical data corresponding to tensor spec. It takes into account:
//   * Sharding: Each shard will be padded to nearest page (if needed)
//     ** This is mostly for logical sharding, since logical shards may not be aligned to page in general
//     ** For interleaved, it will be handled as a "logically sharded" tensor with same shard shard height/width
//        as the original tensor dims at -2 and -1. In the future, interleaved may be generalized as sharded.
//     ** This means padding may be inserted in the middle of logical data (if needed)
//   * Layout: Each aligned shard will be tilized (if needed)
//     ** Tilization happens after first inserting padding to align shards (if needed)
//     ** For the last shard, we only align to nearest page instead of full shard size for partial shards
//   * After conversion, size of physical data will match 2D physical size indicated by tensor_spec.physical_shape()
template <typename T>
std::vector<T> encode_tensor_data(tt::stl::Span<const T> logical_data, const TensorSpec& tensor_spec, T pad_value = 0);

// Converts physical data into logical data based on tensor spec (see encode_tensor_data for details)
// - Physical data: Flat container of physical data corresponding to tensor spec
//   * Assumes that the physical data already matches tensor spec
//   * There is a bare minimum check that size of physical data matches size indicated by tensor_spec.physical_shape()
// - Logical data: Flat container of row major data corresponding to some ND logical shape
//   * To get logical data, perform the exact inverse process of encode_tensor_data
//   * Resulting data is safe to be converted to python tensors or general consumption with just a ND logical shape
template <typename T>
std::vector<T> decode_tensor_data(tt::stl::Span<const T> physical_data, const TensorSpec& tensor_spec);

}  // namespace tt::tt_metal::tensor_impl
