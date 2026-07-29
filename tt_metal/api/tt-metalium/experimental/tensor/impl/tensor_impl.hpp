// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/tilize_utils.hpp>

#include <tt_stl/span.hpp>
#include <vector>

/**
 * Functions in this file are internal utilities for Runtime Tensors.
 * They are exported out in the public API area as a transiet state,
 * many of them are used by ttnn python binding.
 * We should disperse these functions as public APIs in tensor_apis.hpp or make them private to tt_metal.
 */

namespace tt::tt_metal::tensor_impl {

// ======================================================================================
//                           Data reader, writer, and initializers
// ======================================================================================

// Converts physical data into logical data based on tensor spec
// - Physical data: Flat container of physical data corresponding to tensor spec
//   * Assumes that the physical data already matches tensor spec
//   * There is a bare minimum check that size of physical data matches size indicated by tensor_spec.physical_shape()
// - Logical data: Flat container of row major data corresponding to some ND logical shape
//   * Inverse of the private encode_tensor_data helper (padding / untilize)
//   * Resulting data is safe to be converted to python tensors or general consumption with just a ND logical shape
template <typename T>
std::vector<T> decode_tensor_data(ttsl::Span<const T> physical_data, const TensorSpec& tensor_spec);

// ======================================================================================
//                                  Layout converters
// ======================================================================================

template <typename T>
std::vector<T> to_row_major_layout(const Shape2D& shape, const Tile& tile, ttsl::Span<const T> data_to_convert) {
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

template <typename T>
std::vector<T> to_tile_major_layout(const Shape2D& shape, const Tile& tile, ttsl::Span<const T> data_to_convert);

// Empty structs to facilitate Tensor template logic.
struct bfloat4_b {};
struct bfloat8_b {};

// Utility to convert runtime DataType to compile-time constant and dispatch the function call
template <typename Func, typename... Args>
auto dispatch(DataType dtype, Func&& func, Args&&... args) {
    switch (dtype) {
        case DataType::BFLOAT16:
            return (std::forward<Func>(func)).template operator()<bfloat16>(std::forward<Args>(args)...);
        case DataType::FLOAT32:
            return (std::forward<Func>(func)).template operator()<float>(std::forward<Args>(args)...);
        case DataType::INT32:
            return (std::forward<Func>(func)).template operator()<int32_t>(std::forward<Args>(args)...);
        case DataType::UINT32:
            return (std::forward<Func>(func)).template operator()<uint32_t>(std::forward<Args>(args)...);
        case DataType::UINT16:
            return (std::forward<Func>(func)).template operator()<uint16_t>(std::forward<Args>(args)...);
        case DataType::UINT8:
            return (std::forward<Func>(func)).template operator()<uint8_t>(std::forward<Args>(args)...);
        case DataType::BFLOAT8_B:
            return (std::forward<Func>(func)).template operator()<bfloat8_b>(std::forward<Args>(args)...);
        case DataType::BFLOAT4_B:
            return (std::forward<Func>(func)).template operator()<bfloat4_b>(std::forward<Args>(args)...);
        case DataType::FP8_E4M3:
            return (std::forward<Func>(func)).template operator()<float8_e4m3>(std::forward<Args>(args)...);
        default: TT_THROW("Unsupported data type");
    }
}

}  // namespace tt::tt_metal::tensor_impl
