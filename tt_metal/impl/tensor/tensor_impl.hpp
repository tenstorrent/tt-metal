// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <utility>
#include <vector>

#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/shape2d.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/span.hpp>

/**
 * Private tensor_impl utilities for Runtime Tensors.
 *
 * Not part of the installed public API. Prefer HostTensor / MeshTensor /
 * tensor_apis surfaces at call sites outside this directory.
 */

namespace tt::tt_metal::tensor_impl {

template <typename T>
std::vector<T> encode_tensor_data(ttsl::Span<const T> logical_data, const TensorSpec& tensor_spec, T pad_value = 0);

template <typename T>
std::vector<T> to_tile_major_layout(const Shape2D& shape, const Tile& tile, ttsl::Span<const T> data_to_convert);

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
