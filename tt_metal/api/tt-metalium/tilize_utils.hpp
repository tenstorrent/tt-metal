// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

//
// This header contains tile conversion functions used in tests on the host.
//

#pragma once

#include <tt_stl/span.hpp>
#include <array>
#include <cstdint>
#include <optional>
#include <vector>
#include <iosfwd>

// Used in ops
std::uint32_t round_up_to_mul32(std::uint32_t val);

// Used in sdpa decode op and tt-train examples
std::uint32_t round_up_to_tile(int val, int tile_val);

// Used in programming examples
template <typename T>
std::vector<T> tilize_nfaces(const std::vector<T>& input, uint32_t m, uint32_t n);

// Used in programming examples
template <typename T>
std::vector<T> untilize_nfaces(const std::vector<T>& input, uint32_t m, uint32_t n);

// Below are functions used either only in tests/ or for implementing tensor

// Only used in tensor_impl
enum class TensorLayoutType {
    LIN_ROW_MAJOR = 0,   // standard element-wise row-major
    TILED_SWIZZLED = 1,  // row-major of tiles, each tile is row-major-swizzled
    TILED_NFACES = 2,    // row-major of tiles, each tile is N (N = 1, 2, or 4) faces, each face is
                         // row-major, faces are swizzled
};
std::ostream& operator<<(std::ostream& os, TensorLayoutType layout);

// Awful name
using PhysicalSize = std::array<uint32_t, 2>;

// Used only in tests/
template <class T>
std::vector<T> convert_layout_tile_swizzled_to_tile_nfaces(
    tt::stl::Span<const T> data,
    std::optional<PhysicalSize> tile_shape = std::nullopt,
    std::optional<PhysicalSize> face_shape = std::nullopt,
    bool transpose_face = false,
    bool transpose_face_order = false);

// Used only in tests/
template <class T>
std::vector<T> convert_layout_tile_nfaces_to_tile_swizzled(
    tt::stl::Span<const T> data,
    std::optional<PhysicalSize> tile_shape = std::nullopt,
    std::optional<PhysicalSize> face_shape = std::nullopt,
    bool transpose_face = false,
    bool transpose_face_order = false);

// Used only in tests and tensor impl
template <typename T>
std::vector<T> convert_layout(
    tt::stl::Span<const T> data,
    const PhysicalSize& shape,
    TensorLayoutType inL,
    TensorLayoutType outL,
    std::optional<PhysicalSize> tile_shape = std::nullopt,
    std::optional<PhysicalSize> face_shape = std::nullopt,
    bool transpose_within_face = false,
    bool transpose_of_faces = false);

// Used only in tests and tensor impl
template <typename T>
std::vector<T> convert_layout(
    tt::stl::Span<const T> data,
    tt::stl::Span<const uint32_t> shape,
    TensorLayoutType inL,
    TensorLayoutType outL,
    std::optional<PhysicalSize> tile_shape = std::nullopt,
    std::optional<PhysicalSize> face_shape = std::nullopt,
    bool transpose_within_face = false,
    bool transpose_of_faces = false);

// Those are specific cases that convert_layout can do, but left for compatibility with existing codebase
// Converts from/to row major
// Used only in tests
template <typename T>
std::vector<T> tilize_swizzled(const std::vector<T>& input, uint32_t m, uint32_t n);

// Used only in tests
template <typename T>
std::vector<T> untilize_swizzled(const std::vector<T>& input, uint32_t m, uint32_t n);
