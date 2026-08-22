// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt_stl/span.hpp>
#include <array>
#include <cstdint>
#include <optional>
#include <vector>
#include <iosfwd>

enum class TensorLayoutType {
    LIN_ROW_MAJOR = 0,   // standard element-wise row-major
    TILED_SWIZZLED = 1,  // row-major of tiles, each tile is row-major-swizzled
    TILED_NFACES = 2,    // row-major of tiles, each tile is N (N = 1, 2, or 4) faces, each face is
                         // row-major, faces are swizzled
};
std::ostream& operator<<(std::ostream& os, TensorLayoutType layout);

using PhysicalSize = std::array<uint32_t, 2>;

std::uint32_t round_up_to_mul32(std::uint32_t val);

template <typename T>
std::vector<T> convert_layout(
    ttsl::Span<const T> inp,
    const PhysicalSize& shape,
    TensorLayoutType inL,
    TensorLayoutType outL,
    std::optional<PhysicalSize> tile_shape = std::nullopt,
    std::optional<PhysicalSize> face_shape = std::nullopt,
    bool transpose_within_face = false,
    bool transpose_of_faces = false);

template <typename T>
std::vector<T> convert_layout(
    ttsl::Span<const T> inp,
    ttsl::Span<const uint32_t> shape,
    TensorLayoutType inL,
    TensorLayoutType outL,
    std::optional<PhysicalSize> tile_shape = std::nullopt,
    std::optional<PhysicalSize> face_shape = std::nullopt,
    bool transpose_within_face = false,
    bool transpose_of_faces = false);
