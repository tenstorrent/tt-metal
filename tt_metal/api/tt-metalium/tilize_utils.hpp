// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

//
// This header contains tile conversion functions used in tests on the host.
//

#pragma once

#include <tracy/Tracy.hpp>
#include <tt-metalium/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt_stl/span.hpp>
#include <array>
#include <concepts>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <vector>

namespace tests::utils {
enum class TensorLayoutType {
    LIN_ROW_MAJOR = 0,   // standard element-wise row-major
    TILED_SWIZZLED = 1,  // row-major of tiles, each tile is row-major-swizzled
    TILED_NFACES =
        2,  // row-major of tiles, each tile is N (N = 1, 2, or 4) faces, each face is row-major, faces are swizzled
};
}  // namespace tests::utils

using PhysicalSize = std::array<uint32_t, 2>;

struct TensAddr {
    std::vector<std::uint32_t> sh;

    TensAddr(const std::vector<std::uint32_t>& shape);
    std::uint32_t numel() const;
    int offs(int n, int c, int h, int w);
};

std::uint32_t round_up_to_mul16(std::uint32_t val);

std::uint32_t round_up_to_mul32(std::uint32_t val);

std::uint32_t round_up_to_tile(int val, int tile_val);

template <class T>
std::vector<T> convert_to_tile_layout(
    tt::stl::Span<const T> data,
    std::optional<PhysicalSize> tile_shape = std::nullopt,
    std::optional<PhysicalSize> face_shape = std::nullopt,
    bool transpose_face = false,
    bool transpose_face_order = false);

template <class T>
std::vector<T> convert_to_flat_layout(
    tt::stl::Span<const T> data,
    std::optional<PhysicalSize> tile_shape = std::nullopt,
    std::optional<PhysicalSize> face_shape = std::nullopt,
    bool transpose_face = false,
    bool transpose_face_order = false);

template <typename T>
std::vector<T> convert_layout(
    tt::stl::Span<const T> data,
    const PhysicalSize& shape,
    tests::utils::TensorLayoutType inL,
    tests::utils::TensorLayoutType outL,
    std::optional<PhysicalSize> tile_shape = std::nullopt,
    std::optional<PhysicalSize> face_shape = std::nullopt,
    bool transpose_within_face = false,
    bool transpose_of_faces = false);

template <typename T>
std::vector<T> convert_layout(
    tt::stl::Span<const T> data,
    tt::stl::Span<const uint32_t> shape,
    tests::utils::TensorLayoutType inL,
    tests::utils::TensorLayoutType outL,
    std::optional<PhysicalSize> tile_shape = std::nullopt,
    std::optional<PhysicalSize> face_shape = std::nullopt,
    bool transpose_within_face = false,
    bool transpose_of_faces = false);

template <typename T>
void tilize(std::vector<T>& input, uint32_t m, uint32_t n);

template <typename T>
void untilize(std::vector<T>& input, uint32_t m, uint32_t n);
