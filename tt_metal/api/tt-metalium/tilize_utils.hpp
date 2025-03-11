// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

//
// This header contains tile conversion functions used in tests on the host.
//

#pragma once

#include <cstdint>
#include <vector>
#include <optional>
#include <concepts>
#include <type_traits>

#include <tt-metalium/span.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/assert.hpp>
#include <tt-metalium/math.hpp>

#include <tracy/Tracy.hpp>

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

    TensAddr(std::vector<std::uint32_t> shape);
    std::uint32_t numel() const;
    int offs(int n, int c, int h, int w);
};

std::uint32_t round_up_to_mul16(std::uint32_t val);

std::uint32_t round_up_to_mul32(std::uint32_t val);

std::uint32_t round_up_to_tile(int val, int tile_val);

/*
    BufferProxy represents a data source.
    It is a wrapper around anything supporting operator[] and size()
    like std::vector or any other kind of data source (e.g. TT-NN OwnedStorage or BorrowedStorage)
*/
template <class T>
class BufferProxy {
public:
    template <typename BufferType>
    BufferProxy(const BufferType& buffer) : buffer_(buffer.data(), buffer.size()) {}

    // Constructor specifically for span
    BufferProxy(tt::stl::Span<T> buffer) : buffer_(buffer) {}

    // Copy constructor
    BufferProxy(const BufferProxy& other) : buffer_(other.buffer_) {}

    const T& operator[](std::size_t index) const { return buffer_[index]; }

    std::size_t size() const { return buffer_.size(); }

private:
    tt::stl::Span<const T> buffer_;
};

template <class T>
std::vector<T> convert_to_tile_layout(
    const BufferProxy<T>& data,
    std::optional<PhysicalSize> tile_shape = std::nullopt,
    std::optional<PhysicalSize> face_shape = std::nullopt,
    bool transpose_face = false,
    bool transpose_face_order = false);

template <class T>
std::vector<T> convert_to_flat_layout(
    const BufferProxy<T>& data,
    std::optional<PhysicalSize> tile_shape = std::nullopt,
    std::optional<PhysicalSize> face_shape = std::nullopt,
    bool transpose_face = false,
    bool transpose_face_order = false);

// Converts a 32-swizzled tilized row-major tensor to a linear 32-zero-padded row-major tensor
template <typename T>
std::vector<T> untilize_nchw(
    const BufferProxy<T>& in, const PhysicalSize& shape, std::optional<PhysicalSize> tile_shape = std::nullopt);

// Converts a linear non-zero-padded row-major tensor to 32-swizzled tilized row-major tensor
template <typename T>
std::vector<T> tilize_nchw(
    const BufferProxy<T>& in_rowmajor,
    const PhysicalSize& shape,
    std::optional<PhysicalSize> tile_shape = std::nullopt);

template <typename T>
std::vector<T> convert_layout(
    const BufferProxy<T>& inp,
    const PhysicalSize& shape,
    tests::utils::TensorLayoutType inL,
    tests::utils::TensorLayoutType outL,
    std::optional<PhysicalSize> tile_shape = std::nullopt,
    std::optional<PhysicalSize> face_shape = std::nullopt,
    bool transpose_within_face = false,
    bool transpose_of_faces = false);

template <typename T>
std::vector<T> convert_layout(
    const BufferProxy<T>& inp,
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
