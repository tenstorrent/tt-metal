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
#include "span.hpp"
#include "constants.hpp"
#include "assert.hpp"
#include "tracy/Tracy.hpp"
#include "math.hpp"

namespace tests::utils {
enum class TensorLayoutType {
    LIN_ROW_MAJOR = 0, // standard element-wise row-major
    TILED_SWIZZLED = 1, // row-major of tiles, each tile is row-major-swizzled
    TILED_NFACES = 2,  // row-major of tiles, each tile is N (N = 1, 2, or 4) faces, each face is row-major, faces are swizzled
};
} // namespace tests::utils

using PhysicalSize = std::array<uint32_t, 2>;

template <class T, template <typename...> typename BufferType>
std::vector<T> convert_to_tile_layout(
    const BufferType<T>& data,
    std::optional<PhysicalSize> tile_shape = std::nullopt,
    std::optional<PhysicalSize> face_shape = std::nullopt,
    const bool transpose_face = false,
    const bool transpose_face_order = false) {
    ZoneScoped;
    std::vector<T> result;
    if(data.size() == 0) {
        return result;
    }

    result.reserve(data.size());
    auto tile_H = tile_shape.has_value() ? tile_shape.value()[0] : tt::constants::TILE_HEIGHT;
    auto tile_W = tile_shape.has_value() ? tile_shape.value()[1] : tt::constants::TILE_WIDTH;
    auto face_H = face_shape.has_value() ? face_shape.value()[0] : tt::constants::FACE_HEIGHT;
    auto face_W = face_shape.has_value() ? face_shape.value()[1] : tt::constants::FACE_WIDTH;
    auto tile_HW = tile_H * tile_W;
    auto face_HW = face_H * face_W;
    TT_ASSERT(data.size() % tile_HW == 0);
    int num_tiles = data.size() / tile_HW;
    for(int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        std::vector<T> top_left;
        std::vector<T> top_right;
        std::vector<T> bottom_left;
        std::vector<T> bottom_right;

        if (transpose_face) {
            for(int col = 0; col < tile_W; col++) {
                int index = tile_idx * tile_HW + col;
                for(int row = 0; row < tile_H; row++) {
                    if(row < face_H and col < face_W) {
                        top_left.push_back(data[index]);
                    } else if(row < face_H and col >= face_W) {
                        top_right.push_back(data[index]);
                    } else if(row >= face_H and col < face_W) {
                        bottom_left.push_back(data[index]);
                    } else if(row >= face_H and col >= face_W) {
                        bottom_right.push_back(data[index]);
                    } else {
                        TT_ASSERT(false);
                    }
                    index += tile_W;
                }
            }
        } else {
            int index = tile_idx * tile_HW;
            for(int row = 0; row < tile_H; row++) {
                for(int col = 0; col < tile_W; col++) {
                    if(row < face_H and col < face_W) {
                        top_left.push_back(data[index]);
                    } else if(row < face_H and col >= face_W) {
                        top_right.push_back(data[index]);
                    } else if(row >= face_H and col < face_W) {
                        bottom_left.push_back(data[index]);
                    } else if(row >= face_H and col >= face_W) {
                        bottom_right.push_back(data[index]);
                    } else {
                        TT_ASSERT(false);
                    }
                    index++;
                }
            }
        }
        TT_ASSERT(top_left.size() == face_HW);
        TT_ASSERT((top_right.size() == 0) or (top_right.size() == face_HW));
        TT_ASSERT((bottom_left.size() == 0) or (bottom_left.size() == face_HW));
        TT_ASSERT((bottom_right.size() == 0) or (bottom_right.size() == face_HW));

        if (transpose_face_order) {
            result.insert(result.end(), top_left.begin(), top_left.end());
            result.insert(result.end(), bottom_left.begin(), bottom_left.end());
            result.insert(result.end(), top_right.begin(), top_right.end());
            result.insert(result.end(), bottom_right.begin(), bottom_right.end());
        } else {
            result.insert(result.end(), top_left.begin(), top_left.end());
            result.insert(result.end(), top_right.begin(), top_right.end());
            result.insert(result.end(), bottom_left.begin(), bottom_left.end());
            result.insert(result.end(), bottom_right.begin(), bottom_right.end());
        }
    }

    return result;
}

template <class T, template <typename...> typename BufferTyp>
std::vector<T> convert_to_flat_layout(
    const BufferTyp<T>& data,
    std::optional<PhysicalSize> tile_shape = std::nullopt,
    std::optional<PhysicalSize> face_shape = std::nullopt,
    const bool transpose_face = false,
    const bool transpose_face_order = false) {
    ZoneScoped;
    std::vector<T> result;
    if(data.size() == 0) {
        return result;
    }
    result.reserve(data.size());
    auto tile_H = tile_shape.has_value() ? tile_shape.value()[0] : tt::constants::TILE_HEIGHT;
    auto tile_W = tile_shape.has_value() ? tile_shape.value()[1] : tt::constants::TILE_WIDTH;
    auto face_H = face_shape.has_value() ? face_shape.value()[0] : tt::constants::FACE_HEIGHT;
    auto face_W = face_shape.has_value() ? face_shape.value()[1] : tt::constants::FACE_WIDTH;
    auto tile_HW = tile_H * tile_W;
    auto face_HW = face_H * face_W;
    auto num_faces_col = tile_W / face_W;
    auto num_faces_row = tile_H / face_H;
    TT_ASSERT(data.size() % tile_HW == 0);
    int num_tiles = data.size() / tile_HW;
    for(int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        int tile_start = tile_idx * tile_HW;

        if (transpose_face) {
            if (num_faces_row >= 1 && num_faces_col <= 1) { // 32x16
                for(int face_y = 0; face_y < num_faces_row; face_y++) {
                    int start = tile_start + face_y * (face_H * tile_W);
                    for(int col = 0; col < face_W; col++) {
                        for(int row = 0; row < face_H; row++) {
                            result.push_back(data[start + col + row * face_W]);
                        }
                    }
                }
            } else if (num_faces_row <= 1 && num_faces_col >= 1) { // 16x32
                for(int col = 0; col < face_W; col++) {
                    int start = tile_start + col;
                    for(int face_x = 0; face_x < num_faces_col; face_x++) {
                        int offset = face_x * face_HW;
                        for(int row = 0; row < face_H; row++) {
                            result.push_back(data[start + offset + row * face_W]);
                        }
                    }
                }
            } else {
                for(int face_x = 0; face_x < num_faces_col; face_x++) {
                    for(int col = 0; col < face_W; col++) {
                        int start = tile_start + face_x * face_HW + col;
                        for(int face_y = 0; face_y < num_faces_row; face_y++) {
                            int offset = face_y * (face_H * tile_W);
                            for(int row = 0; row < face_H; row++) {
                                result.push_back(data[start + offset + row * face_W]);
                            }
                        }
                    }
                }
            }
        } else {
            for(int face_y = 0; face_y < num_faces_row; face_y++) {
                for(int row = 0; row < face_H; row++) {
                    int start = tile_start + face_y * (face_H * tile_W) + row * face_W;
                    for(int face_x = 0; face_x < num_faces_col; face_x++) {
                        int offset = face_x * face_HW;
                        for(int col = offset; col < offset + face_W; col++) {
                            result.push_back(data[start + col]);
                        }
                    }
                }
            }
        }
    }

    return result;
}

// Converts a 32-swizzled tilized row-major tensor to a linear 32-zero-padded row-major tensor
template <typename T, template <typename...> typename BufferType>
inline std::vector<T> untilize_nchw(
    const BufferType<T>& in, const PhysicalSize& shape, std::optional<PhysicalSize> tile_shape = std::nullopt) {
    ZoneScoped;
    std::vector<T> result;
    if(in.size() == 0) {
        return result;
    }

    auto tile_H = tile_shape.has_value() ? tile_shape.value()[0] : tt::constants::TILE_HEIGHT;
    auto tile_W = tile_shape.has_value() ? tile_shape.value()[1] : tt::constants::TILE_WIDTH;

    TT_ASSERT(shape[0] % tile_H == 0 && shape[1] % tile_W == 0);

    // Untilize into row major
    uint32_t H = shape[0];
    uint32_t W = shape[1];

    result.resize(H * W);
    uint64_t linear = 0;
    for (auto hs = 0; hs < H; hs += tile_H) {           // iterate over h with stride 32
        for (auto ws = 0; ws < W; ws += tile_W) {       // iterate over w with stride 32
            for (auto ht = 0; ht < tile_H; ht++) {      // hs + ht = h
                for (auto wt = 0; wt < tile_W; wt++) {  // ws + wt = w
                    T val = in[linear];
                    auto w = wt + ws;
                    auto h = ht + hs;
                    auto offs = w + h * W;  // + batch_index * H * W;
                    result[offs] = val;
                    linear++;
                }
            }
        }
    }

    return result;
}

inline std::uint32_t round_up_to_mul16(std::uint32_t val) { return ((val & 15) == 0) ? val : (val | 15)+1; }

inline std::uint32_t round_up_to_mul32(std::uint32_t val) { return ((val & 31) == 0) ? val : (val | 31)+1; }

inline std::uint32_t round_up_to_tile(int val, int tile_val) { return (val + tile_val - 1) & ~(tile_val - 1); }

// Converts a linear non-zero-padded row-major tensor to 32-swizzled tilized row-major tensor
template <typename T, template <typename...> typename BufferType>
inline std::vector<T> tilize_nchw(
    const BufferType<T>& in_rowmajor,
    const PhysicalSize& shape,
    std::optional<PhysicalSize> tile_shape = std::nullopt) {
    ZoneScoped;
    std::vector<T> tilized_result;
    if(in_rowmajor.size() == 0) {
        return tilized_result;
    }

    auto tile_H = tile_shape.has_value() ? tile_shape.value()[0] : tt::constants::TILE_HEIGHT;
    auto tile_W = tile_shape.has_value() ? tile_shape.value()[1] : tt::constants::TILE_WIDTH;

    TT_ASSERT(shape[0] % tile_H == 0 && shape[1] % tile_W == 0);

    uint32_t H = shape[0];
    uint32_t W = shape[1];

    tilized_result.resize(H * W);
    uint64_t out_index = 0;
    for (auto hs = 0; hs < H; hs += tile_H) {
        for (auto ws = 0; ws < W; ws += tile_W) {
            for (auto ht = 0; ht < tile_H; ht++) {
                for (auto wt = 0; wt < tile_W; wt++) {
                    auto w = wt + ws;
                    auto h = ht + hs;
                    auto in_offs = w + h * W;
                    auto val = in_rowmajor[in_offs];
                    tilized_result[out_index] = val;
                    out_index++;
                }
            }
        }
    }

    return tilized_result;
}

struct TensAddr {
    std::vector<std::uint32_t> sh;

    std::uint32_t numel() const {
        std::uint32_t prod = 1;
        for (int j = 0; j < sh.size(); j ++)
            prod *= sh[j];
        return prod;
    }

    TensAddr(std::vector<std::uint32_t> shape) : sh(shape) {}
    int offs(int n, int c, int h, int w) {
        TT_ASSERT(std::uint32_t(n) < sh[0] && std::uint32_t(c) < sh[1] && std::uint32_t(h) < sh[2] && std::uint32_t(w) < sh[3]);
        return w + sh[3]*h + sh[2]*sh[3]*c + sh[1]*sh[2]*sh[3]*n;
    }
};

template <typename T, template <typename...> typename BufferType>
inline std::vector<T> convert_layout(
    const BufferType<T>& inp,
    const PhysicalSize& shape,
    tests::utils::TensorLayoutType inL,
    tests::utils::TensorLayoutType outL,
    std::optional<PhysicalSize> tile_shape = std::nullopt,
    std::optional<PhysicalSize> face_shape = std::nullopt,
    const bool transpose_within_face = false,
    const bool transpose_of_faces = false) {
    ZoneScoped;
    if(inp.size() == 0) {
        return std::vector<T>();
    }

    switch (inL) {
        case tests::utils::TensorLayoutType::TILED_SWIZZLED:
            if (outL == tests::utils::TensorLayoutType::TILED_NFACES) {
                return convert_to_tile_layout<T>(inp, tile_shape, face_shape, transpose_within_face, transpose_of_faces);
            } else if (outL == tests::utils::TensorLayoutType::LIN_ROW_MAJOR) {
                return untilize_nchw<T>(inp, shape, tile_shape);
            } else
                TT_ASSERT(false && "Unsupported conversion.");
        break;
        case tests::utils::TensorLayoutType::LIN_ROW_MAJOR:
            if (outL == tests::utils::TensorLayoutType::TILED_SWIZZLED) {
                return tilize_nchw<T>(inp, shape, tile_shape);
            } else if (outL == tests::utils::TensorLayoutType::TILED_NFACES) {
                auto swiz32 = tilize_nchw<T>(inp, shape, tile_shape);
                return convert_to_tile_layout<T>(
                    swiz32, tile_shape, face_shape, transpose_within_face, transpose_of_faces);
            } else
                TT_ASSERT(false && "Unsupported conversion.");
        break;
        case tests::utils::TensorLayoutType::TILED_NFACES:
            if (outL == tests::utils::TensorLayoutType::TILED_SWIZZLED) {
                return convert_to_flat_layout<T>(inp, tile_shape, face_shape, transpose_within_face, transpose_of_faces);
            } else if (outL == tests::utils::TensorLayoutType::LIN_ROW_MAJOR) {
                auto swiz32 =
                    convert_to_flat_layout<T>(inp, tile_shape, face_shape, transpose_within_face, transpose_of_faces);
                return untilize_nchw<T>(swiz32, shape, tile_shape);
            } else {
                TT_ASSERT(false && "Unsupported conversion");
            }
        break;
        default:
            TT_ASSERT(false && "Unsupported conversion");
    }
    return std::vector<T>();
}

template <typename T, template <typename...> typename BufferType>
inline std::vector<T> convert_layout(
    const BufferType<T>& inp,
    tt::stl::Span<const uint32_t> shape,
    tests::utils::TensorLayoutType inL,
    tests::utils::TensorLayoutType outL,
    std::optional<PhysicalSize> tile_shape = std::nullopt,
    std::optional<PhysicalSize> face_shape = std::nullopt,
    const bool transpose_within_face = false,
    const bool transpose_of_faces = false) {
    ZoneScoped;

    TT_ASSERT(shape.size() >= 2, "Shape size {} must be at least rank 2!", shape.size());
    uint32_t H = shape[shape.size() - 2];
    uint32_t W = shape[shape.size() - 1];
    for (int i = 0; i < shape.size() - 2; i++) {
        H *= shape[i];
    }
    return convert_layout<T, BufferType>(
        inp, PhysicalSize{H, W}, inL, outL, tile_shape, face_shape, transpose_within_face, transpose_of_faces);
}
