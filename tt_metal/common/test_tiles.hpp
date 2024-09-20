// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

//
// This header contains tile conversion functions used in tests on the host.
//

#pragma once

#include <cstdint>
#include <vector>
#include "tt_metal/common/assert.hpp"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"
#include "math.hpp"

enum TensorLayout {
    LIN_ROW_MAJOR = 0, // standard element-wise row-major
    TILED_SWIZZLED = 1, // row-major of tiles, each tile is row-major-swizzled
    TILED_NFACES = 2,  // rowm major of tiles, each tile is N (N = 1, 2, or 4) faces, each face is row-major, faces are swizzled
};

template <class T, template <typename...> typename BufferType>
std::vector<T> convert_to_tile_layout(
    const BufferType<T>& data,
    const std::optional<std::vector<uint32_t>>& tile_shape = std::nullopt,
    const std::optional<const std::vector<uint32_t>>& face_shape = std::nullopt) {
    ZoneScoped;
    std::vector<T> result;
    auto tile_H = tile_shape.has_value() ? tile_shape.value()[0] : 32;
    auto tile_W = tile_shape.has_value() ? tile_shape.value()[1] : 32;
    auto face_H = face_shape.has_value() ? face_shape.value()[0] : 16;
    auto face_W = face_shape.has_value() ? face_shape.value()[1] : 16;
    auto tile_HW = tile_H * tile_W;
    auto face_HW = face_H * face_W;
    TT_ASSERT(data.size() / tile_HW > 0);
    TT_ASSERT(data.size() % tile_HW == 0);
    int num_tiles = data.size() / tile_HW;
    for(int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        std::vector<T> top_left;
        std::vector<T> top_right;
        std::vector<T> bottom_left;
        std::vector<T> bottom_right;

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
        TT_ASSERT(top_left.size() == face_HW);
        TT_ASSERT((top_right.size() == 0) or (top_right.size() == face_HW));
        TT_ASSERT((bottom_left.size() == 0) or (bottom_left.size() == face_HW));
        TT_ASSERT((bottom_right.size() == 0) or (bottom_right.size() == face_HW));

        result.insert(result.end(), top_left.begin(), top_left.end());
        result.insert(result.end(), top_right.begin(), top_right.end());
        result.insert(result.end(), bottom_left.begin(), bottom_left.end());
        result.insert(result.end(), bottom_right.begin(), bottom_right.end());
    }

    return result;
}

template <class T, template <typename...> typename BufferTyp>
std::vector<T> convert_to_flat_layout(
    const BufferTyp<T>& data,
    const std::optional<std::vector<uint32_t>>& tile_shape = std::nullopt,
    const std::optional<const std::vector<uint32_t>>& face_shape = std::nullopt) {
    ZoneScoped;
    std::vector<T> result;
    auto tile_H = tile_shape.has_value() ? tile_shape.value()[0] : 32;
    auto tile_W = tile_shape.has_value() ? tile_shape.value()[1] : 32;
    auto face_H = face_shape.has_value() ? face_shape.value()[0] : 16;
    auto face_W = face_shape.has_value() ? face_shape.value()[1] : 16;
    auto tile_HW = tile_H * tile_W;
    auto face_HW = face_H * face_W;
    auto num_faces_row = tile_W / face_W;
    auto num_faces_col = tile_H / face_H;
    TT_ASSERT(data.size() / tile_HW > 0);
    TT_ASSERT(data.size() % tile_HW == 0);
    int num_tiles = data.size() / tile_HW;
    for(int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        int tile_start = tile_idx * tile_HW;
        for(int face_y = 0; face_y < num_faces_col; face_y++) {
            for(int row = 0; row < face_H; row++) {
                int start = tile_start + face_y * (face_H * tile_W) + row * face_W;
                for(int face_x = 0; face_x < num_faces_row; face_x++) {
                    int offset = face_x * face_HW;
                    for(int col = offset; col < offset + face_W; col++) {
                        result.push_back(data[start + col]);
                    }
                }
            }
        }
    }

    return result;
}

// Converts a 32-swizzled tilized row-major tensor to a linear 32-zero-padded row-major tensor
template <typename T, template <typename...> typename BufferType>
inline std::vector<T> untilize_nchw(const BufferType<T>& in, const std::vector<std::uint32_t>& shape, const std::optional<std::vector<uint32_t>>& tile_shape = std::nullopt) {
    ZoneScoped;
    auto tile_H = tile_shape.has_value() ? tile_shape.value()[0] : 32;
    auto tile_W = tile_shape.has_value() ? tile_shape.value()[1] : 32;

    TT_ASSERT(shape[shape.size() - 2] % tile_H == 0 && shape[shape.size() - 1] % tile_W == 0);

    std::vector<T> result;
    // Untilize into row major
    int H = shape[shape.size() - 2], W = shape[shape.size() - 1];
    auto batch_size = 1;
    for (int i = 0; i < shape.size() - 2; i++) {
        batch_size *= shape[i];
    }
    result.resize(batch_size * H * W);
    uint32_t linear = 0;
    for (auto batch_index = 0; batch_index < batch_size; batch_index++) {
        for (int hs32 = 0; hs32 < H; hs32 += tile_H) {        // iterate over h with stride 32
            for (int ws32 = 0; ws32 < W; ws32 += tile_W) {    // iterate over w with stride 32
                for (int h32 = 0; h32 < tile_H; h32++) {      // hs32 + h32 = h
                    for (int w32 = 0; w32 < tile_W; w32++) {  // ws32 + w32 = w
                        T val = in[linear];
                        auto w = w32 + ws32;
                        auto h = h32 + hs32;
                        auto offs = w + h * W + batch_index * H * W;
                        result[offs] = val;
                        linear++;
                    }
                }
            }
        }
    }

    return result;
}

inline std::uint32_t round_up_to_mul16(std::uint32_t val) { return ((val & 15) == 0) ? val : (val | 15)+1; }

inline std::uint32_t round_up_to_mul32(std::uint32_t val) { return ((val & 31) == 0) ? val : (val | 31)+1; }

inline std::uint32_t round_up_to_tile(int val, int tile_val) { return (val + tile_val - 1) & ~(tile_val - 1); }

// Converts a linear non-zero-padded row-major tensor to zero-padded-32 32-swizzled tilized row-major tensor
template <typename T, template <typename...> typename BufferType>
inline std::vector<T> tilize_nchw(const BufferType<T>& in_rowmajor, const std::vector<std::uint32_t>& shape, const std::optional<std::vector<uint32_t>>& tile_shape = std::nullopt) {
    ZoneScoped;
    int H = shape[shape.size() - 2], W = shape[shape.size() - 1];
    auto batch_size = 1;
    for (int i = 0; i < shape.size() - 2; i++) {
        batch_size *= shape[i];
    }
    int input_volume = batch_size * H * W;
    auto tile_H = tile_shape.has_value() ? tile_shape.value()[0] : 32;
    auto tile_W = tile_shape.has_value() ? tile_shape.value()[1] : 32;
    int OH = round_up_to_tile(H, tile_H);
    int OW = round_up_to_tile(W, tile_W);
    std::vector<T> tilized_result;
    tilized_result.resize(batch_size * OH * OW);
    std::fill(tilized_result.begin(), tilized_result.end(), 0);
    int out_index = 0;
    for (auto batch_index = 0; batch_index < batch_size; batch_index++) {
        for (int hs32 = 0; hs32 < H; hs32 += tile_H) {
            for (int ws32 = 0; ws32 < W; ws32 += tile_W) {
                for (int h32 = 0; h32 < tile_H; h32++) {
                    for (int w32 = 0; w32 < tile_W; w32++) {
                        auto w = w32 + ws32;
                        auto h = h32 + hs32;
                        auto in_offs = w + h * W + batch_index * H * W;
                        auto val = (w >= W || h >= H || in_offs >= input_volume) ? 0 : in_rowmajor[in_offs];
                        int out_w = (out_index % OW);
                        int out_h = (out_index / OW) % OH;
                        TT_ASSERT(w < OW);
                        TT_ASSERT(h < OH);
                        int out_offs = out_w + out_h * OW + batch_index * OH * OW;
                        tilized_result[out_offs] = val;
                        out_index++;
                    }
                }
            }
        }
    }
    TT_ASSERT(tilized_result.size() == batch_size * OH * OW);

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
    const std::vector<uint32_t>& shape,
    TensorLayout inL,
    TensorLayout outL,
    const std::optional<std::vector<uint32_t>>& tile_shape = std::nullopt,
    const std::optional<const std::vector<uint32_t>>& face_shape = std::nullopt) {
    ZoneScoped;
    switch (inL) {
        case TILED_SWIZZLED:
            if (outL == TILED_NFACES) {
                return convert_to_tile_layout<T>(inp, tile_shape, face_shape);
            } else if (outL == LIN_ROW_MAJOR) {
                return untilize_nchw<T>(inp, shape, tile_shape);
            } else
                TT_ASSERT(false && "Unsupported conversion.");
        break;
        case LIN_ROW_MAJOR:
            if (outL == TILED_SWIZZLED) {
                return tilize_nchw<T>(inp, shape, tile_shape);
            } else if (outL == TILED_NFACES) {
                auto swiz32 = convert_layout<T>(inp, shape, inL, TILED_SWIZZLED, tile_shape, face_shape);
                return convert_layout<T>(swiz32, shape, TILED_SWIZZLED, outL, tile_shape, face_shape);
            } else
                TT_ASSERT(false && "Unsupported conversion.");
        break;
        case TILED_NFACES:
            if (outL == TILED_SWIZZLED) {
                return convert_to_flat_layout<T>(inp, tile_shape, face_shape);
            } else if (outL == LIN_ROW_MAJOR) {
                auto swiz32 = convert_layout<T>(inp, shape, inL, TILED_SWIZZLED, tile_shape, face_shape);
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
