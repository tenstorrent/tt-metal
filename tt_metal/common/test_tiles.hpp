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
#include "tt_metal/tt_stl/span.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/common/assert.hpp"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"
#include "math.hpp"

namespace tests::utils {
enum class TensorLayoutType {
    LIN_ROW_MAJOR = 0, // standard element-wise row-major
    TILED_SWIZZLED = 1, // row-major of tiles, each tile is row-major-swizzled
    TILED_NFACES = 2,  // row-major of tiles, each tile is N (N = 1, 2, or 4) faces, each face is row-major, faces are swizzled
};
} // namespace tests::utils

template <class T, template <typename...> typename BufferType>
std::vector<T> convert_to_tile_layout_old(
    const BufferType<T>& data,
    std::optional<tt::stl::Span<const uint32_t>> tile_shape = std::nullopt,
    std::optional<tt::stl::Span<const uint32_t>> face_shape = std::nullopt,
    const std::optional<bool>& transpose_within_face = std::nullopt,
    const std::optional<bool>& transpose_of_faces = std::nullopt) {
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
    bool transpose_face = transpose_within_face.has_value() ? transpose_within_face.value() : false;
    bool transpose_face_order = transpose_of_faces.has_value() ? transpose_of_faces.value() : false;
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

template <class T, template <typename...> typename BufferType>
std::vector<T> convert_to_tile_layout(
    const BufferType<T>& data,
    std::optional<tt::stl::Span<const uint32_t>> tile_shape = std::nullopt,
    std::optional<tt::stl::Span<const uint32_t>> face_shape = std::nullopt) {
    ZoneScoped;

    if (data.size() == 0) {
        return {};
    }

    auto tile_H = tile_shape.has_value() ? tile_shape.value()[0] : tt::constants::TILE_HEIGHT;
    auto tile_W = tile_shape.has_value() ? tile_shape.value()[1] : tt::constants::TILE_WIDTH;
    auto face_H = face_shape.has_value() ? face_shape.value()[0] : tt::constants::FACE_HEIGHT;
    auto face_W = face_shape.has_value() ? face_shape.value()[1] : tt::constants::FACE_WIDTH;
    auto tile_HW = tile_H * tile_W;
    auto face_HW = face_H * face_W;

    TT_ASSERT(data.size() % tile_HW == 0);
    int num_tiles = data.size() / tile_HW;

    // Precompute sizes for the quadrants
    size_t size_top_left = face_H * face_W;
    size_t size_top_right = face_H * (tile_W - face_W);
    size_t size_bottom_left = (tile_H - face_H) * face_W;
    size_t size_bottom_right = (tile_H - face_H) * (tile_W - face_W);

    // Initialize the result vector with the appropriate size
    std::vector<T> result(data.size());

    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        size_t base_offset = tile_idx * tile_HW;

        // Calculate offsets for each quadrant within the current tile
        size_t offset_top_left = base_offset;
        size_t offset_top_right = offset_top_left + size_top_left;
        size_t offset_bottom_left = offset_top_right + size_top_right;
        size_t offset_bottom_right = offset_bottom_left + size_bottom_left;

        // Initialize pointers for writing into the result vector
        size_t ptr_top_left = offset_top_left;
        size_t ptr_top_right = offset_top_right;
        size_t ptr_bottom_left = offset_bottom_left;
        size_t ptr_bottom_right = offset_bottom_right;

        int index = tile_idx * tile_HW;

        for (int row = 0; row < tile_H; row++) {
            for (int col = 0; col < tile_W; col++) {
                if (row < face_H && col < face_W) {
                    result[ptr_top_left++] = data[index];
                } else if (row < face_H && col >= face_W) {
                    result[ptr_top_right++] = data[index];
                } else if (row >= face_H && col < face_W) {
                    result[ptr_bottom_left++] = data[index];
                } else if (row >= face_H && col >= face_W) {
                    result[ptr_bottom_right++] = data[index];
                } else {
                    TT_ASSERT(false);
                }
                index++;
            }
        }

        // Ensure that all data has been written correctly
        TT_ASSERT(ptr_top_left == offset_top_left + size_top_left);
        TT_ASSERT(ptr_top_right == offset_top_right + size_top_right);
        TT_ASSERT(ptr_bottom_left == offset_bottom_left + size_bottom_left);
        TT_ASSERT(ptr_bottom_right == offset_bottom_right + size_bottom_right);
    }

    return result;
}


template <class T, template <typename...> typename BufferTyp>
std::vector<T> convert_to_flat_layout(
    const BufferTyp<T>& data,
    std::optional<tt::stl::Span<const uint32_t>> tile_shape = std::nullopt,
    std::optional<tt::stl::Span<const uint32_t>> face_shape = std::nullopt,
    const std::optional<bool>& transpose_within_face = std::nullopt,
    const std::optional<bool>& transpose_of_faces = std::nullopt) {
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
    bool transpose_face = transpose_within_face.has_value() ? transpose_within_face.value() : false;
    bool transpose_face_order = transpose_of_faces.has_value() ? transpose_of_faces.value() : false;
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
inline std::vector<T> untilize_nchw(const BufferType<T>& in, tt::stl::Span<const uint32_t> shape, std::optional<tt::stl::Span<const uint32_t>> tile_shape = std::nullopt) {
    ZoneScoped;
    auto tile_H = tile_shape.has_value() ? tile_shape.value()[0] : tt::constants::TILE_HEIGHT;
    auto tile_W = tile_shape.has_value() ? tile_shape.value()[1] : tt::constants::TILE_WIDTH;

    std::vector<T> result;
    if(in.size() == 0) {
        return result;
    }

    TT_ASSERT(shape[shape.size() - 2] % tile_H == 0 && shape[shape.size() - 1] % tile_W == 0);

    // Untilize into row major
    int H = shape[shape.size() - 2], W = shape[shape.size() - 1];
    auto batch_size = 1;
    for (int i = 0; i < shape.size() - 2; i++) {
        batch_size *= shape[i];
    }
    result.resize(batch_size * H * W);
    uint32_t linear = 0;
    for (auto batch_index = 0; batch_index < batch_size; batch_index++) {
        for (int hs = 0; hs < H; hs += tile_H) {        // iterate over h with stride 32
            for (int ws = 0; ws < W; ws += tile_W) {    // iterate over w with stride 32
                for (int ht = 0; ht < tile_H; ht++) {      // hs + ht = h
                    for (int wt = 0; wt < tile_W; wt++) {  // ws + wt = w
                        T val = in[linear];
                        auto w = wt + ws;
                        auto h = ht + hs;
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
inline std::vector<T> tilize_nchw_old(const BufferType<T>& in_rowmajor, tt::stl::Span<const uint32_t> shape, std::optional<tt::stl::Span<const uint32_t>> tile_shape = std::nullopt) {
    ZoneScoped;
    std::vector<T> tilized_result;
    if(in_rowmajor.size() == 0) {
        return tilized_result;
    }

    int H = shape[shape.size() - 2], W = shape[shape.size() - 1];
    auto batch_size = 1;
    for (int i = 0; i < shape.size() - 2; i++) {
        batch_size *= shape[i];
    }
    int input_volume = batch_size * H * W;
    auto tile_H = tile_shape.has_value() ? tile_shape.value()[0] : tt::constants::TILE_HEIGHT;
    auto tile_W = tile_shape.has_value() ? tile_shape.value()[1] : tt::constants::TILE_WIDTH;
    int OH = round_up_to_tile(H, tile_H);
    int OW = round_up_to_tile(W, tile_W);
    tilized_result.resize(batch_size * OH * OW);
    std::fill(tilized_result.begin(), tilized_result.end(), 0);
    int out_index = 0;
    for (auto batch_index = 0; batch_index < batch_size; batch_index++) {
        for (int hs = 0; hs < H; hs += tile_H) {
            for (int ws = 0; ws < W; ws += tile_W) {
                for (int ht = 0; ht < tile_H; ht++) {
                    for (int wt = 0; wt < tile_W; wt++) {
                        auto w = wt + ws;
                        auto h = ht + hs;
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

template <typename T, template <typename...> typename BufferType>
inline std::vector<T> tilize_nchw(const BufferType<T>& in_rowmajor, tt::stl::Span<const uint32_t> shape, std::optional<tt::stl::Span<const uint32_t>> tile_shape = std::nullopt) {
    ZoneScoped;
    std::vector<T> tilized_result;
    if (in_rowmajor.size() == 0) {
        return tilized_result;
    }

    const int H = shape[shape.size() - 2];
    const int W = shape[shape.size() - 1];
    int batch_size = 1;
    for (size_t i = 0; i < shape.size() - 2; i++) {
        batch_size *= shape[i];
    }

    const int tile_H = tile_shape.has_value() ? tile_shape.value()[0] : tt::constants::TILE_HEIGHT;
    const int tile_W = tile_shape.has_value() ? tile_shape.value()[1] : tt::constants::TILE_WIDTH;
    const int OH = round_up_to_tile(H, tile_H);
    const int OW = round_up_to_tile(W, tile_W);

    tilized_result.resize(batch_size * OH * OW, 0);
    int out_w = 0;
    int out_h = 0;
    for (int batch_index = 0; batch_index < batch_size; batch_index++) {
        const int batch_offset_in = batch_index * H * W;
        const int batch_offset_out = batch_index * OH * OW;
        for (int hs = 0; hs < H; hs += tile_H) {
            for (int ws = 0; ws < W; ws += tile_W) {
                for (int ht = 0; ht < tile_H; ht++) {
                    const int h = ht + hs;
                    if (h >= H) continue;
                    const int h_W = h * W;
                    for (int wt = 0; wt < tile_W; wt++) {
                        const int w = wt + ws;
                        if (w >= W) continue;
                        //int out_w = (out_index % OW);
                        //int out_h = (out_index / OW) % OH;
                        const int out_offs = out_w + out_h * OW + batch_offset_out;
                        const int in_offs = w + h_W + batch_offset_in;
                        T val = in_rowmajor[in_offs];
                        tilized_result[out_offs] = val;

                        out_w++;
                        if (out_w == OW) {
                            out_w = 0;
                            out_h++;
                            if (out_h == OH) {
                                out_h = 0;
                            }
                        }
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
    tt::stl::Span<const uint32_t> shape,
    tests::utils::TensorLayoutType inL,
    tests::utils::TensorLayoutType outL,
    std::optional<tt::stl::Span<const uint32_t>> tile_shape = std::nullopt,
    std::optional<const tt::stl::Span<const uint32_t>> face_shape = std::nullopt,
    const std::optional<bool>& transpose_within_face = std::nullopt,
    const std::optional<bool>& transpose_of_faces = std::nullopt) {
    ZoneScoped;
    if(inp.size() == 0) {
        return std::vector<T>();
    }

    switch (inL) {
        case tests::utils::TensorLayoutType::TILED_SWIZZLED:
            if (outL == tests::utils::TensorLayoutType::TILED_NFACES) {
                return convert_to_tile_layout<T>(inp, tile_shape, face_shape /*, transpose_within_face, transpose_of_faces*/);
            } else if (outL == tests::utils::TensorLayoutType::LIN_ROW_MAJOR) {
                return untilize_nchw<T>(inp, shape, tile_shape);
            } else
                TT_ASSERT(false && "Unsupported conversion.");
        break;
        case tests::utils::TensorLayoutType::LIN_ROW_MAJOR:
            if (outL == tests::utils::TensorLayoutType::TILED_SWIZZLED) {
                return tilize_nchw<T>(inp, shape, tile_shape);
            } else if (outL == tests::utils::TensorLayoutType::TILED_NFACES) {
                auto swiz32 = convert_layout<T>(inp, shape, inL, tests::utils::TensorLayoutType::TILED_SWIZZLED, tile_shape, face_shape, transpose_within_face, transpose_of_faces);
                return convert_layout<T>(swiz32, shape, tests::utils::TensorLayoutType::TILED_SWIZZLED, outL, tile_shape, face_shape, transpose_within_face, transpose_of_faces);
            } else
                TT_ASSERT(false && "Unsupported conversion.");
        break;
        case tests::utils::TensorLayoutType::TILED_NFACES:
            if (outL == tests::utils::TensorLayoutType::TILED_SWIZZLED) {
                return convert_to_flat_layout<T>(inp, tile_shape, face_shape, transpose_within_face, transpose_of_faces);
            } else if (outL == tests::utils::TensorLayoutType::LIN_ROW_MAJOR) {
                auto swiz32 = convert_layout<T>(inp, shape, inL, tests::utils::TensorLayoutType::TILED_SWIZZLED, tile_shape, face_shape, transpose_within_face, transpose_of_faces);
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
