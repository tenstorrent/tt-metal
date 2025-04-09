// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <boost/core/span.hpp>
#include <tracy/Tracy.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <cstddef>
#include <type_traits>
#include <ostream>

#include "assert.hpp"
#include "constants.hpp"
#include <tt_stl/span.hpp>

std::ostream& operator<<(std::ostream& os, TensorLayoutType layout) {
    switch (layout) {
        case TensorLayoutType::LIN_ROW_MAJOR: os << "LIN_ROW_MAJOR"; break;
        case TensorLayoutType::TILED_SWIZZLED: os << "TILED_SWIZZLED"; break;
        case TensorLayoutType::TILED_NFACES: os << "TILED_NFACES"; break;
    }
    return os;
}

TensAddr::TensAddr(const std::vector<std::uint32_t>& shape) : sh(shape) {}

std::uint32_t TensAddr::numel() const {
    std::uint32_t prod = 1;
    for (int j = 0; j < sh.size(); j++) {
        prod *= sh[j];
    }
    return prod;
}

int TensAddr::offs(int n, int c, int h, int w) {
    TT_ASSERT(
        std::uint32_t(n) < sh[0] && std::uint32_t(c) < sh[1] && std::uint32_t(h) < sh[2] && std::uint32_t(w) < sh[3]);
    return w + sh[3] * h + sh[2] * sh[3] * c + sh[1] * sh[2] * sh[3] * n;
}

std::uint32_t round_up_to_mul16(std::uint32_t val) { return ((val & 15) == 0) ? val : (val | 15) + 1; }

std::uint32_t round_up_to_mul32(std::uint32_t val) { return ((val & 31) == 0) ? val : (val | 31) + 1; }

std::uint32_t round_up_to_tile(int val, int tile_val) { return (val + tile_val - 1) & ~(tile_val - 1); }

// Converts a linear non-zero-padded row-major tensor to 32-swizzled tilized row-major tensor
template <typename T>
std::vector<T> convert_layout_row_major_to_tile_swizzled(
    tt::stl::Span<const T> in_row_major, const PhysicalSize& shape, std::optional<PhysicalSize> tile_shape) {
    ZoneScoped;
    std::vector<T> tilized_result;
    if (in_row_major.size() == 0) {
        return tilized_result;
    }

    auto tile_H = tile_shape.has_value() ? tile_shape.value()[0] : tt::constants::TILE_HEIGHT;
    auto tile_W = tile_shape.has_value() ? tile_shape.value()[1] : tt::constants::TILE_WIDTH;

    uint32_t H = shape[0];
    uint32_t W = shape[1];
    uint32_t B = in_row_major.size() / (H * W);

    TT_FATAL(in_row_major.size() > 0 and H > 0 and W > 0, "None of the input size, H, nor W can be 0");
    TT_FATAL((in_row_major.size() % (H * W)) == 0, "Input size must be divisible by H and W");
    TT_FATAL((H % tile_H == 0) and (W % tile_W == 0), "H and W must be divisible by {} and {}", tile_H, tile_W);

    tilized_result.resize(in_row_major.size());
    uint64_t out_index = 0;
    for (auto b = 0; b < B; b++) {
        for (auto hs = 0; hs < H; hs += tile_H) {
            for (auto ws = 0; ws < W; ws += tile_W) {
                for (auto ht = 0; ht < tile_H; ht++) {
                    size_t src_idx = b * H * W + (hs + ht) * W + ws;
                    size_t dst_idx = b * H * W + hs * W + (ws * tile_H) + (ht * tile_W);
                    std::memcpy(&tilized_result[dst_idx], &in_row_major[src_idx], tile_W * sizeof(T));
                }
            }
        }
    }

    return tilized_result;
}

// Converts a 32-swizzled tilized row-major tensor to a linear 32-zero-padded row-major tensor
template <typename T>
std::vector<T> convert_layout_tile_swizzled_to_row_major(
    tt::stl::Span<const T> in_tile_swizzled, const PhysicalSize& shape, std::optional<PhysicalSize> tile_shape) {
    ZoneScoped;
    std::vector<T> result;
    if (in_tile_swizzled.size() == 0) {
        return result;
    }

    auto tile_H = tile_shape.has_value() ? tile_shape.value()[0] : tt::constants::TILE_HEIGHT;
    auto tile_W = tile_shape.has_value() ? tile_shape.value()[1] : tt::constants::TILE_WIDTH;

    // Untilize into row major
    uint32_t H = shape[0];
    uint32_t W = shape[1];
    uint32_t B = in_tile_swizzled.size() / (H * W);

    TT_FATAL(in_tile_swizzled.size() > 0 and H > 0 and W > 0, "None of the input size, H, nor W can be 0");
    TT_FATAL((in_tile_swizzled.size() % (H * W)) == 0, "Input size must be divisible by H and W");
    TT_FATAL((H % tile_H == 0) and (W % tile_W == 0), "H and W must be divisible by {} and {}", tile_H, tile_W);

    result.resize(in_tile_swizzled.size());
    uint64_t linear = 0;
    for (auto b = 0; b < B; b++) {
        for (auto hs = 0; hs < H; hs += tile_H) {
            for (auto ws = 0; ws < W; ws += tile_W) {
                for (auto ht = 0; ht < tile_H; ht++) {
                    // Note: the only difference with tilize_row_major - switched src and dst indices
                    size_t src_idx = b * H * W + hs * W + (ws * tile_H) + (ht * tile_W);
                    size_t dst_idx = b * H * W + (hs + ht) * W + ws;
                    std::memcpy(&result[dst_idx], &in_tile_swizzled[src_idx], tile_W * sizeof(T));
                }
            }
        }
    }
    return result;
}

template <class T>
std::vector<T> convert_layout_tile_swizzled_to_tile_nfaces(
    tt::stl::Span<const T> in_tile_swizzled,
    std::optional<PhysicalSize> tile_shape,
    std::optional<PhysicalSize> face_shape,
    const bool transpose_face,
    const bool transpose_face_order) {
    ZoneScoped;
    std::vector<T> result;
    if (in_tile_swizzled.size() == 0) {
        return result;
    }

    result.reserve(in_tile_swizzled.size());
    auto tile_H = tile_shape.has_value() ? tile_shape.value()[0] : tt::constants::TILE_HEIGHT;
    auto tile_W = tile_shape.has_value() ? tile_shape.value()[1] : tt::constants::TILE_WIDTH;
    auto face_H = face_shape.has_value() ? face_shape.value()[0] : tt::constants::FACE_HEIGHT;
    auto face_W = face_shape.has_value() ? face_shape.value()[1] : tt::constants::FACE_WIDTH;
    auto tile_HW = tile_H * tile_W;
    auto face_HW = face_H * face_W;
    TT_FATAL(in_tile_swizzled.size() % tile_HW == 0, "Input size must be divisible by tile size");
    int num_tiles = in_tile_swizzled.size() / tile_HW;
    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        std::vector<T> top_left, top_right, bottom_left, bottom_right;
        top_left.resize(face_HW);
        top_right.resize(face_HW);
        bottom_left.resize(face_HW);
        bottom_right.resize(face_HW);

        if (transpose_face) {
            int tile_offset = tile_idx * tile_HW;
            for (int col = 0; col < face_H; col++) {
                for (int row = 0; row < face_W; row++) {
                    size_t dst_index = row * face_H + col;
                    top_left[dst_index] = in_tile_swizzled[tile_offset + (col)*tile_W + row];
                    top_right[dst_index] = in_tile_swizzled[tile_offset + (col)*tile_W + row + face_W];
                    bottom_left[dst_index] = in_tile_swizzled[tile_offset + (col + face_H) * tile_W + row];
                    bottom_right[dst_index] = in_tile_swizzled[tile_offset + (col + face_H) * tile_W + row + face_W];
                }
            }
        } else {
            int index = tile_idx * tile_HW;
            for (int row = 0; row < face_H; row++) {
                std::memcpy(&top_left[row * face_W], &in_tile_swizzled[index], face_W * sizeof(T));
                std::memcpy(&top_right[row * face_W], &in_tile_swizzled[index + face_W], face_W * sizeof(T));
                std::memcpy(&bottom_left[row * face_W], &in_tile_swizzled[index + tile_W * face_H], face_W * sizeof(T));
                std::memcpy(
                    &bottom_right[row * face_W],
                    &in_tile_swizzled[index + tile_W * face_H + face_W],
                    face_W * sizeof(T));

                index += tile_W;
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

template <class T>
std::vector<T> convert_layout_tile_nfaces_to_tile_swizzled(
    tt::stl::Span<const T> in_tile_nfaces,
    std::optional<PhysicalSize> tile_shape,
    std::optional<PhysicalSize> face_shape,
    const bool transpose_face,
    const bool /*transpose_face_order*/) {
    ZoneScoped;
    std::vector<T> result(in_tile_nfaces.size());
    if (in_tile_nfaces.size() == 0) {
        return result;
    }
    auto tile_H = tile_shape.has_value() ? tile_shape.value()[0] : tt::constants::TILE_HEIGHT;
    auto tile_W = tile_shape.has_value() ? tile_shape.value()[1] : tt::constants::TILE_WIDTH;
    auto face_H = face_shape.has_value() ? face_shape.value()[0] : tt::constants::FACE_HEIGHT;
    auto face_W = face_shape.has_value() ? face_shape.value()[1] : tt::constants::FACE_WIDTH;
    auto tile_HW = tile_H * tile_W;
    auto face_HW = face_H * face_W;
    auto num_faces_col = tile_W / face_W;
    auto num_faces_row = tile_H / face_H;
    TT_FATAL(in_tile_nfaces.size() % tile_HW == 0, "Input size must be divisible by tile size");
    int num_tiles = in_tile_nfaces.size() / tile_HW;
    size_t dest_idx = 0;
    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        int tile_start = tile_idx * tile_HW;

        if (transpose_face) {
            if (num_faces_row >= 1 && num_faces_col <= 1) {  // e.g. 32x16 case
                for (int face_y = 0; face_y < num_faces_row; face_y++) {
                    const int start = tile_start + face_y * (face_H * tile_W);
                    for (int col = 0; col < face_W; col++) {
                        for (int row = 0; row < face_H; row++) {
                            result[dest_idx++] = in_tile_nfaces[start + col + row * face_W];
                        }
                    }
                }
            } else if (num_faces_row <= 1 && num_faces_col >= 1) {  // e.g. 16x32 case
                for (int col = 0; col < face_W; col++) {
                    const int start = tile_start + col;
                    for (int face_x = 0; face_x < num_faces_col; face_x++) {
                        const int offset = face_x * face_HW;
                        for (int row = 0; row < face_H; row++) {
                            result[dest_idx++] = in_tile_nfaces[start + offset + row * face_W];
                        }
                    }
                }
            } else {
                for (int face_x = 0; face_x < num_faces_col; face_x++) {
                    for (int col = 0; col < face_W; col++) {
                        const int start = tile_start + face_x * face_HW + col;
                        for (int face_y = 0; face_y < num_faces_row; face_y++) {
                            const int offset = face_y * (face_H * tile_W);
                            for (int row = 0; row < face_H; row++) {
                                result[dest_idx++] = in_tile_nfaces[start + offset + row * face_W];
                            }
                        }
                    }
                }
            }
        } else {
            for (int face_y = 0; face_y < num_faces_row; face_y++) {
                for (int face_x = 0; face_x < num_faces_col; face_x++) {
                    size_t src_face_start = tile_start + face_y * (face_H * tile_W) + face_x * face_HW;
                    size_t dst_face_start = tile_start + face_y * (face_H * tile_W) + face_x * face_W;
                    for (int row = 0; row < face_H; row++) {
                        size_t src_idx = src_face_start + row * face_W;
                        size_t dst_idx = dst_face_start + row * tile_W;
                        std::memcpy(&result[dst_idx], &in_tile_nfaces[src_idx], face_W * sizeof(T));
                    }
                }
            }
        }
    }

    return result;
}

// TODO: implement transpose_within_face and transpose_of_faces
template <typename T>
std::vector<T> convert_layout_row_major_to_tile_nfaces(
    tt::stl::Span<const T> in_row_major,
    const PhysicalSize& shape,
    std::optional<PhysicalSize> tile_shape,
    std::optional<PhysicalSize> face_shape,
    const bool transpose_face,
    const bool transpose_face_order) {
    ZoneScoped;

    uint32_t H = shape[0];
    uint32_t W = shape[1];
    uint32_t batch_size = H * W;
    uint32_t B = in_row_major.size() / batch_size;  // Number of batches

    std::vector<T> tilized_input;
    tilized_input.reserve(in_row_major.size());

    auto tile_H = tile_shape.has_value() ? tile_shape.value()[0] : tt::constants::TILE_HEIGHT;
    auto tile_W = tile_shape.has_value() ? tile_shape.value()[1] : tt::constants::TILE_WIDTH;
    auto face_H = face_shape.has_value() ? face_shape.value()[0] : tt::constants::FACE_HEIGHT;
    auto face_W = face_shape.has_value() ? face_shape.value()[1] : tt::constants::FACE_WIDTH;

    uint32_t row_tiles = H / tile_H;
    uint32_t col_tiles = W / tile_W;
    uint32_t row_of_tiles_num_elements = tile_H * W;

    TT_FATAL(in_row_major.size() > 0 and H > 0 and W > 0, "None of the input size, H, nor W can be 0");
    TT_FATAL((in_row_major.size() % (H * W)) == 0, "Input size must be divisible by H and W");
    TT_FATAL((H % tile_H == 0) and (W % tile_W == 0), "H and W must be divisible by {} and {}", tile_H, tile_W);

    auto write_face = [&](uint32_t face_idx, uint32_t face_height, uint32_t face_width, uint32_t stride) {
        size_t offset = tilized_input.size();
        tilized_input.resize(offset + face_height * face_width);
        T* dst = tilized_input.data() + offset;
        const T* src = in_row_major.data() + face_idx;
        for (uint32_t i = 0; i < face_height; i++) {
            std::memcpy(dst, src, face_width * sizeof(T));
            dst += face_width;
            src += stride;
        }
    };

    uint32_t batch_start = 0;
    for (size_t b = 0; b < B; b++) {
        uint32_t tile_start = batch_start;
        for (uint32_t row_tile = 0; row_tile < row_tiles; row_tile++) {
            uint32_t row_tile_start = tile_start;
            for (uint32_t col_tile = 0; col_tile < col_tiles; col_tile++) {
                uint32_t face0_id = row_tile_start;
                uint32_t face1_id = face0_id + face_W;
                uint32_t face2_id = face0_id + W * face_H;
                uint32_t face3_id = face2_id + face_W;

                write_face(face0_id, face_H, face_W, W);
                write_face(face1_id, face_H, face_W, W);
                write_face(face2_id, face_H, face_W, W);
                write_face(face3_id, face_H, face_W, W);
                row_tile_start += tile_W;
            }
            tile_start += row_of_tiles_num_elements;
        }
        batch_start += batch_size;
    }

    return tilized_input;
}

// TODO: implement transpose_within_face and transpose_of_faces
template <typename T>
std::vector<T> convert_layout_tile_nfaces_to_row_major(
    tt::stl::Span<const T> in_nfaces,
    const PhysicalSize& shape,
    std::optional<PhysicalSize> tile_shape,
    std::optional<PhysicalSize> face_shape,
    const bool transpose_face,
    const bool transpose_face_order) {
    ZoneScoped;

    uint32_t H = shape[0];
    uint32_t W = shape[1];
    uint32_t batch_size = H * W;
    uint32_t B = in_nfaces.size() / batch_size;

    std::vector<T> untilized_input(in_nfaces.size());

    auto tile_H = tile_shape.has_value() ? tile_shape.value()[0] : tt::constants::TILE_HEIGHT;
    auto tile_W = tile_shape.has_value() ? tile_shape.value()[1] : tt::constants::TILE_WIDTH;
    auto face_H = face_shape.has_value() ? face_shape.value()[0] : tt::constants::FACE_HEIGHT;
    auto face_W = face_shape.has_value() ? face_shape.value()[1] : tt::constants::FACE_WIDTH;
    uint32_t row_tiles = H / tile_H;
    uint32_t col_tiles = W / tile_W;
    uint32_t row_of_tiles_num_elements = tile_H * W;

    TT_FATAL(in_nfaces.size() > 0 and H > 0 and W > 0, "None of the input size, H, nor W can be 0");
    TT_FATAL((in_nfaces.size() % (H * W)) == 0, "Input size must be divisible by H and W");
    TT_FATAL((H % tile_H == 0) and (W % tile_W == 0), "H and W must be divisible by {} and {}", tile_H, tile_W);

    const auto untilize_row_of_tiles =
        [&](std::vector<T>& out_data, tt::stl::Span<const T> in_data, uint32_t row_tile_start) {
            uint32_t face_stride = face_H * face_W;

            for (uint32_t j = 0; j < W / tile_W; j++) {
                for (uint32_t face_h_in_tile = 0; face_h_in_tile < tile_H / face_H; face_h_in_tile++) {
                    for (uint32_t face_w_in_tile = 0; face_w_in_tile < tile_W / face_W; face_w_in_tile++) {
                        for (uint32_t h = 0; h < face_H; h++) {
                            size_t src_idx = row_tile_start + j * tile_H * tile_W + face_h_in_tile * face_H * tile_W +
                                             face_w_in_tile * face_H * face_W + h * face_H;
                            size_t dst_idx = row_tile_start + (face_h_in_tile * face_H + h) * W + j * tile_W +
                                             face_w_in_tile * face_W;
                            std::memcpy(&out_data[dst_idx], &in_data[src_idx], face_W * sizeof(T));
                        }
                    }
                }
            }
        };

    uint32_t batch_start = 0;
    for (size_t i = 0; i < B; i++) {
        uint32_t row_tile_start = batch_start;
        for (uint32_t row_tile = 0; row_tile < row_tiles; row_tile++) {
            untilize_row_of_tiles(untilized_input, in_nfaces, row_tile_start);
            row_tile_start += row_of_tiles_num_elements;
        }
        batch_start += batch_size;
    }

    return untilized_input;
}

template <typename T>
std::vector<T> convert_layout(
    tt::stl::Span<const T> inp,
    const PhysicalSize& shape,
    TensorLayoutType inL,
    TensorLayoutType outL,
    std::optional<PhysicalSize> tile_shape,
    std::optional<PhysicalSize> face_shape,
    const bool transpose_within_face,
    const bool transpose_of_faces) {
    ZoneScoped;
    if (inp.size() == 0) {
        return std::vector<T>();
    }

    switch (inL) {
        case TensorLayoutType::TILED_SWIZZLED:
            if (outL == TensorLayoutType::TILED_NFACES) {
                return convert_layout_tile_swizzled_to_tile_nfaces<T>(
                    inp, tile_shape, face_shape, transpose_within_face, transpose_of_faces);
            } else if (outL == TensorLayoutType::LIN_ROW_MAJOR) {
                return convert_layout_tile_swizzled_to_row_major<T>(inp, shape, tile_shape);
            } else {
                TT_ASSERT(false && "Unsupported conversion.");
            }
            break;
        case TensorLayoutType::LIN_ROW_MAJOR:
            if (outL == TensorLayoutType::TILED_SWIZZLED) {
                return convert_layout_row_major_to_tile_swizzled<T>(inp, shape, tile_shape);
            } else if (outL == TensorLayoutType::TILED_NFACES) {
                return convert_layout_row_major_to_tile_nfaces(
                    inp, shape, tile_shape, face_shape, transpose_within_face, transpose_of_faces);
            } else {
                TT_ASSERT(false && "Unsupported conversion.");
            }
            break;
        case TensorLayoutType::TILED_NFACES:
            if (outL == TensorLayoutType::TILED_SWIZZLED) {
                return convert_layout_tile_nfaces_to_tile_swizzled<T>(
                    inp, tile_shape, face_shape, transpose_within_face, transpose_of_faces);
            } else if (outL == TensorLayoutType::LIN_ROW_MAJOR) {
                return convert_layout_tile_nfaces_to_row_major(
                    inp, shape, tile_shape, face_shape, transpose_within_face, transpose_of_faces);
            } else {
                TT_ASSERT(false && "Unsupported conversion");
            }
            break;
        default: TT_ASSERT(false && "Unsupported conversion");
    }
    return std::vector<T>();
}

template <typename T>
std::vector<T> convert_layout(
    tt::stl::Span<const T> inp,
    tt::stl::Span<const uint32_t> shape,
    TensorLayoutType inL,
    TensorLayoutType outL,
    std::optional<PhysicalSize> tile_shape,
    std::optional<PhysicalSize> face_shape,
    const bool transpose_within_face,
    const bool transpose_of_faces) {
    ZoneScoped;
    TT_ASSERT(shape.size() >= 2, "Shape size {} must be at least rank 2!", shape.size());
    uint32_t H = shape[shape.size() - 2];
    uint32_t W = shape[shape.size() - 1];
    for (int i = 0; i < shape.size() - 2; i++) {
        H *= shape[i];
    }
    return convert_layout(
        inp, PhysicalSize{H, W}, inL, outL, tile_shape, face_shape, transpose_within_face, transpose_of_faces);
}

template <typename T>
std::vector<T> tilize_swizzled(const std::vector<T>& input, uint32_t m, uint32_t n) {
    TT_FATAL(input.size() > 0 and m > 0 and n > 0, "None of the input size, m, nor n can be 0");
    TT_FATAL((input.size() % (m * n)) == 0, "Input size must be divisible by m  and n");

    return convert_layout<T>(
        input, PhysicalSize{m, n}, TensorLayoutType::LIN_ROW_MAJOR, TensorLayoutType::TILED_SWIZZLED);
}

template <typename T>
std::vector<T> untilize_swizzled(const std::vector<T>& input, uint32_t m, uint32_t n) {
    TT_FATAL(input.size() > 0 and m > 0 and n > 0, "None of the input size, m, nor n can be 0");
    TT_FATAL((input.size() % (m * n)) == 0, "Input size must be divisible by m  and n");

    return convert_layout<T>(
        input, PhysicalSize{m, n}, TensorLayoutType::TILED_SWIZZLED, TensorLayoutType::LIN_ROW_MAJOR);
}

template <typename T>
std::vector<T> tilize_nfaces(const std::vector<T>& input, uint32_t m, uint32_t n) {
    TT_FATAL(input.size() > 0 and m > 0 and n > 0, "None of the input size, m, nor n can be 0");
    TT_FATAL((input.size() % (m * n)) == 0, "Input size must be divisible by m  and n");

    return convert_layout<T>(
        input, PhysicalSize{m, n}, TensorLayoutType::LIN_ROW_MAJOR, TensorLayoutType::TILED_NFACES);
}

template <typename T>
std::vector<T> untilize_nfaces(const std::vector<T>& input, uint32_t m, uint32_t n) {
    TT_FATAL(input.size() > 0 and m > 0 and n > 0, "None of the input size, m, nor n can be 0");
    TT_FATAL((input.size() % (m * n)) == 0, "Input size must be divisible by m  and n");

    return convert_layout<T>(
        input, PhysicalSize{m, n}, TensorLayoutType::TILED_NFACES, TensorLayoutType::LIN_ROW_MAJOR);
}

// Explicit instantiations
// clang-format off
template std::vector<float> convert_layout_tile_swizzled_to_tile_nfaces<float>(tt::stl::Span<const float>, std::optional<PhysicalSize>, std::optional<PhysicalSize>, const bool, const bool);
template std::vector<uint16_t> convert_layout_tile_swizzled_to_tile_nfaces<uint16_t>(tt::stl::Span<const uint16_t>, std::optional<PhysicalSize>, std::optional<PhysicalSize>, const bool, const bool);
template std::vector<uint32_t> convert_layout_tile_swizzled_to_tile_nfaces<uint32_t>(tt::stl::Span<const uint32_t>, std::optional<PhysicalSize>, std::optional<PhysicalSize>, const bool, const bool);
template std::vector<bfloat16> convert_layout_tile_swizzled_to_tile_nfaces<bfloat16>(tt::stl::Span<const bfloat16>, std::optional<PhysicalSize>, std::optional<PhysicalSize>, const bool, const bool);

template std::vector<float> convert_layout_tile_nfaces_to_tile_swizzled<float>(tt::stl::Span<const float>, std::optional<PhysicalSize>, std::optional<PhysicalSize>, const bool, const bool);
template std::vector<uint16_t> convert_layout_tile_nfaces_to_tile_swizzled<uint16_t>(tt::stl::Span<const uint16_t>, std::optional<PhysicalSize>, std::optional<PhysicalSize>, const bool, const bool);
template std::vector<uint32_t> convert_layout_tile_nfaces_to_tile_swizzled<uint32_t>(tt::stl::Span<const uint32_t>, std::optional<PhysicalSize>, std::optional<PhysicalSize>, const bool, const bool);
template std::vector<bfloat16> convert_layout_tile_nfaces_to_tile_swizzled<bfloat16>(tt::stl::Span<const bfloat16>, std::optional<PhysicalSize>, std::optional<PhysicalSize>, const bool, const bool);

template std::vector<float> convert_layout<float>(tt::stl::Span<const float>, const PhysicalSize&, TensorLayoutType, TensorLayoutType, std::optional<PhysicalSize>, std::optional<PhysicalSize>, const bool, const bool);
template std::vector<int> convert_layout<int>(tt::stl::Span<const int>, const PhysicalSize&, TensorLayoutType, TensorLayoutType, std::optional<PhysicalSize>, std::optional<PhysicalSize>, const bool, const bool);
template std::vector<uint8_t> convert_layout<uint8_t>(tt::stl::Span<const uint8_t>, const PhysicalSize&, TensorLayoutType, TensorLayoutType, std::optional<PhysicalSize>, std::optional<PhysicalSize>, const bool, const bool);
template std::vector<uint16_t> convert_layout<uint16_t>(tt::stl::Span<const uint16_t>, const PhysicalSize&, TensorLayoutType, TensorLayoutType, std::optional<PhysicalSize>, std::optional<PhysicalSize>, const bool, const bool);
template std::vector<uint32_t> convert_layout<uint32_t>(tt::stl::Span<const uint32_t>, const PhysicalSize&, TensorLayoutType, TensorLayoutType, std::optional<PhysicalSize>, std::optional<PhysicalSize>, const bool, const bool);
template std::vector<bfloat16> convert_layout<bfloat16>(tt::stl::Span<const bfloat16>, const PhysicalSize&, TensorLayoutType, TensorLayoutType, std::optional<PhysicalSize>, std::optional<PhysicalSize>, const bool, const bool);

template std::vector<float> convert_layout<float>(tt::stl::Span<const float>, tt::stl::Span<const uint32_t>, TensorLayoutType, TensorLayoutType, std::optional<PhysicalSize>, std::optional<PhysicalSize>, const bool, const bool);
template std::vector<int> convert_layout<int>(tt::stl::Span<const int>, tt::stl::Span<const uint32_t>, TensorLayoutType, TensorLayoutType, std::optional<PhysicalSize>, std::optional<PhysicalSize>, const bool, const bool);
template std::vector<uint8_t> convert_layout<uint8_t>(tt::stl::Span<const uint8_t>, tt::stl::Span<const uint32_t>, TensorLayoutType, TensorLayoutType, std::optional<PhysicalSize>, std::optional<PhysicalSize>, const bool, const bool);
template std::vector<uint16_t> convert_layout<uint16_t>(tt::stl::Span<const uint16_t>, tt::stl::Span<const uint32_t>, TensorLayoutType, TensorLayoutType, std::optional<PhysicalSize>, std::optional<PhysicalSize>, const bool, const bool);
template std::vector<uint32_t> convert_layout<uint32_t>(tt::stl::Span<const uint32_t>, tt::stl::Span<const uint32_t>, TensorLayoutType, TensorLayoutType, std::optional<PhysicalSize>, std::optional<PhysicalSize>, const bool, const bool);
template std::vector<bfloat16> convert_layout<bfloat16>(tt::stl::Span<const bfloat16>, tt::stl::Span<const uint32_t>, TensorLayoutType, TensorLayoutType, std::optional<PhysicalSize>, std::optional<PhysicalSize>, const bool, const bool);

template std::vector<uint16_t> tilize_swizzled<uint16_t>(const std::vector<uint16_t>& input, uint32_t m, uint32_t n);
template std::vector<uint32_t> tilize_swizzled<uint32_t>(const std::vector<uint32_t>& input, uint32_t m, uint32_t n);
template std::vector<bfloat16> tilize_swizzled<bfloat16>(const std::vector<bfloat16>& input, uint32_t m, uint32_t n);
template std::vector<float> tilize_swizzled<float>(const std::vector<float>& input, uint32_t m, uint32_t n);

template std::vector<uint16_t> untilize_swizzled<uint16_t>(const std::vector<uint16_t>& input, uint32_t m, uint32_t n);
template std::vector<uint32_t> untilize_swizzled<uint32_t>(const std::vector<uint32_t>& input, uint32_t m, uint32_t n);
template std::vector<bfloat16> untilize_swizzled<bfloat16>(const std::vector<bfloat16>& input, uint32_t m, uint32_t n);
template std::vector<float> untilize_swizzled<float>(const std::vector<float>& input, uint32_t m, uint32_t n);

template std::vector<uint16_t> tilize_nfaces<uint16_t>(const std::vector<uint16_t>& input, uint32_t m, uint32_t n);
template std::vector<uint32_t> tilize_nfaces<uint32_t>(const std::vector<uint32_t>& input, uint32_t m, uint32_t n);
template std::vector<bfloat16> tilize_nfaces<bfloat16>(const std::vector<bfloat16>& input, uint32_t m, uint32_t n);
template std::vector<float> tilize_nfaces<float>(const std::vector<float>& input, uint32_t m, uint32_t n);

template std::vector<uint16_t> untilize_nfaces<uint16_t>(const std::vector<uint16_t>& input, uint32_t m, uint32_t n);
template std::vector<uint32_t> untilize_nfaces<uint32_t>(const std::vector<uint32_t>& input, uint32_t m, uint32_t n);
template std::vector<bfloat16> untilize_nfaces<bfloat16>(const std::vector<bfloat16>& input, uint32_t m, uint32_t n);
template std::vector<float> untilize_nfaces<float>(const std::vector<float>& input, uint32_t m, uint32_t n);

// clang-format on
