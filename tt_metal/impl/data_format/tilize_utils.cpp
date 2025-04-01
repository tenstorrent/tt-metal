// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <boost/core/span.hpp>
#include <tracy/Tracy.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <cstddef>
#include <type_traits>

#include "assert.hpp"
#include "constants.hpp"
#include "span.hpp"

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

// Converts a 32-swizzled tilized row-major tensor to a linear 32-zero-padded row-major tensor
template <typename T>
std::vector<T> untilize_nchw(
    tt::stl::Span<const T> in, const PhysicalSize& shape, std::optional<PhysicalSize> tile_shape) {
    ZoneScoped;
    std::vector<T> result;
    if (in.size() == 0) {
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

// Converts a linear non-zero-padded row-major tensor to 32-swizzled tilized row-major tensor
template <typename T>
std::vector<T> tilize_nchw(
    tt::stl::Span<const T> in_rowmajor, const PhysicalSize& shape, std::optional<PhysicalSize> tile_shape) {
    ZoneScoped;
    std::vector<T> tilized_result;
    if (in_rowmajor.size() == 0) {
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

template <class T>
std::vector<T> convert_to_tile_layout(
    tt::stl::Span<const T> data,
    std::optional<PhysicalSize> tile_shape,
    std::optional<PhysicalSize> face_shape,
    const bool transpose_face,
    const bool transpose_face_order) {
    ZoneScoped;
    std::vector<T> result;
    if (data.size() == 0) {
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
    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        std::vector<T> top_left;
        std::vector<T> top_right;
        std::vector<T> bottom_left;
        std::vector<T> bottom_right;

        if (transpose_face) {
            for (int col = 0; col < tile_W; col++) {
                int index = tile_idx * tile_HW + col;
                for (int row = 0; row < tile_H; row++) {
                    if (row < face_H and col < face_W) {
                        top_left.push_back(data[index]);
                    } else if (row < face_H and col >= face_W) {
                        top_right.push_back(data[index]);
                    } else if (row >= face_H and col < face_W) {
                        bottom_left.push_back(data[index]);
                    } else if (row >= face_H and col >= face_W) {
                        bottom_right.push_back(data[index]);
                    } else {
                        TT_ASSERT(false);
                    }
                    index += tile_W;
                }
            }
        } else {
            int index = tile_idx * tile_HW;
            for (int row = 0; row < tile_H; row++) {
                for (int col = 0; col < tile_W; col++) {
                    if (row < face_H and col < face_W) {
                        top_left.push_back(data[index]);
                    } else if (row < face_H and col >= face_W) {
                        top_right.push_back(data[index]);
                    } else if (row >= face_H and col < face_W) {
                        bottom_left.push_back(data[index]);
                    } else if (row >= face_H and col >= face_W) {
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

template <class T>
std::vector<T> convert_to_flat_layout(
    tt::stl::Span<const T> data,
    std::optional<PhysicalSize> tile_shape,
    std::optional<PhysicalSize> face_shape,
    const bool transpose_face,
    const bool transpose_face_order) {
    ZoneScoped;
    std::vector<T> result;
    if (data.size() == 0) {
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
    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        int tile_start = tile_idx * tile_HW;

        if (transpose_face) {
            if (num_faces_row >= 1 && num_faces_col <= 1) {  // 32x16
                for (int face_y = 0; face_y < num_faces_row; face_y++) {
                    int start = tile_start + face_y * (face_H * tile_W);
                    for (int col = 0; col < face_W; col++) {
                        for (int row = 0; row < face_H; row++) {
                            result.push_back(data[start + col + row * face_W]);
                        }
                    }
                }
            } else if (num_faces_row <= 1 && num_faces_col >= 1) {  // 16x32
                for (int col = 0; col < face_W; col++) {
                    int start = tile_start + col;
                    for (int face_x = 0; face_x < num_faces_col; face_x++) {
                        int offset = face_x * face_HW;
                        for (int row = 0; row < face_H; row++) {
                            result.push_back(data[start + offset + row * face_W]);
                        }
                    }
                }
            } else {
                for (int face_x = 0; face_x < num_faces_col; face_x++) {
                    for (int col = 0; col < face_W; col++) {
                        int start = tile_start + face_x * face_HW + col;
                        for (int face_y = 0; face_y < num_faces_row; face_y++) {
                            int offset = face_y * (face_H * tile_W);
                            for (int row = 0; row < face_H; row++) {
                                result.push_back(data[start + offset + row * face_W]);
                            }
                        }
                    }
                }
            }
        } else {
            for (int face_y = 0; face_y < num_faces_row; face_y++) {
                for (int row = 0; row < face_H; row++) {
                    int start = tile_start + face_y * (face_H * tile_W) + row * face_W;
                    for (int face_x = 0; face_x < num_faces_col; face_x++) {
                        int offset = face_x * face_HW;
                        for (int col = offset; col < offset + face_W; col++) {
                            result.push_back(data[start + col]);
                        }
                    }
                }
            }
        }
    }

    return result;
}

template <typename T>
std::vector<T> convert_layout(
    tt::stl::Span<const T> inp,
    const PhysicalSize& shape,
    tests::utils::TensorLayoutType inL,
    tests::utils::TensorLayoutType outL,
    std::optional<PhysicalSize> tile_shape,
    std::optional<PhysicalSize> face_shape,
    const bool transpose_within_face,
    const bool transpose_of_faces) {
    ZoneScoped;
    if (inp.size() == 0) {
        return std::vector<T>();
    }

    switch (inL) {
        case tests::utils::TensorLayoutType::TILED_SWIZZLED:
            if (outL == tests::utils::TensorLayoutType::TILED_NFACES) {
                return convert_to_tile_layout<T>(
                    inp, tile_shape, face_shape, transpose_within_face, transpose_of_faces);
            } else if (outL == tests::utils::TensorLayoutType::LIN_ROW_MAJOR) {
                return untilize_nchw<T>(inp, shape, tile_shape);
            } else {
                TT_ASSERT(false && "Unsupported conversion.");
            }
            break;
        case tests::utils::TensorLayoutType::LIN_ROW_MAJOR:
            if (outL == tests::utils::TensorLayoutType::TILED_SWIZZLED) {
                return tilize_nchw<T>(inp, shape, tile_shape);
            } else if (outL == tests::utils::TensorLayoutType::TILED_NFACES) {
                auto swiz32 = tilize_nchw<T>(inp, shape, tile_shape);
                return convert_to_tile_layout<T>(
                    swiz32, tile_shape, face_shape, transpose_within_face, transpose_of_faces);
            } else {
                TT_ASSERT(false && "Unsupported conversion.");
            }
            break;
        case tests::utils::TensorLayoutType::TILED_NFACES:
            if (outL == tests::utils::TensorLayoutType::TILED_SWIZZLED) {
                return convert_to_flat_layout<T>(
                    inp, tile_shape, face_shape, transpose_within_face, transpose_of_faces);
            } else if (outL == tests::utils::TensorLayoutType::LIN_ROW_MAJOR) {
                auto swiz32 =
                    convert_to_flat_layout<T>(inp, tile_shape, face_shape, transpose_within_face, transpose_of_faces);
                return untilize_nchw<T>(swiz32, shape, tile_shape);
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
    tests::utils::TensorLayoutType inL,
    tests::utils::TensorLayoutType outL,
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
void tilize(std::vector<T>& input, uint32_t m, uint32_t n) {
    TT_FATAL(input.size() > 0 and m > 0 and n > 0, "None of the input size, m, nor n can be 0");
    TT_FATAL((input.size() % (m * n)) == 0, "Input size must be divisible by m  and n");

    std::vector<T> tilized_input;
    tilized_input.reserve(input.size());

    uint32_t block_num_elements = m * n;
    uint32_t num_blocks = input.size() / block_num_elements;

    const auto write_face = [](std::vector<T>& tilized_input,
                               const std::vector<T>& input,
                               uint32_t face_height,
                               uint32_t face_width,
                               uint32_t face_idx,
                               uint32_t n) -> void {
        for (uint32_t i = 0; i < face_height; i++) {
            for (uint32_t j = 0; j < face_width; j++) {
                tilized_input.push_back(input[face_idx + j]);
            }
            face_idx += n;
        }
    };

    if constexpr (std::is_same<T, bfloat16>()) {
        uint32_t TILE_HEIGHT = 32;
        uint32_t TILE_WIDTH = 32;
        uint32_t FACE_HEIGHT = 16;
        uint32_t FACE_WIDTH = 16;
        uint32_t row_tiles = m / TILE_HEIGHT;
        uint32_t col_tiles = n / TILE_WIDTH;
        uint32_t row_of_tiles_num_elements = TILE_HEIGHT * n;
        TT_FATAL((m % TILE_HEIGHT == 0) and (n % TILE_WIDTH == 0), "m and n must be divisible by 32");
        uint32_t block_start = 0;
        for (size_t i = 0; i < num_blocks; i++) {
            uint32_t tile_start = block_start;
            for (uint32_t row_tile = 0; row_tile < row_tiles; row_tile++) {
                uint32_t row_tile_start = tile_start;
                for (uint32_t col_tile = 0; col_tile < col_tiles; col_tile++) {
                    uint32_t face0_id = row_tile_start;
                    uint32_t face1_id = face0_id + FACE_WIDTH;
                    uint32_t face2_id = face0_id + n * FACE_HEIGHT;
                    uint32_t face3_id = face2_id + FACE_WIDTH;

                    write_face(tilized_input, input, FACE_HEIGHT, FACE_WIDTH, face0_id, n);
                    write_face(tilized_input, input, FACE_HEIGHT, FACE_WIDTH, face1_id, n);
                    write_face(tilized_input, input, FACE_HEIGHT, FACE_WIDTH, face2_id, n);
                    write_face(tilized_input, input, FACE_HEIGHT, FACE_WIDTH, face3_id, n);
                    row_tile_start += TILE_WIDTH;
                }
                tile_start += row_of_tiles_num_elements;
            }
            block_start += block_num_elements;
        }
    } else {
        TT_THROW("Invalid type passed into tilize");
    }

    input = std::move(tilized_input);
}

template <typename T>
void untilize(std::vector<T>& input, uint32_t m, uint32_t n) {
    TT_FATAL(input.size() > 0 and m > 0 and n > 0, "None of the input size, m, nor n can be 0");
    TT_FATAL((input.size() % (m * n)) == 0, "Input size must be divisible by m  and n");

    std::vector<T> untilized_input;
    untilized_input.reserve(input.size());

    uint32_t block_num_elements = m * n;
    uint32_t num_blocks = input.size() / block_num_elements;

    const auto untilize_row = [](std::vector<T>& untilized_input,
                                 const std::vector<T>& input,
                                 uint32_t face_height,
                                 uint32_t face_width,
                                 uint32_t tile_idx,
                                 uint32_t TILE_WIDTH,
                                 uint32_t n) -> void {
        uint32_t face_num_elements = face_height * face_width;
        uint32_t face_start = tile_idx;
        for (uint32_t m = 0; m < 2; m++) {
            for (uint32_t i = 0; i < face_height; i++) {
                uint32_t row_start = face_start + i * face_width;
                for (uint32_t j = 0; j < n / TILE_WIDTH; j++) {  // Iterates over all the column tiles
                    // Grab 16 elements from tile j, face 0/2
                    for (uint32_t k = 0; k < face_width; k++) {
                        untilized_input.push_back(input[row_start + k]);
                    }

                    // Grab 16 elements from tile j, face 1/3
                    row_start += face_height * face_width;
                    for (uint32_t k = 0; k < face_width; k++) {
                        untilized_input.push_back(input[row_start + k]);
                    }
                    row_start += face_height * face_width * 3;  // If on face 1, need to get to face 0 of next tile, and
                                                                // if on face 3, need to get to face 2 of next tile
                }
            }
            face_start += face_height * face_width * 2;  // Get to face 2 of current tile
        }
    };

    if constexpr (std::is_same<T, bfloat16>()) {
        uint32_t TILE_HEIGHT = 32;
        uint32_t TILE_WIDTH = 32;
        uint32_t FACE_HEIGHT = 16;
        uint32_t FACE_WIDTH = 16;
        uint32_t row_tiles = m / TILE_HEIGHT;
        uint32_t col_tiles = n / TILE_WIDTH;
        uint32_t row_of_tiles_num_elements = TILE_HEIGHT * n;
        TT_FATAL((m % TILE_HEIGHT == 0) and (n % TILE_WIDTH == 0), "m and n must be divisible by 32");
        uint32_t block_start = 0;
        for (size_t i = 0; i < num_blocks; i++) {
            uint32_t row_tile_start = block_start;
            for (uint32_t row_tile = 0; row_tile < row_tiles; row_tile++) {
                untilize_row(untilized_input, input, FACE_HEIGHT, FACE_WIDTH, row_tile_start, TILE_WIDTH, n);
                row_tile_start += row_of_tiles_num_elements;
            }
            block_start += block_num_elements;
        }
    } else {
        TT_THROW("Invalid type passed into untilize");
    }

    input = std::move(untilized_input);
}

// Explicit instantiations
// clang-format off
template std::vector<float> convert_to_tile_layout<float>(tt::stl::Span<const float>, std::optional<PhysicalSize>, std::optional<PhysicalSize>, const bool, const bool);
template std::vector<uint16_t> convert_to_tile_layout<uint16_t>(tt::stl::Span<const uint16_t>, std::optional<PhysicalSize>, std::optional<PhysicalSize>, const bool, const bool);
template std::vector<uint32_t> convert_to_tile_layout<uint32_t>(tt::stl::Span<const uint32_t>, std::optional<PhysicalSize>, std::optional<PhysicalSize>, const bool, const bool);
template std::vector<bfloat16> convert_to_tile_layout<bfloat16>(tt::stl::Span<const bfloat16>, std::optional<PhysicalSize>, std::optional<PhysicalSize>, const bool, const bool);

template std::vector<float> convert_to_flat_layout<float>(tt::stl::Span<const float>, std::optional<PhysicalSize>, std::optional<PhysicalSize>, const bool, const bool);
template std::vector<uint16_t> convert_to_flat_layout<uint16_t>(tt::stl::Span<const uint16_t>, std::optional<PhysicalSize>, std::optional<PhysicalSize>, const bool, const bool);
template std::vector<uint32_t> convert_to_flat_layout<uint32_t>(tt::stl::Span<const uint32_t>, std::optional<PhysicalSize>, std::optional<PhysicalSize>, const bool, const bool);
template std::vector<bfloat16> convert_to_flat_layout<bfloat16>(tt::stl::Span<const bfloat16>, std::optional<PhysicalSize>, std::optional<PhysicalSize>, const bool, const bool);

template std::vector<float> convert_layout<float>(tt::stl::Span<const float>, const PhysicalSize&, tests::utils::TensorLayoutType, tests::utils::TensorLayoutType, std::optional<PhysicalSize>, std::optional<PhysicalSize>, const bool, const bool);
template std::vector<int> convert_layout<int>(tt::stl::Span<const int>, const PhysicalSize&, tests::utils::TensorLayoutType, tests::utils::TensorLayoutType, std::optional<PhysicalSize>, std::optional<PhysicalSize>, const bool, const bool);
template std::vector<uint8_t> convert_layout<uint8_t>(tt::stl::Span<const uint8_t>, const PhysicalSize&, tests::utils::TensorLayoutType, tests::utils::TensorLayoutType, std::optional<PhysicalSize>, std::optional<PhysicalSize>, const bool, const bool);
template std::vector<uint16_t> convert_layout<uint16_t>(tt::stl::Span<const uint16_t>, const PhysicalSize&, tests::utils::TensorLayoutType, tests::utils::TensorLayoutType, std::optional<PhysicalSize>, std::optional<PhysicalSize>, const bool, const bool);
template std::vector<uint32_t> convert_layout<uint32_t>(tt::stl::Span<const uint32_t>, const PhysicalSize&, tests::utils::TensorLayoutType, tests::utils::TensorLayoutType, std::optional<PhysicalSize>, std::optional<PhysicalSize>, const bool, const bool);
template std::vector<bfloat16> convert_layout<bfloat16>(tt::stl::Span<const bfloat16>, const PhysicalSize&, tests::utils::TensorLayoutType, tests::utils::TensorLayoutType, std::optional<PhysicalSize>, std::optional<PhysicalSize>, const bool, const bool);

template std::vector<float> convert_layout<float>(tt::stl::Span<const float>, tt::stl::Span<const uint32_t>, tests::utils::TensorLayoutType, tests::utils::TensorLayoutType, std::optional<PhysicalSize>, std::optional<PhysicalSize>, const bool, const bool);
template std::vector<int> convert_layout<int>(tt::stl::Span<const int>, tt::stl::Span<const uint32_t>, tests::utils::TensorLayoutType, tests::utils::TensorLayoutType, std::optional<PhysicalSize>, std::optional<PhysicalSize>, const bool, const bool);
template std::vector<uint8_t> convert_layout<uint8_t>(tt::stl::Span<const uint8_t>, tt::stl::Span<const uint32_t>, tests::utils::TensorLayoutType, tests::utils::TensorLayoutType, std::optional<PhysicalSize>, std::optional<PhysicalSize>, const bool, const bool);
template std::vector<uint16_t> convert_layout<uint16_t>(tt::stl::Span<const uint16_t>, tt::stl::Span<const uint32_t>, tests::utils::TensorLayoutType, tests::utils::TensorLayoutType, std::optional<PhysicalSize>, std::optional<PhysicalSize>, const bool, const bool);
template std::vector<uint32_t> convert_layout<uint32_t>(tt::stl::Span<const uint32_t>, tt::stl::Span<const uint32_t>, tests::utils::TensorLayoutType, tests::utils::TensorLayoutType, std::optional<PhysicalSize>, std::optional<PhysicalSize>, const bool, const bool);
template std::vector<bfloat16> convert_layout<bfloat16>(tt::stl::Span<const bfloat16>, tt::stl::Span<const uint32_t>, tests::utils::TensorLayoutType, tests::utils::TensorLayoutType, std::optional<PhysicalSize>, std::optional<PhysicalSize>, const bool, const bool);

template void tilize<uint16_t>(std::vector<uint16_t>& input, uint32_t m, uint32_t n);
template void tilize<uint32_t>(std::vector<uint32_t>& input, uint32_t m, uint32_t n);
template void tilize<bfloat16>(std::vector<bfloat16>& input, uint32_t m, uint32_t n);
template void tilize<float>(std::vector<float>& input, uint32_t m, uint32_t n);

template void untilize<uint16_t>(std::vector<uint16_t>& input, uint32_t m, uint32_t n);
template void untilize<uint32_t>(std::vector<uint32_t>& input, uint32_t m, uint32_t n);
template void untilize<bfloat16>(std::vector<bfloat16>& input, uint32_t m, uint32_t n);
template void untilize<float>(std::vector<float>& input, uint32_t m, uint32_t n);

// clang-format on
