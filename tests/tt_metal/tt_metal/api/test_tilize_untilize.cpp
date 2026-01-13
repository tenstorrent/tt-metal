// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <limits>
#include <vector>
#include <type_traits>
#include <sys/types.h>

#include <gtest/gtest.h>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/span.hpp>

namespace reference {
template <typename T>
std::vector<T> untilize_nchw(
    tt::stl::Span<const T> in, const PhysicalSize& shape, std::optional<PhysicalSize> tile_shape) {
    std::vector<T> result;
    if (in.size() == 0) {
        return result;
    }

    auto tile_H = tile_shape.has_value() ? tile_shape.value()[0] : tt::constants::TILE_HEIGHT;
    auto tile_W = tile_shape.has_value() ? tile_shape.value()[1] : tt::constants::TILE_WIDTH;

    TT_FATAL(shape[0] % tile_H == 0 && shape[1] % tile_W == 0, "Shape dimensions must be divisible by tile dimensions");

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
                    auto offs = w + (h * W);  // + batch_index * H * W;
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
    std::vector<T> tilized_result;
    if (in_rowmajor.size() == 0) {
        return tilized_result;
    }

    auto tile_H = tile_shape.has_value() ? tile_shape.value()[0] : tt::constants::TILE_HEIGHT;
    auto tile_W = tile_shape.has_value() ? tile_shape.value()[1] : tt::constants::TILE_WIDTH;

    TT_FATAL(shape[0] % tile_H == 0 && shape[1] % tile_W == 0, "Shape dimensions must be divisible by tile dimensions");

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
                    auto in_offs = w + (h * W);
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
    TT_FATAL(data.size() % tile_HW == 0, "data size must be divisible by tile_HW");
    int num_tiles = data.size() / tile_HW;
    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        std::vector<T> top_left;
        std::vector<T> top_right;
        std::vector<T> bottom_left;
        std::vector<T> bottom_right;

        if (transpose_face) {
            for (int col = 0; col < tile_W; col++) {
                int index = (tile_idx * tile_HW) + col;
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
                        TT_FATAL(false, "Unexpected face index");
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
                        TT_FATAL(false, "Unexpected face index");
                    }
                    index++;
                }
            }
        }
        TT_FATAL(top_left.size() == face_HW, "top_left size must equal face_HW");
        TT_FATAL((top_right.size() == 0) or (top_right.size() == face_HW), "top_right size must be 0 or face_HW");
        TT_FATAL((bottom_left.size() == 0) or (bottom_left.size() == face_HW), "bottom_left size must be 0 or face_HW");
        TT_FATAL(
            (bottom_right.size() == 0) or (bottom_right.size() == face_HW), "bottom_right size must be 0 or face_HW");

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
    const bool /*transpose_face_order*/) {
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
    TT_FATAL(data.size() % tile_HW == 0, "data size must be divisible by tile_HW");
    int num_tiles = data.size() / tile_HW;
    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        int tile_start = tile_idx * tile_HW;

        if (transpose_face) {
            if (num_faces_row >= 1 && num_faces_col <= 1) {  // 32x16
                for (int face_y = 0; face_y < num_faces_row; face_y++) {
                    int start = tile_start + (face_y * (face_H * tile_W));
                    for (int col = 0; col < face_W; col++) {
                        for (int row = 0; row < face_H; row++) {
                            result.push_back(data[start + col + (row * face_W)]);
                        }
                    }
                }
            } else if (num_faces_row <= 1 && num_faces_col >= 1) {  // 16x32
                for (int col = 0; col < face_W; col++) {
                    int start = tile_start + col;
                    for (int face_x = 0; face_x < num_faces_col; face_x++) {
                        int offset = face_x * face_HW;
                        for (int row = 0; row < face_H; row++) {
                            result.push_back(data[start + offset + (row * face_W)]);
                        }
                    }
                }
            } else {
                for (int face_x = 0; face_x < num_faces_col; face_x++) {
                    for (int col = 0; col < face_W; col++) {
                        int start = tile_start + (face_x * face_HW) + col;
                        for (int face_y = 0; face_y < num_faces_row; face_y++) {
                            int offset = face_y * (face_H * tile_W);
                            for (int row = 0; row < face_H; row++) {
                                result.push_back(data[start + offset + (row * face_W)]);
                            }
                        }
                    }
                }
            }
        } else {
            for (int face_y = 0; face_y < num_faces_row; face_y++) {
                for (int row = 0; row < face_H; row++) {
                    int start = tile_start + (face_y * (face_H * tile_W)) + (row * face_W);
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
    TensorLayoutType inL,
    TensorLayoutType outL,
    std::optional<PhysicalSize> tile_shape,
    std::optional<PhysicalSize> face_shape,
    const bool transpose_within_face,
    const bool transpose_of_faces) {
    if (inp.size() == 0) {
        return std::vector<T>();
    }

    switch (inL) {
        case TensorLayoutType::TILED_SWIZZLED:
            if (outL == TensorLayoutType::TILED_NFACES) {
                return convert_to_tile_layout<T>(
                    inp, tile_shape, face_shape, transpose_within_face, transpose_of_faces);
            } else if (outL == TensorLayoutType::LIN_ROW_MAJOR) {
                return untilize_nchw<T>(inp, shape, tile_shape);
            } else {
                TT_FATAL(false, "Unsupported conversion.");
            }
            break;
        case TensorLayoutType::LIN_ROW_MAJOR:
            if (outL == TensorLayoutType::TILED_SWIZZLED) {
                return tilize_nchw<T>(inp, shape, tile_shape);
            } else if (outL == TensorLayoutType::TILED_NFACES) {
                auto swiz32 = tilize_nchw<T>(inp, shape, tile_shape);
                return convert_to_tile_layout<T>(
                    swiz32, tile_shape, face_shape, transpose_within_face, transpose_of_faces);
            } else {
                TT_FATAL(false, "Unsupported conversion.");
            }
            break;
        case TensorLayoutType::TILED_NFACES:
            if (outL == TensorLayoutType::TILED_SWIZZLED) {
                return convert_to_flat_layout<T>(
                    inp, tile_shape, face_shape, transpose_within_face, transpose_of_faces);
            } else if (outL == TensorLayoutType::LIN_ROW_MAJOR) {
                auto swiz32 =
                    convert_to_flat_layout<T>(inp, tile_shape, face_shape, transpose_within_face, transpose_of_faces);
                return untilize_nchw<T>(swiz32, shape, tile_shape);
            } else {
                TT_FATAL(false, "Unsupported conversion");
            }
            break;
        default: TT_FATAL(false, "Unsupported conversion");
    }
    return std::vector<T>();
}

}  // namespace reference

namespace {
template <typename T>
std::vector<T>& get_test_data(size_t n_elements = 128 * 128) {
    static std::vector<T> data;
    static size_t current_size = 0;

    if (n_elements > current_size) {
        data.resize(n_elements);

        for (size_t i = 0; i < n_elements; ++i) {
            if constexpr (std::is_floating_point_v<T>) {
                data[i] = static_cast<T>(i);
            } else if constexpr (std::is_integral_v<T>) {
                data[i] = static_cast<T>(i % (static_cast<size_t>(std::numeric_limits<T>::max()) + 1));
            } else {
                data[i] = static_cast<T>(static_cast<float>(i));
            }
        }

        current_size = n_elements;
    }

    return data;
}
}  // namespace

// Note: tuple is used for ::testing::Combine
using TilizeUntilizeParams = std::tuple<
    int,
    PhysicalSize,
    TensorLayoutType,
    TensorLayoutType,
    std::optional<PhysicalSize>,
    std::optional<PhysicalSize>,
    bool,
    bool>;

class TilizeUntilizeTestsFixture : public ::testing::TestWithParam<TilizeUntilizeParams> {};

TEST_P(TilizeUntilizeTestsFixture, ConvertLayout) {
    auto params = GetParam();
    int n_batches = std::get<0>(params);
    PhysicalSize shape = std::get<1>(params);
    auto from_layout = std::get<2>(params);
    auto to_layout = std::get<3>(params);
    auto tile_shape = std::get<4>(params);
    auto face_shape = std::get<5>(params);
    bool transpose_within_face = std::get<6>(params);
    bool transpose_of_faces = std::get<7>(params);

    if (from_layout == to_layout) {
        return;
    }

    if ((transpose_within_face || transpose_of_faces) && from_layout == TensorLayoutType::TILED_NFACES) {
        GTEST_SKIP() << "Transpose of faces / within faces is not supported in reference TILED_NFACES->XXX";
    }

    // reference doesn't work well with to_layout == TILED_NFACES when tile_shape == face_shape
    auto tile_H = tile_shape.has_value() ? tile_shape.value()[0] : tt::constants::TILE_HEIGHT;
    auto tile_W = tile_shape.has_value() ? tile_shape.value()[1] : tt::constants::TILE_WIDTH;
    auto face_H = face_shape.has_value() ? face_shape.value()[0] : tt::constants::FACE_HEIGHT;
    auto face_W = face_shape.has_value() ? face_shape.value()[1] : tt::constants::FACE_WIDTH;
    if (to_layout == TensorLayoutType::TILED_NFACES && tile_H <= face_H && tile_W <= face_W) {
        GTEST_SKIP() << "Reference doesn't work well with TILED_NFACES->TILED_SWIZZLED when tile_shape == face_shape";
    }

    uint32_t n_rows = shape[0];
    uint32_t n_cols = shape[1];
    size_t n_elements = n_batches * n_rows * n_cols;

    auto run_for_type = [&](auto type) {
        using Type = decltype(type);
        const auto& data = get_test_data<Type>();
        tt::stl::Span<const Type> input(data.data(), n_elements);

        auto output = convert_layout(
            input, shape, from_layout, to_layout, tile_shape, face_shape, transpose_within_face, transpose_of_faces);

        auto output_ref = reference::convert_layout(
            input, shape, from_layout, to_layout, tile_shape, face_shape, transpose_within_face, transpose_of_faces);

        ASSERT_EQ(output.size(), output_ref.size());
        ASSERT_EQ(output, output_ref);
    };

    // Test all interesting types
    run_for_type(bfloat16{});
    run_for_type(float{});
    run_for_type(int32_t{});
    run_for_type(uint32_t{});
}

TEST_P(TilizeUntilizeTestsFixture, TilizeUntilize) {
    auto params = GetParam();
    int n_batches = std::get<0>(params);
    PhysicalSize shape = std::get<1>(params);
    auto from_layout = std::get<2>(params);
    auto to_layout = std::get<3>(params);
    auto tile_shape = std::get<4>(params);
    auto face_shape = std::get<5>(params);
    bool transpose_within_face = std::get<6>(params);
    bool transpose_of_faces = std::get<7>(params);

    if (from_layout == to_layout) {
        return;
    }

    uint32_t n_rows = shape[0];
    uint32_t n_cols = shape[1];
    size_t n_elements = n_batches * n_rows * n_cols;

    auto run_for_type = [&](auto type) {
        using Type = decltype(type);
        const auto& data = get_test_data<Type>();
        tt::stl::Span<const Type> input(data.data(), n_elements);

        auto converted = convert_layout(
            input, shape, from_layout, to_layout, tile_shape, face_shape, transpose_within_face, transpose_of_faces);

        auto converted_back = convert_layout(
            tt::stl::make_const_span(converted),
            shape,
            to_layout,
            from_layout,
            tile_shape,
            face_shape,
            transpose_within_face,
            transpose_of_faces);

        auto converted_back_span = tt::stl::make_const_span(converted_back);
        ASSERT_EQ(input.size(), converted_back.size());
        ASSERT_TRUE(std::equal(input.begin(), input.end(), converted_back_span.begin()));
    };

    // Test all interesting types
    run_for_type(bfloat16{});
    run_for_type(float{});
    run_for_type(int32_t{});
    run_for_type(uint32_t{});
}

INSTANTIATE_TEST_SUITE_P(
    TilizeUntilizeTests,
    TilizeUntilizeTestsFixture,
    ::testing::Combine(
        ::testing::Values(1),  // n_batches not supported in reference, so only 1 batch
        ::testing::Values(
            PhysicalSize{0, 0}, PhysicalSize{32, 32}, PhysicalSize{128, 64}, PhysicalSize{64, 128}),  // shape
        ::testing::Values(
            TensorLayoutType::LIN_ROW_MAJOR, TensorLayoutType::TILED_SWIZZLED, TensorLayoutType::TILED_NFACES),
        ::testing::Values(
            TensorLayoutType::LIN_ROW_MAJOR, TensorLayoutType::TILED_SWIZZLED, TensorLayoutType::TILED_NFACES),
        ::testing::Values(std::nullopt, PhysicalSize{16, 16}),  // tile_shape.
        ::testing::Values(std::nullopt),                        // face_shape
        ::testing::Bool(),  // transpose_within_face  true doesn't work even in reference
        ::testing::Bool()   // transpose_of_faces     true doesn't work even in reference
        ));

// Test that tilize and then untilize give the same result as the original data

// Note: tuple is used for ::testing::Combine
using ThrowableTilizeUntilizeParams = std::tuple<PhysicalSize, TensorLayoutType, TensorLayoutType, size_t>;

class ThrowableTilizeUntilizeFixture : public ::testing::TestWithParam<ThrowableTilizeUntilizeParams> {};
TEST_P(ThrowableTilizeUntilizeFixture, TilizeUntilize) {
    auto params = GetParam();
    PhysicalSize shape = std::get<0>(params);
    auto from_layout = std::get<1>(params);
    auto to_layout = std::get<2>(params);
    size_t input_size = std::get<3>(params);

    if (from_layout == to_layout) {
        return;
    }

    auto run_for_type = [&](auto type) {
        using Type = decltype(type);
        std::vector<Type> input(input_size);

        EXPECT_ANY_THROW(convert_layout(tt::stl::make_const_span(input), shape, from_layout, to_layout));
    };

    // Test all interesting types
    run_for_type(bfloat16{});
    run_for_type(float{});
    run_for_type(int32_t{});
    run_for_type(uint32_t{});
}

INSTANTIATE_TEST_SUITE_P(
    ThrowableTilizeUntilize,
    ThrowableTilizeUntilizeFixture,
    ::testing::Values(
        // shape
        std::make_tuple(
            PhysicalSize{32, 32},
            TensorLayoutType::LIN_ROW_MAJOR,
            TensorLayoutType::TILED_NFACES,
            12),  // Input too small
        std::make_tuple(
            PhysicalSize{33, 32},  // Bad H
            TensorLayoutType::LIN_ROW_MAJOR,
            TensorLayoutType::TILED_NFACES,
            1024),
        std::make_tuple(
            PhysicalSize{32, 31},  // Bad W
            TensorLayoutType::LIN_ROW_MAJOR,
            TensorLayoutType::TILED_NFACES,
            1024)));

using NonSquareTilesParams =
    std::tuple<PhysicalSize, TensorLayoutType, TensorLayoutType, std::pair<PhysicalSize, PhysicalSize>, bool>;

class NonSquareTilesTestFixture : public ::testing::TestWithParam<NonSquareTilesParams> {};

TEST_P(NonSquareTilesTestFixture, ConvertLayout) {
    auto params = GetParam();
    PhysicalSize shape = std::get<0>(params);
    auto from_layout = std::get<1>(params);
    auto to_layout = std::get<2>(params);
    auto [tile_shape, face_shape] = std::get<3>(params);
    auto transpose = std::get<4>(params);

    if (from_layout == to_layout) {
        return;
    }

    // Not supported by reference
    if (tile_shape[0] < 16 && transpose) {
        return;
    }

    uint32_t n_rows = shape[0];
    uint32_t n_cols = shape[1];
    size_t n_elements = n_rows * n_cols;

    auto run_for_type = [&](auto type) {
        using Type = decltype(type);
        const auto& data = get_test_data<Type>();
        tt::stl::Span<const Type> input(data.data(), n_elements);

        auto output =
            convert_layout(input, shape, from_layout, to_layout, tile_shape, face_shape, transpose, transpose);

        auto output_ref = reference::convert_layout(
            input, shape, from_layout, to_layout, tile_shape, face_shape, transpose, transpose);

        ASSERT_EQ(output.size(), output_ref.size());
        ASSERT_EQ(output, output_ref);
    };

    run_for_type(bfloat16{});
    run_for_type(float{});
    run_for_type(int32_t{});
    run_for_type(uint32_t{});
}

INSTANTIATE_TEST_SUITE_P(
    NonSquareTilesTest,
    NonSquareTilesTestFixture,
    ::testing::Combine(
        ::testing::Values(PhysicalSize{32, 32}, PhysicalSize{128, 64}, PhysicalSize{64, 128}),  // shape
        ::testing::Values(TensorLayoutType::LIN_ROW_MAJOR, TensorLayoutType::TILED_NFACES),
        ::testing::Values(TensorLayoutType::LIN_ROW_MAJOR, TensorLayoutType::TILED_NFACES),

        // Shapes from tt_metal/api/tt-metalium/tile.hpp
        ::testing::Values(
            std::make_pair(PhysicalSize{32, 16}, PhysicalSize{16, 16}),
            std::make_pair(PhysicalSize{16, 32}, PhysicalSize{16, 16}),
            std::make_pair(PhysicalSize{8, 32}, PhysicalSize{8, 16}),
            std::make_pair(PhysicalSize{4, 32}, PhysicalSize{4, 16}),
            std::make_pair(PhysicalSize{2, 32}, PhysicalSize{2, 16}),
            std::make_pair(PhysicalSize{1, 32}, PhysicalSize{1, 16}),
            std::make_pair(PhysicalSize{8, 16}, PhysicalSize{8, 16}),
            std::make_pair(PhysicalSize{4, 16}, PhysicalSize{4, 16}),
            std::make_pair(PhysicalSize{2, 16}, PhysicalSize{2, 16}),
            std::make_pair(PhysicalSize{1, 16}, PhysicalSize{1, 16})),  // tile_shape.
        ::testing::Bool()                                               // transpose
        ));
