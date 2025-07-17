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
#include <tt-metalium/assert.hpp>
#include <tt_stl/span.hpp>

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
using TilizeUntilizeBigBuffersParams = std::tuple<
    PhysicalSize,
    TensorLayoutType,
    TensorLayoutType,
    bool,  // transpose_within_face
    bool   // transpose_of_faces
    >;

void TilizeUntilizeBigBufferImpl(const TilizeUntilizeBigBuffersParams& params) {
    PhysicalSize shape = std::get<0>(params);
    auto from_layout = std::get<1>(params);
    auto to_layout = std::get<2>(params);
    bool transpose_within_face = std::get<3>(params);
    bool transpose_of_faces = std::get<4>(params);

    if (from_layout == to_layout) {
        return;
    }

    size_t n_rows = shape[0];
    size_t n_cols = shape[1];
    size_t n_elements = n_rows * n_cols;

    auto run_for_type = [&](auto type) {
        using Type = decltype(type);
        const auto& data = get_test_data<Type>(n_elements);
        tt::stl::Span<const Type> input(data.data(), n_elements);

        auto converted = convert_layout(
            input,
            shape,
            from_layout,
            to_layout,
            std::nullopt,
            std::nullopt,
            transpose_within_face,
            transpose_of_faces);

        auto converted_back = convert_layout(
            tt::stl::make_const_span(converted),
            shape,
            to_layout,
            from_layout,
            std::nullopt,
            std::nullopt,
            transpose_within_face,
            transpose_of_faces);

        auto converted_back_span = tt::stl::make_const_span(converted_back);
        ASSERT_EQ(input.size(), converted_back.size());
        ASSERT_TRUE(std::equal(input.begin(), input.end(), converted_back_span.begin()));
    };

    run_for_type(uint8_t{});
}

class TilizeUntilizeBigBufferTestsFixture : public ::testing::TestWithParam<TilizeUntilizeBigBuffersParams> {};
class TilizeUntilizeBigBufferShortTestsFixture : public ::testing::TestWithParam<TilizeUntilizeBigBuffersParams> {};

// Disabled by default, since they take a lot of time to run. To still run them, use --gtest_also_run_disabled_tests
TEST_P(TilizeUntilizeBigBufferTestsFixture, DISABLED_TilizeUntilizeBigBuffer) {
    auto params = GetParam();
    TilizeUntilizeBigBufferImpl(params);
}

// The purpose of this test is to be run always to catch any regressions in tilize/untilize logic even though it doesn't
// cover all layout conversions. It tests the most important Row-major -> Tiled_NFaces conversion.
TEST_P(TilizeUntilizeBigBufferShortTestsFixture, TilizeUntilizeBigBuffer) {
    auto params = GetParam();
    TilizeUntilizeBigBufferImpl(params);
}

INSTANTIATE_TEST_SUITE_P(
    TilizeUntilizeBigBufferTests,
    TilizeUntilizeBigBufferTestsFixture,
    ::testing::Combine(
        ::testing::Values(PhysicalSize{1ULL << 16, 1ULL << 17}),  // shape
        ::testing::Values(
            TensorLayoutType::LIN_ROW_MAJOR, TensorLayoutType::TILED_SWIZZLED, TensorLayoutType::TILED_NFACES),
        ::testing::Values(
            TensorLayoutType::LIN_ROW_MAJOR, TensorLayoutType::TILED_SWIZZLED, TensorLayoutType::TILED_NFACES),
        ::testing::Bool(),
        ::testing::Bool()));

INSTANTIATE_TEST_SUITE_P(
    TilizeUntilizeBigBufferTests,
    TilizeUntilizeBigBufferShortTestsFixture,
    ::testing::Combine(
        ::testing::Values(PhysicalSize{1ULL << 16, (1ULL << 16) + 32}),  // shape
        ::testing::Values(TensorLayoutType::LIN_ROW_MAJOR),
        ::testing::Values(TensorLayoutType::TILED_NFACES),
        ::testing::Values(false),
        ::testing::Values(false)));
