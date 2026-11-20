// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <limits>
#include <array>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/shape.hpp>
#include "ttnn/device.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::operations::unary::test {

struct SubUnaryParam {
    int32_t scalar1;
    int32_t scalar2;
    uint32_t h;
    uint32_t w;
};

class SubUnaryFixture : public TTNNFixtureWithSuiteDevice<SubUnaryFixture>,
                        public testing::WithParamInterface<SubUnaryParam> {};

TEST_P(SubUnaryFixture, CompareWithTorchReference) {
    auto param = GetParam();
    auto& device = *device_;
    std::array<uint32_t, 2> dimensions = {param.h, param.w};
    ttnn::Shape shape(dimensions);

    constexpr DataType dtype = DataType::INT32;

    // Create zero tensor with zeros
    auto zero_tensor = ttnn::zeros(shape, dtype, ttnn::TILE_LAYOUT, device);

    // create input tensor with fill value
    auto input_tensor = ttnn::fill(zero_tensor, param.scalar1);

    // Run TTNN unary sub operation
    auto ttnn_output = ttnn::sub_sfpu(input_tensor, param.scalar2);

    int32_t expected_scalar_result = param.scalar1 - param.scalar2;

    // Create expected tensor (should be filled with the expected scalar value)
    auto expected_tensor = ttnn::full(shape, expected_scalar_result, dtype, ttnn::TILE_LAYOUT, device);

    // Compare results using allclose
    auto expected_host = ttnn::from_device(expected_tensor);
    auto output_host = ttnn::from_device(ttnn_output);

    TT_FATAL(
        ttnn::allclose<int32_t>(expected_host, output_host), "TTNN unary sub result does not match expected result");
}

INSTANTIATE_TEST_SUITE_P(
    SubUnaryTests,
    SubUnaryFixture,
    ::testing::Values(
        SubUnaryParam{-2147483648, -89, 128, 32},
        SubUnaryParam{2147483647, 5, 128, 32},
        SubUnaryParam{-45, -5, 32, 32},
        SubUnaryParam{-5, 67, 64, 64},
        SubUnaryParam{0, 78, 4, 4},
        SubUnaryParam{17, -96, 64, 64},
        SubUnaryParam{100, 458, 128, 128},
        SubUnaryParam{564, -118, 64, 128},
        SubUnaryParam{0, 16777217, 32, 32},
        SubUnaryParam{2147483647, 2147483647, 32, 32}));

namespace sub_uint32 {

constexpr uint32_t kTwoPow31 = 2147483648u;
constexpr uint32_t kUint32Max = 4294967295u;

// 16777217 is the first integer float cannot represent, so it checks that the scalar is not passed through float.
constexpr std::array<uint32_t, 8> kUint32Scalars = {
    0u, 1u, 89u, 16777217u, kTwoPow31, 3000000000u, kUint32Max - 1u, kUint32Max};

template <typename T>
ttnn::Tensor make_tensor(ttnn::MeshDevice* device, const std::vector<T>& data, DataType dtype) {
    const ttnn::Shape shape({32, 64});
    const TensorSpec spec(
        shape, tt::tt_metal::TensorLayout(dtype, tt::tt_metal::PageConfig(Layout::TILE), MemoryConfig{}));
    return Tensor::from_vector(data, spec, device);
}

// Float scalars that no INT32 value equals: fractional, non-finite, or just outside [-2^31, 2^31).
const std::array<float, 6> kInvalidInt32Scalars = {
    2.5f,
    2147483648.0f,
    -2147483904.0f,
    std::numeric_limits<float>::infinity(),
    -std::numeric_limits<float>::infinity(),
    std::numeric_limits<float>::quiet_NaN()};

// Float scalars that no UINT32 value equals: negative, fractional, non-finite, or at or above 2^32.
const std::array<float, 5> kInvalidUint32Scalars = {
    -1.0f, 0.5f, 4294967296.0f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()};

// Cycles through edge values, including those on either side of 2^31 and 0, where an int32 reading would differ.
std::vector<uint32_t> make_uint32_input(uint32_t scalar) {
    const std::array<uint32_t, 10> edges = {
        0u, 1u, 2147483647u, kTwoPow31, kTwoPow31 + 1u, 3000000000u, kUint32Max - 1u, kUint32Max, ~scalar, scalar};
    std::vector<uint32_t> data(32 * 64);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = edges[i % edges.size()] + static_cast<uint32_t>(i / edges.size());
    }
    return data;
}

}  // namespace sub_uint32

class SubUnaryUint32Fixture : public TTNNFixtureWithSuiteDevice<SubUnaryUint32Fixture>,
                              public testing::WithParamInterface<uint32_t> {
protected:
    void SetUp() override {
        if (device_->arch() == tt::ARCH::QUASAR) {
            GTEST_SKIP() << "UInt32 sub_unary_tile is not supported on Quasar.";
        }
    }
};

TEST_P(SubUnaryUint32Fixture, WrapsModulo2Pow32) {
    const uint32_t scalar = GetParam();
    const auto input = sub_uint32::make_uint32_input(scalar);

    const auto output =
        ttnn::sub_sfpu(sub_uint32::make_tensor(device_, input, DataType::UINT32), scalar).cpu().to_vector<uint32_t>();

    ASSERT_EQ(output.size(), input.size());
    size_t mismatches = 0;
    for (size_t i = 0; i < input.size(); ++i) {
        const uint32_t expected = input[i] - scalar;
        if (output[i] != expected) {
            if (mismatches < 8) {
                ADD_FAILURE() << "sub_sfpu(" << input[i] << ", " << scalar << ") = " << output[i] << ", expected "
                              << expected << " at index " << i;
            }
            ++mismatches;
        }
    }
    EXPECT_EQ(mismatches, 0u);
}

// sub_sfpu(scalar, tensor) runs RSUB.
TEST_P(SubUnaryUint32Fixture, ScalarFirstWrapsModulo2Pow32) {
    const uint32_t scalar = GetParam();
    const auto input = sub_uint32::make_uint32_input(scalar);

    const auto output =
        ttnn::sub_sfpu(scalar, sub_uint32::make_tensor(device_, input, DataType::UINT32)).cpu().to_vector<uint32_t>();

    ASSERT_EQ(output.size(), input.size());
    size_t mismatches = 0;
    for (size_t i = 0; i < input.size(); ++i) {
        const uint32_t expected = scalar - input[i];
        if (output[i] != expected) {
            if (mismatches < 8) {
                ADD_FAILURE() << "sub_sfpu(" << scalar << ", " << input[i] << ") = " << output[i] << ", expected "
                              << expected << " at index " << i;
            }
            ++mismatches;
        }
    }
    EXPECT_EQ(mismatches, 0u);
}

INSTANTIATE_TEST_SUITE_P(
    SubUnaryUint32Tests,
    SubUnaryUint32Fixture,
    ::testing::ValuesIn(sub_uint32::kUint32Scalars),
    [](const testing::TestParamInfo<uint32_t>& info) { return std::to_string(info.param); });

class SubUnaryScalarRangeFixture : public TTNNFixtureWithSuiteDevice<SubUnaryScalarRangeFixture> {
protected:
    void SetUp() override {
        if (device_->arch() == tt::ARCH::QUASAR) {
            GTEST_SKIP() << "Integer sub_unary_tile is not supported on Quasar.";
        }
    }
};

// sub_sfpu(tensor, scalar) runs SUB_UNARY_SFPU and sub_sfpu(scalar, tensor) runs RSUB; both check the scalar.
TEST_F(SubUnaryScalarRangeFixture, RejectsScalarsOutsideInputDtypeRange) {
    using ::testing::HasSubstr;
    using ::testing::ThrowsMessage;
    constexpr const char* int32_reason = "on INT32 data requires an integer scalar in [-2147483648, 2147483647], got";
    constexpr const char* uint32_reason = "on UINT32 data requires an integer scalar in [0, 4294967295], got";
    const auto int32_tensor = sub_uint32::make_tensor(device_, std::vector<int32_t>(32 * 64, 7), DataType::INT32);
    const auto uint32_tensor = sub_uint32::make_tensor(device_, std::vector<uint32_t>(32 * 64, 7u), DataType::UINT32);

    for (const float scalar : sub_uint32::kInvalidInt32Scalars) {
        EXPECT_THAT(
            [&] { ttnn::sub_sfpu(int32_tensor, scalar); }, ThrowsMessage<std::runtime_error>(HasSubstr(int32_reason)))
            << "INT32, scalar " << scalar;
        EXPECT_THAT(
            [&] { ttnn::sub_sfpu(scalar, int32_tensor); }, ThrowsMessage<std::runtime_error>(HasSubstr(int32_reason)))
            << "INT32, scalar first, scalar " << scalar;
    }
    for (const float scalar : sub_uint32::kInvalidUint32Scalars) {
        EXPECT_THAT(
            [&] { ttnn::sub_sfpu(uint32_tensor, scalar); }, ThrowsMessage<std::runtime_error>(HasSubstr(uint32_reason)))
            << "UINT32, scalar " << scalar;
        EXPECT_THAT(
            [&] { ttnn::sub_sfpu(scalar, uint32_tensor); }, ThrowsMessage<std::runtime_error>(HasSubstr(uint32_reason)))
            << "UINT32, scalar first, scalar " << scalar;
    }

    // Integer scalars outside the input dtype's range.
    EXPECT_THAT(
        [&] { ttnn::sub_sfpu(int32_tensor, 2147483648u); }, ThrowsMessage<std::runtime_error>(HasSubstr(int32_reason)));
    EXPECT_THAT(
        [&] { ttnn::sub_sfpu(2147483648u, int32_tensor); }, ThrowsMessage<std::runtime_error>(HasSubstr(int32_reason)));
    EXPECT_THAT(
        [&] { ttnn::sub_sfpu(uint32_tensor, -1); }, ThrowsMessage<std::runtime_error>(HasSubstr(uint32_reason)));
    EXPECT_THAT(
        [&] { ttnn::sub_sfpu(-1, uint32_tensor); }, ThrowsMessage<std::runtime_error>(HasSubstr(uint32_reason)));

    // The range ends are accepted.
    EXPECT_NO_THROW(ttnn::sub_sfpu(int32_tensor, -2147483648.0f));
    EXPECT_NO_THROW(ttnn::sub_sfpu(-2147483648.0f, int32_tensor));
    EXPECT_NO_THROW(ttnn::sub_sfpu(uint32_tensor, 4294967040.0f));
    EXPECT_NO_THROW(ttnn::sub_sfpu(uint32_tensor, 4294967295u));
    EXPECT_NO_THROW(ttnn::sub_sfpu(int32_tensor, -2147483647 - 1));
    EXPECT_NO_THROW(ttnn::sub_sfpu(4294967040.0f, uint32_tensor));
}

}  // namespace ttnn::operations::unary::test
