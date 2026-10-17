// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <array>
#include <memory>
#include <optional>
#include <span>
#include <iostream>

#include <tt_stl/assert.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/shape.hpp>
#include "ttnn/device.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::operations::unary::test {

struct RelationalUnaryParam {
    int32_t scalar_input;
    int32_t scalar_param;
    uint32_t h;
    uint32_t w;
};

class RelationalUnaryFixture : public TTNNFixtureWithSuiteDevice<RelationalUnaryFixture>,
                               public testing::WithParamInterface<RelationalUnaryParam> {};

static ttnn::Tensor make_int_tensor(ttnn::MeshDevice& device, uint32_t h, uint32_t w, int32_t fill_value) {
    std::array<uint32_t, 2> dimensions = {h, w};
    ttnn::Shape shape(dimensions);
    constexpr DataType dtype = DataType::INT32;
    auto zero_tensor = ttnn::zeros(shape, dtype, ttnn::TILE_LAYOUT, device);
    return ttnn::fill(zero_tensor, fill_value);
}

template <typename UnaryOp>
void test_relational_unary_impl(
    ttnn::MeshDevice& device,
    const RelationalUnaryParam& p,
    const UnaryOp& unary_op,
    int32_t expected_value,
    const char* op_name) {
    auto input_tensor = make_int_tensor(device, p.h, p.w, p.scalar_input);
    auto output = unary_op(input_tensor, p.scalar_param);

    auto expected = make_int_tensor(device, p.h, p.w, expected_value);

    auto expected_host = ttnn::from_device(expected);
    auto output_host = ttnn::from_device(output);

    TT_FATAL(ttnn::allclose<int32_t>(expected_host, output_host), "{} result mismatch", op_name);
}

TEST_P(RelationalUnaryFixture, EqUnaryMatchesExpected) {
    auto p = GetParam();
    test_relational_unary_impl(
        *device_,
        p,
        [](const auto& t, auto s) { return ttnn::eq_unary(t, s); },
        (p.scalar_input == p.scalar_param) ? 1 : 0,
        "eq_unary");
}

TEST_P(RelationalUnaryFixture, NeUnaryMatchesExpected) {
    auto p = GetParam();
    test_relational_unary_impl(
        *device_,
        p,
        [](const auto& t, auto s) { return ttnn::ne_unary(t, s); },
        (p.scalar_input != p.scalar_param) ? 1 : 0,
        "ne_unary");
}

TEST_P(RelationalUnaryFixture, GtUnaryMatchesExpected) {
    auto p = GetParam();
    test_relational_unary_impl(
        *device_,
        p,
        [](const auto& t, auto s) { return ttnn::gt_unary(t, s); },
        (p.scalar_input > p.scalar_param) ? 1 : 0,
        "gt_unary");
}

TEST_P(RelationalUnaryFixture, LtUnaryMatchesExpected) {
    auto p = GetParam();
    test_relational_unary_impl(
        *device_,
        p,
        [](const auto& t, auto s) { return ttnn::lt_unary(t, s); },
        (p.scalar_input < p.scalar_param) ? 1 : 0,
        "lt_unary");
}

TEST_P(RelationalUnaryFixture, GeUnaryMatchesExpected) {
    auto p = GetParam();
    test_relational_unary_impl(
        *device_,
        p,
        [](const auto& t, auto s) { return ttnn::ge_unary(t, s); },
        (p.scalar_input >= p.scalar_param) ? 1 : 0,
        "ge_unary");
}

TEST_P(RelationalUnaryFixture, LeUnaryMatchesExpected) {
    auto p = GetParam();
    test_relational_unary_impl(
        *device_,
        p,
        [](const auto& t, auto s) { return ttnn::le_unary(t, s); },
        (p.scalar_input <= p.scalar_param) ? 1 : 0,
        "le_unary");
}

INSTANTIATE_TEST_SUITE_P(
    RelationalUnaryTests,
    RelationalUnaryFixture,
    ::testing::Values(
        RelationalUnaryParam{2147483647, 2147483647, 128, 32},    // equality at max int
        RelationalUnaryParam{-2147483647, -2147483647, 128, 32},  // equality at min int for negative case
        RelationalUnaryParam{2147483647, 2147483646, 128, 32},    // gt, ne
        RelationalUnaryParam{2147483647, -1, 64, 32},             // gt, ge, ne
        RelationalUnaryParam{-45, -5, 32, 32},                    // lt, le, ne
        RelationalUnaryParam{5, -5, 32, 32},                      // gt, ge, ne
        RelationalUnaryParam{-5, 67, 64, 64},                     // lt, ne
        RelationalUnaryParam{0, 0, 4, 4},                         // eq, ge, le
        RelationalUnaryParam{17, -96, 64, 64},                    // gt, ge, ne
        RelationalUnaryParam{100, 458, 128, 128},                 // lt, ne
        RelationalUnaryParam{564, -118, 64, 128}));               // gt, ne

// A float scalar against an INT32 or UINT32 tensor: the host converts the scalar by value, so an
// integral float compares as that integer. The inputs include the bit pattern of the float scalar
// itself (1088421888 == bits of 7.0f), which a bit-cast of the scalar would wrongly report as
// equal, and a uint32 value >= 2^31, which a conversion through int32_t would corrupt.
struct RelationalUnaryFloatScalarParam {
    DataType dtype;        // INT32 or UINT32
    int64_t scalar_input;  // value held by the tensor
    float scalar_param;    // integral-valued, within the dtype's range
    uint32_t h;
    uint32_t w;
};

class RelationalUnaryFloatScalarFixture : public TTNNFixtureWithSuiteDevice<RelationalUnaryFloatScalarFixture>,
                                          public testing::WithParamInterface<RelationalUnaryFloatScalarParam> {};

static ttnn::Tensor make_int_tensor(ttnn::MeshDevice& device, DataType dtype, uint32_t h, uint32_t w, int64_t value) {
    std::array<uint32_t, 2> dimensions = {h, w};
    ttnn::Shape shape(dimensions);
    auto zero_tensor = ttnn::zeros(shape, dtype, ttnn::TILE_LAYOUT, device);
    if (dtype == DataType::UINT32) {
        return ttnn::fill(zero_tensor, static_cast<uint32_t>(value));
    }
    return ttnn::fill(zero_tensor, static_cast<int32_t>(value));
}

static bool int_tensors_match(DataType dtype, const ttnn::Tensor& expected_host, const ttnn::Tensor& output_host) {
    if (dtype == DataType::UINT32) {
        return ttnn::allclose<uint32_t>(expected_host, output_host);
    }
    return ttnn::allclose<int32_t>(expected_host, output_host);
}

template <typename Op>
void test_relational_float_scalar_impl(
    ttnn::MeshDevice& device,
    const RelationalUnaryFloatScalarParam& p,
    const Op& op,
    int64_t expected_value,
    const char* op_name) {
    auto input_tensor = make_int_tensor(device, p.dtype, p.h, p.w, p.scalar_input);
    auto output = op(input_tensor, p.scalar_param);

    auto expected = make_int_tensor(device, p.dtype, p.h, p.w, expected_value);

    auto expected_host = ttnn::from_device(expected);
    auto output_host = ttnn::from_device(output);

    TT_FATAL(int_tensors_match(p.dtype, expected_host, output_host), "{} result mismatch", op_name);
}

TEST_P(RelationalUnaryFloatScalarFixture, EqUnaryFloatScalar) {
    auto p = GetParam();
    const auto s = static_cast<int64_t>(p.scalar_param);
    test_relational_float_scalar_impl(
        *device_, p, [](const auto& t, float v) { return ttnn::eq_unary(t, v); }, p.scalar_input == s, "eq_unary");
}

TEST_P(RelationalUnaryFloatScalarFixture, NeUnaryFloatScalar) {
    auto p = GetParam();
    const auto s = static_cast<int64_t>(p.scalar_param);
    test_relational_float_scalar_impl(
        *device_, p, [](const auto& t, float v) { return ttnn::ne_unary(t, v); }, p.scalar_input != s, "ne_unary");
}

TEST_P(RelationalUnaryFloatScalarFixture, GtUnaryFloatScalar) {
    auto p = GetParam();
    const auto s = static_cast<int64_t>(p.scalar_param);
    test_relational_float_scalar_impl(
        *device_, p, [](const auto& t, float v) { return ttnn::gt_unary(t, v); }, p.scalar_input > s, "gt_unary");
}

TEST_P(RelationalUnaryFloatScalarFixture, LtUnaryFloatScalar) {
    auto p = GetParam();
    const auto s = static_cast<int64_t>(p.scalar_param);
    test_relational_float_scalar_impl(
        *device_, p, [](const auto& t, float v) { return ttnn::lt_unary(t, v); }, p.scalar_input < s, "lt_unary");
}

TEST_P(RelationalUnaryFloatScalarFixture, GeUnaryFloatScalar) {
    auto p = GetParam();
    const auto s = static_cast<int64_t>(p.scalar_param);
    test_relational_float_scalar_impl(
        *device_, p, [](const auto& t, float v) { return ttnn::ge_unary(t, v); }, p.scalar_input >= s, "ge_unary");
}

TEST_P(RelationalUnaryFloatScalarFixture, LeUnaryFloatScalar) {
    auto p = GetParam();
    const auto s = static_cast<int64_t>(p.scalar_param);
    test_relational_float_scalar_impl(
        *device_, p, [](const auto& t, float v) { return ttnn::le_unary(t, v); }, p.scalar_input <= s, "le_unary");
}

// The public scalar-first overloads route to the tensor-scalar compare with the operands swapped.
TEST_P(RelationalUnaryFloatScalarFixture, ScalarFirstNe) {
    auto p = GetParam();
    const auto s = static_cast<int64_t>(p.scalar_param);
    test_relational_float_scalar_impl(
        *device_, p, [](const auto& t, float v) { return ttnn::ne(v, t); }, s != p.scalar_input, "ne(scalar, tensor)");
}

TEST_P(RelationalUnaryFloatScalarFixture, ScalarFirstGt) {
    auto p = GetParam();
    const auto s = static_cast<int64_t>(p.scalar_param);
    test_relational_float_scalar_impl(
        *device_, p, [](const auto& t, float v) { return ttnn::gt(v, t); }, s > p.scalar_input, "gt(scalar, tensor)");
}

TEST_P(RelationalUnaryFloatScalarFixture, ScalarFirstLt) {
    auto p = GetParam();
    const auto s = static_cast<int64_t>(p.scalar_param);
    test_relational_float_scalar_impl(
        *device_, p, [](const auto& t, float v) { return ttnn::lt(v, t); }, s < p.scalar_input, "lt(scalar, tensor)");
}

INSTANTIATE_TEST_SUITE_P(
    RelationalUnaryFloatScalarTests,
    RelationalUnaryFloatScalarFixture,
    ::testing::Values(
        RelationalUnaryFloatScalarParam{DataType::INT32, 7, 7.0f, 32, 32},           // equal
        RelationalUnaryFloatScalarParam{DataType::INT32, 1088421888, 7.0f, 32, 32},  // input is the bits of 7.0f
        RelationalUnaryFloatScalarParam{DataType::INT32, -5, -5.0f, 32, 32},         // equal, negative
        RelationalUnaryFloatScalarParam{DataType::INT32, 3, 2.0f, 32, 32},           // gt, ge, ne
        RelationalUnaryFloatScalarParam{DataType::UINT32, 5, 7.0f, 32, 32},          // lt, le, ne
        RelationalUnaryFloatScalarParam{DataType::UINT32, 3000000000, 3000000000.0f, 32, 32}));  // equal, >= 2^31

// Host side only: the scalar word the generator emits for the int32 compare kernels is the
// scalar's value, not its float bit pattern, and a uint32 value >= 2^31 keeps its bits.
TEST(RelationalUnaryScalarWord, FloatScalarIsConvertedByValue) {
    using ttnn::operations::unary::utils::get_defines;
    const auto func = [](UnaryOpType op, float scalar, DataType dtype) {
        const std::array<float, 1> params = {scalar};
        return get_defines<float>(op, std::span<const float>(params), "0", "0", dtype).at("SFPU_OP_FUNC_0");
    };
    EXPECT_EQ(func(UnaryOpType::UNARY_NE, 7.0f, DataType::INT32), "unary_ne_tile_int32(0, 7u);");
    EXPECT_EQ(func(UnaryOpType::UNARY_EQ, -5.0f, DataType::INT32), "unary_eq_tile_int32(0, 4294967291u);");
    EXPECT_EQ(func(UnaryOpType::UNARY_GT, 2.0f, DataType::INT32), "unary_gt_tile_int32(0, 2);");
    EXPECT_EQ(func(UnaryOpType::UNARY_LT, 2.0f, DataType::INT32), "unary_lt_tile_int32(0, 2);");
    EXPECT_EQ(func(UnaryOpType::UNARY_GE, 3000000000.0f, DataType::UINT32), "unary_ge_tile_int32(0, 3000000000);");
    EXPECT_EQ(func(UnaryOpType::UNARY_LE, 3000000000.0f, DataType::UINT32), "unary_le_tile_int32(0, 3000000000);");
}

TEST(RelationalUnaryScalarWord, UnrepresentableFloatScalarIsRejected) {
    using ::testing::HasSubstr;
    using ::testing::ThrowsMessage;
    using ttnn::operations::unary::utils::get_defines;
    const auto emit = [](UnaryOpType op, float scalar, DataType dtype) {
        const std::array<float, 1> params = {scalar};
        get_defines<float>(op, std::span<const float>(params), "0", "0", dtype);
    };
    constexpr const char* reason = "cannot represent the scalar";
    EXPECT_THAT(
        [&] { emit(UnaryOpType::UNARY_GE, 2.5f, DataType::INT32); },
        ThrowsMessage<std::runtime_error>(HasSubstr(reason)));
    EXPECT_THAT(
        [&] { emit(UnaryOpType::UNARY_LT, std::numeric_limits<float>::infinity(), DataType::INT32); },
        ThrowsMessage<std::runtime_error>(HasSubstr(reason)));
    EXPECT_THAT(
        [&] { emit(UnaryOpType::UNARY_GT, 3000000000.0f, DataType::INT32); },
        ThrowsMessage<std::runtime_error>(HasSubstr(reason)));
    EXPECT_THAT(
        [&] { emit(UnaryOpType::UNARY_NE, -1.0f, DataType::UINT32); },
        ThrowsMessage<std::runtime_error>(HasSubstr(reason)));
    EXPECT_THAT(
        [&] { emit(UnaryOpType::UNARY_EQ, std::numeric_limits<float>::quiet_NaN(), DataType::UINT32); },
        ThrowsMessage<std::runtime_error>(HasSubstr(reason)));
    // Integral values inside the range are accepted.
    EXPECT_NO_THROW(emit(UnaryOpType::UNARY_LE, -2147483648.0f, DataType::INT32));
    EXPECT_NO_THROW(emit(UnaryOpType::UNARY_LE, 4294967040.0f, DataType::UINT32));
}

// A float scalar the integer dtype cannot hold exactly (fractional, non-finite, out of range) is
// rejected at the host instead of being truncated or converted through undefined behaviour.
struct RelationalUnaryRejectParam {
    DataType dtype;
    float scalar_param;
};

class RelationalUnaryRejectFixture : public TTNNFixtureWithSuiteDevice<RelationalUnaryRejectFixture>,
                                     public testing::WithParamInterface<RelationalUnaryRejectParam> {};

TEST_P(RelationalUnaryRejectFixture, UnrepresentableFloatScalarIsRejected) {
    using ::testing::HasSubstr;
    using ::testing::ThrowsMessage;
    const auto p = GetParam();
    auto t = make_int_tensor(*device_, p.dtype, 32, 32, 1);
    const float s = p.scalar_param;
    constexpr const char* reason = "cannot represent the scalar";

    EXPECT_THAT([&] { ttnn::eq_unary(t, s); }, ThrowsMessage<std::runtime_error>(HasSubstr(reason)));
    EXPECT_THAT([&] { ttnn::ne_unary(t, s); }, ThrowsMessage<std::runtime_error>(HasSubstr(reason)));
    EXPECT_THAT([&] { ttnn::gt_unary(t, s); }, ThrowsMessage<std::runtime_error>(HasSubstr(reason)));
    EXPECT_THAT([&] { ttnn::lt_unary(t, s); }, ThrowsMessage<std::runtime_error>(HasSubstr(reason)));
    EXPECT_THAT([&] { ttnn::ge_unary(t, s); }, ThrowsMessage<std::runtime_error>(HasSubstr(reason)));
    EXPECT_THAT([&] { ttnn::le_unary(t, s); }, ThrowsMessage<std::runtime_error>(HasSubstr(reason)));
    // Scalar-first public overloads.
    EXPECT_THAT([&] { ttnn::ne(s, t); }, ThrowsMessage<std::runtime_error>(HasSubstr(reason)));
    EXPECT_THAT([&] { ttnn::gt(s, t); }, ThrowsMessage<std::runtime_error>(HasSubstr(reason)));
    EXPECT_THAT([&] { ttnn::lt(s, t); }, ThrowsMessage<std::runtime_error>(HasSubstr(reason)));
}

INSTANTIATE_TEST_SUITE_P(
    RelationalUnaryRejectTests,
    RelationalUnaryRejectFixture,
    ::testing::Values(
        RelationalUnaryRejectParam{DataType::INT32, 2.5f},  // fractional
        RelationalUnaryRejectParam{DataType::INT32, std::numeric_limits<float>::infinity()},
        RelationalUnaryRejectParam{DataType::INT32, 3000000000.0f},  // above INT32 range
        RelationalUnaryRejectParam{DataType::UINT32, -1.0f},         // below UINT32 range
        RelationalUnaryRejectParam{DataType::UINT32, std::numeric_limits<float>::quiet_NaN()}));

}  // namespace ttnn::operations::unary::test
