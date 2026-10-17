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
        RelationalUnaryParam{2147483647, 2147483647, 128, 32},           // equality at max int
        RelationalUnaryParam{-2147483647, -2147483647, 128, 32},         // equality at min int for negative case
        RelationalUnaryParam{-2147483647 - 1, -2147483647 - 1, 32, 32},  // equality at INT32_MIN (word 0x80000000)
        RelationalUnaryParam{2147483647, 2147483646, 128, 32},           // gt, ne
        RelationalUnaryParam{2147483647, -1, 64, 32},                    // gt, ge, ne
        RelationalUnaryParam{-45, -5, 32, 32},                           // lt, le, ne
        RelationalUnaryParam{5, -5, 32, 32},                             // gt, ge, ne
        RelationalUnaryParam{-5, 67, 64, 64},                            // lt, ne
        RelationalUnaryParam{0, 0, 4, 4},                                // eq, ge, le
        RelationalUnaryParam{17, -96, 64, 64},                           // gt, ge, ne
        RelationalUnaryParam{100, 458, 128, 128},                        // lt, ne
        RelationalUnaryParam{564, -118, 64, 128}));                      // gt, ne

// A float scalar against an INT32 or UINT32 tensor: the host converts the scalar by value, so an
// integral float compares as that integer. The inputs include the bit pattern of the float scalar
// itself (1088421888 == bits of 7.0f), which a bit-cast of the scalar would wrongly report as equal.
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
        RelationalUnaryFloatScalarParam{DataType::UINT32, 5, 7.0f, 32, 32}));        // lt, le, ne

// A UINT32 scalar at or above 2^31. EQ/NE compare bits, so they accept it and a conversion through
// int32_t would corrupt it. GT/LT/GE/LE are signed int32 compares in the kernel and would read it as
// negative (gt_unary(uint32 5, 3000000000.0f) returned 1), so the host rejects it instead.
struct RelationalUnaryUint32HighScalarParam {
    int64_t scalar_input;  // value held by the tensor
    float scalar_param;    // integral-valued, in [2^31, 2^32)
};

class RelationalUnaryUint32HighScalarFixture
    : public TTNNFixtureWithSuiteDevice<RelationalUnaryUint32HighScalarFixture>,
      public testing::WithParamInterface<RelationalUnaryUint32HighScalarParam> {};

TEST_P(RelationalUnaryUint32HighScalarFixture, EqNeUnaryCompareBits) {
    const auto p = GetParam();
    const auto s = static_cast<int64_t>(p.scalar_param);
    const RelationalUnaryFloatScalarParam q{DataType::UINT32, p.scalar_input, p.scalar_param, 32, 32};
    test_relational_float_scalar_impl(
        *device_, q, [](const auto& t, float v) { return ttnn::eq_unary(t, v); }, p.scalar_input == s, "eq_unary");
    test_relational_float_scalar_impl(
        *device_, q, [](const auto& t, float v) { return ttnn::ne_unary(t, v); }, p.scalar_input != s, "ne_unary");
    test_relational_float_scalar_impl(
        *device_, q, [](const auto& t, float v) { return ttnn::ne(v, t); }, s != p.scalar_input, "ne(scalar, tensor)");
}

TEST_P(RelationalUnaryUint32HighScalarFixture, OrderedCompareIsRejected) {
    using ::testing::HasSubstr;
    using ::testing::ThrowsMessage;
    const auto p = GetParam();
    auto t = make_int_tensor(*device_, DataType::UINT32, 32, 32, p.scalar_input);
    const float s = p.scalar_param;
    const auto s_u32 = static_cast<uint32_t>(p.scalar_param);  // the same word through the uint32_t scalar overload
    constexpr const char* reason = "below 2147483648";

    EXPECT_THAT([&] { ttnn::gt_unary(t, s); }, ThrowsMessage<std::runtime_error>(HasSubstr(reason)));
    EXPECT_THAT([&] { ttnn::lt_unary(t, s); }, ThrowsMessage<std::runtime_error>(HasSubstr(reason)));
    EXPECT_THAT([&] { ttnn::ge_unary(t, s); }, ThrowsMessage<std::runtime_error>(HasSubstr(reason)));
    EXPECT_THAT([&] { ttnn::le_unary(t, s); }, ThrowsMessage<std::runtime_error>(HasSubstr(reason)));
    EXPECT_THAT([&] { ttnn::gt_unary(t, s_u32); }, ThrowsMessage<std::runtime_error>(HasSubstr(reason)));
    EXPECT_THAT([&] { ttnn::le_unary(t, s_u32); }, ThrowsMessage<std::runtime_error>(HasSubstr(reason)));
    // Scalar-first public overloads.
    EXPECT_THAT([&] { ttnn::gt(s, t); }, ThrowsMessage<std::runtime_error>(HasSubstr(reason)));
    EXPECT_THAT([&] { ttnn::lt(s, t); }, ThrowsMessage<std::runtime_error>(HasSubstr(reason)));
    // A negative int32_t scalar is not a UINT32 value: rejected as unrepresentable, not as "below 2^31".
    constexpr const char* unrepresentable = "cannot represent the scalar";
    EXPECT_THAT(
        [&] { ttnn::gt_unary(t, std::int32_t{-1}); }, ThrowsMessage<std::runtime_error>(HasSubstr(unrepresentable)));
    EXPECT_THAT(
        [&] { ttnn::ne_unary(t, std::int32_t{-1}); }, ThrowsMessage<std::runtime_error>(HasSubstr(unrepresentable)));
}

INSTANTIATE_TEST_SUITE_P(
    RelationalUnaryUint32HighScalarTests,
    RelationalUnaryUint32HighScalarFixture,
    ::testing::Values(
        RelationalUnaryUint32HighScalarParam{3000000000, 3000000000.0f},    // equal
        RelationalUnaryUint32HighScalarParam{5, 3000000000.0f},             // lt, le, ne: the signed read says gt, ge
        RelationalUnaryUint32HighScalarParam{2147483648, 2147483648.0f},    // equal at exactly 2^31 (word 0x80000000)
        RelationalUnaryUint32HighScalarParam{4294967040, 4294967040.0f}));  // equal at the largest float32 below 2^32

// EQ/NE on UINT32 through the uint32_t scalar overload: a bit compare over the whole range, including
// the words no float32 holds (0xFFFFFFFF) and exactly 2^31 against 0 in both roles.
struct RelationalUnaryUint32WordParam {
    uint32_t scalar_input;  // value held by the tensor
    uint32_t scalar_param;
};

class RelationalUnaryUint32WordFixture : public TTNNFixtureWithSuiteDevice<RelationalUnaryUint32WordFixture>,
                                         public testing::WithParamInterface<RelationalUnaryUint32WordParam> {};

TEST_P(RelationalUnaryUint32WordFixture, EqNeUnaryCompareBits) {
    const auto p = GetParam();
    const RelationalUnaryFloatScalarParam q{DataType::UINT32, p.scalar_input, 0.0f, 32, 32};
    const uint32_t s = p.scalar_param;
    test_relational_float_scalar_impl(
        *device_, q, [s](const auto& t, float) { return ttnn::eq_unary(t, s); }, p.scalar_input == s, "eq_unary");
    test_relational_float_scalar_impl(
        *device_, q, [s](const auto& t, float) { return ttnn::ne_unary(t, s); }, p.scalar_input != s, "ne_unary");
}

INSTANTIATE_TEST_SUITE_P(
    RelationalUnaryUint32WordTests,
    RelationalUnaryUint32WordFixture,
    ::testing::Values(
        RelationalUnaryUint32WordParam{4294967295u, 4294967295u},  // equal at UINT32_MAX
        RelationalUnaryUint32WordParam{2147483648u, 2147483648u},  // equal at exactly 2^31
        RelationalUnaryUint32WordParam{2147483648u, 0u},           // 2^31 is not 0
        RelationalUnaryUint32WordParam{0u, 2147483648u},           // 0 is not 2^31
        RelationalUnaryUint32WordParam{4294967295u, 2147483647u}));

// Host side only: the scalar word the generator emits for the int32 compare kernels is the
// scalar's value, not its float bit pattern, and for EQ/NE a uint32 value >= 2^31 keeps its bits.
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
    EXPECT_EQ(func(UnaryOpType::UNARY_EQ, 3000000000.0f, DataType::UINT32), "unary_eq_tile_int32(0, 3000000000u);");
    EXPECT_EQ(func(UnaryOpType::UNARY_NE, 4294967040.0f, DataType::UINT32), "unary_ne_tile_int32(0, 4294967040u);");
    EXPECT_EQ(func(UnaryOpType::UNARY_GE, 2147483520.0f, DataType::UINT32), "unary_ge_tile_int32(0, 2147483520);");
}

// Host side only: GT/LT/GE/LE on UINT32 reject a scalar word at or above 2^31 (float or uint32_t
// scalar) because the kernel compares as signed int32; EQ/NE and INT32 are unaffected.
TEST(RelationalUnaryScalarWord, Uint32OrderedCompareRejectsScalarAtOrAbove2To31) {
    using ::testing::HasSubstr;
    using ::testing::ThrowsMessage;
    using ttnn::operations::unary::utils::get_defines;
    const auto emit = [](UnaryOpType op, auto scalar, DataType dtype) {
        using T = decltype(scalar);
        const std::array<T, 1> params = {scalar};
        get_defines<T>(op, std::span<const T>(params), "0", "0", dtype);
    };
    constexpr const char* reason = "below 2147483648";
    EXPECT_THAT(
        [&] { emit(UnaryOpType::UNARY_GT, 2147483648.0f, DataType::UINT32); },
        ThrowsMessage<std::runtime_error>(HasSubstr(reason)));
    EXPECT_THAT(
        [&] { emit(UnaryOpType::UNARY_LT, 3000000000.0f, DataType::UINT32); },
        ThrowsMessage<std::runtime_error>(HasSubstr(reason)));
    EXPECT_THAT(
        [&] { emit(UnaryOpType::UNARY_GE, 3000000000.0f, DataType::UINT32); },
        ThrowsMessage<std::runtime_error>(HasSubstr(reason)));
    EXPECT_THAT(
        [&] { emit(UnaryOpType::UNARY_LE, 4294967040.0f, DataType::UINT32); },
        ThrowsMessage<std::runtime_error>(HasSubstr(reason)));
    EXPECT_THAT(
        [&] { emit(UnaryOpType::UNARY_GT, std::uint32_t{3000000000u}, DataType::UINT32); },
        ThrowsMessage<std::runtime_error>(HasSubstr(reason)));
    // Below 2^31 (2147483520 is the largest float32 below it), EQ/NE at any UINT32 value, and INT32 pass.
    EXPECT_NO_THROW(emit(UnaryOpType::UNARY_GT, 2147483520.0f, DataType::UINT32));
    EXPECT_NO_THROW(emit(UnaryOpType::UNARY_LE, std::uint32_t{2147483647u}, DataType::UINT32));
    EXPECT_NO_THROW(emit(UnaryOpType::UNARY_EQ, 3000000000.0f, DataType::UINT32));
    EXPECT_NO_THROW(emit(UnaryOpType::UNARY_NE, std::uint32_t{4294967295u}, DataType::UINT32));
    EXPECT_NO_THROW(emit(UnaryOpType::UNARY_GT, -5.0f, DataType::INT32));
    EXPECT_NO_THROW(emit(UnaryOpType::UNARY_LT, std::int32_t{-2147483647 - 1}, DataType::INT32));
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
    EXPECT_NO_THROW(emit(UnaryOpType::UNARY_EQ, 4294967040.0f, DataType::UINT32));
}

// Host side only: an integer scalar outside the op dtype's range is rejected like an unrepresentable
// float (the kernel would read a negative int32_t as a large UINT32 and a uint32_t >= 2^31 as negative).
TEST(RelationalUnaryScalarWord, IntegerScalarOutsideDtypeRangeIsRejected) {
    using ::testing::HasSubstr;
    using ::testing::ThrowsMessage;
    using ttnn::operations::unary::utils::get_defines;
    const auto emit = [](UnaryOpType op, auto scalar, DataType dtype) {
        using T = decltype(scalar);
        const std::array<T, 1> params = {scalar};
        get_defines<T>(op, std::span<const T>(params), "0", "0", dtype);
    };
    constexpr const char* reason = "cannot represent the scalar";
    EXPECT_THAT(
        [&] { emit(UnaryOpType::UNARY_GT, std::int32_t{-1}, DataType::UINT32); },
        ThrowsMessage<std::runtime_error>(HasSubstr(reason)));
    EXPECT_THAT(
        [&] { emit(UnaryOpType::UNARY_EQ, std::int32_t{-1}, DataType::UINT32); },
        ThrowsMessage<std::runtime_error>(HasSubstr(reason)));
    EXPECT_THAT(
        [&] { emit(UnaryOpType::UNARY_LT, std::uint32_t{2147483648u}, DataType::INT32); },
        ThrowsMessage<std::runtime_error>(HasSubstr(reason)));
    EXPECT_THAT(
        [&] { emit(UnaryOpType::UNARY_NE, std::uint32_t{3000000000u}, DataType::INT32); },
        ThrowsMessage<std::runtime_error>(HasSubstr(reason)));
    // In range: accepted, and the word is the value.
    EXPECT_NO_THROW(emit(UnaryOpType::UNARY_GT, std::int32_t{0}, DataType::UINT32));
    EXPECT_NO_THROW(emit(UnaryOpType::UNARY_LE, std::uint32_t{2147483647u}, DataType::INT32));
    EXPECT_NO_THROW(emit(UnaryOpType::UNARY_GE, std::int32_t{-5}, DataType::INT32));
    const std::array<std::int32_t, 1> five = {5};
    EXPECT_EQ(
        get_defines<std::int32_t>(
            UnaryOpType::UNARY_GT, std::span<const std::int32_t>(five), "0", "0", DataType::UINT32)
            .at("SFPU_OP_FUNC_0"),
        "unary_gt_tile_int32(0, 5);");
}

// Host side only: an op placed after a TYPECAST is generated for the cast's target dtype. A relational
// op on an INT32 tensor cast to FLOAT32 gets the float compare with the scalar's float bit pattern
// (and a fractional scalar is accepted there); one on a FLOAT32 tensor cast to INT32 gets the int32
// compare with the integer value.
TEST(RelationalUnaryScalarWord, OpsAfterTypecastUseTheCastDtype) {
    using ttnn::operations::unary::utils::get_block_defines;
    const auto typecast = [](DataType from, DataType to) {
        return EltwiseUnaryWithParam(UnaryOpType::TYPECAST, static_cast<float>(from), static_cast<float>(to));
    };
    const auto func_1 = [](const std::vector<EltwiseUnaryWithParam>& chain, DataType input_dtype) {
        return get_block_defines(chain, "0", "0", input_dtype).at("SFPU_OP_CHAIN_0_FUNC_1");
    };
    const std::vector<EltwiseUnaryWithParam> int_to_float_gt = {
        typecast(DataType::INT32, DataType::FLOAT32), EltwiseUnaryWithParam(UnaryOpType::UNARY_GT, 2.0f)};
    EXPECT_EQ(func_1(int_to_float_gt, DataType::INT32), "unary_gt_tile(0, 0x40000000u);");
    const std::vector<EltwiseUnaryWithParam> int_to_float_gt_half = {
        typecast(DataType::INT32, DataType::FLOAT32), EltwiseUnaryWithParam(UnaryOpType::UNARY_GT, 0.5f)};
    EXPECT_EQ(func_1(int_to_float_gt_half, DataType::INT32), "unary_gt_tile(0, 0x3f000000u);");
    const std::vector<EltwiseUnaryWithParam> float_to_int_gt = {
        typecast(DataType::FLOAT32, DataType::INT32), EltwiseUnaryWithParam(UnaryOpType::UNARY_GT, 2.0f)};
    EXPECT_EQ(func_1(float_to_int_gt, DataType::FLOAT32), "unary_gt_tile_int32(0, 2);");
    // The TYPECAST parameters may be int32_t or uint32_t backed; the cast's dtype is read from the active type.
    const std::vector<EltwiseUnaryWithParam> int_params_gt_half = {
        EltwiseUnaryWithParam(
            UnaryOpType::TYPECAST,
            static_cast<std::int32_t>(DataType::INT32),
            static_cast<std::int32_t>(DataType::FLOAT32)),
        EltwiseUnaryWithParam(UnaryOpType::UNARY_GT, 0.5f)};
    EXPECT_EQ(func_1(int_params_gt_half, DataType::INT32), "unary_gt_tile(0, 0x3f000000u);");
    const std::vector<EltwiseUnaryWithParam> uint_params_gt = {
        EltwiseUnaryWithParam(
            UnaryOpType::TYPECAST,
            static_cast<std::uint32_t>(DataType::INT32),
            static_cast<std::uint32_t>(DataType::FLOAT32)),
        EltwiseUnaryWithParam(UnaryOpType::UNARY_GT, 2.0f)};
    EXPECT_EQ(func_1(uint_params_gt, DataType::INT32), "unary_gt_tile(0, 0x40000000u);");
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
