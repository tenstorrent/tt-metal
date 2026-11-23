// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <tt-metalium/shape.hpp>
#include "ttnn/device.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::operations::binary::test {

// bitwise_and/or/xor have both an int32_t and a uint32_t scalar overload; every integer scalar type must still
// resolve to exactly one of them.
template <typename T>
concept BitwiseScalarCallable = requires(const Tensor& t, T s) {
    ttnn::bitwise_and(t, s);
    ttnn::bitwise_or(t, s);
    ttnn::bitwise_xor(t, s);
};

static_assert(BitwiseScalarCallable<int8_t>);
static_assert(BitwiseScalarCallable<int16_t>);
static_assert(BitwiseScalarCallable<int32_t>);
static_assert(BitwiseScalarCallable<int64_t>);
static_assert(BitwiseScalarCallable<long>);
static_assert(BitwiseScalarCallable<long long>);
static_assert(BitwiseScalarCallable<uint8_t>);
static_assert(BitwiseScalarCallable<uint16_t>);
static_assert(BitwiseScalarCallable<uint32_t>);
static_assert(BitwiseScalarCallable<uint64_t>);
static_assert(BitwiseScalarCallable<std::size_t>);

template <typename T>
concept SelectsUInt32Overload = requires(const Tensor& t, T s) { ttnn::bitwise_and<T>(t, s); };

static_assert(SelectsUInt32Overload<uint32_t>);
static_assert(!SelectsUInt32Overload<int32_t>);
static_assert(!SelectsUInt32Overload<int64_t>);
static_assert(!SelectsUInt32Overload<uint64_t>);

enum class BitwiseOp { And, Or, Xor };
// The bool selects the binary_ng tensor-scalar path instead of the unary SFPU fast path.
using BitwiseUint32Param = std::tuple<BitwiseOp, uint32_t, bool>;
using BitwiseUint16Param = std::tuple<BitwiseOp, int32_t, bool>;

namespace {

using unary::EltwiseUnaryWithParam;
using unary::UnaryOpType;

std::string op_name(BitwiseOp op) {
    switch (op) {
        case BitwiseOp::And: return "bitwise_and";
        case BitwiseOp::Or: return "bitwise_or";
        case BitwiseOp::Xor: return "bitwise_xor";
    }
    return "unknown";
}

template <typename T>
T reference(BitwiseOp op, T x, T s) {
    switch (op) {
        case BitwiseOp::And: return x & s;
        case BitwiseOp::Or: return x | s;
        case BitwiseOp::Xor: return x ^ s;
    }
    return 0;
}

// RELU is an identity on unsigned inputs; as an lhs activation it forces the binary_ng path.
template <typename S>
ttnn::Tensor run_op(BitwiseOp op, const ttnn::Tensor& t, S s, bool use_binary_ng = false) {
    const std::array<EltwiseUnaryWithParam, 1> relu = {EltwiseUnaryWithParam(UnaryOpType::RELU)};
    const auto lhs_activations =
        use_binary_ng ? ttsl::Span<const EltwiseUnaryWithParam>(relu) : ttsl::Span<const EltwiseUnaryWithParam>();
    switch (op) {
        case BitwiseOp::And: return ttnn::bitwise_and(t, s, std::nullopt, std::nullopt, {}, lhs_activations);
        case BitwiseOp::Or: return ttnn::bitwise_or(t, s, std::nullopt, std::nullopt, {}, lhs_activations);
        case BitwiseOp::Xor: return ttnn::bitwise_xor(t, s, std::nullopt, std::nullopt, {}, lhs_activations);
    }
    TT_THROW("unreachable");
}

template <typename T>
ttnn::Tensor make_tensor(ttnn::MeshDevice* device, const std::vector<T>& data, DataType dtype) {
    const ttnn::Shape shape({32, 64});
    const TensorSpec spec(
        shape, tt::tt_metal::TensorLayout(dtype, tt::tt_metal::PageConfig(Layout::TILE), MemoryConfig{}));
    return Tensor::from_vector(data, spec, device);
}

constexpr uint32_t kInt32Max = 2147483647u;   // 2^31 - 1
constexpr uint32_t kTwoPow31 = 2147483648u;   // 2^31
constexpr uint32_t kUint32Max = 4294967295u;  // 2^32 - 1
constexpr uint32_t kUint16Max = 65535u;

// Half edge values, half random values covering the whole dtype range.
template <typename T>
std::vector<T> make_input(uint32_t seed, size_t volume, const std::vector<T>& edge_values) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<uint32_t> dist(0u, std::numeric_limits<T>::max());
    std::vector<T> data(volume);
    for (size_t i = 0; i < volume; ++i) {
        data[i] = (i % 2 == 0) ? edge_values[(i / 2) % edge_values.size()] : static_cast<T>(dist(gen));
    }
    return data;
}

template <typename T>
void expect_matches_reference(BitwiseOp op, const std::vector<T>& input, T scalar, const std::vector<T>& output) {
    ASSERT_EQ(output.size(), input.size());
    size_t mismatches = 0;
    for (size_t i = 0; i < input.size(); ++i) {
        const T expected = reference(op, input[i], scalar);
        if (output[i] != expected) {
            if (mismatches < 8) {
                ADD_FAILURE() << op_name(op) << "(" << input[i] << ", " << scalar << ") = " << output[i]
                              << ", expected " << expected << " at index " << i;
            }
            ++mismatches;
        }
    }
    EXPECT_EQ(mismatches, 0u);
}

std::string path_name(bool use_binary_ng) { return use_binary_ng ? "binary_ng" : "unary"; }

}  // namespace

class BitwiseUint32ScalarFixture : public TTNNFixtureWithSuiteDevice<BitwiseUint32ScalarFixture>,
                                   public testing::WithParamInterface<BitwiseUint32Param> {
protected:
    void SetUp() override {
        if (device_->arch() == tt::ARCH::QUASAR) {
            GTEST_SKIP() << "UInt32 bitwise ops are not supported on Quasar.";
        }
    }
};

TEST_P(BitwiseUint32ScalarFixture, MatchesUnsignedReference) {
    const auto [op, scalar, use_binary_ng] = GetParam();
    const auto input = make_input<uint32_t>(
        scalar,
        32 * 64,
        {0u, 1u, kInt32Max, kTwoPow31, kTwoPow31 + 1u, 3000000000u, kUint32Max - 1u, kUint32Max, ~scalar, scalar});
    const auto input_tensor = make_tensor(device_, input, DataType::UINT32);

    const auto output = run_op(op, input_tensor, scalar, use_binary_ng).cpu().to_vector<uint32_t>();

    expect_matches_reference(op, input, scalar, output);
}

INSTANTIATE_TEST_SUITE_P(
    BitwiseUint32ScalarTests,
    BitwiseUint32ScalarFixture,
    ::testing::Combine(
        ::testing::Values(BitwiseOp::And, BitwiseOp::Or, BitwiseOp::Xor),
        ::testing::Values(0u, 1u, kInt32Max, kTwoPow31, 3000000000u, kUint32Max - 1u, kUint32Max),
        ::testing::Bool()),
    [](const testing::TestParamInfo<BitwiseUint32Param>& info) {
        return op_name(std::get<0>(info.param)) + "_" + std::to_string(std::get<1>(info.param)) + "_" +
               path_name(std::get<2>(info.param));
    });

class BitwiseUint16ScalarFixture : public TTNNFixtureWithSuiteDevice<BitwiseUint16ScalarFixture>,
                                   public testing::WithParamInterface<BitwiseUint16Param> {};

TEST_P(BitwiseUint16ScalarFixture, MatchesUnsignedReference) {
    const auto [op, scalar, use_binary_ng] = GetParam();
    const auto scalar_u16 = static_cast<uint16_t>(scalar);
    const auto input = make_input<uint16_t>(
        scalar, 32 * 64, {0, 1, 0x1234, 0x8000, kUint16Max - 1, kUint16Max, static_cast<uint16_t>(~scalar_u16)});
    const auto input_tensor = make_tensor(device_, input, DataType::UINT16);

    const auto output = run_op(op, input_tensor, scalar, use_binary_ng).cpu().to_vector<uint16_t>();

    expect_matches_reference(op, input, scalar_u16, output);
}

INSTANTIATE_TEST_SUITE_P(
    BitwiseUint16ScalarTests,
    BitwiseUint16ScalarFixture,
    ::testing::Combine(
        ::testing::Values(BitwiseOp::And, BitwiseOp::Or, BitwiseOp::Xor),
        ::testing::Values(0, 1, 0x5E00, 0x8000, static_cast<int32_t>(kUint16Max)),
        ::testing::Bool()),
    [](const testing::TestParamInfo<BitwiseUint16Param>& info) {
        return op_name(std::get<0>(info.param)) + "_" + std::to_string(std::get<1>(info.param)) + "_" +
               path_name(std::get<2>(info.param));
    });

class BitwiseScalarRangeFixture : public TTNNFixtureWithSuiteDevice<BitwiseScalarRangeFixture>,
                                  public testing::WithParamInterface<BitwiseOp> {};

TEST_P(BitwiseScalarRangeFixture, RejectsScalarsOutsideInputDtypeRange) {
    using ::testing::HasSubstr;
    using ::testing::ThrowsMessage;
    const auto op = GetParam();
    const auto int32_t_tensor = make_tensor(device_, std::vector<int32_t>(32 * 64, 7), DataType::INT32);
    const auto uint16_t_tensor = make_tensor(device_, std::vector<uint16_t>(32 * 64, 7), DataType::UINT16);

    EXPECT_THAT(
        [&] { run_op(op, int32_t_tensor, kTwoPow31); },
        ThrowsMessage<std::runtime_error>(
            HasSubstr("requires a scalar in [-2147483648, 2147483647], got 2147483648.")));
    EXPECT_THAT(
        [&] { run_op(op, int32_t_tensor, kUint32Max); },
        ThrowsMessage<std::runtime_error>(
            HasSubstr("requires a scalar in [-2147483648, 2147483647], got 4294967295.")));
    EXPECT_THAT(
        [&] { run_op(op, uint16_t_tensor, int32_t{65536}); },
        ThrowsMessage<std::runtime_error>(HasSubstr("requires a scalar in [0, 65535], got 65536.")));
    EXPECT_THAT(
        [&] { run_op(op, uint16_t_tensor, int32_t{-1}); },
        ThrowsMessage<std::runtime_error>(HasSubstr("requires a scalar in [0, 65535], got -1.")));
    EXPECT_THAT(
        [&] { run_op(op, uint16_t_tensor, 3000000000u); },
        ThrowsMessage<std::runtime_error>(HasSubstr("requires a scalar in [0, 65535], got 3000000000.")));

    if (device_->arch() != tt::ARCH::QUASAR) {
        const auto uint32_t_tensor = make_tensor(device_, std::vector<uint32_t>(32 * 64, 7u), DataType::UINT32);
        EXPECT_THAT(
            [&] { run_op(op, uint32_t_tensor, int32_t{-1}); },
            ThrowsMessage<std::runtime_error>(HasSubstr("requires a scalar in [0, 4294967295], got -1.")));
        EXPECT_THAT(
            [&] { run_op(op, uint32_t_tensor, int32_t{-32}); },
            ThrowsMessage<std::runtime_error>(HasSubstr("requires a scalar in [0, 4294967295], got -32.")));
    }
}

INSTANTIATE_TEST_SUITE_P(
    BitwiseScalarRangeTests,
    BitwiseScalarRangeFixture,
    ::testing::Values(BitwiseOp::And, BitwiseOp::Or, BitwiseOp::Xor),
    [](const testing::TestParamInfo<BitwiseOp>& info) { return op_name(info.param); });

}  // namespace ttnn::operations::binary::test
