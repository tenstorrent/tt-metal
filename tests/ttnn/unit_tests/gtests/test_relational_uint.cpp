// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include <tt-metalium/shape.hpp>
#include "ttnn/device.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::operations::unary::test {

namespace {

enum class RelationalOp { Eq, Ne, Gt, Ge, Lt, Le };

using RelationalUintParam = std::tuple<RelationalOp, uint32_t>;

std::string op_name(RelationalOp op) {
    switch (op) {
        case RelationalOp::Eq: return "eq";
        case RelationalOp::Ne: return "ne";
        case RelationalOp::Gt: return "gt";
        case RelationalOp::Ge: return "ge";
        case RelationalOp::Lt: return "lt";
        case RelationalOp::Le: return "le";
    }
    return "unknown";
}

uint32_t reference(RelationalOp op, uint32_t x, uint32_t s) {
    switch (op) {
        case RelationalOp::Eq: return x == s;
        case RelationalOp::Ne: return x != s;
        case RelationalOp::Gt: return x > s;
        case RelationalOp::Ge: return x >= s;
        case RelationalOp::Lt: return x < s;
        case RelationalOp::Le: return x <= s;
    }
    return 0;
}

ttnn::Tensor run_op(RelationalOp op, const ttnn::Tensor& t, uint32_t s) {
    switch (op) {
        case RelationalOp::Eq: return ttnn::eq_unary(t, s);
        case RelationalOp::Ne: return ttnn::ne_unary(t, s);
        case RelationalOp::Gt: return ttnn::gt_unary(t, s);
        case RelationalOp::Ge: return ttnn::ge_unary(t, s);
        case RelationalOp::Lt: return ttnn::lt_unary(t, s);
        case RelationalOp::Le: return ttnn::le_unary(t, s);
    }
    TT_THROW("unreachable");
}

// Values at or above 2^31 read as negative when treated as int32, so they are the cases a signed compare gets wrong.
constexpr uint32_t kInt32Max = 2147483647u;   // 2^31 - 1
constexpr uint32_t kTwoPow31 = 2147483648u;   // 2^31
constexpr uint32_t kUint32Max = 4294967295u;  // 2^32 - 1

std::vector<uint32_t> make_input(uint32_t scalar, size_t volume) {
    const std::array<uint32_t, 13> edge_values = {
        0u,
        1u,
        kInt32Max - 1u,
        kInt32Max,
        kTwoPow31,
        kTwoPow31 + 1u,
        3000000000u,
        kUint32Max - 1u,
        kUint32Max,
        scalar - 1u,  // wraps to kUint32Max when scalar is 0
        scalar,
        scalar + 1u,         // wraps to 0 when scalar is kUint32Max
        scalar + kTwoPow31,  // same low 31 bits as scalar, in the other half of the range
    };
    std::mt19937 gen(scalar);
    std::uniform_int_distribution<uint32_t> dist(0u, kUint32Max);
    std::vector<uint32_t> data(volume);
    for (size_t i = 0; i < volume; ++i) {
        data[i] = (i % 2 == 0) ? edge_values[(i / 2) % edge_values.size()] : dist(gen);
    }
    return data;
}

}  // namespace

class RelationalUintFixture : public TTNNFixtureWithSuiteDevice<RelationalUintFixture>,
                              public testing::WithParamInterface<RelationalUintParam> {};

TEST_P(RelationalUintFixture, MatchesUnsignedReference) {
    const auto [op, scalar] = GetParam();
    const ttnn::Shape shape({64, 64});
    const TensorSpec spec(
        shape, tt::tt_metal::TensorLayout(DataType::UINT32, tt::tt_metal::PageConfig(Layout::TILE), MemoryConfig{}));

    const auto input = make_input(scalar, shape.volume());
    const auto input_tensor = Tensor::from_vector(input, spec, device_);

    const auto output = run_op(op, input_tensor, scalar).cpu().to_vector<uint32_t>();

    ASSERT_EQ(output.size(), input.size());
    size_t mismatches = 0;
    for (size_t i = 0; i < input.size(); ++i) {
        const uint32_t expected = reference(op, input[i], scalar);
        if (output[i] != expected) {
            if (mismatches < 8) {
                ADD_FAILURE() << op_name(op) << "_unary(" << input[i] << ", " << scalar << ") = " << output[i]
                              << ", expected " << expected << " at index " << i;
            }
            ++mismatches;
        }
    }
    EXPECT_EQ(mismatches, 0u);
}

INSTANTIATE_TEST_SUITE_P(
    RelationalUintTests,
    RelationalUintFixture,
    ::testing::Combine(
        ::testing::Values(
            RelationalOp::Eq, RelationalOp::Ne, RelationalOp::Gt, RelationalOp::Ge, RelationalOp::Lt, RelationalOp::Le),
        ::testing::Values(0u, 1u, 12345u, kInt32Max, kTwoPow31, 3000000000u, kUint32Max)),
    [](const testing::TestParamInfo<RelationalUintParam>& info) {
        return op_name(std::get<0>(info.param)) + "_" + std::to_string(std::get<1>(info.param));
    });

class RelationalUintScalarRangeFixture : public TTNNFixtureWithSuiteDevice<RelationalUintScalarRangeFixture> {
protected:
    ttnn::Tensor make_tensor() const {
        const ttnn::Shape shape({32, 32});
        const TensorSpec spec(
            shape,
            tt::tt_metal::TensorLayout(DataType::UINT32, tt::tt_metal::PageConfig(Layout::TILE), MemoryConfig{}));
        return Tensor::from_vector(std::vector<uint32_t>(shape.volume(), 7u), spec, device_);
    }
};

TEST_F(RelationalUintScalarRangeFixture, RejectsScalarsOutsideUint32Range) {
    const auto t = make_tensor();
    EXPECT_ANY_THROW(ttnn::gt_unary(t, int32_t{-1}));
    EXPECT_ANY_THROW(ttnn::eq_unary(t, int32_t{-2147483647}));
    EXPECT_ANY_THROW(ttnn::lt_unary(t, -1.0f));
    EXPECT_ANY_THROW(ttnn::ge_unary(t, 2.5f));
    EXPECT_ANY_THROW(ttnn::le_unary(t, 4294967296.0f));
    EXPECT_ANY_THROW(ttnn::ne_unary(t, std::numeric_limits<float>::quiet_NaN()));
    EXPECT_ANY_THROW(ttnn::ne_unary(t, std::numeric_limits<float>::infinity()));
}

TEST_F(RelationalUintScalarRangeFixture, AcceptsIntegralScalarsOfAnyType) {
    const auto t = make_tensor();
    EXPECT_EQ(ttnn::gt_unary(t, int32_t{6}).cpu().to_vector<uint32_t>(), std::vector<uint32_t>(32 * 32, 1u));
    EXPECT_EQ(ttnn::lt_unary(t, 3000000000.0f).cpu().to_vector<uint32_t>(), std::vector<uint32_t>(32 * 32, 1u));
    EXPECT_EQ(ttnn::eq_unary(t, 7.0f).cpu().to_vector<uint32_t>(), std::vector<uint32_t>(32 * 32, 1u));
}

}  // namespace ttnn::operations::unary::test
