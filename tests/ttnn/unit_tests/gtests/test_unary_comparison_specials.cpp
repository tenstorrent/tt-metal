// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The unary comparisons against the special values, on the registered ops themselves.
//
// A comparison built from a subtract and an int32 test is not an IEEE comparison: two equal
// infinities subtract to a NaN so they never compare equal, a NaN operand subtracts to a NaN whose
// sign decides the answer, and -0.0 is a negative non-zero word.

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/shape.hpp>
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::operations::unary::test {
namespace {

using tt::tt_metal::distributed::MeshDevice;

struct Operand {
    const char* name;
    uint16_t bf16_bits;
    uint32_t fp32_bits;
};

// Bit patterns, because bfloat16(float) canonicalises every NaN to 0x7FC0.
constexpr std::array<Operand, 13> kOperands = {{
    {"+inf", 0x7F80, 0x7F800000},
    {"-inf", 0xFF80, 0xFF800000},
    {"+0.0", 0x0000, 0x00000000},
    {"-0.0", 0x8000, 0x80000000},
    {"1.0", 0x3F80, 0x3F800000},
    {"-1.0", 0xBF80, 0xBF800000},
    {"2.0", 0x4000, 0x40000000},
    {"max", 0x7F7F, 0x7F7FFFFF},
    {"-max", 0xFF7F, 0xFF7FFFFF},
    {"min subnormal", 0x0001, 0x00000001},
    {"-min subnormal", 0x8001, 0x80000001},
    {"+NaN", 0x7FC1, 0x7FC00001},
    {"-NaN", 0xFFC1, 0xFFC00001},
}};

struct Scalar {
    const char* name;
    uint32_t bits;
};

constexpr std::array<Scalar, 7> kScalars = {{
    {"pos_inf", 0x7F800000},
    {"neg_inf", 0xFF800000},
    {"nan", 0x7FC00001},
    {"one", 0x3F800000},
    {"pos_zero", 0x00000000},
    {"neg_zero", 0x80000000},
    {"min_subnormal", 0x00000001},
}};

struct OpUnderTest {
    const char* name;
    ttnn::Tensor (*call)(const ttnn::Tensor&, float);
    bool (*reference)(float, float);
};

constexpr std::array<OpUnderTest, 6> kOps = {{
    {"eq_unary",
     [](const ttnn::Tensor& t, float s) { return ttnn::eq_unary(t, s); },
     [](float a, float b) { return a == b; }},
    {"ne_unary",
     [](const ttnn::Tensor& t, float s) { return ttnn::ne_unary(t, s); },
     [](float a, float b) { return a != b; }},
    {"gt_unary",
     [](const ttnn::Tensor& t, float s) { return ttnn::gt_unary(t, s); },
     [](float a, float b) { return a > b; }},
    {"lt_unary",
     [](const ttnn::Tensor& t, float s) { return ttnn::lt_unary(t, s); },
     [](float a, float b) { return a < b; }},
    {"ge_unary",
     [](const ttnn::Tensor& t, float s) { return ttnn::ge_unary(t, s); },
     [](float a, float b) { return a >= b; }},
    {"le_unary",
     [](const ttnn::Tensor& t, float s) { return ttnn::le_unary(t, s); },
     [](float a, float b) { return a <= b; }},
}};

template <typename HostT>
HostT from_bf16_bits(uint16_t bits) {
    if constexpr (std::is_same_v<HostT, ::bfloat16>) {
        return std::bit_cast<::bfloat16>(bits);
    } else {
        return std::bit_cast<float>(static_cast<uint32_t>(bits) << 16);
    }
}

// The kernel reads "both operands are zero" off abs(v) + abs(s), and the SFPU add flushes
// subnormals, so these comparisons are DAZ. The reference flushes the same way.
float daz(float x) { return std::fpclassify(x) == FP_SUBNORMAL ? std::copysign(0.0f, x) : x; }

template <typename HostT>
ttnn::Tensor to_device(MeshDevice& device, std::vector<HostT> values, DataType dtype, uint32_t rows) {
    values.resize(static_cast<size_t>(rows) * tt::constants::TILE_WIDTH, HostT(0.0f));
    tt::tt_metal::TensorSpec spec(
        tt::tt_metal::Shape({1, 1, rows, tt::constants::TILE_WIDTH}),
        tt::tt_metal::TensorLayout(
            dtype, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), tt::tt_metal::MemoryConfig{}));
    return ttnn::Tensor::from_vector(std::move(values), spec).to_device(&device);
}

template <typename HostT>
std::vector<float> run(MeshDevice& device, const OpUnderTest& op, const std::vector<HostT>& host, DataType dtype,
                       float scalar, uint32_t rows) {
    const auto output = op.call(to_device(device, host, dtype, rows), scalar);
    const auto read = ttnn::from_device(output).template to_vector<HostT>();
    std::vector<float> values;
    values.reserve(read.size());
    for (const auto& v : read) {
        values.push_back(static_cast<float>(v));
    }
    return values;
}

class UnaryComparisonSpecials : public TTNNFixtureWithSuiteDevice<UnaryComparisonSpecials>,
                                public testing::WithParamInterface<Scalar> {};

template <typename HostT>
void check_specials(MeshDevice& device, DataType dtype, const Scalar& scalar) {
    const float s = std::bit_cast<float>(scalar.bits);

    std::vector<HostT> host;
    std::vector<float> widened;
    for (const auto& operand : kOperands) {
        if constexpr (std::is_same_v<HostT, ::bfloat16>) {
            host.push_back(std::bit_cast<::bfloat16>(operand.bf16_bits));
        } else {
            host.push_back(std::bit_cast<float>(operand.fp32_bits));
        }
        widened.push_back(static_cast<float>(host.back()));
    }

    for (const auto& op : kOps) {
        const auto got = run(device, op, host, dtype, s, tt::constants::TILE_HEIGHT);
        for (size_t i = 0; i < kOperands.size(); i++) {
            const float want = op.reference(daz(widened[i]), daz(s)) ? 1.0f : 0.0f;
            EXPECT_EQ(got[i], want) << op.name << "(" << kOperands[i].name << ", " << scalar.name << ") in "
                                    << (std::is_same_v<HostT, ::bfloat16> ? "bfloat16" : "float32");
        }
    }
}

TEST_P(UnaryComparisonSpecials, Bfloat16) { check_specials<::bfloat16>(*device_, DataType::BFLOAT16, GetParam()); }

TEST_P(UnaryComparisonSpecials, Float32) { check_specials<float>(*device_, DataType::FLOAT32, GetParam()); }

INSTANTIATE_TEST_SUITE_P(Scalars, UnaryComparisonSpecials, testing::ValuesIn(kScalars), [](const auto& info) {
    return std::string(info.param.name);
});

class UnaryComparisonExhaustive : public TTNNFixtureWithSuiteDevice<UnaryComparisonExhaustive> {};

// Every bfloat16 bit pattern, including all 254 NaNs, both zeros and all 254 subnormals.
template <typename HostT>
void check_every_pattern(MeshDevice& device, DataType dtype) {
    constexpr uint32_t kRows = 65536 / tt::constants::TILE_WIDTH;

    std::vector<HostT> host;
    std::vector<float> widened;
    host.reserve(65536);
    widened.reserve(65536);
    for (uint32_t bits = 0; bits < 65536; bits++) {
        host.push_back(from_bf16_bits<HostT>(static_cast<uint16_t>(bits)));
        widened.push_back(static_cast<float>(host.back()));
    }

    for (const Scalar& scalar : {kScalars[0], kScalars[1], kScalars[4]}) {
        const float s = std::bit_cast<float>(scalar.bits);
        for (const auto& op : kOps) {
            const auto got = run(device, op, host, dtype, s, kRows);
            size_t wrong = 0;
            uint32_t first = 0;
            for (uint32_t bits = 0; bits < 65536; bits++) {
                const float want = op.reference(daz(widened[bits]), daz(s)) ? 1.0f : 0.0f;
                if (got[bits] != want) {
                    if (wrong == 0) {
                        first = bits;
                    }
                    wrong++;
                }
            }
            EXPECT_EQ(wrong, 0u) << op.name << " against " << scalar.name << " in "
                                 << (std::is_same_v<HostT, ::bfloat16> ? "bfloat16" : "float32") << ": " << wrong
                                 << " of 65536 patterns differ, first at 0x" << std::hex << first;
        }
    }
}

TEST_F(UnaryComparisonExhaustive, Bfloat16) { check_every_pattern<::bfloat16>(*device_, DataType::BFLOAT16); }

TEST_F(UnaryComparisonExhaustive, Float32) { check_every_pattern<float>(*device_, DataType::FLOAT32); }

}  // namespace
}  // namespace ttnn::operations::unary::test
