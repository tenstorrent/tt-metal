// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <tt-metalium/shape.hpp>
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::operations::unary::test {

// RDIV kernel codegen reads params[1] as the rounding mode and uses it to index a
// 3-entry ckernel::RoundingMode table. These host-side tests pin that param
// contract, so a caller cannot regress to passing a single param -- which read
// past the end of the span -- without a test failing.

TEST(RdivCodegen, RoundingModeSelectsMatchingTemplate) {
    const std::array<std::pair<float, std::string_view>, 3> cases = {{
        {0.0f, "ckernel::RoundingMode::None"},
        {1.0f, "ckernel::RoundingMode::Trunc"},
        {2.0f, "ckernel::RoundingMode::Floor"},
    }};

    for (const auto& [mode, expected] : cases) {
        const std::vector<float> params = {2.0f, mode};
        const auto [init, func] = utils::get_op_init_and_func<float>(UnaryOpType::RDIV, params);
        EXPECT_EQ(init, "rdiv_tile_init();");
        EXPECT_NE(func.find(expected), std::string::npos) << "mode " << mode << " generated " << func;
    }
}

// Regression: div_sfpu(float, Tensor) built RDIV with only the divisor, so
// params[1] was an out-of-bounds read on a 1-element span.
TEST(RdivCodegen, RejectsSingleParam) {
    const std::vector<float> params = {2.0f};
    EXPECT_ANY_THROW(utils::get_op_init_and_func<float>(UnaryOpType::RDIV, params));
}

TEST(RdivCodegen, RejectsOutOfRangeRoundingMode) {
    const std::vector<float> params = {2.0f, 3.0f};  // valid modes are 0, 1, 2
    EXPECT_ANY_THROW(utils::get_op_init_and_func<float>(UnaryOpType::RDIV, params));
}

// Device coverage for the scalar-first overload: ttnn::div_sfpu(scalar, tensor)
// computes scalar / tensor, matching ttnn::rdiv(tensor, scalar).

struct DivSfpuParam {
    float scalar;
    float input_value;
    uint32_t h;
    uint32_t w;
};

class DivSfpuScalarFirstFixture : public TTNNFixtureWithSuiteDevice<DivSfpuScalarFirstFixture>,
                                  public testing::WithParamInterface<DivSfpuParam> {};

TEST_P(DivSfpuScalarFirstFixture, MatchesScalarDividedByInput) {
    const auto param = GetParam();
    auto& device = *device_;
    const std::array<uint32_t, 2> dimensions = {param.h, param.w};
    const ttnn::Shape shape(dimensions);

    constexpr DataType dtype = DataType::FLOAT32;

    const auto input_tensor = ttnn::full(shape, param.input_value, dtype, ttnn::TILE_LAYOUT, device);
    const auto ttnn_output = ttnn::div_sfpu(param.scalar, input_tensor);

    const float expected_scalar_result = param.scalar / param.input_value;
    const auto expected_tensor = ttnn::full(shape, expected_scalar_result, dtype, ttnn::TILE_LAYOUT, device);

    const auto expected_host = ttnn::from_device(expected_tensor);
    const auto output_host = ttnn::from_device(ttnn_output);

    EXPECT_TRUE(ttnn::allclose<float>(expected_host, output_host, 1e-2f, 1e-3f))
        << param.scalar << " / " << param.input_value;
}

INSTANTIATE_TEST_SUITE_P(
    DivSfpuScalarFirstTests,
    DivSfpuScalarFirstFixture,
    ::testing::Values(
        DivSfpuParam{1.0f, 2.0f, 32, 32},
        DivSfpuParam{-6.0f, 3.0f, 32, 32},
        DivSfpuParam{7.5f, -2.5f, 64, 64},
        DivSfpuParam{100.0f, 8.0f, 64, 128},
        DivSfpuParam{0.5f, 0.25f, 128, 32}));

}  // namespace ttnn::operations::unary::test
