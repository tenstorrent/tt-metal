// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <array>
#include <memory>
#include <optional>

#include <tt_stl/assert.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/shape.hpp>
#include "ttnn/decorators.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::operations::unary::test {

struct RsubUnaryParam {
    int32_t scalar1;
    int32_t scalar2;
    uint32_t h;
    uint32_t w;
};

class RsubUnaryFixture : public TTNNFixtureWithSuiteDevice<RsubUnaryFixture>,
                         public testing::WithParamInterface<RsubUnaryParam> {};

TEST_P(RsubUnaryFixture, CompareWithTorchReference) {
    auto param = GetParam();
    auto& device = *device_;
    std::array<uint32_t, 2> dimensions = {param.h, param.w};
    ttnn::Shape shape(dimensions);

    constexpr DataType dtype = DataType::INT32;

    // Create zero tensor with zeros
    auto zero_tensor = ttnn::zeros(shape, dtype, ttnn::TILE_LAYOUT, device);

    // create input tensor with fill value
    auto input_tensor = ttnn::fill(zero_tensor, param.scalar1);

    // Run TTNN unary rsub operation
    auto ttnn_output = ttnn::rsub_sfpu(input_tensor, param.scalar2);

    int32_t expected_scalar_result = param.scalar2 - param.scalar1;

    // Create expected tensor (should be filled with the expected scalar value)
    auto expected_tensor = ttnn::full(shape, expected_scalar_result, dtype, ttnn::TILE_LAYOUT, device);

    // Compare results using allclose
    auto expected_host = ttnn::from_device(expected_tensor);
    auto output_host = ttnn::from_device(ttnn_output);

    TT_FATAL(
        ttnn::allclose<int32_t>(expected_host, output_host), "TTNN unary rsub result does not match expected result");
}

INSTANTIATE_TEST_SUITE_P(
    RsubUnaryTests,
    RsubUnaryFixture,
    ::testing::Values(
        RsubUnaryParam{-2147483648, -89, 128, 32},
        RsubUnaryParam{2147483647, 5, 128, 32},
        RsubUnaryParam{-45, -5, 32, 32},
        RsubUnaryParam{-5, 67, 64, 64},
        RsubUnaryParam{0, 78, 4, 4},
        RsubUnaryParam{17, -96, 64, 64},
        RsubUnaryParam{100, 458, 128, 128},
        RsubUnaryParam{564, -118, 64, 128}));

}  // namespace ttnn::operations::unary::test
