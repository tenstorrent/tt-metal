// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::operations::binary::test {

struct SumTensorLastDimParam {
    int32_t dim;
    uint32_t h;
    uint32_t w;
};

class SumTensorLastDimFixture : public TTNNFixtureWithSuiteDevice<SumTensorLastDimFixture>,
                                public testing::WithParamInterface<SumTensorLastDimParam> {};

TEST_P(SumTensorLastDimFixture, SumTensorCorrectly) {
    auto param = GetParam();
    auto& device = *device_;
    std::array<uint32_t, 2> dimensions = {param.h, param.w};
    ttnn::Shape shape(dimensions);
    std::array<uint32_t, 2> reduced_dimensions = {param.h, 1};
    ttnn::Shape reduced_shape(reduced_dimensions);

    {
        const auto input_tensor = ttnn::ones(shape, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
        const auto output_tensor = ttnn::sum(
            input_tensor,
            param.dim,
            true);  // keepdim true allows output of reduce to be returned directly without reshape on device call
        TT_FATAL(
            output_tensor.logical_shape() == reduced_shape,
            "Shapes are not equal output tensor shape {} vs reduced shape {}",
            output_tensor.logical_shape(),
            reduced_shape);
        auto output_vector = output_tensor.to_vector<bfloat16>();
        float expected = (float)param.w;
        for (uint32_t i = 0; i < param.h; i++) {
            float value = output_vector[i];
            TT_FATAL(value == expected, "{} != {} @ {}", value, expected, i);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    SumTensorLastDimTests, SumTensorLastDimFixture, ::testing::Values(SumTensorLastDimParam{-1, 32, 64}));

}  // namespace ttnn::operations::binary::test
