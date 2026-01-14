// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
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
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::operations::binary::test {

struct Add1DTensorAndScalarParam {
    float scalar;
    uint32_t h;
    uint32_t w;
};

class Add1DTensorAndScalarFixture : public TTNNFixtureWithSuiteDevice<Add1DTensorAndScalarFixture>,
                                    public testing::WithParamInterface<Add1DTensorAndScalarParam> {};

TEST_P(Add1DTensorAndScalarFixture, AddsScalarCorrectly) {
    auto param = GetParam();
    auto& device = *device_;
    std::array<uint32_t, 2> dimensions = {param.h, param.w};
    ttnn::Shape shape(dimensions);

    {
        const auto input_tensor = ttnn::zeros(shape, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
        const auto output_tensor = input_tensor + param.scalar;
        const auto expected_tensor = ttnn::full(shape, param.scalar, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
        TT_FATAL(
            ttnn::allclose<::bfloat16>(ttnn::from_device(expected_tensor), ttnn::from_device(output_tensor)), "Error");
    }
}

INSTANTIATE_TEST_SUITE_P(
    Add1DTensorAndScalarTests, Add1DTensorAndScalarFixture, ::testing::Values(Add1DTensorAndScalarParam{3.0f, 32, 64}));

}  // namespace ttnn::operations::binary::test
