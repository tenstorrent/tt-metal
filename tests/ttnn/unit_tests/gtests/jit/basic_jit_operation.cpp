// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <stdint.h>
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

#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

class BasicJistOperationFixture : public ttnn::TTNNFixtureWithDevice {};

TEST_F(BasicJistOperationFixture, CreateEmptyTensor) {
    auto empty_tensor = ttnn::empty(
        ttnn::Shape{1, 2, 3}, ttnn::DataType::BFLOAT16, ttnn::Layout::TILE, device_, ttnn::DRAM_MEMORY_CONFIG);
    ASSERT_EQ(empty_tensor.logical_shape(), (ttnn::Shape{1, 2, 3}));
    ASSERT_EQ(empty_tensor.dtype(), ttnn::DataType::BFLOAT16);
    ASSERT_EQ(empty_tensor.layout(), ttnn::Layout::TILE);
    ASSERT_EQ(empty_tensor.device(), device_);
}
