// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/reshape_op.hpp"

#include <gtest/gtest.h>

#include <array>
#include <stdexcept>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"

class ReshapeOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(ReshapeOpTest, RejectsDifferentLogicalVolumeWithinTilePadding) {
    using namespace ttml;

    auto* device = &autograd::ctx().get_device();
    auto input = autograd::create_tensor(core::zeros(ttnn::Shape({1U, 1U, 1U, 32U}), device));
    std::array<uint32_t, 4> padding_as_data_shape{1U, 1U, 32U, 32U};

    EXPECT_THROW(ops::reshape(input, padding_as_data_shape), std::logic_error);
}
