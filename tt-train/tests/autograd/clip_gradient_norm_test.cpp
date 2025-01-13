// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "autograd/clip_gradient_norm.hpp"

#include <gtest/gtest.h>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"

class ClipGradientNormTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(ClipGradientNormTest, GradNormTensor_0) {
    auto* device = &ttml::autograd::ctx().get_device();

    std::vector<float> data(81, -1.F);
    auto shape = ttml::core::create_shape({1, 1, 9, 9});
    auto tensor = ttml::core::from_vector(data, shape, device);

    ttml::autograd::clip_tensor_norm_(tensor, 3.F);

    auto clipped_vec = ttml::core::to_vector(tensor);
    auto norm = 0.F;
    for (auto& value : clipped_vec) {
        norm += value * value;
    }
    norm = std::sqrt(norm);
    EXPECT_NEAR(norm, 3.F, 1e-2);
    for (const auto& value : clipped_vec) {
        EXPECT_NEAR(value, -1.F / 3.F, 1e-2);
    }
}

TEST_F(ClipGradientNormTest, GradNormTensor_1) {
    auto* device = &ttml::autograd::ctx().get_device();

    std::vector<float> data(81, -1.F);
    auto shape = ttml::core::create_shape({1, 1, 9, 9});
    auto tensor = ttml::core::from_vector(data, shape, device);

    ttml::autograd::clip_tensor_norm_(tensor, 10.F);

    auto clipped_vec = ttml::core::to_vector(tensor);
    auto norm = 0.F;
    for (auto& value : clipped_vec) {
        norm += value * value;
    }
    norm = std::sqrt(norm);
    EXPECT_NEAR(norm, 9.F, 1e-2);
    for (const auto& value : clipped_vec) {
        EXPECT_NEAR(value, -1.F, 1e-2);
    }
}

TEST_F(ClipGradientNormTest, GradNormTensor_2) {
    auto* device = &ttml::autograd::ctx().get_device();

    std::vector<float> data(81, -1.F);
    auto shape = ttml::core::create_shape({1, 1, 9, 9});
    auto tensor = ttml::core::from_vector(data, shape, device);

    ttml::autograd::clip_tensor_norm_(tensor, 1.F);

    auto clipped_vec = ttml::core::to_vector(tensor);
    auto norm = 0.F;
    for (auto& value : clipped_vec) {
        norm += value * value;
    }
    norm = std::sqrt(norm);
    EXPECT_NEAR(norm, 1.F, 1e-2);
    for (const auto& value : clipped_vec) {
        EXPECT_NEAR(value, -1.F / 9.F, 1e-2);
    }
}
