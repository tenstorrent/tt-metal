// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "autograd/clip_gradient_norm.hpp"

#include <gtest/gtest.h>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"

TEST(ClipGradientNormTest, GradNormTensor_0) {
    auto* device = &ttml::autograd::ctx().get_device();

    std::vector<float> data(81, -1.F);
    auto shape = ttml::core::create_shape({1, 1, 9, 9});
    auto tensor = ttml::core::from_vector(data, shape, device);

    std::vector<tt::tt_metal::Tensor> tensors = {tensor};
    ttml::autograd::clip_tensor_norm_(tensors, 3.F, 2.F);

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

TEST(ClipGradientNormTest, GradNormTensor_1) {
    auto* device = &ttml::autograd::ctx().get_device();

    std::vector<float> data(81, -1.F);
    auto shape = ttml::core::create_shape({1, 1, 9, 9});
    auto tensor = ttml::core::from_vector(data, shape, device);

    std::vector<tt::tt_metal::Tensor> tensors = {tensor};
    ttml::autograd::clip_tensor_norm_(tensors, 10.F, 2.F);

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

TEST(ClipGradientNormTest, GradNormTensor_2) {
    auto* device = &ttml::autograd::ctx().get_device();

    std::vector<float> data(81, -1.F);
    auto shape = ttml::core::create_shape({1, 1, 9, 9});
    auto tensor = ttml::core::from_vector(data, shape, device);

    std::vector<tt::tt_metal::Tensor> tensors = {tensor};
    ttml::autograd::clip_tensor_norm_(tensors, 1.F, 2.F);

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
