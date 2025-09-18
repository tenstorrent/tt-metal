// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/autocast_tensor.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"

using namespace ttml;

class AutogradTensorTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(AutogradTensorTest, AutogradTensorFLOAT32) {
    auto tensor = autograd::create_tensor(
        ttml::core::ones(ttnn::Shape({1, 1, 1, 32}), &autograd::ctx().get_device(), ttnn::DataType::FLOAT32));
    const auto& half_precision_tensor = tensor->get_value();
    const auto& full_precision_tensor = tensor->get_value(autograd::PreferredPrecision::FULL);

    EXPECT_EQ(half_precision_tensor.dtype(), ttnn::DataType::BFLOAT16);
    EXPECT_EQ(full_precision_tensor.dtype(), ttnn::DataType::FLOAT32);
}

TEST_F(AutogradTensorTest, AutogradTensorBFLOAT16) {
    auto tensor = autograd::create_tensor(
        ttml::core::ones(ttnn::Shape({1, 1, 1, 32}), &autograd::ctx().get_device(), ttnn::DataType::BFLOAT16));
    const auto& half_precision_tensor = tensor->get_value();
    const auto& full_precision_tensor = tensor->get_value(autograd::PreferredPrecision::FULL);

    EXPECT_EQ(half_precision_tensor.dtype(), ttnn::DataType::BFLOAT16);
    EXPECT_EQ(full_precision_tensor.dtype(), ttnn::DataType::BFLOAT16);
}

TEST_F(AutogradTensorTest, AutocastTensor) {
    auto tt_tensor =
        ttml::core::ones(ttnn::Shape({1, 1, 1, 32}), &autograd::ctx().get_device(), ttnn::DataType::FLOAT32);
    auto autocast_tensor = autograd::AutocastTensor(tt_tensor);
    const auto& half_precision_tensor = autocast_tensor.get_tensor();
    const auto& full_precision_tensor = autocast_tensor.get_tensor(autograd::PreferredPrecision::FULL);

    EXPECT_EQ(half_precision_tensor.dtype(), ttnn::DataType::BFLOAT16);
    EXPECT_EQ(full_precision_tensor.dtype(), ttnn::DataType::FLOAT32);
}
