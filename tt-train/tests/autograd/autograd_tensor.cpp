// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

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
    EXPECT_EQ(full_precision_tensor.dtype(), ttnn::DataType::FLOAT32);
}

TEST_F(AutogradTensorTest, AutocastTensorFromFLOAT32) {
    auto tt_tensor =
        ttml::core::ones(ttnn::Shape({1, 1, 1, 32}), &autograd::ctx().get_device(), ttnn::DataType::FLOAT32);
    auto autocast_tensor = autograd::AutocastTensor(tt_tensor);
    const auto& half_precision_tensor = autocast_tensor.get_tensor();
    const auto& full_precision_tensor = autocast_tensor.get_tensor(autograd::PreferredPrecision::FULL);

    EXPECT_EQ(half_precision_tensor.dtype(), ttnn::DataType::BFLOAT16);
    EXPECT_EQ(full_precision_tensor.dtype(), ttnn::DataType::FLOAT32);
}

TEST_F(AutogradTensorTest, AutocastTensorFromBFLOAT16) {
    auto tt_tensor =
        ttml::core::ones(ttnn::Shape({1, 1, 1, 32}), &autograd::ctx().get_device(), ttnn::DataType::BFLOAT16);
    auto autocast_tensor = autograd::AutocastTensor(tt_tensor);
    const auto& half_precision_tensor = autocast_tensor.get_tensor();
    const auto& full_precision_tensor = autocast_tensor.get_tensor(autograd::PreferredPrecision::FULL);

    EXPECT_EQ(half_precision_tensor.dtype(), ttnn::DataType::BFLOAT16);
    EXPECT_EQ(full_precision_tensor.dtype(), ttnn::DataType::FLOAT32);
}

TEST_F(AutogradTensorTest, AutocastTensorLazyHalfNotCreatedWhenUnused) {
    auto tt_tensor =
        ttml::core::ones(ttnn::Shape({1, 1, 1, 32}), &autograd::ctx().get_device(), ttnn::DataType::FLOAT32);
    auto autocast_tensor = autograd::AutocastTensor(tt_tensor);

    // Only request FULL — the BF16 half should not be created
    const auto& full = autocast_tensor.get_tensor(autograd::PreferredPrecision::FULL);
    EXPECT_EQ(full.dtype(), ttnn::DataType::FLOAT32);

    // Now request HALF — it should be lazily created
    const auto& half = autocast_tensor.get_tensor(autograd::PreferredPrecision::HALF);
    EXPECT_EQ(half.dtype(), ttnn::DataType::BFLOAT16);
}

TEST_F(AutogradTensorTest, AutocastTensorLazyFullNotCreatedWhenUnused) {
    auto tt_tensor =
        ttml::core::ones(ttnn::Shape({1, 1, 1, 32}), &autograd::ctx().get_device(), ttnn::DataType::BFLOAT16);
    auto autocast_tensor = autograd::AutocastTensor(tt_tensor);

    // Only request HALF — the FP32 full should not be created
    const auto& half = autocast_tensor.get_tensor(autograd::PreferredPrecision::HALF);
    EXPECT_EQ(half.dtype(), ttnn::DataType::BFLOAT16);

    // Now request FULL — it should be lazily created
    const auto& full = autocast_tensor.get_tensor(autograd::PreferredPrecision::FULL);
    EXPECT_EQ(full.dtype(), ttnn::DataType::FLOAT32);
}

TEST_F(AutogradTensorTest, AutocastTensorSetTensorInvalidatesCache) {
    auto fp32_tensor =
        ttml::core::ones(ttnn::Shape({1, 1, 1, 32}), &autograd::ctx().get_device(), ttnn::DataType::FLOAT32);
    auto autocast_tensor = autograd::AutocastTensor(fp32_tensor);

    // Trigger lazy BF16 creation
    const auto& half = autocast_tensor.get_tensor(autograd::PreferredPrecision::HALF);
    EXPECT_EQ(half.dtype(), ttnn::DataType::BFLOAT16);

    // Overwrite with a BF16 tensor — should invalidate the FP32 cache
    auto bf16_tensor =
        ttml::core::zeros(ttnn::Shape({1, 1, 1, 32}), &autograd::ctx().get_device(), ttnn::DataType::BFLOAT16);
    autocast_tensor.set_tensor(bf16_tensor);

    const auto& new_half = autocast_tensor.get_tensor(autograd::PreferredPrecision::HALF);
    EXPECT_EQ(new_half.dtype(), ttnn::DataType::BFLOAT16);

    // FP32 should be lazily recreated from the new BF16 tensor
    const auto& new_full = autocast_tensor.get_tensor(autograd::PreferredPrecision::FULL);
    EXPECT_EQ(new_full.dtype(), ttnn::DataType::FLOAT32);
}
