// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/losses.hpp"

namespace ttml::ops::tests {

class NllLossTest : public ::testing::Test {
public:
    static void SetUpTestSuite() {
        autograd::ctx().open_device();
    }

    static void TearDownTestSuite() {
        autograd::ctx().close_device();
    }

protected:
    void TearDown() override {
        autograd::ctx().reset_graph();
    }
};

namespace {

constexpr uint32_t kBatch = 2U;
constexpr uint32_t kClasses = 35U;

autograd::TensorPtr make_prediction() {
    std::vector<float> values(kBatch * kClasses, -0.25F);
    values[1] = -1.0F;
    values[kClasses] = -9.0F;
    values[kClasses + 2U] = -3.0F;
    return autograd::create_tensor(
        core::from_vector(values, ttnn::Shape({kBatch, 1U, 1U, kClasses}), &autograd::ctx().get_device()),
        /* requires_grad */ true);
}

template <typename TargetType, ttnn::DataType TargetDtype>
autograd::TensorPtr make_target(const ttnn::Shape& shape, ttnn::Layout layout) {
    return autograd::create_tensor(core::from_vector<TargetType, TargetDtype>(
        std::vector<TargetType>{static_cast<TargetType>(1), static_cast<TargetType>(2)},
        shape,
        &autograd::ctx().get_device(),
        layout));
}

void expect_forward_and_backward(const autograd::TensorPtr& target) {
    auto prediction = make_prediction();
    auto loss = nll_loss(prediction, target);

    const auto loss_values = core::to_vector<float>(loss->get_value());
    ASSERT_EQ(loss_values.size(), 1U);
    EXPECT_NEAR(loss_values[0], 2.0F, 1e-2F);

    loss->backward();
    const auto grad = core::to_vector<float>(prediction->get_grad());
    ASSERT_EQ(grad.size(), kBatch * kClasses);
    for (std::size_t i = 0; i < grad.size(); ++i) {
        const float expected = (i == 1U || i == kClasses + 2U) ? -0.5F : 0.0F;
        EXPECT_NEAR(grad[i], expected, 1e-2F) << "gradient mismatch at flattened index " << i;
    }
}

}  // namespace

TEST_F(NllLossTest, NormalizesCurrentUint32RowMajorTarget) {
    expect_forward_and_backward(
        make_target<uint32_t, ttnn::DataType::UINT32>(ttnn::Shape({kBatch, 1U}), ttnn::Layout::ROW_MAJOR));
}

TEST_F(NllLossTest, RepacksMultidimensionalInt32TileTarget) {
    expect_forward_and_backward(
        make_target<int32_t, ttnn::DataType::INT32>(ttnn::Shape({kBatch, 1U}), ttnn::Layout::TILE));
}

TEST_F(NllLossTest, PreservesLegacyFlatInt32TileTarget) {
    expect_forward_and_backward(make_target<int32_t, ttnn::DataType::INT32>(ttnn::Shape({kBatch}), ttnn::Layout::TILE));
}

TEST_F(NllLossTest, RejectsMalformedContractsBeforeLaunch) {
    auto target = make_target<uint32_t, ttnn::DataType::UINT32>(ttnn::Shape({kBatch, 1U}), ttnn::Layout::ROW_MAJOR);

    auto rank_two_prediction = autograd::create_tensor(core::from_vector(
        std::vector<float>(kBatch * kClasses, -1.0F), ttnn::Shape({kBatch, kClasses}), &autograd::ctx().get_device()));
    EXPECT_THROW(nll_loss(rank_two_prediction, target), std::logic_error);

    auto prediction = make_prediction();
    auto wrong_volume_target = autograd::create_tensor(core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        std::vector<uint32_t>{1U}, ttnn::Shape({1U, 1U}), &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR));
    EXPECT_THROW(nll_loss(prediction, wrong_volume_target), std::logic_error);
}

}  // namespace ttml::ops::tests
