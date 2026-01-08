// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <sys/types.h>

#include <cassert>
#include <core/ttnn_all_includes.hpp>
#include <cstddef>
#include <cstdint>
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>
#include <umd/device/cluster.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>

#include "autograd/auto_context.hpp"
#include "core/random.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/losses.hpp"

class GrpoLossTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
        ttml::autograd::ctx().set_seed(42);
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

// Golden reference values for regression testing
// These values were captured from actual runs and verified to be correct

TEST_F(GrpoLossTest, GrpoLoss_Basic_Test) {
    using namespace ttml;

    const uint32_t N = 2, H = 3;
    const uint32_t vocab_size = 4;

    // Create prediction tensor (batch=2, seq_len=3, vocab=4)
    xt::xarray<float> prediction_data = {
        {{{1.0F, 2.0F, 0.5F, 1.5F}, {2.0F, 1.0F, 3.0F, 0.5F}, {0.5F, 1.5F, 2.5F, 1.0F}}}};

    // Broadcast to (N, 1, H, vocab_size)
    std::array<size_t, 4> shape = {N, 1, H, vocab_size};
    xt::xarray<float> prediction_tensor = xt::broadcast(prediction_data, shape);
    auto prediction = core::from_xtensor(prediction_tensor, &autograd::ctx().get_device());

    // Create target tensor (batch=2, seq_len=3)
    xt::xarray<uint32_t> target_tensor = {{1, 2, 1}, {0, 2, 3}};
    auto target = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    // Create advantages tensor using negative answer length as reward
    // For sequences of length 3, use -3 as advantage (negative length)
    xt::xarray<float> advantages_tensor = {-3.0F, -3.0F};
    auto advantages = core::from_xtensor(advantages_tensor, &autograd::ctx().get_device());

    float normalization_factor = static_cast<float>(N * H);  // 6 tokens total

    auto result = ttml::ops::grpo_loss(
        autograd::create_tensor(prediction),
        autograd::create_tensor(target),
        autograd::create_tensor(advantages),
        normalization_factor);

    // Golden reference value - captured from actual run and verified
    float expected_value = -7.09375F;

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result->get_value());
    float actual_value = result_xtensor(0, 0, 0, 0);
    EXPECT_NEAR(actual_value, expected_value, 1e-3F);
}

TEST_F(GrpoLossTest, GrpoLoss_Different_Advantages) {
    using namespace ttml;

    const uint32_t N = 2, H = 2;
    const uint32_t vocab_size = 3;

    // Create prediction tensor
    xt::xarray<float> prediction_data = {{{{1.0F, 2.0F, 3.0F}, {0.5F, 1.5F, 2.0F}}}};
    std::array<size_t, 4> shape2 = {N, 1, H, vocab_size};
    xt::xarray<float> prediction_tensor = xt::broadcast(prediction_data, shape2);
    auto prediction = core::from_xtensor(prediction_tensor, &autograd::ctx().get_device());

    // Create target tensor
    xt::xarray<uint32_t> target_tensor = {{2, 1}, {0, 2}};
    auto target = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    // Use different negative lengths as advantages (e.g., different answer lengths)
    xt::xarray<float> advantages_tensor = {-2.0F, -4.0F};  // Different sequence lengths
    auto advantages = core::from_xtensor(advantages_tensor, &autograd::ctx().get_device());

    float normalization_factor = static_cast<float>(N * H);

    auto result = ttml::ops::grpo_loss(
        autograd::create_tensor(prediction),
        autograd::create_tensor(target),
        autograd::create_tensor(advantages),
        normalization_factor);

    // Golden reference value - captured from actual run and verified
    float expected_value = -6.78125F;

    auto result_xtensor = core::to_xtensor(result->get_value());
    float actual_value = result_xtensor(0, 0, 0, 0);
    EXPECT_NEAR(actual_value, expected_value, 1e-3F);
}

TEST_F(GrpoLossTest, GrpoLoss_Large_Batch) {
    using namespace ttml;

    const uint32_t N = 4, H = 8;
    const uint32_t vocab_size = 16;

    // Create random prediction tensor
    std::mt19937 gen(42);
    std::array<size_t, 4> shape3 = {N, 1, H, vocab_size};
    xt::xarray<float> prediction_tensor = xt::empty<float>(shape3);
    auto& rng = ttml::autograd::ctx().get_generator();
    uint32_t seed = rng();
    ttml::core::parallel_generate(
        std::span{prediction_tensor.data(), prediction_tensor.size()},
        []() { return std::uniform_real_distribution<float>(-2.0F, 2.0F); },
        seed);
    auto prediction = core::from_xtensor(prediction_tensor, &autograd::ctx().get_device());

    // Create random target tensor
    std::array<size_t, 2> target_shape2 = {N, H};
    xt::xarray<uint32_t> target_tensor = xt::zeros<uint32_t>(target_shape2);
    std::uniform_int_distribution<uint32_t> class_dist(0, vocab_size - 1);
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t h = 0; h < H; ++h) {
            target_tensor(n, h) = class_dist(gen);
        }
    }
    auto target = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    // Use negative answer lengths as advantages
    xt::xarray<float> advantages_tensor = {-5.0F, -3.0F, -7.0F, -4.0F};
    auto advantages = core::from_xtensor(advantages_tensor, &autograd::ctx().get_device());

    float normalization_factor = static_cast<float>(N * H);

    auto result = ttml::ops::grpo_loss(
        autograd::create_tensor(prediction),
        autograd::create_tensor(target),
        autograd::create_tensor(advantages),
        normalization_factor);

    // Golden reference value - captured from actual run and verified
    float expected_value = -65.0F;

    auto result_xtensor = core::to_xtensor(result->get_value());
    float actual_value = result_xtensor(0, 0, 0, 0);
    EXPECT_NEAR(actual_value, expected_value, 1e-2F);
}
