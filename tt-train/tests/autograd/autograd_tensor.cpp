// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>

#include "autograd/auto_context.hpp"
#include "autograd/autocast_tensor.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "optimizers/adamw.hpp"
#include "optimizers/sgd.hpp"
#include "test_utils/random_data.hpp"

using namespace ttml;

class AutogradTensorTest : public ::testing::Test {
public:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }

    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }
};

namespace {

// Fused optimizers update the parameter in place. A FULL view read before the step must show the
// updated values afterwards.
template <typename Optimizer, typename Config>
void expect_full_view_tracks_fused_step(const Config& config) {
    const std::array<std::size_t, 4> shape = {1, 1, 32, 32};
    autograd::ctx().set_seed(123U);
    auto& gen = autograd::ctx().get_generator();
    const xt::xarray<float> w0 = test_utils::make_uniform_xarray<float>(shape, -1.0F, 1.0F, gen());
    const xt::xarray<float> g0 = test_utils::make_uniform_xarray<float>(shape, 0.25F, 1.0F, gen());

    auto* device = &autograd::ctx().get_device();
    auto theta = autograd::create_tensor(core::from_xtensor(w0, device), /* requires_grad */ true);
    ASSERT_EQ(theta->get_value(autograd::PreferredPrecision::NATIVE).dtype(), ttnn::DataType::BFLOAT16);

    const auto full_before = core::to_xtensor(theta->get_value(autograd::PreferredPrecision::FULL));

    theta->set_grad(core::from_xtensor(g0, device));
    Optimizer optimizer(serialization::NamedParameters{{"theta", theta}}, config);
    optimizer.step();

    const auto half_after = core::to_xtensor(theta->get_value(autograd::PreferredPrecision::HALF));
    ASSERT_FALSE(xt::allclose(half_after, full_before, 0.0, 0.0)) << "the step did not change the parameter";

    // Exact comparison: bf16 -> fp32 is lossless, so the FULL view must match bit for bit.
    const auto full_after = core::to_xtensor(theta->get_value(autograd::PreferredPrecision::FULL));
    EXPECT_TRUE(xt::allclose(full_after, half_after, 0.0, 0.0)) << "FULL view is stale after an in-place step";
}

}  // namespace

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

    EXPECT_TRUE(autocast_tensor.has_full());
    EXPECT_FALSE(autocast_tensor.has_half());

    const auto& full = autocast_tensor.get_tensor(autograd::PreferredPrecision::FULL);
    EXPECT_EQ(full.dtype(), ttnn::DataType::FLOAT32);
    EXPECT_FALSE(autocast_tensor.has_half());

    const auto& half = autocast_tensor.get_tensor(autograd::PreferredPrecision::HALF);
    EXPECT_EQ(half.dtype(), ttnn::DataType::BFLOAT16);
    EXPECT_TRUE(autocast_tensor.has_half());
}

TEST_F(AutogradTensorTest, AutocastTensorFromBFLOAT16) {
    auto tt_tensor =
        ttml::core::ones(ttnn::Shape({1, 1, 1, 32}), &autograd::ctx().get_device(), ttnn::DataType::BFLOAT16);
    auto autocast_tensor = autograd::AutocastTensor(tt_tensor);

    EXPECT_TRUE(autocast_tensor.has_half());
    EXPECT_FALSE(autocast_tensor.has_full());

    const auto& half = autocast_tensor.get_tensor(autograd::PreferredPrecision::HALF);
    EXPECT_EQ(half.dtype(), ttnn::DataType::BFLOAT16);
    EXPECT_FALSE(autocast_tensor.has_full());

    const auto& full = autocast_tensor.get_tensor(autograd::PreferredPrecision::FULL);
    EXPECT_EQ(full.dtype(), ttnn::DataType::FLOAT32);
    EXPECT_TRUE(autocast_tensor.has_full());
}

TEST_F(AutogradTensorTest, AutocastTensorSetTensorInvalidatesCache) {
    auto fp32_tensor =
        ttml::core::ones(ttnn::Shape({1, 1, 1, 32}), &autograd::ctx().get_device(), ttnn::DataType::FLOAT32);
    auto autocast_tensor = autograd::AutocastTensor(fp32_tensor);

    EXPECT_TRUE(autocast_tensor.has_full());
    EXPECT_FALSE(autocast_tensor.has_half());

    [[maybe_unused]] const auto& half = autocast_tensor.get_tensor(autograd::PreferredPrecision::HALF);
    EXPECT_TRUE(autocast_tensor.has_full());
    EXPECT_TRUE(autocast_tensor.has_half());

    auto bf16_tensor =
        ttml::core::zeros(ttnn::Shape({1, 1, 1, 32}), &autograd::ctx().get_device(), ttnn::DataType::BFLOAT16);
    autocast_tensor.set_tensor(bf16_tensor);

    EXPECT_TRUE(autocast_tensor.has_half());
    EXPECT_FALSE(autocast_tensor.has_full());

    [[maybe_unused]] const auto& full = autocast_tensor.get_tensor(autograd::PreferredPrecision::FULL);
    EXPECT_TRUE(autocast_tensor.has_half());
    EXPECT_TRUE(autocast_tensor.has_full());
}

// Disabled: fused optimizers leave a cached FULL view stale — https://github.com/tenstorrent/tt-metal/issues/41657
TEST_F(AutogradTensorTest, DISABLED_FullViewTracksFusedAdamWStep) {
    optimizers::AdamWConfig config;
    config.lr = 1e-2F;
    expect_full_view_tracks_fused_step<optimizers::AdamW>(config);
}

// Disabled: fused optimizers leave a cached FULL view stale — https://github.com/tenstorrent/tt-metal/issues/41657
TEST_F(AutogradTensorTest, DISABLED_FullViewTracksFusedSGDStep) {
    optimizers::SGDConfig config;
    config.lr = 1e-1F;
    expect_full_view_tracks_fused_step<optimizers::SGD>(config);
}
