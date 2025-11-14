// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cmath>
#include <core/ttnn_all_includes.hpp>
#include <tuple>

#include "autograd/auto_context.hpp"
#include "core/random.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/ops/layernorm_fw/layernorm_fw.hpp"

// Reference implementation using xtensor
std::tuple<xt::xarray<float>, xt::xarray<float>, xt::xarray<float>> layernorm_forward_reference_(
    const xt::xarray<float>& x,
    const xt::xarray<float>& gamma,
    const xt::xarray<float>& beta,
    uint32_t batch_size,
    uint32_t features,
    float eps = 1e-6f) {
    // Reshape input to (batch_size, features) for easier manipulation
    auto x_reshaped = xt::reshape_view(x, {batch_size, features});

    // Compute mean along features axis
    xt::xarray<float> mu = xt::mean(x_reshaped, {1});

    // Compute variance along features axis
    xt::xarray<float> x_centered = x_reshaped - xt::view(mu, xt::all(), xt::newaxis());
    xt::xarray<float> var = xt::mean(xt::square(x_centered), {1});

    // Compute reciprocal standard deviation (rstd)
    xt::xarray<float> rstd = 1.0f / xt::sqrt(var + eps);

    // Normalize
    xt::xarray<float> x_hat = x_centered * xt::view(rstd, xt::all(), xt::newaxis());

    // Scale and shift
    xt::xarray<float> y = x_hat * xt::view(gamma, xt::newaxis(), xt::all()) + xt::view(beta, xt::newaxis(), xt::all());

    // Flatten outputs back to 1D
    y = xt::flatten(y);

    return std::make_tuple(y, mu, rstd);
}

class LayerNormForwardOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

// Helper function to compare metal kernel results against xtensor reference
static void CompareKernelVsXArray(
    uint32_t batch_size,
    const uint32_t seq_len,
    const uint32_t heads,
    const uint32_t features,
    const int num_iterations = 3) {
    using namespace ttml;

    for (int iter = 0; iter < num_iterations; iter++) {
        // Generate test data using xtensor
        uint32_t total_elements = batch_size * seq_len * heads * features;
        uint32_t combined_batch = batch_size * seq_len * heads;

        xt::xarray<float> x_data = xt::empty<float>({total_elements});
        auto rng = autograd::ctx().get_generator();
        uint32_t seed1 = rng();
        core::parallel_generate<float>(
            x_data, []() { return std::uniform_real_distribution<float>(-1.0F, 1.0F); }, seed1);

        xt::xarray<float> gamma_data = xt::empty<float>({features});
        uint32_t seed2 = rng();
        core::parallel_generate<float>(
            gamma_data, []() { return std::uniform_real_distribution<float>(0.5F, 1.5F); }, seed2);

        xt::xarray<float> beta_data = xt::empty<float>({features});
        uint32_t seed3 = rng();
        core::parallel_generate<float>(
            beta_data, []() { return std::uniform_real_distribution<float>(-0.1F, 0.1F); }, seed3);

        // Compute reference results
        auto [y_ref, mu_ref, rstd_ref] =
            layernorm_forward_reference_(x_data, gamma_data, beta_data, combined_batch, features, 1e-6f);

        // Copy and reshape data to 4D for device tensors (copy to avoid corrupting reference data)
        xt::xarray<float> x_4d = x_data;
        x_4d.reshape({batch_size, heads, seq_len, features});
        xt::xarray<float> gamma_4d = gamma_data;
        gamma_4d.reshape({1, 1, 1, features});
        xt::xarray<float> beta_4d = beta_data;
        beta_4d.reshape({1, 1, 1, features});

        // Create tensors on device using from_xtensor
        auto input_tensor = core::from_xtensor(x_4d, &autograd::ctx().get_device());
        auto gamma_tensor = core::from_xtensor(gamma_4d, &autograd::ctx().get_device());
        auto beta_tensor = core::from_xtensor(beta_4d, &autograd::ctx().get_device());

        // Run metal kernel
        auto output_tensors = metal::ops::layernorm_fw::LayerNormForwardOperation::invoke(
            input_tensor, gamma_tensor, beta_tensor, 1e-6f, /* return_mean_rstd */ true);

        auto metal_y_xtensor = core::to_xtensor(output_tensors[0].value());
        auto metal_mu_xtensor = core::to_xtensor(output_tensors[1].value());
        auto metal_rstd_xtensor = core::to_xtensor(output_tensors[2].value());

        // Flatten metal results for comparison
        xt::xarray<float> metal_y_flat = xt::flatten(metal_y_xtensor);
        xt::xarray<float> metal_mu_flat = xt::flatten(metal_mu_xtensor);
        xt::xarray<float> metal_rstd_flat = xt::flatten(metal_rstd_xtensor);

        // Compare shapes
        ASSERT_EQ(y_ref.shape(), metal_y_flat.shape());
        ASSERT_EQ(mu_ref.shape(), metal_mu_flat.shape());
        ASSERT_EQ(rstd_ref.shape(), metal_rstd_flat.shape());

        // Compare values
        EXPECT_TRUE(xt::allclose(metal_y_flat, y_ref, 1.0e-3F, 5e-2F));
        EXPECT_TRUE(xt::allclose(metal_mu_flat, mu_ref, 1.0e-3F, 5e-2F));
        EXPECT_TRUE(xt::allclose(metal_rstd_flat, rstd_ref, 1.0e-3F, 5e-2F));
    }
}

TEST_F(LayerNormForwardOpTest, MetalLayerNormFw_OneTile) {
    CompareKernelVsXArray(1, 32, 1, 32);
}

TEST_F(LayerNormForwardOpTest, MetalLayerNormFw_OneIncompleteTile) {
    CompareKernelVsXArray(1, 12, 1, 19);
}

TEST_F(LayerNormForwardOpTest, NIGHTLY_MetalLayerNormFw_MediumTensorFitsInL1) {
    CompareKernelVsXArray(2, 182, 1, 2083);
}

TEST_F(LayerNormForwardOpTest, NIGHTLY_MetalLayerNormFw_LargeTensor_DoesNotFitInL1) {
    CompareKernelVsXArray(4, 324, 1, 9132);
}

TEST_F(LayerNormForwardOpTest, MetalLayerNormFw_HeadsDimNot1) {
    CompareKernelVsXArray(2, 8, 4, 512);
}
