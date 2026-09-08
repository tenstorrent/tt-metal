// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <tt-metalium/constants.hpp>
#include <tuple>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/ops/layernorm_fw/layernorm_fw.hpp"
#include "test_utils/random_data.hpp"

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
public:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }

    static void TearDownTestSuite() {
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

        auto& rng = autograd::ctx().get_generator();
        uint32_t seed1 = rng();
        xt::xarray<float> x_data =
            test_utils::make_uniform_xarray<float>(std::array<std::size_t, 1>{total_elements}, -1.0F, 1.0F, seed1);

        uint32_t seed2 = rng();
        xt::xarray<float> gamma_data =
            test_utils::make_uniform_xarray<float>(std::array<std::size_t, 1>{features}, 0.5F, 1.5F, seed2);

        uint32_t seed3 = rng();
        xt::xarray<float> beta_data =
            test_utils::make_uniform_xarray<float>(std::array<std::size_t, 1>{features}, -0.1F, 0.1F, seed3);

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
        auto output_tensors =
            metal::layernorm_fw(input_tensor, gamma_tensor, beta_tensor, 1e-6f, /* return_mean_rstd */ true);

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

static void CompareMixedStatsFormatsAcrossMultipleRows() {
    using namespace ttml;

    auto& device = autograd::ctx().get_device();
    const auto grid = device.compute_with_storage_grid_size();
    const uint32_t tile_rows = static_cast<uint32_t>(grid.x * grid.y) + 1U;
    const uint32_t batch_size = 1U;
    const uint32_t seq_len = tile_rows * tt::constants::TILE_HEIGHT;
    const uint32_t heads = 1U;
    const uint32_t features = 64U;
    const uint32_t total_elements = batch_size * seq_len * heads * features;

    xt::xarray<float> x_data = xt::zeros<float>(std::array<std::size_t, 1>{total_elements});
    xt::xarray<float> gamma_data = xt::zeros<float>(std::array<std::size_t, 1>{features});
    xt::xarray<float> beta_data = xt::zeros<float>(std::array<std::size_t, 1>{features});

    for (uint32_t i = 0; i < total_elements; ++i) {
        x_data.data()[i] = (i % 2U == 0U) ? 0.0F : 2.0F;
    }
    for (uint32_t i = 0; i < features; ++i) {
        gamma_data.data()[i] = 0.5F + static_cast<float>(i % 8U) * 0.125F;
        beta_data.data()[i] = -0.25F + static_cast<float>(i % 4U) * 0.125F;
    }

    auto [y_ref, mu_ref, rstd_ref] =
        layernorm_forward_reference_(x_data, gamma_data, beta_data, seq_len, features, 1e-6F);

    x_data.reshape({batch_size, heads, seq_len, features});
    gamma_data.reshape({1U, 1U, 1U, features});
    beta_data.reshape({1U, 1U, 1U, features});

    auto input_tensor = core::from_xtensor(x_data, &device);
    auto gamma_tensor = core::from_xtensor(gamma_data, &device);
    auto beta_tensor = core::from_xtensor(beta_data, &device);

    auto output_tensors =
        metal::layernorm_fw(input_tensor, gamma_tensor, beta_tensor, 1e-6F, /* return_mean_rstd */ true);
    auto metal_y = xt::flatten(core::to_xtensor(output_tensors[0].value()));
    auto metal_mu = xt::flatten(core::to_xtensor(output_tensors[1].value()));
    auto metal_rstd = xt::flatten(core::to_xtensor(output_tensors[2].value()));

    ASSERT_EQ(metal_y.shape(), y_ref.shape());
    ASSERT_EQ(metal_mu.shape(), mu_ref.shape());
    ASSERT_EQ(metal_rstd.shape(), rstd_ref.shape());

    EXPECT_TRUE(xt::allclose(metal_y, y_ref, 1.0e-3F, 2.0e-2F))
        << "y max_abs_diff=" << xt::amax(xt::abs(metal_y - y_ref))();
    EXPECT_TRUE(xt::allclose(metal_mu, mu_ref, 1.0e-3F, 1.0e-2F))
        << "mean max_abs_diff=" << xt::amax(xt::abs(metal_mu - mu_ref))();
    EXPECT_TRUE(xt::allclose(metal_rstd, rstd_ref, 1.0e-3F, 1.0e-2F))
        << "rstd max_abs_diff=" << xt::amax(xt::abs(metal_rstd - rstd_ref))();
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

TEST_F(LayerNormForwardOpTest, MetalLayerNormFw_MixedStatsFormatsAcrossMultipleRows) {
    CompareMixedStatsFormatsAcrossMultipleRows();
}
