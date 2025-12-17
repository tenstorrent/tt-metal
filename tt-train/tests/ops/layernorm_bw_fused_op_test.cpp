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
#include "metal/ops/layernorm_bw/layernorm_bw.hpp"

// Reference implementation using xtensor
struct LayerNormCache {
    xt::xarray<float> x;
    xt::xarray<float> x_hat;
    xt::xarray<float> mu;
    xt::xarray<float> s;
    xt::xarray<float> gamma;
    xt::xarray<float> beta;
    uint32_t batch_size;
    uint32_t features;
    float eps;
};

std::tuple<xt::xarray<float>, LayerNormCache> layernorm_forward_reference(
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

    // Compute standard deviation with epsilon
    xt::xarray<float> s = xt::sqrt(var + eps);

    // Normalize
    xt::xarray<float> x_hat = x_centered / xt::view(s, xt::all(), xt::newaxis());

    // Scale and shift
    xt::xarray<float> y = x_hat * xt::view(gamma, xt::newaxis(), xt::all()) + xt::view(beta, xt::newaxis(), xt::all());

    // Flatten outputs back to 1D
    y = xt::flatten(y);
    x_hat = xt::flatten(x_hat);

    LayerNormCache cache{x, x_hat, mu, s, gamma, beta, batch_size, features, eps};
    return std::make_tuple(y, cache);
}

std::tuple<xt::xarray<float>, xt::xarray<float>, xt::xarray<float>> layernorm_backward_reference(
    const xt::xarray<float>& dy, const LayerNormCache& cache) {
    // Reshape to (batch_size, features)
    auto dy_reshaped = xt::reshape_view(dy, {cache.batch_size, cache.features});
    auto x_hat_reshaped = xt::reshape_view(cache.x_hat, {cache.batch_size, cache.features});

    // Compute dgamma and dbeta - sum over batch dimension
    xt::xarray<float> dgamma = xt::sum(dy_reshaped * x_hat_reshaped, {0});
    xt::xarray<float> dbeta = xt::sum(dy_reshaped, {0});

    // Compute dxhat = dy * gamma
    xt::xarray<float> dxhat = dy_reshaped * xt::view(cache.gamma, xt::newaxis(), xt::all());

    // Compute mean_dxhat - mean along features axis
    xt::xarray<float> mean_dxhat = xt::mean(dxhat, {1});

    // Compute mean_dxhat_xhat - mean along features axis
    xt::xarray<float> mean_dxhat_xhat = xt::mean(dxhat * x_hat_reshaped, {1});

    // Compute final dx
    xt::xarray<float> dx = (dxhat - xt::view(mean_dxhat, xt::all(), xt::newaxis()) -
                            x_hat_reshaped * xt::view(mean_dxhat_xhat, xt::all(), xt::newaxis())) /
                           xt::view(cache.s, xt::all(), xt::newaxis());

    // Flatten dx back to 1D
    dx = xt::flatten(dx);

    return std::make_tuple(dx, dgamma, dbeta);
}

class LayerNormBackwardOpTest : public ::testing::Test {
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
    const uint32_t batch_size,
    const uint32_t seq_len,
    const uint32_t heads,
    const uint32_t features,
    const int num_iterations = 3) {
    using namespace ttml;

    for (int iter = 0; iter < num_iterations; iter++) {
        // Generate test data using flattened 1D arrays
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
            gamma_data, []() { return std::uniform_real_distribution<float>(0.0F, 1.0F); }, seed2);

        xt::xarray<float> beta_data = xt::empty<float>({features});
        uint32_t seed3 = rng();
        core::parallel_generate<float>(
            beta_data, []() { return std::uniform_real_distribution<float>(0.0F, 1.0F); }, seed3);

        xt::xarray<float> dy_data = xt::empty<float>({total_elements});
        uint32_t seed4 = rng();
        core::parallel_generate<float>(
            dy_data, []() { return std::uniform_real_distribution<float>(-1.0F, 1.0F); }, seed4);

        // Compute reference results
        auto [y_ref, cache] =
            layernorm_forward_reference(x_data, gamma_data, beta_data, combined_batch, features, 1e-6f);
        auto [dx_ref, dgamma_ref, dbeta_ref] = layernorm_backward_reference(dy_data, cache);

        // Copy and reshape data to 4D for device tensors (copy to avoid corrupting reference data)
        xt::xarray<float> x_4d = x_data;
        x_4d.reshape({batch_size, heads, seq_len, features});
        xt::xarray<float> gamma_4d = gamma_data;
        gamma_4d.reshape({1, 1, 1, features});
        xt::xarray<float> dy_4d = dy_data;
        dy_4d.reshape({batch_size, heads, seq_len, features});
        xt::xarray<float> mu_4d = cache.mu;
        mu_4d.reshape({batch_size, heads, seq_len, 1});

        // Compute rstd from s and reshape
        xt::xarray<float> rstd_data = 1.0f / cache.s;
        rstd_data.reshape({batch_size, heads, seq_len, 1});

        // Create tensors on device using from_xtensor
        auto input_tensor = core::from_xtensor(x_4d, &autograd::ctx().get_device());
        auto gamma_tensor = core::from_xtensor(gamma_4d, &autograd::ctx().get_device());
        auto mean_tensor = core::from_xtensor(mu_4d, &autograd::ctx().get_device());
        auto rstd_tensor = core::from_xtensor(rstd_data, &autograd::ctx().get_device());
        auto dy_tensor = core::from_xtensor(dy_4d, &autograd::ctx().get_device());

        auto output_tensors = metal::ops::layernorm_bw::LayerNormBackwardOperation::invoke(
            input_tensor, gamma_tensor, mean_tensor, rstd_tensor, dy_tensor);

        auto metal_dx_xtensor = core::to_xtensor(output_tensors[0].value());
        auto metal_dgamma_xtensor = core::to_xtensor(output_tensors[1].value());
        auto metal_dbeta_xtensor = core::to_xtensor(output_tensors[2].value());

        // Flatten metal results for comparison
        xt::xarray<float> metal_dx_flat = xt::flatten(metal_dx_xtensor);
        xt::xarray<float> metal_dgamma_flat = xt::flatten(metal_dgamma_xtensor);
        xt::xarray<float> metal_dbeta_flat = xt::flatten(metal_dbeta_xtensor);

        // Compare shapes
        ASSERT_EQ(dx_ref.shape(), metal_dx_flat.shape());
        ASSERT_EQ(dgamma_ref.shape(), metal_dgamma_flat.shape());
        ASSERT_EQ(dbeta_ref.shape(), metal_dbeta_flat.shape());

        // Compare values
        EXPECT_TRUE(xt::allclose(metal_dx_flat, dx_ref, 1.0e-3F, 5e-1F));
        EXPECT_TRUE(xt::allclose(metal_dgamma_flat, dgamma_ref, 1.0e-3F, 5e-1F));
        EXPECT_TRUE(xt::allclose(metal_dbeta_flat, dbeta_ref, 1.0e-3F, 5e-1F));
    }
}

// ============================================================================
// Test Cases - LayerNorm Backward Metal Kernel vs XArray Reference
// ============================================================================

TEST_F(LayerNormBackwardOpTest, MetalLayerNormBw_OneTile) {
    CompareKernelVsXArray(1, 13, 1, 20);
}

TEST_F(LayerNormBackwardOpTest, MetalLayerNormBw_TwoIncompleteTiles) {
    CompareKernelVsXArray(1, 32, 1, 33);
}

TEST_F(LayerNormBackwardOpTest, NIGHTLY_MetalLayerNormBw_LargeFeatures_NoL1Fit) {
    CompareKernelVsXArray(3, 273, 1, 8462);
}

TEST_F(LayerNormBackwardOpTest, MetalLayerNormBw_DoesNotFitInL1_WtNotDivisibleBy4) {
    CompareKernelVsXArray(3, 100, 1, 8191, 10);
}

TEST_F(LayerNormBackwardOpTest, MetalLayerNormBw_OneTilePerRow) {
    CompareKernelVsXArray(1, 19, 1, 213, 10);
}

// ============================================================================
// Bug Reproduction Tests - Expose accumulation bug in compute_dy_gamma_sum
// ============================================================================
//
// These tests use deterministic (constant) inputs instead of random data
// to definitively expose the accumulation bug in the block-based code path.
//
// Bug description:
// In layernorm_bw_kernel.cpp:compute_dy_gamma_sum(), only the last tile uses
// add_binary_tile for accumulation. For tiles 0 to Wt-2, mul_tiles_bcast_rows
// overwrites sum_register instead of accumulating. This causes ~99% data loss
// for large tensors.
//
// These tests use:
// 1. Constant positive values (not random mean-zero data)
// 2. Tight tolerance (atol=0.01 instead of 0.5)
// 3. Many tiles to maximize error visibility
// ============================================================================

// Helper function using deterministic inputs to expose accumulation bug
static void CompareKernelVsXArrayDeterministic(
    const uint32_t batch_size,
    const uint32_t seq_len,
    const uint32_t heads,
    const uint32_t features,
    const float dy_value,     // Constant value for dy (use non-zero)
    const float gamma_value,  // Constant value for gamma (use non-zero)
    const float x_value,      // Constant value for x
    const float rtol = 1.0e-2F,
    const float atol = 1.0e-2F) {
    using namespace ttml;

    uint32_t total_elements = batch_size * seq_len * heads * features;
    uint32_t combined_batch = batch_size * seq_len * heads;

    // Create deterministic test data with constant values
    xt::xarray<float> x_data = xt::ones<float>({total_elements}) * x_value;
    xt::xarray<float> gamma_data = xt::ones<float>({features}) * gamma_value;
    xt::xarray<float> beta_data = xt::zeros<float>({features});
    xt::xarray<float> dy_data = xt::ones<float>({total_elements}) * dy_value;

    // Compute reference results
    auto [y_ref, cache] = layernorm_forward_reference(x_data, gamma_data, beta_data, combined_batch, features, 1e-6f);
    auto [dx_ref, dgamma_ref, dbeta_ref] = layernorm_backward_reference(dy_data, cache);

    // Copy and reshape data to 4D for device tensors
    xt::xarray<float> x_4d = x_data;
    x_4d.reshape({batch_size, heads, seq_len, features});
    xt::xarray<float> gamma_4d = gamma_data;
    gamma_4d.reshape({1, 1, 1, features});
    xt::xarray<float> dy_4d = dy_data;
    dy_4d.reshape({batch_size, heads, seq_len, features});
    xt::xarray<float> mu_4d = cache.mu;
    mu_4d.reshape({batch_size, heads, seq_len, 1});

    // Compute rstd from s and reshape
    xt::xarray<float> rstd_data = 1.0f / cache.s;
    rstd_data.reshape({batch_size, heads, seq_len, 1});

    // Create tensors on device
    auto input_tensor = core::from_xtensor(x_4d, &autograd::ctx().get_device());
    auto gamma_tensor = core::from_xtensor(gamma_4d, &autograd::ctx().get_device());
    auto mean_tensor = core::from_xtensor(mu_4d, &autograd::ctx().get_device());
    auto rstd_tensor = core::from_xtensor(rstd_data, &autograd::ctx().get_device());
    auto dy_tensor = core::from_xtensor(dy_4d, &autograd::ctx().get_device());

    auto output_tensors = metal::ops::layernorm_bw::LayerNormBackwardOperation::invoke(
        input_tensor, gamma_tensor, mean_tensor, rstd_tensor, dy_tensor);

    auto metal_dx_xtensor = core::to_xtensor(output_tensors[0].value());
    auto metal_dgamma_xtensor = core::to_xtensor(output_tensors[1].value());
    auto metal_dbeta_xtensor = core::to_xtensor(output_tensors[2].value());

    // Flatten metal results for comparison
    xt::xarray<float> metal_dx_flat = xt::flatten(metal_dx_xtensor);
    xt::xarray<float> metal_dgamma_flat = xt::flatten(metal_dgamma_xtensor);
    xt::xarray<float> metal_dbeta_flat = xt::flatten(metal_dbeta_xtensor);

    // Compare shapes
    ASSERT_EQ(dx_ref.shape(), metal_dx_flat.shape());
    ASSERT_EQ(dgamma_ref.shape(), metal_dgamma_flat.shape());
    ASSERT_EQ(dbeta_ref.shape(), metal_dbeta_flat.shape());

    // Compare values with TIGHT tolerance
    // These will FAIL if the accumulation bug exists
    bool dx_close = xt::allclose(metal_dx_flat, dx_ref, rtol, atol);
    bool dgamma_close = xt::allclose(metal_dgamma_flat, dgamma_ref, rtol, atol);
    bool dbeta_close = xt::allclose(metal_dbeta_flat, dbeta_ref, rtol, atol);

    // Print detailed error info on failure
    if (!dx_close) {
        float max_diff = xt::amax(xt::abs(metal_dx_flat - dx_ref))();
        float mean_diff = xt::mean(xt::abs(metal_dx_flat - dx_ref))();
        std::cout << "dx FAILED: max_diff=" << max_diff << ", mean_diff=" << mean_diff << std::endl;
    }
    if (!dgamma_close) {
        float max_diff = xt::amax(xt::abs(metal_dgamma_flat - dgamma_ref))();
        float mean_diff = xt::mean(xt::abs(metal_dgamma_flat - dgamma_ref))();
        std::cout << "dgamma FAILED: max_diff=" << max_diff << ", mean_diff=" << mean_diff << std::endl;
    }
    if (!dbeta_close) {
        float max_diff = xt::amax(xt::abs(metal_dbeta_flat - dbeta_ref))();
        float mean_diff = xt::mean(xt::abs(metal_dbeta_flat - dbeta_ref))();
        std::cout << "dbeta FAILED: max_diff=" << max_diff << ", mean_diff=" << mean_diff << std::endl;
    }

    EXPECT_TRUE(dx_close) << "dx mismatch with deterministic inputs (features=" << features << ")";
    EXPECT_TRUE(dgamma_close) << "dgamma mismatch with deterministic inputs (features=" << features << ")";
    EXPECT_TRUE(dbeta_close) << "dbeta mismatch with deterministic inputs (features=" << features << ")";
}

// Test with 256 tiles (8192 features) - block-based path, exposes accumulation bug
// With constant dy=1.0, gamma=1.0: correct sum = 8192, buggy sum ≈ 64
TEST_F(LayerNormBackwardOpTest, BugRepro_Deterministic_256Tiles) {
    // 8192 features = 256 tiles, uses block-based path (doesn't fit in L1)
    // dy=1.0, gamma=1.0, x=0.5
    CompareKernelVsXArrayDeterministic(1, 100, 1, 8192, 1.0f, 1.0f, 0.5f);
}

// Test with 128 tiles (4096 features) - smaller but still exposes bug
TEST_F(LayerNormBackwardOpTest, BugRepro_Deterministic_128Tiles) {
    // 4096 features = 128 tiles
    CompareKernelVsXArrayDeterministic(1, 100, 1, 4096, 1.0f, 1.0f, 0.5f);
}

// Test with different constant values to ensure bug is not data-dependent
TEST_F(LayerNormBackwardOpTest, BugRepro_Deterministic_DifferentValues) {
    // Use different constant values: dy=0.5, gamma=2.0, x=1.0
    CompareKernelVsXArrayDeterministic(1, 100, 1, 8192, 0.5f, 2.0f, 1.0f);
}

// Test matching the original NIGHTLY test parameters but with deterministic inputs
TEST_F(LayerNormBackwardOpTest, BugRepro_Deterministic_8462Features) {
    // Same as NIGHTLY_MetalLayerNormBw_LargeFeatures_NoL1Fit but deterministic
    // 8462 features = 265 tiles
    CompareKernelVsXArrayDeterministic(3, 273, 1, 8462, 1.0f, 1.0f, 0.5f);
}

// Test with smaller feature count that still uses block-based path
TEST_F(LayerNormBackwardOpTest, BugRepro_Deterministic_2048Features) {
    // 2048 features = 64 tiles, may or may not fit in L1 depending on config
    CompareKernelVsXArrayDeterministic(1, 100, 1, 2048, 1.0f, 1.0f, 0.5f);
}

// ============================================================================
// Tight Tolerance Tests - Same as original but with strict tolerance
// ============================================================================
// These use random data but with tight tolerance (atol=0.01 instead of 0.5)
// They may be flaky but will catch gross errors from accumulation bug

static void CompareKernelVsXArrayTightTolerance(
    const uint32_t batch_size,
    const uint32_t seq_len,
    const uint32_t heads,
    const uint32_t features,
    const int num_iterations = 1) {
    using namespace ttml;

    for (int iter = 0; iter < num_iterations; iter++) {
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
            gamma_data, []() { return std::uniform_real_distribution<float>(0.0F, 1.0F); }, seed2);

        xt::xarray<float> beta_data = xt::empty<float>({features});
        uint32_t seed3 = rng();
        core::parallel_generate<float>(
            beta_data, []() { return std::uniform_real_distribution<float>(0.0F, 1.0F); }, seed3);

        xt::xarray<float> dy_data = xt::empty<float>({total_elements});
        uint32_t seed4 = rng();
        core::parallel_generate<float>(
            dy_data, []() { return std::uniform_real_distribution<float>(-1.0F, 1.0F); }, seed4);

        auto [y_ref, cache] =
            layernorm_forward_reference(x_data, gamma_data, beta_data, combined_batch, features, 1e-6f);
        auto [dx_ref, dgamma_ref, dbeta_ref] = layernorm_backward_reference(dy_data, cache);

        xt::xarray<float> x_4d = x_data;
        x_4d.reshape({batch_size, heads, seq_len, features});
        xt::xarray<float> gamma_4d = gamma_data;
        gamma_4d.reshape({1, 1, 1, features});
        xt::xarray<float> dy_4d = dy_data;
        dy_4d.reshape({batch_size, heads, seq_len, features});
        xt::xarray<float> mu_4d = cache.mu;
        mu_4d.reshape({batch_size, heads, seq_len, 1});

        xt::xarray<float> rstd_data = 1.0f / cache.s;
        rstd_data.reshape({batch_size, heads, seq_len, 1});

        auto input_tensor = core::from_xtensor(x_4d, &autograd::ctx().get_device());
        auto gamma_tensor = core::from_xtensor(gamma_4d, &autograd::ctx().get_device());
        auto mean_tensor = core::from_xtensor(mu_4d, &autograd::ctx().get_device());
        auto rstd_tensor = core::from_xtensor(rstd_data, &autograd::ctx().get_device());
        auto dy_tensor = core::from_xtensor(dy_4d, &autograd::ctx().get_device());

        auto output_tensors = metal::ops::layernorm_bw::LayerNormBackwardOperation::invoke(
            input_tensor, gamma_tensor, mean_tensor, rstd_tensor, dy_tensor);

        auto metal_dx_xtensor = core::to_xtensor(output_tensors[0].value());
        auto metal_dgamma_xtensor = core::to_xtensor(output_tensors[1].value());
        auto metal_dbeta_xtensor = core::to_xtensor(output_tensors[2].value());

        xt::xarray<float> metal_dx_flat = xt::flatten(metal_dx_xtensor);
        xt::xarray<float> metal_dgamma_flat = xt::flatten(metal_dgamma_xtensor);
        xt::xarray<float> metal_dbeta_flat = xt::flatten(metal_dbeta_xtensor);

        ASSERT_EQ(dx_ref.shape(), metal_dx_flat.shape());
        ASSERT_EQ(dgamma_ref.shape(), metal_dgamma_flat.shape());
        ASSERT_EQ(dbeta_ref.shape(), metal_dbeta_flat.shape());

        // TIGHT tolerance: rtol=0.01, atol=0.01 (instead of rtol=0.001, atol=0.5)
        EXPECT_TRUE(xt::allclose(metal_dx_flat, dx_ref, 1.0e-2F, 1.0e-2F))
            << "dx failed with tight tolerance (iter=" << iter << ")";
        EXPECT_TRUE(xt::allclose(metal_dgamma_flat, dgamma_ref, 1.0e-2F, 1.0e-2F))
            << "dgamma failed with tight tolerance (iter=" << iter << ")";
        EXPECT_TRUE(xt::allclose(metal_dbeta_flat, dbeta_ref, 1.0e-2F, 1.0e-2F))
            << "dbeta failed with tight tolerance (iter=" << iter << ")";
    }
}

// Tight tolerance version of the NIGHTLY test
TEST_F(LayerNormBackwardOpTest, BugRepro_TightTolerance_8462Features) {
    CompareKernelVsXArrayTightTolerance(3, 273, 1, 8462, 1);
}

// Tight tolerance with 8192 features
TEST_F(LayerNormBackwardOpTest, BugRepro_TightTolerance_8192Features) {
    CompareKernelVsXArrayTightTolerance(1, 100, 1, 8192, 1);
}
