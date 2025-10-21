// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cmath>
#include <core/ttnn_all_includes.hpp>
#include <random>
#include <ttnn/tensor/xtensor/xtensor_all_includes.hpp>
#include <tuple>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/ops/layernorm_bw/layernorm_bw.hpp"
#include "ops/layernorm_op.hpp"

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
    float eps = 1e-5f) {
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

class LayerNormFusedOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(LayerNormFusedOpTest, CompositeLayerNormForward_AgainstXTensor) {
    using namespace ttml;

    uint32_t batch_size = 2;
    uint32_t seq_len = 64;
    uint32_t heads = 1;
    uint32_t features = 512;

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    uint32_t total_elements = batch_size * seq_len * heads * features;

    // Generate test data
    std::vector<float> test_data;
    for (uint32_t i = 0; i < total_elements; ++i) {
        test_data.push_back(dist(gen));
    }

    std::vector<float> gamma_data;
    std::vector<float> beta_data;
    std::normal_distribution<float> param_dist(1.0f, 0.2f);
    for (uint32_t i = 0; i < features; ++i) {
        gamma_data.push_back(param_dist(gen));
        beta_data.push_back(dist(gen) * 0.1f);
    }

    // Convert to xtensor and get reference output using xtensor implementation
    uint32_t combined_batch = batch_size * seq_len * heads;
    auto x_xtensor = xt::adapt(test_data, {combined_batch * features});
    auto gamma_xtensor = xt::adapt(gamma_data, {features});
    auto beta_xtensor = xt::adapt(beta_data, {features});

    auto [y_ref, cache] =
        layernorm_forward_reference(x_xtensor, gamma_xtensor, beta_xtensor, combined_batch, features, 1e-5f);

    // Create tensors for composite operation
    auto input_tensor = core::from_vector(
        test_data, ttnn::Shape({batch_size, heads, seq_len, features}), &autograd::ctx().get_device());
    auto gamma_tensor = core::from_vector(gamma_data, ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device());
    auto beta_tensor = core::from_vector(beta_data, ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device());

    // Call composite layernorm forward
    auto output_tensor = ops::composite_layernorm(
        autograd::create_tensor(input_tensor),
        autograd::create_tensor(gamma_tensor),
        autograd::create_tensor(beta_tensor));

    // Convert to host and compare
    auto composite_output = core::to_vector(output_tensor->get_value());

    float tolerance = 1e-1f;
    uint32_t mismatches = 0;
    float max_error = 0.0f;

    for (uint32_t i = 0; i < total_elements; ++i) {
        float error = std::abs(composite_output[i] - y_ref.data()[i]);
        max_error = std::max(max_error, error);

        if (error > tolerance) {
            mismatches++;
            if (mismatches <= 10) {  // Print first 10 mismatches
                std::cout << "Mismatch at index " << i << ": composite=" << composite_output[i]
                          << ", reference=" << y_ref.data()[i] << ", error=" << error << std::endl;
            }
        }

        EXPECT_NEAR(composite_output[i], y_ref.data()[i], tolerance) << "Output mismatch at index " << i;
    }

    std::cout << "Test completed. Max error: " << max_error << ", Total mismatches: " << mismatches << "/"
              << total_elements << std::endl;
}

TEST_F(LayerNormFusedOpTest, MetalLayerNormBw_LargeFeatures_NoL1Fit) {
    using namespace ttml;

    uint32_t batch_size = 3;  // Increased to provide sufficient work
    uint32_t seq_len = 273;   // Must be divisible by 32 (tile size)
    uint32_t heads = 1;
    uint32_t features = 8462;  // Must be divisible by 32 (tile size)

    std::mt19937 gen(1213);
    std::normal_distribution<float> dist(0.0f, 2.0f);  // Larger variance

    uint32_t total_elements = batch_size * seq_len * heads * features;
    for (int i = 0; i < 3; i++) {
        std::vector<float> test_data;
        for (uint32_t i = 0; i < total_elements; ++i) {
            test_data.push_back(dist(gen));
        }

        std::vector<float> gamma_data;
        std::vector<float> beta_data;
        std::normal_distribution<float> param_dist(1.0f, 0.2f);
        for (uint32_t i = 0; i < features; ++i) {
            gamma_data.push_back(param_dist(gen));
            beta_data.push_back(dist(gen) * 0.1f);
        }

        std::vector<float> dy_data;
        std::normal_distribution<float> grad_dist(0.0f, 0.2f);
        for (uint32_t i = 0; i < total_elements; ++i) {
            dy_data.push_back(grad_dist(gen));
        }

        uint32_t combined_batch = batch_size * seq_len * heads;
        auto x_xtensor = xt::adapt(test_data, {combined_batch * features});
        auto gamma_xtensor = xt::adapt(gamma_data, {features});
        auto beta_xtensor = xt::adapt(beta_data, {features});
        auto dy_xtensor = xt::adapt(dy_data, {combined_batch * features});

        auto [y_ref, cache] =
            layernorm_forward_reference(x_xtensor, gamma_xtensor, beta_xtensor, combined_batch, features, 1e-5f);
        auto [dx_ref, dgamma_ref, dbeta_ref] = layernorm_backward_reference(dy_xtensor, cache);

        auto input_tensor = core::from_vector(
            test_data, ttnn::Shape({batch_size, heads, seq_len, features}), &autograd::ctx().get_device());
        auto gamma_tensor =
            core::from_vector(gamma_data, ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device());

        std::vector<float> mu_data(cache.mu.data(), cache.mu.data() + cache.mu.size());
        auto mean_tensor =
            core::from_vector(mu_data, ttnn::Shape({batch_size, heads, seq_len, 1}), &autograd::ctx().get_device());

        std::vector<float> rstd_data;
        for (uint32_t b = 0; b < combined_batch; ++b) {
            rstd_data.push_back(1.0f / cache.s.data()[b]);
        }
        auto rstd_tensor =
            core::from_vector(rstd_data, ttnn::Shape({batch_size, heads, seq_len, 1}), &autograd::ctx().get_device());
        auto dy_tensor = core::from_vector(
            dy_data, ttnn::Shape({batch_size, heads, seq_len, features}), &autograd::ctx().get_device());

        auto output_tensors = metal::ops::layernorm_bw::LayerNormBackwardOperation::invoke(
            input_tensor, gamma_tensor, mean_tensor, rstd_tensor, dy_tensor);

        auto metal_dx = core::to_vector(output_tensors[0].value());
        auto metal_dgamma = core::to_vector(output_tensors[1].value());

        float tolerance = 1e-1f;

        for (uint32_t i = 0; i < total_elements; ++i) {
            EXPECT_NEAR(metal_dx[i], dx_ref.data()[i], tolerance) << "Input gradient mismatch at index " << i;
        }

        float max_error = 0.0f;
        for (uint32_t i = 0; i < features; ++i) {
            if (std::abs(metal_dgamma[i] - dgamma_ref.data()[i]) > tolerance) {
                max_error = std::max(max_error, std::abs(metal_dgamma[i] - dgamma_ref.data()[i]));
                std::cout << "Gamma gradient mismatch at index " << i << ": metal=" << metal_dgamma[i]
                          << ", reference=" << dgamma_ref.data()[i]
                          << ", error=" << std::abs(metal_dgamma[i] - dgamma_ref.data()[i]) << std::endl;
            }
        }
        std::cout << "Test completed. Max error: " << max_error << std::endl;
    }
}

TEST_F(LayerNormFusedOpTest, MetalLayerNormBw_Everything_Small_L1Fit) {
    using namespace ttml;

    uint32_t batch_size = 3;  // Increased to provide sufficient work
    uint32_t seq_len = 100;   // Must be divisible by 32 (tile size)
    uint32_t heads = 1;
    uint32_t features = 324;  // Must be divisible by 32 (tile size)

    std::mt19937 gen(1213);
    std::normal_distribution<float> dist(0.0f, 2.0f);  // Larger variance

    uint32_t total_elements = batch_size * seq_len * heads * features;
    for (int i = 0; i < 10; i++) {
        std::vector<float> test_data;
        for (uint32_t i = 0; i < total_elements; ++i) {
            test_data.push_back(dist(gen));
        }

        std::vector<float> gamma_data;
        std::vector<float> beta_data;
        std::normal_distribution<float> param_dist(1.0f, 0.2f);
        for (uint32_t i = 0; i < features; ++i) {
            gamma_data.push_back(param_dist(gen));
            beta_data.push_back(dist(gen) * 0.1f);
        }

        std::vector<float> dy_data;
        std::normal_distribution<float> grad_dist(0.0f, 0.2f);
        for (uint32_t i = 0; i < total_elements; ++i) {
            dy_data.push_back(grad_dist(gen));
        }

        uint32_t combined_batch = batch_size * seq_len * heads;
        auto x_xtensor = xt::adapt(test_data, {combined_batch * features});
        auto gamma_xtensor = xt::adapt(gamma_data, {features});
        auto beta_xtensor = xt::adapt(beta_data, {features});
        auto dy_xtensor = xt::adapt(dy_data, {combined_batch * features});

        auto [y_ref, cache] =
            layernorm_forward_reference(x_xtensor, gamma_xtensor, beta_xtensor, combined_batch, features, 1e-5f);
        auto [dx_ref, dgamma_ref, dbeta_ref] = layernorm_backward_reference(dy_xtensor, cache);

        auto input_tensor = core::from_vector(
            test_data, ttnn::Shape({batch_size, heads, seq_len, features}), &autograd::ctx().get_device());
        auto gamma_tensor =
            core::from_vector(gamma_data, ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device());

        std::vector<float> x_hat_data(cache.x_hat.data(), cache.x_hat.data() + cache.x_hat.size());
        auto mean_tensor = core::from_vector(
            x_hat_data, ttnn::Shape({batch_size, heads, seq_len, features}), &autograd::ctx().get_device());

        std::vector<float> rstd_data;
        for (uint32_t b = 0; b < combined_batch; ++b) {
            rstd_data.push_back(1.0f / cache.s.data()[b]);
        }
        auto rstd_tensor =
            core::from_vector(rstd_data, ttnn::Shape({batch_size, heads, seq_len, 1}), &autograd::ctx().get_device());
        auto dy_tensor = core::from_vector(
            dy_data, ttnn::Shape({batch_size, heads, seq_len, features}), &autograd::ctx().get_device());

        auto output_tensors = metal::ops::layernorm_bw::LayerNormBackwardOperation::invoke(
            input_tensor, gamma_tensor, mean_tensor, rstd_tensor, dy_tensor);

        auto metal_dx = core::to_vector(output_tensors[0].value());
        auto metal_dgamma = core::to_vector(output_tensors[1].value());

        float tolerance = 1e-1f;

        for (uint32_t i = 0; i < total_elements; ++i) {
            EXPECT_NEAR(metal_dx[i], dx_ref.data()[i], tolerance) << "Input gradient mismatch at index " << i;
        }

        float max_error = 0.0f;
        for (uint32_t i = 0; i < features; ++i) {
            if (std::abs(metal_dgamma[i] - dgamma_ref.data()[i]) > tolerance) {
                max_error = std::max(max_error, std::abs(metal_dgamma[i] - dgamma_ref.data()[i]));
                std::cout << "Gamma gradient mismatch at index " << i << ": metal=" << metal_dgamma[i]
                          << ", reference=" << dgamma_ref.data()[i]
                          << ", error=" << std::abs(metal_dgamma[i] - dgamma_ref.data()[i]) << std::endl;
            }
        }
        std::cout << "Test completed. Max error: " << max_error << std::endl;
    }
}

TEST_F(LayerNormFusedOpTest, MetalLayerNormBw_One_Tile_Row) {
    using namespace ttml;

    uint32_t batch_size = 1;  // Increased to provide sufficient work
    uint32_t seq_len = 19;    // Must be divisible by 32 (tile size)
    uint32_t heads = 1;
    uint32_t features = 213;  // Must be divisible by 32 (tile size)

    std::mt19937 gen(1213);
    std::normal_distribution<float> dist(0.0f, 2.0f);  // Larger variance

    uint32_t total_elements = batch_size * seq_len * heads * features;
    for (int i = 0; i < 10; i++) {
        std::vector<float> test_data;
        for (uint32_t i = 0; i < total_elements; ++i) {
            test_data.push_back(dist(gen));
        }

        std::vector<float> gamma_data;
        std::vector<float> beta_data;
        std::normal_distribution<float> param_dist(1.0f, 0.2f);
        for (uint32_t i = 0; i < features; ++i) {
            gamma_data.push_back(param_dist(gen));
            beta_data.push_back(dist(gen) * 0.1f);
        }

        std::vector<float> dy_data;
        std::normal_distribution<float> grad_dist(0.0f, 0.2f);
        for (uint32_t i = 0; i < total_elements; ++i) {
            dy_data.push_back(grad_dist(gen));
        }

        uint32_t combined_batch = batch_size * seq_len * heads;
        auto x_xtensor = xt::adapt(test_data, {combined_batch * features});
        auto gamma_xtensor = xt::adapt(gamma_data, {features});
        auto beta_xtensor = xt::adapt(beta_data, {features});
        auto dy_xtensor = xt::adapt(dy_data, {combined_batch * features});

        auto [y_ref, cache] =
            layernorm_forward_reference(x_xtensor, gamma_xtensor, beta_xtensor, combined_batch, features, 1e-5f);
        auto [dx_ref, dgamma_ref, dbeta_ref] = layernorm_backward_reference(dy_xtensor, cache);

        auto input_tensor = core::from_vector(
            test_data, ttnn::Shape({batch_size, heads, seq_len, features}), &autograd::ctx().get_device());
        auto gamma_tensor =
            core::from_vector(gamma_data, ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device());

        std::vector<float> x_hat_data(cache.x_hat.data(), cache.x_hat.data() + cache.x_hat.size());
        auto mean_tensor = core::from_vector(
            x_hat_data, ttnn::Shape({batch_size, heads, seq_len, features}), &autograd::ctx().get_device());

        std::vector<float> rstd_data;
        for (uint32_t b = 0; b < combined_batch; ++b) {
            rstd_data.push_back(1.0f / cache.s.data()[b]);
        }
        auto rstd_tensor =
            core::from_vector(rstd_data, ttnn::Shape({batch_size, heads, seq_len, 1}), &autograd::ctx().get_device());
        auto dy_tensor = core::from_vector(
            dy_data, ttnn::Shape({batch_size, heads, seq_len, features}), &autograd::ctx().get_device());

        auto output_tensors = metal::ops::layernorm_bw::LayerNormBackwardOperation::invoke(
            input_tensor, gamma_tensor, mean_tensor, rstd_tensor, dy_tensor);

        auto metal_dx = core::to_vector(output_tensors[0].value());
        auto metal_dgamma = core::to_vector(output_tensors[1].value());

        float tolerance = 1e-1f;

        for (uint32_t i = 0; i < total_elements; ++i) {
            EXPECT_NEAR(metal_dx[i], dx_ref.data()[i], tolerance) << "Input gradient mismatch at index " << i;
        }
        float max_error = 0.0f;
        for (uint32_t i = 0; i < features; ++i) {
            if (std::abs(metal_dgamma[i] - dgamma_ref.data()[i]) > tolerance) {
                max_error = std::max(max_error, std::abs(metal_dgamma[i] - dgamma_ref.data()[i]));
                std::cout << "Gamma gradient mismatch at index " << i << ": metal=" << metal_dgamma[i]
                          << ", reference=" << dgamma_ref.data()[i]
                          << ", error=" << std::abs(metal_dgamma[i] - dgamma_ref.data()[i]) << std::endl;
            }
        }
        std::cout << "Test completed. Max error: " << max_error << std::endl;
    }
}
