// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cmath>
#include <core/ttnn_all_includes.hpp>
#include <random>
#include <tuple>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/ops/layernorm_bw/layernorm_bw.hpp"
#include "ops/layernorm_op.hpp"

// Reference implementation using standard C++
struct LayerNormCache {
    std::vector<float> x;
    std::vector<float> x_hat;
    std::vector<float> mu;
    std::vector<float> s;
    std::vector<float> gamma;
    std::vector<float> beta;
    uint32_t batch_size;
    uint32_t features;
    float eps;
};

std::tuple<std::vector<float>, LayerNormCache> layernorm_forward_reference(
    const std::vector<float>& x,
    const std::vector<float>& gamma,
    const std::vector<float>& beta,
    uint32_t batch_size,
    uint32_t features,
    float eps = 1e-5f) {
    std::vector<float> y(x.size());
    std::vector<float> x_hat(x.size());
    std::vector<float> mu(batch_size);
    std::vector<float> s(batch_size);

    // For each batch element
    for (uint32_t b = 0; b < batch_size; ++b) {
        // Compute mean
        float mean = 0.0f;
        for (uint32_t d = 0; d < features; ++d) {
            mean += x[b * features + d];
        }
        mean /= features;
        mu[b] = mean;

        // Compute variance
        float var = 0.0f;
        for (uint32_t d = 0; d < features; ++d) {
            float centered = x[b * features + d] - mean;
            var += centered * centered;
        }
        var /= features;

        float std_dev = std::sqrt(var + eps);
        s[b] = std_dev;

        // Normalize and scale
        for (uint32_t d = 0; d < features; ++d) {
            float x_normalized = (x[b * features + d] - mean) / std_dev;
            x_hat[b * features + d] = x_normalized;
            y[b * features + d] = x_normalized * gamma[d] + beta[d];
        }
    }

    LayerNormCache cache{x, x_hat, mu, s, gamma, beta, batch_size, features, eps};
    return std::make_tuple(y, cache);
}

std::tuple<std::vector<float>, std::vector<float>, std::vector<float>> layernorm_backward_reference(
    const std::vector<float>& dy, const LayerNormCache& cache) {
    std::vector<float> dx(dy.size());
    std::vector<float> dgamma(cache.features, 0.0f);
    std::vector<float> dbeta(cache.features, 0.0f);

    // Compute dgamma and dbeta
    for (uint32_t b = 0; b < cache.batch_size; ++b) {
        for (uint32_t d = 0; d < cache.features; ++d) {
            dgamma[d] += dy[b * cache.features + d] * cache.x_hat[b * cache.features + d];
            // std::cout << "dgamma[d]: " << dy[b * cache.features + d] << " "<< cache.x_hat[b * cache.features + d] <<
            // " " << dy[b * cache.features + d] * cache.x_hat[b * cache.features + d] << std::endl;
            dbeta[d] += dy[b * cache.features + d];
        }
    }

    // Compute dx
    for (uint32_t b = 0; b < cache.batch_size; ++b) {
        // Compute dxhat = dy * gamma
        std::vector<float> dxhat(cache.features);
        for (uint32_t d = 0; d < cache.features; ++d) {
            dxhat[d] = dy[b * cache.features + d] * cache.gamma[d];
        }

        // Compute mean_dxhat
        float mean_dxhat = 0.0f;
        for (uint32_t d = 0; d < cache.features; ++d) {
            mean_dxhat += dxhat[d];
        }
        mean_dxhat /= cache.features;

        // Compute mean_dxhat_xhat
        float mean_dxhat_xhat = 0.0f;
        for (uint32_t d = 0; d < cache.features; ++d) {
            mean_dxhat_xhat += dxhat[d] * cache.x_hat[b * cache.features + d];
        }
        mean_dxhat_xhat /= cache.features;
        // Compute final dx
        for (uint32_t d = 0; d < cache.features; ++d) {
            dx[b * cache.features + d] =
                (1.0f / cache.s[b]) * (dxhat[d] - mean_dxhat - cache.x_hat[b * cache.features + d] * mean_dxhat_xhat);
        }
    }

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

    // Get reference output using xtensor implementation
    uint32_t combined_batch = batch_size * seq_len * heads;
    auto [y_ref, cache] =
        layernorm_forward_reference(test_data, gamma_data, beta_data, combined_batch, features, 1e-5f);

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
        float error = std::abs(composite_output[i] - y_ref[i]);
        max_error = std::max(max_error, error);

        if (error > tolerance) {
            mismatches++;
            if (mismatches <= 10) {  // Print first 10 mismatches
                std::cout << "Mismatch at index " << i << ": composite=" << composite_output[i]
                          << ", reference=" << y_ref[i] << ", error=" << error << std::endl;
            }
        }

        EXPECT_NEAR(composite_output[i], y_ref[i], tolerance) << "Output mismatch at index " << i;
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
        auto [y_ref, cache] =
            layernorm_forward_reference(test_data, gamma_data, beta_data, combined_batch, features, 1e-5f);
        auto [dx_ref, dgamma_ref, dbeta_ref] = layernorm_backward_reference(dy_data, cache);

        auto input_tensor = core::from_vector(
            test_data, ttnn::Shape({batch_size, heads, seq_len, features}), &autograd::ctx().get_device());
        auto gamma_tensor =
            core::from_vector(gamma_data, ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device());
        auto mean_tensor =
            core::from_vector(cache.mu, ttnn::Shape({batch_size, heads, seq_len, 1}), &autograd::ctx().get_device());

        std::vector<float> rstd_data;
        for (uint32_t b = 0; b < combined_batch; ++b) {
            rstd_data.push_back(1.0f / cache.s[b]);
        }
        auto rstd_tensor =
            core::from_vector(rstd_data, ttnn::Shape({batch_size, heads, seq_len, 1}), &autograd::ctx().get_device());
        auto dy_tensor = core::from_vector(
            dy_data, ttnn::Shape({batch_size, heads, seq_len, features}), &autograd::ctx().get_device());

        auto output_tensors = metal::ops::layernorm_bw::LayerNormBackwardOperation::invoke(
            input_tensor, gamma_tensor, mean_tensor, rstd_tensor, dy_tensor);

        auto metal_dx = core::to_vector(output_tensors[0].value());
        auto metal_dgamma = core::to_vector(output_tensors[1].value());

        float tolerance = 5e-2f;

        for (uint32_t i = 0; i < total_elements; ++i) {
            EXPECT_NEAR(metal_dx[i], dx_ref[i], tolerance) << "Input gradient mismatch at index " << i;
        }

        float max_error = 0.0f;
        for (uint32_t i = 0; i < features; ++i) {
            if (std::abs(metal_dgamma[i] - dgamma_ref[i]) > tolerance) {
                max_error = std::max(max_error, std::abs(metal_dgamma[i] - dgamma_ref[i]));
                std::cout << "Gamma gradient mismatch at index " << i << ": metal=" << metal_dgamma[i]
                          << ", reference=" << dgamma_ref[i] << ", error=" << std::abs(metal_dgamma[i] - dgamma_ref[i])
                          << std::endl;
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
        auto [y_ref, cache] =
            layernorm_forward_reference(test_data, gamma_data, beta_data, combined_batch, features, 1e-5f);
        auto [dx_ref, dgamma_ref, dbeta_ref] = layernorm_backward_reference(dy_data, cache);

        auto input_tensor = core::from_vector(
            test_data, ttnn::Shape({batch_size, heads, seq_len, features}), &autograd::ctx().get_device());
        auto gamma_tensor =
            core::from_vector(gamma_data, ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device());
        auto mean_tensor = core::from_vector(
            cache.x_hat, ttnn::Shape({batch_size, heads, seq_len, features}), &autograd::ctx().get_device());

        std::vector<float> rstd_data;
        for (uint32_t b = 0; b < combined_batch; ++b) {
            rstd_data.push_back(1.0f / cache.s[b]);
        }
        auto rstd_tensor =
            core::from_vector(rstd_data, ttnn::Shape({batch_size, heads, seq_len, 1}), &autograd::ctx().get_device());
        auto dy_tensor = core::from_vector(
            dy_data, ttnn::Shape({batch_size, heads, seq_len, features}), &autograd::ctx().get_device());

        auto output_tensors = metal::ops::layernorm_bw::LayerNormBackwardOperation::invoke(
            input_tensor, gamma_tensor, mean_tensor, rstd_tensor, dy_tensor);

        auto metal_dx = core::to_vector(output_tensors[0].value());
        auto metal_dgamma = core::to_vector(output_tensors[1].value());

        float tolerance = 5e-2f;

        for (uint32_t i = 0; i < total_elements; ++i) {
            EXPECT_NEAR(metal_dx[i], dx_ref[i], tolerance) << "Input gradient mismatch at index " << i;
        }

        float max_error = 0.0f;
        for (uint32_t i = 0; i < features; ++i) {
            if (std::abs(metal_dgamma[i] - dgamma_ref[i]) > tolerance) {
                max_error = std::max(max_error, std::abs(metal_dgamma[i] - dgamma_ref[i]));
                std::cout << "Gamma gradient mismatch at index " << i << ": metal=" << metal_dgamma[i]
                          << ", reference=" << dgamma_ref[i] << ", error=" << std::abs(metal_dgamma[i] - dgamma_ref[i])
                          << std::endl;
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
        auto [y_ref, cache] =
            layernorm_forward_reference(test_data, gamma_data, beta_data, combined_batch, features, 1e-5f);
        auto [dx_ref, dgamma_ref, dbeta_ref] = layernorm_backward_reference(dy_data, cache);

        auto input_tensor = core::from_vector(
            test_data, ttnn::Shape({batch_size, heads, seq_len, features}), &autograd::ctx().get_device());
        auto gamma_tensor =
            core::from_vector(gamma_data, ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device());
        auto mean_tensor = core::from_vector(
            cache.x_hat, ttnn::Shape({batch_size, heads, seq_len, features}), &autograd::ctx().get_device());

        std::vector<float> rstd_data;
        for (uint32_t b = 0; b < combined_batch; ++b) {
            rstd_data.push_back(1.0f / cache.s[b]);
        }
        auto rstd_tensor =
            core::from_vector(rstd_data, ttnn::Shape({batch_size, heads, seq_len, 1}), &autograd::ctx().get_device());
        auto dy_tensor = core::from_vector(
            dy_data, ttnn::Shape({batch_size, heads, seq_len, features}), &autograd::ctx().get_device());

        auto output_tensors = metal::ops::layernorm_bw::LayerNormBackwardOperation::invoke(
            input_tensor, gamma_tensor, mean_tensor, rstd_tensor, dy_tensor);

        auto metal_dx = core::to_vector(output_tensors[0].value());
        auto metal_dgamma = core::to_vector(output_tensors[1].value());

        float tolerance = 5e-2f;

        for (uint32_t i = 0; i < total_elements; ++i) {
            EXPECT_NEAR(metal_dx[i], dx_ref[i], tolerance) << "Input gradient mismatch at index " << i;
        }
        float max_error = 0.0f;
        for (uint32_t i = 0; i < features; ++i) {
            if (std::abs(metal_dgamma[i] - dgamma_ref[i]) > tolerance) {
                max_error = std::max(max_error, std::abs(metal_dgamma[i] - dgamma_ref[i]));
                std::cout << "Gamma gradient mismatch at index " << i << ": metal=" << metal_dgamma[i]
                          << ", reference=" << dgamma_ref[i] << ", error=" << std::abs(metal_dgamma[i] - dgamma_ref[i])
                          << std::endl;
            }
        }
        std::cout << "Test completed. Max error: " << max_error << std::endl;
    }
}
