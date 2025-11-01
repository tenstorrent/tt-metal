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

TEST_F(LayerNormFusedOpTest, MetalLayerNormBw_One_Tile) {
    using namespace ttml;

    uint32_t batch_size = 1;
    uint32_t seq_len = 13;
    uint32_t heads = 1;
    uint32_t features = 20;

    std::cout << "\n=== Test: MetalLayerNormBw_LargeFeatures_NoL1Fit ===" << std::endl;
    std::cout << "Hyperparameters:" << std::endl;
    std::cout << "  batch_size: " << batch_size << std::endl;
    std::cout << "  seq_len: " << seq_len << std::endl;
    std::cout << "  heads: " << heads << std::endl;
    std::cout << "  features: " << features << std::endl;
    std::cout << "  eps: 1e-5" << std::endl;
    std::cout << "  tolerance: 1e-1" << std::endl;
    std::cout << "  total_elements: " << (batch_size * seq_len * heads * features) << std::endl;
    std::cout << "  iterations: 3" << std::endl;

    std::mt19937 gen(1213);
    std::normal_distribution<float> dist(0.0f, 10.0f);  // Larger variance

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
        auto metal_dbeta = core::to_vector(output_tensors[2].value());

        float tolerance = 1e-1f;

        // Test dx accuracy
        uint32_t dx_mismatches = 0;
        float max_dx_error = 0.0f;
        float dx_sum_error = 0.0f;

        std::cout << "\n=== Testing dx accuracy ===" << std::endl;
        for (uint32_t i = 0; i < total_elements; ++i) {
            float error = std::abs(metal_dx[i] - dx_ref.data()[i]);
            max_dx_error = std::max(max_dx_error, error);
            dx_sum_error += error;

            if (error > tolerance) {
                dx_mismatches++;
                if (dx_mismatches <= 5) {  // Print first 5 mismatches
                    std::cout << "dx mismatch at index " << i << ": computed=" << metal_dx[i]
                              << ", reference=" << dx_ref.data()[i] << ", error=" << error << std::endl;
                }
            }
        }

        float dx_avg_error = dx_sum_error / total_elements;
        float dx_variance = 0.0f;
        for (uint32_t i = 0; i < total_elements; ++i) {
            float error = std::abs(metal_dx[i] - dx_ref.data()[i]);
            float diff = error - dx_avg_error;
            dx_variance += diff * diff;
        }
        dx_variance /= total_elements;

        std::cout << "dx: Max error: " << max_dx_error << ", Avg error: " << dx_avg_error
                  << ", Variance: " << dx_variance << ", Total mismatches: " << dx_mismatches << "/" << total_elements
                  << std::endl;

        // Test dgamma accuracy
        uint32_t dgamma_mismatches = 0;
        float max_dgamma_error = 0.0f;
        float dgamma_sum_error = 0.0f;

        std::cout << "\n=== Testing dgamma accuracy ===" << std::endl;
        for (uint32_t i = 0; i < features; ++i) {
            float error = std::abs(metal_dgamma[i] - dgamma_ref.data()[i]);
            max_dgamma_error = std::max(max_dgamma_error, error);
            dgamma_sum_error += error;

            if (error > tolerance) {
                dgamma_mismatches++;
                if (dgamma_mismatches <= 5) {  // Print first 5 mismatches
                    std::cout << "dgamma mismatch at index " << i << ": computed=" << metal_dgamma[i]
                              << ", reference=" << dgamma_ref.data()[i] << ", error=" << error << std::endl;
                }
            }
        }

        float dgamma_avg_error = dgamma_sum_error / features;
        float dgamma_variance = 0.0f;
        for (uint32_t i = 0; i < features; ++i) {
            float error = std::abs(metal_dgamma[i] - dgamma_ref.data()[i]);
            float diff = error - dgamma_avg_error;
            dgamma_variance += diff * diff;
        }
        dgamma_variance /= features;

        std::cout << "dgamma: Max error: " << max_dgamma_error << ", Avg error: " << dgamma_avg_error
                  << ", Variance: " << dgamma_variance << ", Total mismatches: " << dgamma_mismatches << "/" << features
                  << std::endl;

        // Test dbeta accuracy
        uint32_t dbeta_mismatches = 0;
        float max_dbeta_error = 0.0f;
        float dbeta_sum_error = 0.0f;

        std::cout << "\n=== Testing dbeta accuracy ===" << std::endl;
        for (uint32_t i = 0; i < features; ++i) {
            float error = std::abs(metal_dbeta[i] - dbeta_ref.data()[i]);
            max_dbeta_error = std::max(max_dbeta_error, error);
            dbeta_sum_error += error;

            if (error > tolerance) {
                dbeta_mismatches++;
                if (dbeta_mismatches <= 5) {  // Print first 5 mismatches
                    std::cout << "dbeta mismatch at index " << i << ": computed=" << metal_dbeta[i]
                              << ", reference=" << dbeta_ref.data()[i] << ", error=" << error << std::endl;
                }
            }
        }

        float dbeta_avg_error = dbeta_sum_error / features;
        float dbeta_variance = 0.0f;
        for (uint32_t i = 0; i < features; ++i) {
            float error = std::abs(metal_dbeta[i] - dbeta_ref.data()[i]);
            float diff = error - dbeta_avg_error;
            dbeta_variance += diff * diff;
        }
        dbeta_variance /= features;

        std::cout << "dbeta: Max error: " << max_dbeta_error << ", Avg error: " << dbeta_avg_error
                  << ", Variance: " << dbeta_variance << ", Total mismatches: " << dbeta_mismatches << "/" << features
                  << std::endl;
    }
}

TEST_F(LayerNormFusedOpTest, MetalLayerNormBw_Two_Incomplete_Tiles) {
    using namespace ttml;

    uint32_t batch_size = 1;
    uint32_t seq_len = 32;
    uint32_t heads = 1;
    uint32_t features = 33;

    std::cout << "\n=== Test: MetalLayerNormBw_LargeFeatures_NoL1Fit ===" << std::endl;
    std::cout << "Hyperparameters:" << std::endl;
    std::cout << "  batch_size: " << batch_size << std::endl;
    std::cout << "  seq_len: " << seq_len << std::endl;
    std::cout << "  heads: " << heads << std::endl;
    std::cout << "  features: " << features << std::endl;
    std::cout << "  eps: 1e-5" << std::endl;
    std::cout << "  tolerance: 1e-1" << std::endl;
    std::cout << "  total_elements: " << (batch_size * seq_len * heads * features) << std::endl;
    std::cout << "  iterations: 3" << std::endl;

    std::mt19937 gen(1213);
    std::normal_distribution<float> dist(0.0f, 10.0f);  // Larger variance

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
        auto metal_dbeta = core::to_vector(output_tensors[2].value());

        float tolerance = 1e-1f;

        // Test dx accuracy
        uint32_t dx_mismatches = 0;
        float max_dx_error = 0.0f;
        float dx_sum_error = 0.0f;

        std::cout << "\n=== Testing dx accuracy ===" << std::endl;
        for (uint32_t i = 0; i < total_elements; ++i) {
            float error = std::abs(metal_dx[i] - dx_ref.data()[i]);
            max_dx_error = std::max(max_dx_error, error);
            dx_sum_error += error;

            if (error > tolerance) {
                dx_mismatches++;
                if (dx_mismatches <= 5) {  // Print first 5 mismatches
                    std::cout << "dx mismatch at index " << i << ": computed=" << metal_dx[i]
                              << ", reference=" << dx_ref.data()[i] << ", error=" << error << std::endl;
                }
            }
        }

        float dx_avg_error = dx_sum_error / total_elements;
        float dx_variance = 0.0f;
        for (uint32_t i = 0; i < total_elements; ++i) {
            float error = std::abs(metal_dx[i] - dx_ref.data()[i]);
            float diff = error - dx_avg_error;
            dx_variance += diff * diff;
        }
        dx_variance /= total_elements;

        std::cout << "dx: Max error: " << max_dx_error << ", Avg error: " << dx_avg_error
                  << ", Variance: " << dx_variance << ", Total mismatches: " << dx_mismatches << "/" << total_elements
                  << std::endl;

        // Test dgamma accuracy
        uint32_t dgamma_mismatches = 0;
        float max_dgamma_error = 0.0f;
        float dgamma_sum_error = 0.0f;

        std::cout << "\n=== Testing dgamma accuracy ===" << std::endl;
        for (uint32_t i = 0; i < features; ++i) {
            float error = std::abs(metal_dgamma[i] - dgamma_ref.data()[i]);
            max_dgamma_error = std::max(max_dgamma_error, error);
            dgamma_sum_error += error;

            if (error > tolerance) {
                dgamma_mismatches++;
                if (dgamma_mismatches <= 5) {  // Print first 5 mismatches
                    std::cout << "dgamma mismatch at index " << i << ": computed=" << metal_dgamma[i]
                              << ", reference=" << dgamma_ref.data()[i] << ", error=" << error << std::endl;
                }
            }
        }

        float dgamma_avg_error = dgamma_sum_error / features;
        float dgamma_variance = 0.0f;
        for (uint32_t i = 0; i < features; ++i) {
            float error = std::abs(metal_dgamma[i] - dgamma_ref.data()[i]);
            float diff = error - dgamma_avg_error;
            dgamma_variance += diff * diff;
        }
        dgamma_variance /= features;

        std::cout << "dgamma: Max error: " << max_dgamma_error << ", Avg error: " << dgamma_avg_error
                  << ", Variance: " << dgamma_variance << ", Total mismatches: " << dgamma_mismatches << "/" << features
                  << std::endl;

        // Test dbeta accuracy
        uint32_t dbeta_mismatches = 0;
        float max_dbeta_error = 0.0f;
        float dbeta_sum_error = 0.0f;

        std::cout << "\n=== Testing dbeta accuracy ===" << std::endl;
        for (uint32_t i = 0; i < features; ++i) {
            float error = std::abs(metal_dbeta[i] - dbeta_ref.data()[i]);
            max_dbeta_error = std::max(max_dbeta_error, error);
            dbeta_sum_error += error;

            if (error > tolerance) {
                dbeta_mismatches++;
                if (dbeta_mismatches <= 5) {  // Print first 5 mismatches
                    std::cout << "dbeta mismatch at index " << i << ": computed=" << metal_dbeta[i]
                              << ", reference=" << dbeta_ref.data()[i] << ", error=" << error << std::endl;
                }
            }
        }

        float dbeta_avg_error = dbeta_sum_error / features;
        float dbeta_variance = 0.0f;
        for (uint32_t i = 0; i < features; ++i) {
            float error = std::abs(metal_dbeta[i] - dbeta_ref.data()[i]);
            float diff = error - dbeta_avg_error;
            dbeta_variance += diff * diff;
        }
        dbeta_variance /= features;

        std::cout << "dbeta: Max error: " << max_dbeta_error << ", Avg error: " << dbeta_avg_error
                  << ", Variance: " << dbeta_variance << ", Total mismatches: " << dbeta_mismatches << "/" << features
                  << std::endl;
    }
}

TEST_F(LayerNormFusedOpTest, MetalLayerNormBw_LargeFeatures_NoL1Fit) {
    using namespace ttml;

    uint32_t batch_size = 3;
    uint32_t seq_len = 273;
    uint32_t heads = 1;
    uint32_t features = 8462;

    std::cout << "\n=== Test: MetalLayerNormBw_LargeFeatures_NoL1Fit ===" << std::endl;
    std::cout << "Hyperparameters:" << std::endl;
    std::cout << "  batch_size: " << batch_size << std::endl;
    std::cout << "  seq_len: " << seq_len << std::endl;
    std::cout << "  heads: " << heads << std::endl;
    std::cout << "  features: " << features << std::endl;
    std::cout << "  eps: 1e-5" << std::endl;
    std::cout << "  tolerance: 1e-1" << std::endl;
    std::cout << "  total_elements: " << (batch_size * seq_len * heads * features) << std::endl;
    std::cout << "  iterations: 3" << std::endl;

    std::mt19937 gen(1213);
    std::normal_distribution<float> dist(0.0f, 10.0f);  // Larger variance

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
        auto metal_dbeta = core::to_vector(output_tensors[2].value());

        float tolerance = 1e-1f;

        // Test dx accuracy
        uint32_t dx_mismatches = 0;
        float max_dx_error = 0.0f;
        float dx_sum_error = 0.0f;

        std::cout << "\n=== Testing dx accuracy ===" << std::endl;
        for (uint32_t i = 0; i < total_elements; ++i) {
            float error = std::abs(metal_dx[i] - dx_ref.data()[i]);
            max_dx_error = std::max(max_dx_error, error);
            dx_sum_error += error;

            if (error > tolerance) {
                dx_mismatches++;
                if (dx_mismatches <= 5) {  // Print first 5 mismatches
                    std::cout << "dx mismatch at index " << i << ": computed=" << metal_dx[i]
                              << ", reference=" << dx_ref.data()[i] << ", error=" << error << std::endl;
                }
            }
        }

        float dx_avg_error = dx_sum_error / total_elements;
        float dx_variance = 0.0f;
        for (uint32_t i = 0; i < total_elements; ++i) {
            float error = std::abs(metal_dx[i] - dx_ref.data()[i]);
            float diff = error - dx_avg_error;
            dx_variance += diff * diff;
        }
        dx_variance /= total_elements;

        std::cout << "dx: Max error: " << max_dx_error << ", Avg error: " << dx_avg_error
                  << ", Variance: " << dx_variance << ", Total mismatches: " << dx_mismatches << "/" << total_elements
                  << std::endl;

        // Test dgamma accuracy
        uint32_t dgamma_mismatches = 0;
        float max_dgamma_error = 0.0f;
        float dgamma_sum_error = 0.0f;

        std::cout << "\n=== Testing dgamma accuracy ===" << std::endl;
        for (uint32_t i = 0; i < features; ++i) {
            float error = std::abs(metal_dgamma[i] - dgamma_ref.data()[i]);
            max_dgamma_error = std::max(max_dgamma_error, error);
            dgamma_sum_error += error;

            if (error > tolerance) {
                dgamma_mismatches++;
                if (dgamma_mismatches <= 5) {  // Print first 5 mismatches
                    std::cout << "dgamma mismatch at index " << i << ": computed=" << metal_dgamma[i]
                              << ", reference=" << dgamma_ref.data()[i] << ", error=" << error << std::endl;
                }
            }
        }

        float dgamma_avg_error = dgamma_sum_error / features;
        float dgamma_variance = 0.0f;
        for (uint32_t i = 0; i < features; ++i) {
            float error = std::abs(metal_dgamma[i] - dgamma_ref.data()[i]);
            float diff = error - dgamma_avg_error;
            dgamma_variance += diff * diff;
        }
        dgamma_variance /= features;

        std::cout << "dgamma: Max error: " << max_dgamma_error << ", Avg error: " << dgamma_avg_error
                  << ", Variance: " << dgamma_variance << ", Total mismatches: " << dgamma_mismatches << "/" << features
                  << std::endl;

        // Test dbeta accuracy
        uint32_t dbeta_mismatches = 0;
        float max_dbeta_error = 0.0f;
        float dbeta_sum_error = 0.0f;

        std::cout << "\n=== Testing dbeta accuracy ===" << std::endl;
        for (uint32_t i = 0; i < features; ++i) {
            float error = std::abs(metal_dbeta[i] - dbeta_ref.data()[i]);
            max_dbeta_error = std::max(max_dbeta_error, error);
            dbeta_sum_error += error;

            if (error > tolerance) {
                dbeta_mismatches++;
                if (dbeta_mismatches <= 5) {  // Print first 5 mismatches
                    std::cout << "dbeta mismatch at index " << i << ": computed=" << metal_dbeta[i]
                              << ", reference=" << dbeta_ref.data()[i] << ", error=" << error << std::endl;
                }
            }
        }

        float dbeta_avg_error = dbeta_sum_error / features;
        float dbeta_variance = 0.0f;
        for (uint32_t i = 0; i < features; ++i) {
            float error = std::abs(metal_dbeta[i] - dbeta_ref.data()[i]);
            float diff = error - dbeta_avg_error;
            dbeta_variance += diff * diff;
        }
        dbeta_variance /= features;

        std::cout << "dbeta: Max error: " << max_dbeta_error << ", Avg error: " << dbeta_avg_error
                  << ", Variance: " << dbeta_variance << ", Total mismatches: " << dbeta_mismatches << "/" << features
                  << std::endl;
    }
}

TEST_F(LayerNormFusedOpTest, MetalLayerNormBw_Everything_Small_L1Fit) {
    using namespace ttml;

    uint32_t batch_size = 3;
    uint32_t seq_len = 100;
    uint32_t heads = 1;
    uint32_t features = 324;

    std::cout << "\n=== Test: MetalLayerNormBw_Everything_Small_L1Fit ===" << std::endl;
    std::cout << "Hyperparameters:" << std::endl;
    std::cout << "  batch_size: " << batch_size << std::endl;
    std::cout << "  seq_len: " << seq_len << std::endl;
    std::cout << "  heads: " << heads << std::endl;
    std::cout << "  features: " << features << std::endl;
    std::cout << "  eps: 1e-5" << std::endl;
    std::cout << "  tolerance: 1e-1" << std::endl;
    std::cout << "  total_elements: " << (batch_size * seq_len * heads * features) << std::endl;
    std::cout << "  iterations: 10" << std::endl;

    std::mt19937 gen(1213);
    std::normal_distribution<float> dist(0.0f, 10.0f);  // Larger variance

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
        auto metal_dbeta = core::to_vector(output_tensors[2].value());

        float tolerance = 1e-1f;

        // Test dx accuracy
        uint32_t dx_mismatches = 0;
        float max_dx_error = 0.0f;
        float dx_sum_error = 0.0f;

        std::cout << "\n=== Testing dx accuracy ===" << std::endl;
        for (uint32_t i = 0; i < total_elements; ++i) {
            float error = std::abs(metal_dx[i] - dx_ref.data()[i]);
            max_dx_error = std::max(max_dx_error, error);
            dx_sum_error += error;

            if (error > tolerance) {
                dx_mismatches++;
                if (dx_mismatches <= 5) {  // Print first 5 mismatches
                    std::cout << "dx mismatch at index " << i << ": computed=" << metal_dx[i]
                              << ", reference=" << dx_ref.data()[i] << ", error=" << error << std::endl;
                }
            }
        }

        float dx_avg_error = dx_sum_error / total_elements;
        float dx_variance = 0.0f;
        for (uint32_t i = 0; i < total_elements; ++i) {
            float error = std::abs(metal_dx[i] - dx_ref.data()[i]);
            float diff = error - dx_avg_error;
            dx_variance += diff * diff;
        }
        dx_variance /= total_elements;

        std::cout << "dx: Max error: " << max_dx_error << ", Avg error: " << dx_avg_error
                  << ", Variance: " << dx_variance << ", Total mismatches: " << dx_mismatches << "/" << total_elements
                  << std::endl;

        for (uint32_t i = 0; i < total_elements; ++i) {
            EXPECT_NEAR(metal_dx[i], dx_ref.data()[i], tolerance) << "Input gradient mismatch at index " << i;
        }

        // Test dgamma accuracy
        uint32_t dgamma_mismatches = 0;
        float max_dgamma_error = 0.0f;
        float dgamma_sum_error = 0.0f;

        std::cout << "\n=== Testing dgamma accuracy ===" << std::endl;
        for (uint32_t i = 0; i < features; ++i) {
            float error = std::abs(metal_dgamma[i] - dgamma_ref.data()[i]);
            max_dgamma_error = std::max(max_dgamma_error, error);
            dgamma_sum_error += error;

            if (error > tolerance) {
                dgamma_mismatches++;
                if (dgamma_mismatches <= 5) {  // Print first 5 mismatches
                    std::cout << "dgamma mismatch at index " << i << ": computed=" << metal_dgamma[i]
                              << ", reference=" << dgamma_ref.data()[i] << ", error=" << error << std::endl;
                }
            }
        }

        float dgamma_avg_error = dgamma_sum_error / features;
        float dgamma_variance = 0.0f;
        for (uint32_t i = 0; i < features; ++i) {
            float error = std::abs(metal_dgamma[i] - dgamma_ref.data()[i]);
            float diff = error - dgamma_avg_error;
            dgamma_variance += diff * diff;
        }
        dgamma_variance /= features;

        std::cout << "dgamma: Max error: " << max_dgamma_error << ", Avg error: " << dgamma_avg_error
                  << ", Variance: " << dgamma_variance << ", Total mismatches: " << dgamma_mismatches << "/" << features
                  << std::endl;

        // Test dbeta accuracy
        uint32_t dbeta_mismatches = 0;
        float max_dbeta_error = 0.0f;
        float dbeta_sum_error = 0.0f;

        std::cout << "\n=== Testing dbeta accuracy ===" << std::endl;
        for (uint32_t i = 0; i < features; ++i) {
            float error = std::abs(metal_dbeta[i] - dbeta_ref.data()[i]);
            max_dbeta_error = std::max(max_dbeta_error, error);
            dbeta_sum_error += error;

            if (error > tolerance) {
                dbeta_mismatches++;
                if (dbeta_mismatches <= 5) {  // Print first 5 mismatches
                    std::cout << "dbeta mismatch at index " << i << ": computed=" << metal_dbeta[i]
                              << ", reference=" << dbeta_ref.data()[i] << ", error=" << error << std::endl;
                }
            }
        }

        float dbeta_avg_error = dbeta_sum_error / features;
        float dbeta_variance = 0.0f;
        for (uint32_t i = 0; i < features; ++i) {
            float error = std::abs(metal_dbeta[i] - dbeta_ref.data()[i]);
            float diff = error - dbeta_avg_error;
            dbeta_variance += diff * diff;
        }
        dbeta_variance /= features;

        std::cout << "dbeta: Max error: " << max_dbeta_error << ", Avg error: " << dbeta_avg_error
                  << ", Variance: " << dbeta_variance << ", Total mismatches: " << dbeta_mismatches << "/" << features
                  << std::endl;
    }
}

TEST_F(LayerNormFusedOpTest, MetalLayerNormBw_One_Tile_Row) {
    using namespace ttml;

    uint32_t batch_size = 1;
    uint32_t seq_len = 19;
    uint32_t heads = 1;
    uint32_t features = 213;

    std::cout << "\n=== Test: MetalLayerNormBw_One_Tile_Row ===" << std::endl;
    std::cout << "Hyperparameters:" << std::endl;
    std::cout << "  batch_size: " << batch_size << std::endl;
    std::cout << "  seq_len: " << seq_len << std::endl;
    std::cout << "  heads: " << heads << std::endl;
    std::cout << "  features: " << features << std::endl;
    std::cout << "  eps: 1e-5" << std::endl;
    std::cout << "  tolerance: 1e-1" << std::endl;
    std::cout << "  total_elements: " << (batch_size * seq_len * heads * features) << std::endl;
    std::cout << "  iterations: 10" << std::endl;

    std::mt19937 gen(1213);
    std::normal_distribution<float> dist(0.0f, 10.0f);  // Larger variance

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
        auto metal_dbeta = core::to_vector(output_tensors[2].value());

        float tolerance = 1e-1f;

        // Test dx accuracy
        uint32_t dx_mismatches = 0;
        float max_dx_error = 0.0f;
        float dx_sum_error = 0.0f;

        std::cout << "\n=== Testing dx accuracy ===" << std::endl;
        for (uint32_t i = 0; i < total_elements; ++i) {
            float error = std::abs(metal_dx[i] - dx_ref.data()[i]);
            max_dx_error = std::max(max_dx_error, error);
            dx_sum_error += error;

            if (error > tolerance) {
                dx_mismatches++;
                if (dx_mismatches <= 5) {  // Print first 5 mismatches
                    std::cout << "dx mismatch at index " << i << ": computed=" << metal_dx[i]
                              << ", reference=" << dx_ref.data()[i] << ", error=" << error << std::endl;
                }
            }
        }

        float dx_avg_error = dx_sum_error / total_elements;
        float dx_variance = 0.0f;
        for (uint32_t i = 0; i < total_elements; ++i) {
            float error = std::abs(metal_dx[i] - dx_ref.data()[i]);
            float diff = error - dx_avg_error;
            dx_variance += diff * diff;
        }
        dx_variance /= total_elements;

        std::cout << "dx: Max error: " << max_dx_error << ", Avg error: " << dx_avg_error
                  << ", Variance: " << dx_variance << ", Total mismatches: " << dx_mismatches << "/" << total_elements
                  << std::endl;

        for (uint32_t i = 0; i < total_elements; ++i) {
            EXPECT_NEAR(metal_dx[i], dx_ref.data()[i], tolerance) << "Input gradient mismatch at index " << i;
        }

        // Test dgamma accuracy
        uint32_t dgamma_mismatches = 0;
        float max_dgamma_error = 0.0f;
        float dgamma_sum_error = 0.0f;

        std::cout << "\n=== Testing dgamma accuracy ===" << std::endl;
        for (uint32_t i = 0; i < features; ++i) {
            float error = std::abs(metal_dgamma[i] - dgamma_ref.data()[i]);
            max_dgamma_error = std::max(max_dgamma_error, error);
            dgamma_sum_error += error;

            if (error > tolerance) {
                dgamma_mismatches++;
                if (dgamma_mismatches <= 5) {  // Print first 5 mismatches
                    std::cout << "dgamma mismatch at index " << i << ": computed=" << metal_dgamma[i]
                              << ", reference=" << dgamma_ref.data()[i] << ", error=" << error << std::endl;
                }
            }
        }

        float dgamma_avg_error = dgamma_sum_error / features;
        float dgamma_variance = 0.0f;
        for (uint32_t i = 0; i < features; ++i) {
            float error = std::abs(metal_dgamma[i] - dgamma_ref.data()[i]);
            float diff = error - dgamma_avg_error;
            dgamma_variance += diff * diff;
        }
        dgamma_variance /= features;

        std::cout << "dgamma: Max error: " << max_dgamma_error << ", Avg error: " << dgamma_avg_error
                  << ", Variance: " << dgamma_variance << ", Total mismatches: " << dgamma_mismatches << "/" << features
                  << std::endl;

        // Test dbeta accuracy
        uint32_t dbeta_mismatches = 0;
        float max_dbeta_error = 0.0f;
        float dbeta_sum_error = 0.0f;

        std::cout << "\n=== Testing dbeta accuracy ===" << std::endl;
        for (uint32_t i = 0; i < features; ++i) {
            float error = std::abs(metal_dbeta[i] - dbeta_ref.data()[i]);
            max_dbeta_error = std::max(max_dbeta_error, error);
            dbeta_sum_error += error;

            if (error > tolerance) {
                dbeta_mismatches++;
                if (dbeta_mismatches <= 5) {  // Print first 5 mismatches
                    std::cout << "dbeta mismatch at index " << i << ": computed=" << metal_dbeta[i]
                              << ", reference=" << dbeta_ref.data()[i] << ", error=" << error << std::endl;
                }
            }
        }

        float dbeta_avg_error = dbeta_sum_error / features;
        float dbeta_variance = 0.0f;
        for (uint32_t i = 0; i < features; ++i) {
            float error = std::abs(metal_dbeta[i] - dbeta_ref.data()[i]);
            float diff = error - dbeta_avg_error;
            dbeta_variance += diff * diff;
        }
        dbeta_variance /= features;

        std::cout << "dbeta: Max error: " << max_dbeta_error << ", Avg error: " << dbeta_avg_error
                  << ", Variance: " << dbeta_variance << ", Total mismatches: " << dbeta_mismatches << "/" << features
                  << std::endl;
    }
}

TEST_F(LayerNormFusedOpTest, CompositeLayerNormBackward_AgainstXTensor_NotThatLargeFeatures) {
    using namespace ttml;

    uint32_t batch_size = 3;
    uint32_t seq_len = 100;
    uint32_t heads = 1;
    uint32_t features = 324;

    std::cout << "\n=== Test: CompositeLayerNormBackward_AgainstXTensor_NotThatLargeFeatures ===" << std::endl;
    std::cout << "Hyperparameters:" << std::endl;
    std::cout << "  batch_size: " << batch_size << std::endl;
    std::cout << "  seq_len: " << seq_len << std::endl;
    std::cout << "  heads: " << heads << std::endl;
    std::cout << "  features: " << features << std::endl;
    std::cout << "  eps: 1e-5" << std::endl;
    std::cout << "  tolerance: 1e-1" << std::endl;
    std::cout << "  total_elements: " << (batch_size * seq_len * heads * features) << std::endl;

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 10.0f);

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

    // Generate gradient data (dy)
    std::vector<float> dy_data;
    std::normal_distribution<float> grad_dist(0.0f, 0.2f);
    for (uint32_t i = 0; i < total_elements; ++i) {
        dy_data.push_back(grad_dist(gen));
    }

    // Convert to xtensor and get reference outputs using xtensor implementation
    uint32_t combined_batch = batch_size * seq_len * heads;
    auto x_xtensor = xt::adapt(test_data, {combined_batch * features});
    auto gamma_xtensor = xt::adapt(gamma_data, {features});
    auto beta_xtensor = xt::adapt(beta_data, {features});
    auto dy_xtensor = xt::adapt(dy_data, {combined_batch * features});

    // Get reference forward and backward results
    auto [y_ref, cache] =
        layernorm_forward_reference(x_xtensor, gamma_xtensor, beta_xtensor, combined_batch, features, 1e-5f);
    auto [dx_ref, dgamma_ref, dbeta_ref] = layernorm_backward_reference(dy_xtensor, cache);

    // Create tensors for composite operation with autograd enabled
    auto input_tensor = autograd::create_tensor(core::from_vector(
        test_data, ttnn::Shape({batch_size, heads, seq_len, features}), &autograd::ctx().get_device()));
    auto gamma_tensor = autograd::create_tensor(
        core::from_vector(gamma_data, ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device()));
    auto beta_tensor = autograd::create_tensor(
        core::from_vector(beta_data, ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device()));

    // Call composite layernorm forward
    auto output_tensor = ops::composite_layernorm(input_tensor, gamma_tensor, beta_tensor);

    // Create gradient tensor and trigger backward pass
    auto grad_output =
        core::from_vector(dy_data, ttnn::Shape({batch_size, heads, seq_len, features}), &autograd::ctx().get_device());
    output_tensor->set_grad(grad_output);
    output_tensor->backward();

    // Get computed gradients
    auto dx_computed = core::to_vector(input_tensor->get_grad());
    auto dgamma_computed = core::to_vector(gamma_tensor->get_grad());
    auto dbeta_computed = core::to_vector(beta_tensor->get_grad());

    // Test dx accuracy
    float tolerance = 1e-1f;
    uint32_t dx_mismatches = 0;
    float dx_max_error = 0.0f;
    float dx_sum_error = 0.0f;

    std::cout << "\n=== Testing dx accuracy ===" << std::endl;
    for (uint32_t i = 0; i < total_elements; ++i) {
        float error = std::abs(dx_computed[i] - dx_ref.data()[i]);
        dx_max_error = std::max(dx_max_error, error);
        dx_sum_error += error;

        if (error > tolerance) {
            dx_mismatches++;
            if (dx_mismatches <= 5) {  // Print first 5 mismatches
                std::cout << "dx mismatch at index " << i << ": computed=" << dx_computed[i]
                          << ", reference=" << dx_ref.data()[i] << ", error=" << error << std::endl;
            }
        }
    }

    float dx_avg_error = dx_sum_error / total_elements;
    float dx_variance = 0.0f;
    for (uint32_t i = 0; i < total_elements; ++i) {
        float error = std::abs(dx_computed[i] - dx_ref.data()[i]);
        float diff = error - dx_avg_error;
        dx_variance += diff * diff;
    }
    dx_variance /= total_elements;

    std::cout << "dx: Max error: " << dx_max_error << ", Avg error: " << dx_avg_error << ", Variance: " << dx_variance
              << ", Total mismatches: " << dx_mismatches << "/" << total_elements << std::endl;

    // Test dgamma accuracy
    uint32_t dgamma_mismatches = 0;
    float dgamma_max_error = 0.0f;
    float dgamma_sum_error = 0.0f;

    std::cout << "\n=== Testing dgamma accuracy ===" << std::endl;
    for (uint32_t i = 0; i < features; ++i) {
        float error = std::abs(dgamma_computed[i] - dgamma_ref.data()[i]);
        dgamma_max_error = std::max(dgamma_max_error, error);
        dgamma_sum_error += error;

        if (error > tolerance) {
            dgamma_mismatches++;
            if (dgamma_mismatches <= 5) {  // Print first 5 mismatches
                std::cout << "dgamma mismatch at index " << i << ": computed=" << dgamma_computed[i]
                          << ", reference=" << dgamma_ref.data()[i] << ", error=" << error << std::endl;
            }
        }
    }

    float dgamma_avg_error = dgamma_sum_error / features;
    float dgamma_variance = 0.0f;
    for (uint32_t i = 0; i < features; ++i) {
        float error = std::abs(dgamma_computed[i] - dgamma_ref.data()[i]);
        float diff = error - dgamma_avg_error;
        dgamma_variance += diff * diff;
    }
    dgamma_variance /= features;

    std::cout << "dgamma: Max error: " << dgamma_max_error << ", Avg error: " << dgamma_avg_error
              << ", Variance: " << dgamma_variance << ", Total mismatches: " << dgamma_mismatches << "/" << features
              << std::endl;

    // Test dbeta accuracy
    uint32_t dbeta_mismatches = 0;
    float dbeta_max_error = 0.0f;
    float dbeta_sum_error = 0.0f;

    std::cout << "\n=== Testing dbeta accuracy ===" << std::endl;
    for (uint32_t i = 0; i < features; ++i) {
        float error = std::abs(dbeta_computed[i] - dbeta_ref.data()[i]);
        dbeta_max_error = std::max(dbeta_max_error, error);
        dbeta_sum_error += error;

        if (error > tolerance) {
            dbeta_mismatches++;
            if (dbeta_mismatches <= 5) {  // Print first 5 mismatches
                std::cout << "dbeta mismatch at index " << i << ": computed=" << dbeta_computed[i]
                          << ", reference=" << dbeta_ref.data()[i] << ", error=" << error << std::endl;
            }
        }
    }

    float dbeta_avg_error = dbeta_sum_error / features;
    float dbeta_variance = 0.0f;
    for (uint32_t i = 0; i < features; ++i) {
        float error = std::abs(dbeta_computed[i] - dbeta_ref.data()[i]);
        float diff = error - dbeta_avg_error;
        dbeta_variance += diff * diff;
    }
    dbeta_variance /= features;

    std::cout << "dbeta: Max error: " << dbeta_max_error << ", Avg error: " << dbeta_avg_error
              << ", Variance: " << dbeta_variance << ", Total mismatches: " << dbeta_mismatches << "/" << features
              << std::endl;

    std::cout << "\n=== Test Summary ===" << std::endl;
    std::cout << "All gradients (dx, dgamma, dbeta) tested successfully against xarray reference!" << std::endl;
}

TEST_F(LayerNormFusedOpTest, CompositeLayerNormBackward_AgainstXTensor_LargeFeatures) {
    using namespace ttml;

    uint32_t batch_size = 3;
    uint32_t seq_len = 273;
    uint32_t heads = 1;
    uint32_t features = 8462;

    std::cout << "\n=== Test: CompositeLayerNormBackward_AgainstXTensor_LargeFeatures ===" << std::endl;
    std::cout << "Hyperparameters:" << std::endl;
    std::cout << "  batch_size: " << batch_size << std::endl;
    std::cout << "  seq_len: " << seq_len << std::endl;
    std::cout << "  heads: " << heads << std::endl;
    std::cout << "  features: " << features << std::endl;
    std::cout << "  eps: 1e-5" << std::endl;
    std::cout << "  tolerance: 1e-1" << std::endl;
    std::cout << "  total_elements: " << (batch_size * seq_len * heads * features) << std::endl;

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 10.0f);

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

    // Generate gradient data (dy)
    std::vector<float> dy_data;
    std::normal_distribution<float> grad_dist(0.0f, 0.2f);
    for (uint32_t i = 0; i < total_elements; ++i) {
        dy_data.push_back(grad_dist(gen));
    }

    // Convert to xtensor and get reference outputs using xtensor implementation
    uint32_t combined_batch = batch_size * seq_len * heads;
    auto x_xtensor = xt::adapt(test_data, {combined_batch * features});
    auto gamma_xtensor = xt::adapt(gamma_data, {features});
    auto beta_xtensor = xt::adapt(beta_data, {features});
    auto dy_xtensor = xt::adapt(dy_data, {combined_batch * features});

    // Get reference forward and backward results
    auto [y_ref, cache] =
        layernorm_forward_reference(x_xtensor, gamma_xtensor, beta_xtensor, combined_batch, features, 1e-5f);
    auto [dx_ref, dgamma_ref, dbeta_ref] = layernorm_backward_reference(dy_xtensor, cache);

    // Create tensors for composite operation with autograd enabled
    auto input_tensor = autograd::create_tensor(core::from_vector(
        test_data, ttnn::Shape({batch_size, heads, seq_len, features}), &autograd::ctx().get_device()));
    auto gamma_tensor = autograd::create_tensor(
        core::from_vector(gamma_data, ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device()));
    auto beta_tensor = autograd::create_tensor(
        core::from_vector(beta_data, ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device()));

    // Call composite layernorm forward
    auto output_tensor = ops::composite_layernorm(input_tensor, gamma_tensor, beta_tensor);

    // Create gradient tensor and trigger backward pass
    auto grad_output =
        core::from_vector(dy_data, ttnn::Shape({batch_size, heads, seq_len, features}), &autograd::ctx().get_device());
    output_tensor->set_grad(grad_output);
    output_tensor->backward();

    // Get computed gradients
    auto dx_computed = core::to_vector(input_tensor->get_grad());
    auto dgamma_computed = core::to_vector(gamma_tensor->get_grad());
    auto dbeta_computed = core::to_vector(beta_tensor->get_grad());

    // Test dx accuracy
    float tolerance = 1e-1f;
    uint32_t dx_mismatches = 0;
    float dx_max_error = 0.0f;
    float dx_sum_error = 0.0f;

    std::cout << "\n=== Testing dx accuracy ===" << std::endl;
    for (uint32_t i = 0; i < total_elements; ++i) {
        float error = std::abs(dx_computed[i] - dx_ref.data()[i]);
        dx_max_error = std::max(dx_max_error, error);
        dx_sum_error += error;

        if (error > tolerance) {
            dx_mismatches++;
            if (dx_mismatches <= 5) {  // Print first 5 mismatches
                std::cout << "dx mismatch at index " << i << ": computed=" << dx_computed[i]
                          << ", reference=" << dx_ref.data()[i] << ", error=" << error << std::endl;
            }
        }
    }

    float dx_avg_error = dx_sum_error / total_elements;
    float dx_variance = 0.0f;
    for (uint32_t i = 0; i < total_elements; ++i) {
        float error = std::abs(dx_computed[i] - dx_ref.data()[i]);
        float diff = error - dx_avg_error;
        dx_variance += diff * diff;
    }
    dx_variance /= total_elements;

    std::cout << "dx: Max error: " << dx_max_error << ", Avg error: " << dx_avg_error << ", Variance: " << dx_variance
              << ", Total mismatches: " << dx_mismatches << "/" << total_elements << std::endl;

    // Test dgamma accuracy
    uint32_t dgamma_mismatches = 0;
    float dgamma_max_error = 0.0f;
    float dgamma_sum_error = 0.0f;

    std::cout << "\n=== Testing dgamma accuracy ===" << std::endl;
    for (uint32_t i = 0; i < features; ++i) {
        float error = std::abs(dgamma_computed[i] - dgamma_ref.data()[i]);
        dgamma_max_error = std::max(dgamma_max_error, error);
        dgamma_sum_error += error;

        if (error > tolerance) {
            dgamma_mismatches++;
            if (dgamma_mismatches <= 5) {  // Print first 5 mismatches
                std::cout << "dgamma mismatch at index " << i << ": computed=" << dgamma_computed[i]
                          << ", reference=" << dgamma_ref.data()[i] << ", error=" << error << std::endl;
            }
        }
    }

    float dgamma_avg_error = dgamma_sum_error / features;
    float dgamma_variance = 0.0f;
    for (uint32_t i = 0; i < features; ++i) {
        float error = std::abs(dgamma_computed[i] - dgamma_ref.data()[i]);
        float diff = error - dgamma_avg_error;
        dgamma_variance += diff * diff;
    }
    dgamma_variance /= features;

    std::cout << "dgamma: Max error: " << dgamma_max_error << ", Avg error: " << dgamma_avg_error
              << ", Variance: " << dgamma_variance << ", Total mismatches: " << dgamma_mismatches << "/" << features
              << std::endl;

    // Test dbeta accuracy
    uint32_t dbeta_mismatches = 0;
    float dbeta_max_error = 0.0f;
    float dbeta_sum_error = 0.0f;

    std::cout << "\n=== Testing dbeta accuracy ===" << std::endl;
    for (uint32_t i = 0; i < features; ++i) {
        float error = std::abs(dbeta_computed[i] - dbeta_ref.data()[i]);
        dbeta_max_error = std::max(dbeta_max_error, error);
        dbeta_sum_error += error;

        if (error > tolerance) {
            dbeta_mismatches++;
            if (dbeta_mismatches <= 5) {  // Print first 5 mismatches
                std::cout << "dbeta mismatch at index " << i << ": computed=" << dbeta_computed[i]
                          << ", reference=" << dbeta_ref.data()[i] << ", error=" << error << std::endl;
            }
        }
    }

    float dbeta_avg_error = dbeta_sum_error / features;
    float dbeta_variance = 0.0f;
    for (uint32_t i = 0; i < features; ++i) {
        float error = std::abs(dbeta_computed[i] - dbeta_ref.data()[i]);
        float diff = error - dbeta_avg_error;
        dbeta_variance += diff * diff;
    }
    dbeta_variance /= features;

    std::cout << "dbeta: Max error: " << dbeta_max_error << ", Avg error: " << dbeta_avg_error
              << ", Variance: " << dbeta_variance << ", Total mismatches: " << dbeta_mismatches << "/" << features
              << std::endl;

    std::cout << "\n=== Test Summary ===" << std::endl;
    std::cout << "All gradients (dx, dgamma, dbeta) tested successfully against xarray reference!" << std::endl;
}

TEST_F(LayerNormFusedOpTest, MorehLayerNormBackward_AgainstXTensor_LargeFeatures) {
    using namespace ttml;

    uint32_t batch_size = 3;
    uint32_t seq_len = 273;
    uint32_t heads = 1;
    uint32_t features = 8462;

    std::cout << "\n=== Test: MorehLayerNormBackward_LargeFeatures ===" << std::endl;
    std::cout << "Hyperparameters:" << std::endl;
    std::cout << "  batch_size: " << batch_size << std::endl;
    std::cout << "  seq_len: " << seq_len << std::endl;
    std::cout << "  heads: " << heads << std::endl;
    std::cout << "  features: " << features << std::endl;
    std::cout << "  eps: 1e-6" << std::endl;
    std::cout << "  tolerance: 1e-1" << std::endl;
    std::cout << "  total_elements: " << (batch_size * seq_len * heads * features) << std::endl;

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 10.0f);

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

    // Generate gradient data (dy)
    std::vector<float> dy_data;
    std::normal_distribution<float> grad_dist(0.0f, 0.2f);
    for (uint32_t i = 0; i < total_elements; ++i) {
        dy_data.push_back(grad_dist(gen));
    }

    // Convert to xtensor and get reference outputs using xtensor implementation
    uint32_t combined_batch = batch_size * seq_len * heads;
    auto x_xtensor = xt::adapt(test_data, {combined_batch * features});
    auto gamma_xtensor = xt::adapt(gamma_data, {features});
    auto beta_xtensor = xt::adapt(beta_data, {features});
    auto dy_xtensor = xt::adapt(dy_data, {combined_batch * features});

    // Note: moreh_layer_norm uses eps=1e-6, while the reference uses 1e-5
    // We'll use 1e-6 to match moreh_layer_norm
    auto [y_ref, cache] =
        layernorm_forward_reference(x_xtensor, gamma_xtensor, beta_xtensor, combined_batch, features, 1e-6f);
    auto [dx_ref, dgamma_ref, dbeta_ref] = layernorm_backward_reference(dy_xtensor, cache);

    // Create tensors for moreh operation with autograd enabled
    auto input_tensor = autograd::create_tensor(core::from_vector(
        test_data, ttnn::Shape({batch_size, heads, seq_len, features}), &autograd::ctx().get_device()));
    auto gamma_tensor = autograd::create_tensor(
        core::from_vector(gamma_data, ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device()));
    auto beta_tensor = autograd::create_tensor(
        core::from_vector(beta_data, ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device()));

    // Call moreh layernorm forward (via ops::layernorm)
    auto output_tensor = ops::layernorm(input_tensor, gamma_tensor, beta_tensor);

    // Create gradient tensor and trigger backward pass
    auto grad_output =
        core::from_vector(dy_data, ttnn::Shape({batch_size, heads, seq_len, features}), &autograd::ctx().get_device());
    output_tensor->set_grad(grad_output);
    output_tensor->backward();

    // Get computed gradients
    auto dx_computed = core::to_vector(input_tensor->get_grad());
    auto dgamma_computed = core::to_vector(gamma_tensor->get_grad());
    auto dbeta_computed = core::to_vector(beta_tensor->get_grad());

    // Test dx accuracy
    float tolerance = 1e-1f;
    uint32_t dx_mismatches = 0;
    float dx_max_error = 0.0f;
    float dx_sum_error = 0.0f;

    std::cout << "\n=== Testing dx accuracy (moreh_layer_norm) ===" << std::endl;
    for (uint32_t i = 0; i < total_elements; ++i) {
        float error = std::abs(dx_computed[i] - dx_ref.data()[i]);
        dx_max_error = std::max(dx_max_error, error);
        dx_sum_error += error;

        if (error > tolerance) {
            dx_mismatches++;
            if (dx_mismatches <= 5) {  // Print first 5 mismatches
                std::cout << "dx mismatch at index " << i << ": computed=" << dx_computed[i]
                          << ", reference=" << dx_ref.data()[i] << ", error=" << error << std::endl;
            }
        }
    }

    float dx_avg_error = dx_sum_error / total_elements;
    float dx_variance = 0.0f;
    for (uint32_t i = 0; i < total_elements; ++i) {
        float error = std::abs(dx_computed[i] - dx_ref.data()[i]);
        float diff = error - dx_avg_error;
        dx_variance += diff * diff;
    }
    dx_variance /= total_elements;

    std::cout << "dx: Max error: " << dx_max_error << ", Avg error: " << dx_avg_error << ", Variance: " << dx_variance
              << ", Total mismatches: " << dx_mismatches << "/" << total_elements << std::endl;

    // Test dgamma accuracy
    uint32_t dgamma_mismatches = 0;
    float dgamma_max_error = 0.0f;
    float dgamma_sum_error = 0.0f;

    std::cout << "\n=== Testing dgamma accuracy (moreh_layer_norm) ===" << std::endl;
    for (uint32_t i = 0; i < features; ++i) {
        float error = std::abs(dgamma_computed[i] - dgamma_ref.data()[i]);
        dgamma_max_error = std::max(dgamma_max_error, error);
        dgamma_sum_error += error;

        if (error > tolerance) {
            dgamma_mismatches++;
            if (dgamma_mismatches <= 5) {  // Print first 5 mismatches
                std::cout << "dgamma mismatch at index " << i << ": computed=" << dgamma_computed[i]
                          << ", reference=" << dgamma_ref.data()[i] << ", error=" << error << std::endl;
            }
        }
    }

    float dgamma_avg_error = dgamma_sum_error / features;
    float dgamma_variance = 0.0f;
    for (uint32_t i = 0; i < features; ++i) {
        float error = std::abs(dgamma_computed[i] - dgamma_ref.data()[i]);
        float diff = error - dgamma_avg_error;
        dgamma_variance += diff * diff;
    }
    dgamma_variance /= features;

    std::cout << "dgamma: Max error: " << dgamma_max_error << ", Avg error: " << dgamma_avg_error
              << ", Variance: " << dgamma_variance << ", Total mismatches: " << dgamma_mismatches << "/" << features
              << std::endl;

    // Test dbeta accuracy
    uint32_t dbeta_mismatches = 0;
    float dbeta_max_error = 0.0f;
    float dbeta_sum_error = 0.0f;

    std::cout << "\n=== Testing dbeta accuracy (moreh_layer_norm) ===" << std::endl;
    for (uint32_t i = 0; i < features; ++i) {
        float error = std::abs(dbeta_computed[i] - dbeta_ref.data()[i]);
        dbeta_max_error = std::max(dbeta_max_error, error);
        dbeta_sum_error += error;

        if (error > tolerance) {
            dbeta_mismatches++;
            if (dbeta_mismatches <= 5) {  // Print first 5 mismatches
                std::cout << "dbeta mismatch at index " << i << ": computed=" << dbeta_computed[i]
                          << ", reference=" << dbeta_ref.data()[i] << ", error=" << error << std::endl;
            }
        }
    }

    float dbeta_avg_error = dbeta_sum_error / features;
    float dbeta_variance = 0.0f;
    for (uint32_t i = 0; i < features; ++i) {
        float error = std::abs(dbeta_computed[i] - dbeta_ref.data()[i]);
        float diff = error - dbeta_avg_error;
        dbeta_variance += diff * diff;
    }
    dbeta_variance /= features;

    std::cout << "dbeta: Max error: " << dbeta_max_error << ", Avg error: " << dbeta_avg_error
              << ", Variance: " << dbeta_variance << ", Total mismatches: " << dbeta_mismatches << "/" << features
              << std::endl;

    std::cout << "\n=== Test Summary ===" << std::endl;
    std::cout << "All gradients (dx, dgamma, dbeta) for moreh_layer_norm tested successfully against xarray reference!"
              << std::endl;
}

TEST_F(LayerNormFusedOpTest, MorehLayerNormBackward_AgainstXTensor_NotThatLargeFeatures) {
    using namespace ttml;

    uint32_t batch_size = 3;
    uint32_t seq_len = 100;
    uint32_t heads = 1;
    uint32_t features = 324;

    std::cout << "\n=== Test: MorehLayerNormBackward_AgainstXTensor_NotThatLargeFeatures ===" << std::endl;
    std::cout << "Hyperparameters:" << std::endl;
    std::cout << "  batch_size: " << batch_size << std::endl;
    std::cout << "  seq_len: " << seq_len << std::endl;
    std::cout << "  heads: " << heads << std::endl;
    std::cout << "  features: " << features << std::endl;
    std::cout << "  eps: 1e-6" << std::endl;
    std::cout << "  tolerance: 1e-1" << std::endl;
    std::cout << "  total_elements: " << (batch_size * seq_len * heads * features) << std::endl;

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 10.0f);

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

    // Generate gradient data (dy)
    std::vector<float> dy_data;
    std::normal_distribution<float> grad_dist(0.0f, 0.2f);
    for (uint32_t i = 0; i < total_elements; ++i) {
        dy_data.push_back(grad_dist(gen));
    }

    // Convert to xtensor and get reference outputs using xtensor implementation
    uint32_t combined_batch = batch_size * seq_len * heads;
    auto x_xtensor = xt::adapt(test_data, {combined_batch * features});
    auto gamma_xtensor = xt::adapt(gamma_data, {features});
    auto beta_xtensor = xt::adapt(beta_data, {features});
    auto dy_xtensor = xt::adapt(dy_data, {combined_batch * features});

    // Note: moreh_layer_norm uses eps=1e-6, while the reference uses 1e-5
    // We'll use 1e-6 to match moreh_layer_norm
    auto [y_ref, cache] =
        layernorm_forward_reference(x_xtensor, gamma_xtensor, beta_xtensor, combined_batch, features, 1e-6f);
    auto [dx_ref, dgamma_ref, dbeta_ref] = layernorm_backward_reference(dy_xtensor, cache);

    // Create tensors for moreh operation with autograd enabled
    auto input_tensor = autograd::create_tensor(core::from_vector(
        test_data, ttnn::Shape({batch_size, heads, seq_len, features}), &autograd::ctx().get_device()));
    auto gamma_tensor = autograd::create_tensor(
        core::from_vector(gamma_data, ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device()));
    auto beta_tensor = autograd::create_tensor(
        core::from_vector(beta_data, ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device()));

    // Call moreh layernorm forward (via ops::layernorm)
    auto output_tensor = ops::layernorm(input_tensor, gamma_tensor, beta_tensor);

    // Create gradient tensor and trigger backward pass
    auto grad_output =
        core::from_vector(dy_data, ttnn::Shape({batch_size, heads, seq_len, features}), &autograd::ctx().get_device());
    output_tensor->set_grad(grad_output);
    output_tensor->backward();

    // Get computed gradients
    auto dx_computed = core::to_vector(input_tensor->get_grad());
    auto dgamma_computed = core::to_vector(gamma_tensor->get_grad());
    auto dbeta_computed = core::to_vector(beta_tensor->get_grad());

    // Test dx accuracy
    float tolerance = 1e-1f;
    uint32_t dx_mismatches = 0;
    float dx_max_error = 0.0f;
    float dx_sum_error = 0.0f;

    std::cout << "\n=== Testing dx accuracy (moreh_layer_norm) ===" << std::endl;
    for (uint32_t i = 0; i < total_elements; ++i) {
        float error = std::abs(dx_computed[i] - dx_ref.data()[i]);
        dx_max_error = std::max(dx_max_error, error);
        dx_sum_error += error;

        if (error > tolerance) {
            dx_mismatches++;
            if (dx_mismatches <= 5) {  // Print first 5 mismatches
                std::cout << "dx mismatch at index " << i << ": computed=" << dx_computed[i]
                          << ", reference=" << dx_ref.data()[i] << ", error=" << error << std::endl;
            }
        }
    }

    float dx_avg_error = dx_sum_error / total_elements;
    float dx_variance = 0.0f;
    for (uint32_t i = 0; i < total_elements; ++i) {
        float error = std::abs(dx_computed[i] - dx_ref.data()[i]);
        float diff = error - dx_avg_error;
        dx_variance += diff * diff;
    }
    dx_variance /= total_elements;

    std::cout << "dx: Max error: " << dx_max_error << ", Avg error: " << dx_avg_error << ", Variance: " << dx_variance
              << ", Total mismatches: " << dx_mismatches << "/" << total_elements << std::endl;

    // Test dgamma accuracy
    uint32_t dgamma_mismatches = 0;
    float dgamma_max_error = 0.0f;
    float dgamma_sum_error = 0.0f;

    std::cout << "\n=== Testing dgamma accuracy (moreh_layer_norm) ===" << std::endl;
    for (uint32_t i = 0; i < features; ++i) {
        float error = std::abs(dgamma_computed[i] - dgamma_ref.data()[i]);
        dgamma_max_error = std::max(dgamma_max_error, error);
        dgamma_sum_error += error;

        if (error > tolerance) {
            dgamma_mismatches++;
            if (dgamma_mismatches <= 5) {  // Print first 5 mismatches
                std::cout << "dgamma mismatch at index " << i << ": computed=" << dgamma_computed[i]
                          << ", reference=" << dgamma_ref.data()[i] << ", error=" << error << std::endl;
            }
        }
    }

    float dgamma_avg_error = dgamma_sum_error / features;
    float dgamma_variance = 0.0f;
    for (uint32_t i = 0; i < features; ++i) {
        float error = std::abs(dgamma_computed[i] - dgamma_ref.data()[i]);
        float diff = error - dgamma_avg_error;
        dgamma_variance += diff * diff;
    }
    dgamma_variance /= features;

    std::cout << "dgamma: Max error: " << dgamma_max_error << ", Avg error: " << dgamma_avg_error
              << ", Variance: " << dgamma_variance << ", Total mismatches: " << dgamma_mismatches << "/" << features
              << std::endl;

    // Test dbeta accuracy
    uint32_t dbeta_mismatches = 0;
    float dbeta_max_error = 0.0f;
    float dbeta_sum_error = 0.0f;

    std::cout << "\n=== Testing dbeta accuracy (moreh_layer_norm) ===" << std::endl;
    for (uint32_t i = 0; i < features; ++i) {
        float error = std::abs(dbeta_computed[i] - dbeta_ref.data()[i]);
        dbeta_max_error = std::max(dbeta_max_error, error);
        dbeta_sum_error += error;

        if (error > tolerance) {
            dbeta_mismatches++;
            if (dbeta_mismatches <= 5) {  // Print first 5 mismatches
                std::cout << "dbeta mismatch at index " << i << ": computed=" << dbeta_computed[i]
                          << ", reference=" << dbeta_ref.data()[i] << ", error=" << error << std::endl;
            }
        }
    }

    float dbeta_avg_error = dbeta_sum_error / features;
    float dbeta_variance = 0.0f;
    for (uint32_t i = 0; i < features; ++i) {
        float error = std::abs(dbeta_computed[i] - dbeta_ref.data()[i]);
        float diff = error - dbeta_avg_error;
        dbeta_variance += diff * diff;
    }
    dbeta_variance /= features;

    std::cout << "dbeta: Max error: " << dbeta_max_error << ", Avg error: " << dbeta_avg_error
              << ", Variance: " << dbeta_variance << ", Total mismatches: " << dbeta_mismatches << "/" << features
              << std::endl;

    std::cout << "\n=== Test Summary ===" << std::endl;
    std::cout << "All gradients (dx, dgamma, dbeta) for moreh_layer_norm tested successfully against xarray reference!"
              << std::endl;
}

TEST_F(LayerNormFusedOpTest, LayerNormFwFitsL1) {
    using namespace ttml;

    uint32_t batch_size = 6;
    uint32_t seq_len = 13;
    uint32_t heads = 16;
    uint32_t features = 333;

    uint32_t size = batch_size * seq_len * heads;

    std::vector<float> test_data;
    test_data.reserve((size_t)batch_size * seq_len * heads * features);
    for (uint32_t i = 0; i < batch_size * seq_len * heads; i++) {
        float mean = (float)i / (float)size;
        float stddev = 1.F + (float)i / (float)(size * 2);
        std::mt19937 gen(i);
        std::normal_distribution<float> dist(mean, stddev);
        for (uint32_t j = 0; j < features; j++) {
            test_data.push_back(dist(gen));
        }
    }

    auto tensor = autograd::create_tensor(core::from_vector(
        test_data, ttnn::Shape({batch_size, seq_len, heads, features}), &autograd::ctx().get_device()));

    auto gamma = autograd::create_tensor(core::ones(ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device()));
    auto beta = autograd::create_tensor(core::zeros(ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device()));

    auto result = ops::layernorm(tensor, gamma, beta);

    auto result_tensor = result->get_value();
    auto result_data = core::to_vector(result_tensor);
    for (uint32_t i = 0; i < batch_size * seq_len * heads; i++) {
        uint32_t idx = i * features;

        float exp_mean = 0.F;
        float exp_var = 0.F;
        for (uint32_t j = 0; j < features; ++j) {
            exp_mean += result_data[idx + j];
            exp_var += result_data[idx + j] * result_data[idx + j];
        }

        exp_mean /= (float)features;
        exp_var /= (float)features;
        exp_var = exp_var - exp_mean * exp_mean;

        EXPECT_NEAR(exp_mean, 0.F, 5e-2);
        EXPECT_NEAR(exp_var, 1.F, 5e-2);
    }
}
