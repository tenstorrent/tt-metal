// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
#include "metal/ops/layernorm_fw/layernorm_fw.hpp"
#include "ops/layernorm_op.hpp"

// Reference implementation using xtensor
struct LayerNormForwardCache {
    xt::xarray<float> x;
    xt::xarray<float> x_hat;
    xt::xarray<float> mu;
    xt::xarray<float> rstd;
    xt::xarray<float> gamma;
    xt::xarray<float> beta;
    uint32_t batch_size;
    uint32_t features;
    float eps;
};

std::tuple<xt::xarray<float>, xt::xarray<float>, xt::xarray<float>> layernorm_forward_reference_(
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

// Test Metal LayerNorm Forward vs XTensor Reference
TEST_F(LayerNormForwardOpTest, MetalLayerNormFw_SmallTensor) {
    using namespace ttml;

    uint32_t batch_size = 1;
    uint32_t seq_len = 32;
    uint32_t heads = 1;
    uint32_t features = 32;

    std::cout << "\n=== Test: MetalLayerNormFw_SmallTensor ===" << std::endl;
    std::cout << "Hyperparameters:" << std::endl;
    std::cout << "  batch_size: " << batch_size << std::endl;
    std::cout << "  seq_len: " << seq_len << std::endl;
    std::cout << "  heads: " << heads << std::endl;
    std::cout << "  features: " << features << std::endl;
    std::cout << "  eps: 1e-5" << std::endl;

    uint32_t combined_batch = batch_size * seq_len * heads;
    uint32_t total_elements = combined_batch * features;

    std::mt19937 gen(1234);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> test_data;
    test_data.reserve(total_elements);
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

    // Compute reference using xtensor
    auto x_xtensor = xt::adapt(test_data, {combined_batch * features});
    auto gamma_xtensor = xt::adapt(gamma_data, {features});
    auto beta_xtensor = xt::adapt(beta_data, {features});

    auto [y_ref, mu_ref, rstd_ref] =
        layernorm_forward_reference_(x_xtensor, gamma_xtensor, beta_xtensor, combined_batch, features, 1e-5f);

    // Create tensors for metal operation
    auto input_tensor = core::from_vector(
        test_data, ttnn::Shape({batch_size, heads, seq_len, features}), &autograd::ctx().get_device());
    auto gamma_tensor = core::from_vector(gamma_data, ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device());
    auto beta_tensor = core::from_vector(beta_data, ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device());

    // Call metal layernorm forward
    auto output_tensors = metal::ops::layernorm_fw::LayerNormForwardOperation::invoke(
        input_tensor, gamma_tensor, beta_tensor, 1e-5f, true);

    // Get computed output
    auto y_computed = core::to_vector(output_tensors[0].value());
    auto mu_computed = core::to_vector(output_tensors[1].value());
    auto rstd_computed = core::to_vector(output_tensors[2].value());

    // Test output accuracy
    float tolerance = 5e-2f;
    uint32_t y_mismatches = 0;
    float y_max_error = 0.0f;
    float y_sum_error = 0.0f;

    std::cout << "\n=== Testing output accuracy (metal_layernorm_fw) ===" << std::endl;
    for (uint32_t i = 0; i < total_elements; ++i) {
        float error = std::abs(y_computed[i] - y_ref(i));
        y_sum_error += error;
        y_max_error = std::max(y_max_error, error);
        if (error > tolerance) {
            y_mismatches++;
            if (y_mismatches <= 10) {
                std::cout << "Mismatch at index " << i << ": computed=" << y_computed[i] << ", reference=" << y_ref(i)
                          << ", error=" << error << std::endl;
            }
        }
    }

    float y_avg_error = y_sum_error / total_elements;
    std::cout << "Output statistics:" << std::endl;
    std::cout << "  Mismatches: " << y_mismatches << " / " << total_elements << std::endl;
    std::cout << "  Max error: " << y_max_error << std::endl;
    std::cout << "  Avg error: " << y_avg_error << std::endl;

    EXPECT_LT(y_mismatches, total_elements * 0.01);  // Less than 1% mismatches
    EXPECT_LT(y_max_error, 0.1f);

    // Test mean accuracy
    std::cout << "\n=== Testing mean accuracy ===" << std::endl;
    uint32_t mu_mismatches = 0;
    float mu_max_error = 0.0f;
    float mu_sum_error = 0.0f;
    for (uint32_t i = 0; i < combined_batch; ++i) {
        float error = std::abs(mu_computed[i] - mu_ref(i));
        mu_sum_error += error;
        mu_max_error = std::max(mu_max_error, error);
        if (error > tolerance) {
            mu_mismatches++;
            if (mu_mismatches <= 10) {
                std::cout << "Mean mismatch at index " << i << ": computed=" << mu_computed[i]
                          << ", reference=" << mu_ref(i) << ", error=" << error << std::endl;
            }
        }
    }

    float mu_avg_error = mu_sum_error / combined_batch;
    std::cout << "Mean statistics:" << std::endl;
    std::cout << "  Mismatches: " << mu_mismatches << " / " << combined_batch << std::endl;
    std::cout << "  Max error: " << mu_max_error << std::endl;
    std::cout << "  Avg error: " << mu_avg_error << std::endl;

    EXPECT_EQ(mu_mismatches, 0);

    // Test rstd accuracy
    std::cout << "\n=== Testing rstd accuracy ===" << std::endl;
    uint32_t rstd_mismatches = 0;
    float rstd_max_error = 0.0f;
    float rstd_sum_error = 0.0f;
    for (uint32_t i = 0; i < combined_batch; ++i) {
        float error = std::abs(rstd_computed[i] - rstd_ref(i));
        rstd_sum_error += error;
        rstd_max_error = std::max(rstd_max_error, error);
        if (error > tolerance) {
            rstd_mismatches++;
            if (rstd_mismatches <= 10) {
                std::cout << "Rstd mismatch at index " << i << ": computed=" << rstd_computed[i]
                          << ", reference=" << rstd_ref(i) << ", error=" << error << std::endl;
            }
        }
    }

    float rstd_avg_error = rstd_sum_error / combined_batch;
    std::cout << "Rstd statistics:" << std::endl;
    std::cout << "  Mismatches: " << rstd_mismatches << " / " << combined_batch << std::endl;
    std::cout << "  Max error: " << rstd_max_error << std::endl;
    std::cout << "  Avg error: " << rstd_avg_error << std::endl;

    EXPECT_EQ(rstd_mismatches, 0);
}

// Test Metal LayerNorm Forward vs XTensor Reference
TEST_F(LayerNormForwardOpTest, MetalLayerNormFw_MediumTensor) {
    using namespace ttml;

    uint32_t batch_size = 2;
    uint32_t seq_len = 182;
    uint32_t heads = 1;
    uint32_t features = 2083;

    std::cout << "\n=== Test: MetalLayerNormFw_SmallTensor ===" << std::endl;
    std::cout << "Hyperparameters:" << std::endl;
    std::cout << "  batch_size: " << batch_size << std::endl;
    std::cout << "  seq_len: " << seq_len << std::endl;
    std::cout << "  heads: " << heads << std::endl;
    std::cout << "  features: " << features << std::endl;
    std::cout << "  eps: 1e-5" << std::endl;

    uint32_t combined_batch = batch_size * seq_len * heads;
    uint32_t total_elements = combined_batch * features;

    std::mt19937 gen(1234);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> test_data;
    test_data.reserve(total_elements);
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

    // Compute reference using xtensor
    auto x_xtensor = xt::adapt(test_data, {combined_batch * features});
    auto gamma_xtensor = xt::adapt(gamma_data, {features});
    auto beta_xtensor = xt::adapt(beta_data, {features});

    auto [y_ref, mu_ref, rstd_ref] =
        layernorm_forward_reference_(x_xtensor, gamma_xtensor, beta_xtensor, combined_batch, features, 1e-5f);

    // Create tensors for metal operation
    auto input_tensor = core::from_vector(
        test_data, ttnn::Shape({batch_size, heads, seq_len, features}), &autograd::ctx().get_device());
    auto gamma_tensor = core::from_vector(gamma_data, ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device());
    auto beta_tensor = core::from_vector(beta_data, ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device());

    // Call metal layernorm forward
    auto output_tensors = metal::ops::layernorm_fw::LayerNormForwardOperation::invoke(
        input_tensor, gamma_tensor, beta_tensor, 1e-5f, true);

    // Get computed output
    auto y_computed = core::to_vector(output_tensors[0].value());
    auto mu_computed = core::to_vector(output_tensors[1].value());
    auto rstd_computed = core::to_vector(output_tensors[2].value());

    // Test output accuracy
    float tolerance = 5e-2f;
    uint32_t y_mismatches = 0;
    float y_max_error = 0.0f;
    float y_sum_error = 0.0f;

    std::cout << "\n=== Testing output accuracy (metal_layernorm_fw) ===" << std::endl;
    for (uint32_t i = 0; i < total_elements; ++i) {
        float error = std::abs(y_computed[i] - y_ref(i));
        y_sum_error += error;
        y_max_error = std::max(y_max_error, error);
        if (error > tolerance) {
            y_mismatches++;
            if (y_mismatches <= 10) {
                std::cout << "Mismatch at index " << i << ": computed=" << y_computed[i] << ", reference=" << y_ref(i)
                          << ", error=" << error << std::endl;
            }
        }
    }

    float y_avg_error = y_sum_error / total_elements;
    std::cout << "Output statistics:" << std::endl;
    std::cout << "  Mismatches: " << y_mismatches << " / " << total_elements << std::endl;
    std::cout << "  Max error: " << y_max_error << std::endl;
    std::cout << "  Avg error: " << y_avg_error << std::endl;

    EXPECT_LT(y_mismatches, total_elements * 0.01);  // Less than 1% mismatches
    EXPECT_LT(y_max_error, 0.1f);

    // Test mean accuracy
    std::cout << "\n=== Testing mean accuracy ===" << std::endl;
    uint32_t mu_mismatches = 0;
    float mu_max_error = 0.0f;
    float mu_sum_error = 0.0f;
    for (uint32_t i = 0; i < combined_batch; ++i) {
        float error = std::abs(mu_computed[i] - mu_ref(i));
        mu_sum_error += error;
        mu_max_error = std::max(mu_max_error, error);
        if (error > tolerance) {
            mu_mismatches++;
            if (mu_mismatches <= 10) {
                std::cout << "Mean mismatch at index " << i << ": computed=" << mu_computed[i]
                          << ", reference=" << mu_ref(i) << ", error=" << error << std::endl;
            }
        }
    }

    float mu_avg_error = mu_sum_error / combined_batch;
    std::cout << "Mean statistics:" << std::endl;
    std::cout << "  Mismatches: " << mu_mismatches << " / " << combined_batch << std::endl;
    std::cout << "  Max error: " << mu_max_error << std::endl;
    std::cout << "  Avg error: " << mu_avg_error << std::endl;

    EXPECT_EQ(mu_mismatches, 0);

    // Test rstd accuracy
    std::cout << "\n=== Testing rstd accuracy ===" << std::endl;
    uint32_t rstd_mismatches = 0;
    float rstd_max_error = 0.0f;
    float rstd_sum_error = 0.0f;
    for (uint32_t i = 0; i < combined_batch; ++i) {
        float error = std::abs(rstd_computed[i] - rstd_ref(i));
        rstd_sum_error += error;
        rstd_max_error = std::max(rstd_max_error, error);
        if (error > tolerance) {
            rstd_mismatches++;
            if (rstd_mismatches <= 10) {
                std::cout << "Rstd mismatch at index " << i << ": computed=" << rstd_computed[i]
                          << ", reference=" << rstd_ref(i) << ", error=" << error << std::endl;
            }
        }
    }

    float rstd_avg_error = rstd_sum_error / combined_batch;
    std::cout << "Rstd statistics:" << std::endl;
    std::cout << "  Mismatches: " << rstd_mismatches << " / " << combined_batch << std::endl;
    std::cout << "  Max error: " << rstd_max_error << std::endl;
    std::cout << "  Avg error: " << rstd_avg_error << std::endl;

    EXPECT_EQ(rstd_mismatches, 0);
}

// Test Metal LayerNorm Forward vs XTensor Reference
TEST_F(LayerNormForwardOpTest, MetalLayerNormFw_LargeTensor) {
    using namespace ttml;

    uint32_t batch_size = 4;
    uint32_t seq_len = 324;
    uint32_t heads = 1;
    uint32_t features = 9132;

    std::cout << "\n=== Test: MetalLayerNormFw_LargeTensor ===" << std::endl;
    std::cout << "Hyperparameters:" << std::endl;
    std::cout << "  batch_size: " << batch_size << std::endl;
    std::cout << "  seq_len: " << seq_len << std::endl;
    std::cout << "  heads: " << heads << std::endl;
    std::cout << "  features: " << features << std::endl;
    std::cout << "  eps: 1e-5" << std::endl;

    uint32_t combined_batch = batch_size * seq_len * heads;
    uint32_t total_elements = combined_batch * features;

    std::mt19937 gen(1234);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> test_data;
    test_data.reserve(total_elements);
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

    // Compute reference using xtensor
    auto x_xtensor = xt::adapt(test_data, {combined_batch * features});
    auto gamma_xtensor = xt::adapt(gamma_data, {features});
    auto beta_xtensor = xt::adapt(beta_data, {features});

    auto [y_ref, mu_ref, rstd_ref] =
        layernorm_forward_reference_(x_xtensor, gamma_xtensor, beta_xtensor, combined_batch, features, 1e-5f);

    // Create tensors for metal operation
    auto input_tensor = core::from_vector(
        test_data, ttnn::Shape({batch_size, heads, seq_len, features}), &autograd::ctx().get_device());
    auto gamma_tensor = core::from_vector(gamma_data, ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device());
    auto beta_tensor = core::from_vector(beta_data, ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device());

    // Call metal layernorm forward
    auto output_tensors = metal::ops::layernorm_fw::LayerNormForwardOperation::invoke(
        input_tensor, gamma_tensor, beta_tensor, 1e-5f, true);

    // Get computed output
    auto y_computed = core::to_vector(output_tensors[0].value());
    auto mu_computed = core::to_vector(output_tensors[1].value());
    auto rstd_computed = core::to_vector(output_tensors[2].value());

    // Test output accuracy
    float tolerance = 5e-2f;
    uint32_t y_mismatches = 0;
    float y_max_error = 0.0f;
    float y_sum_error = 0.0f;

    std::cout << "\n=== Testing output accuracy (metal_layernorm_fw) ===" << std::endl;
    for (uint32_t i = 0; i < total_elements; ++i) {
        float error = std::abs(y_computed[i] - y_ref(i));
        y_sum_error += error;
        y_max_error = std::max(y_max_error, error);
        if (error > tolerance) {
            y_mismatches++;
            if (y_mismatches <= 10) {
                std::cout << "Mismatch at index " << i << ": computed=" << y_computed[i] << ", reference=" << y_ref(i)
                          << ", error=" << error << std::endl;
            }
        }
    }

    float y_avg_error = y_sum_error / total_elements;
    std::cout << "Output statistics:" << std::endl;
    std::cout << "  Mismatches: " << y_mismatches << " / " << total_elements << std::endl;
    std::cout << "  Max error: " << y_max_error << std::endl;
    std::cout << "  Avg error: " << y_avg_error << std::endl;

    EXPECT_LT(y_mismatches, total_elements * 0.01);  // Less than 1% mismatches
    EXPECT_LT(y_max_error, 0.1f);

    // Test mean accuracy
    std::cout << "\n=== Testing mean accuracy ===" << std::endl;
    uint32_t mu_mismatches = 0;
    float mu_max_error = 0.0f;
    float mu_sum_error = 0.0f;
    for (uint32_t i = 0; i < combined_batch; ++i) {
        float error = std::abs(mu_computed[i] - mu_ref(i));
        mu_sum_error += error;
        mu_max_error = std::max(mu_max_error, error);
        if (error > tolerance) {
            mu_mismatches++;
            if (mu_mismatches <= 10) {
                std::cout << "Mean mismatch at index " << i << ": computed=" << mu_computed[i]
                          << ", reference=" << mu_ref(i) << ", error=" << error << std::endl;
            }
        }
    }

    float mu_avg_error = mu_sum_error / combined_batch;
    std::cout << "Mean statistics:" << std::endl;
    std::cout << "  Mismatches: " << mu_mismatches << " / " << combined_batch << std::endl;
    std::cout << "  Max error: " << mu_max_error << std::endl;
    std::cout << "  Avg error: " << mu_avg_error << std::endl;

    EXPECT_EQ(mu_mismatches, 0);

    // Test rstd accuracy
    std::cout << "\n=== Testing rstd accuracy ===" << std::endl;
    uint32_t rstd_mismatches = 0;
    float rstd_max_error = 0.0f;
    float rstd_sum_error = 0.0f;
    for (uint32_t i = 0; i < combined_batch; ++i) {
        float error = std::abs(rstd_computed[i] - rstd_ref(i));
        rstd_sum_error += error;
        rstd_max_error = std::max(rstd_max_error, error);
        if (error > tolerance) {
            rstd_mismatches++;
            if (rstd_mismatches <= 10) {
                std::cout << "Rstd mismatch at index " << i << ": computed=" << rstd_computed[i]
                          << ", reference=" << rstd_ref(i) << ", error=" << error << std::endl;
            }
        }
    }

    float rstd_avg_error = rstd_sum_error / combined_batch;
    std::cout << "Rstd statistics:" << std::endl;
    std::cout << "  Mismatches: " << rstd_mismatches << " / " << combined_batch << std::endl;
    std::cout << "  Max error: " << rstd_max_error << std::endl;
    std::cout << "  Avg error: " << rstd_avg_error << std::endl;

    EXPECT_EQ(rstd_mismatches, 0);
}

// Test Metal LayerNorm Forward with larger features (tests non-L1 fit case)
TEST_F(LayerNormForwardOpTest, MetalLayerNormFw_LargeFeatures) {
    using namespace ttml;

    uint32_t batch_size = 2;
    uint32_t seq_len = 8;
    uint32_t heads = 4;
    uint32_t features = 512;

    std::cout << "\n=== Test: MetalLayerNormFw_LargeFeatures ===" << std::endl;
    std::cout << "Hyperparameters:" << std::endl;
    std::cout << "  batch_size: " << batch_size << std::endl;
    std::cout << "  seq_len: " << seq_len << std::endl;
    std::cout << "  heads: " << heads << std::endl;
    std::cout << "  features: " << features << std::endl;

    uint32_t combined_batch = batch_size * seq_len * heads;
    uint32_t total_elements = combined_batch * features;

    std::mt19937 gen(5678);
    std::normal_distribution<float> dist(0.0f, 2.0f);

    std::vector<float> test_data;
    test_data.reserve(total_elements);
    for (uint32_t i = 0; i < total_elements; ++i) {
        test_data.push_back(dist(gen));
    }

    std::vector<float> gamma_data;
    std::vector<float> beta_data;
    for (uint32_t i = 0; i < features; ++i) {
        gamma_data.push_back(1.0f);
        beta_data.push_back(0.0f);
    }

    // Compute reference using xtensor
    auto x_xtensor = xt::adapt(test_data, {combined_batch * features});
    auto gamma_xtensor = xt::adapt(gamma_data, {features});
    auto beta_xtensor = xt::adapt(beta_data, {features});

    auto [y_ref, mu_ref, rstd_ref] =
        layernorm_forward_reference_(x_xtensor, gamma_xtensor, beta_xtensor, combined_batch, features, 1e-5f);

    // Create tensors for metal operation
    auto input_tensor = core::from_vector(
        test_data, ttnn::Shape({batch_size, heads, seq_len, features}), &autograd::ctx().get_device());
    auto gamma_tensor = core::from_vector(gamma_data, ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device());
    auto beta_tensor = core::from_vector(beta_data, ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device());

    // Call metal layernorm forward
    auto output_tensors = metal::ops::layernorm_fw::LayerNormForwardOperation::invoke(
        input_tensor, gamma_tensor, beta_tensor, 1e-5f, true);

    // Get computed output
    auto y_computed = core::to_vector(output_tensors[0].value());

    // Test output accuracy
    float tolerance = 5e-2f;
    uint32_t y_mismatches = 0;
    float y_max_error = 0.0f;
    float y_sum_error = 0.0f;

    for (uint32_t i = 0; i < total_elements; ++i) {
        float error = std::abs(y_computed[i] - y_ref(i));
        y_sum_error += error;
        y_max_error = std::max(y_max_error, error);
        if (error > tolerance) {
            y_mismatches++;
        }
    }

    float y_avg_error = y_sum_error / total_elements;
    std::cout << "Output statistics:" << std::endl;
    std::cout << "  Mismatches: " << y_mismatches << " / " << total_elements << std::endl;
    std::cout << "  Max error: " << y_max_error << std::endl;
    std::cout << "  Avg error: " << y_avg_error << std::endl;

    EXPECT_LT(y_mismatches, total_elements * 0.01);  // Less than 1% mismatches
    EXPECT_LT(y_max_error, 0.1f);
}

// Test Moreh LayerNorm Forward vs XTensor Reference
TEST_F(LayerNormForwardOpTest, MorehLayerNormFw_VsReference) {
    using namespace ttml;

    uint32_t batch_size = 2;
    uint32_t seq_len = 4;
    uint32_t heads = 2;
    uint32_t features = 256;

    std::cout << "\n=== Test: MorehLayerNormFw_VsReference ===" << std::endl;

    uint32_t combined_batch = batch_size * seq_len * heads;
    uint32_t total_elements = combined_batch * features;

    std::mt19937 gen(9012);
    std::normal_distribution<float> dist(0.0f, 1.5f);

    std::vector<float> test_data;
    test_data.reserve(total_elements);
    for (uint32_t i = 0; i < total_elements; ++i) {
        test_data.push_back(dist(gen));
    }

    std::vector<float> gamma_data;
    std::vector<float> beta_data;
    std::normal_distribution<float> param_dist(1.0f, 0.1f);
    for (uint32_t i = 0; i < features; ++i) {
        gamma_data.push_back(param_dist(gen));
        beta_data.push_back(dist(gen) * 0.05f);
    }

    // Compute reference using xtensor (moreh uses eps=1e-6)
    auto x_xtensor = xt::adapt(test_data, {combined_batch * features});
    auto gamma_xtensor = xt::adapt(gamma_data, {features});
    auto beta_xtensor = xt::adapt(beta_data, {features});

    auto [y_ref, mu_ref, rstd_ref] =
        layernorm_forward_reference_(x_xtensor, gamma_xtensor, beta_xtensor, combined_batch, features, 1e-6f);

    // Create tensors for moreh operation
    auto input_tensor = core::from_vector(
        test_data, ttnn::Shape({batch_size, heads, seq_len, features}), &autograd::ctx().get_device());
    auto gamma_tensor = core::from_vector(gamma_data, ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device());
    auto beta_tensor = core::from_vector(beta_data, ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device());

    // Call moreh layernorm forward
    auto mean = core::empty(
        ttnn::Shape({batch_size, heads, seq_len, 1}), &autograd::ctx().get_device(), input_tensor.memory_config());
    auto rstd = ttnn::empty_like(mean);
    auto output = ttnn::empty_like(input_tensor);

    auto out_tensors = ttnn::moreh_layer_norm(
        input_tensor, 1, 1e-6F, gamma_tensor, beta_tensor, output, mean, rstd, std::nullopt, std::nullopt);

    // Get computed output
    auto y_computed = core::to_vector(out_tensors[0].value());

    // Test output accuracy
    float tolerance = 5e-2f;
    uint32_t y_mismatches = 0;
    float y_max_error = 0.0f;
    float y_sum_error = 0.0f;

    std::cout << "\n=== Testing output accuracy (moreh_layer_norm) ===" << std::endl;
    for (uint32_t i = 0; i < total_elements; ++i) {
        float error = std::abs(y_computed[i] - y_ref(i));
        y_sum_error += error;
        y_max_error = std::max(y_max_error, error);
        if (error > tolerance) {
            y_mismatches++;
            if (y_mismatches <= 10) {
                std::cout << "Mismatch at index " << i << ": computed=" << y_computed[i] << ", reference=" << y_ref(i)
                          << ", error=" << error << std::endl;
            }
        }
    }

    float y_avg_error = y_sum_error / total_elements;
    std::cout << "Output statistics:" << std::endl;
    std::cout << "  Mismatches: " << y_mismatches << " / " << total_elements << std::endl;
    std::cout << "  Max error: " << y_max_error << std::endl;
    std::cout << "  Avg error: " << y_avg_error << std::endl;

    EXPECT_LT(y_mismatches, total_elements * 0.01);
    EXPECT_LT(y_max_error, 0.1f);
}

// Test Composite LayerNorm Forward vs XTensor Reference
TEST_F(LayerNormForwardOpTest, CompositeLayerNormFw_VsReference) {
    using namespace ttml;

    uint32_t batch_size = 1;
    uint32_t seq_len = 8;
    uint32_t heads = 4;
    uint32_t features = 192;

    std::cout << "\n=== Test: CompositeLayerNormFw_VsReference ===" << std::endl;

    uint32_t combined_batch = batch_size * seq_len * heads;
    uint32_t total_elements = combined_batch * features;

    std::mt19937 gen(3456);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> test_data;
    test_data.reserve(total_elements);
    for (uint32_t i = 0; i < total_elements; ++i) {
        test_data.push_back(dist(gen));
    }

    std::vector<float> gamma_data;
    std::vector<float> beta_data;
    for (uint32_t i = 0; i < features; ++i) {
        gamma_data.push_back(1.0f);
        beta_data.push_back(0.0f);
    }

    // Compute reference using xtensor (composite uses eps=1e-6)
    auto x_xtensor = xt::adapt(test_data, {combined_batch * features});
    auto gamma_xtensor = xt::adapt(gamma_data, {features});
    auto beta_xtensor = xt::adapt(beta_data, {features});

    auto [y_ref, mu_ref, rstd_ref] =
        layernorm_forward_reference_(x_xtensor, gamma_xtensor, beta_xtensor, combined_batch, features, 1e-6f);

    // Create tensors for composite operation
    auto input_tensor = autograd::create_tensor(core::from_vector(
        test_data, ttnn::Shape({batch_size, heads, seq_len, features}), &autograd::ctx().get_device()));
    auto gamma_tensor = autograd::create_tensor(
        core::from_vector(gamma_data, ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device()));
    auto beta_tensor = autograd::create_tensor(
        core::from_vector(beta_data, ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device()));

    // Call composite layernorm forward
    auto output_tensor = ops::composite_layernorm(input_tensor, gamma_tensor, beta_tensor);

    // Get computed output
    auto y_computed = core::to_vector(output_tensor->get_value());

    // Test output accuracy
    float tolerance = 5e-2f;
    uint32_t y_mismatches = 0;
    float y_max_error = 0.0f;
    float y_sum_error = 0.0f;

    std::cout << "\n=== Testing output accuracy (composite_layernorm) ===" << std::endl;
    for (uint32_t i = 0; i < total_elements; ++i) {
        float error = std::abs(y_computed[i] - y_ref(i));
        y_sum_error += error;
        y_max_error = std::max(y_max_error, error);
        if (error > tolerance) {
            y_mismatches++;
            if (y_mismatches <= 10) {
                std::cout << "Mismatch at index " << i << ": computed=" << y_computed[i] << ", reference=" << y_ref(i)
                          << ", error=" << error << std::endl;
            }
        }
    }

    float y_avg_error = y_sum_error / total_elements;
    std::cout << "Output statistics:" << std::endl;
    std::cout << "  Mismatches: " << y_mismatches << " / " << total_elements << std::endl;
    std::cout << "  Max error: " << y_max_error << std::endl;
    std::cout << "  Avg error: " << y_avg_error << std::endl;

    EXPECT_LT(y_mismatches, total_elements * 0.01);
    EXPECT_LT(y_max_error, 0.1f);
}
