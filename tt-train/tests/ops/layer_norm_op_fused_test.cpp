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
    float eps = 1e-6f) {
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

TEST_F(LayerNormFusedOpTest, MetalLayerNormBw_VariedInputs) {
    using namespace ttml;

    uint32_t batch_size = 1;  // Increased to provide sufficient work
    uint32_t seq_len = 32;    // Must be divisible by 32 (tile size)
    uint32_t heads = 1;
    uint32_t features = 32;  // Must be divisible by 32 (tile size)

    std::vector<float> test_data;
    std::mt19937 gen(1213);
    std::normal_distribution<float> dist(0.0f, 2.0f);  // Larger variance

    uint32_t total_elements = batch_size * seq_len * heads * features;
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
        layernorm_forward_reference(test_data, gamma_data, beta_data, combined_batch, features, 1e-6f);
    auto [dx_ref, dgamma_ref, dbeta_ref] = layernorm_backward_reference(dy_data, cache);

    auto input_tensor = core::from_vector(
        test_data, ttnn::Shape({batch_size, heads, seq_len, features}), &autograd::ctx().get_device());
    auto gamma_tensor = core::from_vector(gamma_data, ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device());
    auto x_hat_tensor = core::from_vector(
        cache.x_hat, ttnn::Shape({batch_size, heads, seq_len, features}), &autograd::ctx().get_device());

    std::cout << "Gamma tensor values:" << std::endl;
    auto gamma_host_data = core::to_vector(gamma_tensor);
    for (size_t i = 0; i < gamma_host_data.size(); ++i) {
        std::cout << gamma_host_data[i] << " ";
    }

    std::vector<float> rstd_data;
    for (uint32_t b = 0; b < combined_batch; ++b) {
        rstd_data.push_back(1.0f / cache.s[b]);
    }
    auto rstd_tensor =
        core::from_vector(rstd_data, ttnn::Shape({batch_size, heads, seq_len, 1}), &autograd::ctx().get_device());
    auto dy_tensor =
        core::from_vector(dy_data, ttnn::Shape({batch_size, heads, seq_len, features}), &autograd::ctx().get_device());

    std::cout << "dy tensor values:" << std::endl;
    auto dy_host_data = core::to_vector(dy_tensor);
    for (size_t i = 0; i < dy_host_data.size(); ++i) {
        std::cout << dy_host_data[i] << " ";
    }
    std::cout << std::endl;

    auto output_tensors = metal::ops::layernorm_bw::LayerNormBackwardOperation::invoke(
        input_tensor, gamma_tensor, x_hat_tensor, rstd_tensor, dy_tensor);

    auto metal_dx = core::to_vector(output_tensors[0].value());
    auto metal_dgamma = core::to_vector(output_tensors[1].value());

    // float tolerance = 3e-2f;

    // for (uint32_t i = 0; i < total_elements; ++i) {
    //     EXPECT_NEAR(metal_dx[i], dx_ref[i], tolerance)
    //         << "Input gradient mismatch at index " << i;
    // }

    // for (uint32_t i = 0; i < features; ++i) {
    //     EXPECT_NEAR(metal_dgamma[i], dgamma_ref[i], tolerance)
    //         << "Gamma gradient mismatch at index " << i;
    // }
}

// // ============================================================================
// // Large-scale tests for non-L1 code path (data doesn't fit in L1)
// // ============================================================================

// TEST_F(LayerNormFusedOpTest, MetalLayerNormBw_LargeFeatures_NoL1Fit) {
//     using namespace ttml;

//     uint32_t batch_size = 4;
//     uint32_t seq_len = 64;
//     uint32_t heads = 1;
//     uint32_t features = 1024;  // Large features to exceed L1 capacity

//     std::vector<float> test_data;
//     std::mt19937 gen(2024);
//     std::normal_distribution<float> dist(0.0f, 1.0f);

//     uint32_t total_elements = batch_size * seq_len * heads * features;
//     test_data.reserve(total_elements);
//     for (uint32_t i = 0; i < total_elements; ++i) {
//         test_data.push_back(dist(gen));
//     }

//     std::vector<float> gamma_data;
//     std::vector<float> beta_data;
//     for (uint32_t i = 0; i < features; ++i) {
//         gamma_data.push_back(1.0f + dist(gen) * 0.1f);
//         beta_data.push_back(dist(gen) * 0.1f);
//     }

//     std::vector<float> dy_data;
//     std::normal_distribution<float> grad_dist(0.0f, 0.1f);
//     for (uint32_t i = 0; i < total_elements; ++i) {
//         dy_data.push_back(grad_dist(gen));
//     }

//     // Run reference forward pass
//     uint32_t combined_batch = batch_size * seq_len * heads;
//     auto [y_ref, cache] = layernorm_forward_reference(test_data, gamma_data, beta_data, combined_batch, features,
//     1e-6f); auto [dx_ref, dgamma_ref, dbeta_ref] = layernorm_backward_reference(dy_data, cache);

//     // Prepare metal tensors
//     auto input_tensor = core::from_vector(test_data, ttnn::Shape({batch_size, heads, seq_len, features}),
//     &autograd::ctx().get_device()); auto gamma_tensor = core::from_vector(gamma_data, ttnn::Shape({1, 1, 1,
//     features}), &autograd::ctx().get_device()); auto x_hat_tensor = core::from_vector(cache.x_hat,
//     ttnn::Shape({batch_size, heads, seq_len, features}), &autograd::ctx().get_device());

//     std::vector<float> rstd_data;
//     for (uint32_t b = 0; b < combined_batch; ++b) {
//         rstd_data.push_back(1.0f / cache.s[b]);
//     }
//     auto rstd_tensor = core::from_vector(rstd_data, ttnn::Shape({batch_size, heads, seq_len, 1}),
//     &autograd::ctx().get_device()); auto dy_tensor = core::from_vector(dy_data, ttnn::Shape({batch_size, heads,
//     seq_len, features}), &autograd::ctx().get_device());

//     // Call metal operation
//     auto output_tensors = metal::ops::layernorm_bw::LayerNormBackwardOperation::invoke(
//         input_tensor, gamma_tensor, x_hat_tensor, rstd_tensor, dy_tensor);

//     ASSERT_GE(output_tensors.size(), 2);
//     auto metal_dx = core::to_vector(output_tensors[0].value());
//     auto metal_dgamma = core::to_vector(output_tensors[1].value());

//     float tolerance = 3e-2f;

//     // Sample check - testing subset for large data
//     for (uint32_t i = 0; i < std::min(200u, total_elements); ++i) {
//         EXPECT_NEAR(metal_dx[i], dx_ref[i], tolerance)
//             << "Input gradient mismatch at index " << i;
//     }

//     for (uint32_t i = 0; i < features; ++i) {
//         EXPECT_NEAR(metal_dgamma[i], dgamma_ref[i], tolerance)
//             << "Gamma gradient mismatch at index " << i;
//     }
// }

// TEST_F(LayerNormFusedOpTest, MetalLayerNormBw_VeryLargeFeatures_NoL1Fit) {
//     using namespace ttml;

//     uint32_t batch_size = 2;
//     uint32_t seq_len = 32;
//     uint32_t heads = 1;
//     uint32_t features = 2048;  // Very large features - definitely won't fit in L1

//     std::vector<float> test_data;
//     std::mt19937 gen(2025);
//     std::normal_distribution<float> dist(0.0f, 1.0f);

//     uint32_t total_elements = batch_size * seq_len * heads * features;
//     test_data.reserve(total_elements);
//     for (uint32_t i = 0; i < total_elements; ++i) {
//         test_data.push_back(dist(gen));
//     }

//     std::vector<float> gamma_data;
//     std::vector<float> beta_data;
//     for (uint32_t i = 0; i < features; ++i) {
//         gamma_data.push_back(1.0f + dist(gen) * 0.1f);
//         beta_data.push_back(dist(gen) * 0.1f);
//     }

//     std::vector<float> dy_data;
//     std::normal_distribution<float> grad_dist(0.0f, 0.1f);
//     for (uint32_t i = 0; i < total_elements; ++i) {
//         dy_data.push_back(grad_dist(gen));
//     }

//     // Run reference
//     uint32_t combined_batch = batch_size * seq_len * heads;
//     auto [y_ref, cache] = layernorm_forward_reference(test_data, gamma_data, beta_data, combined_batch, features,
//     1e-6f); auto [dx_ref, dgamma_ref, dbeta_ref] = layernorm_backward_reference(dy_data, cache);

//     // Prepare metal tensors
//     auto input_tensor = core::from_vector(test_data, ttnn::Shape({batch_size, heads, seq_len, features}),
//     &autograd::ctx().get_device()); auto gamma_tensor = core::from_vector(gamma_data, ttnn::Shape({1, 1, 1,
//     features}), &autograd::ctx().get_device()); auto x_hat_tensor = core::from_vector(cache.x_hat,
//     ttnn::Shape({batch_size, heads, seq_len, features}), &autograd::ctx().get_device());

//     std::vector<float> rstd_data;
//     for (uint32_t b = 0; b < combined_batch; ++b) {
//         rstd_data.push_back(1.0f / cache.s[b]);
//     }
//     auto rstd_tensor = core::from_vector(rstd_data, ttnn::Shape({batch_size, heads, seq_len, 1}),
//     &autograd::ctx().get_device()); auto dy_tensor = core::from_vector(dy_data, ttnn::Shape({batch_size, heads,
//     seq_len, features}), &autograd::ctx().get_device());

//     auto output_tensors = metal::ops::layernorm_bw::LayerNormBackwardOperation::invoke(
//         input_tensor, gamma_tensor, x_hat_tensor, rstd_tensor, dy_tensor);

//     auto metal_dx = core::to_vector(output_tensors[0].value());
//     auto metal_dgamma = core::to_vector(output_tensors[1].value());

//     float tolerance = 3e-2f;

//     // Sample check on a subset
//     for (uint32_t i = 0; i < std::min(200u, total_elements); ++i) {
//         EXPECT_NEAR(metal_dx[i], dx_ref[i], tolerance);
//     }

//     for (uint32_t i = 0; i < features; ++i) {
//         EXPECT_NEAR(metal_dgamma[i], dgamma_ref[i], tolerance);
//     }
// }

// TEST_F(LayerNormFusedOpTest, MetalLayerNormBw_LargeSequenceLength_NoL1Fit) {
//     using namespace ttml;

//     uint32_t batch_size = 2;
//     uint32_t seq_len = 256;  // Large sequence length
//     uint32_t heads = 1;
//     uint32_t features = 512;  // Combined with features, won't fit in L1

//     std::vector<float> test_data;
//     std::mt19937 gen(2026);
//     std::normal_distribution<float> dist(0.0f, 1.0f);

//     uint32_t total_elements = batch_size * seq_len * heads * features;
//     test_data.reserve(total_elements);
//     for (uint32_t i = 0; i < total_elements; ++i) {
//         test_data.push_back(dist(gen));
//     }

//     std::vector<float> gamma_data;
//     std::vector<float> beta_data;
//     for (uint32_t i = 0; i < features; ++i) {
//         gamma_data.push_back(1.0f + dist(gen) * 0.1f);
//         beta_data.push_back(dist(gen) * 0.1f);
//     }

//     std::vector<float> dy_data;
//     std::normal_distribution<float> grad_dist(0.0f, 0.1f);
//     for (uint32_t i = 0; i < total_elements; ++i) {
//         dy_data.push_back(grad_dist(gen));
//     }

//     // Run reference
//     uint32_t combined_batch = batch_size * seq_len * heads;
//     auto [y_ref, cache] = layernorm_forward_reference(test_data, gamma_data, beta_data, combined_batch, features,
//     1e-6f); auto [dx_ref, dgamma_ref, dbeta_ref] = layernorm_backward_reference(dy_data, cache);

//     // Prepare metal tensors
//     auto input_tensor = core::from_vector(test_data, ttnn::Shape({batch_size, heads, seq_len, features}),
//     &autograd::ctx().get_device()); auto gamma_tensor = core::from_vector(gamma_data, ttnn::Shape({1, 1, 1,
//     features}), &autograd::ctx().get_device()); auto x_hat_tensor = core::from_vector(cache.x_hat,
//     ttnn::Shape({batch_size, heads, seq_len, features}), &autograd::ctx().get_device());

//     std::vector<float> rstd_data;
//     for (uint32_t b = 0; b < combined_batch; ++b) {
//         rstd_data.push_back(1.0f / cache.s[b]);
//     }
//     auto rstd_tensor = core::from_vector(rstd_data, ttnn::Shape({batch_size, heads, seq_len, 1}),
//     &autograd::ctx().get_device()); auto dy_tensor = core::from_vector(dy_data, ttnn::Shape({batch_size, heads,
//     seq_len, features}), &autograd::ctx().get_device());

//     auto output_tensors = metal::ops::layernorm_bw::LayerNormBackwardOperation::invoke(
//         input_tensor, gamma_tensor, x_hat_tensor, rstd_tensor, dy_tensor);

//     auto metal_dx = core::to_vector(output_tensors[0].value());
//     auto metal_dgamma = core::to_vector(output_tensors[1].value());

//     float tolerance = 3e-2f;

//     // Sample check
//     for (uint32_t i = 0; i < std::min(200u, total_elements); ++i) {
//         EXPECT_NEAR(metal_dx[i], dx_ref[i], tolerance);
//     }

//     for (uint32_t i = 0; i < features; ++i) {
//         EXPECT_NEAR(metal_dgamma[i], dgamma_ref[i], tolerance);
//     }
// }

// TEST_F(LayerNormFusedOpTest, MetalLayerNormBw_LargeBatchAndSeq_NoL1Fit) {
//     using namespace ttml;

//     uint32_t batch_size = 8;
//     uint32_t seq_len = 1024;
//     uint32_t heads = 1;
//     uint32_t features = 512;  // Large total workload

//     std::vector<float> test_data;
//     std::mt19937 gen(2027);
//     std::normal_distribution<float> dist(0.0f, 1.0f);

//     uint32_t total_elements = batch_size * seq_len * heads * features;
//     test_data.reserve(total_elements);
//     for (uint32_t i = 0; i < total_elements; ++i) {
//         test_data.push_back(dist(gen));
//     }

//     std::vector<float> gamma_data;
//     std::vector<float> beta_data;
//     for (uint32_t i = 0; i < features; ++i) {
//         gamma_data.push_back(1.0f + dist(gen) * 0.1f);
//         beta_data.push_back(dist(gen) * 0.1f);
//     }

//     std::vector<float> dy_data;
//     std::normal_distribution<float> grad_dist(0.0f, 0.1f);
//     for (uint32_t i = 0; i < total_elements; ++i) {
//         dy_data.push_back(grad_dist(gen));
//     }

//     // Run reference
//     uint32_t combined_batch = batch_size * seq_len * heads;
//     auto [y_ref, cache] = layernorm_forward_reference(test_data, gamma_data, beta_data, combined_batch, features,
//     1e-6f); auto [dx_ref, dgamma_ref, dbeta_ref] = layernorm_backward_reference(dy_data, cache);

//     // Prepare metal tensors
//     auto input_tensor = core::from_vector(test_data, ttnn::Shape({batch_size, heads, seq_len, features}),
//     &autograd::ctx().get_device()); auto gamma_tensor = core::from_vector(gamma_data, ttnn::Shape({1, 1, 1,
//     features}), &autograd::ctx().get_device()); auto x_hat_tensor = core::from_vector(cache.x_hat,
//     ttnn::Shape({batch_size, heads, seq_len, features}), &autograd::ctx().get_device());

//     std::vector<float> rstd_data;
//     for (uint32_t b = 0; b < combined_batch; ++b) {
//         rstd_data.push_back(1.0f / cache.s[b]);
//     }
//     auto rstd_tensor = core::from_vector(rstd_data, ttnn::Shape({batch_size, heads, seq_len, 1}),
//     &autograd::ctx().get_device()); auto dy_tensor = core::from_vector(dy_data, ttnn::Shape({batch_size, heads,
//     seq_len, features}), &autograd::ctx().get_device());

//     auto output_tensors = metal::ops::layernorm_bw::LayerNormBackwardOperation::invoke(
//         input_tensor, gamma_tensor, x_hat_tensor, rstd_tensor, dy_tensor);

//     auto metal_dx = core::to_vector(output_tensors[0].value());
//     auto metal_dgamma = core::to_vector(output_tensors[1].value());

//     float tolerance = 3e-2f;

//     // Sample check on subset
//     for (uint32_t i = 0; i < std::min(200u, total_elements); ++i) {
//         EXPECT_NEAR(metal_dx[i], dx_ref[i], tolerance);
//     }

//     for (uint32_t i = 0; i < features; ++i) {
//         EXPECT_NEAR(metal_dgamma[i], dgamma_ref[i], tolerance);
//     }
// }

// // ============================================================================
// // Additional Large-scale tests for various configurations
// // ============================================================================

// TEST_F(LayerNormFusedOpTest, MetalLayerNormBw_MultiHead_LargeFeatures) {
//     using namespace ttml;

//     uint32_t batch_size = 4;
//     uint32_t seq_len = 128;
//     uint32_t heads = 8;  // Multiple heads
//     uint32_t features = 1024;

//     std::vector<float> test_data;
//     std::mt19937 gen(3001);
//     std::normal_distribution<float> dist(0.0f, 1.0f);

//     uint32_t total_elements = batch_size * seq_len * heads * features;
//     test_data.reserve(total_elements);
//     for (uint32_t i = 0; i < total_elements; ++i) {
//         test_data.push_back(dist(gen));
//     }

//     std::vector<float> gamma_data;
//     std::vector<float> beta_data;
//     for (uint32_t i = 0; i < features; ++i) {
//         gamma_data.push_back(1.0f + dist(gen) * 0.1f);
//         beta_data.push_back(dist(gen) * 0.1f);
//     }

//     std::vector<float> dy_data;
//     std::normal_distribution<float> grad_dist(0.0f, 0.1f);
//     for (uint32_t i = 0; i < total_elements; ++i) {
//         dy_data.push_back(grad_dist(gen));
//     }

//     // Run reference
//     uint32_t combined_batch = batch_size * seq_len * heads;
//     auto [y_ref, cache] = layernorm_forward_reference(test_data, gamma_data, beta_data, combined_batch, features,
//     1e-6f); auto [dx_ref, dgamma_ref, dbeta_ref] = layernorm_backward_reference(dy_data, cache);

//     // Prepare metal tensors
//     auto input_tensor = core::from_vector(test_data, ttnn::Shape({batch_size, heads, seq_len, features}),
//     &autograd::ctx().get_device()); auto gamma_tensor = core::from_vector(gamma_data, ttnn::Shape({1, 1, 1,
//     features}), &autograd::ctx().get_device()); auto x_hat_tensor = core::from_vector(cache.x_hat,
//     ttnn::Shape({batch_size, heads, seq_len, features}), &autograd::ctx().get_device());

//     std::vector<float> rstd_data;
//     for (uint32_t b = 0; b < combined_batch; ++b) {
//         rstd_data.push_back(1.0f / cache.s[b]);
//     }
//     auto rstd_tensor = core::from_vector(rstd_data, ttnn::Shape({batch_size, heads, seq_len, 1}),
//     &autograd::ctx().get_device()); auto dy_tensor = core::from_vector(dy_data, ttnn::Shape({batch_size, heads,
//     seq_len, features}), &autograd::ctx().get_device());

//     auto output_tensors = metal::ops::layernorm_bw::LayerNormBackwardOperation::invoke(
//         input_tensor, gamma_tensor, x_hat_tensor, rstd_tensor, dy_tensor);

//     auto metal_dx = core::to_vector(output_tensors[0].value());
//     auto metal_dgamma = core::to_vector(output_tensors[1].value());

//     float tolerance = 3e-2f;

//     // Sample check
//     for (uint32_t i = 0; i < std::min(500u, total_elements); ++i) {
//         EXPECT_NEAR(metal_dx[i], dx_ref[i], tolerance)
//             << "Input gradient mismatch at index " << i;
//     }

//     for (uint32_t i = 0; i < features; ++i) {
//         EXPECT_NEAR(metal_dgamma[i], dgamma_ref[i], tolerance)
//             << "Gamma gradient mismatch at index " << i;
//     }
// }

// TEST_F(LayerNormFusedOpTest, MetalLayerNormBw_GPT3_Small_Config) {
//     using namespace ttml;

//     // GPT-3 Small: 12 layers, 12 heads, 768 hidden size, 2048 context
//     uint32_t batch_size = 4;
//     uint32_t seq_len = 512;  // Reduced from 2048 for memory
//     uint32_t heads = 12;
//     uint32_t features = 768;

//     std::vector<float> test_data;
//     std::mt19937 gen(3002);
//     std::normal_distribution<float> dist(0.0f, 0.8f);

//     uint32_t total_elements = batch_size * seq_len * heads * features;
//     test_data.reserve(total_elements);
//     for (uint32_t i = 0; i < total_elements; ++i) {
//         test_data.push_back(dist(gen));
//     }

//     std::vector<float> gamma_data;
//     std::vector<float> beta_data;
//     for (uint32_t i = 0; i < features; ++i) {
//         gamma_data.push_back(1.0f + dist(gen) * 0.1f);
//         beta_data.push_back(dist(gen) * 0.05f);
//     }

//     std::vector<float> dy_data;
//     std::normal_distribution<float> grad_dist(0.0f, 0.1f);
//     for (uint32_t i = 0; i < total_elements; ++i) {
//         dy_data.push_back(grad_dist(gen));
//     }

//     // Run reference
//     uint32_t combined_batch = batch_size * seq_len * heads;
//     auto [y_ref, cache] = layernorm_forward_reference(test_data, gamma_data, beta_data, combined_batch, features,
//     1e-6f); auto [dx_ref, dgamma_ref, dbeta_ref] = layernorm_backward_reference(dy_data, cache);

//     // Prepare metal tensors
//     auto input_tensor = core::from_vector(test_data, ttnn::Shape({batch_size, heads, seq_len, features}),
//     &autograd::ctx().get_device()); auto gamma_tensor = core::from_vector(gamma_data, ttnn::Shape({1, 1, 1,
//     features}), &autograd::ctx().get_device()); auto x_hat_tensor = core::from_vector(cache.x_hat,
//     ttnn::Shape({batch_size, heads, seq_len, features}), &autograd::ctx().get_device());

//     std::vector<float> rstd_data;
//     for (uint32_t b = 0; b < combined_batch; ++b) {
//         rstd_data.push_back(1.0f / cache.s[b]);
//     }
//     auto rstd_tensor = core::from_vector(rstd_data, ttnn::Shape({batch_size, heads, seq_len, 1}),
//     &autograd::ctx().get_device()); auto dy_tensor = core::from_vector(dy_data, ttnn::Shape({batch_size, heads,
//     seq_len, features}), &autograd::ctx().get_device());

//     auto output_tensors = metal::ops::layernorm_bw::LayerNormBackwardOperation::invoke(
//         input_tensor, gamma_tensor, x_hat_tensor, rstd_tensor, dy_tensor);

//     auto metal_dx = core::to_vector(output_tensors[0].value());
//     auto metal_dgamma = core::to_vector(output_tensors[1].value());

//     float tolerance = 3e-2f;

//     // Sample check
//     for (uint32_t i = 0; i < std::min(500u, total_elements); ++i) {
//         EXPECT_NEAR(metal_dx[i], dx_ref[i], tolerance);
//     }

//     for (uint32_t i = 0; i < features; ++i) {
//         EXPECT_NEAR(metal_dgamma[i], dgamma_ref[i], tolerance);
//     }
// }

// TEST_F(LayerNormFusedOpTest, MetalLayerNormBw_LLaMA_Style_Config) {
//     using namespace ttml;

//     // LLaMA-style: 32 heads, 4096 hidden size
//     uint32_t batch_size = 2;
//     uint32_t seq_len = 128;  // Reduced for memory
//     uint32_t heads = 32;
//     uint32_t features = 4096;

//     std::vector<float> test_data;
//     std::mt19937 gen(3003);
//     std::normal_distribution<float> dist(0.0f, 0.8f);

//     uint32_t total_elements = batch_size * seq_len * heads * features;
//     test_data.reserve(total_elements);
//     for (uint32_t i = 0; i < total_elements; ++i) {
//         test_data.push_back(dist(gen));
//     }

//     std::vector<float> gamma_data;
//     std::vector<float> beta_data;
//     for (uint32_t i = 0; i < features; ++i) {
//         gamma_data.push_back(1.0f + dist(gen) * 0.1f);
//         beta_data.push_back(dist(gen) * 0.05f);
//     }

//     std::vector<float> dy_data;
//     std::normal_distribution<float> grad_dist(0.0f, 0.1f);
//     for (uint32_t i = 0; i < total_elements; ++i) {
//         dy_data.push_back(grad_dist(gen));
//     }

//     // Run reference
//     uint32_t combined_batch = batch_size * seq_len * heads;
//     auto [y_ref, cache] = layernorm_forward_reference(test_data, gamma_data, beta_data, combined_batch, features,
//     1e-6f); auto [dx_ref, dgamma_ref, dbeta_ref] = layernorm_backward_reference(dy_data, cache);

//     // Prepare metal tensors
//     auto input_tensor = core::from_vector(test_data, ttnn::Shape({batch_size, heads, seq_len, features}),
//     &autograd::ctx().get_device()); auto gamma_tensor = core::from_vector(gamma_data, ttnn::Shape({1, 1, 1,
//     features}), &autograd::ctx().get_device()); auto x_hat_tensor = core::from_vector(cache.x_hat,
//     ttnn::Shape({batch_size, heads, seq_len, features}), &autograd::ctx().get_device());

//     std::vector<float> rstd_data;
//     for (uint32_t b = 0; b < combined_batch; ++b) {
//         rstd_data.push_back(1.0f / cache.s[b]);
//     }
//     auto rstd_tensor = core::from_vector(rstd_data, ttnn::Shape({batch_size, heads, seq_len, 1}),
//     &autograd::ctx().get_device()); auto dy_tensor = core::from_vector(dy_data, ttnn::Shape({batch_size, heads,
//     seq_len, features}), &autograd::ctx().get_device());

//     auto output_tensors = metal::ops::layernorm_bw::LayerNormBackwardOperation::invoke(
//         input_tensor, gamma_tensor, x_hat_tensor, rstd_tensor, dy_tensor);

//     auto metal_dx = core::to_vector(output_tensors[0].value());
//     auto metal_dgamma = core::to_vector(output_tensors[1].value());

//     float tolerance = 4e-2f;  // Slightly relaxed for very large dimensions

//     // Sample check
//     for (uint32_t i = 0; i < std::min(500u, total_elements); ++i) {
//         EXPECT_NEAR(metal_dx[i], dx_ref[i], tolerance);
//     }

//     for (uint32_t i = 0; i < features; ++i) {
//         EXPECT_NEAR(metal_dgamma[i], dgamma_ref[i], tolerance);
//     }
// }

// TEST_F(LayerNormFusedOpTest, MetalLayerNormBw_VeryLargeBatch) {
//     using namespace ttml;

//     uint32_t batch_size = 32;  // Very large batch
//     uint32_t seq_len = 128;
//     uint32_t heads = 4;
//     uint32_t features = 512;

//     std::vector<float> test_data;
//     std::mt19937 gen(3004);
//     std::normal_distribution<float> dist(0.0f, 1.0f);

//     uint32_t total_elements = batch_size * seq_len * heads * features;
//     test_data.reserve(total_elements);
//     for (uint32_t i = 0; i < total_elements; ++i) {
//         test_data.push_back(dist(gen));
//     }

//     std::vector<float> gamma_data;
//     std::vector<float> beta_data;
//     for (uint32_t i = 0; i < features; ++i) {
//         gamma_data.push_back(1.0f + dist(gen) * 0.1f);
//         beta_data.push_back(dist(gen) * 0.1f);
//     }

//     std::vector<float> dy_data;
//     std::normal_distribution<float> grad_dist(0.0f, 0.1f);
//     for (uint32_t i = 0; i < total_elements; ++i) {
//         dy_data.push_back(grad_dist(gen));
//     }

//     // Run reference
//     uint32_t combined_batch = batch_size * seq_len * heads;
//     auto [y_ref, cache] = layernorm_forward_reference(test_data, gamma_data, beta_data, combined_batch, features,
//     1e-6f); auto [dx_ref, dgamma_ref, dbeta_ref] = layernorm_backward_reference(dy_data, cache);

//     // Prepare metal tensors
//     auto input_tensor = core::from_vector(test_data, ttnn::Shape({batch_size, heads, seq_len, features}),
//     &autograd::ctx().get_device()); auto gamma_tensor = core::from_vector(gamma_data, ttnn::Shape({1, 1, 1,
//     features}), &autograd::ctx().get_device()); auto x_hat_tensor = core::from_vector(cache.x_hat,
//     ttnn::Shape({batch_size, heads, seq_len, features}), &autograd::ctx().get_device());

//     std::vector<float> rstd_data;
//     for (uint32_t b = 0; b < combined_batch; ++b) {
//         rstd_data.push_back(1.0f / cache.s[b]);
//     }
//     auto rstd_tensor = core::from_vector(rstd_data, ttnn::Shape({batch_size, heads, seq_len, 1}),
//     &autograd::ctx().get_device()); auto dy_tensor = core::from_vector(dy_data, ttnn::Shape({batch_size, heads,
//     seq_len, features}), &autograd::ctx().get_device());

//     auto output_tensors = metal::ops::layernorm_bw::LayerNormBackwardOperation::invoke(
//         input_tensor, gamma_tensor, x_hat_tensor, rstd_tensor, dy_tensor);

//     auto metal_dx = core::to_vector(output_tensors[0].value());
//     auto metal_dgamma = core::to_vector(output_tensors[1].value());

//     float tolerance = 3e-2f;

//     // Sample check
//     for (uint32_t i = 0; i < std::min(500u, total_elements); ++i) {
//         EXPECT_NEAR(metal_dx[i], dx_ref[i], tolerance);
//     }

//     for (uint32_t i = 0; i < features; ++i) {
//         EXPECT_NEAR(metal_dgamma[i], dgamma_ref[i], tolerance);
//     }
// }

// TEST_F(LayerNormFusedOpTest, MetalLayerNormBw_ExtremeLongSequence) {
//     using namespace ttml;

//     uint32_t batch_size = 2;
//     uint32_t seq_len = 2048;  // Very long sequence
//     uint32_t heads = 4;
//     uint32_t features = 256;

//     std::vector<float> test_data;
//     std::mt19937 gen(3005);
//     std::normal_distribution<float> dist(0.0f, 1.0f);

//     uint32_t total_elements = batch_size * seq_len * heads * features;
//     test_data.reserve(total_elements);
//     for (uint32_t i = 0; i < total_elements; ++i) {
//         test_data.push_back(dist(gen));
//     }

//     std::vector<float> gamma_data;
//     std::vector<float> beta_data;
//     for (uint32_t i = 0; i < features; ++i) {
//         gamma_data.push_back(1.0f + dist(gen) * 0.1f);
//         beta_data.push_back(dist(gen) * 0.1f);
//     }

//     std::vector<float> dy_data;
//     std::normal_distribution<float> grad_dist(0.0f, 0.1f);
//     for (uint32_t i = 0; i < total_elements; ++i) {
//         dy_data.push_back(grad_dist(gen));
//     }

//     // Run reference
//     uint32_t combined_batch = batch_size * seq_len * heads;
//     auto [y_ref, cache] = layernorm_forward_reference(test_data, gamma_data, beta_data, combined_batch, features,
//     1e-6f); auto [dx_ref, dgamma_ref, dbeta_ref] = layernorm_backward_reference(dy_data, cache);

//     // Prepare metal tensors
//     auto input_tensor = core::from_vector(test_data, ttnn::Shape({batch_size, heads, seq_len, features}),
//     &autograd::ctx().get_device()); auto gamma_tensor = core::from_vector(gamma_data, ttnn::Shape({1, 1, 1,
//     features}), &autograd::ctx().get_device()); auto x_hat_tensor = core::from_vector(cache.x_hat,
//     ttnn::Shape({batch_size, heads, seq_len, features}), &autograd::ctx().get_device());

//     std::vector<float> rstd_data;
//     for (uint32_t b = 0; b < combined_batch; ++b) {
//         rstd_data.push_back(1.0f / cache.s[b]);
//     }
//     auto rstd_tensor = core::from_vector(rstd_data, ttnn::Shape({batch_size, heads, seq_len, 1}),
//     &autograd::ctx().get_device()); auto dy_tensor = core::from_vector(dy_data, ttnn::Shape({batch_size, heads,
//     seq_len, features}), &autograd::ctx().get_device());

//     auto output_tensors = metal::ops::layernorm_bw::LayerNormBackwardOperation::invoke(
//         input_tensor, gamma_tensor, x_hat_tensor, rstd_tensor, dy_tensor);

//     auto metal_dx = core::to_vector(output_tensors[0].value());
//     auto metal_dgamma = core::to_vector(output_tensors[1].value());

//     float tolerance = 3e-2f;

//     // Sample check
//     for (uint32_t i = 0; i < std::min(500u, total_elements); ++i) {
//         EXPECT_NEAR(metal_dx[i], dx_ref[i], tolerance);
//     }

//     for (uint32_t i = 0; i < features; ++i) {
//         EXPECT_NEAR(metal_dgamma[i], dgamma_ref[i], tolerance);
//     }
// }

// TEST_F(LayerNormFusedOpTest, MetalLayerNormBw_MaximalWorkload) {
//     using namespace ttml;

//     // Maximal practical workload test
//     uint32_t batch_size = 16;
//     uint32_t seq_len = 512;
//     uint32_t heads = 16;
//     uint32_t features = 1024;

//     std::vector<float> test_data;
//     std::mt19937 gen(3006);
//     std::normal_distribution<float> dist(0.0f, 1.0f);

//     uint32_t total_elements = batch_size * seq_len * heads * features;
//     test_data.reserve(total_elements);
//     for (uint32_t i = 0; i < total_elements; ++i) {
//         test_data.push_back(dist(gen));
//     }

//     std::vector<float> gamma_data;
//     std::vector<float> beta_data;
//     for (uint32_t i = 0; i < features; ++i) {
//         gamma_data.push_back(1.0f + dist(gen) * 0.1f);
//         beta_data.push_back(dist(gen) * 0.1f);
//     }

//     std::vector<float> dy_data;
//     std::normal_distribution<float> grad_dist(0.0f, 0.1f);
//     for (uint32_t i = 0; i < total_elements; ++i) {
//         dy_data.push_back(grad_dist(gen));
//     }

//     // Run reference
//     uint32_t combined_batch = batch_size * seq_len * heads;
//     auto [y_ref, cache] = layernorm_forward_reference(test_data, gamma_data, beta_data, combined_batch, features,
//     1e-6f); auto [dx_ref, dgamma_ref, dbeta_ref] = layernorm_backward_reference(dy_data, cache);

//     // Prepare metal tensors
//     auto input_tensor = core::from_vector(test_data, ttnn::Shape({batch_size, heads, seq_len, features}),
//     &autograd::ctx().get_device()); auto gamma_tensor = core::from_vector(gamma_data, ttnn::Shape({1, 1, 1,
//     features}), &autograd::ctx().get_device()); auto x_hat_tensor = core::from_vector(cache.x_hat,
//     ttnn::Shape({batch_size, heads, seq_len, features}), &autograd::ctx().get_device());

//     std::vector<float> rstd_data;
//     for (uint32_t b = 0; b < combined_batch; ++b) {
//         rstd_data.push_back(1.0f / cache.s[b]);
//     }
//     auto rstd_tensor = core::from_vector(rstd_data, ttnn::Shape({batch_size, heads, seq_len, 1}),
//     &autograd::ctx().get_device()); auto dy_tensor = core::from_vector(dy_data, ttnn::Shape({batch_size, heads,
//     seq_len, features}), &autograd::ctx().get_device());

//     auto output_tensors = metal::ops::layernorm_bw::LayerNormBackwardOperation::invoke(
//         input_tensor, gamma_tensor, x_hat_tensor, rstd_tensor, dy_tensor);

//     auto metal_dx = core::to_vector(output_tensors[0].value());
//     auto metal_dgamma = core::to_vector(output_tensors[1].value());

//     float tolerance = 4e-2f;  // Slightly relaxed for massive workload

//     // Sample check on multiple points throughout the tensor
//     uint32_t stride = total_elements / 1000;
//     for (uint32_t i = 0; i < 1000 && i * stride < total_elements; ++i) {
//         uint32_t idx = i * stride;
//         EXPECT_NEAR(metal_dx[idx], dx_ref[idx], tolerance)
//             << "Input gradient mismatch at index " << idx;
//     }

//     for (uint32_t i = 0; i < features; ++i) {
//         EXPECT_NEAR(metal_dgamma[i], dgamma_ref[i], tolerance)
//             << "Gamma gradient mismatch at index " << i;
//     }
// }

// TEST_F(LayerNormFusedOpTest, MetalLayerNormBw_NonPowerOfTwo_Dimensions) {
//     using namespace ttml;

//     // Non-power-of-2 but tile-aligned dimensions
//     uint32_t batch_size = 6;
//     uint32_t seq_len = 192;  // 6 * 32
//     uint32_t heads = 6;
//     uint32_t features = 672;  // 21 * 32

//     std::vector<float> test_data;
//     std::mt19937 gen(3007);
//     std::normal_distribution<float> dist(0.0f, 1.0f);

//     uint32_t total_elements = batch_size * seq_len * heads * features;
//     test_data.reserve(total_elements);
//     for (uint32_t i = 0; i < total_elements; ++i) {
//         test_data.push_back(dist(gen));
//     }

//     std::vector<float> gamma_data;
//     std::vector<float> beta_data;
//     for (uint32_t i = 0; i < features; ++i) {
//         gamma_data.push_back(1.0f + dist(gen) * 0.1f);
//         beta_data.push_back(dist(gen) * 0.1f);
//     }

//     std::vector<float> dy_data;
//     std::normal_distribution<float> grad_dist(0.0f, 0.1f);
//     for (uint32_t i = 0; i < total_elements; ++i) {
//         dy_data.push_back(grad_dist(gen));
//     }

//     // Run reference
//     uint32_t combined_batch = batch_size * seq_len * heads;
//     auto [y_ref, cache] = layernorm_forward_reference(test_data, gamma_data, beta_data, combined_batch, features,
//     1e-6f); auto [dx_ref, dgamma_ref, dbeta_ref] = layernorm_backward_reference(dy_data, cache);

//     // Prepare metal tensors
//     auto input_tensor = core::from_vector(test_data, ttnn::Shape({batch_size, heads, seq_len, features}),
//     &autograd::ctx().get_device()); auto gamma_tensor = core::from_vector(gamma_data, ttnn::Shape({1, 1, 1,
//     features}), &autograd::ctx().get_device()); auto x_hat_tensor = core::from_vector(cache.x_hat,
//     ttnn::Shape({batch_size, heads, seq_len, features}), &autograd::ctx().get_device());

//     std::vector<float> rstd_data;
//     for (uint32_t b = 0; b < combined_batch; ++b) {
//         rstd_data.push_back(1.0f / cache.s[b]);
//     }
//     auto rstd_tensor = core::from_vector(rstd_data, ttnn::Shape({batch_size, heads, seq_len, 1}),
//     &autograd::ctx().get_device()); auto dy_tensor = core::from_vector(dy_data, ttnn::Shape({batch_size, heads,
//     seq_len, features}), &autograd::ctx().get_device());

//     auto output_tensors = metal::ops::layernorm_bw::LayerNormBackwardOperation::invoke(
//         input_tensor, gamma_tensor, x_hat_tensor, rstd_tensor, dy_tensor);

//     auto metal_dx = core::to_vector(output_tensors[0].value());
//     auto metal_dgamma = core::to_vector(output_tensors[1].value());

//     float tolerance = 3e-2f;

//     // Sample check
//     for (uint32_t i = 0; i < std::min(500u, total_elements); ++i) {
//         EXPECT_NEAR(metal_dx[i], dx_ref[i], tolerance);
//     }

//     for (uint32_t i = 0; i < features; ++i) {
//         EXPECT_NEAR(metal_dgamma[i], dgamma_ref[i], tolerance);
//     }
// }
