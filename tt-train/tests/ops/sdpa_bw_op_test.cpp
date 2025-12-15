// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <sys/types.h>

#include <cmath>
#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/random.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "ttnn_fixed/matmuls.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

class SDPABackwardTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
        ttml::autograd::ctx().set_seed(42);
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

xt::xarray<float> generate_attn_mask(const xt::xarray<float>& query) {
    auto shape = query.shape();
    size_t S = shape[2];
    // Create mask with shape (1, 1, S, S) - same mask for all batches/heads
    xt::xarray<float> mask = xt::zeros<float>({1UL, 1UL, S, S});

    for (size_t s = 0; s < S; ++s) {
        for (size_t w = 0; w <= s; ++w) {
            mask(0, 0, s, w) = 1.0F;  // causal mask - lower triangular part
        }
    }
    return mask;
}

// Error analysis utility function
void print_error_analysis(
    const xt::xarray<float>& result,
    const xt::xarray<float>& groundtruth,
    std::pair<float, float> threshold,  // {atol, rtol}
    size_t num_elements_to_print,
    const std::string& name) {
    assert(result.shape() == groundtruth.shape() && "Tensors must have the same shape");

    // float atol = threshold.first;
    // float rtol = threshold.second;
    // auto& [atol, rtol] = threshold;
    double atol = static_cast<double>(threshold.first);
    double rtol = static_cast<double>(threshold.second);
    size_t total_elements = result.size();
    auto shape = result.shape();

    // Compute statistics
    float mean_error = 0.0f;
    float mse = 0.0f;
    float max_diff = 0.0f;

    for (size_t i = 0; i < total_elements; ++i) {
        float diff = std::abs(result.flat(i) - groundtruth.flat(i));
        mean_error += diff;
        mse += diff * diff;
        max_diff = std::max(max_diff, diff);
    }
    mean_error /= total_elements;
    mse /= total_elements;

    // Print statistics
    fmt::print("=== Error Analysis: {} ===\n", name);
    fmt::print("Shape: {}\n", shape);
    fmt::print("Total elements: {}\n", total_elements);
    fmt::print("Mean Absolute Error: {:.6e}\n", mean_error);
    fmt::print("Mean Squared Error (MSE): {:.6e}\n", mse);
    fmt::print("Max absolute difference: {:.6e}\n", max_diff);
    fmt::print("Threshold: atol={:.6e}, rtol={:.6e}\n\n", atol, rtol);

    // Find and print first k elements where error exceeds threshold
    fmt::print("First {} elements where error exceeds threshold:\n", num_elements_to_print);
    fmt::print("(index: result_val, groundtruth_val, abs_diff, threshold)\n");

    size_t error_count = 0;

    for (size_t b = 0; b < shape[0]; ++b) {
        for (size_t h = 0; h < shape[1]; ++h) {
            for (size_t s = 0; s < shape[2]; ++s) {
                for (size_t d = 0; d < shape[3]; ++d) {
                    double result_val = static_cast<double>(result(b, h, s, d));
                    double gt_val = static_cast<double>(groundtruth(b, h, s, d));
                    double diff = std::abs(result_val - gt_val);
                    double tolerance = std::max(atol, rtol * std::abs(gt_val));
                    bool is_close = (diff <= atol || diff <= rtol * std::max(std::abs(result_val), std::abs(gt_val)));
                    if (!is_close) {
                        if (error_count <= num_elements_to_print) {
                            fmt::print(
                                "[{},{},{},{}]: {:.3e}, {:.3e}, {:.3e}, {:.3e}\n",
                                b,
                                h,
                                s,
                                d,
                                result_val,
                                gt_val,
                                diff,
                                tolerance);
                        }
                        error_count++;
                    }
                }
            }
        }
    }

    fmt::print(" Total elements exceeding threshold: {}\n", error_count);
    fmt::print("=== End Error Analysis ===\n\n");
}

xt::xarray<float> dot_product(const xt::xarray<float>& input_0, const xt::xarray<float>& input_1) {
    assert(input_0.shape() == input_1.shape());
    auto shape = input_0.shape();
    size_t B = shape[0], H = shape[1], S = shape[2], D = shape[3];
    shape[3] = 1U;  // output shape is (B,H,S,1)
    xt::xarray<float> result = xt::zeros<float>(shape);

    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t s = 0; s < S; ++s) {
                float sum = 0.0F;
                for (size_t d = 0; d < D; ++d) {
                    sum += input_0(b, h, s, d) * input_1(b, h, s, d);
                }
                result(b, h, s, 0) = sum;
            }
        }
    }
    return result;
}

// Helper function to scale a tensor by a scalar factor
// Used to pre-scale Query or Key tensors to match kernel's mathematical flow
xt::xarray<float> scale_tensor(const xt::xarray<float>& tensor, float scale_factor) {
    return tensor * scale_factor;
}

// Pure float implementation of SDPA backward pass using xtensor
// This serves as the reference ground truth for testing
// Returns: {dQ, dK, dV, intermediates}
std::vector<xt::xarray<float>> float_sdpa_backward(
    const xt::xarray<float>& Q,
    const xt::xarray<float>& K,
    const xt::xarray<float>& V,
    const xt::xarray<float>& grad_output,
    const std::optional<xt::xarray<float>>& attn_mask = std::nullopt) {
    auto shape = Q.shape();
    size_t B = shape[0], H = shape[1], S = shape[2], D = shape[3], intermediate_size = 64;

    auto kv_shape = K.shape();
    size_t G = kv_shape[1];  // number of KV heads (groups)

    const float scale = 1.0F / std::sqrt(static_cast<float>(D));

    // ========== Forward Pass (need intermediate values) ==========

    // Step 1: Scale query
    xt::xarray<float> Q_scaled = Q * scale;

    // Step 2: Compute attention scores: S = Q_scaled @ K^T
    // Handle grouped attention: each KV head serves multiple Q heads
    xt::xarray<float> scores = xt::zeros<float>({B, H, S, S});

    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            size_t kv_head = h / (H / G);  // which KV head this Q head uses
            for (size_t i = 0; i < S; ++i) {
                for (size_t j = 0; j < S; ++j) {
                    float sum = 0.0F;
                    for (size_t d = 0; d < D; ++d) {
                        sum += Q_scaled(b, h, i, d) * K(b, kv_head, j, d);
                    }
                    scores(b, h, i, j) = sum;
                }
            }
        }
    }

    // Step 3: Apply attention mask if provided
    // Mask is (1, 1, S, S) - same mask for all batches/heads
    if (attn_mask.has_value()) {
        const auto& mask = attn_mask.value();
        for (size_t b = 0; b < B; ++b) {
            for (size_t h = 0; h < H; ++h) {
                for (size_t i = 0; i < S; ++i) {
                    for (size_t j = 0; j < S; ++j) {
                        // mask: 1.0 = keep, 0.0 = mask out
                        if (mask(0, 0, i, j) < 0.5F) {
                            scores(b, h, i, j) = -1e9F;
                        }
                    }
                }
            }
        }
    }

    // Step 4: Softmax over last dimension and store intermediates
    // Intermediates shape: (B, H, S, 64) where position 0 = max_val, position 32 = recip_sum_exp
    xt::xarray<float> intermediates = xt::zeros<float>({B, H, S, intermediate_size});
    xt::xarray<float> attention_weights = xt::zeros<float>({B, H, S, S});
    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t i = 0; i < S; ++i) {
                // Find max for numerical stability
                float max_val = scores(b, h, i, 0);
                for (size_t j = 1; j < S; ++j) {
                    max_val = std::max(max_val, scores(b, h, i, j));
                }

                // Store max_val at position 0
                intermediates(b, h, i, 0) = max_val;

                // Compute exp and sum
                float sum_exp = 0.0F;
                for (size_t j = 0; j < S; ++j) {
                    float exp_val = std::exp(scores(b, h, i, j) - max_val);
                    attention_weights(b, h, i, j) = exp_val;
                    sum_exp += exp_val;
                }

                // Store recip_sum_exp at position 32
                intermediates(b, h, i, 32) = 1.0F / sum_exp;

                // Normalize
                for (size_t j = 0; j < S; ++j) {
                    attention_weights(b, h, i, j) /= sum_exp;
                }
            }
        }
    }

    // Step 5: attention_weights @ V (forward output, not returned but needed for backward)
    // We don't actually need to compute this for backward pass

    // ========== Backward Pass ==========

    const auto& dO = grad_output;  // gradient w.r.t. output [B, H, S, D]

    // Step 1: Compute dL/dV = attention_weights^T @ dO
    // For grouped attention, sum over Q heads that share the same KV head
    xt::xarray<float> dV = xt::zeros<float>({B, G, S, D});

    for (size_t b = 0; b < B; ++b) {
        for (size_t g = 0; g < G; ++g) {
            // Sum contributions from all Q heads in this group
            for (size_t h_in_group = 0; h_in_group < (H / G); ++h_in_group) {
                size_t h = g * (H / G) + h_in_group;
                for (size_t j = 0; j < S; ++j) {  // V's sequence dimension
                    for (size_t d = 0; d < D; ++d) {
                        float sum = 0.0F;
                        for (size_t i = 0; i < S; ++i) {
                            sum += attention_weights(b, h, i, j) * dO(b, h, i, d);
                        }
                        dV(b, g, j, d) += sum;
                    }
                }
            }
        }
    }

    // Step 2: Compute dL/dP (gradient w.r.t. attention weights) = dO @ V^T
    xt::xarray<float> dP = xt::zeros<float>({B, H, S, S});

    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            size_t kv_head = h / (H / G);
            for (size_t i = 0; i < S; ++i) {
                for (size_t j = 0; j < S; ++j) {
                    float sum = 0.0F;
                    for (size_t d = 0; d < D; ++d) {
                        sum += dO(b, h, i, d) * V(b, kv_head, j, d);
                    }
                    dP(b, h, i, j) = sum;
                }
            }
        }
    }

    // Step 3: Softmax backward: dL/dS = P * (dP - sum(P * dP))
    xt::xarray<float> dS = xt::zeros<float>({B, H, S, S});

    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t i = 0; i < S; ++i) {
                // Compute sum(P * dP) for this row
                float sum_p_dp = 0.0F;
                for (size_t j = 0; j < S; ++j) {
                    sum_p_dp += attention_weights(b, h, i, j) * dP(b, h, i, j);
                }

                // Compute dS = P * (dP - sum_p_dp)
                for (size_t j = 0; j < S; ++j) {
                    dS(b, h, i, j) = attention_weights(b, h, i, j) * (dP(b, h, i, j) - sum_p_dp);
                }
            }
        }
    }

    // Step 4: Apply scale to dS (IMPORTANT: scale before matmul for numerical stability!)
    dS = dS * scale;

    // Step 5: Compute dL/dQ = dS @ K
    xt::xarray<float> dQ = xt::zeros<float>({B, H, S, D});

    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            size_t kv_head = h / (H / G);
            for (size_t i = 0; i < S; ++i) {
                for (size_t d = 0; d < D; ++d) {
                    float sum = 0.0F;
                    for (size_t j = 0; j < S; ++j) {
                        sum += dS(b, h, i, j) * K(b, kv_head, j, d);
                    }
                    dQ(b, h, i, d) = sum;
                }
            }
        }
    }

    // Step 6: Compute dL/dK = dS^T @ Q
    // dS already has scale applied (line 317), so we use UNSCALED Q to match composite
    // Composite does: dL_dK = (dL_dscaled_dot * scale)^T @ query (line 513, 520-522)
    // This is mathematically: dK = (grad_softmax * scale)^T @ Q
    // For grouped attention, sum over Q heads that share the same KV head
    xt::xarray<float> dK = xt::zeros<float>({B, G, S, D});

    for (size_t b = 0; b < B; ++b) {
        for (size_t g = 0; g < G; ++g) {
            // Sum contributions from all Q heads in this group
            for (size_t h_in_group = 0; h_in_group < (H / G); ++h_in_group) {
                size_t h = g * (H / G) + h_in_group;
                for (size_t j = 0; j < S; ++j) {  // K's sequence dimension
                    for (size_t d = 0; d < D; ++d) {
                        float sum = 0.0F;
                        for (size_t i = 0; i < S; ++i) {
                            sum += dS(b, h, i, j) * Q(b, h, i, d);  // Use UNSCALED Q!
                        }
                        dK(b, g, j, d) += sum;
                    }
                }
            }
        }
    }

    return {dQ, dK, dV, intermediates};
}

// Wrapper around matmul to handle sharing of KV heads across groups of query
// heads.
// For e.g. Q @ V, there are two cases:
// - G == H: (B, H, S, S) x (B, H, S, V) -> (B, H, S, V)
// - G != H:
//    - In this case value has shape (B,G,S,V):
//      1. Reshape attention_weights to (B*G, H/G, S, S).
//      2. Reshape value to (B*G, 1, S, V).
//      3. Manually broadcast values over groupsize.
//      4. Matmul.
//      5. Reshape the result to (B, H, S, V).
//   - Summary of intermediate shapes:
//     (B*G, H/G, S, S) x (B*G, 1, S, V) -> (B*G, H/G, S, V) -> (B, H, S, V)
ttnn::Tensor groups_shared_matmul(
    const ttnn::Tensor& query_tensor,
    const ttnn::Tensor& kv_tensor,
    bool transpose_a = false,
    bool transpose_b = false) {
    using namespace ttml;
    auto [batch_num, heads, seq_len, embedding_dim] = query_tensor.logical_shape().to_array_4D();
    auto [batch_num_v, groups, seq_len_v, embedding_dim_v] = kv_tensor.logical_shape().to_array_4D();
    if (batch_num != batch_num_v) {
        throw std::invalid_argument(
            fmt::format(
                "query_tensor and kv_tensor must have the same batch size, got shapes {} and {} respectively",
                query_tensor.logical_shape(),
                kv_tensor.logical_shape()));
    }
    if (heads == groups) {
        // no broadcasting needed
        return ttnn_fixed::matmul(query_tensor, kv_tensor, transpose_a, transpose_b);
    }
    // result will have shape (batch_num, heads, M, N)
    // we determine M,N based on the transpose options
    auto M = transpose_a ? embedding_dim : seq_len;
    auto N = transpose_b ? seq_len_v : embedding_dim_v;

    // - G != H:
    //   bcast kv_tensor to groups in query_tensor then reshape back to query_tensor_shape:
    //   (B*G,H/G,M,E) x (B*G, 1, E,N) -> (B*G, H/G, M, N) -> (B, H, M, N)
    auto query_tensor_grouped =
        ttnn::reshape(query_tensor, ttnn::Shape{batch_num * groups, heads / groups, seq_len, embedding_dim});
    auto kv_tensor_batched = ttnn::reshape(kv_tensor, ttnn::Shape{batch_num * groups, 1U, seq_len_v, embedding_dim_v});

    // repeat kv_tensor to group size for each group (manual bcast)
    ttnn::Tensor kv_tensor_repeated = ttnn::repeat(kv_tensor_batched, ttnn::Shape{1U, heads / groups, 1U, 1U});
    auto bcasted_mm = ttnn_fixed::matmul(query_tensor_grouped, kv_tensor_repeated, transpose_a, transpose_b);
    auto reshaped_mm = ttnn::reshape(bcasted_mm, ttnn::Shape{batch_num, heads, M, N});
    return reshaped_mm;
}

// helper function to collect grads from the query groups associated
// with each key/value
ttnn::Tensor sum_over_groups(const ttnn::Tensor& ungrouped_grads, uint32_t groups) {
    using namespace ttml;
    if (ungrouped_grads.logical_shape().rank() != 4) {
        throw std::invalid_argument(
            fmt::format("ungrouped_grads must have rank 4, but got rank {}", ungrouped_grads.logical_shape().rank()));
    }
    // [B,H,S,E]
    auto [batch_num, num_heads, seq_len, embedding_dim] = ungrouped_grads.logical_shape().to_array_4D();
    if (groups == num_heads) {
        // group size is 1, nothing to do
        return ungrouped_grads;
    }
    // sum over groups:
    // [B,H,S,E] -> [B*G,H/G,S,E] -> [B*G,1,S,E] -> [B,G,S,E]
    auto grouped_grads =
        ttnn::reshape(ungrouped_grads, ttnn::Shape{batch_num * groups, num_heads / groups, seq_len, embedding_dim});
    auto summed_grads = ttnn_fixed::sum_moreh(grouped_grads, /*dim=*/1, /*keep_dim=*/true);
    return ttnn::reshape(summed_grads, ttnn::Shape{batch_num, groups, seq_len, embedding_dim});
}

std::vector<ttnn::Tensor> composite_sdpa(
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    const ttnn::Tensor& grad_output,
    const std::optional<ttnn::Tensor>& attn_mask,
    bool return_intermediate = false) {
    // Forward pass with intermediates returned
    // Backward pass return 3 gradients
    using namespace ttml;

    auto [batch_num, heads, seq_len, embedding_dim] = query.logical_shape().to_array_4D();
    auto groups = value.logical_shape().to_array_4D()[1];

    const float scale = 1.0F / std::sqrt(static_cast<float>(embedding_dim));
    constexpr auto none = ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam>{};
    auto q_scaled = ttnn::multiply(query, scale, std::nullopt, std::nullopt, std::nullopt, none, none, none, false);

    // σQ @ K
    ttnn::Tensor qk_scaled = groups_shared_matmul(q_scaled, key, /*transpose_a=*/false, /*transpose_b=*/true);

    if (attn_mask.has_value()) {
        auto mask_tensor = attn_mask.value();
        // ttnn::where when mask is not of the same shape as qk_scaled
        qk_scaled = ttnn::add(
            ttnn::multiply(mask_tensor, qk_scaled, std::nullopt, std::nullopt, std::nullopt, none, none, none, false),
            ttnn::multiply(
                ttnn::subtract(mask_tensor, 1.F, std::nullopt, std::nullopt, std::nullopt, none, none, none, false),
                1e9F,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                none,
                none,
                none,
                false),
            std::nullopt,
            std::nullopt,
            std::nullopt,
            none,
            none,
            none,
            false);
    }
    // Calculate intermediate results to test against kernel implementation
    auto max_value = ttnn::max(qk_scaled, /* dim */ 3, /* keepdim */ true);
    auto qk_scaled_sub_max = ttnn::subtract(qk_scaled, max_value);
    auto exp_qk_scaled = ttnn::exp(qk_scaled_sub_max);
    auto sum_exp = ttnn::sum(exp_qk_scaled, /* dim */ 3, /* keepdim */ true);
    auto recip_sum_exp = ttnn::reciprocal(sum_exp);

    // (B, H, S, S)
    auto attention_weights = ttml::metal::softmax(qk_scaled, /* axis */ 3);
    // TODO: add dropout here

    // softmax(σQ@K+mask) @ V
    auto attention_qkv = groups_shared_matmul(attention_weights, value, /*transpose_a=*/false, /*transpose_b=*/false);

    // compute gradients(backward pass)
    auto dL_dout = grad_output;  // (B, H, S, embedding_dim)
    // dL_d(softmax(σQK+mask)) = dL_dout @ value^T
    ttnn::Tensor dL_dattention_weights =
        groups_shared_matmul(dL_dout, value, /*transpose_a=*/false, /*transpose_b=*/true);

    auto dL_dscaled_dot = ttnn::moreh_softmax_backward(
        attention_weights,
        dL_dattention_weights,
        /* axis */ 3,
        /* output */ std::nullopt,
        ttnn::operations::moreh::moreh_softmax_backward::MorehSoftmaxBackwardOp::SOFTMAX,
        ttnn::operations::moreh::moreh_softmax_backward::MorehSoftmaxBackwardOpParallelizationStrategy::NONE,
        /* output_mem_config */ std::nullopt,
        /* compute_kernel_config */ core::ComputeKernelConfig::precise());
    dL_dattention_weights.deallocate();

    dL_dscaled_dot = ttnn::multiply(
        dL_dscaled_dot, scale, std::nullopt, std::nullopt, std::nullopt, none, none, none, false);  // [B,H,S,S]

    // dL_dQ = dL_dscaled_dot @ key
    ttnn::Tensor dL_dQ = groups_shared_matmul(dL_dscaled_dot, key, /*transpose_a=*/false, /*transpose_b=*/false);

    // dL_dK = Σ_g [dL_dscaled_dot^T @ query]
    ttnn::Tensor dL_dK = ttnn_fixed::matmul(
        dL_dscaled_dot,
        query,
        /*transpose_a=*/true,
        /*transpose_b=*/false);
    dL_dK = sum_over_groups(dL_dK, groups);  // no-op when groups == heads

    // dL_dV = Σ_g [attention_weights^T @ dL_dout]
    ttnn::Tensor dL_dV = ttnn_fixed::matmul(
        attention_weights,
        dL_dout,
        /*transpose_a=*/true,
        /*transpose_b=*/false);
    dL_dV = sum_over_groups(dL_dV, groups);  // no-op when groups == heads

    return {/* forward pass output*/ attention_qkv,
            /* per row max value*/ max_value,
            /* recip sum exp */ recip_sum_exp,
            /* dL_dQ */ dL_dQ,
            /* dL_dK */ dL_dK,
            /* dL_dV */ dL_dV,
            /*attention_weights*/ qk_scaled};
}

// ========== Test Configuration ==========
struct SDPABackwardTestConfig {
    uint32_t batch_size;
    uint32_t sequence_length;
    uint32_t query_dim;
    uint32_t key_value_dim;
    uint32_t num_query_heads;
    uint32_t num_kv_heads;
    float dropout_prob = 0.0F;
    bool fp32_dest_acc_en = true;
    float atol = 3e-2F;
    float rtol = 3e-2F;
    std::string test_name = "SDPA Backward Test";
};

void run_sdpa_backward_test(const SDPABackwardTestConfig& config) {
    using namespace ttml;

    fmt::print("\n========== {} ==========\n", config.test_name);
    fmt::print("Config: B={}, S={}, qD={}, kvD={}, qNH={}, kvNH={}\n",
               config.batch_size, config.sequence_length, config.query_dim, config.key_value_dim,
               config.num_query_heads, config.num_kv_heads);
    fmt::print("Tolerances: atol={:.2e}, rtol={:.2e}\n\n", config.atol, config.rtol);

    const uint32_t B = config.batch_size;
    const uint32_t qNH = config.num_query_heads;
    const uint32_t kvNH = config.num_kv_heads;
    const uint32_t S = config.sequence_length;
    const uint32_t qD = config.query_dim;
    const uint32_t kvD = config.key_value_dim;
    const float dropout_probability = config.dropout_prob;
    const bool fp32_dest_acc_en = config.fp32_dest_acc_en;
    const float atol = config.atol;
    const float rtol = config.rtol;
    const float scale_factor = 1.0F / std::sqrt(static_cast<float>(qD));

    auto* device = &autograd::ctx().get_device();

    std::mt19937 gen(42);
    auto& rng = ttml::autograd::ctx().get_generator();
    uint32_t seed = rng();

    // Generate input tensors
    xt::xarray<float> query_tensor = xt::empty<float>({B, qNH, S, qD});
    ttml::core::parallel_generate(
        std::span{query_tensor.data(), query_tensor.size()},
        []() { return std::uniform_real_distribution<float>(-1.0F, 1.0F); },
        seed);

    xt::xarray<float> key_tensor = xt::empty<float>({B, kvNH, S, kvD});
    ttml::core::parallel_generate(
        std::span{key_tensor.data(), key_tensor.size()},
        []() { return std::uniform_real_distribution<float>(-1.0F, 1.0F); },
        seed);

    xt::xarray<float> value_tensor = xt::empty<float>({B, kvNH, S, kvD});
    ttml::core::parallel_generate(
        std::span{value_tensor.data(), value_tensor.size()},
        []() { return std::uniform_real_distribution<float>(-1.0F, 1.0F); },
        seed);

    // Create attention mask in kernel-expected format (1, 1, S, S) - broadcasted across batches/heads
    xt::xarray<float> attn_mask_tensor = generate_attn_mask(query_tensor);

    xt::xarray<float> grad_output_tensor = xt::empty<float>({B, qNH, S, qD});
    ttml::core::parallel_generate(
        std::span{grad_output_tensor.data(), grad_output_tensor.size()},
        []() { return std::uniform_real_distribution<float>(-1.0F, 1.0F); },
        seed);

    xt::xarray<float> scale_query_tensor = scale_tensor(query_tensor, scale_factor);
    auto scaled_query = core::from_xtensor(scale_query_tensor, device);

    auto query = core::from_xtensor(query_tensor, device);
    auto key = core::from_xtensor(key_tensor, device);
    auto value = core::from_xtensor(value_tensor, device);
    auto attn_mask = core::from_xtensor(attn_mask_tensor, device);
    auto grad_output = core::from_xtensor(grad_output_tensor, device);

    // ========== Pure Float Reference (Ground Truth) ==========
    fmt::print("Computing pure float reference (ground truth)...\n");
    auto float_gradients =
        float_sdpa_backward(query_tensor, key_tensor, value_tensor, grad_output_tensor, attn_mask_tensor);
    auto float_dQ = float_gradients[0];
    auto float_dK = float_gradients[1];
    auto float_dV = float_gradients[2];
    auto float_intermediates = float_gradients[3];
    fmt::print("Float reference computed.\n\n");

    // ========== Composite Implementation (uses ttnn ops) ==========
    fmt::print("Computing composite SDPA backward (ttnn ops)...\n");
    auto composite_output = composite_sdpa(query, key, value, grad_output, attn_mask, /*return_intermediate=*/true);
    auto composite_attn_output = /* attn_output */ composite_output[0];
    auto max_value = /* max_value */ composite_output[1];
    auto recip_sum_exp = /* recip_sum_exp */ composite_output[2];
    auto dL_dQ = /* dL_dQ */ composite_output[3];
    auto dL_dK = /* dL_dK */ composite_output[4];
    auto dL_dV = /* dL_dV */ composite_output[5];
    [[maybe_unused]] auto attention_weights = /* attention_weights */ composite_output[6];

    auto padded_interm = core::zeros(ttnn::Shape{B, qNH, S, 32U}, device, ttnn::DataType::BFLOAT16);
    max_value = ttnn::add(padded_interm, max_value);
    recip_sum_exp = ttnn::add(padded_interm, recip_sum_exp);

    auto composite_intermediates = ttnn::concat(std::vector<ttnn::Tensor>{max_value, recip_sum_exp}, 3);

    // Diagnostic: Check intermediate values that might affect numerical stability
    xt::xarray<float> max_val_cpu = core::to_xtensor(max_value);
    xt::xarray<float> recip_cpu = core::to_xtensor(recip_sum_exp);
    fmt::print("\n=== Intermediates for Numerical Stability Check ===\n");
    fmt::print("Q row 0 (seq 0-31): max={:.6e}, recip_sum={:.6e}\n", max_val_cpu(0, 0, 0, 0), recip_cpu(0, 0, 0, 0));
    if (S > 32) {
        fmt::print("Q row 32 (seq 32): max={:.6e}, recip_sum={:.6e}\n", max_val_cpu(0, 0, 32, 0), recip_cpu(0, 0, 32, 0));
    }

    // ========== SDPA Forward Kernel (get attn_output and intermediates) ==========
    fmt::print("\nComputing SDPA forward kernel...\n");
    auto sdpa_fw_result = ttml::metal::sdpa_fw(
        query,
        key,
        value,
        attn_mask,
        dropout_probability,
        /*return_intermediates=*/true,
        fp32_dest_acc_en);

    // sdpa_fw returns std::vector<std::optional<ttnn::Tensor>>, unwrap with .value()
    auto kernel_attn_output = sdpa_fw_result[0].value();
    auto kernel_intermediates = sdpa_fw_result[1].value();
    fmt::print("SDPA forward kernel output shape: {}\n", kernel_attn_output.logical_shape());
    fmt::print("SDPA forward kernel intermediates shape: {}\n", kernel_intermediates.logical_shape());

    // ========== SDPA Backward Kernel (using forward kernel outputs) ==========
    fmt::print("\nComputing SDPA backward kernel (using sdpa_fw outputs)...\n");
    auto op_result = ttml::metal::sdpa_bw(
        grad_output,
        kernel_attn_output,
        query,
        key,
        value,
        attn_mask,
        kernel_intermediates,
        dropout_probability,
        fp32_dest_acc_en);

    fmt::print("\n=== Converting tensors to xtensor ===\n");

    // Convert forward kernel outputs for comparison
    xt::xarray<float> kernel_attn_output_cpu = core::to_xtensor(kernel_attn_output);
    xt::xarray<float> kernel_intermediates_cpu = core::to_xtensor(kernel_intermediates);
    xt::xarray<float> composite_attn_output_cpu = core::to_xtensor(composite_attn_output);
    xt::xarray<float> composite_intermediates_cpu = core::to_xtensor(composite_intermediates);
    fmt::print("Converted forward outputs\n");

    xt::xarray<float> sdpa_bw_dQ = core::to_xtensor(op_result[0]);  // dL_dQ
    fmt::print("Converted kernel dQ\n");
    xt::xarray<float> composite_dQ = core::to_xtensor(dL_dQ);
    fmt::print("Converted composite dQ\n");

    xt::xarray<float> sdpa_bw_dK = core::to_xtensor(op_result[1]);  // dL_dK
    fmt::print("Converted kernel dK\n");
    xt::xarray<float> composite_dK = core::to_xtensor(dL_dK);
    fmt::print("Converted composite dK\n");

    xt::xarray<float> sdpa_bw_dV = core::to_xtensor(op_result[2]);  // dL_dV
    fmt::print("Converted kernel dV\n");
    xt::xarray<float> composite_dV = core::to_xtensor(dL_dV);
    fmt::print("Converted composite dV\n");

    // Print shapes before checking
    fmt::print("\n=== Tensor Shapes ===\n");
    fmt::print("float_dQ shape: {}\n", float_dQ.shape());
    fmt::print("float_dK shape: {}\n", float_dK.shape());
    fmt::print("float_dV shape: {}\n", float_dV.shape());
    fmt::print("composite_dQ shape: {}\n", composite_dQ.shape());
    fmt::print("composite_dK shape: {}\n", composite_dK.shape());
    fmt::print("composite_dV shape: {}\n", composite_dV.shape());
    fmt::print("kernel_dQ shape: {}\n", sdpa_bw_dQ.shape());
    fmt::print("kernel_dK shape: {}\n", sdpa_bw_dK.shape());
    fmt::print("kernel_dV shape: {}\n", sdpa_bw_dV.shape());

    // Verify shapes match
    ASSERT_EQ(sdpa_bw_dQ.shape(), composite_dQ.shape()) << "kernel_dQ shape != composite_dQ shape";
    ASSERT_EQ(sdpa_bw_dQ.shape(), float_dQ.shape()) << "kernel_dQ shape != float_dQ shape";
    ASSERT_EQ(sdpa_bw_dK.shape(), composite_dK.shape()) << "kernel_dK shape != composite_dK shape";
    ASSERT_EQ(sdpa_bw_dK.shape(), float_dK.shape()) << "kernel_dK shape != float_dK shape";
    ASSERT_EQ(sdpa_bw_dV.shape(), composite_dV.shape()) << "kernel_dV shape != composite_dV shape";
    ASSERT_EQ(sdpa_bw_dV.shape(), float_dV.shape()) << "kernel_dV shape != float_dV shape";

    // Debug: Check for NaN/Inf values
    fmt::print("\n=== DEBUG: Checking for invalid values ===\n");

    auto has_nan_or_inf = [](const xt::xarray<float>& arr, const std::string& name) {
        size_t nan_count = 0, inf_count = 0;
        for (size_t i = 0; i < arr.size(); ++i) {
            if (std::isnan(arr.flat(i)))
                nan_count++;
            if (std::isinf(arr.flat(i)))
                inf_count++;
        }
        fmt::print("{}: shape={}, NaNs={}, Infs={}\n", name, arr.shape(), nan_count, inf_count);
    };

    has_nan_or_inf(float_dQ, "float_dQ");
    has_nan_or_inf(composite_dQ, "composite_dQ");
    has_nan_or_inf(sdpa_bw_dQ, "kernel_dQ");
    has_nan_or_inf(float_dK, "float_dK");
    has_nan_or_inf(composite_dK, "composite_dK");
    has_nan_or_inf(sdpa_bw_dK, "kernel_dK");
    has_nan_or_inf(float_dV, "float_dV");
    has_nan_or_inf(composite_dV, "composite_dV");
    has_nan_or_inf(sdpa_bw_dV, "kernel_dV");

    // ========== Comparisons ==========
    fmt::print("\n=== COMPARISON RESULTS ===\n\n");

    // 0. Forward pass: Kernel vs Composite vs Float
    fmt::print("------- FORWARD PASS -------\n");
    print_error_analysis(
        kernel_attn_output_cpu, composite_attn_output_cpu, {atol, rtol}, 50, "Attn Output (Kernel FW vs Composite)");
    // Note: Composite intermediates have broadcast format (value in all 32 positions per tile)
    // while kernel/float have sparse format (value only at position 0 and 32).
    // So we compare kernel vs float for intermediates validation.
    print_error_analysis(
        kernel_intermediates_cpu, float_intermediates, {atol, rtol}, 50, "Intermediates (Kernel FW vs Float)");

    // 1. Float vs Composite
    fmt::print("------- FLOAT REFERENCE VS COMPOSITE -------\n");
    print_error_analysis(composite_dQ, float_dQ, {atol, rtol}, 50, "dQ (Composite vs Float)");
    print_error_analysis(composite_dK, float_dK, {atol, rtol}, 50, "dK (Composite vs Float)");
    print_error_analysis(composite_dV, float_dV, {atol, rtol}, 50, "dV (Composite vs Float)");

    // 2. Float vs Kernel (using sdpa_fw outputs)
    fmt::print("------- FLOAT REFERENCE VS KERNEL (using sdpa_fw) -------\n");
    print_error_analysis(sdpa_bw_dQ, float_dQ, {atol, rtol}, 50, "dQ (Kernel vs Float)");
    print_error_analysis(sdpa_bw_dK, float_dK, {atol, rtol}, 50, "dK (Kernel vs Float)");
    print_error_analysis(sdpa_bw_dV, float_dV, {atol, rtol}, 50, "dV (Kernel vs Float)");

    // DEBUG: Check xt::isclose to understand second tile issue
    fmt::print("\n=== DEBUG: Analyzing xt::isclose for second tile issue ===\n");

    // Final assertions
    // Forward pass checks
    bool fw_attn_output_matches = xt::allclose(kernel_attn_output_cpu, composite_attn_output_cpu, rtol, atol);
    // Compare kernel intermediates vs float (same sparse format: values at pos 0 and 32 only)
    bool fw_intermediates_matches = xt::allclose(kernel_intermediates_cpu, float_intermediates, rtol, atol);

    // Backward pass checks
    bool kernel_dQ_matches_float = xt::allclose(sdpa_bw_dQ, float_dQ, rtol, atol);
    bool kernel_dK_matches_float = xt::allclose(sdpa_bw_dK, float_dK, rtol, atol);
    bool kernel_dV_matches_float = xt::allclose(sdpa_bw_dV, float_dV, rtol, atol);
    bool kernel_dQ_matches_composite = xt::allclose(sdpa_bw_dQ, composite_dQ, rtol, atol);
    bool kernel_dK_matches_composite = xt::allclose(sdpa_bw_dK, composite_dK, rtol, atol);
    bool kernel_dV_matches_composite = xt::allclose(sdpa_bw_dV, composite_dV, rtol, atol);
    bool composite_dQ_matches_float = xt::allclose(composite_dQ, float_dQ, rtol, atol);
    bool composite_dK_matches_float = xt::allclose(composite_dK, float_dK, rtol, atol);
    bool composite_dV_matches_float = xt::allclose(composite_dV, float_dV, rtol, atol);

    fmt::print("\n=== FINAL RESULTS ({}) ===\n", config.test_name);

    // Forward pass results
    fmt::print("Forward Pass:\n");
    fmt::print("  Attn Output (Kernel vs Composite): {}\n", fw_attn_output_matches ? "PASS" : "FAIL");
    fmt::print("  Intermediates (Kernel vs Float): {}\n", fw_intermediates_matches ? "PASS" : "FAIL");

    // Backward pass results
    fmt::print("\nBackward Pass (sdpa_bw kernel using sdpa_fw outputs):\n");
    fmt::print(
        "dQ: Kernel vs Float: {}, Kernel vs Composite: {}\n",
        kernel_dQ_matches_float ? "PASS" : "FAIL",
        kernel_dQ_matches_composite ? "PASS" : "FAIL");
    fmt::print(
        "dK: Kernel vs Float: {}, Kernel vs Composite: {}\n",
        kernel_dK_matches_float ? "PASS" : "FAIL",
        kernel_dK_matches_composite ? "PASS" : "FAIL");
    fmt::print(
        "dV: Kernel vs Float: {}, Kernel vs Composite: {}\n",
        kernel_dV_matches_float ? "PASS" : "FAIL",
        kernel_dV_matches_composite ? "PASS" : "FAIL");

    fmt::print("\nComposite vs Float Reference:\n");
    fmt::print("dQ: {}\n", composite_dQ_matches_float ? "PASS" : "FAIL");
    fmt::print("dK: {}\n", composite_dK_matches_float ? "PASS" : "FAIL");
    fmt::print("dV: {}\n", composite_dV_matches_float ? "PASS" : "FAIL");

    // Assertions
    EXPECT_TRUE(fw_attn_output_matches) << "Forward attn output mismatch in " << config.test_name;
    EXPECT_TRUE(fw_intermediates_matches) << "Forward intermediates mismatch in " << config.test_name;
    EXPECT_TRUE(kernel_dQ_matches_float) << "Kernel dQ vs Float mismatch in " << config.test_name;
    EXPECT_TRUE(kernel_dK_matches_float) << "Kernel dK vs Float mismatch in " << config.test_name;
    EXPECT_TRUE(kernel_dV_matches_float) << "Kernel dV vs Float mismatch in " << config.test_name;
    EXPECT_TRUE(kernel_dQ_matches_composite) << "Kernel dQ vs Composite mismatch in " << config.test_name;
    EXPECT_TRUE(kernel_dK_matches_composite) << "Kernel dK vs Composite mismatch in " << config.test_name;
    EXPECT_TRUE(kernel_dV_matches_composite) << "Kernel dV vs Composite mismatch in " << config.test_name;
}

// ========== Test Cases ==========

TEST_F(SDPABackwardTest, SmallBatch) {
    SDPABackwardTestConfig config{
        .batch_size = 2U,
        .sequence_length = 128U,
        .query_dim = 64U,
        .key_value_dim = 64U,
        .num_query_heads = 4U,
        .num_kv_heads = 4U,
        .dropout_prob = 0.0F,
        .fp32_dest_acc_en = true,
        .atol = 3e-2F,
        .rtol = 3e-2F,
        .test_name = "SmallBatch (B=2, S=128, D=64, H=4)"};
    run_sdpa_backward_test(config);
}

TEST_F(SDPABackwardTest, NanoGPTConfig) {
    // Match nano_gpt training config
    SDPABackwardTestConfig config{
        .batch_size = 64U,
        .sequence_length = 256U,
        .query_dim = 128U,
        .key_value_dim = 128U,
        .num_query_heads = 6U,
        .num_kv_heads = 6U,
        .dropout_prob = 0.0F,
        .fp32_dest_acc_en = true,
        .atol = 3e-2F,
        .rtol = 3e-2F,
        .test_name = "NanoGPTConfig (B=64, S=256, D=128, H=6)"};
    run_sdpa_backward_test(config);
}

TEST_F(SDPABackwardTest, LargerSequence) {
    SDPABackwardTestConfig config{
        .batch_size = 4U,
        .sequence_length = 512U,
        .query_dim = 128U,
        .key_value_dim = 128U,
        .num_query_heads = 8U,
        .num_kv_heads = 8U,
        .dropout_prob = 0.0F,
        .fp32_dest_acc_en = true,
        .atol = 3e-2F,
        .rtol = 3e-2F,
        .test_name = "LargerSequence (B=4, S=512, D=128, H=8)"};
    run_sdpa_backward_test(config);
}

TEST_F(SDPABackwardTest, GroupedQueryAttention) {
    // Test GQA: more query heads than kv heads
    SDPABackwardTestConfig config{
        .batch_size = 2U,
        .sequence_length = 128U,
        .query_dim = 64U,
        .key_value_dim = 64U,
        .num_query_heads = 8U,
        .num_kv_heads = 2U,  // 4 query heads per kv head
        .dropout_prob = 0.0F,
        .fp32_dest_acc_en = true,
        .atol = 3e-2F,
        .rtol = 3e-2F,
        .test_name = "GroupedQueryAttention (qH=8, kvH=2)"};
    run_sdpa_backward_test(config);
}

TEST_F(SDPABackwardTest, TinyLlamaConfig) {
    // Match TinyLlama training config from configs/training_shakespeare_tinyllama.yaml
    // num_heads: 32, num_groups: 4, embedding_dim: 2048, max_sequence_length: 2048
    // head_dim = 2048 / 32 = 64
    // heads_per_group = 32 / 4 = 8
    SDPABackwardTestConfig config{
        .batch_size = 1U,
        .sequence_length = 256U,  // Using smaller seq for faster test (full is 2048)
        .query_dim = 64U,         // head_dim = embedding_dim / num_heads = 2048 / 32
        .key_value_dim = 64U,
        .num_query_heads = 32U,   // num_heads from config
        .num_kv_heads = 4U,       // num_groups from config (8 query heads per kv head)
        .dropout_prob = 0.0F,
        .fp32_dest_acc_en = true,
        .atol = 3e-2F,
        .rtol = 3e-2F,
        .test_name = "TinyLlamaConfig (B=1, S=256, D=64, qH=32, kvH=4)"};
    run_sdpa_backward_test(config);
}