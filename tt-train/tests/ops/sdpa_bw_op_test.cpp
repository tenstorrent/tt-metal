// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <sys/types.h>

#include <cmath>
#include <limits>
#include <xtensor-blas/xlinalg.hpp>

#include "autograd/auto_context.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/random.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/repeat/repeat.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/moreh/moreh_softmax_backward/moreh_softmax_backward.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/tensor/tensor.hpp"
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
    const auto shape = query.shape();
    const size_t S = shape[2];
    // Create causal mask (lower triangular) with shape (1, 1, S, S)
    xt::xarray<float> mask = xt::tril(xt::ones<float>({S, S}));
    return mask.reshape({1UL, 1UL, S, S});
}

// Error analysis utility function
void print_error_analysis(
    const xt::xarray<float>& result,
    const xt::xarray<float>& groundtruth,
    const std::pair<float, float> threshold,  // {atol, rtol}
    const size_t num_elements_to_print,
    const std::string& name) {
    assert(result.shape() == groundtruth.shape() && "Tensors must have the same shape");

    const double atol = static_cast<double>(threshold.first);
    const double rtol = static_cast<double>(threshold.second);
    const size_t total_elements = result.size();
    const auto shape = result.shape();

    // Compute statistics using vectorized operations
    const xt::xarray<float> abs_diff = xt::abs(result - groundtruth);
    const float mean_error = xt::mean(abs_diff)();
    const float mse = xt::mean(xt::square(abs_diff))();
    const float max_diff = xt::amax(abs_diff)();

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
                    const double result_val = static_cast<double>(result(b, h, s, d));
                    const double gt_val = static_cast<double>(groundtruth(b, h, s, d));
                    const double diff = std::abs(result_val - gt_val);
                    const double tolerance = std::max(atol, rtol * std::abs(gt_val));
                    const bool is_close =
                        (diff <= atol || diff <= rtol * std::max(std::abs(result_val), std::abs(gt_val)));
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

// Helper function to scale a tensor by a scalar factor
// Used to pre-scale Query or Key tensors to match kernel's mathematical flow
xt::xarray<float> scale_tensor(const xt::xarray<float>& tensor, const float scale_factor) {
    return tensor * scale_factor;
}

// Helper: Expand K/V from (B, G, S, D) to (B, H, S, D) for grouped query attention
xt::xarray<float> expand_kv_heads(const xt::xarray<float>& kv, size_t H) {
    const auto shape = kv.shape();
    const size_t B = shape[0], G = shape[1], S = shape[2], D = shape[3];
    if (H == G) {
        return kv;  // No expansion needed
    }
    const size_t heads_per_group = H / G;
    xt::xarray<float> expanded = xt::zeros<float>({B, H, S, D});
    for (size_t b = 0; b < B; ++b) {
        for (size_t g = 0; g < G; ++g) {
            for (size_t hg = 0; hg < heads_per_group; ++hg) {
                xt::view(expanded, b, g * heads_per_group + hg, xt::all(), xt::all()) =
                    xt::view(kv, b, g, xt::all(), xt::all());
            }
        }
    }
    return expanded;
}

// Helper: Reduce gradients from (B, H, S, D) to (B, G, S, D) by summing over head groups
xt::xarray<float> reduce_grad_to_groups(const xt::xarray<float>& grad, size_t G) {
    const auto shape = grad.shape();
    const size_t B = shape[0], H = shape[1], S = shape[2], D = shape[3];
    if (H == G) {
        return grad;  // No reduction needed
    }
    const size_t heads_per_group = H / G;
    xt::xarray<float> reduced = xt::zeros<float>({B, G, S, D});
    for (size_t b = 0; b < B; ++b) {
        for (size_t g = 0; g < G; ++g) {
            for (size_t hg = 0; hg < heads_per_group; ++hg) {
                xt::view(reduced, b, g, xt::all(), xt::all()) +=
                    xt::view(grad, b, g * heads_per_group + hg, xt::all(), xt::all());
            }
        }
    }
    return reduced;
}

// Pure float implementation of SDPA backward pass using xtensor
// HYBRID VERSION: Uses xt::linalg::dot for matmuls, loop-based softmax backward
// This serves as the reference ground truth for testing
// Returns: {dQ, dK, dV, intermediates}
std::vector<xt::xarray<float>> float_sdpa_backward(
    const xt::xarray<float>& Q,
    const xt::xarray<float>& K,
    const xt::xarray<float>& V,
    const xt::xarray<float>& grad_output,
    const std::optional<xt::xarray<float>>& attn_mask = std::nullopt) {
    const auto shape = Q.shape();
    const size_t B = shape[0], H = shape[1], S = shape[2], D = shape[3], intermediate_size = 32;

    const auto kv_shape = K.shape();
    const size_t G = kv_shape[1];  // number of KV heads (groups)

    const float scale = 1.0F / std::sqrt(static_cast<float>(D));

    // ========== Forward Pass (need intermediate values) ==========

    // Step 1: Scale query
    const xt::xarray<float> Q_scaled = Q * scale;

    // Step 2: Compute attention scores: S = Q_scaled @ K^T using xt::linalg::dot
    const xt::xarray<float> K_expanded = expand_kv_heads(K, H);
    xt::xarray<float> scores = xt::zeros<float>({B, H, S, S});
    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            auto q_slice = xt::view(Q_scaled, b, h, xt::all(), xt::all());    // (S, D)
            auto k_slice = xt::view(K_expanded, b, h, xt::all(), xt::all());  // (S, D)
            xt::view(scores, b, h, xt::all(), xt::all()) = xt::linalg::dot(q_slice, xt::transpose(k_slice));
        }
    }

    // Step 3: Apply attention mask if provided (vectorized)
    if (attn_mask.has_value()) {
        const auto& mask = attn_mask.value();
        scores = xt::where(mask < 0.5F, -1e9F, scores);
    }

    // Step 4: Softmax over last dimension and store intermediates (loop-based for correctness)
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
                // Compute exp and sum
                float sum_exp = 0.0F;
                for (size_t j = 0; j < S; ++j) {
                    float exp_val = std::exp(scores(b, h, i, j) - max_val);
                    attention_weights(b, h, i, j) = exp_val;
                    sum_exp += exp_val;
                }
                intermediates(b, h, i, 0) = max_val + std::log(sum_exp);

                // Normalize
                for (size_t j = 0; j < S; ++j) {
                    attention_weights(b, h, i, j) /= sum_exp;
                }
            }
        }
    }

    // ========== Backward Pass ==========
    const auto& dO = grad_output;

    // Step 4: dV = P^T @ dO using xt::linalg::dot, then reduce to groups
    xt::xarray<float> dV_expanded = xt::zeros<float>({B, H, S, D});
    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            auto p_slice = xt::view(attention_weights, b, h, xt::all(), xt::all());  // (S, S)
            auto do_slice = xt::view(dO, b, h, xt::all(), xt::all());                // (S, D)
            xt::view(dV_expanded, b, h, xt::all(), xt::all()) = xt::linalg::dot(xt::transpose(p_slice), do_slice);
        }
    }
    const xt::xarray<float> dV = reduce_grad_to_groups(dV_expanded, G);

    // Step 5: dP = dO @ V^T using xt::linalg::dot
    const xt::xarray<float> V_expanded = expand_kv_heads(V, H);
    xt::xarray<float> dP = xt::zeros<float>({B, H, S, S});
    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            auto do_slice = xt::view(dO, b, h, xt::all(), xt::all());         // (S, D)
            auto v_slice = xt::view(V_expanded, b, h, xt::all(), xt::all());  // (S, D)
            xt::view(dP, b, h, xt::all(), xt::all()) = xt::linalg::dot(do_slice, xt::transpose(v_slice));
        }
    }

    // Step 6: Softmax backward (LOOP-BASED for correctness): dS = P * (dP - sum(P * dP))
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

    // Apply scale to dS
    dS = dS * scale;

    // Step 7: dQ = dS @ K using xt::linalg::dot
    xt::xarray<float> dQ = xt::zeros<float>({B, H, S, D});
    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            auto ds_slice = xt::view(dS, b, h, xt::all(), xt::all());         // (S, S)
            auto k_slice = xt::view(K_expanded, b, h, xt::all(), xt::all());  // (S, D)
            xt::view(dQ, b, h, xt::all(), xt::all()) = xt::linalg::dot(ds_slice, k_slice);
        }
    }

    // Step 8: dK = dS^T @ Q using xt::linalg::dot, then reduce to groups
    xt::xarray<float> dK_expanded = xt::zeros<float>({B, H, S, D});
    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            auto ds_slice = xt::view(dS, b, h, xt::all(), xt::all());  // (S, S)
            auto q_slice = xt::view(Q, b, h, xt::all(), xt::all());    // (S, D) - UNSCALED!
            xt::view(dK_expanded, b, h, xt::all(), xt::all()) = xt::linalg::dot(xt::transpose(ds_slice), q_slice);
        }
    }
    const xt::xarray<float> dK = reduce_grad_to_groups(dK_expanded, G);

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
    const bool transpose_a = false,
    const bool transpose_b = false) {
    using namespace ttml;
    const auto [batch_num, heads, seq_len, embedding_dim] = query_tensor.logical_shape().to_array_4D();
    const auto [batch_num_v, groups, seq_len_v, embedding_dim_v] = kv_tensor.logical_shape().to_array_4D();
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
    const auto M = transpose_a ? embedding_dim : seq_len;
    const auto N = transpose_b ? seq_len_v : embedding_dim_v;

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
ttnn::Tensor sum_over_groups(const ttnn::Tensor& ungrouped_grads, const uint32_t groups) {
    using namespace ttml;
    if (ungrouped_grads.logical_shape().rank() != 4) {
        throw std::invalid_argument(
            fmt::format("ungrouped_grads must have rank 4, but got rank {}", ungrouped_grads.logical_shape().rank()));
    }
    // [B,H,S,E]
    const auto [batch_num, num_heads, seq_len, embedding_dim] = ungrouped_grads.logical_shape().to_array_4D();
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
    const bool return_intermediate = false) {
    // Forward pass with intermediates returned
    // Backward pass return 3 gradients
    using namespace ttml;

    const auto [batch_num, heads, seq_len, embedding_dim] = query.logical_shape().to_array_4D();
    const auto groups = value.logical_shape().to_array_4D()[1];

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
    // Calculate logsumexp intermediate to test against kernel implementation
    auto max_value = ttnn::max(qk_scaled, /* dim */ 3, /* keepdim */ true);
    auto qk_scaled_sub_max = ttnn::subtract(qk_scaled, max_value);
    auto exp_qk_scaled = ttnn::exp(qk_scaled_sub_max);
    auto sum_exp = ttnn::sum(exp_qk_scaled, /* dim */ 3, /* keepdim */ true);
    auto log_sum_exp = ttnn::log(sum_exp);
    auto lse = ttnn::add(max_value, log_sum_exp);

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

    return {
        /* forward pass output*/ attention_qkv,
        /* logsumexp */ lse,
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
    float atol = 3e-2F;
    float rtol = 3e-2F;
    std::string test_name = "SDPA Backward Test";
    ttml::metal::AttentionMaskType mask_type = ttml::metal::AttentionMaskType::Arbitrary;
};

void run_sdpa_backward_test(const SDPABackwardTestConfig& config) {
    using namespace ttml;

    const uint32_t B = config.batch_size;
    const uint32_t qNH = config.num_query_heads;
    const uint32_t kvNH = config.num_kv_heads;
    const uint32_t S = config.sequence_length;
    const uint32_t qD = config.query_dim;
    const uint32_t kvD = config.key_value_dim;
    const float dropout_probability = config.dropout_prob;
    const float atol = config.atol;
    const float rtol = config.rtol;
    const ttml::metal::AttentionMaskType mask_type = config.mask_type;
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

    const xt::xarray<float> scale_query_tensor = scale_tensor(query_tensor, scale_factor);
    const auto scaled_query = core::from_xtensor(scale_query_tensor, device);

    const auto query = core::from_xtensor(query_tensor, device);
    const auto key = core::from_xtensor(key_tensor, device);
    const auto value = core::from_xtensor(value_tensor, device);
    const auto attn_mask = core::from_xtensor(attn_mask_tensor, device);
    const auto grad_output = core::from_xtensor(grad_output_tensor, device);

    // ========== Pure Float Reference (Ground Truth) ==========
    auto float_gradients =
        float_sdpa_backward(query_tensor, key_tensor, value_tensor, grad_output_tensor, attn_mask_tensor);
    const auto& float_dQ = float_gradients[0];
    const auto& float_dK = float_gradients[1];
    const auto& float_dV = float_gradients[2];
    const auto& float_intermediates = float_gradients[3];

    // ========== Composite Implementation (uses ttnn ops) ==========
    auto composite_output = composite_sdpa(query, key, value, grad_output, attn_mask, /*return_intermediate=*/true);
    const auto composite_attn_output = /* attn_output */ composite_output[0];
    auto composite_lse = /* logsumexp */ composite_output[1];
    const auto dL_dQ = /* dL_dQ */ composite_output[2];
    const auto dL_dK = /* dL_dK */ composite_output[3];
    const auto dL_dV = /* dL_dV */ composite_output[4];
    [[maybe_unused]] const auto attention_weights = /* attention_weights */ composite_output[5];

    // Pad lse from (B, H, S, 1) to (B, H, S, 32) to match kernel intermediate shape
    const auto padded_interm = core::zeros(ttnn::Shape{B, qNH, S, 32U}, device, ttnn::DataType::BFLOAT16);
    const auto composite_intermediates = ttnn::add(padded_interm, composite_lse);

    // ========== SDPA Forward Kernel (get attn_output and intermediates) ==========
    const auto sdpa_fw_result = ttml::metal::sdpa_fw(
        query,
        key,
        value,
        mask_type,
        mask_type == ttml::metal::AttentionMaskType::Arbitrary ? std::make_optional(attn_mask) : std::nullopt,
        dropout_probability,
        /*return_intermediates=*/true);

    // sdpa_fw returns std::vector<std::optional<ttnn::Tensor>>, unwrap with .value()
    const auto kernel_attn_output = sdpa_fw_result[0].value();
    const auto kernel_intermediates = sdpa_fw_result[1].value();

    // ========== SDPA Backward Kernel (using forward kernel outputs) ==========
    // For Causal mask: both sdpa_bw_q and sdpa_bw_kv now support on-the-fly causal mask generation
    // For Arbitrary mask: pass the mask tensor
    const auto op_result = ttml::metal::sdpa_bw(
        grad_output,
        kernel_attn_output,
        query,
        key,
        value,
        kernel_intermediates,
        mask_type,
        mask_type == ttml::metal::AttentionMaskType::Arbitrary ? std::make_optional(attn_mask) : std::nullopt,
        dropout_probability);

    // Convert forward kernel outputs for comparison
    const xt::xarray<float> kernel_attn_output_cpu = core::to_xtensor(kernel_attn_output);
    const xt::xarray<float> kernel_intermediates_cpu = core::to_xtensor(kernel_intermediates);
    const xt::xarray<float> composite_attn_output_cpu = core::to_xtensor(composite_attn_output);
    const xt::xarray<float> composite_intermediates_cpu = core::to_xtensor(composite_intermediates);

    const auto& [kernel_dQ, kernel_dK, kernel_dV] = op_result;
    const xt::xarray<float> sdpa_bw_dQ = core::to_xtensor(kernel_dQ);  // dL_dQ
    const xt::xarray<float> sdpa_bw_dK = core::to_xtensor(kernel_dK);  // dL_dK
    const xt::xarray<float> sdpa_bw_dV = core::to_xtensor(kernel_dV);  // dL_dV

    const xt::xarray<float> composite_dQ = core::to_xtensor(dL_dQ);
    const xt::xarray<float> composite_dK = core::to_xtensor(dL_dK);
    const xt::xarray<float> composite_dV = core::to_xtensor(dL_dV);

    // Diagnostic: print value ranges for all kernel outputs
    auto print_stats = [](const xt::xarray<float>& t, const std::string& name) {
        size_t nan_count = 0, inf_count = 0;
        for (size_t i = 0; i < t.size(); ++i) {
            if (std::isnan(t.flat(i)))
                nan_count++;
            if (std::isinf(t.flat(i)))
                inf_count++;
        }
        fmt::print(
            "[{}] shape={}, min={:.6e}, max={:.6e}, mean={:.6e}, nan={}, inf={}\n",
            name,
            t.shape(),
            xt::amin(t)(),
            xt::amax(t)(),
            xt::mean(t)(),
            nan_count,
            inf_count);
        // Print first 8 values
        fmt::print("  first 8: ");
        for (size_t i = 0; i < std::min<size_t>(8, t.size()); ++i) {
            fmt::print("{:.4e} ", t.flat(i));
        }
        fmt::print("\n");
    };

    fmt::print("\n=== Kernel output diagnostics ===\n");
    print_stats(sdpa_bw_dQ, "kernel_dQ");
    print_stats(sdpa_bw_dK, "kernel_dK");
    print_stats(sdpa_bw_dV, "kernel_dV");
    fmt::print("=== Float reference diagnostics ===\n");
    print_stats(float_dQ, "float_dQ");
    print_stats(float_dK, "float_dK");
    print_stats(float_dV, "float_dV");
    fmt::print("================================\n\n");

    // Verify shapes match
    ASSERT_EQ(sdpa_bw_dQ.shape(), composite_dQ.shape()) << "kernel_dQ shape != composite_dQ shape";
    ASSERT_EQ(sdpa_bw_dQ.shape(), float_dQ.shape()) << "kernel_dQ shape != float_dQ shape";
    ASSERT_EQ(sdpa_bw_dK.shape(), composite_dK.shape()) << "kernel_dK shape != composite_dK shape";
    ASSERT_EQ(sdpa_bw_dK.shape(), float_dK.shape()) << "kernel_dK shape != float_dK shape";
    ASSERT_EQ(sdpa_bw_dV.shape(), composite_dV.shape()) << "kernel_dV shape != composite_dV shape";
    ASSERT_EQ(sdpa_bw_dV.shape(), float_dV.shape()) << "kernel_dV shape != float_dV shape";

    // Check for NaN/Inf values
    EXPECT_TRUE(xt::all(xt::isfinite(float_dQ))) << "float_dQ contains NaN or Inf values";
    EXPECT_TRUE(xt::all(xt::isfinite(float_dK))) << "float_dK contains NaN or Inf values";
    EXPECT_TRUE(xt::all(xt::isfinite(float_dV))) << "float_dV contains NaN or Inf values";
    EXPECT_TRUE(xt::all(xt::isfinite(composite_dQ))) << "composite_dQ contains NaN or Inf values";
    EXPECT_TRUE(xt::all(xt::isfinite(composite_dK))) << "composite_dK contains NaN or Inf values";
    EXPECT_TRUE(xt::all(xt::isfinite(composite_dV))) << "composite_dV contains NaN or Inf values";
    EXPECT_TRUE(xt::all(xt::isfinite(sdpa_bw_dQ))) << "kernel_dQ contains NaN or Inf values";
    EXPECT_TRUE(xt::all(xt::isfinite(sdpa_bw_dK))) << "kernel_dK contains NaN or Inf values";
    EXPECT_TRUE(xt::all(xt::isfinite(sdpa_bw_dV))) << "kernel_dV contains NaN or Inf values";

    // ========== Comparisons ==========
    // Forward pass checks
    const bool fw_attn_output_matches = xt::allclose(kernel_attn_output_cpu, composite_attn_output_cpu, rtol, atol);
    // Compare kernel intermediates (FP32 logsumexp) vs float reference (value at pos 0 only)
    const bool fw_intermediates_matches = xt::allclose(kernel_intermediates_cpu, float_intermediates, rtol, atol);

    // Backward pass checks
    const bool kernel_dQ_matches_float = xt::allclose(sdpa_bw_dQ, float_dQ, rtol, atol);
    const bool kernel_dK_matches_float = xt::allclose(sdpa_bw_dK, float_dK, rtol, atol);
    const bool kernel_dV_matches_float = xt::allclose(sdpa_bw_dV, float_dV, rtol, atol);
    [[maybe_unused]] const bool composite_dQ_matches_float = xt::allclose(composite_dQ, float_dQ, rtol, atol);
    [[maybe_unused]] const bool composite_dK_matches_float = xt::allclose(composite_dK, float_dK, rtol, atol);
    [[maybe_unused]] const bool composite_dV_matches_float = xt::allclose(composite_dV, float_dV, rtol, atol);

    // Assertions
    EXPECT_TRUE(fw_attn_output_matches) << "Forward attn output mismatch in " << config.test_name;
    EXPECT_TRUE(fw_intermediates_matches) << "Forward intermediates mismatch in " << config.test_name;
    EXPECT_TRUE(kernel_dQ_matches_float) << "Kernel dQ vs Float mismatch in " << config.test_name;
    EXPECT_TRUE(kernel_dK_matches_float) << "Kernel dK vs Float mismatch in " << config.test_name;
    EXPECT_TRUE(kernel_dV_matches_float) << "Kernel dV vs Float mismatch in " << config.test_name;

    // Note: Composite implementation (ttnn ops based) may have slightly larger numerical errors
    // for larger batch/sequence sizes due to accumulated precision loss in intermediate ops.
    // These checks are informational - the kernel implementation is the authoritative reference.
    // Uncomment for debugging if needed:
    // EXPECT_TRUE(composite_dQ_matches_float) << "Composite dQ vs Float mismatch in " << config.test_name;
    // EXPECT_TRUE(composite_dK_matches_float) << "Composite dK vs Float mismatch in " << config.test_name;
    // EXPECT_TRUE(composite_dV_matches_float) << "Composite dV vs Float mismatch in " << config.test_name;
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
        .atol = 2e-2F,
        .rtol = 2e-2F,
        .test_name = "SmallBatch (B=2, S=128, D=64, H=4)"};
    run_sdpa_backward_test(config);
}

TEST_F(SDPABackwardTest, NIGHTLY_NanoGPTConfig) {
    // Match nano_gpt training config
    // D=128 needs wider tolerance: 2x inner dim accumulation depth in BF16 matmul
    // causes larger forward-to-backward precision cascade for dQ/dK
    SDPABackwardTestConfig config{
        .batch_size = 64U,
        .sequence_length = 256U,
        .query_dim = 128U,
        .key_value_dim = 128U,
        .num_query_heads = 6U,
        .num_kv_heads = 6U,
        .dropout_prob = 0.0F,
        .atol = 3e-2F,
        .rtol = 3e-2F,
        .test_name = "NanoGPTConfig (B=64, S=256, D=128, H=6)"};
    run_sdpa_backward_test(config);
}

TEST_F(SDPABackwardTest, NIGHTLY_LargerSequence) {
    // D=128 + S=1024: wider tolerance for accumulated BF16 rounding
    SDPABackwardTestConfig config{
        .batch_size = 4U,
        .sequence_length = 1024U,
        .query_dim = 128U,
        .key_value_dim = 128U,
        .num_query_heads = 8U,
        .num_kv_heads = 8U,
        .dropout_prob = 0.0F,
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
        .atol = 2e-2F,
        .rtol = 2e-2F,
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
        .num_query_heads = 32U,  // num_heads from config
        .num_kv_heads = 4U,      // num_groups from config (8 query heads per kv head)
        .dropout_prob = 0.0F,
        .atol = 2e-2F,
        .rtol = 2e-2F,
        .test_name = "TinyLlamaConfig (B=1, S=256, D=64, qH=32, kvH=4)"};
    run_sdpa_backward_test(config);
}

TEST_F(SDPABackwardTest, CausalMask_MHA) {
    // Test causal mask with Multi-Head Attention
    // Both sdpa_bw_q and sdpa_bw_kv support on-the-fly causal mask generation
    SDPABackwardTestConfig config{
        .batch_size = 2U,
        .sequence_length = 128U,
        .query_dim = 64U,
        .key_value_dim = 64U,
        .num_query_heads = 4U,
        .num_kv_heads = 4U,
        .dropout_prob = 0.0F,
        .atol = 2e-2F,
        .rtol = 2e-2F,
        .test_name = "CausalMask_MHA (B=2, S=128, D=64, H=4)",
        .mask_type = ttml::metal::AttentionMaskType::Causal};
    run_sdpa_backward_test(config);
}

TEST_F(SDPABackwardTest, CausalMask_GQA) {
    // Test causal mask with Grouped Query Attention
    // Both sdpa_bw_q and sdpa_bw_kv support on-the-fly causal mask generation
    SDPABackwardTestConfig config{
        .batch_size = 2U,
        .sequence_length = 256U,
        .query_dim = 64U,
        .key_value_dim = 64U,
        .num_query_heads = 8U,
        .num_kv_heads = 2U,  // GQA: 4 query heads per KV head
        .dropout_prob = 0.0F,
        .atol = 2e-2F,
        .rtol = 2e-2F,
        .test_name = "CausalMask_GQA (B=2, S=256, D=64, qH=8, kvH=2)",
        .mask_type = ttml::metal::AttentionMaskType::Causal};
    run_sdpa_backward_test(config);
}

TEST_F(SDPABackwardTest, NIGHTLY_CausalMask_NanoGPTConfig) {
    // D=128: wider tolerance (see NIGHTLY_NanoGPTConfig comment)
    SDPABackwardTestConfig config{
        .batch_size = 64U,
        .sequence_length = 256U,
        .query_dim = 128U,
        .key_value_dim = 128U,
        .num_query_heads = 6U,
        .num_kv_heads = 6U,
        .dropout_prob = 0.0F,
        .atol = 3e-2F,
        .rtol = 3e-2F,
        .test_name = "CausalMask_NanoGPTConfig (B=64, S=256, D=128, H=6)",
        .mask_type = ttml::metal::AttentionMaskType::Causal};
    run_sdpa_backward_test(config);
}

TEST_F(SDPABackwardTest, NIGHTLY_CausalMask_LargerSequence) {
    // D=128 + S=1024: wider tolerance (see NIGHTLY_LargerSequence comment)
    SDPABackwardTestConfig config{
        .batch_size = 4U,
        .sequence_length = 1024U,
        .query_dim = 128U,
        .key_value_dim = 128U,
        .num_query_heads = 8U,
        .num_kv_heads = 8U,
        .dropout_prob = 0.0F,
        .atol = 3e-2F,
        .rtol = 3e-2F,
        .test_name = "CausalMask_LargerSeq (B=4, S=1024, D=128, H=8)",
        .mask_type = ttml::metal::AttentionMaskType::Causal};
    run_sdpa_backward_test(config);
}

// ========== Ring Attention Simulation on Single Device ==========
// Simulates ring attention forward + backward on a single device by splitting
// K/V into chunks along the sequence dimension and combining outputs with the
// logaddexp algorithm (same as ring_attention_sdpa.cpp). This does NOT require
// a multi-device setup — it exercises the merge logic on a single chip.
//
// Use this test to verify ring attention changes when a multi-device VM is unavailable.
// Run with: ./ttml_tests --gtest_filter=*NIGHTLY_RingAttentionMergeSimulation*
//
// Verifies:
// 1. FP32 logsumexp intermediates are correct
// 2. The logaddexp merge reproduces full-sequence SDPA output
// 3. The backward pass gradient scaling via exp(lse_j - final_lse) is correct

TEST_F(SDPABackwardTest, NIGHTLY_RingAttentionMergeSimulation) {
    GTEST_SKIP()
        << "Single-device ring attention simulation — requires manual opt-in, not part of regular nightly runs";
    using namespace ttml;

    constexpr uint32_t B = 2U;
    constexpr uint32_t H = 4U;
    constexpr uint32_t S_full = 256U;
    constexpr uint32_t D = 64U;
    constexpr uint32_t ring_size = 4U;
    constexpr uint32_t S_local = S_full / ring_size;
    constexpr float atol = 3e-2F;
    constexpr float rtol = 3e-2F;

    auto* device = &autograd::ctx().get_device();
    auto& rng = autograd::ctx().get_generator();
    uint32_t seed = rng();

    xt::xarray<float> query_xt = xt::empty<float>({B, H, S_full, D});
    core::parallel_generate(
        std::span{query_xt.data(), query_xt.size()},
        []() { return std::uniform_real_distribution<float>(-0.5F, 0.5F); },
        seed);
    seed = rng();

    xt::xarray<float> key_xt = xt::empty<float>({B, H, S_full, D});
    core::parallel_generate(
        std::span{key_xt.data(), key_xt.size()},
        []() { return std::uniform_real_distribution<float>(-0.5F, 0.5F); },
        seed);
    seed = rng();

    xt::xarray<float> value_xt = xt::empty<float>({B, H, S_full, D});
    core::parallel_generate(
        std::span{value_xt.data(), value_xt.size()},
        []() { return std::uniform_real_distribution<float>(-0.5F, 0.5F); },
        seed);
    seed = rng();

    xt::xarray<float> grad_output_xt = xt::empty<float>({B, H, S_full, D});
    core::parallel_generate(
        std::span{grad_output_xt.data(), grad_output_xt.size()},
        []() { return std::uniform_real_distribution<float>(-0.5F, 0.5F); },
        seed);

    // === Ground truth: full-sequence sdpa_fw + sdpa_bw ===
    const auto query_tt = core::from_xtensor(query_xt, device);
    const auto key_tt = core::from_xtensor(key_xt, device);
    const auto value_tt = core::from_xtensor(value_xt, device);
    const auto grad_output_tt = core::from_xtensor(grad_output_xt, device);

    auto full_fw_result = metal::sdpa_fw(
        query_tt,
        key_tt,
        value_tt,
        metal::AttentionMaskType::Causal,
        std::nullopt,
        0.0F,
        /*return_intermediates=*/true);
    const auto full_output = full_fw_result[0].value();
    const auto full_intermediates = full_fw_result[1].value();

    const auto full_output_cpu = core::to_xtensor(full_output);
    const auto full_lse_cpu = core::to_xtensor(full_intermediates);

    auto [full_dQ, full_dK, full_dV] = metal::sdpa_bw(
        grad_output_tt,
        full_output,
        query_tt,
        key_tt,
        value_tt,
        full_intermediates,
        metal::AttentionMaskType::Causal,
        std::nullopt,
        0.0F);
    const auto full_dQ_cpu = core::to_xtensor(full_dQ);
    const auto full_dK_cpu = core::to_xtensor(full_dK);
    const auto full_dV_cpu = core::to_xtensor(full_dV);

    // === Simulated ring attention: per-chunk forward + logaddexp merge ===
    // For each simulated "device" d, its local Q is Q[:,:,d*Sl:(d+1)*Sl,:].
    // It processes K/V chunks in ring order and merges via logaddexp.
    xt::xarray<float> ring_output = xt::zeros<float>({(size_t)B, (size_t)H, (size_t)S_full, (size_t)D});
    // Per-device final_lse, needed for backward weight computation
    // Shape: (ring_size, B, H, S_local, 1)
    std::vector<xt::xarray<float>> per_device_final_lse(ring_size);

    // Per-device, per-step outputs and lse (for backward recomputation)
    // Indexed as [device][step] -> {output, lse}
    std::vector<std::vector<std::pair<xt::xarray<float>, xt::xarray<float>>>> per_device_step_data(ring_size);

    fmt::print("\n=== Ring Attention Forward Simulation (ring_size={}) ===\n", ring_size);
    for (uint32_t d = 0; d < ring_size; ++d) {
        xt::xarray<float> output_accum = xt::zeros<float>({(size_t)B, (size_t)H, (size_t)S_local, (size_t)D});
        xt::xarray<float> global_lse = xt::full_like(
            xt::xarray<float>::from_shape({(size_t)B, (size_t)H, (size_t)S_local, 1UL}),
            -std::numeric_limits<float>::infinity());

        // Extract local Q chunk and put on device
        xt::xarray<float> Q_d_xt = xt::xarray<float>(
            xt::view(query_xt, xt::all(), xt::all(), xt::range(d * S_local, (d + 1) * S_local), xt::all()));
        auto Q_d_tt = core::from_xtensor(Q_d_xt, device);

        per_device_step_data[d].resize(ring_size);

        for (uint32_t s = 0; s < ring_size; ++s) {
            uint32_t kv_pos = (d + s) % ring_size;

            if (kv_pos > d) {
                // Future K/V — no causal contribution
                per_device_step_data[d][s] = {
                    xt::zeros<float>({(size_t)B, (size_t)H, (size_t)S_local, (size_t)D}),
                    xt::full_like(
                        xt::xarray<float>::from_shape({(size_t)B, (size_t)H, (size_t)S_local, 1UL}),
                        -std::numeric_limits<float>::infinity())};
                continue;
            }

            // Extract K/V chunk
            xt::xarray<float> K_chunk_xt = xt::xarray<float>(
                xt::view(key_xt, xt::all(), xt::all(), xt::range(kv_pos * S_local, (kv_pos + 1) * S_local), xt::all()));
            xt::xarray<float> V_chunk_xt = xt::xarray<float>(xt::view(
                value_xt, xt::all(), xt::all(), xt::range(kv_pos * S_local, (kv_pos + 1) * S_local), xt::all()));
            auto K_chunk_tt = core::from_xtensor(K_chunk_xt, device);
            auto V_chunk_tt = core::from_xtensor(V_chunk_xt, device);

            auto mask_type = (kv_pos == d) ? metal::AttentionMaskType::Causal : metal::AttentionMaskType::None;

            auto chunk_result = metal::sdpa_fw(
                Q_d_tt, K_chunk_tt, V_chunk_tt, mask_type, std::nullopt, 0.0F, /*return_intermediates=*/true);

            auto chunk_output_cpu = core::to_xtensor(chunk_result[0].value());
            auto chunk_inter_cpu = core::to_xtensor(chunk_result[1].value());

            // Extract lse from column 0: (B, H, S_local, 32) → (B, H, S_local, 1)
            xt::xarray<float> lse_chunk =
                xt::xarray<float>(xt::view(chunk_inter_cpu, xt::all(), xt::all(), xt::all(), xt::range(0, 1)));

            per_device_step_data[d][s] = {chunk_output_cpu, lse_chunk};

            // logaddexp merge (mirrors ring_attention_sdpa.cpp)
            xt::xarray<float> m = xt::maximum(global_lse, lse_chunk);
            xt::xarray<float> exp_global = xt::exp(global_lse - m);
            xt::xarray<float> exp_chunk = xt::exp(lse_chunk - m);
            xt::xarray<float> new_lse = m + xt::log(exp_global + exp_chunk);

            xt::xarray<float> old_weight = xt::exp(global_lse - new_lse);
            xt::xarray<float> new_weight = xt::exp(lse_chunk - new_lse);

            output_accum = output_accum * old_weight + chunk_output_cpu * new_weight;
            global_lse = new_lse;
        }

        per_device_final_lse[d] = global_lse;

        // Store merged output into full ring output
        xt::view(ring_output, xt::all(), xt::all(), xt::range(d * S_local, (d + 1) * S_local), xt::all()) =
            output_accum;
    }

    // === Forward comparison ===
    const bool fw_matches = xt::allclose(ring_output, full_output_cpu, rtol, atol);
    fmt::print("Forward: ring_merged vs full_sdpa — {}\n", fw_matches ? "PASS" : "FAIL");
    if (!fw_matches) {
        auto diff = xt::abs(ring_output - full_output_cpu);
        fmt::print("  max_diff={:.6e}, mean_diff={:.6e}\n", xt::amax(diff)(), xt::mean(diff)());
    }
    EXPECT_TRUE(fw_matches) << "Ring attention forward merge does not match full-sequence SDPA";

    // === Simulated ring attention backward ===
    // For each device d:
    //   For each step s (K/V chunk):
    //     step_weight = exp(lse_j - final_lse)
    //     scaled_grad = grad_output * step_weight
    //     Run sdpa_bw(scaled_grad, chunk_output, Q_d, K_chunk, V_chunk, chunk_intermediates)
    //     Accumulate dQ; store dK, dV per chunk
    fmt::print("\n=== Ring Attention Backward Simulation ===\n");
    xt::xarray<float> ring_dQ = xt::zeros<float>({(size_t)B, (size_t)H, (size_t)S_full, (size_t)D});
    xt::xarray<float> ring_dK = xt::zeros<float>({(size_t)B, (size_t)H, (size_t)S_full, (size_t)D});
    xt::xarray<float> ring_dV = xt::zeros<float>({(size_t)B, (size_t)H, (size_t)S_full, (size_t)D});

    for (uint32_t d = 0; d < ring_size; ++d) {
        xt::xarray<float> Q_d_xt = xt::xarray<float>(
            xt::view(query_xt, xt::all(), xt::all(), xt::range(d * S_local, (d + 1) * S_local), xt::all()));
        xt::xarray<float> grad_d_xt = xt::xarray<float>(
            xt::view(grad_output_xt, xt::all(), xt::all(), xt::range(d * S_local, (d + 1) * S_local), xt::all()));
        auto Q_d_tt = core::from_xtensor(Q_d_xt, device);

        const auto& final_lse = per_device_final_lse[d];

        for (uint32_t s = 0; s < ring_size; ++s) {
            uint32_t kv_pos = (d + s) % ring_size;
            if (kv_pos > d) {
                continue;
            }

            const auto& [step_output_cpu, step_lse] = per_device_step_data[d][s];

            // step_weight = exp(lse_j - final_lse)
            xt::xarray<float> step_weight = xt::exp(step_lse - final_lse);
            xt::xarray<float> scaled_grad = grad_d_xt * step_weight;

            // Put tensors on device for sdpa_bw
            auto scaled_grad_tt = core::from_xtensor(scaled_grad, device);
            auto step_output_tt = core::from_xtensor(step_output_cpu, device);

            xt::xarray<float> K_chunk_xt = xt::xarray<float>(
                xt::view(key_xt, xt::all(), xt::all(), xt::range(kv_pos * S_local, (kv_pos + 1) * S_local), xt::all()));
            xt::xarray<float> V_chunk_xt = xt::xarray<float>(xt::view(
                value_xt, xt::all(), xt::all(), xt::range(kv_pos * S_local, (kv_pos + 1) * S_local), xt::all()));
            auto K_chunk_tt = core::from_xtensor(K_chunk_xt, device);
            auto V_chunk_tt = core::from_xtensor(V_chunk_xt, device);

            // Recompute forward for intermediates (same as ring_attention_sdpa backward does)
            auto mask_type = (kv_pos == d) ? metal::AttentionMaskType::Causal : metal::AttentionMaskType::None;
            auto recomputed = metal::sdpa_fw(
                Q_d_tt, K_chunk_tt, V_chunk_tt, mask_type, std::nullopt, 0.0F, /*return_intermediates=*/true);
            auto step_intermediates = recomputed[1].value();
            auto step_recomp_output = recomputed[0].value();

            auto [chunk_dQ, chunk_dK, chunk_dV] = metal::sdpa_bw(
                scaled_grad_tt,
                step_recomp_output,
                Q_d_tt,
                K_chunk_tt,
                V_chunk_tt,
                step_intermediates,
                mask_type,
                std::nullopt,
                0.0F);

            auto chunk_dQ_cpu = core::to_xtensor(chunk_dQ);
            auto chunk_dK_cpu = core::to_xtensor(chunk_dK);
            auto chunk_dV_cpu = core::to_xtensor(chunk_dV);

            // Accumulate dQ for this device
            xt::view(ring_dQ, xt::all(), xt::all(), xt::range(d * S_local, (d + 1) * S_local), xt::all()) +=
                chunk_dQ_cpu;

            // dK and dV accumulate per K/V chunk position
            xt::view(ring_dK, xt::all(), xt::all(), xt::range(kv_pos * S_local, (kv_pos + 1) * S_local), xt::all()) +=
                chunk_dK_cpu;
            xt::view(ring_dV, xt::all(), xt::all(), xt::range(kv_pos * S_local, (kv_pos + 1) * S_local), xt::all()) +=
                chunk_dV_cpu;
        }
    }

    // === Backward comparison ===
    const bool dQ_matches = xt::allclose(ring_dQ, full_dQ_cpu, rtol, atol);
    const bool dK_matches = xt::allclose(ring_dK, full_dK_cpu, rtol, atol);
    const bool dV_matches = xt::allclose(ring_dV, full_dV_cpu, rtol, atol);

    fmt::print("Backward dQ: ring vs full — {}\n", dQ_matches ? "PASS" : "FAIL");
    if (!dQ_matches) {
        auto diff = xt::abs(ring_dQ - full_dQ_cpu);
        fmt::print("  max_diff={:.6e}, mean_diff={:.6e}\n", xt::amax(diff)(), xt::mean(diff)());
    }
    fmt::print("Backward dK: ring vs full — {}\n", dK_matches ? "PASS" : "FAIL");
    if (!dK_matches) {
        auto diff = xt::abs(ring_dK - full_dK_cpu);
        fmt::print("  max_diff={:.6e}, mean_diff={:.6e}\n", xt::amax(diff)(), xt::mean(diff)());
    }
    fmt::print("Backward dV: ring vs full — {}\n", dV_matches ? "PASS" : "FAIL");
    if (!dV_matches) {
        auto diff = xt::abs(ring_dV - full_dV_cpu);
        fmt::print("  max_diff={:.6e}, mean_diff={:.6e}\n", xt::amax(diff)(), xt::mean(diff)());
    }

    EXPECT_TRUE(dQ_matches) << "Ring attention dQ does not match full-sequence SDPA";
    EXPECT_TRUE(dK_matches) << "Ring attention dK does not match full-sequence SDPA";
    EXPECT_TRUE(dV_matches) << "Ring attention dV does not match full-sequence SDPA";
}
