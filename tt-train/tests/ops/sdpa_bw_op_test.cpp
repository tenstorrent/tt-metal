// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <sys/types.h>

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
    size_t B = shape[0], H = shape[1], S = shape[2];
    xt::xarray<float> mask = xt::zeros<float>({B, H, S, S});

    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t s = 0; s < S; ++s) {
                for (size_t w = 0; w <= s; ++w) {
                    mask(b, h, s, w) = 1.0F;  // causal mask - upper triangular part
                }
            }
        }
    }
    return mask;
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
            /* dL_dV */ dL_dV};
}

TEST_F(SDPABackwardTest, SDPABackwardTest_SmallBatch) {
    using namespace ttml;
    const uint32_t B = 1U, qNH = 1U, kvNH = 1U, S = 64U, qD = 64U, kvD = 64U;
    const float dropout_probability = 0.0F;
    const bool fp32_dest_acc_en = true;

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

    // Create attention mask in kernel-expected format (B, qNH, S, S)
    xt::xarray<float> attn_mask_tensor = generate_attn_mask(query_tensor);

    xt::xarray<float> grad_output_tensor = xt::empty<float>({B, qNH, S, qD});
    ttml::core::parallel_generate(
        std::span{grad_output_tensor.data(), grad_output_tensor.size()},
        []() { return std::uniform_real_distribution<float>(-1.0F, 1.0F); },
        seed);

    auto query = core::from_xtensor(query_tensor, &autograd::ctx().get_device());
    auto key = core::from_xtensor(key_tensor, &autograd::ctx().get_device());
    auto value = core::from_xtensor(value_tensor, &autograd::ctx().get_device());
    auto attn_mask = core::from_xtensor(attn_mask_tensor, &autograd::ctx().get_device());
    auto grad_output = core::from_xtensor(grad_output_tensor, &autograd::ctx().get_device());

    auto composite_output = composite_sdpa(query, key, value, grad_output, attn_mask, /*return_intermediate=*/true);
    auto attn_output = /* attn_output */ composite_output[0];
    auto max_value = /* max_value */ composite_output[1];
    auto recip_sum_exp = /* recip_sum_exp */ composite_output[2];
    auto dL_dQ = /* dL_dQ */ composite_output[3];
    auto dL_dK = /* dL_dK */ composite_output[4];
    auto dL_dV = /* dL_dV */ composite_output[5];

    auto forward_intermediates = ttnn::concat(std::vector<ttnn::Tensor>{max_value, recip_sum_exp}, 3);

    auto op_result = ttml::metal::sdpa_bw(
        grad_output,
        attn_output,
        query,
        key,
        value,
        attn_mask,
        forward_intermediates,
        dropout_probability,
        fp32_dest_acc_en);

    auto sdpa_bw_dV = core::to_xtensor(op_result[2]);  // dL_dV

    auto composite_dV = core::to_xtensor(dL_dV);

    assert(sdpa_bw_dV.shape() == composite_dV.shape());
    EXPECT_TRUE(xt::allclose(sdpa_bw_dV, composite_dV, 2e-2F, 2e-2F));
    EXPECT_TRUE(false);
}