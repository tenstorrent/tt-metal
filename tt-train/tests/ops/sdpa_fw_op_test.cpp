// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <sys/types.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt_stl/reflection.hpp>
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/system_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/common/const_utils.hpp"
#include "metal/operations.hpp"
#include "metal/ops/sdpa_fw/device/sdpa_fw_config.hpp"
#include "metal/ops/sdpa_fw/device/sdpa_fw_device_operation_types.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/repeat/repeat.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn_fixed/matmuls.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"
#include "xtensor/generators/xbuilder.hpp"

using ::testing::HasSubstr;
using ::testing::ThrowsMessage;

class SDPAForwardTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
        ttml::autograd::ctx().set_seed(42);
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

xt::xarray<float> generate_mask(const xt::xarray<float>& query) {
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

// Split (B, 1, S, d)  -->  (B, H, S, d/H)
// Assumes: d % num_heads == 0
xt::xarray<float> split_heads(const xt::xarray<float>& input, std::uint32_t num_heads) {
    const std::size_t B = input.shape()[0];
    const std::size_t S = input.shape()[2];
    const std::size_t d = input.shape()[3];
    const std::size_t H = static_cast<std::size_t>(num_heads);
    const std::size_t Dh = d / H;

    xt::xarray<float> out = xt::xarray<float>::from_shape({B, H, S, Dh});
    std::fill(out.begin(), out.end(), 0.0f);

    for (std::size_t b = 0; b < B; ++b) {
        for (std::size_t s = 0; s < S; ++s) {
            for (std::size_t h = 0; h < H; ++h) {
                const std::size_t in_base = h * Dh;  // slice in last dim
                for (std::size_t t = 0; t < Dh; ++t) {
                    out(b, h, s, t) = input(b, 0, s, in_base + t);
                }
            }
        }
    }
    return out;
}

// Fuse (B, H, S, Dh)  -->  (B, 1, S, H*Dh)
// Assumes: input.shape()[1] == num_heads
xt::xarray<float> fuse_heads(const xt::xarray<float>& heads, std::uint32_t num_heads) {
    const std::size_t B = heads.shape()[0];
    const std::size_t H = heads.shape()[1];  // should equal num_heads
    const std::size_t S = heads.shape()[2];
    const std::size_t Dh = heads.shape()[3];
    (void)num_heads;  // kept for symmetry with split; not used

    const std::size_t d = H * Dh;

    xt::xarray<float> out = xt::xarray<float>::from_shape({B, std::size_t(1), S, d});
    std::fill(out.begin(), out.end(), 0.0f);

    for (std::size_t b = 0; b < B; ++b) {
        for (std::size_t s = 0; s < S; ++s) {
            for (std::size_t h = 0; h < H; ++h) {
                const std::size_t out_base = h * Dh;  // slice in last dim
                for (std::size_t t = 0; t < Dh; ++t) {
                    out(b, 0, s, out_base + t) = heads(b, h, s, t);
                }
            }
        }
    }
    return out;
}

// Naive reference SDPA with grouped KV (no validation).
// Inputs (physical):
//   Q: (B, 1, S, qD)
//   K: (B, 1, S, kvD)
//   V: (B, 1, S, kvD)
//   attn_mask: (B, 1, S, S)   // additive: 0 keep, large negative to mask
// Heads (passed in):
//   query_heads -> cast to q_heads
//   key_heads   -> cast to kv_heads
// Assumes: qD / q_heads == kvD / kv_heads
// Returns: (B, 1, S, qD)
xt::xarray<float> sdpa_grouped_naive(
    const xt::xarray<float>& Q,
    const xt::xarray<float>& K,
    const xt::xarray<float>& V,
    const xt::xarray<float>& attn_mask,
    std::uint32_t query_heads,
    std::uint32_t key_heads) {
    // local aliases with your preferred names
    const std::size_t q_heads = static_cast<std::size_t>(query_heads);
    const std::size_t kv_heads = static_cast<std::size_t>(key_heads);

    const std::size_t B = Q.shape()[0];
    const std::size_t S = Q.shape()[2];
    const std::size_t qD = Q.shape()[3];

    const std::size_t Dh_q = qD / q_heads;
    const std::size_t Dh = Dh_q;  // assume Dh_q == Dh_kv

    xt::xarray<float> Out = xt::xarray<float>::from_shape({B, std::size_t(1), S, qD});
    std::fill(Out.begin(), Out.end(), 0.0F);

    auto group_of_head = [&](std::size_t h) -> std::size_t {
        // contiguous block mapping
        return (h * kv_heads) / q_heads;
    };

    const float scale = 1.0F / std::sqrt(static_cast<float>(Dh));
    std::vector<float> scores_row(S);

    for (std::size_t b = 0; b < B; ++b) {
        for (std::size_t h = 0; h < q_heads; ++h) {
            const std::size_t g = group_of_head(h);
            const std::size_t q_off = h * Dh;   // Q slice
            const std::size_t kv_off = g * Dh;  // KV slice

            for (std::size_t i = 0; i < S; ++i) {
                // scores_row[j] = (q_i · k_j) * scale + mask(i,j)
                float rmax = -INFINITY;
                for (std::size_t j = 0; j < S; ++j) {
                    float dot = 0.0f;
                    for (std::size_t t = 0; t < Dh; ++t) {
                        dot += Q(b, 0, i, q_off + t) * K(b, 0, j, kv_off + t);
                    }
                    const float m = attn_mask(0, 0, i, j);  // expected 0 or 1, mask is (1,1,S,S)
                    const float s = m * (dot * scale) + (m - 1.0F) * 1e9F;
                    // float s = dot * scale + attn_mask(b, 0, i, j);
                    scores_row[j] = s;
                    rmax = std::max(s, rmax);  // <- changed line
                }

                // softmax over j
                float denom = 0.0F;
                for (std::size_t j = 0; j < S; ++j) denom += std::exp(scores_row[j] - rmax);
                denom = std::max(denom, 1e-20F);

                // out_i[h] = sum_j softmax_ij * V[j]
                for (std::size_t t = 0; t < Dh; ++t) {
                    float acc = 0.0F;
                    for (std::size_t j = 0; j < S; ++j) {
                        float w = std::exp(scores_row[j] - rmax) / denom;
                        acc += w * V(b, 0, j, kv_off + t);
                    }
                    Out(b, 0, i, q_off + t) = acc;
                }
            }
        }
    }

    return Out;
}

// Extended version that also returns intermediate results (1/sum_exp)
// This version expects UNSPLIT tensors (B, 1, S, D) with head count parameters
std::pair<xt::xarray<float>, xt::xarray<float>> sdpa_grouped_naive_with_intermediates(
    const xt::xarray<float>& Q,
    const xt::xarray<float>& K,
    const xt::xarray<float>& V,
    const xt::xarray<float>& attn_mask,
    std::uint32_t query_heads,
    std::uint32_t key_heads) {
    // local aliases with your preferred names
    const std::size_t q_heads = static_cast<std::size_t>(query_heads);
    const std::size_t kv_heads = static_cast<std::size_t>(key_heads);

    const std::size_t B = Q.shape()[0];
    const std::size_t S = Q.shape()[2];
    const std::size_t qD = Q.shape()[3];

    const std::size_t Dh_q = qD / q_heads;
    const std::size_t Dh = Dh_q;  // assume Dh_q == Dh_kv

    xt::xarray<float> Out = xt::xarray<float>::from_shape({B, std::size_t(1), S, qD});
    std::fill(Out.begin(), Out.end(), 0.0F);

    // Intermediates: (B, q_heads, S, 1) - logsumexp per head per sequence position
    xt::xarray<float> Intermediates = xt::xarray<float>::from_shape({B, q_heads, S, std::size_t(1)});
    std::fill(Intermediates.begin(), Intermediates.end(), 0.0F);

    auto group_of_head = [&](std::size_t h) -> std::size_t {
        // contiguous block mapping
        return (h * kv_heads) / q_heads;
    };

    const float scale = 1.0F / std::sqrt(static_cast<float>(Dh));
    std::vector<float> scores_row(S);

    for (std::size_t b = 0; b < B; ++b) {
        for (std::size_t h = 0; h < q_heads; ++h) {
            const std::size_t g = group_of_head(h);
            const std::size_t q_off = h * Dh;   // Q slice
            const std::size_t kv_off = g * Dh;  // KV slice

            for (std::size_t i = 0; i < S; ++i) {
                // scores_row[j] = (q_i · k_j) * scale + mask(i,j)
                float rmax = -INFINITY;
                for (std::size_t j = 0; j < S; ++j) {
                    float dot = 0.0f;
                    for (std::size_t t = 0; t < Dh; ++t) {
                        dot += Q(b, 0, i, q_off + t) * K(b, 0, j, kv_off + t);
                    }
                    const float m = attn_mask(0, 0, i, j);  // expected 0 or 1, mask is (1,1,S,S)
                    const float s = m * (dot * scale) + (m - 1.0F) * 1e9F;
                    scores_row[j] = s;
                    rmax = std::max(s, rmax);
                }

                // softmax over j
                float denom = 0.0F;
                for (std::size_t j = 0; j < S; ++j) denom += std::exp(scores_row[j] - rmax);
                denom = std::max(denom, 1e-20F);

                // Store intermediate: logsumexp = max + log(sum_exp)
                Intermediates(b, h, i, 0) = rmax + std::log(denom);

                // out_i[h] = sum_j softmax_ij * V[j]
                for (std::size_t t = 0; t < Dh; ++t) {
                    float acc = 0.0F;
                    for (std::size_t j = 0; j < S; ++j) {
                        float w = std::exp(scores_row[j] - rmax) / denom;
                        acc += w * V(b, 0, j, kv_off + t);
                    }
                    Out(b, 0, i, q_off + t) = acc;
                }
            }
        }
    }

    return std::make_pair(Out, Intermediates);
}

// New version that works with SPLIT-BY-HEADS tensors (B, H, S, D/H) and outputs SPLIT format (B, H, S, Dh)
std::pair<xt::xarray<float>, xt::xarray<float>> sdpa_split_heads_naive_with_intermediates(
    const xt::xarray<float>& Q_split,
    const xt::xarray<float>& K_split,
    const xt::xarray<float>& V_split,
    const xt::xarray<float>& attn_mask) {
    const std::size_t B = Q_split.shape()[0];
    const std::size_t q_heads = Q_split.shape()[1];
    const std::size_t kv_heads = K_split.shape()[1];
    const std::size_t S = Q_split.shape()[2];
    const std::size_t Dh_qk = Q_split.shape()[3];  // Q/K head dim (used for QK^T dot product)
    const std::size_t Dh_v = V_split.shape()[3];   // V head dim (can differ, determines output width)

    // Output in SPLIT format (B, H, S, Dh_v) - heads NOT fused, output width matches V
    xt::xarray<float> Out = xt::xarray<float>::from_shape({B, q_heads, S, Dh_v});
    std::fill(Out.begin(), Out.end(), 0.0F);

    // Intermediates: (B, q_heads, S, 32) - logsumexp at col 0, rest zero-padded
    constexpr std::size_t kIntermediateWidth = 32U;
    xt::xarray<float> Intermediates = xt::xarray<float>::from_shape({B, q_heads, S, kIntermediateWidth});
    std::fill(Intermediates.begin(), Intermediates.end(), 0.0F);

    auto group_of_head = [&](std::size_t h) -> std::size_t {
        // contiguous block mapping for grouped KV
        return (h * kv_heads) / q_heads;
    };

    const float scale = 1.0F / std::sqrt(static_cast<float>(Dh_qk));
    std::vector<float> scores_row(S);

    for (std::size_t b = 0; b < B; ++b) {
        for (std::size_t h = 0; h < q_heads; ++h) {
            const std::size_t g = group_of_head(h);

            for (std::size_t i = 0; i < S; ++i) {
                // scores_row[j] = (q_i · k_j) * scale + mask(i,j)
                float rmax = -INFINITY;
                for (std::size_t j = 0; j < S; ++j) {
                    float dot = 0.0f;
                    for (std::size_t t = 0; t < Dh_qk; ++t) {
                        dot += Q_split(b, h, i, t) * K_split(b, g, j, t);
                    }
                    const float m = attn_mask(0, 0, i, j);  // expected 0 or 1, mask is (1,1,S,S)
                    const float s = m * (dot * scale) + (m - 1.0F) * 1e9F;
                    scores_row[j] = s;
                    rmax = std::max(s, rmax);
                }

                // softmax over j
                float denom = 0.0F;
                for (std::size_t j = 0; j < S; ++j) denom += std::exp(scores_row[j] - rmax);
                denom = std::max(denom, 1e-20F);

                // Store intermediate: logsumexp = max + log(sum_exp)
                Intermediates(b, h, i, 0) = rmax + std::log(denom);

                // out_i[h] = sum_j softmax_ij * V[j] - store in SPLIT format (B, H, S, Dh_v)
                for (std::size_t t = 0; t < Dh_v; ++t) {
                    float acc = 0.0F;
                    for (std::size_t j = 0; j < S; ++j) {
                        float w = std::exp(scores_row[j] - rmax) / denom;
                        acc += w * V_split(b, g, j, t);
                    }
                    Out(b, h, i, t) = acc;  // Store in split format (B, H, S, Dh_v)
                }
            }
        }
    }

    return std::make_pair(Out, Intermediates);
}

float compute_mse(const xt::xarray<float>& expected, const xt::xarray<float>& result) {
    assert(result.shape() == expected.shape());
    xt::xarray<float> diff = expected - result;
    float mse = xt::mean(xt::square(diff))();
    return mse;
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
ttnn::Tensor group_shared_matmul(
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

// Round every element to the nearest bf16 (ties to even), which is what from_xtensor does on upload.
// Host references built from a rounded gate see exactly the gate the kernel sees.
xt::xarray<float> round_to_bf16(xt::xarray<float> values) {
    for (auto& v : values) {
        v = static_cast<float>(bfloat16(v));
    }
    return values;
}

// Reference gating: out * sigmoid(gate), elementwise. Both tensors are (B, H, S, Dh_v).
xt::xarray<float> apply_gate_reference(const xt::xarray<float>& out, const xt::xarray<float>& gate) {
    assert(out.shape() == gate.shape());
    xt::xarray<float> sig = 1.0F / (1.0F + xt::exp(-gate));
    return out * sig;
}

std::vector<ttnn::Tensor> composite_sdpa_fw(
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    const std::optional<ttnn::Tensor>& attn_mask,
    const std::optional<ttnn::Tensor>& gate = std::nullopt) {
    // std::vector<ttnn::Tensor> result;
    // result.reserve(2U);  // one for output, one for intermediate if needed

    using namespace ttml;
    auto [batch_num, heads, seq_len, embedding_dim] = query.logical_shape().to_array_4D();

    const float scale = 1.0F / std::sqrt(static_cast<float>(embedding_dim));
    constexpr auto none = ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam>{};
    auto q_scaled = ttnn::multiply(query, scale, std::nullopt, std::nullopt, std::nullopt, none, none, none);
    ttnn::Tensor qk_scaled = group_shared_matmul(q_scaled, key, /*transpose_a=*/false, /*transpose_b=*/true);

    // σQ @ K
    if (attn_mask.has_value()) {
        ttnn::Tensor mask_tensor = attn_mask.value();
        qk_scaled = ttnn::add(
            ttnn::multiply(mask_tensor, qk_scaled, std::nullopt, std::nullopt, std::nullopt, none, none, none),
            ttnn::multiply(
                ttnn::subtract(mask_tensor, 1.F, std::nullopt, std::nullopt, std::nullopt, none, none, none),
                1e9F,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                none,
                none,
                none),
            std::nullopt,
            std::nullopt,
            std::nullopt,
            none,
            none,
            none);
    }

    // Calculate intermediate results to test against kernel implementation
    auto max_value = ttnn::max(qk_scaled, /* dim */ 3, /* keepdim */ true);  // (B, H, S, 1)
    auto qk_scaled_sub_max = ttnn::subtract(qk_scaled, max_value);
    auto exp_qk_scaled = ttnn::exp(qk_scaled_sub_max);
    auto sum_exp = ttnn::sum(exp_qk_scaled, /* dim */ 3, /* keepdim */ true);

    // Build intermediates tensor with shape (B, H, S, 32)
    // Format: logsumexp = max + log(sum_exp) at col 0, rest zero-padded
    auto* device = query.device();
    auto lse = ttnn::add(max_value, ttnn::log(sum_exp));  // (B, H, S, 1)
    auto padded_zeros = core::zeros(ttnn::Shape{batch_num, heads, seq_len, 31U}, device, lse.dtype());
    auto intermediates = ttnn::concat(std::vector<ttnn::Tensor>{lse, padded_zeros}, 3);

    auto attention_weights = ttml::metal::softmax(qk_scaled, /* axis */ 3);

    auto attention_qkv = group_shared_matmul(attention_weights, value, /*transpose_a=*/false, /*transpose_b=*/false);

    // Optional output gating: out = out * sigmoid(gate). Gate is (B, H, S, Dh_v), same as the output.
    if (gate.has_value()) {
        auto sig_gate = ttnn::sigmoid(gate.value());
        attention_qkv =
            ttnn::multiply(attention_qkv, sig_gate, std::nullopt, std::nullopt, std::nullopt, none, none, none);
    }
    return {attention_qkv, intermediates};
}

struct SDPATestConfig {
    uint32_t batch_size;
    uint32_t sequence_length;
    uint32_t query_dim;
    uint32_t key_value_dim;
    uint32_t value_dim =
        0U;  // 0 means same as key_value_dim (for backward compat); total V dim = num_key_heads * head_dim_v
    uint32_t num_query_heads;
    uint32_t num_key_heads;
    ttml::metal::AttentionMaskType mask_type = ttml::metal::AttentionMaskType::Causal;  // default: causal mask
    // Optional output gate: when true a gate tensor of shape (B, num_query_heads, S, head_dim_v) is generated,
    // passed to the kernel, and the expected output becomes sdpa(Q,K,V) * sigmoid(gate).
    bool use_gate = false;
    float gate_min = -4.0F;  // sigmoid(-4) ~ 0.018
    float gate_max = 4.0F;   // sigmoid(+4) ~ 0.982
    float dropout_prob = 0.0F;
    // 0 = program factory default.
    ttml::metal::SDPAForwardProgramConfig program_config{};
    float result_atol = 2e-2F;
    float result_rtol = 2e-2F;
    float intermediate_atol = 2e-2F;
    float intermediate_rtol = 2e-2F;
    bool return_intermediates = true;
    std::optional<uint32_t> seed;  // input RNG seed; unset = fresh from the context generator
    std::string test_name = "SDPA Test";
};

// kernel_output, when given, receives the kernel's output so callers can compare runs with each other.
void run_sdpa_test(const SDPATestConfig& config, xt::xarray<float>* kernel_output = nullptr) {
    using namespace ttml;

    ASSERT_GT(config.num_query_heads, 0U) << "num_query_heads must be greater than zero";
    ASSERT_GT(config.num_key_heads, 0U) << "num_key_heads must be greater than zero";
    const uint32_t effective_value_dim = config.value_dim > 0 ? config.value_dim : config.key_value_dim;
    ASSERT_EQ(config.query_dim % config.num_query_heads, 0U) << "query_dim must be divisible by num_query_heads";
    ASSERT_EQ(config.key_value_dim % config.num_key_heads, 0U) << "key_value_dim must be divisible by num_key_heads";
    ASSERT_EQ(effective_value_dim % config.num_key_heads, 0U) << "value_dim must be divisible by num_key_heads";

    // Generate already split-by-heads tensors directly
    const uint32_t head_dim_q = config.query_dim / config.num_query_heads;
    const uint32_t head_dim_kv = config.key_value_dim / config.num_key_heads;
    const uint32_t head_dim_v = effective_value_dim / config.num_key_heads;

    const uint32_t seed = config.seed.has_value() ? config.seed.value() : ttml::autograd::ctx().get_generator()();

    const std::array<std::size_t, 4> query_shape{
        config.batch_size, config.num_query_heads, config.sequence_length, head_dim_q};
    const std::array<std::size_t, 4> kv_shape{
        config.batch_size, config.num_key_heads, config.sequence_length, head_dim_kv};

    xt::xarray<float> query_tensor = ttml::test_utils::make_uniform_xarray<float>(query_shape, -1.0F, 1.0F, seed);

    xt::xarray<float> key_tensor = ttml::test_utils::make_uniform_xarray<float>(kv_shape, -1.0F, 1.0F, seed);

    const std::array<std::size_t, 4> value_shape{
        config.batch_size, config.num_key_heads, config.sequence_length, head_dim_v};
    xt::xarray<float> value_tensor = ttml::test_utils::make_uniform_xarray<float>(value_shape, -1.0F, 1.0F, seed);

    // Optional gate: (B, num_query_heads, S, head_dim_v) - same shape as the kernel output.
    // Use a different seed so the gate is decorrelated from V (same shape, same seed would give identical data).
    std::optional<xt::xarray<float>> gate_tensor;
    if (config.use_gate) {
        const std::array<std::size_t, 4> gate_shape{
            config.batch_size, config.num_query_heads, config.sequence_length, head_dim_v};
        gate_tensor = round_to_bf16(
            ttml::test_utils::make_uniform_xarray<float>(gate_shape, config.gate_min, config.gate_max, seed + 1U));
    }

    // Create attention mask in kernel-expected format (1, 1, S, S) - broadcasted across batches/heads.
    // For None the kernel computes unmasked attention, so the reference mask must keep
    // every position (all-ones) instead of the causal pattern.
    xt::xarray<float> attn_mask_tensor;
    if (config.mask_type == ttml::metal::AttentionMaskType::None) {
        const auto seq = static_cast<std::size_t>(config.sequence_length);
        attn_mask_tensor = xt::ones<float>({1UL, 1UL, seq, seq});
    } else {
        attn_mask_tensor = generate_mask(query_tensor);
    }

    // Convert to device tensors
    auto query = core::from_xtensor(query_tensor, &autograd::ctx().get_device());
    auto key = core::from_xtensor(key_tensor, &autograd::ctx().get_device());
    auto value = core::from_xtensor(value_tensor, &autograd::ctx().get_device());
    const bool return_intermediates = config.return_intermediates;

    // For Causal mask_type: kernel generates mask on-the-fly, we pass std::nullopt
    // For Arbitrary mask_type: we pass attn_mask tensor to kernel
    // For None mask_type: no mask at all
    std::optional<ttnn::Tensor> kernel_mask = std::nullopt;
    if (config.mask_type == ttml::metal::AttentionMaskType::Arbitrary) {
        kernel_mask = core::from_xtensor(attn_mask_tensor, &autograd::ctx().get_device());
    }

    std::optional<ttnn::Tensor> kernel_gate = std::nullopt;
    if (gate_tensor.has_value()) {
        kernel_gate = core::from_xtensor(gate_tensor.value(), &autograd::ctx().get_device());
    }

    // Run SDPA kernel with new interface - this is our reference implementation
    auto result = ttml::metal::sdpa_fw(
        query,
        key,
        value,
        config.mask_type,
        kernel_mask,
        config.dropout_prob,
        return_intermediates,
        kernel_gate,
        config.program_config);
    xt::xarray<float> result_xtensor =
        core::to_xtensor(result[0].value());  // Kernel returns (B, H, S, D) - heads NOT fused
    if (kernel_output != nullptr) {
        *kernel_output = result_xtensor;
    }
    xt::xarray<float> interm_xtensor;
    if (return_intermediates) {
        ASSERT_TRUE(result[1].has_value()) << "intermediates requested but not returned in " << config.test_name;
        interm_xtensor = core::to_xtensor(result[1].value());
    } else {
        ASSERT_FALSE(result[1].has_value()) << "intermediates returned although not requested in " << config.test_name;
    }

    // Run composite SDPA implementation with split tensors - output is (B, H, S, D)
    // Composite always needs the mask tensor for comparison (even when kernel generates it on-the-fly)
    auto attn_mask_device = core::from_xtensor(attn_mask_tensor, &autograd::ctx().get_device());
    auto composite_result_split = composite_sdpa_fw(query, key, value, attn_mask_device, kernel_gate);
    xt::xarray<float> composite_result_xtensor = core::to_xtensor(composite_result_split[0]);  // Already (B, H, S, D)
    xt::xarray<float> composite_interm_xtensor = core::to_xtensor(composite_result_split[1]);

    // Run float reference implementation with split tensors - now outputs (B, H, S, D) format
    auto [float_result, float_intermediates] =
        sdpa_split_heads_naive_with_intermediates(query_tensor, key_tensor, value_tensor, attn_mask_tensor);
    if (gate_tensor.has_value()) {
        float_result = apply_gate_reference(float_result, gate_tensor.value());
    }

    // Gate-specific checks against the kernel's own ungated output. These do not depend on the composite or
    // float references, so they isolate the gating step from the rest of the attention math. The gate step is
    // one fp32 multiply and one bf16 rounding, so with the gate already rounded to bf16 on the host the result
    // must land within ~2 bf16 ULP of ungated * sigmoid(gate): rtol 8e-3 covers 2 ULP of the mantissa and atol
    // 1e-4 only absorbs values that underflow bf16 resolution around zero.
    if (gate_tensor.has_value()) {
        constexpr float kGateAtol = 1e-4F;
        constexpr float kGateRtol = 8e-3F;
        // The gate CB takes L1, so the gated run may resolve different chunks than an ungated run would.
        // Reproduce the gated run's pick and pin the ungated run to it: both must use the same numerics.
        auto* device = &autograd::ctx().get_device();
        const uint64_t l1_budget_bytes =
            device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
        const auto grid = device->compute_with_storage_grid_size();
        const auto picked = ttml::metal::ops::sdpa_fw::device::resolve_chunks(
            config.program_config,
            ttml::metal::ops::sdpa_fw::device::SDPAForwardShape{
                .batch = config.batch_size,
                .q_heads = config.num_query_heads,
                .kv_heads = config.num_key_heads,
                .St = config.sequence_length / 32U,
                .qWt = head_dim_kv / 32U,
                .vWt = head_dim_v / 32U,
                .mask_type = config.mask_type,
                .has_gate = true,
                .return_intermediates = return_intermediates},
            l1_budget_bytes,
            grid.x * grid.y);
        const ttml::metal::SDPAForwardProgramConfig pinned{
            .q_chunk_size = picked.Sq_chunk_t * 32U,
            .k_chunk_size = picked.Sk_chunk_t * 32U,
            .heads_per_item = picked.heads_per_item};
        auto ungated = ttml::metal::sdpa_fw(
            query,
            key,
            value,
            config.mask_type,
            kernel_mask,
            config.dropout_prob,
            return_intermediates,
            std::nullopt,
            pinned);
        xt::xarray<float> ungated_xtensor = core::to_xtensor(ungated[0].value());
        xt::xarray<float> ungated_interm_xtensor = core::to_xtensor(ungated[1].value());

        ASSERT_EQ(result_xtensor.shape(), ungated_xtensor.shape())
            << "Gated and ungated kernel outputs must have the same shape in " << config.test_name;

        // (1) gated == ungated * sigmoid(gate)
        xt::xarray<float> expected_from_ungated = apply_gate_reference(ungated_xtensor, gate_tensor.value());
        EXPECT_TRUE(xt::allclose(result_xtensor, expected_from_ungated, kGateRtol, kGateAtol))
            << "Gated kernel output != ungated kernel output * sigmoid(gate) in " << config.test_name
            << " (MSE: " << compute_mse(expected_from_ungated, result_xtensor) << ")";

        // (2) the logsumexp is packed before the gate path runs, so it must be bit-identical.
        EXPECT_TRUE(xt::all(xt::equal(interm_xtensor, ungated_interm_xtensor)))
            << "Gate changed the intermediates (logsumexp) in " << config.test_name;
    }

    // All results are now in split format (B, H, S, D) - heads NOT fused
    // Shape validation - all should be in split format (B, H, S, D)
    ASSERT_EQ(result_xtensor.shape(), float_result.shape()) << "Kernel result shape mismatch in " << config.test_name;
    ASSERT_EQ(composite_result_xtensor.shape(), float_result.shape())
        << "Composite result shape mismatch in " << config.test_name;
    if (return_intermediates) {
        ASSERT_EQ(interm_xtensor.shape(), float_intermediates.shape())
            << "Intermediate shape mismatch in " << config.test_name;
    }

    // Compute MSE for validation
    float mse_kernel_vs_composite = compute_mse(result_xtensor, composite_result_xtensor);
    float mse_kernel_vs_float = compute_mse(result_xtensor, float_result);
    float mse_composite_vs_float = compute_mse(composite_result_xtensor, float_result);
    float mse_kernel_vs_composite_interm =
        return_intermediates ? compute_mse(interm_xtensor, composite_interm_xtensor) : 0.0F;

    // Primary validation: Kernel vs Composite (most reliable - both use same implementation approach)
    EXPECT_TRUE(xt::allclose(result_xtensor, composite_result_xtensor, config.result_atol, config.result_rtol))
        << "Kernel vs Composite comparison failed in " << config.test_name << " (MSE: " << mse_kernel_vs_composite
        << ")";

    // Secondary validation: Compare with float reference (may have numerical precision differences)
    bool float_impl_reliable =
        xt::allclose(composite_result_xtensor, float_result, config.result_atol * 10, config.result_rtol * 10);

    if (float_impl_reliable) {
        // Float implementation seems reliable, use normal tolerances
        EXPECT_TRUE(xt::allclose(result_xtensor, float_result, config.result_atol, config.result_rtol))
            << "Kernel vs Float result comparison failed in " << config.test_name << " (MSE: " << mse_kernel_vs_float
            << ")";

        EXPECT_TRUE(xt::allclose(composite_result_xtensor, float_result, config.result_atol, config.result_rtol))
            << "Composite vs Float result comparison failed in " << config.test_name
            << " (MSE: " << mse_composite_vs_float << ")";

        if (return_intermediates) {
            EXPECT_TRUE(
                xt::allclose(interm_xtensor, float_intermediates, config.intermediate_atol, config.intermediate_rtol))
                << "Intermediate result comparison failed in " << config.test_name;
        }
    } else if (return_intermediates) {
        // Float implementation unreliable, compare intermediates between kernel and composite instead
        EXPECT_TRUE(
            xt::allclose(interm_xtensor, composite_interm_xtensor, config.intermediate_atol, config.intermediate_rtol))
            << "Kernel vs Composite intermediate result comparison failed in " << config.test_name
            << " (MSE: " << mse_kernel_vs_composite_interm << ")";
    }
}

TEST_F(SDPAForwardTest, SDPAForwardTest_SmallBatch) {
    SDPATestConfig config{
        .batch_size = 1U,
        .sequence_length = 128U,
        .query_dim = 128U,
        .key_value_dim = 128U,
        .num_query_heads = 2U,
        .num_key_heads = 2U,
        .mask_type = ttml::metal::AttentionMaskType::Arbitrary,
        .test_name = "SmallBatch_2H_2KV"};
    run_sdpa_test(config);
}

TEST_F(SDPAForwardTest, SDPAForwardTest_SingleHead) {
    SDPATestConfig config{
        .batch_size = 1U,
        .sequence_length = 128U,
        .query_dim = 128U,
        .key_value_dim = 128U,
        .num_query_heads = 1U,
        .num_key_heads = 1U,
        .mask_type = ttml::metal::AttentionMaskType::Arbitrary,
        .test_name = "SingleHead_1H_1KV"};
    run_sdpa_test(config);
}

// =============================================================================
// CAUSAL MASK TESTS - Testing on-the-fly causal mask generation
// =============================================================================

TEST_F(SDPAForwardTest, SDPAForwardTest_CausalMask_Small) {
    // Simple causal mask test with small shapes
    SDPATestConfig config{
        .batch_size = 1U,
        .sequence_length = 128U,
        .query_dim = 128U,
        .key_value_dim = 128U,
        .num_query_heads = 2U,
        .num_key_heads = 2U,
        .mask_type = ttml::metal::AttentionMaskType::Causal,
        .test_name = "CausalMask_Small_2H"};
    run_sdpa_test(config);
}

TEST_F(SDPAForwardTest, SDPAForwardTest_CausalMask_SingleHead) {
    SDPATestConfig config{
        .batch_size = 1U,
        .sequence_length = 128U,
        .query_dim = 128U,
        .key_value_dim = 128U,
        .num_query_heads = 1U,
        .num_key_heads = 1U,
        .mask_type = ttml::metal::AttentionMaskType::Causal,
        .test_name = "CausalMask_SingleHead"};
    run_sdpa_test(config);
}

TEST_F(SDPAForwardTest, SDPAForwardTest_CausalMask_SingleTile) {
    // Single tile test (32 seq len = 1 tile row) - everything on one core, simplest case
    SDPATestConfig config{
        .batch_size = 1U,
        .sequence_length = 32U,
        .query_dim = 64U,
        .key_value_dim = 64U,
        .num_query_heads = 1U,
        .num_key_heads = 1U,
        .mask_type = ttml::metal::AttentionMaskType::Causal,
        .test_name = "CausalMask_SingleTile"};
    run_sdpa_test(config);
}

// Was disabled for non-deterministic accuracy failures on the old softmax path
// (https://github.com/tenstorrent/tt-metal/issues/46121); passes on the chunked kernel.
TEST_F(SDPAForwardTest, SDPAForwardTest_CausalMask_MHA_Batch4_Seq256) {
    SKIP_FOR_LLK_ASSERTS("Skip due to too large code size when assert is enabled.");
    // Multi-head attention with equal query and KV heads (standard MHA)
    // batch=4, seq=256 (8 tile rows), 6 heads with 128 dim per head
    SDPATestConfig config{
        .batch_size = 4U,
        .sequence_length = 256U,
        .query_dim = 768U,  // 6 heads * 128 dim per head
        .key_value_dim = 768U,
        .num_query_heads = 6U,
        .num_key_heads = 6U,
        .mask_type = ttml::metal::AttentionMaskType::Causal,
        .test_name = "CausalMask_MHA_4B_256S_6H"};
    run_sdpa_test(config);
}

// Was disabled for non-deterministic accuracy failures on the old softmax path
// (https://github.com/tenstorrent/tt-metal/issues/46121); passes on the chunked kernel.
TEST_F(SDPAForwardTest, NIGHTLY_SDPAForwardTest_CausalMask_GQA_Batch16_Seq512) {
    SKIP_FOR_LLK_ASSERTS("Skip due to too large code size when assert is enabled.");
    // Grouped Query Attention with different query and KV heads
    // batch=16, seq=512 (16 tile rows), 8 query heads, 4 KV heads (2:1 ratio)
    SDPATestConfig config{
        .batch_size = 16U,
        .sequence_length = 512U,
        .query_dim = 1024U,     // 8 heads * 128 dim per head
        .key_value_dim = 512U,  // 4 heads * 128 dim per head
        .num_query_heads = 8U,
        .num_key_heads = 4U,
        .mask_type = ttml::metal::AttentionMaskType::Causal,
        .test_name = "CausalMask_GQA_16B_512S_8Q_4KV"};
    run_sdpa_test(config);
}

TEST_F(SDPAForwardTest, SDPAForwardTest_SmallBatch_2Heads_1Group) {
    SDPATestConfig config{
        .batch_size = 1U,
        .sequence_length = 128U,
        .query_dim = 128U,
        .key_value_dim = 64U,
        .num_query_heads = 2U,
        .num_key_heads = 1U,
        .mask_type = ttml::metal::AttentionMaskType::Arbitrary,
        .test_name = "SmallBatch_2H_1KV_Grouped"};
    run_sdpa_test(config);
}

TEST_F(SDPAForwardTest, NIGHTLY_SDPAForwardTest_SmallBatch_12Heads_6Group) {
    SDPATestConfig config{
        .batch_size = 1U,
        .sequence_length = 1024U,
        .query_dim = 768U,
        .key_value_dim = 384U,
        .num_query_heads = 12U,
        .num_key_heads = 6U,
        .mask_type = ttml::metal::AttentionMaskType::Arbitrary,
        .test_name = "SmallBatch_12H_6KV_Grouped"};
    run_sdpa_test(config);
}

TEST_F(SDPAForwardTest, NIGHTLY_SDPAForwardTest_Batch_12Heads_6Group) {
    SDPATestConfig config{
        .batch_size = 16U,
        .sequence_length = 1024U,
        .query_dim = 768U,
        .key_value_dim = 384U,
        .num_query_heads = 12U,
        .num_key_heads = 6U,
        .mask_type = ttml::metal::AttentionMaskType::Arbitrary,
        .test_name = "Batch_16B_12H_6KV_Production"};
    run_sdpa_test(config);
}

// =============================================================================
// VALIDATION TESTS - Testing Error Conditions and Edge Cases
// =============================================================================

// Was disabled for non-deterministic accuracy failures on the old softmax path
// (https://github.com/tenstorrent/tt-metal/issues/46121); passes on the chunked kernel.
TEST_F(SDPAForwardTest, ValidationTest_EdgeCaseDimensions) {
    using namespace ttml;

    // Test Case 1: Minimum viable dimensions
    {
        SDPATestConfig config{
            .batch_size = 1U,
            .sequence_length = 32U,  // Minimum tile size
            .query_dim = 32U,        // Minimum tile size
            .key_value_dim = 32U,
            .num_query_heads = 1U,
            .num_key_heads = 1U,
            .mask_type = ttml::metal::AttentionMaskType::Arbitrary,
            .test_name = "EdgeCase_MinDimensions"};

        EXPECT_NO_THROW({ run_sdpa_test(config); }) << "Should handle minimum tile dimensions correctly";
    }

    // Test Case 2: Single head configuration
    {
        SDPATestConfig config{
            .batch_size = 1U,
            .sequence_length = 128U,
            .query_dim = 64U,
            .key_value_dim = 64U,
            .num_query_heads = 1U,
            .num_key_heads = 1U,
            .mask_type = ttml::metal::AttentionMaskType::Arbitrary,
            .test_name = "EdgeCase_SingleHead"};

        EXPECT_NO_THROW({ run_sdpa_test(config); }) << "Should handle single head attention correctly";
    }

    // Test Case 3: Progressive grouping ratios
    {
        SDPATestConfig config{
            .batch_size = 1U,
            .sequence_length = 128U,
            .query_dim = 128U,     // 4 heads * 32 dim per head
            .key_value_dim = 32U,  // 1 head * 32 dim per head
            .num_query_heads = 4U,
            .num_key_heads = 1U,
            .mask_type = ttml::metal::AttentionMaskType::Arbitrary,
            .test_name = "EdgeCase_Grouping_4to1"};

        EXPECT_NO_THROW({ run_sdpa_test(config); }) << "Should handle 4:1 grouping ratio correctly";
    }

    {
        SDPATestConfig config{
            .batch_size = 1U,
            .sequence_length = 128U,
            .query_dim = 256U,     // 8 heads * 32 dim per head
            .key_value_dim = 32U,  // 1 head * 32 dim per head
            .num_query_heads = 8U,
            .num_key_heads = 1U,
            .mask_type = ttml::metal::AttentionMaskType::Arbitrary,
            .test_name = "EdgeCase_MaxGrouping_8to1"};

        EXPECT_NO_THROW({ run_sdpa_test(config); }) << "Should handle 8:1 grouping ratio correctly";
    }
}

TEST_F(SDPAForwardTest, ValidationTest_IntermediateReturnModes) {
    using namespace ttml;

    const uint32_t B = 1U, S = 128U, d = 64U;

    // Create split-by-heads tensors for the new interface
    const uint32_t num_heads = 2U;
    const uint32_t head_dim = d / num_heads;

    auto& rng = ttml::autograd::ctx().get_generator();
    uint32_t seed = rng();

    const std::array<std::size_t, 4> split_shape{B, num_heads, S, head_dim};

    xt::xarray<float> query_tensor = ttml::test_utils::make_uniform_xarray<float>(split_shape, -1.0F, 1.0F, seed);

    xt::xarray<float> key_tensor = ttml::test_utils::make_uniform_xarray<float>(split_shape, -1.0F, 1.0F, seed);

    xt::xarray<float> value_tensor = ttml::test_utils::make_uniform_xarray<float>(split_shape, -1.0F, 1.0F, seed);

    // Create attention mask in kernel-expected format (1, 1, S, S) - broadcasted across batches/heads
    xt::xarray<float> attn_mask_tensor = generate_mask(query_tensor);

    // Test Case 1: return_intermediates = false
    {
        auto query = core::from_xtensor(query_tensor, &autograd::ctx().get_device());
        auto key = core::from_xtensor(key_tensor, &autograd::ctx().get_device());
        auto value = core::from_xtensor(value_tensor, &autograd::ctx().get_device());
        auto attn_mask = core::from_xtensor(attn_mask_tensor, &autograd::ctx().get_device());

        auto result =
            ttml::metal::sdpa_fw(query, key, value, ttml::metal::AttentionMaskType::Arbitrary, attn_mask, 0.0F, false);

        EXPECT_TRUE(result[0].has_value()) << "Main result should always be present";
        EXPECT_FALSE(result[1].has_value()) << "Intermediate should be null when return_intermediates=false";

        xt::xarray<float> result_xtensor = core::to_xtensor(result[0].value());
        // Kernel returns split format (B, H, S, Dh) - heads NOT fused
        std::vector<size_t> expected_shape = {B, num_heads, S, head_dim};
        EXPECT_EQ(result_xtensor.shape(), expected_shape) << "Result should be in split format (B, H, S, Dh)";
    }

    // Test Case 2: return_intermediates = true
    {
        auto query = core::from_xtensor(query_tensor, &autograd::ctx().get_device());
        auto key = core::from_xtensor(key_tensor, &autograd::ctx().get_device());
        auto value = core::from_xtensor(value_tensor, &autograd::ctx().get_device());
        auto attn_mask = core::from_xtensor(attn_mask_tensor, &autograd::ctx().get_device());

        auto result =
            ttml::metal::sdpa_fw(query, key, value, ttml::metal::AttentionMaskType::Arbitrary, attn_mask, 0.0F, true);

        EXPECT_TRUE(result[0].has_value()) << "Main result should be present";
        EXPECT_TRUE(result[1].has_value()) << "Intermediate should be present when return_intermediates=true";

        xt::xarray<float> result_xtensor = core::to_xtensor(result[0].value());
        xt::xarray<float> interm_xtensor = core::to_xtensor(result[1].value());

        // Kernel returns split format (B, H, S, Dh) - heads NOT fused
        std::vector<size_t> expected_shape = {B, num_heads, S, head_dim};
        EXPECT_EQ(result_xtensor.shape(), expected_shape) << "Result should be in split format (B, H, S, Dh)";

        // Check intermediate shape: (B, num_query_heads, S, 32)
        constexpr size_t kIntermediateWidth = 32U;
        std::vector<size_t> expected_interm_shape = {B, num_heads, S, kIntermediateWidth};
        EXPECT_EQ(interm_xtensor.shape(), expected_interm_shape) << "Intermediate shape should be (B, q_heads, S, 32)";

        // Verify logsumexp values at position 0 are finite
        for (size_t b_idx = 0; b_idx < B; ++b_idx) {
            for (size_t h_idx = 0; h_idx < num_heads; ++h_idx) {
                for (size_t s_idx = 0; s_idx < S; ++s_idx) {
                    float lse = interm_xtensor(b_idx, h_idx, s_idx, 0);
                    EXPECT_TRUE(std::isfinite(lse)) << "logsumexp should be finite";
                }
            }
        }
    }
}

// =============================================================================
// DIFFERENT INNER DIM TESTS - V head dim differs from Q/K head dim
// =============================================================================

TEST_F(SDPAForwardTest, SDPAForwardTest_DifferentVDim_CausalMask_SmallV) {
    // Q/K head_dim=64, V head_dim=32 (V smaller than Q/K)
    SDPATestConfig config{
        .batch_size = 1U,
        .sequence_length = 128U,
        .query_dim = 128U,      // 2 heads * 64 dim per head
        .key_value_dim = 128U,  // 2 heads * 64 dim per head (K)
        .value_dim = 64U,       // 2 heads * 32 dim per head (V)
        .num_query_heads = 2U,
        .num_key_heads = 2U,
        .mask_type = ttml::metal::AttentionMaskType::Causal,
        .test_name = "DifferentVDim_CausalMask_SmallV"};
    run_sdpa_test(config);
}

TEST_F(SDPAForwardTest, SDPAForwardTest_DifferentVDim_CausalMask_LargeV) {
    // Q/K head_dim=32, V head_dim=64 (V larger than Q/K)
    SDPATestConfig config{
        .batch_size = 1U,
        .sequence_length = 128U,
        .query_dim = 64U,      // 2 heads * 32 dim per head
        .key_value_dim = 64U,  // 2 heads * 32 dim per head (K)
        .value_dim = 128U,     // 2 heads * 64 dim per head (V)
        .num_query_heads = 2U,
        .num_key_heads = 2U,
        .mask_type = ttml::metal::AttentionMaskType::Causal,
        .test_name = "DifferentVDim_CausalMask_LargeV"};
    run_sdpa_test(config);
}

TEST_F(SDPAForwardTest, SDPAForwardTest_DifferentVDim_ArbitraryMask) {
    // Q/K head_dim=64, V head_dim=32 with arbitrary mask
    SDPATestConfig config{
        .batch_size = 1U,
        .sequence_length = 128U,
        .query_dim = 128U,      // 2 heads * 64 dim per head
        .key_value_dim = 128U,  // 2 heads * 64 dim per head (K)
        .value_dim = 64U,       // 2 heads * 32 dim per head (V)
        .num_query_heads = 2U,
        .num_key_heads = 2U,
        .mask_type = ttml::metal::AttentionMaskType::Arbitrary,
        .test_name = "DifferentVDim_ArbitraryMask"};
    run_sdpa_test(config);
}

TEST_F(SDPAForwardTest, SDPAForwardTest_DifferentVDim_GQA) {
    // GQA with different V dim: Q has 4 heads, KV has 2 heads
    // Q/K head_dim=64, V head_dim=32
    SDPATestConfig config{
        .batch_size = 1U,
        .sequence_length = 128U,
        .query_dim = 256U,      // 4 heads * 64 dim per head
        .key_value_dim = 128U,  // 2 heads * 64 dim per head (K)
        .value_dim = 64U,       // 2 heads * 32 dim per head (V)
        .num_query_heads = 4U,
        .num_key_heads = 2U,
        .mask_type = ttml::metal::AttentionMaskType::Causal,
        .test_name = "DifferentVDim_GQA_4Q_2KV"};
    run_sdpa_test(config);
}

TEST_F(SDPAForwardTest, SDPAForwardTest_DifferentVDim_SingleTile) {
    // Minimum viable: Q/K head_dim=64, V head_dim=32, single tile seq
    SDPATestConfig config{
        .batch_size = 1U,
        .sequence_length = 32U,
        .query_dim = 64U,      // 1 head * 64 dim
        .key_value_dim = 64U,  // 1 head * 64 dim (K)
        .value_dim = 32U,      // 1 head * 32 dim (V)
        .num_query_heads = 1U,
        .num_key_heads = 1U,
        .mask_type = ttml::metal::AttentionMaskType::Causal,
        .test_name = "DifferentVDim_SingleTile"};
    run_sdpa_test(config);
}

// Was disabled for non-deterministic accuracy failures on the old softmax path
// (https://github.com/tenstorrent/tt-metal/issues/46121); passes on the chunked kernel.
TEST_F(SDPAForwardTest, SDPAForwardTest_DifferentVDim_MultiBatch) {
    // Multi-batch with different V dim
    SDPATestConfig config{
        .batch_size = 4U,
        .sequence_length = 128U,
        .query_dim = 128U,      // 2 heads * 64 dim per head
        .key_value_dim = 128U,  // 2 heads * 64 dim per head (K)
        .value_dim = 64U,       // 2 heads * 32 dim per head (V)
        .num_query_heads = 2U,
        .num_key_heads = 2U,
        .mask_type = ttml::metal::AttentionMaskType::Causal,
        .test_name = "DifferentVDim_MultiBatch"};
    run_sdpa_test(config);
}

TEST_F(SDPAForwardTest, SDPAForwardTest_CausalMask_TwoTileRows) {
    // 64 seq len = 2 tile rows -> Ht=2 -> Sk_chunk_t=2. Forces the chunked inner loop with
    // exactly one chunk per row and exercises the diagonal-chunk masking path for the
    // smallest non-trivial chunk size.
    SDPATestConfig config{
        .batch_size = 1U,
        .sequence_length = 64U,
        .query_dim = 64U,
        .key_value_dim = 64U,
        .num_query_heads = 1U,
        .num_key_heads = 1U,
        .mask_type = ttml::metal::AttentionMaskType::Causal,
        .test_name = "CausalMask_TwoTileRows"};
    run_sdpa_test(config);
}

TEST_F(SDPAForwardTest, SDPAForwardTest_NoMask_Small) {
    // mask_type=None: compute skips mask application entirely, so the reader must not
    // push mask tiles into a CB nothing consumes. Always-on guard for the no-mask path.
    SDPATestConfig config{
        .batch_size = 1U,
        .sequence_length = 128U,
        .query_dim = 128U,
        .key_value_dim = 128U,
        .num_query_heads = 2U,
        .num_key_heads = 2U,
        .mask_type = ttml::metal::AttentionMaskType::None,
        .test_name = "NoMask_Small_2H"};
    run_sdpa_test(config);
}

TEST_F(SDPAForwardTest, SDPAForwardTest_NoMask_TwoTileRows) {
    // 64 seq len -> Ht=2 -> Sk_chunk_t=2 on the no-mask compute path: chunked inner loop
    // with no mask CB traffic.
    SDPATestConfig config{
        .batch_size = 1U,
        .sequence_length = 64U,
        .query_dim = 64U,
        .key_value_dim = 64U,
        .num_query_heads = 1U,
        .num_key_heads = 1U,
        .mask_type = ttml::metal::AttentionMaskType::None,
        .test_name = "NoMask_TwoTileRows"};
    run_sdpa_test(config);
}

TEST_F(SDPAForwardTest, SDPAForwardTest_ArbitraryMask_TwoTileRows) {
    // 64 seq len -> Ht=2 -> Sk_chunk_t=2 on the arbitrary-mask compute path (USE_ATTN_MASK).
    SDPATestConfig config{
        .batch_size = 1U,
        .sequence_length = 64U,
        .query_dim = 64U,
        .key_value_dim = 64U,
        .num_query_heads = 1U,
        .num_key_heads = 1U,
        .mask_type = ttml::metal::AttentionMaskType::Arbitrary,
        .test_name = "ArbitraryMask_TwoTileRows"};
    run_sdpa_test(config);
}

// =============================================================================
// GATE TESTS - optional gate input: out = sdpa(Q, K, V) * sigmoid(gate)
// Gate shape is (B, num_query_heads, S, head_dim_v), i.e. identical to the output shape.
//
// Three things in the gate epilogue can break, and the table below covers each of them once:
//   * the finalize block loop: Dh_v 96 / 128 / 160 give vWt 3 / 4 / 5 -> block_size 3 / 2 / 1;
//   * the reader's row index: batch, head, GQA head->group mapping, and non-consecutive items on a core;
//   * the gate_args offset after mask_args, with (Arbitrary) and without (Causal / None) a DRAM mask.
// run_sdpa_test with use_gate=true additionally checks, against the kernel's own ungated output, that
// gated == ungated * sigmoid(gate) to ~2 bf16 ULP and that the logsumexp intermediates are bit-identical.
// The rows are NIGHTLY_ because each one compiles its own kernels. In Merge Gate, GateTest_IsElementwiseNotBroadcast
// covers the multi-block finalize loop and the batch / head / row indexing. Both Merge Gate gate tests are causal,
// so the gate_args offset after a DRAM mask is only checked nightly.
// =============================================================================

class SDPAForwardGateTest : public ::testing::TestWithParam<SDPATestConfig> {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
        ttml::autograd::ctx().set_seed(42);
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

TEST_P(SDPAForwardGateTest, MatchesUngatedTimesSigmoid) {
    run_sdpa_test(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    GateShapes,
    SDPAForwardGateTest,
    ::testing::Values(
        // Dh_v = 96 -> vWt = 3 -> block_size = 3: three output tiles plus the shared scratch slot fill DST.
        SDPATestConfig{
            .batch_size = 1U,
            .sequence_length = 128U,
            .query_dim = 128U,
            .key_value_dim = 128U,
            .value_dim = 192U,  // 2 heads * 96
            .num_query_heads = 2U,
            .num_key_heads = 2U,
            .mask_type = ttml::metal::AttentionMaskType::Causal,
            .use_gate = true,
            .test_name = "NIGHTLY_Dv96_Causal"},
        // Dh_v = 128 -> vWt = 4 -> block_size = 2: copy_tile must offset by tile_idx into the second block.
        // Two batches and a DRAM mask: the gate address follows the mask runtime args, and batch 1 must read
        // gate[1], not gate[0].
        SDPATestConfig{
            .batch_size = 2U,
            .sequence_length = 64U,
            .query_dim = 128U,
            .key_value_dim = 128U,
            .value_dim = 256U,  // 2 heads * 128
            .num_query_heads = 2U,
            .num_key_heads = 2U,
            .mask_type = ttml::metal::AttentionMaskType::Arbitrary,
            .use_gate = true,
            .test_name = "NIGHTLY_Dv128_Arbitrary_2B"},
        // Dh_v = 160 -> vWt = 5 -> block_size = 1: five single-tile blocks, no mask CB traffic at all.
        SDPATestConfig{
            .batch_size = 1U,
            .sequence_length = 128U,
            .query_dim = 128U,
            .key_value_dim = 128U,
            .value_dim = 320U,  // 2 heads * 160
            .num_query_heads = 2U,
            .num_key_heads = 2U,
            .mask_type = ttml::metal::AttentionMaskType::None,
            .use_gate = true,
            .test_name = "NIGHTLY_Dv160_NoMask"},
        // GQA: 4 query heads share 2 KV heads. The gate has num_query_heads heads and is indexed by query head.
        SDPATestConfig{
            .batch_size = 1U,
            .sequence_length = 128U,
            .query_dim = 256U,      // 4 heads * 64
            .key_value_dim = 128U,  // 2 heads * 64
            .num_query_heads = 4U,
            .num_key_heads = 2U,
            .mask_type = ttml::metal::AttentionMaskType::Causal,
            .use_gate = true,
            .test_name = "NIGHTLY_GQA_4Q_2KV"},
        // Many causal items (B=4, H=12, S=256): every core gets several items of different cost and the gate
        // reads have to follow that item order.
        SDPATestConfig{
            .batch_size = 4U,
            .sequence_length = 256U,
            .query_dim = 768U,  // 12 heads * 64
            .key_value_dim = 768U,
            .num_query_heads = 12U,
            .num_key_heads = 12U,
            .mask_type = ttml::metal::AttentionMaskType::Causal,
            .use_gate = true,
            .test_name = "NIGHTLY_BalancedParallelism_4B_12H_256S"}),
    [](const ::testing::TestParamInfo<SDPATestConfig>& info) { return info.param.test_name; });

// --- Structured gates with analytically known results ------------------------

namespace {

struct GateFixtureTensors {
    xt::xarray<float> query;
    xt::xarray<float> key;
    xt::xarray<float> value;
    std::array<std::size_t, 4> out_shape;  // (B, H, S, Dh_v)
};

// Small MHA problem used by the structured-gate tests below.
GateFixtureTensors make_gate_fixture(
    uint32_t B = 1U, uint32_t H = 2U, uint32_t S = 64U, uint32_t Dh_qk = 64U, uint32_t Dh_v = 64U) {
    auto& rng = ttml::autograd::ctx().get_generator();
    const uint32_t seed = rng();
    const std::array<std::size_t, 4> qk_shape{B, H, S, Dh_qk};
    const std::array<std::size_t, 4> v_shape{B, H, S, Dh_v};
    GateFixtureTensors t;
    t.query = ttml::test_utils::make_uniform_xarray<float>(qk_shape, -1.0F, 1.0F, seed);
    t.key = ttml::test_utils::make_uniform_xarray<float>(qk_shape, -1.0F, 1.0F, seed + 1U);
    t.value = ttml::test_utils::make_uniform_xarray<float>(v_shape, -1.0F, 1.0F, seed + 2U);
    t.out_shape = v_shape;
    return t;
}

// Runs the kernel (causal, intermediates on) with an optional gate and returns {output, intermediates} on host.
std::pair<xt::xarray<float>, xt::xarray<float>> run_kernel_causal(
    const GateFixtureTensors& t, const std::optional<xt::xarray<float>>& gate) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    auto q = core::from_xtensor(t.query, device);
    auto k = core::from_xtensor(t.key, device);
    auto v = core::from_xtensor(t.value, device);
    std::optional<ttnn::Tensor> g = std::nullopt;
    if (gate.has_value()) {
        g = core::from_xtensor(gate.value(), device);
    }
    auto res = metal::sdpa_fw(q, k, v, metal::AttentionMaskType::Causal, std::nullopt, 0.0F, true, g);
    return {core::to_xtensor(res[0].value()), core::to_xtensor(res[1].value())};
}

}  // namespace

TEST_F(SDPAForwardTest, GateTest_IsElementwiseNotBroadcast) {
    // Each gate element is independently +16 (open) or -16 (closed), so a gate tile read from the wrong batch,
    // head, row or column block, or broadcast over any axis, zeroes some open elements or lets some closed ones
    // through. Dh_v = 192 gives vWt = 6 -> block_size = 3: two finalize blocks that each fill DST, so the gate
    // copy has to offset by tile_idx into the second block and that block has to be re-initialised after the
    // first block's gate code. B is sized from the grid so there are more items than cores: some cores get two
    // items, so the per-item gate read and cb_pop_front run more than once per core.
    const uint32_t H = 2U, S = 64U, Dh_qk = 64U, Dh_v = 192U;
    const uint32_t St = S / 32U;
    const auto grid = ttml::autograd::ctx().get_device().compute_with_storage_grid_size();
    const uint32_t B = static_cast<uint32_t>(grid.x * grid.y) / (H * St) + 1U;
    auto t = make_gate_fixture(B, H, S, Dh_qk, Dh_v);
    auto [ungated, ungated_lse] = run_kernel_causal(t, std::nullopt);

    auto& rng = ttml::autograd::ctx().get_generator();
    const xt::xarray<float> coin = ttml::test_utils::make_uniform_xarray<float>(t.out_shape, -1.0F, 1.0F, rng());
    const xt::xarray<bool> open = coin >= 0.0F;
    const xt::xarray<float> gate = xt::where(open, 16.0F, -16.0F);
    // sigmoid(+16) is 1 - 1.1e-7 and sigmoid(-16) is ~1.1e-7, and the kernel keeps both in fp32 DST. The open
    // elements are within rtol of O and the closed ones are |O| * 1.1e-7, far below atol, so the gated output is
    // the ungated output with the closed elements zeroed.
    const xt::xarray<float> expected = xt::where(open, ungated, 0.0F);

    auto [gated, gated_lse] = run_kernel_causal(t, gate);
    ASSERT_EQ(gated.shape(), expected.shape());
    // Same ~2 bf16 ULP bound as the gated == ungated * sigmoid(gate) check in run_sdpa_test.
    EXPECT_TRUE(xt::allclose(gated, expected, 8e-3F, 1e-4F))
        << "Random +/-16 gate mismatch: the gate is not being applied elementwise with the right indexing (MSE: "
        << compute_mse(expected, gated) << ")";
    // The logsumexp is packed before the gate path runs, so it must be bit-identical.
    EXPECT_TRUE(xt::all(xt::equal(gated_lse, ungated_lse))) << "Gate changed the intermediates (logsumexp)";
}

TEST_F(SDPAForwardTest, NIGHTLY_GateTest_WorksWithoutIntermediates) {
    // Gate path with return_intermediates = false must still produce the gated output and no intermediates.
    using namespace ttml;
    auto t = make_gate_fixture();
    auto* device = &autograd::ctx().get_device();
    auto q = core::from_xtensor(t.query, device);
    auto k = core::from_xtensor(t.key, device);
    auto v = core::from_xtensor(t.value, device);

    xt::xarray<float> gate_host = ttml::test_utils::make_uniform_xarray<float>(t.out_shape, -4.0F, 4.0F, 1234U);
    auto gate = core::from_xtensor(gate_host, device);

    auto ungated = metal::sdpa_fw(q, k, v, metal::AttentionMaskType::Causal, std::nullopt, 0.0F, false, std::nullopt);
    auto gated = metal::sdpa_fw(q, k, v, metal::AttentionMaskType::Causal, std::nullopt, 0.0F, false, gate);

    ASSERT_TRUE(gated[0].has_value());
    EXPECT_FALSE(gated[1].has_value()) << "Intermediates must be absent when return_intermediates=false";

    xt::xarray<float> gated_host = core::to_xtensor(gated[0].value());
    xt::xarray<float> expected = apply_gate_reference(core::to_xtensor(ungated[0].value()), gate_host);
    std::vector<size_t> expected_shape(t.out_shape.begin(), t.out_shape.end());
    EXPECT_EQ(gated_host.shape(), expected_shape) << "Gated output must keep the (B, H, S, Dh_v) shape";
    EXPECT_TRUE(xt::allclose(gated_host, expected, 2e-2F, 2e-2F))
        << "Gated output mismatch without intermediates (MSE: " << compute_mse(expected, gated_host) << ")";
}

TEST_F(SDPAForwardTest, GateTest_RejectsWrongShapes) {
    // The op validates the gate as (B, num_query_heads, S, head_dim_v). Each case below breaks exactly one dim.
    using namespace ttml;
    const uint32_t B = 1U, Hq = 4U, Hkv = 2U, S = 64U, Dh_qk = 64U, Dh_v = 32U;
    auto* device = &autograd::ctx().get_device();
    const std::array<std::size_t, 4> q_shape{B, Hq, S, Dh_qk};
    const std::array<std::size_t, 4> k_shape{B, Hkv, S, Dh_qk};
    const std::array<std::size_t, 4> v_shape{B, Hkv, S, Dh_v};
    auto q = core::from_xtensor(ttml::test_utils::make_uniform_xarray<float>(q_shape, -1.0F, 1.0F, 1U), device);
    auto k = core::from_xtensor(ttml::test_utils::make_uniform_xarray<float>(k_shape, -1.0F, 1.0F, 2U), device);
    auto v = core::from_xtensor(ttml::test_utils::make_uniform_xarray<float>(v_shape, -1.0F, 1.0F, 3U), device);

    auto make_gate = [&](std::array<std::size_t, 4> shape) {
        return core::from_xtensor(ttml::test_utils::make_uniform_xarray<float>(shape, -1.0F, 1.0F, 4U), device);
    };
    auto run = [&](const ttnn::Tensor& gate) {
        return metal::sdpa_fw(q, k, v, metal::AttentionMaskType::Causal, std::nullopt, 0.0F, false, gate);
    };

    // Correct shape must be accepted.
    EXPECT_NO_THROW(run(make_gate({B, Hq, S, Dh_v}))) << "Correct gate shape (B, Hq, S, Dh_v) was rejected";

    // Wrong head count: KV heads instead of query heads.
    EXPECT_ANY_THROW(run(make_gate({B, Hkv, S, Dh_v}))) << "Gate with num_key_heads heads should be rejected";
    // Wrong last dim: Q/K head dim instead of V head dim.
    EXPECT_ANY_THROW(run(make_gate({B, Hq, S, Dh_qk}))) << "Gate with Q/K head dim should be rejected";
    // Wrong sequence length.
    EXPECT_ANY_THROW(run(make_gate({B, Hq, S / 2U, Dh_v}))) << "Gate with wrong S should be rejected";
    // Wrong batch.
    EXPECT_ANY_THROW(run(make_gate({B + 1U, Hq, S, Dh_v}))) << "Gate with wrong B should be rejected";
    // Fused-heads layout (B, 1, S, Hq*Dh_v) is not what the op expects.
    EXPECT_ANY_THROW(run(make_gate({B, 1U, S, Hq * Dh_v}))) << "Gate in fused-heads layout should be rejected";
}

TEST_F(SDPAForwardTest, GateTest_ProgramCacheHitReadsNewGate) {
    // override_runtime_arguments patches the gate buffer address on a program-cache hit. Run the gated op twice
    // at identical shapes with two different gates alive at the same time, so the second gate cannot land at the
    // first gate's address: a stale address would make the second run return the first run's output.
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    device->enable_program_cache();

    auto t = make_gate_fixture();
    auto q = core::from_xtensor(t.query, device);
    auto k = core::from_xtensor(t.key, device);
    auto v = core::from_xtensor(t.value, device);

    const xt::xarray<float> gate_a_host =
        round_to_bf16(ttml::test_utils::make_uniform_xarray<float>(t.out_shape, -4.0F, 4.0F, 4321U));
    // sigmoid(-g) == 1 - sigmoid(g), so the two gates differ everywhere g != 0.
    const xt::xarray<float> gate_b_host = -gate_a_host;
    auto gate_a = core::from_xtensor(gate_a_host, device);
    auto gate_b = core::from_xtensor(gate_b_host, device);
    ASSERT_NE(gate_a.buffer()->address(), gate_b.buffer()->address())
        << "Both gates must be alive at distinct device addresses for this test to mean anything";

    auto ungated = metal::sdpa_fw(q, k, v, metal::AttentionMaskType::Causal, std::nullopt, 0.0F, true);
    const xt::xarray<float> ungated_host = core::to_xtensor(ungated[0].value());

    const auto entries_before = device->num_program_cache_entries();
    auto out_a = metal::sdpa_fw(q, k, v, metal::AttentionMaskType::Causal, std::nullopt, 0.0F, true, gate_a);
    const auto entries_after_first = device->num_program_cache_entries();
    // Guard against a vacuous test: the first gated run must actually populate the cache.
    ASSERT_GT(entries_after_first, entries_before) << "program cache not populated by the first gated run";

    auto out_b = metal::sdpa_fw(q, k, v, metal::AttentionMaskType::Causal, std::nullopt, 0.0F, true, gate_b);
    EXPECT_EQ(device->num_program_cache_entries(), entries_after_first)
        << "second gated run at identical shapes compiled a new program instead of hitting the cache";

    const xt::xarray<float> out_a_host = core::to_xtensor(out_a[0].value());
    const xt::xarray<float> out_b_host = core::to_xtensor(out_b[0].value());
    const xt::xarray<float> expected_b = apply_gate_reference(ungated_host, gate_b_host);
    EXPECT_TRUE(xt::allclose(out_b_host, expected_b, 8e-3F, 1e-4F))
        << "Cache-hit run did not apply the second gate (MSE vs expected: " << compute_mse(expected_b, out_b_host)
        << ")";
    EXPECT_FALSE(xt::allclose(out_b_host, out_a_host, 8e-3F, 1e-4F))
        << "Cache-hit run reproduced the first gate's output: stale gate address";
}

// =============================================================================
// q_chunk_size / k_chunk_size knobs: same numbers, different chunking.
// =============================================================================

struct ChunkKnobs {
    uint32_t q_chunk_size;
    uint32_t k_chunk_size;
    uint32_t heads_per_item = 0U;
};

// One Q row with a wide K step, many Q rows with one K tile, square, and both rectangles.
const std::vector<ChunkKnobs> kChunkKnobSweep = {{32, 256}, {128, 32}, {64, 64}, {128, 64}, {64, 256}};

void run_chunk_knob_sweep(SDPATestConfig base, const std::vector<ChunkKnobs>& sweep = kChunkKnobSweep) {
    for (const auto& knobs : sweep) {
        SDPATestConfig config = base;
        config.program_config = {
            .q_chunk_size = knobs.q_chunk_size,
// Every chunking computes the same fp32 math in a different order and rounds once to bf16, so besides the
// reference checks each row is held against the default pick: a mean absolute gap of 5e-4 is a few
// bf16 ULP at the ~0.03-0.1 output magnitudes here, where the 2e-2 reference tolerance would let a
// uniform few-percent rescale error through.
constexpr float kCrossChunkingMeanAbsTol = 5e-4F;

            .k_chunk_size = knobs.k_chunk_size,
    if (!base.seed.has_value()) {
        base.seed = ttml::autograd::ctx().get_generator()();  // every row sees the same inputs
    }
    xt::xarray<float> default_output;
    {
        SDPATestConfig config = base;
        config.program_config = {};
        config.test_name = base.test_name + "_default";
        SCOPED_TRACE(config.test_name);
        run_sdpa_test(config, &default_output);
    }
            .heads_per_item = knobs.heads_per_item};
        config.test_name =
            fmt::format("{}_q{}_k{}_h{}", base.test_name, knobs.q_chunk_size, knobs.k_chunk_size, knobs.heads_per_item);
        SCOPED_TRACE(config.test_name);
        xt::xarray<float> output;
        run_sdpa_test(config, &output);
        const float mean_abs_gap = xt::mean(xt::abs(output - default_output))();
        EXPECT_LT(mean_abs_gap, kCrossChunkingMeanAbsTol)
            << config.test_name << " differs from the default chunking by " << mean_abs_gap << " on average";
    }
}

TEST_F(SDPAForwardTest, NIGHTLY_SDPAForwardTest_ChunkKnobs_Causal) {
    SKIP_FOR_LLK_ASSERTS("Skip due to too large code size when assert is enabled.");
    // Few items: most cores get one.
    run_chunk_knob_sweep(SDPATestConfig{
        .batch_size = 1U,
        .sequence_length = 256U,
        .query_dim = 128U,
        .key_value_dim = 128U,
        .num_query_heads = 2U,
        .num_key_heads = 2U,
        .mask_type = ttml::metal::AttentionMaskType::Causal,
        .test_name = "ChunkKnobs_Causal"});
}

TEST_F(SDPAForwardTest, NIGHTLY_SDPAForwardTest_ChunkKnobs_Causal_GQA_ManySeqs) {
    SKIP_FOR_LLK_ASSERTS("Skip due to too large code size when assert is enabled.");
    // 144 sequences, GQA: many items per core at every chunking.
    run_chunk_knob_sweep(SDPATestConfig{
        .batch_size = 9U,
        .sequence_length = 256U,
        .query_dim = 1024U,     // 16 heads * 64
        .key_value_dim = 256U,  // 4 kv heads * 64
        .num_query_heads = 16U,
        .num_key_heads = 4U,
        .mask_type = ttml::metal::AttentionMaskType::Causal,
        .test_name = "ChunkKnobs_Causal_GQA_ManySeqs"});
}

TEST_F(SDPAForwardTest, NIGHTLY_SDPAForwardTest_ChunkKnobs_Causal_NonPow2) {
    SKIP_FOR_LLK_ASSERTS("Skip due to too large code size when assert is enabled.");
    // S = 12 tiles: 3-tile chunks give odd sub-blocks and a causal step count that is not a power of two.
    run_chunk_knob_sweep(
        SDPATestConfig{
            .batch_size = 1U,
            .sequence_length = 384U,
            .query_dim = 128U,
            .key_value_dim = 128U,
            .num_query_heads = 2U,
            .num_key_heads = 2U,
            .mask_type = ttml::metal::AttentionMaskType::Causal,
            .test_name = "ChunkKnobs_Causal_NonPow2"},
        {{96, 96}, {32, 96}, {96, 32}, {128, 96}, {96, 384}});
}

TEST_F(SDPAForwardTest, NIGHTLY_SDPAForwardTest_ChunkKnobs_Causal_HeadStacking) {
    SKIP_FOR_LLK_ASSERTS("Skip due to too large code size when assert is enabled.");
    // GQA with 4 heads per group: stacks of 2 and 4 heads, with 1..4 Q rows each.
    run_chunk_knob_sweep(
        SDPATestConfig{
            .batch_size = 2U,
            .sequence_length = 256U,
            .query_dim = 512U,      // 8 heads * 64
            .key_value_dim = 128U,  // 2 kv heads * 64
            .num_query_heads = 8U,
            .num_key_heads = 2U,
            .mask_type = ttml::metal::AttentionMaskType::Causal,
            .test_name = "ChunkKnobs_Causal_HeadStacking"},
        {{32, 64, 4}, {128, 256, 2}, {64, 32, 4}, {128, 64, 4}, {0, 0, 2}});
}

TEST_F(SDPAForwardTest, SDPAForwardTest_ChunkKnobs_Arbitrary_HeadStacking) {
    SKIP_FOR_LLK_ASSERTS("Skip due to too large code size when assert is enabled.");
    // The (1, 1, S, S) mask is read once per step and shared by the stacked heads.
    run_chunk_knob_sweep(
        SDPATestConfig{
// One blocking row per compiled variant (causal, arbitrary, no mask + gate); the full sweeps are nightly.
TEST_F(SDPAForwardTest, SDPAForwardTest_ChunkKnobs_Smoke) {
    SKIP_FOR_LLK_ASSERTS("Skip due to too large code size when assert is enabled.");
    run_chunk_knob_sweep(
        SDPATestConfig{
            .batch_size = 1U,
            .sequence_length = 256U,
            .query_dim = 128U,
            .key_value_dim = 128U,
            .num_query_heads = 2U,
            .num_key_heads = 2U,
            .mask_type = ttml::metal::AttentionMaskType::Causal,
            .test_name = "ChunkKnobs_Smoke_Causal"},
        {{128, 256, 1}});
    run_chunk_knob_sweep(
        SDPATestConfig{
            .batch_size = 1U,
            .sequence_length = 256U,
            .query_dim = 128U,
            .key_value_dim = 64U,
            .num_query_heads = 2U,
            .num_key_heads = 1U,
            .mask_type = ttml::metal::AttentionMaskType::Arbitrary,
            .test_name = "ChunkKnobs_Smoke_Arbitrary"},
        {{64, 64, 2}});
    run_chunk_knob_sweep(
        SDPATestConfig{
            .batch_size = 1U,
            .sequence_length = 256U,
            .query_dim = 256U,
            .key_value_dim = 256U,
            .value_dim = 128U,
            .num_query_heads = 2U,
            .num_key_heads = 2U,
            .mask_type = ttml::metal::AttentionMaskType::None,
            .use_gate = true,
            .test_name = "ChunkKnobs_Smoke_None_Gate"},
        {{64, 128, 1}});
}

TEST_F(SDPAForwardTest, SDPAForwardTest_ChunkKnobs_NoIntermediates) {
    SKIP_FOR_LLK_ASSERTS("Skip due to too large code size when assert is enabled.");
    // The RETURN_INTERMEDIATES-less compile with explicit chunking, checked against the references.
    run_chunk_knob_sweep(
        SDPATestConfig{
            .batch_size = 1U,
            .sequence_length = 256U,
            .query_dim = 128U,
            .key_value_dim = 128U,
            .num_query_heads = 2U,
            .num_key_heads = 2U,
            .mask_type = ttml::metal::AttentionMaskType::Causal,
            .return_intermediates = false,
            .test_name = "ChunkKnobs_NoIntermediates"},
        {{128, 256, 1}, {64, 64, 1}});
}

TEST_F(SDPAForwardTest, NIGHTLY_SDPAForwardTest_ChunkKnobs_Arbitrary_NonPow2) {
    SKIP_FOR_LLK_ASSERTS("Skip due to too large code size when assert is enabled.");
    // 3 x 2 mask tiles per step: the mask stamp's last DST pass is a 2-tile tail.
    run_chunk_knob_sweep(
        SDPATestConfig{
            .batch_size = 1U,
            .sequence_length = 384U,
            .query_dim = 128U,
            .key_value_dim = 128U,
            .num_query_heads = 2U,
            .num_key_heads = 2U,
            .mask_type = ttml::metal::AttentionMaskType::Arbitrary,
            .test_name = "ChunkKnobs_Arbitrary_NonPow2"},
        {{96, 64, 1}, {96, 96, 1}});
}

            .batch_size = 1U,
            .sequence_length = 256U,
            .query_dim = 256U,     // 4 heads * 64
            .key_value_dim = 64U,  // 1 kv head * 64
            .num_query_heads = 4U,
            .num_key_heads = 1U,
            .mask_type = ttml::metal::AttentionMaskType::Arbitrary,
            .test_name = "ChunkKnobs_Arbitrary_HeadStacking"},
        {{64, 64, 2}, {32, 128, 4}});
}

TEST_F(SDPAForwardTest, NIGHTLY_SDPAForwardTest_ChunkKnobs_None_Gate_HeadStacking) {
    SKIP_FOR_LLK_ASSERTS("Skip due to too large code size when assert is enabled.");
    // No mask, vWt < qWt, gate: the gate rows are stacked like the Q rows.
    run_chunk_knob_sweep(
        SDPATestConfig{
            .batch_size = 1U,
            .sequence_length = 256U,
            .query_dim = 512U,      // 4 heads * 128
            .key_value_dim = 256U,  // 2 kv heads * 128
            .value_dim = 128U,      // 2 kv heads * 64
            .num_query_heads = 4U,
            .num_key_heads = 2U,
            .mask_type = ttml::metal::AttentionMaskType::None,
            .use_gate = true,
            .test_name = "ChunkKnobs_None_Gate_HeadStacking"},
        {{64, 128, 2}, {128, 64, 2}});
}

TEST_F(SDPAForwardTest, NIGHTLY_SDPAForwardTest_ChunkKnobs_Arbitrary) {
    SKIP_FOR_LLK_ASSERTS("Skip due to too large code size when assert is enabled.");
    run_chunk_knob_sweep(SDPATestConfig{
        .batch_size = 1U,
        .sequence_length = 256U,
        .query_dim = 128U,
        .key_value_dim = 128U,
        .num_query_heads = 2U,
        .num_key_heads = 2U,
        .mask_type = ttml::metal::AttentionMaskType::Arbitrary,
        .test_name = "ChunkKnobs_Arbitrary"});
}

TEST_F(SDPAForwardTest, NIGHTLY_SDPAForwardTest_ChunkKnobs_None_DiffVDim_Gate) {
    SKIP_FOR_LLK_ASSERTS("Skip due to too large code size when assert is enabled.");
    // No mask, vWt < qWt, gate: non-square PV geometry.
    run_chunk_knob_sweep(SDPATestConfig{
        .batch_size = 1U,
        .sequence_length = 256U,
        .query_dim = 256U,
        .key_value_dim = 256U,
        .value_dim = 128U,
        .num_query_heads = 2U,
        .num_key_heads = 2U,
        .mask_type = ttml::metal::AttentionMaskType::None,
        .use_gate = true,
        .test_name = "ChunkKnobs_None_DiffVDim_Gate"});
}

TEST_F(SDPAForwardTest, SDPAForwardTest_ChunkKnobs_DefaultPick) {
    // With the knobs at 0 the cost model picks the chunking: larger than main's one Q row / four K tiles
    // where L1 allows, and always something that fits.
    using namespace ttml::metal::ops::sdpa_fw::device;
    constexpr uint64_t kBudget = 1'300'000U;  // a Blackhole core's usable L1
    constexpr uint32_t kCores = 120U;
    const SDPAForwardShape tinyllama{
        .batch = 1U,
        .q_heads = 32U,
        .kv_heads = 4U,
        .St = 64U,
        .qWt = 2U,
        .vWt = 2U,
        .mask_type = ttml::metal::AttentionMaskType::Causal,
        .has_gate = false,
        .return_intermediates = true};
    const auto picked = resolve_chunks({}, tinyllama, kBudget, kCores);
    EXPECT_GE(picked.rows(), 4U) << "Sq=" << picked.Sq_chunk_t << " h=" << picked.heads_per_item;
    EXPECT_GE(picked.Sk_chunk_t, 4U);
    EXPECT_LE(l1_bytes(cb_tiles(tinyllama, picked)), kBudget);

    SDPAForwardShape wide = tinyllama;  // head dim 512: the biggest chunks do not fit, the pick must degrade
    wide.qWt = 16U;
    wide.vWt = 16U;
    const auto picked_wide = resolve_chunks({}, wide, kBudget, kCores);
    EXPECT_LE(l1_bytes(cb_tiles(wide, picked_wide)), kBudget);

    // A user knob pins its axis; the others are still picked.
    const auto pinned = resolve_chunks({.k_chunk_size = 32U}, tinyllama, kBudget, kCores);
    EXPECT_EQ(pinned.Sk_chunk_t, 1U);
    EXPECT_GE(pinned.rows(), 4U);
}

TEST_F(SDPAForwardTest, SDPAForwardTest_ChunkKnobs_DefaultFitsLargeHeadDim) {
    SKIP_FOR_LLK_ASSERTS("Skip due to too large code size when assert is enabled.");
    run_sdpa_test(SDPATestConfig{
        .batch_size = 1U,
        .sequence_length = 256U,
        .query_dim = 1024U,  // 2 heads * 512
        .key_value_dim = 1024U,
        .num_query_heads = 2U,
        .num_key_heads = 2U,
        .mask_type = ttml::metal::AttentionMaskType::Causal,
        .test_name = "ChunkKnobs_DefaultFitsLargeHeadDim"});
}

TEST_F(SDPAForwardTest, SDPAForwardTest_ChunkKnobs_Rejected) {
    using namespace ttml;
    const uint32_t S = 256U;
    const std::array<std::size_t, 4> shape{1U, 2U, S, 64U};
    auto make = [&](uint32_t seed) {
        return core::from_xtensor(
            test_utils::make_uniform_xarray<float>(shape, -1.0F, 1.0F, seed), &autograd::ctx().get_device());
    };
    auto query = make(1U);
    auto key = make(2U);
    auto value = make(3U);
    auto run = [&](uint32_t q_chunk_size, uint32_t k_chunk_size, uint32_t heads_per_item = 0U) {
        return metal::sdpa_fw(
            query,
            key,
            value,
            metal::AttentionMaskType::Causal,
            std::nullopt,
            0.0F,
            false,
            std::nullopt,
            {.q_chunk_size = q_chunk_size, .k_chunk_size = k_chunk_size, .heads_per_item = heads_per_item});
    };
    EXPECT_NO_THROW(run(0U, 0U));
    EXPECT_NO_THROW(run(S, S));
    EXPECT_THAT([&] { run(48U, 0U); }, ThrowsMessage<std::exception>(HasSubstr("q_chunk_size (48)")));
    EXPECT_THAT([&] { run(0U, 48U); }, ThrowsMessage<std::exception>(HasSubstr("k_chunk_size (48)")));
    EXPECT_THAT([&] { run(96U, 0U); }, ThrowsMessage<std::exception>(HasSubstr("q_chunk_size (96)")));
    EXPECT_THAT([&] { run(0U, 2U * S); }, ThrowsMessage<std::exception>(HasSubstr("k_chunk_size (512)")));
    EXPECT_THAT([&] { run(0U, 0U, 2U); }, ThrowsMessage<std::exception>(HasSubstr("heads_per_item (2)")));

    // Head dim 512 with whole-sequence chunks: the circular buffers do not fit L1.
    const std::array<std::size_t, 4> wide_shape{1U, 2U, S, 512U};
    auto make_wide = [&](uint32_t seed) {
        return core::from_xtensor(
            test_utils::make_uniform_xarray<float>(wide_shape, -1.0F, 1.0F, seed), &autograd::ctx().get_device());
    };
    auto wide_query = make_wide(4U);
    auto wide_key = make_wide(5U);
    auto wide_value = make_wide(6U);
    EXPECT_THAT(
        [&] {
            metal::sdpa_fw(
                wide_query,
                wide_key,
                wide_value,
                metal::AttentionMaskType::Causal,
                std::nullopt,
                0.0F,
                false,
                std::nullopt,
                {.q_chunk_size = S, .k_chunk_size = S});
        },
        ThrowsMessage<std::exception>(HasSubstr("SDPA forward circular buffers need")))
        << "a user config that overflows L1 must be refused with a message, not die in the allocator";
}

TEST_F(SDPAForwardTest, SDPAForwardTest_ChunkKnobs_ChangeProgramHash) {
TEST_F(SDPAForwardTest, SDPAForwardTest_ChunkKnobs_ProgramCacheMissAndHit) {
    // Two chunkings on the same tensors are two cached programs; the first one again is a hit.
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    device->enable_program_cache();
    const std::array<std::size_t, 4> shape{1U, 2U, 256U, 64U};
    auto make = [&](uint32_t seed) {
        return core::from_xtensor(test_utils::make_uniform_xarray<float>(shape, -1.0F, 1.0F, seed), device);
    };
    auto query = make(1U);
    auto key = make(2U);
    auto value = make(3U);
    auto run = [&](const metal::SDPAForwardProgramConfig& cfg) {
        return core::to_xtensor(
            metal::sdpa_fw(
                query, key, value, metal::AttentionMaskType::Causal, std::nullopt, 0.0F, false, std::nullopt, cfg)[0]
                .value());
    };
    const metal::SDPAForwardProgramConfig a{.q_chunk_size = 128U, .k_chunk_size = 256U};
    const metal::SDPAForwardProgramConfig b{.q_chunk_size = 32U, .k_chunk_size = 64U};
    const auto entries0 = device->num_program_cache_entries();
    const xt::xarray<float> out_a = run(a);
    const auto entries_a = device->num_program_cache_entries();
    ASSERT_GT(entries_a, entries0) << "first config did not populate the cache";
    const xt::xarray<float> out_b = run(b);
    EXPECT_EQ(device->num_program_cache_entries(), entries_a + 1U) << "second config must miss: different program";
    const xt::xarray<float> out_a_again = run(a);
    EXPECT_EQ(device->num_program_cache_entries(), entries_a + 1U) << "first config again must hit";
    EXPECT_TRUE(xt::all(xt::equal(out_a, out_a_again))) << "cache hit must reproduce the first run";
    EXPECT_LT(xt::mean(xt::abs(out_a - out_b))(), kCrossChunkingMeanAbsTol) << "both chunkings compute the same op";
}

    // The op hashes its attributes by reflection; the chunk knobs must take part, or a second config on
    // the same shapes would silently reuse the first config's cached program.
    using Params = ttml::metal::ops::sdpa_fw::device::SDPAForwardParams;
    const Params base{.mask_type = ttml::metal::AttentionMaskType::Causal};
    Params q_changed = base;
    q_changed.program_config.q_chunk_size = 128U;
    Params k_changed = base;
    k_changed.program_config.k_chunk_size = 128U;
    Params h_changed = base;
    h_changed.program_config.heads_per_item = 2U;
    EXPECT_NE(ttsl::hash::hash_objects_with_default_seed(base), ttsl::hash::hash_objects_with_default_seed(h_changed));
    EXPECT_NE(ttsl::hash::hash_objects_with_default_seed(base), ttsl::hash::hash_objects_with_default_seed(q_changed));
    EXPECT_NE(ttsl::hash::hash_objects_with_default_seed(base), ttsl::hash::hash_objects_with_default_seed(k_changed));
    EXPECT_NE(
        ttsl::hash::hash_objects_with_default_seed(q_changed), ttsl::hash::hash_objects_with_default_seed(k_changed));
}
