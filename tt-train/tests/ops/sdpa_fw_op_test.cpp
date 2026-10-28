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
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt_stl/reflection.hpp>
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/system_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/common/const_utils.hpp"
#include "metal/common/program_utils.hpp"
#include "metal/operations.hpp"
#include "metal/ops/sdpa_fw/device/sdpa_fw_config.hpp"
#include "metal/ops/sdpa_fw/device/sdpa_fw_device_operation_types.hpp"
#include "ops/scaled_dot_product_attention.hpp"
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
    // V's range. Zero-mean values make the late (long) rows average out to ~0, where an absolute
    // tolerance hides relative errors; the split tests use [0, 1] so every output is ~0.5.
    float value_min = -1.0F;
    float value_max = 1.0F;
    float dropout_prob = 0.0F;
    // 0 = program factory default.
    ttml::metal::SDPAForwardProgramConfig program_config{};
    bool return_intermediates = true;
    std::optional<uint32_t> seed;  // input RNG seed; unset = fresh from the context generator
    float result_atol = 2e-2F;
    float result_rtol = 2e-2F;
    float intermediate_atol = 2e-2F;
    float intermediate_rtol = 2e-2F;
    std::string test_name = "SDPA Test";
};

// kernel_output / kernel_lse, when given, receive the kernel's output and logsumexp so callers can compare
// runs with each other.
void run_sdpa_test(
    const SDPATestConfig& config,
    xt::xarray<float>* kernel_output = nullptr,
    xt::xarray<float>* kernel_lse = nullptr,
    xt::xarray<float>* reference_lse = nullptr) {
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

    // Each input gets its own seed (the same seed on equal shapes would make Q, K and V identical), and
    // is rounded to bf16 up front so the references see exactly what the device sees.
    xt::xarray<float> query_tensor =
        round_to_bf16(ttml::test_utils::make_uniform_xarray<float>(query_shape, -1.0F, 1.0F, seed));

    xt::xarray<float> key_tensor =
        round_to_bf16(ttml::test_utils::make_uniform_xarray<float>(kv_shape, -1.0F, 1.0F, seed + 1U));

    const std::array<std::size_t, 4> value_shape{
        config.batch_size, config.num_key_heads, config.sequence_length, head_dim_v};
    xt::xarray<float> value_tensor = round_to_bf16(
        ttml::test_utils::make_uniform_xarray<float>(value_shape, config.value_min, config.value_max, seed + 2U));

    // Optional gate: (B, num_query_heads, S, head_dim_v) - same shape as the kernel output.
    std::optional<xt::xarray<float>> gate_tensor;
    if (config.use_gate) {
        const std::array<std::size_t, 4> gate_shape{
            config.batch_size, config.num_query_heads, config.sequence_length, head_dim_v};
        gate_tensor = round_to_bf16(
            ttml::test_utils::make_uniform_xarray<float>(gate_shape, config.gate_min, config.gate_max, seed + 3U));
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
        if (kernel_lse != nullptr) {
            *kernel_lse = interm_xtensor;
        }
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
        const uint64_t l1_budget_bytes = ttml::metal::available_l1_bytes(*device);
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
            .heads_per_item = picked.heads_per_item,
            .max_pieces = config.program_config.max_pieces,  // same schedule too: the merge order is part of
            .num_cores = config.program_config.num_cores};   // the numerics the two runs must share
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

        ASSERT_EQ(result_xtensor.shape(), ungated_xtensor.shape())
            << "Gated and ungated kernel outputs must have the same shape in " << config.test_name;

        // (1) gated == ungated * sigmoid(gate)
        xt::xarray<float> expected_from_ungated = apply_gate_reference(ungated_xtensor, gate_tensor.value());
        EXPECT_TRUE(xt::allclose(result_xtensor, expected_from_ungated, kGateRtol, kGateAtol))
            << "Gated kernel output != ungated kernel output * sigmoid(gate) in " << config.test_name
            << " (MSE: " << compute_mse(expected_from_ungated, result_xtensor) << ")";

        // (2) the logsumexp is packed before the gate path runs, so it must be bit-identical.
        if (return_intermediates) {
            EXPECT_TRUE(xt::all(xt::equal(interm_xtensor, core::to_xtensor(ungated[1].value()))))
                << "Gate changed the intermediates (logsumexp) in " << config.test_name;
        }
    }

    // All results are now in split format (B, H, S, D) - heads NOT fused
    // Shape validation - all should be in split format (B, H, S, D)
    ASSERT_EQ(result_xtensor.shape(), float_result.shape()) << "Kernel result shape mismatch in " << config.test_name;
    ASSERT_EQ(composite_result_xtensor.shape(), float_result.shape())
        << "Composite result shape mismatch in " << config.test_name;
    if (return_intermediates) {
        ASSERT_EQ(interm_xtensor.shape(), float_intermediates.shape())
            << "Intermediate shape mismatch in " << config.test_name;
        // The logsumexp is ~log(S), so the relative tolerance below is blind to its absolute error: print it.
        fmt::print(
            "{}: logsumexp max abs gap vs float reference {:.3e}\n",
            config.test_name,
            xt::amax(xt::abs(interm_xtensor - float_intermediates))());
        if (reference_lse != nullptr) {
            *reference_lse = float_intermediates;
        }
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
    // 64 seq len = 2 tile rows: the smallest shape where the K/V loop can take more than one step and
    // the diagonal-chunk masking path runs.
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
    // 64 seq len = 2 tile rows on the no-mask compute path: no mask CB traffic.
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
    // 64 seq len = 2 tile rows on the arbitrary-mask compute path (USE_ATTN_MASK).
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
    uint32_t max_pieces = 0U;
    uint32_t num_cores = 0U;
};

// One Q row with a wide K step, many Q rows with one K tile, square, and both rectangles.
const std::vector<ChunkKnobs> kChunkKnobSweep = {{32, 256}, {128, 32}, {64, 64}, {128, 64}, {64, 256}};

// Every chunking computes the same fp32 math in a different order and rounds once to bf16, so besides the
// reference checks each row is held against the default pick: a mean absolute gap of 5e-4 is a few
// bf16 ULP at the ~0.03-0.1 output magnitudes here, where the 2e-2 reference tolerance would let a
// uniform few-percent rescale error through.
constexpr float kCrossChunkingMeanAbsTol = 5e-4F;

ttml::metal::SDPAForwardProgramConfig program_config_of(const ChunkKnobs& knobs) {
    return {
        .q_chunk_size = knobs.q_chunk_size,
        .k_chunk_size = knobs.k_chunk_size,
        .heads_per_item = knobs.heads_per_item,
        .max_pieces = knobs.max_pieces,
        .num_cores = knobs.num_cores};
}

// The shape and chunks the program factory would resolve for config on this device.
std::pair<ttml::metal::ops::sdpa_fw::device::SDPAForwardShape, ttml::metal::ops::sdpa_fw::device::SDPAForwardChunks>
host_shape_and_chunks(const SDPATestConfig& config) {
    using namespace ttml::metal::ops::sdpa_fw::device;
    auto* device = &ttml::autograd::ctx().get_device();
    const auto grid = device->compute_with_storage_grid_size();
    const uint32_t value_dim = config.value_dim > 0U ? config.value_dim : config.key_value_dim;
    const SDPAForwardShape shape{
        .batch = config.batch_size,
        .q_heads = config.num_query_heads,
        .kv_heads = config.num_key_heads,
        .St = config.sequence_length / 32U,
        .qWt = config.key_value_dim / config.num_key_heads / 32U,
        .vWt = value_dim / config.num_key_heads / 32U,
        .mask_type = config.mask_type,
        .has_gate = config.use_gate,
        .return_intermediates = config.return_intermediates};
    return {
        shape, resolve_chunks(config.program_config, shape, ttml::metal::available_l1_bytes(*device), grid.x * grid.y)};
}

// The schedule the program factory would build for config on this device.
ttml::metal::ops::sdpa_fw::device::SDPAForwardSchedule host_schedule(const SDPATestConfig& config) {
    using namespace ttml::metal::ops::sdpa_fw::device;
    auto* device = &ttml::autograd::ctx().get_device();
    const auto grid = device->compute_with_storage_grid_size();
    const auto [shape, chunks] = host_shape_and_chunks(config);
    const uint32_t max_pieces = effective_max_pieces(config.program_config);
    const uint32_t num_cores = std::min(
        effective_num_cores(config.program_config, grid.x * grid.y), num_work_items(shape, chunks) * max_pieces);
    return make_schedule(shape, chunks, num_cores, max_pieces);
}

// The terms of the split-vs-unsplit error bounds, from the split schedule: the most partials a merger folds
// in, and how many more K/V steps the unsplit run's longest item takes than the longest merger piece (each
// step of the running sum passes the FPU once, so the two runs' sums carry different truncation counts).
struct FoldErrorTerms {
    uint32_t partials = 0;
    uint32_t extra_unsplit_steps = 0;
};

FoldErrorTerms fold_error_terms(
    const SDPATestConfig& split_config, const ttml::metal::ops::sdpa_fw::device::SDPAForwardSchedule& split_schedule) {
    using namespace ttml::metal::ops::sdpa_fw::device;
    const auto [shape, chunks] = host_shape_and_chunks(split_config);
    uint32_t unsplit_steps = 0;
    for (uint32_t item = 0; item < num_work_items(shape, chunks); ++item) {
        unsplit_steps = std::max(unsplit_steps, num_kv_steps(shape, chunks, item));
    }
    FoldErrorTerms terms;
    uint32_t piece_steps = 0;
    for (const auto& units : split_schedule.per_core) {
        for (const auto& unit : units) {
            if (unit.role == UnitRole::Merger) {
                terms.partials = std::max(terms.partials, unit.num_partials);
                piece_steps = std::max(piece_steps, unit.step_end - unit.step_begin);
            }
        }
    }
    terms.extra_unsplit_steps = unsplit_steps - std::min(unsplit_steps, piece_steps);
    return terms;
}

// Split and unsplit outputs of the same inputs. Both runs multiply the running sum on the FPU, whose inputs
// are 19-bit (TF32): a truncation of up to 2^-10 per K/V step and, in the split run, per partial folded in;
// the merger's own fp32 output accumulator is rescaled in DST and is not truncated. So the two outputs
// differ by one to two bf16 ULP of rounding plus the steps the runs do not share, each worth the truncation
// and about as much again from the rescale chain (the weights of a peaked row reach the output through a
// different product of exps in the two runs; measured, not derived). The callers give V a nonzero mean so
// that no output sits near zero, where a bf16 ULP of the output would be smaller than the fp32 noise of
// the fold. The mean gap is held to the mean truncation, 2^-11 per unshared step.
void expect_split_matches_unsplit(
    const xt::xarray<float>& split_output,
    const xt::xarray<float>& unsplit_output,
    const FoldErrorTerms& terms,
    const std::string& name) {
    constexpr float kTwoUlpRel = 7.8125e-3F;   // 2^-7: two bf16 ULP in [0.25, 0.5), one in [0.5, 1)
    constexpr float kTruncRel = 9.765625e-4F;  // 2^-10: one TF32 truncation, worst case
    const float unshared_steps = static_cast<float>(terms.partials + terms.extra_unsplit_steps);
    const float rel_tol = kTwoUlpRel + 2.0F * kTruncRel * unshared_steps;
    const float mean_rel_tol = 1e-3F + 0.5F * kTruncRel * unshared_steps;
    const xt::xarray<float> gap = xt::abs(split_output - unsplit_output);
    const xt::xarray<float> bound = rel_tol * xt::abs(unsplit_output);
    const float worst_share = xt::amax(gap / bound)();  // of the bound; V > 0 keeps every output away from 0
    const float mean_rel = xt::mean(gap)() / xt::mean(xt::abs(unsplit_output))();
    fmt::print(
        "{}: {} partials, {} extra unsplit steps; split vs unsplit worst element at {:.2f} of the {:.2e} relative "
        "bound, mean relative gap {:.3e} (bound {:.1e})\n",
        name,
        terms.partials,
        terms.extra_unsplit_steps,
        worst_share,
        rel_tol,
        mean_rel,
        mean_rel_tol);
    EXPECT_LE(worst_share, 1.0F) << name << ": a split element is off the unsplit run by more than the bound";
    EXPECT_LT(mean_rel, mean_rel_tol) << name << ": split and unsplit runs differ by " << mean_rel << " on average";
}

// The merger's logsumexp, scale * max + log(sum), is what the backward pass consumes. The max is exact, so
// merging may move it off the float reference by at most the folds' truncation of the sum (two per partial:
// the sender's copy and the merge) on top of the unsplit run's own error.
void expect_split_lse_matches_reference(
    const xt::xarray<float>& split_lse,
    const xt::xarray<float>& unsplit_lse,
    const xt::xarray<float>& reference_lse,
    const FoldErrorTerms& terms,
    const std::string& name) {
    constexpr float kTruncAbs = 9.765625e-4F;  // 2^-10 on log(sum)
    const float split_gap = xt::amax(xt::abs(split_lse - reference_lse))();
    const float unsplit_gap = xt::amax(xt::abs(unsplit_lse - reference_lse))();
    const float bound = unsplit_gap + kTruncAbs * static_cast<float>(2U * terms.partials + 1U);
    fmt::print(
        "{}: logsumexp max gap vs reference: split {:.3e}, unsplit {:.3e} (bound {:.1e})\n",
        name,
        split_gap,
        unsplit_gap,
        bound);
    EXPECT_LE(split_gap, bound) << name << ": the merged logsumexp is further off the reference than the folds allow";
}

// A split schedule folds the partials into the merger's running state in fp32, in a different order than one
// core's inner loop, then rounds once to bf16 like the unsplit run. So the two outputs agree to a couple of
// bf16 ULP per element and far closer on average; a merger that dropped, double-counted or mis-scaled a
// partial would miss by a lot more. The 2e-2 reference tolerance would not see that on the small late-row
// outputs that get cut, which is why the sweep's reference checks are not enough.
void run_split_vs_unsplit(SDPATestConfig base, const ChunkKnobs& knobs) {
    if (!base.seed.has_value()) {
        base.seed = ttml::autograd::ctx().get_generator()();
    }
    auto arm = [&](uint32_t max_pieces, const char* tag) {
        SDPATestConfig config = base;
        config.program_config = program_config_of(knobs);
        config.program_config.max_pieces = max_pieces;
        config.test_name = base.test_name + tag;
        return config;
    };
    base.value_min = 0.0F;  // see expect_split_matches_unsplit
    base.value_max = 1.0F;
    const SDPATestConfig split = arm(knobs.max_pieces, "_split");
    const SDPATestConfig unsplit = arm(1U, "_unsplit");
    // The split arm must really split, or the check proves nothing.
    const auto split_schedule = host_schedule(split);
    ASSERT_GT(split_schedule.num_slots, 0U) << split.test_name << " schedules no split units";
    ASSERT_EQ(host_schedule(unsplit).num_slots, 0U) << unsplit.test_name << " schedules split units";

    xt::xarray<float> split_output;
    xt::xarray<float> unsplit_output;
    xt::xarray<float> split_lse;
    xt::xarray<float> unsplit_lse;
    xt::xarray<float> reference_lse;
    {
        SCOPED_TRACE(split.test_name);
        run_sdpa_test(split, &split_output, &split_lse, &reference_lse);
    }
    {
        SCOPED_TRACE(unsplit.test_name);
        run_sdpa_test(unsplit, &unsplit_output, &unsplit_lse);
    }
    const FoldErrorTerms terms = fold_error_terms(split, split_schedule);
    expect_split_matches_unsplit(split_output, unsplit_output, terms, base.test_name);
    if (base.return_intermediates) {
        expect_split_lse_matches_reference(split_lse, unsplit_lse, reference_lse, terms, base.test_name);
    }
}

void run_chunk_knob_sweep(SDPATestConfig base, const std::vector<ChunkKnobs>& sweep = kChunkKnobSweep) {
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
    for (const auto& knobs : sweep) {
        SDPATestConfig config = base;
        config.program_config = program_config_of(knobs);
        config.test_name = fmt::format(
            "{}_q{}_k{}_h{}_p{}_c{}",
            base.test_name,
            knobs.q_chunk_size,
            knobs.k_chunk_size,
            knobs.heads_per_item,
            knobs.max_pieces,
            knobs.num_cores);
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

// One blocking row per compiled variant (causal, arbitrary, no mask + gate), each its own test so the
// shards spread the kernel compiles; the full sweeps are nightly.
TEST_F(SDPAForwardTest, SDPAForwardTest_ChunkKnobs_Smoke_Causal) {
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
        {{256, 256, 1}});  // whole-sequence chunks: one item per sequence, Sk_chunk_t = St
}

TEST_F(SDPAForwardTest, SDPAForwardTest_ChunkKnobs_Smoke_Arbitrary) {
    SKIP_FOR_LLK_ASSERTS("Skip due to too large code size when assert is enabled.");
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
}

TEST_F(SDPAForwardTest, SDPAForwardTest_ChunkKnobs_Smoke_NoneGate) {
    SKIP_FOR_LLK_ASSERTS("Skip due to too large code size when assert is enabled.");
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
    // The RETURN_INTERMEDIATES-less compile with the gate and an explicit chunking, checked against the
    // references (ValidationTest_IntermediateReturnModes covers it without the gate at the default pick).
    run_chunk_knob_sweep(
        SDPATestConfig{
            .batch_size = 1U,
            .sequence_length = 256U,
            .query_dim = 128U,
            .key_value_dim = 128U,
            .num_query_heads = 2U,
            .num_key_heads = 2U,
            .mask_type = ttml::metal::AttentionMaskType::Causal,
            .use_gate = true,
            .return_intermediates = false,
            .test_name = "ChunkKnobs_NoIntermediates_Gate"},
        {{64, 64, 1}});
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

TEST_F(SDPAForwardTest, NIGHTLY_SDPAForwardTest_ChunkKnobs_Arbitrary_HeadStacking) {
    SKIP_FOR_LLK_ASSERTS("Skip due to too large code size when assert is enabled.");
    // The (1, 1, S, S) mask is read once per step and shared by the stacked heads; the blocking case of
    // this is the {64, 64, 2} row of ChunkKnobs_Smoke_Arbitrary.
    run_chunk_knob_sweep(
        SDPATestConfig{
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

// Few heavy items on many cores: the late items are heavier than a core's fair share, so the schedule cuts
// them into step ranges computed on different cores and merged. 2 heads, S=1024, 64-row chunks: 32 items,
// up to four pieces each.
const SDPATestConfig kSplitUnitsBase{
    .batch_size = 1U,
    .sequence_length = 1024U,
    .query_dim = 128U,
    .key_value_dim = 128U,
    .num_query_heads = 2U,
    .num_key_heads = 2U,
    .mask_type = ttml::metal::AttentionMaskType::Causal,
    .test_name = "ChunkKnobs_SplitUnits"};

TEST_F(SDPAForwardTest, SDPAForwardTest_ChunkKnobs_SplitUnits) {
    SKIP_FOR_LLK_ASSERTS("Skip due to too large code size when assert is enabled.");
    // The one blocking run of the split kernel: the cut rows against the same rows uncut.
    run_split_vs_unsplit(kSplitUnitsBase, {64, 128, 1, 4, 0});
}

TEST_F(SDPAForwardTest, NIGHTLY_SDPAForwardTest_ChunkKnobs_SplitUnits_Sweep) {
    SKIP_FOR_LLK_ASSERTS("Skip due to too large code size when assert is enabled.");
    // The same rows against the references with four pieces, uncut, and on 32 cores so senders, whole items
    // and mergers share cores.
    run_chunk_knob_sweep(kSplitUnitsBase, {{64, 128, 1, 4, 0}, {64, 128, 1, 1, 0}, {64, 128, 1, 0, 32}});
}

TEST_F(SDPAForwardTest, SDPAForwardTest_ChunkKnobs_SplitUnits_Backward) {
    SKIP_FOR_LLK_ASSERTS("Skip due to too large code size when assert is enabled.");
    // The merged logsumexp and output feed the backward kernel. On a shape the default pick cuts, the
    // gradients of the cut forward must match those of the uncut forward, and the autograd op, which passes
    // no program config and so cuts by default, must produce the same gradients as the metal path.
    using namespace ttml;
    SDPATestConfig config = kSplitUnitsBase;
    config.value_min = 0.0F;  // see expect_split_matches_unsplit
    config.test_name = "ChunkKnobs_SplitUnits_Backward";
    const auto split_schedule = host_schedule(config);
    ASSERT_GT(split_schedule.num_slots, 0U) << "the default pick must cut this shape";
    const FoldErrorTerms terms = fold_error_terms(config, split_schedule);

    auto* device = &autograd::ctx().get_device();
    const std::array<std::size_t, 4> shape{1U, 2U, 1024U, 64U};
    const uint32_t seed = 7U;
    auto make = [&](uint32_t s, float lo, float hi) {
        return core::from_xtensor(test_utils::make_uniform_xarray<float>(shape, lo, hi, seed + s), device);
    };
    const ttnn::Tensor query = make(0U, -1.0F, 1.0F);
    const ttnn::Tensor key = make(1U, -1.0F, 1.0F);
    const ttnn::Tensor value = make(2U, 0.0F, 1.0F);
    const ttnn::Tensor grad_output = make(3U, -1.0F, 1.0F);

    struct Grads {
        xt::xarray<float> dQ, dK, dV;
    };
    auto grads_of = [&](const metal::SDPAForwardProgramConfig& cfg) {
        auto fw = metal::sdpa_fw(
            query,
            key,
            value,
            metal::AttentionMaskType::Causal,
            std::nullopt,
            0.0F,
            /*return_intermediates=*/true,
            std::nullopt,
            cfg);
        auto [dQ, dK, dV, dGate] = metal::sdpa_bw(
            grad_output, fw[0].value(), query, key, value, fw[1].value(), metal::AttentionMaskType::Causal);
        return Grads{core::to_xtensor(dQ), core::to_xtensor(dK), core::to_xtensor(dV)};
    };
    const Grads split = grads_of({});
    const Grads unsplit = grads_of({.max_pieces = 1U});
    // The gradients inherit the forward's output and logsumexp gaps (relative, the unshared steps' truncation)
    // and add the backward kernel's own rounding of them; the bound is the forward's mean bound on the mean.
    const float mean_rel_tol = 1e-3F + 4.8828125e-4F * static_cast<float>(terms.partials + terms.extra_unsplit_steps);
    auto expect_close = [&](const xt::xarray<float>& a, const xt::xarray<float>& b, const char* what) {
        ASSERT_EQ(a.shape(), b.shape());
        const float mean_rel = xt::mean(xt::abs(a - b))() / xt::mean(xt::abs(b))();
        fmt::print(
            "{}: split vs unsplit {} mean relative gap {:.3e} (bound {:.1e})\n",
            config.test_name,
            what,
            mean_rel,
            mean_rel_tol);
        EXPECT_LT(mean_rel, mean_rel_tol) << config.test_name << ": " << what << " of the cut forward is off";
        EXPECT_TRUE(xt::all(xt::isfinite(a))) << what;
    };
    expect_close(split.dQ, unsplit.dQ, "dQ");
    expect_close(split.dK, unsplit.dK, "dK");
    expect_close(split.dV, unsplit.dV, "dV");

    // The autograd entry point: forward, then backward with the same upstream gradient.
    auto q = autograd::create_tensor(query, /* requires_grad */ true);
    auto k = autograd::create_tensor(key, /* requires_grad */ true);
    auto v = autograd::create_tensor(value, /* requires_grad */ true);
    auto out = ops::scaled_dot_product_attention(q, k, v);
    out->set_grad(grad_output);
    out->backward();
    EXPECT_TRUE(xt::all(xt::equal(core::to_xtensor(q->get_grad()), split.dQ))) << "autograd dQ != metal dQ";
    EXPECT_TRUE(xt::all(xt::equal(core::to_xtensor(k->get_grad()), split.dK))) << "autograd dK != metal dK";
    EXPECT_TRUE(xt::all(xt::equal(core::to_xtensor(v->get_grad()), split.dV))) << "autograd dV != metal dV";
}

TEST_F(SDPAForwardTest, NIGHTLY_SDPAForwardTest_ChunkKnobs_SplitUnits_Long) {
    SKIP_FOR_LLK_ASSERTS("Skip due to too large code size when assert is enabled.");
    // S=2048: 128-row chunks of up to eight pieces, 64-row chunks of two, 128 one-tile-row items when the
    // shape rule picks (it shrinks Sq to fill the cores), and on 48 cores.
    const SDPATestConfig base{
        .batch_size = 1U,
        .sequence_length = 2048U,
        .query_dim = 128U,
        .key_value_dim = 128U,
        .num_query_heads = 2U,
        .num_key_heads = 2U,
        .mask_type = ttml::metal::AttentionMaskType::Causal,
        .test_name = "ChunkKnobs_SplitUnits_Long"};
    run_chunk_knob_sweep(base, {{128, 256, 1, 4, 0}, {128, 256, 1, 8, 0}, {64, 128, 1, 2, 0}, {64, 128, 1, 0, 48}});
    run_split_vs_unsplit(base, {128, 256, 1, 8, 0});
}

TEST_F(SDPAForwardTest, NIGHTLY_SDPAForwardTest_ChunkKnobs_SplitUnits_ManyPieces) {
    SKIP_FOR_LLK_ASSERTS("Skip due to too large code size when assert is enabled.");
    // One head with one K tile per step: items of up to 64 steps cut into up to 8 and 16 pieces, so a
    // merger folds in more than three partials.
    const SDPATestConfig base{
        .batch_size = 1U,
        .sequence_length = 2048U,
        .query_dim = 64U,
        .key_value_dim = 64U,
        .num_query_heads = 1U,
        .num_key_heads = 1U,
        .mask_type = ttml::metal::AttentionMaskType::Causal,
        .test_name = "ChunkKnobs_SplitUnits_ManyPieces"};
    run_chunk_knob_sweep(base, {{128, 32, 1, 8, 0}, {128, 32, 1, 16, 0}});
    run_split_vs_unsplit(base, {128, 32, 1, 16, 0});
}

// Splits with stacked heads, a different value dim and the gate; the gate is applied by the merger only.
const SDPATestConfig kSplitUnitsGqaGateBase{
    .batch_size = 1U,
    .sequence_length = 512U,
    .query_dim = 512U,      // 4 heads * 128
    .key_value_dim = 256U,  // 2 kv heads * 128
    .value_dim = 128U,      // 2 kv heads * 64
    .num_query_heads = 4U,
    .num_key_heads = 2U,
    .mask_type = ttml::metal::AttentionMaskType::Causal,
    .use_gate = true,
    .test_name = "ChunkKnobs_SplitUnits_GQA_Gate"};

TEST_F(SDPAForwardTest, SDPAForwardTest_ChunkKnobs_SplitUnits_GQA_Gate) {
    SKIP_FOR_LLK_ASSERTS("Skip due to too large code size when assert is enabled.");
    run_split_vs_unsplit(kSplitUnitsGqaGateBase, {64, 128, 2, 4, 0});
    // The gated split kernel without the logsumexp: a variant no other test compiles.
    SDPATestConfig no_intermediates = kSplitUnitsGqaGateBase;
    no_intermediates.program_config = program_config_of({64, 128, 2, 4, 0});
    no_intermediates.return_intermediates = false;
    no_intermediates.test_name = "ChunkKnobs_SplitUnits_GQA_Gate_NoIntermediates";
    ASSERT_GT(host_schedule(no_intermediates).num_slots, 0U);
    run_sdpa_test(no_intermediates);
}

TEST_F(SDPAForwardTest, NIGHTLY_SDPAForwardTest_ChunkKnobs_SplitUnits_GQA_Gate_Sweep) {
    SKIP_FOR_LLK_ASSERTS("Skip due to too large code size when assert is enabled.");
    run_chunk_knob_sweep(kSplitUnitsGqaGateBase, {{64, 128, 2, 4, 0}, {128, 256, 2, 3, 16}});
}

TEST_F(SDPAForwardTest, SDPAForwardTest_ChunkKnobs_NumCores) {
    SKIP_FOR_LLK_ASSERTS("Skip due to too large code size when assert is enabled.");
    run_chunk_knob_sweep(
        SDPATestConfig{
            .batch_size = 2U,
            .sequence_length = 256U,
            .query_dim = 256U,
            .key_value_dim = 256U,
            .num_query_heads = 4U,
            .num_key_heads = 4U,
            .mask_type = ttml::metal::AttentionMaskType::Causal,
            .test_name = "ChunkKnobs_NumCores"},
        {{0, 0, 0, 0, 1}, {0, 0, 0, 0, 7}, {32, 32, 1, 1, 3}});
}

TEST_F(SDPAForwardTest, SDPAForwardTest_ChunkKnobs_DefaultPick) {
    // With the knobs at 0 the shape rule picks the chunking: larger than main's one Q row / four K tiles
    // where L1 allows, and always something that fits.
    using namespace ttml::metal::ops::sdpa_fw::device;
    constexpr uint64_t kBudget = 1'300'000U;  // a Blackhole core's usable L1
    constexpr uint32_t kCores = 130U;         // the Blackhole p150 compute grid
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
    EXPECT_LE(l1_bytes(cb_tiles(tinyllama, picked, /* with_partials */ true)), kBudget);

    SDPAForwardShape wide = tinyllama;  // head dim 512: the biggest chunks do not fit, the pick must degrade
    wide.qWt = 16U;
    wide.vWt = 16U;
    const auto picked_wide = resolve_chunks({}, wide, kBudget, kCores);
    EXPECT_LE(l1_bytes(cb_tiles(wide, picked_wide, /* with_partials */ true)), kBudget);

    // A user knob pins its axis; the others are still picked.
    const auto pinned = resolve_chunks({.k_chunk_size = 32U}, tinyllama, kBudget, kCores);
    EXPECT_EQ(pinned.Sk_chunk_t, 1U);
    EXPECT_GE(pinned.rows(), 4U);

    // Few causal sequences (8 heads, S=2048): four-row chunks give 128 items for 130 cores, so the pick
    // shrinks to two rows per chunk (256 items) and keeps every core busy.
    SDPAForwardShape few_seqs = tinyllama;
    few_seqs.q_heads = 8U;
    few_seqs.kv_heads = 8U;
    const auto picked_few = resolve_chunks({}, few_seqs, kBudget, kCores);
    EXPECT_EQ(picked_few.Sq_chunk_t, 2U);
    EXPECT_GE(num_work_items(few_seqs, picked_few), kCores);
    EXPECT_EQ(picked_few.Sk_chunk_t, picked.Sk_chunk_t) << "the K/V step does not depend on the core count";
}

namespace {

using ttml::metal::ops::sdpa_fw::device::SDPAForwardChunks;
using ttml::metal::ops::sdpa_fw::device::SDPAForwardSchedule;
using ttml::metal::ops::sdpa_fw::device::SDPAForwardShape;
using ttml::metal::ops::sdpa_fw::device::UnitRole;
using ttml::metal::ops::sdpa_fw::device::WorkUnit;

// What every schedule must satisfy for the kernels to compute each item exactly once and for the senders
// and mergers to find each other: full coverage, consecutive slots per item, matching semaphores, and
// senders before whole items before mergers on every core.
void check_schedule(
    const SDPAForwardShape& shape,
    const SDPAForwardChunks& chunks,
    const SDPAForwardSchedule& schedule,
    uint32_t max_pieces) {
    using namespace ttml::metal::ops::sdpa_fw::device;
    const uint32_t total_items = num_work_items(shape, chunks);
    std::vector<uint32_t> whole_count(total_items, 0U);
    struct Cut {
        std::vector<std::pair<uint32_t, uint32_t>> ranges;  // step ranges of every piece
        std::vector<uint32_t> sender_slots;
        std::vector<uint32_t> sender_sems;
        std::vector<uint32_t> sender_merger_cores;
        std::optional<WorkUnit> merger;
        uint32_t merger_core = 0;
    };
    std::map<uint32_t, Cut> cuts;
    std::set<uint32_t> slots;
    for (uint32_t core = 0; core < schedule.per_core.size(); ++core) {
        auto rank = [](UnitRole role) { return role == UnitRole::Sender ? 0 : role == UnitRole::Whole ? 1 : 2; };
        int last_rank = 0;
        std::set<uint32_t> sems_on_core;
        for (const WorkUnit& unit : schedule.per_core[core]) {
            EXPECT_GE(rank(unit.role), last_rank) << "core " << core << ": senders, then whole items, then mergers";
            last_rank = rank(unit.role);
            if (unit.role == UnitRole::Whole) {
                ASSERT_LE(unit.item + unit.num_items, total_items);
                for (uint32_t item = unit.item; item < unit.item + unit.num_items; ++item) {
                    ++whole_count[item];
                }
                continue;
            }
            ASSERT_LT(unit.item, total_items);
            Cut& cut = cuts[unit.item];
            cut.ranges.emplace_back(unit.step_begin, unit.step_end);
            EXPECT_LT(unit.step_begin, unit.step_end) << "empty piece";
            if (unit.role == UnitRole::Sender) {
                EXPECT_TRUE(slots.insert(unit.slot).second) << "slot " << unit.slot << " used twice";
                EXPECT_LT(unit.slot, schedule.num_slots);
                cut.sender_slots.push_back(unit.slot);
                cut.sender_sems.push_back(unit.sem_idx);
                cut.sender_merger_cores.push_back(unit.merger_core);
            } else {
                EXPECT_FALSE(cut.merger.has_value()) << "item " << unit.item << " has two mergers";
                EXPECT_TRUE(sems_on_core.insert(unit.sem_idx).second) << "core " << core << " reuses a semaphore";
                EXPECT_LT(unit.sem_idx, kMaxMergersPerCore);
                cut.merger = unit;
                cut.merger_core = core;
            }
        }
    }
    EXPECT_EQ(slots.size(), schedule.num_slots) << "every slot belongs to one sender";
    for (uint32_t item = 0; item < total_items; ++item) {
        const auto cut = cuts.find(item);
        if (cut == cuts.end()) {
            EXPECT_EQ(whole_count[item], 1U) << "item " << item << " is computed " << whole_count[item] << " times";
            continue;
        }
        EXPECT_EQ(whole_count[item], 0U) << "item " << item << " is both cut and whole";
        Cut& c = cut->second;
        ASSERT_TRUE(c.merger.has_value()) << "item " << item << " is cut but has no merger";
        EXPECT_EQ(c.merger->num_partials, c.sender_slots.size());
        EXPECT_LE(c.ranges.size(), max_pieces);
        EXPECT_GE(c.ranges.size(), 2U);
        std::sort(c.ranges.begin(), c.ranges.end());
        uint32_t next = 0;
        for (const auto& [begin, end] : c.ranges) {
            EXPECT_EQ(begin, next) << "item " << item << ": pieces must tile its K/V steps";
            next = end;
        }
        EXPECT_EQ(next, num_kv_steps(shape, chunks, item)) << "item " << item << ": pieces must cover every step";
        std::sort(c.sender_slots.begin(), c.sender_slots.end());
        for (uint32_t k = 0; k < c.sender_slots.size(); ++k) {
            EXPECT_EQ(c.sender_slots[k], c.merger->slot + k) << "item " << item << ": slots must follow the merger's";
            EXPECT_EQ(c.sender_sems[k], c.merger->sem_idx)
                << "item " << item << ": a sender bumps its merger's semaphore";
            EXPECT_EQ(c.sender_merger_cores[k], c.merger_core)
                << "item " << item << ": a sender targets its merger's core";
        }
    }
}

SDPAForwardShape causal_shape(uint32_t batch, uint32_t q_heads, uint32_t kv_heads, uint32_t St, uint32_t Wt) {
    return SDPAForwardShape{
        .batch = batch,
        .q_heads = q_heads,
        .kv_heads = kv_heads,
        .St = St,
        .qWt = Wt,
        .vWt = Wt,
        .mask_type = ttml::metal::AttentionMaskType::Causal,
        .has_gate = false,
        .return_intermediates = true};
}

uint32_t count_role(const SDPAForwardSchedule& schedule, UnitRole role) {
    uint32_t n = 0;
    for (const auto& units : schedule.per_core) {
        for (const auto& unit : units) {
            n += unit.role == role ? 1U : 0U;
        }
    }
    return n;
}

}  // namespace

TEST_F(SDPAForwardTest, SDPAForwardTest_ChunkKnobs_Schedule) {
    // The scheduler is pure, so its invariants are checked on the host for the cases the device tests can
    // not reach: many pieces per item, the merger cap, the runtime-arg fallback.
    using namespace ttml::metal::ops::sdpa_fw::device;
    constexpr uint32_t kCores = 120U;

    // Enough items per core: nothing is cut, every core gets whole items.
    const SDPAForwardShape tinyllama = causal_shape(1U, 32U, 4U, 64U, 2U);
    const SDPAForwardChunks sq4sk8{.Sq_chunk_t = 4U, .Sk_chunk_t = 8U, .heads_per_item = 1U};
    const SDPAForwardSchedule uncut = make_schedule(tinyllama, sq4sk8, kCores, kDefaultMaxPieces);
    EXPECT_EQ(uncut.num_slots, 0U);
    EXPECT_EQ(count_role(uncut, UnitRole::Sender), 0U);
    EXPECT_EQ(uncut.per_core.size(), kCores);
    check_schedule(tinyllama, sq4sk8, uncut, kDefaultMaxPieces);

    // Two heads, S=2048: the late items are cut. Only items heavier than a core's fair share are cut, so the
    // scratch slots (one per sender) stay under the core count: that bounds the scratch the program owns.
    const SDPAForwardShape two_heads = causal_shape(1U, 2U, 2U, 64U, 2U);
    const SDPAForwardSchedule cut = make_schedule(two_heads, sq4sk8, kCores, kDefaultMaxPieces);
    EXPECT_GT(cut.num_slots, 0U);
    EXPECT_LT(cut.num_slots, kCores);
    EXPECT_EQ(count_role(cut, UnitRole::Sender), cut.num_slots);
    EXPECT_GT(count_role(cut, UnitRole::Merger), 0U);
    check_schedule(two_heads, sq4sk8, cut, kDefaultMaxPieces);

    // max_pieces = 1 never cuts.
    const SDPAForwardSchedule one_piece = make_schedule(two_heads, sq4sk8, kCores, 1U);
    EXPECT_EQ(one_piece.num_slots, 0U);
    check_schedule(two_heads, sq4sk8, one_piece, 1U);

    // One head with one K tile per step: items of up to 64 steps cut into many pieces, more partials than
    // any device test reaches, several mergers on one core.
    const SDPAForwardShape one_head = causal_shape(1U, 1U, 1U, 64U, 2U);
    const SDPAForwardChunks sq4sk1{.Sq_chunk_t = 4U, .Sk_chunk_t = 1U, .heads_per_item = 1U};
    const SDPAForwardSchedule many = make_schedule(one_head, sq4sk1, kCores, 16U);
    uint32_t most_partials = 0;
    for (const auto& units : many.per_core) {
        for (const auto& unit : units) {
            most_partials = std::max(most_partials, unit.num_partials);
        }
    }
    EXPECT_GT(most_partials, 3U);
    EXPECT_LT(many.num_slots, kCores);
    check_schedule(one_head, sq4sk1, many, 16U);

    // More cores than items with no cuts allowed: one core per item.
    const SDPAForwardSchedule few_items = make_schedule(two_heads, sq4sk8, kCores, 1U);
    EXPECT_EQ(few_items.per_core.size(), num_work_items(two_heads, sq4sk8));

    // Longest-first scatters equal-cost items across the cores, so with far more items than the runtime
    // args can list per core the schedule falls back to heavy/light chunk pairs: every core gets the same
    // number of K/V steps within one pair, folded into a few runs.
    const SDPAForwardShape many_items = causal_shape(2U, 32U, 32U, 256U, 2U);
    const SDPAForwardChunks sq1sk8{.Sq_chunk_t = 1U, .Sk_chunk_t = 8U, .heads_per_item = 1U};
    const SDPAForwardSchedule paired = make_schedule(many_items, sq1sk8, kCores, kDefaultMaxPieces);
    const uint32_t total = num_work_items(many_items, sq1sk8);
    ASSERT_EQ(paired.per_core.size(), kCores);
    EXPECT_EQ(paired.num_slots, 0U);
    uint64_t min_steps = std::numeric_limits<uint64_t>::max();
    uint64_t max_steps = 0;
    for (const auto& units : paired.per_core) {
        EXPECT_LE(units.size(), 4U) << "pairs of one stack fold into runs";
        uint64_t steps = 0;
        for (const WorkUnit& unit : units) {
            for (uint32_t item = unit.item; item < unit.item + unit.num_items; ++item) {
                steps += num_kv_steps(many_items, sq1sk8, item);
            }
        }
        min_steps = std::min(min_steps, steps);
        max_steps = std::max(max_steps, steps);
    }
    // 16384 items on 120 cores: 136 or 137 items each, one pair's worth of steps apart at most.
    EXPECT_LE(max_steps - min_steps, static_cast<uint64_t>(num_kv_steps(many_items, sq1sk8, total - 1U) + 1U));
    check_schedule(many_items, sq1sk8, paired, kDefaultMaxPieces);
}

// The production pick (four-row chunks, eight K/V tiles) only lands on device when the items keep the
// cores busy; every other blocking shape is too small for that and resolves to one-row chunks. B is sized
// from the grid: S=512 gives four chunks per sequence, so B*H = cores/2 + 2 sequences give about two items
// per core, as TinyLlama has. D=32 keeps the host references cheap; the pick does not depend on it below 128.
TEST_F(SDPAForwardTest, SDPAForwardTest_ChunkKnobs_DefaultPairedOnDevice) {
    SKIP_FOR_LLK_ASSERTS("Skip due to too large code size when assert is enabled.");
    using namespace ttml::metal::ops::sdpa_fw::device;
    constexpr uint32_t H = 2U, S = 512U, D = 32U;
    auto* device = &ttml::autograd::ctx().get_device();
    const auto grid = device->compute_with_storage_grid_size();
    const uint32_t cores = static_cast<uint32_t>(grid.x * grid.y);
    const uint32_t B = (cores / 2U + 2U + H - 1U) / H;
    const SDPAForwardShape shape{
        .batch = B,
        .q_heads = H,
        .kv_heads = H,
        .St = S / 32U,
        .qWt = D / 32U,
        .vWt = D / 32U,
        .mask_type = ttml::metal::AttentionMaskType::Causal,
        .has_gate = false,
        .return_intermediates = true};
    const auto picked = resolve_chunks({}, shape, ttml::metal::available_l1_bytes(*device), cores);
    ASSERT_EQ(picked.Sq_chunk_t, 4U) << "the shape must take the production pick";
    ASSERT_EQ(picked.Sk_chunk_t, 8U);
    ASSERT_GE(num_work_items(shape, picked), cores) << "an item per core, so the pick keeps four rows";
    run_sdpa_test(SDPATestConfig{
        .batch_size = B,
        .sequence_length = S,
        .query_dim = H * D,
        .key_value_dim = H * D,
        .num_query_heads = H,
        .num_key_heads = H,
        .mask_type = ttml::metal::AttentionMaskType::Causal,
        .test_name = "ChunkKnobs_DefaultPairedOnDevice"});
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
    auto run_cfg = [&](const metal::SDPAForwardProgramConfig& program_config) {
        return metal::sdpa_fw(
            query,
            key,
            value,
            metal::AttentionMaskType::Causal,
            std::nullopt,
            0.0F,
            false,
            std::nullopt,
            program_config);
    };
    auto run = [&](uint32_t q_chunk_size, uint32_t k_chunk_size, uint32_t heads_per_item = 0U) {
        return run_cfg({.q_chunk_size = q_chunk_size, .k_chunk_size = k_chunk_size, .heads_per_item = heads_per_item});
    };
    EXPECT_NO_THROW(run(0U, 0U));
    EXPECT_THAT([&] { run(48U, 0U); }, ThrowsMessage<std::exception>(HasSubstr("q_chunk_size (48)")));
    EXPECT_THAT([&] { run(0U, 48U); }, ThrowsMessage<std::exception>(HasSubstr("k_chunk_size (48)")));
    EXPECT_THAT([&] { run(96U, 0U); }, ThrowsMessage<std::exception>(HasSubstr("q_chunk_size (96)")));
    EXPECT_THAT([&] { run(0U, 2U * S); }, ThrowsMessage<std::exception>(HasSubstr("k_chunk_size (512)")));
    EXPECT_THAT([&] { run(0U, 0U, 2U); }, ThrowsMessage<std::exception>(HasSubstr("heads_per_item (2)")));
    EXPECT_THAT([&] { run_cfg({.max_pieces = 64U}); }, ThrowsMessage<std::exception>(HasSubstr("max_pieces (64)")));
    EXPECT_THAT(
        [&] { run_cfg({.num_cores = 100000U}); }, ThrowsMessage<std::exception>(HasSubstr("num_cores (100000)")));

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

    // validate rejects a config that cannot fit even with the unset chunk sizes at one tile, and accepts one
    // that fits at the floor, without building a program.
    {
        using namespace ttml::metal::ops::sdpa_fw::device;
        const SDPAForwardShape wide = shape_of(
            wide_query,
            wide_key,
            wide_value,
            metal::AttentionMaskType::Causal,
            /*has_gate=*/false,
            /*return_intermediates=*/false);
        const uint64_t budget = ttml::metal::available_l1_bytes(ttml::autograd::ctx().get_device());
        const auto grid = ttml::autograd::ctx().get_device().compute_with_storage_grid_size();
        const uint32_t kGridCores = grid.x * grid.y;
        EXPECT_THAT(
            [&] { check_chunks({.q_chunk_size = S}, wide, budget, kGridCores); },
            ThrowsMessage<std::exception>(HasSubstr("SDPA forward circular buffers need")));
        EXPECT_NO_THROW(check_chunks({.k_chunk_size = 32U}, wide, budget, kGridCores));
        EXPECT_NO_THROW(check_chunks({}, wide, budget, kGridCores));
    }
}

TEST_F(SDPAForwardTest, SDPAForwardTest_ChunkKnobs_ProgramCacheMissAndHit) {
    SKIP_FOR_LLK_ASSERTS("Skip due to too large code size when assert is enabled.");
    // Two chunkings on the same tensors are two cached programs; the first one again is a hit, also on a
    // second set of inputs at other addresses. The first chunking splits, so the hit reuses the cached
    // program's scratch tensor, its semaphores (reset by every merger) and its slots.
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    device->enable_program_cache();
    const SDPATestConfig shape_config{
        .batch_size = 1U,
        .sequence_length = 1024U,
        .query_dim = 128U,
        .key_value_dim = 128U,
        .num_query_heads = 2U,
        .num_key_heads = 2U,
        .mask_type = metal::AttentionMaskType::Causal,
        .return_intermediates = false};
    const std::array<std::size_t, 4> shape{1U, 2U, 1024U, 64U};
    auto make = [&](uint32_t seed, float lo, float hi) {
        return core::from_xtensor(test_utils::make_uniform_xarray<float>(shape, lo, hi, seed), device);
    };
    struct Inputs {
        ttnn::Tensor query, key, value;
    };
    // V in [0, 1]: see expect_split_matches_unsplit.
    const Inputs first{make(1U, -1.0F, 1.0F), make(2U, -1.0F, 1.0F), make(3U, 0.0F, 1.0F)};
    const Inputs second{
        make(4U, -1.0F, 1.0F), make(5U, -1.0F, 1.0F), make(6U, 0.0F, 1.0F)};  // live at once: other addresses
    auto run = [&](const metal::SDPAForwardProgramConfig& cfg, const Inputs& in) {
        return core::to_xtensor(metal::sdpa_fw(
                                    in.query,
                                    in.key,
                                    in.value,
                                    metal::AttentionMaskType::Causal,
                                    std::nullopt,
                                    0.0F,
                                    false,
                                    std::nullopt,
                                    cfg)[0]
                                    .value());
    };
    const metal::SDPAForwardProgramConfig a{.q_chunk_size = 64U, .k_chunk_size = 128U};
    const metal::SDPAForwardProgramConfig a_unsplit{.q_chunk_size = 64U, .k_chunk_size = 128U, .max_pieces = 1U};
    const metal::SDPAForwardProgramConfig b{.q_chunk_size = 128U, .k_chunk_size = 256U};
    SDPATestConfig split_config = shape_config;
    split_config.program_config = a;
    const auto split_schedule = host_schedule(split_config);
    ASSERT_GT(split_schedule.num_slots, 0U) << "the twice-run config must split";
    const auto entries0 = device->num_program_cache_entries();
    const xt::xarray<float> out_a = run(a, first);
    const auto entries_a = device->num_program_cache_entries();
    ASSERT_GT(entries_a, entries0) << "first config did not populate the cache";
    const xt::xarray<float> out_b = run(b, first);
    EXPECT_EQ(device->num_program_cache_entries(), entries_a + 1U) << "second config must miss: different program";
    const xt::xarray<float> out_a_second = run(a, second);
    EXPECT_EQ(device->num_program_cache_entries(), entries_a + 1U) << "first config on new inputs must hit";
    const xt::xarray<float> out_a_again = run(a, first);
    EXPECT_EQ(device->num_program_cache_entries(), entries_a + 1U) << "first config again must hit";
    EXPECT_TRUE(xt::all(xt::equal(out_a, out_a_again))) << "cache hit must reproduce the first run";
    EXPECT_LT(xt::mean(xt::abs(out_a - out_b))(), kCrossChunkingMeanAbsTol) << "both chunkings compute the same op";
    // The hit on the second inputs must have read them (not the first inputs) and merged them right.
    const xt::xarray<float> out_a_second_unsplit = run(a_unsplit, second);
    EXPECT_FALSE(xt::all(xt::equal(out_a_second, out_a))) << "the hit must read the new inputs";
    expect_split_matches_unsplit(
        out_a_second,
        out_a_second_unsplit,
        fold_error_terms(split_config, split_schedule),
        "ProgramCacheMissAndHit_second_inputs");

    // The mask address is a reader runtime arg of its own: a hit with a second live mask tensor (same shape,
    // other values and address) must read the new mask, not the first one.
    constexpr std::size_t kSeq = 1024U;
    const std::array<std::size_t, 4> mask_shape{1U, 1U, kSeq, kSeq};
    xt::xarray<float> lower = xt::tril(xt::ones<float>({kSeq, kSeq}));
    lower.reshape(mask_shape);
    xt::xarray<float> upper = xt::triu(xt::ones<float>({kSeq, kSeq}));
    upper.reshape(mask_shape);
    const ttnn::Tensor causal_mask = core::from_xtensor(lower, device);
    const ttnn::Tensor anticausal_mask = core::from_xtensor(upper, device);  // both stay alive
    const auto entries_before_masked = device->num_program_cache_entries();  // the uploads cache their own ops
    // Mask-tensor programs are never cut, so the run is held against the uncut run on the same inputs.
    auto run_masked = [&](const ttnn::Tensor& mask) {
        return core::to_xtensor(metal::sdpa_fw(
                                    second.query,
                                    second.key,
                                    second.value,
                                    metal::AttentionMaskType::Arbitrary,
                                    mask,
                                    0.0F,
                                    false,
                                    std::nullopt,
                                    a)[0]
                                    .value());
    };
    const xt::xarray<float> out_causal = run_masked(causal_mask);
    const auto entries_masked = device->num_program_cache_entries();
    EXPECT_EQ(entries_masked, entries_before_masked + 1U) << "the mask tensor variant is its own program";
    const xt::xarray<float> out_anticausal = run_masked(anticausal_mask);
    EXPECT_EQ(device->num_program_cache_entries(), entries_masked) << "the second mask must hit";
    EXPECT_LT(xt::mean(xt::abs(out_causal - out_a_second_unsplit))(), kCrossChunkingMeanAbsTol)
        << "the lower-triangular mask tensor must agree with the causal kernel mask";
    EXPECT_FALSE(xt::allclose(out_anticausal, out_causal, 1e-3F, 1e-3F))
        << "hit reproduced the first mask: stale mask address";
}

TEST_F(SDPAForwardTest, SDPAForwardTest_ChunkKnobs_ChangeProgramHash) {
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
    Params p_changed = base;
    p_changed.program_config.max_pieces = 2U;
    EXPECT_NE(ttsl::hash::hash_objects_with_default_seed(base), ttsl::hash::hash_objects_with_default_seed(p_changed));
    Params c_changed = base;
    c_changed.program_config.num_cores = 8U;
    EXPECT_NE(ttsl::hash::hash_objects_with_default_seed(base), ttsl::hash::hash_objects_with_default_seed(c_changed));
    EXPECT_NE(ttsl::hash::hash_objects_with_default_seed(base), ttsl::hash::hash_objects_with_default_seed(h_changed));
    EXPECT_NE(ttsl::hash::hash_objects_with_default_seed(base), ttsl::hash::hash_objects_with_default_seed(q_changed));
    EXPECT_NE(ttsl::hash::hash_objects_with_default_seed(base), ttsl::hash::hash_objects_with_default_seed(k_changed));
    EXPECT_NE(
        ttsl::hash::hash_objects_with_default_seed(q_changed), ttsl::hash::hash_objects_with_default_seed(k_changed));
}
