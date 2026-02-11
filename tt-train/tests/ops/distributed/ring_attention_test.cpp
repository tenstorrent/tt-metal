// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Test for ring_attention_sdpa distributed operation.
 *
 * Ring attention computes full-sequence attention when the sequence is
 * sharded across CP (context parallel) devices.
 */

#include "ops/distributed/ring_attention.hpp"

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>
#include <core/xtensor_utils.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <umd/device/cluster.hpp>

#include "autograd/auto_context.hpp"
#include "core/distributed/socket_manager.hpp"
#include "core/random.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/scaled_dot_product_attention.hpp"
#include "ttnn_fixed/distributed/tt_metal.hpp"

static bool check_32_chips() {
    auto cluster_desc = tt::umd::Cluster::create_cluster_descriptor();
    auto all_chips = cluster_desc->get_all_chips();
    return all_chips.size() == 32;
}

class GalaxyRingAttentionTest : public ::testing::Test {
public:
    static void SetUpTestSuite() {
        if (!check_32_chips()) {
            GTEST_SKIP() << "Skipping Galaxy specific tests";
        }
        ttml::autograd::ctx().initialize_distributed_context(0, nullptr);
        ttml::ttnn_fixed::distributed::enable_fabric(32);
        ttml::autograd::ctx().open_device(tt::tt_metal::distributed::MeshShape(4, 8));
        ttml::autograd::ctx().set_seed(42);
        ttml::autograd::ctx().initialize_socket_manager(ttnn::distributed::SocketType::FABRIC);

        // Configure parallelism context for CP (CP axis will be 0)
        ttml::autograd::ctx().initialize_parallelism_context(
            {.enable_ddp = false, .enable_tp = true, .enable_cp = true});
    }

    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }
};

// Reference SDPA forward and backward implementation in xtensor for comparison
struct SDPARefResult {
    xt::xarray<float> output;
    xt::xarray<float> attention_weights;  // Saved for backward
    float scale;
};

SDPARefResult reference_sdpa_forward(
    const xt::xarray<float>& query,
    const xt::xarray<float>& key,
    const xt::xarray<float>& value,
    const std::optional<xt::xarray<float>>& mask = std::nullopt) {
    // query, key, value: (B, H, S, D)
    auto shape = query.shape();
    size_t B = shape[0];
    size_t H = shape[1];
    size_t S_q = shape[2];
    size_t D = shape[3];
    size_t S_k = key.shape()[2];

    float scale = 1.0F / std::sqrt(static_cast<float>(D));

    // Compute attention scores: Q @ K^T
    // Result shape: (B, H, S_q, S_k)
    xt::xarray<float> scores = xt::zeros<float>({B, H, S_q, S_k});
    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t i = 0; i < S_q; ++i) {
                for (size_t j = 0; j < S_k; ++j) {
                    float dot = 0.0F;
                    for (size_t d = 0; d < D; ++d) {
                        dot += query(b, h, i, d) * key(b, h, j, d);
                    }
                    scores(b, h, i, j) = dot * scale;
                }
            }
        }
    }

    // Apply mask if provided
    if (mask.has_value()) {
        for (size_t b = 0; b < B; ++b) {
            for (size_t h = 0; h < H; ++h) {
                for (size_t i = 0; i < S_q; ++i) {
                    for (size_t j = 0; j < S_k; ++j) {
                        if ((*mask)(0, 0, i, j) < 0.5F) {
                            scores(b, h, i, j) = -1e9F;
                        }
                    }
                }
            }
        }
    }

    // Softmax over last dimension
    xt::xarray<float> attention_weights = xt::zeros<float>({B, H, S_q, S_k});
    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t i = 0; i < S_q; ++i) {
                float max_val = scores(b, h, i, 0);
                for (size_t j = 1; j < S_k; ++j) {
                    max_val = std::max(max_val, scores(b, h, i, j));
                }
                float sum_exp = 0.0F;
                for (size_t j = 0; j < S_k; ++j) {
                    attention_weights(b, h, i, j) = std::exp(scores(b, h, i, j) - max_val);
                    sum_exp += attention_weights(b, h, i, j);
                }
                for (size_t j = 0; j < S_k; ++j) {
                    attention_weights(b, h, i, j) /= sum_exp;
                }
            }
        }
    }

    // Compute output: attention_weights @ V
    xt::xarray<float> output = xt::zeros<float>({B, H, S_q, D});
    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t i = 0; i < S_q; ++i) {
                for (size_t d = 0; d < D; ++d) {
                    float sum = 0.0F;
                    for (size_t j = 0; j < S_k; ++j) {
                        sum += attention_weights(b, h, i, j) * value(b, h, j, d);
                    }
                    output(b, h, i, d) = sum;
                }
            }
        }
    }

    return {output, attention_weights, scale};
}

struct SDPAGrads {
    xt::xarray<float> dQ;
    xt::xarray<float> dK;
    xt::xarray<float> dV;
};

SDPAGrads reference_sdpa_backward(
    const xt::xarray<float>& query,
    const xt::xarray<float>& key,
    const xt::xarray<float>& value,
    const xt::xarray<float>& attention_weights,
    const xt::xarray<float>& grad_output,
    float scale) {
    auto shape = query.shape();
    size_t B = shape[0];
    size_t H = shape[1];
    size_t S_q = shape[2];
    size_t D = shape[3];
    size_t S_k = key.shape()[2];

    // dL/dV = attention_weights^T @ grad_output
    xt::xarray<float> dV = xt::zeros<float>({B, H, S_k, D});
    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t j = 0; j < S_k; ++j) {
                for (size_t d = 0; d < D; ++d) {
                    float sum = 0.0F;
                    for (size_t i = 0; i < S_q; ++i) {
                        sum += attention_weights(b, h, i, j) * grad_output(b, h, i, d);
                    }
                    dV(b, h, j, d) = sum;
                }
            }
        }
    }

    // dL/d(attention_weights) = grad_output @ V^T
    xt::xarray<float> d_attn_weights = xt::zeros<float>({B, H, S_q, S_k});
    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t i = 0; i < S_q; ++i) {
                for (size_t j = 0; j < S_k; ++j) {
                    float sum = 0.0F;
                    for (size_t d = 0; d < D; ++d) {
                        sum += grad_output(b, h, i, d) * value(b, h, j, d);
                    }
                    d_attn_weights(b, h, i, j) = sum;
                }
            }
        }
    }

    // Softmax backward
    xt::xarray<float> d_scores = xt::zeros<float>({B, H, S_q, S_k});
    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t i = 0; i < S_q; ++i) {
                float row_sum = 0.0F;
                for (size_t j = 0; j < S_k; ++j) {
                    row_sum += attention_weights(b, h, i, j) * d_attn_weights(b, h, i, j);
                }
                for (size_t j = 0; j < S_k; ++j) {
                    d_scores(b, h, i, j) = attention_weights(b, h, i, j) * (d_attn_weights(b, h, i, j) - row_sum);
                }
            }
        }
    }

    // dL/dQ = scale * d_scores @ K
    xt::xarray<float> dQ = xt::zeros<float>({B, H, S_q, D});
    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t i = 0; i < S_q; ++i) {
                for (size_t d = 0; d < D; ++d) {
                    float sum = 0.0F;
                    for (size_t j = 0; j < S_k; ++j) {
                        sum += d_scores(b, h, i, j) * key(b, h, j, d);
                    }
                    dQ(b, h, i, d) = sum * scale;
                }
            }
        }
    }

    // dL/dK = scale * d_scores^T @ Q
    xt::xarray<float> dK = xt::zeros<float>({B, H, S_k, D});
    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t j = 0; j < S_k; ++j) {
                for (size_t d = 0; d < D; ++d) {
                    float sum = 0.0F;
                    for (size_t i = 0; i < S_q; ++i) {
                        sum += d_scores(b, h, i, j) * query(b, h, i, d);
                    }
                    dK(b, h, j, d) = sum * scale;
                }
            }
        }
    }

    return {dQ, dK, dV};
}

// Create causal mask for full sequence
// Shape: (1, 1, seq_len, seq_len) - lower triangular
static xt::xarray<float> create_causal_mask(size_t seq_len) {
    xt::xarray<float> mask = xt::zeros<float>({1UL, 1UL, seq_len, seq_len});
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            mask(0, 0, i, j) = 1.0F;
        }
    }
    return mask;
}

static void TestRingAttention(
    const size_t batch,
    const size_t num_heads,
    const size_t seq_len,
    const size_t head_dim,
    const bool use_mask = false,
    const bool test_backward = false,
    const float rtol = 1e-2F,
    const float atol = 5e-1F) {
    using namespace ttml;

    auto* device = &autograd::ctx().get_device();
    auto mesh_shape = device->shape();
    const uint32_t num_devices = mesh_shape.mesh_size();
    const uint32_t mesh_cols = mesh_shape[1];

    const auto& pctx = autograd::ctx().get_parallelism_context();
    const uint32_t cp_axis = pctx.get_cp_axis().value();
    const uint32_t cp_size = pctx.get_cp_size();
    const size_t seq_per_device = seq_len / cp_size;

    auto& rng = autograd::ctx().get_generator();

    // Create full Q, K, V tensors
    std::vector<float> query_data(batch * num_heads * seq_len * head_dim);
    std::vector<float> key_data(batch * num_heads * seq_len * head_dim);
    std::vector<float> value_data(batch * num_heads * seq_len * head_dim);

    auto seed = rng();
    core::parallel_generate(
        std::span{query_data.data(), query_data.size()},
        []() { return std::uniform_real_distribution<float>{0, 2.F}; },
        seed);
    seed = rng();
    core::parallel_generate(
        std::span{key_data.data(), key_data.size()},
        []() { return std::uniform_real_distribution<float>{0, 2.F}; },
        seed);
    seed = rng();
    core::parallel_generate(
        std::span{value_data.data(), value_data.size()},
        []() { return std::uniform_real_distribution<float>{0, 2.F}; },
        seed);

    // Convert to xtensor for reference computation
    xt::xarray<float> query_xt = xt::adapt(query_data, std::vector<size_t>{batch, num_heads, seq_len, head_dim});
    xt::xarray<float> key_xt = xt::adapt(key_data, std::vector<size_t>{batch, num_heads, seq_len, head_dim});
    xt::xarray<float> value_xt = xt::adapt(value_data, std::vector<size_t>{batch, num_heads, seq_len, head_dim});

    // Create reference mask (full causal)
    std::optional<xt::xarray<float>> ref_mask_xt;
    if (use_mask) {
        ref_mask_xt = create_causal_mask(seq_len);
    }
    auto ref_result = reference_sdpa_forward(query_xt, key_xt, value_xt, ref_mask_xt);

    // Create sharded Q, K, V tensors for ring attention
    const auto query_mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, /*dim=*/2, cp_axis);
    const auto key_mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, /*dim=*/2, cp_axis);
    const auto value_mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, /*dim=*/2, cp_axis);

    auto query_tt =
        core::from_xtensor<float, ttnn::DataType::BFLOAT16>(query_xt, device, ttnn::Layout::TILE, query_mapper.get());
    auto key_tt =
        core::from_xtensor<float, ttnn::DataType::BFLOAT16>(key_xt, device, ttnn::Layout::TILE, key_mapper.get());
    auto value_tt =
        core::from_xtensor<float, ttnn::DataType::BFLOAT16>(value_xt, device, ttnn::Layout::TILE, value_mapper.get());

    auto query_tensor = autograd::create_tensor(query_tt);
    auto key_tensor = autograd::create_tensor(key_tt);
    auto value_tensor = autograd::create_tensor(value_tt);

    // Create mask tensor for ring attention
    // Full causal mask (1, 1, S_full, S_full) sharded along Q dim -> (1, 1, S_local, S_full) per device
    std::optional<autograd::TensorPtr> mask_tensor = std::nullopt;
    if (use_mask) {
        xt::xarray<float> full_mask = create_causal_mask(seq_len);
        // Shard along Q dimension (dim 2) so each device gets its Q slice
        const auto mask_mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, /*dim=*/2, cp_axis);
        auto mask_tt = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(
            full_mask, device, ttnn::Layout::TILE, mask_mapper.get());
        mask_tensor = autograd::create_tensor(mask_tt);
    }

    auto output_tensor =
        ops::distributed::ring_attention_sdpa(query_tensor, key_tensor, value_tensor, mask_tensor, use_mask);

    // Gather output from all devices
    auto output_xtensors = core::to_xtensor<float>(output_tensor->get_value(), core::IdentityComposer{});

    xt::xarray<float> gathered_output = xt::zeros<float>({batch, num_heads, seq_len, head_dim});
    for (uint32_t dev = 0; dev < num_devices; ++dev) {
        uint32_t cp_idx = (cp_axis == 0) ? (dev / mesh_cols) : (dev % mesh_cols);
        size_t seq_start = cp_idx * seq_per_device;
        size_t seq_end = seq_start + seq_per_device;

        for (size_t b = 0; b < batch; ++b) {
            for (size_t h = 0; h < num_heads; ++h) {
                for (size_t s = seq_start; s < seq_end; ++s) {
                    for (size_t d = 0; d < head_dim; ++d) {
                        gathered_output(b, h, s, d) = output_xtensors[dev](b, h, s - seq_start, d);
                    }
                }
            }
        }
    }

    EXPECT_TRUE(xt::allclose(ref_result.output, gathered_output, rtol, atol))
        << "Ring attention output does not match reference SDPA output";

    if (test_backward) {
        std::vector<float> grad_output_data(batch * num_heads * seq_len * head_dim);
        seed = rng();
        core::parallel_generate(
            std::span{grad_output_data.data(), grad_output_data.size()},
            []() { return std::uniform_real_distribution<float>{0.F, 2.F}; },
            seed);

        xt::xarray<float> grad_output_xt =
            xt::adapt(grad_output_data, std::vector<size_t>{batch, num_heads, seq_len, head_dim});

        auto ref_grads = reference_sdpa_backward(
            query_xt, key_xt, value_xt, ref_result.attention_weights, grad_output_xt, ref_result.scale);

        const auto grad_mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, /*dim=*/2, cp_axis);
        auto grad_tt = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(
            grad_output_xt, device, ttnn::Layout::TILE, grad_mapper.get());

        output_tensor->set_grad(grad_tt);
        output_tensor->backward();

        auto dQ_xtensors = core::to_xtensor<float>(query_tensor->get_grad(), core::IdentityComposer{});
        auto dK_xtensors = core::to_xtensor<float>(key_tensor->get_grad(), core::IdentityComposer{});
        auto dV_xtensors = core::to_xtensor<float>(value_tensor->get_grad(), core::IdentityComposer{});

        xt::xarray<float> gathered_dQ = xt::zeros<float>({batch, num_heads, seq_len, head_dim});
        xt::xarray<float> gathered_dK = xt::zeros<float>({batch, num_heads, seq_len, head_dim});
        xt::xarray<float> gathered_dV = xt::zeros<float>({batch, num_heads, seq_len, head_dim});

        for (uint32_t dev = 0; dev < num_devices; ++dev) {
            uint32_t cp_idx = (cp_axis == 0) ? (dev / mesh_cols) : (dev % mesh_cols);
            size_t seq_start = cp_idx * seq_per_device;
            size_t seq_end = seq_start + seq_per_device;

            for (size_t b = 0; b < batch; ++b) {
                for (size_t h = 0; h < num_heads; ++h) {
                    for (size_t s = seq_start; s < seq_end; ++s) {
                        for (size_t d = 0; d < head_dim; ++d) {
                            gathered_dQ(b, h, s, d) = dQ_xtensors[dev](b, h, s - seq_start, d);
                            gathered_dK(b, h, s, d) = dK_xtensors[dev](b, h, s - seq_start, d);
                            gathered_dV(b, h, s, d) = dV_xtensors[dev](b, h, s - seq_start, d);
                        }
                    }
                }
            }
        }

        EXPECT_TRUE(xt::allclose(ref_grads.dQ, gathered_dQ, rtol, atol))
            << "Ring attention dQ gradient does not match reference";
        EXPECT_TRUE(xt::allclose(ref_grads.dK, gathered_dK, rtol, atol))
            << "Ring attention dK gradient does not match reference";
        EXPECT_TRUE(xt::allclose(ref_grads.dV, gathered_dV, rtol, atol))
            << "Ring attention dV gradient does not match reference";
    }
}

// Basic ring attention test without mask (forward only)
TEST_F(GalaxyRingAttentionTest, BasicNoMask) {
    TestRingAttention(
        /*batch=*/1,
        /*num_heads=*/4,
        /*seq_len=*/128,  // 32 per device (4 CP devices along axis 0)
        /*head_dim=*/64,
        /*use_mask=*/false,
        /*test_backward=*/false);
}

// Basic ring attention test without mask (forward only)
TEST_F(GalaxyRingAttentionTest, CacheTest) {
    TestRingAttention(
        /*batch=*/1,
        /*num_heads=*/4,
        /*seq_len=*/128,  // 32 per device (4 CP devices along axis 0)
        /*head_dim=*/64,
        /*use_mask=*/false,
        /*test_backward=*/false);
}

// Basic ring attention test with backward pass
TEST_F(GalaxyRingAttentionTest, BasicWithBackward) {
    TestRingAttention(
        /*batch=*/1,
        /*num_heads=*/4,
        /*seq_len=*/128,
        /*head_dim=*/64,
        /*use_mask=*/false,
        /*test_backward=*/true);
}

// Ring attention with causal mask (forward only)
TEST_F(GalaxyRingAttentionTest, WithCausalMask) {
    TestRingAttention(
        /*batch=*/1,
        /*num_heads=*/4,
        /*seq_len=*/128,
        /*head_dim=*/64,
        /*use_mask=*/true,
        /*test_backward=*/false);
}

// Ring attention with causal mask and backward pass
TEST_F(GalaxyRingAttentionTest, WithCausalMaskBackward) {
    TestRingAttention(
        /*batch=*/1,
        /*num_heads=*/4,
        /*seq_len=*/128,
        /*head_dim=*/64,
        /*use_mask=*/true,
        /*test_backward=*/true);
}

// Larger batch size test
TEST_F(GalaxyRingAttentionTest, LargerBatch) {
    TestRingAttention(
        /*batch=*/4,
        /*num_heads=*/8,
        /*seq_len=*/128,
        /*head_dim=*/64,
        /*use_mask=*/false,
        /*test_backward=*/false);
}

// Test with larger sequence
TEST_F(GalaxyRingAttentionTest, LargerSequence) {
    TestRingAttention(
        /*batch=*/2,
        /*num_heads=*/4,
        /*seq_len=*/256,  // 64 per device
        /*head_dim=*/64,
        /*use_mask=*/false,
        /*test_backward=*/false);
}

// Larger batch with causal mask
TEST_F(GalaxyRingAttentionTest, LargerBatchWithCausalMask) {
    TestRingAttention(
        /*batch=*/4,
        /*num_heads=*/8,
        /*seq_len=*/128,
        /*head_dim=*/64,
        /*use_mask=*/true,
        /*test_backward=*/false);
}

// Larger sequence with causal mask
TEST_F(GalaxyRingAttentionTest, LargerSequenceWithCausalMask) {
    TestRingAttention(
        /*batch=*/2,
        /*num_heads=*/4,
        /*seq_len=*/256,
        /*head_dim=*/64,
        /*use_mask=*/true,
        /*test_backward=*/false);
}

// Full test: larger batch with causal mask and backward
TEST_F(GalaxyRingAttentionTest, LargerBatchCausalMaskBackward) {
    TestRingAttention(
        /*batch=*/4,
        /*num_heads=*/8,
        /*seq_len=*/128,
        /*head_dim=*/64,
        /*use_mask=*/true,
        /*test_backward=*/true);
}

// Full test: larger batch with causal mask and backward
TEST_F(GalaxyRingAttentionTest, CacheTestBackward) {
    TestRingAttention(
        /*batch=*/4,
        /*num_heads=*/8,
        /*seq_len=*/128,
        /*head_dim=*/64,
        /*use_mask=*/true,
        /*test_backward=*/true);
}

// Full test: larger sequence with causal mask and backward
TEST_F(GalaxyRingAttentionTest, LargerSequenceCausalMaskBackward) {
    TestRingAttention(
        /*batch=*/2,
        /*num_heads=*/4,
        /*seq_len=*/256,
        /*head_dim=*/64,
        /*use_mask=*/true,
        /*test_backward=*/true);
}
