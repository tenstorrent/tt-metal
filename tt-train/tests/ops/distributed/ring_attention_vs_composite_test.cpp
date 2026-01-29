// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Test for ring_attention_sdpa against composite scaled_dot_product_attention.
 *
 * This test compares ring attention (distributed across CP devices) against
 * the composite SDPA implementation running on a single device. Both should
 * produce identical results for forward and backward passes.
 */

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>
#include <core/xtensor_utils.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <umd/device/cluster.hpp>

#include "autograd/auto_context.hpp"
#include "core/distributed/socket_manager.hpp"
#include "core/random.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/distributed/ring_attention.hpp"
#include "ops/scaled_dot_product_attention.hpp"
#include "ttnn_fixed/distributed/tt_metal.hpp"

static bool check_32_chips() {
    auto cluster_desc = tt::umd::Cluster::create_cluster_descriptor();
    auto all_chips = cluster_desc->get_all_chips();
    return all_chips.size() == 32;
}

class RingAttentionVsCompositeTest : public ::testing::Test {
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
        auto* device = &ttml::autograd::ctx().get_device();
        ttml::autograd::ctx().get_parallelism_context().configure(
            device, /*enable_dp=*/false, /*enable_tp=*/false, /*enable_cp=*/true);
    }

    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }
};

static void TestRingAttentionVsComposite(
    const size_t batch,
    const size_t num_heads,
    const size_t seq_len,
    const size_t head_dim,
    const bool test_backward = false,
    const float rtol = 5e-2F,
    const float atol = 8e-1F) {
    using namespace ttml;

    // Get device and parallelism info
    auto* device = &autograd::ctx().get_device();
    auto mesh_shape = device->shape();
    const uint32_t num_devices = mesh_shape.mesh_size();
    const uint32_t mesh_cols = mesh_shape[1];

    const auto& pctx = autograd::ctx().get_parallelism_context();
    const uint32_t cp_axis = pctx.get_cp_axis().value();
    const uint32_t cp_size = pctx.get_cp_size();

    auto& rng = autograd::ctx().get_generator();

    // Generate random Q, K, V data
    std::vector<float> query_data(batch * num_heads * seq_len * head_dim);
    std::vector<float> key_data(batch * num_heads * seq_len * head_dim);
    std::vector<float> value_data(batch * num_heads * seq_len * head_dim);

    auto seed = rng();
    core::parallel_generate(
        std::span{query_data.data(), query_data.size()},
        []() { return std::uniform_real_distribution<float>{0.0F, 2.0F}; },
        seed);
    seed = rng();
    core::parallel_generate(
        std::span{key_data.data(), key_data.size()},
        []() { return std::uniform_real_distribution<float>{0.0F, 2.0F}; },
        seed);
    seed = rng();
    core::parallel_generate(
        std::span{value_data.data(), value_data.size()},
        []() { return std::uniform_real_distribution<float>{0.0F, 2.0F}; },
        seed);

    // Convert to xtensor
    xt::xarray<float> query_xt = xt::adapt(query_data, std::vector<size_t>{batch, num_heads, seq_len, head_dim});
    xt::xarray<float> key_xt = xt::adapt(key_data, std::vector<size_t>{batch, num_heads, seq_len, head_dim});
    xt::xarray<float> value_xt = xt::adapt(value_data, std::vector<size_t>{batch, num_heads, seq_len, head_dim});

    // ========== Run composite SDPA on full sequence (replicated on all devices) ==========
    // Create replicated tensors (same data on all devices)
    const auto replicate_mapper = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
    auto query_full_tt = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(
        query_xt, device, ttnn::Layout::TILE, replicate_mapper.get());
    auto key_full_tt =
        core::from_xtensor<float, ttnn::DataType::BFLOAT16>(key_xt, device, ttnn::Layout::TILE, replicate_mapper.get());
    auto value_full_tt = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(
        value_xt, device, ttnn::Layout::TILE, replicate_mapper.get());

    auto query_full = autograd::create_tensor(query_full_tt);
    auto key_full = autograd::create_tensor(key_full_tt);
    auto value_full = autograd::create_tensor(value_full_tt);

    // Run composite SDPA
    auto composite_output = ops::scaled_dot_product_attention(query_full, key_full, value_full, std::nullopt);

    // Get composite output (same on all devices, just take first)
    auto composite_xtensors = core::to_xtensor<float>(composite_output->get_value(), core::IdentityComposer{});
    xt::xarray<float> composite_result = composite_xtensors[0];  // All devices have same result

    // ========== Run ring attention SDPA with sequence sharding ==========
    // Create sharded tensors along sequence dimension
    const auto shard_mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, /*dim=*/2, cp_axis);
    auto query_shard_tt =
        core::from_xtensor<float, ttnn::DataType::BFLOAT16>(query_xt, device, ttnn::Layout::TILE, shard_mapper.get());
    auto key_shard_tt =
        core::from_xtensor<float, ttnn::DataType::BFLOAT16>(key_xt, device, ttnn::Layout::TILE, shard_mapper.get());
    auto value_shard_tt =
        core::from_xtensor<float, ttnn::DataType::BFLOAT16>(value_xt, device, ttnn::Layout::TILE, shard_mapper.get());

    auto query_shard = autograd::create_tensor(query_shard_tt);
    auto key_shard = autograd::create_tensor(key_shard_tt);
    auto value_shard = autograd::create_tensor(value_shard_tt);

    // Run ring attention
    auto ring_output = ops::distributed::ring_attention_sdpa(query_shard, key_shard, value_shard, std::nullopt);

    // Gather ring attention output from all devices
    auto ring_xtensors = core::to_xtensor<float>(ring_output->get_value(), core::IdentityComposer{});

    // Reconstruct full sequence from shards
    const size_t seq_per_device = seq_len / cp_size;
    xt::xarray<float> ring_result = xt::zeros<float>({batch, num_heads, seq_len, head_dim});

    for (uint32_t dev = 0; dev < num_devices; ++dev) {
        uint32_t cp_idx = (cp_axis == 0) ? (dev / mesh_cols) : (dev % mesh_cols);
        size_t seq_start = cp_idx * seq_per_device;

        for (size_t b = 0; b < batch; ++b) {
            for (size_t h = 0; h < num_heads; ++h) {
                for (size_t s = 0; s < seq_per_device; ++s) {
                    for (size_t d = 0; d < head_dim; ++d) {
                        ring_result(b, h, seq_start + s, d) = ring_xtensors[dev](b, h, s, d);
                    }
                }
            }
        }
    }

    // ========== Compare forward outputs ==========
    EXPECT_TRUE(xt::allclose(composite_result, ring_result, rtol, atol))
        << "Ring attention output does not match composite SDPA output";

    // ========== Test backward pass if requested ==========
    if (test_backward) {
        // Generate random output gradient
        std::vector<float> grad_output_data(batch * num_heads * seq_len * head_dim);
        seed = rng();
        core::parallel_generate(
            std::span{grad_output_data.data(), grad_output_data.size()},
            []() { return std::uniform_real_distribution<float>{0.0F, 2.0F}; },
            seed);

        xt::xarray<float> grad_output_xt =
            xt::adapt(grad_output_data, std::vector<size_t>{batch, num_heads, seq_len, head_dim});

        // Run backward on composite SDPA (replicated gradient)
        auto grad_full_tt = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(
            grad_output_xt, device, ttnn::Layout::TILE, replicate_mapper.get());
        composite_output->set_grad(grad_full_tt);
        composite_output->backward();

        // Get composite gradients (same on all devices)
        auto composite_dQ_xtensors = core::to_xtensor<float>(query_full->get_grad(), core::IdentityComposer{});
        auto composite_dK_xtensors = core::to_xtensor<float>(key_full->get_grad(), core::IdentityComposer{});
        auto composite_dV_xtensors = core::to_xtensor<float>(value_full->get_grad(), core::IdentityComposer{});

        xt::xarray<float> composite_dQ = composite_dQ_xtensors[0];
        xt::xarray<float> composite_dK = composite_dK_xtensors[0];
        xt::xarray<float> composite_dV = composite_dV_xtensors[0];

        // Run backward on ring attention (sharded gradient)
        auto grad_shard_tt = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(
            grad_output_xt, device, ttnn::Layout::TILE, shard_mapper.get());
        ring_output->set_grad(grad_shard_tt);
        ring_output->backward();

        // Gather ring attention gradients
        auto ring_dQ_xtensors = core::to_xtensor<float>(query_shard->get_grad(), core::IdentityComposer{});
        auto ring_dK_xtensors = core::to_xtensor<float>(key_shard->get_grad(), core::IdentityComposer{});
        auto ring_dV_xtensors = core::to_xtensor<float>(value_shard->get_grad(), core::IdentityComposer{});

        // Reconstruct full gradients from shards
        xt::xarray<float> ring_dQ = xt::zeros<float>({batch, num_heads, seq_len, head_dim});
        xt::xarray<float> ring_dK = xt::zeros<float>({batch, num_heads, seq_len, head_dim});
        xt::xarray<float> ring_dV = xt::zeros<float>({batch, num_heads, seq_len, head_dim});

        for (uint32_t dev = 0; dev < num_devices; ++dev) {
            uint32_t cp_idx = (cp_axis == 0) ? (dev / mesh_cols) : (dev % mesh_cols);
            size_t seq_start = cp_idx * seq_per_device;

            for (size_t b = 0; b < batch; ++b) {
                for (size_t h = 0; h < num_heads; ++h) {
                    for (size_t s = 0; s < seq_per_device; ++s) {
                        for (size_t d = 0; d < head_dim; ++d) {
                            ring_dQ(b, h, seq_start + s, d) = ring_dQ_xtensors[dev](b, h, s, d);
                            ring_dK(b, h, seq_start + s, d) = ring_dK_xtensors[dev](b, h, s, d);
                            ring_dV(b, h, seq_start + s, d) = ring_dV_xtensors[dev](b, h, s, d);
                        }
                    }
                }
            }
        }

        std::cout << "composite_dQ: " << composite_dQ << std::endl;
        std::cout << "ring_dQ: " << ring_dQ << std::endl;
        std::cout << "composite_dK: " << composite_dK << std::endl;
        std::cout << "ring_dK: " << ring_dK << std::endl;
        std::cout << "composite_dV: " << composite_dV << std::endl;
        std::cout << "ring_dV: " << ring_dV << std::endl;
        std::cout << xt::amax(xt::abs(composite_dQ - ring_dQ)) << std::endl;
        std::cout << xt::amax(xt::abs(composite_dK - ring_dK)) << std::endl;
        std::cout << xt::amax(xt::abs(composite_dV - ring_dV)) << std::endl;

        // Compare gradients
        EXPECT_TRUE(xt::allclose(composite_dQ, ring_dQ, rtol, atol))
            << "Ring attention dQ gradient does not match composite SDPA";
        EXPECT_TRUE(xt::allclose(composite_dK, ring_dK, rtol, atol))
            << "Ring attention dK gradient does not match composite SDPA";
        EXPECT_TRUE(xt::allclose(composite_dV, ring_dV, rtol, atol))
            << "Ring attention dV gradient does not match composite SDPA";
    }
}

// ========== Test Cases ==========

// Basic forward-only test
TEST_F(RingAttentionVsCompositeTest, BasicForward) {
    TestRingAttentionVsComposite(
        /*batch=*/1,
        /*num_heads=*/4,
        /*seq_len=*/128,  // 32 per device (4 CP devices)
        /*head_dim=*/64,
        /*test_backward=*/false);
}

// Forward + backward test
TEST_F(RingAttentionVsCompositeTest, BasicWithBackward) {
    TestRingAttentionVsComposite(
        /*batch=*/1,
        /*num_heads=*/4,
        /*seq_len=*/128,
        /*head_dim=*/64,
        /*test_backward=*/true);
}

// Larger batch
TEST_F(RingAttentionVsCompositeTest, LargerBatch) {
    TestRingAttentionVsComposite(
        /*batch=*/2,
        /*num_heads=*/4,
        /*seq_len=*/128,
        /*head_dim=*/64,
        /*test_backward=*/false);
}

// Larger batch with backward
TEST_F(RingAttentionVsCompositeTest, LargerBatchWithBackward) {
    TestRingAttentionVsComposite(
        /*batch=*/2,
        /*num_heads=*/4,
        /*seq_len=*/128,
        /*head_dim=*/64,
        /*test_backward=*/true);
}

// More heads
TEST_F(RingAttentionVsCompositeTest, MoreHeads) {
    TestRingAttentionVsComposite(
        /*batch=*/1,
        /*num_heads=*/8,
        /*seq_len=*/128,
        /*head_dim=*/64,
        /*test_backward=*/false);
}

// More heads with backward
TEST_F(RingAttentionVsCompositeTest, MoreHeadsWithBackward) {
    TestRingAttentionVsComposite(
        /*batch=*/1,
        /*num_heads=*/8,
        /*seq_len=*/128,
        /*head_dim=*/64,
        /*test_backward=*/true);
}

// Larger sequence
TEST_F(RingAttentionVsCompositeTest, LargerSequence) {
    TestRingAttentionVsComposite(
        /*batch=*/1,
        /*num_heads=*/4,
        /*seq_len=*/256,  // 64 per device
        /*head_dim=*/64,
        /*test_backward=*/false);
}

// Larger sequence with backward
TEST_F(RingAttentionVsCompositeTest, LargerSequenceWithBackward) {
    TestRingAttentionVsComposite(
        /*batch=*/1,
        /*num_heads=*/4,
        /*seq_len=*/256,
        /*head_dim=*/64,
        /*test_backward=*/true);
}
