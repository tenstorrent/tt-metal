// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <umd/device/cluster.hpp>

#include "autograd/auto_context.hpp"
#include "core/system_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/distributed/losses.hpp"
#include "ops/losses.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn_fixed/distributed/tt_metal.hpp"

namespace {

auto check_board_is_n300() {
    return tt::umd::Cluster::create_cluster_descriptor()->get_board_type(0) == tt::BoardType::N300;
}

// Reference: standard cross-entropy loss = mean_over_positions( log_normalizer − target_logit )
//   log_normalizer = global_max + log(sum(exp(x − global_max)))
//   target_logit   = x[row, target[row]]
float cross_entropy_loss_reference(const xt::xarray<float>& logits, const xt::xarray<uint32_t>& targets) {
    const uint32_t B = static_cast<uint32_t>(logits.shape(0));
    const uint32_t S = static_cast<uint32_t>(logits.shape(2));
    const uint32_t V = static_cast<uint32_t>(logits.shape(3));
    const uint32_t N = B * S;

    float total = 0.0F;
    for (uint32_t b = 0; b < B; ++b) {
        for (uint32_t s = 0; s < S; ++s) {
            float row_max = -std::numeric_limits<float>::infinity();
            for (uint32_t v = 0; v < V; ++v) {
                row_max = std::max(row_max, logits(b, 0U, s, v));
            }
            float exp_sum = 0.0F;
            for (uint32_t v = 0; v < V; ++v) {
                exp_sum += std::exp(logits(b, 0U, s, v) - row_max);
            }
            float log_norm = row_max + std::log(exp_sum);
            float target_logit = logits(b, 0U, s, targets(b, s));
            total += (log_norm - target_logit);
        }
    }
    return total / static_cast<float>(N);
}

// Reference backward: dL/dx = (softmax − one_hot) / N
xt::xarray<float> cross_entropy_grad_reference(
    const xt::xarray<float>& logits, const xt::xarray<uint32_t>& targets, float grad_scale = 1.0F) {
    const uint32_t B = static_cast<uint32_t>(logits.shape(0));
    const uint32_t S = static_cast<uint32_t>(logits.shape(2));
    const uint32_t V = static_cast<uint32_t>(logits.shape(3));
    const uint32_t N = B * S;

    xt::xarray<float> grad = xt::zeros<float>(logits.shape());

    for (uint32_t b = 0; b < B; ++b) {
        for (uint32_t s = 0; s < S; ++s) {
            float row_max = -std::numeric_limits<float>::infinity();
            for (uint32_t v = 0; v < V; ++v) {
                row_max = std::max(row_max, logits(b, 0U, s, v));
            }
            float exp_sum = 0.0F;
            for (uint32_t v = 0; v < V; ++v) {
                exp_sum += std::exp(logits(b, 0U, s, v) - row_max);
            }
            for (uint32_t v = 0; v < V; ++v) {
                float sm = std::exp(logits(b, 0U, s, v) - row_max) / exp_sum;
                float oh = (v == targets(b, s)) ? 1.0F : 0.0F;
                grad(b, 0U, s, v) = (sm - oh) / static_cast<float>(N) * grad_scale;
            }
        }
    }
    return grad;
}

}  // namespace

class ShardedCrossEntropyLossTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!check_board_is_n300()) {
            GTEST_SKIP() << "Skipping N300 specific tests";
        }
        ttml::ttnn_fixed::distributed::enable_fabric(2U);
        ttml::autograd::ctx().open_device(tt::tt_metal::distributed::MeshShape(1, 2));
        ttml::autograd::ctx().set_seed(42);
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(ShardedCrossEntropyLossTest, ForwardSmall) {
    SKIP_FOR_WATCHER();

    using namespace ttml;

    auto* device = &autograd::ctx().get_device();
    const uint32_t B = 1U, S = 1U;
    const uint32_t local_V = 64U;
    const uint32_t full_V = local_V * 2U;

    xt::xarray<float> logits_xt = xt::empty<float>({B, 1U, S, full_V});
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-2.0F, 2.0F);
    for (auto& v : logits_xt) {
        v = dist(gen);
    }

    xt::xarray<uint32_t> targets_xt = xt::zeros<uint32_t>({B, S});
    targets_xt(0, 0) = 3U;

    auto shard_mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, 3);
    auto logits_dev =
        core::from_xtensor<float, ttnn::DataType::BFLOAT16>(logits_xt, device, ttnn::Layout::TILE, shard_mapper.get());

    auto replicate_mapper = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
    auto targets_dev = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        targets_xt, device, ttnn::Layout::ROW_MAJOR, replicate_mapper.get());

    auto logits_ptr = autograd::create_tensor(logits_dev, true);
    auto targets_ptr = autograd::create_tensor(targets_dev, false);

    auto loss = ops::distributed::vocab_parallel_cross_entropy_loss(logits_ptr, targets_ptr);

    auto loss_xt = core::to_xtensor<float>(loss->get_value(), core::IdentityComposer{});
    float expected_loss = cross_entropy_loss_reference(logits_xt, targets_xt);

    EXPECT_NEAR(loss_xt[0](0, 0, 0, 0), expected_loss, 5e-2F);
    EXPECT_NEAR(loss_xt[1](0, 0, 0, 0), expected_loss, 5e-2F);
}

TEST_F(ShardedCrossEntropyLossTest, BackwardSmall) {
    SKIP_FOR_WATCHER();

    using namespace ttml;

    auto* device = &autograd::ctx().get_device();
    const uint32_t B = 1U, S = 1U;
    const uint32_t local_V = 64U;
    const uint32_t full_V = local_V * 2U;

    xt::xarray<float> logits_xt = xt::empty<float>({B, 1U, S, full_V});
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-2.0F, 2.0F);
    for (auto& v : logits_xt) {
        v = dist(gen);
    }

    xt::xarray<uint32_t> targets_xt = xt::zeros<uint32_t>({B, S});
    targets_xt(0, 0) = 3U;

    auto shard_mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, 3);
    auto logits_dev =
        core::from_xtensor<float, ttnn::DataType::BFLOAT16>(logits_xt, device, ttnn::Layout::TILE, shard_mapper.get());

    auto replicate_mapper = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
    auto targets_dev = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        targets_xt, device, ttnn::Layout::ROW_MAJOR, replicate_mapper.get());

    auto logits_ptr = autograd::create_tensor(logits_dev, true);
    auto targets_ptr = autograd::create_tensor(targets_dev, false);

    auto loss = ops::distributed::vocab_parallel_cross_entropy_loss(logits_ptr, targets_ptr);

    xt::xarray<float> grad_ones = xt::ones<float>({B, 1U, S, 1U});
    auto grad_dev = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(
        grad_ones, device, ttnn::Layout::TILE, replicate_mapper.get());
    loss->set_grad(grad_dev);
    loss->backward();

    ASSERT_TRUE(core::is_tensor_initialized(logits_ptr->get_grad()));

    auto grad_xtensors = core::to_xtensor<float>(logits_ptr->get_grad(), core::IdentityComposer{});
    auto expected_grad = cross_entropy_grad_reference(logits_xt, targets_xt);

    auto expected_shard0 = xt::view(expected_grad, xt::all(), xt::all(), xt::all(), xt::range(0, local_V));
    auto expected_shard1 = xt::view(expected_grad, xt::all(), xt::all(), xt::all(), xt::range(local_V, full_V));

    EXPECT_TRUE(xt::allclose(grad_xtensors[0], expected_shard0, 3e-2F, 1e-2F));
    EXPECT_TRUE(xt::allclose(grad_xtensors[1], expected_shard1, 3e-2F, 1e-2F));
}

TEST_F(ShardedCrossEntropyLossTest, BackwardBatch) {
    SKIP_FOR_WATCHER();

    using namespace ttml;

    auto* device = &autograd::ctx().get_device();
    const uint32_t B = 2U, S = 32U;
    const uint32_t local_V = 128U;
    const uint32_t full_V = local_V * 2U;

    std::mt19937 gen(7);
    xt::xarray<float> logits_xt = xt::empty<float>({B, 1U, S, full_V});
    std::uniform_real_distribution<float> dist(-5.0F, 5.0F);
    for (auto& v : logits_xt) {
        v = dist(gen);
    }

    xt::xarray<uint32_t> targets_xt = xt::zeros<uint32_t>({B, S});
    std::uniform_int_distribution<uint32_t> idx_dist(0U, full_V - 1U);
    for (uint32_t b = 0; b < B; ++b) {
        for (uint32_t s = 0; s < S; ++s) {
            targets_xt(b, s) = idx_dist(gen);
        }
    }

    auto shard_mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, 3);
    auto logits_dev =
        core::from_xtensor<float, ttnn::DataType::BFLOAT16>(logits_xt, device, ttnn::Layout::TILE, shard_mapper.get());

    auto replicate_mapper = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
    auto targets_dev = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        targets_xt, device, ttnn::Layout::ROW_MAJOR, replicate_mapper.get());

    auto logits_ptr = autograd::create_tensor(logits_dev, true);
    auto targets_ptr = autograd::create_tensor(targets_dev, false);

    auto loss = ops::distributed::vocab_parallel_cross_entropy_loss(logits_ptr, targets_ptr);

    auto loss_xt = core::to_xtensor<float>(loss->get_value(), core::IdentityComposer{});
    float expected_loss = cross_entropy_loss_reference(logits_xt, targets_xt);
    EXPECT_NEAR(loss_xt[0](0, 0, 0, 0), expected_loss, 5e-2F);

    xt::xarray<float> grad_ones = xt::ones<float>({1U, 1U, 1U, 1U});
    auto grad_dev = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(
        grad_ones, device, ttnn::Layout::TILE, replicate_mapper.get());
    loss->set_grad(grad_dev);
    loss->backward();

    ASSERT_TRUE(core::is_tensor_initialized(logits_ptr->get_grad()));

    auto grad_xtensors = core::to_xtensor<float>(logits_ptr->get_grad(), core::IdentityComposer{});
    auto expected_grad = cross_entropy_grad_reference(logits_xt, targets_xt);

    auto expected_shard0 = xt::view(expected_grad, xt::all(), xt::all(), xt::all(), xt::range(0, local_V));
    auto expected_shard1 = xt::view(expected_grad, xt::all(), xt::all(), xt::all(), xt::range(local_V, full_V));

    EXPECT_TRUE(xt::allclose(grad_xtensors[0], expected_shard0, 3e-2F, 1e-2F));
    EXPECT_TRUE(xt::allclose(grad_xtensors[1], expected_shard1, 3e-2F, 1e-2F));
}

TEST_F(ShardedCrossEntropyLossTest, BackwardTargetOnSecondShard) {
    SKIP_FOR_WATCHER();

    using namespace ttml;

    auto* device = &autograd::ctx().get_device();
    const uint32_t B = 1U, S = 1U;
    const uint32_t local_V = 64U;
    const uint32_t full_V = local_V * 2U;

    xt::xarray<float> logits_xt = xt::empty<float>({B, 1U, S, full_V});
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    for (auto& v : logits_xt) {
        v = dist(gen);
    }

    xt::xarray<uint32_t> targets_xt = xt::zeros<uint32_t>({B, S});
    targets_xt(0, 0) = local_V + 5U;

    auto shard_mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, 3);
    auto logits_dev =
        core::from_xtensor<float, ttnn::DataType::BFLOAT16>(logits_xt, device, ttnn::Layout::TILE, shard_mapper.get());

    auto replicate_mapper = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
    auto targets_dev = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        targets_xt, device, ttnn::Layout::ROW_MAJOR, replicate_mapper.get());

    auto logits_ptr = autograd::create_tensor(logits_dev, true);
    auto targets_ptr = autograd::create_tensor(targets_dev, false);

    auto loss = ops::distributed::vocab_parallel_cross_entropy_loss(logits_ptr, targets_ptr);

    xt::xarray<float> grad_ones = xt::ones<float>({B, 1U, S, 1U});
    auto grad_dev = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(
        grad_ones, device, ttnn::Layout::TILE, replicate_mapper.get());
    loss->set_grad(grad_dev);
    loss->backward();

    ASSERT_TRUE(core::is_tensor_initialized(logits_ptr->get_grad()));

    auto grad_xtensors = core::to_xtensor<float>(logits_ptr->get_grad(), core::IdentityComposer{});
    auto expected_grad = cross_entropy_grad_reference(logits_xt, targets_xt);

    auto expected_shard0 = xt::view(expected_grad, xt::all(), xt::all(), xt::all(), xt::range(0, local_V));
    auto expected_shard1 = xt::view(expected_grad, xt::all(), xt::all(), xt::all(), xt::range(local_V, full_V));

    // Shard 0 should have pure softmax (no one_hot subtraction since target is on shard 1)
    EXPECT_TRUE(xt::allclose(grad_xtensors[0], expected_shard0, 3e-2F, 1e-2F));
    // Shard 1 should have softmax - one_hot at position 5
    EXPECT_TRUE(xt::allclose(grad_xtensors[1], expected_shard1, 3e-2F, 1e-2F));
}

TEST_F(ShardedCrossEntropyLossTest, NonShardedBackwardMatchesReference) {
    SKIP_FOR_WATCHER();

    using namespace ttml;

    auto* device = &autograd::ctx().get_device();
    const uint32_t B = 2U, S = 32U;
    const uint32_t local_V = 128U;
    const uint32_t full_V = local_V * 2U;

    std::mt19937 gen(55);
    xt::xarray<float> logits_xt = xt::empty<float>({B, 1U, S, full_V});
    std::uniform_real_distribution<float> dist(-4.0F, 4.0F);
    for (auto& v : logits_xt) {
        v = dist(gen);
    }

    xt::xarray<uint32_t> targets_xt = xt::zeros<uint32_t>({B, S});
    std::uniform_int_distribution<uint32_t> idx_dist(0U, full_V - 1U);
    for (uint32_t b = 0; b < B; ++b) {
        for (uint32_t s = 0; s < S; ++s) {
            targets_xt(b, s) = idx_dist(gen);
        }
    }

    auto replicate_mapper = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
    auto logits_dev = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(
        logits_xt, device, ttnn::Layout::TILE, replicate_mapper.get());
    auto targets_dev = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        targets_xt, device, ttnn::Layout::ROW_MAJOR, replicate_mapper.get());

    auto logits_ptr = autograd::create_tensor(logits_dev, true);
    auto targets_ptr = autograd::create_tensor(targets_dev, false);

    auto loss = ops::cross_entropy_loss(logits_ptr, targets_ptr, ops::ReduceType::MEAN);

    float expected_loss = cross_entropy_loss_reference(logits_xt, targets_xt);
    auto loss_xt = core::to_xtensor<float>(loss->get_value(), core::IdentityComposer{});
    EXPECT_NEAR(loss_xt[0](0, 0, 0, 0), expected_loss, 5e-2F);

    xt::xarray<float> grad_ones = xt::ones<float>({1U, 1U, 1U, 1U});
    auto grad_dev = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(
        grad_ones, device, ttnn::Layout::TILE, replicate_mapper.get());
    loss->set_grad(grad_dev);
    loss->backward();

    ASSERT_TRUE(core::is_tensor_initialized(logits_ptr->get_grad()));

    auto grad_xtensors = core::to_xtensor<float>(logits_ptr->get_grad(), core::IdentityComposer{});
    auto expected_grad = cross_entropy_grad_reference(logits_xt, targets_xt);

    EXPECT_TRUE(xt::allclose(grad_xtensors[0], expected_grad, 3e-2F, 1e-2F));
}

TEST_F(ShardedCrossEntropyLossTest, MatchesNonShardedImplementation) {
    SKIP_FOR_WATCHER();

    using namespace ttml;

    auto* device = &autograd::ctx().get_device();
    const uint32_t B = 2U, S = 32U;
    const uint32_t local_V = 128U;
    const uint32_t full_V = local_V * 2U;

    std::mt19937 gen(99);
    xt::xarray<float> logits_xt = xt::empty<float>({B, 1U, S, full_V});
    std::uniform_real_distribution<float> dist(-3.0F, 3.0F);
    for (auto& v : logits_xt) {
        v = dist(gen);
    }

    xt::xarray<uint32_t> targets_xt = xt::zeros<uint32_t>({B, S});
    std::uniform_int_distribution<uint32_t> idx_dist(0U, full_V - 1U);
    for (uint32_t b = 0; b < B; ++b) {
        for (uint32_t s = 0; s < S; ++s) {
            targets_xt(b, s) = idx_dist(gen);
        }
    }

    auto replicate_mapper = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);

    // --- Non-sharded: replicate full logits to all devices ---
    auto full_logits_dev = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(
        logits_xt, device, ttnn::Layout::TILE, replicate_mapper.get());
    auto full_targets_dev = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        targets_xt, device, ttnn::Layout::ROW_MAJOR, replicate_mapper.get());

    auto full_logits_ptr = autograd::create_tensor(full_logits_dev, true);
    auto full_targets_ptr = autograd::create_tensor(full_targets_dev, false);

    auto ref_loss = ops::cross_entropy_loss(full_logits_ptr, full_targets_ptr, ops::ReduceType::MEAN);

    // --- Sharded: shard logits along vocab dim ---
    auto shard_mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, 3);
    auto sharded_logits_dev =
        core::from_xtensor<float, ttnn::DataType::BFLOAT16>(logits_xt, device, ttnn::Layout::TILE, shard_mapper.get());
    auto sharded_targets_dev = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        targets_xt, device, ttnn::Layout::ROW_MAJOR, replicate_mapper.get());

    auto sharded_logits_ptr = autograd::create_tensor(sharded_logits_dev, true);
    auto sharded_targets_ptr = autograd::create_tensor(sharded_targets_dev, false);

    auto sharded_loss = ops::distributed::vocab_parallel_cross_entropy_loss(sharded_logits_ptr, sharded_targets_ptr);

    // --- Compare forward loss ---
    auto ref_loss_xt = core::to_xtensor<float>(ref_loss->get_value(), core::IdentityComposer{});
    auto sharded_loss_xt = core::to_xtensor<float>(sharded_loss->get_value(), core::IdentityComposer{});

    float ref_val = ref_loss_xt[0](0, 0, 0, 0);
    float sharded_val = sharded_loss_xt[0](0, 0, 0, 0);
    EXPECT_NEAR(sharded_val, ref_val, 5e-2F);

    // --- Compare backward gradients ---
    xt::xarray<float> grad_ones = xt::ones<float>({1U, 1U, 1U, 1U});
    auto ref_grad_dev = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(
        grad_ones, device, ttnn::Layout::TILE, replicate_mapper.get());
    ref_loss->set_grad(ref_grad_dev);
    ref_loss->backward();

    auto sharded_grad_dev = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(
        grad_ones, device, ttnn::Layout::TILE, replicate_mapper.get());
    sharded_loss->set_grad(sharded_grad_dev);
    sharded_loss->backward();

    ASSERT_TRUE(core::is_tensor_initialized(full_logits_ptr->get_grad()));
    ASSERT_TRUE(core::is_tensor_initialized(sharded_logits_ptr->get_grad()));

    auto ref_grad_xt = core::to_xtensor<float>(full_logits_ptr->get_grad(), core::IdentityComposer{});
    auto sharded_grad_xt = core::to_xtensor<float>(sharded_logits_ptr->get_grad(), core::IdentityComposer{});

    // Non-sharded grad is replicated [B,1,S,full_V] on each device; compare with sharded halves
    auto ref_shard0 = xt::view(ref_grad_xt[0], xt::all(), xt::all(), xt::all(), xt::range(0, local_V));
    auto ref_shard1 = xt::view(ref_grad_xt[0], xt::all(), xt::all(), xt::all(), xt::range(local_V, full_V));

    EXPECT_TRUE(xt::allclose(sharded_grad_xt[0], ref_shard0, 3e-2F, 1e-2F));
    EXPECT_TRUE(xt::allclose(sharded_grad_xt[1], ref_shard1, 3e-2F, 1e-2F));
}
