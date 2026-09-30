// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "optimizers/adamw_composite.hpp"

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <array>
#include <core/ttnn_all_includes.hpp>
#include <tt-metalium/host_api.hpp>
#include <variant>

#include "autograd/auto_context.hpp"
#include "core/system_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "fmt/base.h"
#include "modules/linear_module.hpp"
#include "ops/losses.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn_fixed/distributed/tt_metal.hpp"

class AdamWCompositeFullTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(AdamWCompositeFullTest, AdamWCompositeTest) {
    using namespace ttml::ops;
    ttml::autograd::ctx().set_seed(42);
    auto* device = &ttml::autograd::ctx().get_device();
    const size_t batch_size = 32;
    const size_t num_features = 64;
    std::vector<float> features;
    features.reserve(batch_size * num_features);
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < num_features; ++j) {
            features.push_back(static_cast<float>(i) * 0.1F);
        }
    }

    std::vector<float> targets;
    targets.reserve(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        targets.push_back(static_cast<float>(i) * 0.1F);
    }

    auto data_tensor = ttml::autograd::create_tensor(
        ttml::core::from_vector(features, ttnn::Shape({batch_size, 1, 1, num_features}), device));

    auto targets_tensor =
        ttml::autograd::create_tensor(ttml::core::from_vector(targets, ttnn::Shape({batch_size, 1, 1, 1}), device));

    auto model = ttml::modules::LinearLayer(num_features, 1);
    auto adamw_config = ttml::optimizers::AdamWCompositeConfig();
    adamw_config.lr = 1e-2F;
    adamw_config.weight_decay = 0.F;
    auto optimizer = ttml::optimizers::AdamWComposite(model.parameters(), adamw_config);

    const size_t steps = 100;
    std::vector<float> losses;
    losses.reserve(steps);
    for (size_t step = 0; step < steps; ++step) {
        optimizer.zero_grad();
        auto prediction = model(data_tensor);
        auto loss = ttml::ops::mse_loss(prediction, targets_tensor);
        auto loss_value = ttml::core::to_vector(loss->get_value())[0];
        losses.emplace_back(loss_value);
        loss->backward();
        optimizer.step();
        ttml::autograd::ctx().reset_graph();
    }
    EXPECT_LT(losses.back(), losses.front());
    EXPECT_LT(losses.back(), 1e-3F);
}

// ====================================================================
// Mesh topology: a step against a mislabelled gradient must not relabel the
// parameter or the optimizer state (the checkpointer gathers by that label).
// ====================================================================

namespace {

// A (1, 1, 64, 64) replicated parameter labelled either N-D ({1, 2} / [Replicate, Replicate], what an explicit
// N-D mapper produces) or collapsed 1-D ({2} / [Replicate], what the default mappers produce).
ttml::autograd::TensorPtr make_replicated_param(ttnn::distributed::MeshDevice* device, bool collapsed_1d_label) {
    const std::array<std::size_t, 4> shape{1U, 1U, 64U, 64U};
    xt::xarray<float> w = ttml::test_utils::make_uniform_xarray<float>(shape, -1.0F, 1.0F, 42U);
    std::unique_ptr<ttnn::distributed::TensorToMesh> mapper =
        collapsed_1d_label ? ttnn::distributed::replicate_tensor_to_mesh_mapper(*device)
                           : ttnn::distributed::create_mesh_mapper(
                                 *device,
                                 ttnn::distributed::MeshMapperConfig{
                                     .placements = {
                                         ttnn::distributed::MeshMapperConfig::Replicate{},
                                         ttnn::distributed::MeshMapperConfig::Replicate{}}});
    auto tensor =
        ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(w, device, ttnn::Layout::TILE, mapper.get());
    return ttml::autograd::create_tensor(tensor, /* requires_grad */ true);
}

// A gradient with the parameter's per-device shape (1, 1, 64, 64) but the wrong label: a (1, 1, 64, 128) host
// array sharded on dim 3 across mesh axis 1, i.e. {1, 2} / [Replicate, Shard(3)] -- what a CCL output that kept
// a stale Shard on the axis it reduced looks like.
ttnn::Tensor make_mislabelled_grad(ttnn::distributed::MeshDevice* device, uint32_t seed) {
    const std::array<std::size_t, 4> wide{1U, 1U, 64U, 128U};
    xt::xarray<float> g = ttml::test_utils::make_uniform_xarray<float>(wide, -0.1F, 0.1F, seed);
    auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, 3, /* cluster_axis */ 1);
    return ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(g, device, ttnn::Layout::TILE, mapper.get());
}

// Two steps against a mislabelled gradient (re-set before each step): the parameter and every tensor leaf of the
// optimizer state must still carry the parameter's original topology afterwards. Before the composites restored
// the topology after set_value, the N-D parameter came back labelled Shard(3) (a checkpoint would then gather it
// as a 2x-wide tensor) and the 1-D one lost its label to the union's rank-mismatch fallback.
template <class Optimizer, class Config>
void expect_step_keeps_topology(const Config& config, bool collapsed_1d_label) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto param = make_replicated_param(device, collapsed_1d_label);
    // By value: tensor_topology() is a reference into attributes the step overwrites.
    const tt::tt_metal::TensorTopology expected =
        param->get_value(ttml::autograd::PreferredPrecision::HALF).tensor_topology();
    ASSERT_EQ(expected.distribution_shape().dims(), collapsed_1d_label ? 1U : 2U) << "precondition: label rank";

    ttml::serialization::NamedParameters params{{"w", param}};
    Optimizer optimizer(params, config);

    for (uint32_t step = 0; step < 2U; ++step) {
        auto grad = make_mislabelled_grad(device, 100U + step);
        ASSERT_NE(grad.tensor_topology(), expected) << "precondition: the gradient must carry a different label";
        param->set_grad(grad);

        optimizer.step();

        EXPECT_EQ(param->get_value(ttml::autograd::PreferredPrecision::HALF).tensor_topology(), expected)
            << "step " << step << " relabelled the parameter";
        size_t checked_leaves = 0;
        for (const auto& [key, entry] : optimizer.get_state_dict()) {
            const auto* leaves = std::get_if<ttml::serialization::NamedParameters>(&entry);
            if (leaves == nullptr) {
                continue;
            }
            for (const auto& [name, leaf] : *leaves) {
                ++checked_leaves;
                EXPECT_EQ(leaf->get_value(ttml::autograd::PreferredPrecision::HALF).tensor_topology(), expected)
                    << "step " << step << " relabelled state " << key << "[" << name << "]";
            }
        }
        EXPECT_GT(checked_leaves, 0U) << "the state dict holds no per-parameter tensors";
    }
}

}  // namespace

// Needs a (1, 2) mesh: guarded on the available chip count rather than the board type so it also runs on larger
// meshes (the mesh graph descriptor comes from TT_MESH_GRAPH_DESC_PATH there, as for the Python tp_mesh fixture).
class AdamWCompositeMeshTopologyTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (tt::tt_metal::GetNumAvailableDevices() < 2U) {
            GTEST_SKIP() << "Skipping: a (1, 2) mesh needs at least two chips";
        }
        ttml::ttnn_fixed::distributed::enable_fabric(2U);
        ttml::autograd::ctx().open_device(tt::tt_metal::distributed::MeshShape(1, 2));
        ttml::autograd::ctx().set_seed(42);
    }

    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(AdamWCompositeMeshTopologyTest, MorehAdamWStepKeepsNDParameterTopology) {
    ttml::optimizers::AdamWCompositeConfig config;  // weight_decay 0.01 default: the decay path is exercised
    expect_step_keeps_topology<ttml::optimizers::MorehAdamW>(config, /* collapsed_1d_label */ false);
}

TEST_F(AdamWCompositeMeshTopologyTest, MorehAdamWStepKeeps1DParameterTopology) {
    ttml::optimizers::AdamWCompositeConfig config;
    expect_step_keeps_topology<ttml::optimizers::MorehAdamW>(config, /* collapsed_1d_label */ true);
}

TEST_F(AdamWCompositeMeshTopologyTest, AdamWCompositeStepKeepsNDParameterTopology) {
    // amsgrad + Kahan: every state tensor (moments, max_exp_avg_sq, kahan_compensation) is written each step.
    ttml::optimizers::AdamWCompositeConfig config;
    config.amsgrad = true;
    config.kahan_summation = true;
    expect_step_keeps_topology<ttml::optimizers::AdamWComposite>(config, /* collapsed_1d_label */ false);
}

TEST_F(AdamWCompositeMeshTopologyTest, AdamWCompositeStepKeeps1DParameterTopology) {
    ttml::optimizers::AdamWCompositeConfig config;
    config.amsgrad = true;
    config.kahan_summation = true;
    expect_step_keeps_topology<ttml::optimizers::AdamWComposite>(config, /* collapsed_1d_label */ true);
}

TEST_F(AdamWCompositeMeshTopologyTest, AdamWCompositePlainStepKeepsNDParameterTopology) {
    // The non-Kahan parameter update path.
    ttml::optimizers::AdamWCompositeConfig config;
    expect_step_keeps_topology<ttml::optimizers::AdamWComposite>(config, /* collapsed_1d_label */ false);
}
