// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared harness for the optimizer mesh-topology tests: a step against a mislabelled gradient must not relabel the
// parameter, the optimizer state (the checkpointer gathers both by their label) or the caller's gradient.

#pragma once

#include <gtest/gtest.h>

#include <array>
#include <core/ttnn_all_includes.hpp>
#include <cstdlib>
#include <memory>
#include <string>
#include <tt-metalium/host_api.hpp>
#include <umd/device/cluster.hpp>
#include <variant>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "serialization/serializable.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn_fixed/distributed/tt_metal.hpp"

namespace ttml::test_utils::optimizer_topology {

// A (1, 1, 64, 64) replicated parameter labelled either N-D ({1, 2} / [Replicate, Replicate], what an explicit
// N-D mapper produces) or collapsed 1-D ({2} / [Replicate], what the default mappers produce).
inline autograd::TensorPtr make_replicated_param(ttnn::distributed::MeshDevice* device, bool collapsed_1d_label) {
    const std::array<std::size_t, 4> shape{1U, 1U, 64U, 64U};
    xt::xarray<float> w = make_uniform_xarray<float>(shape, -1.0F, 1.0F, 42U);
    std::unique_ptr<ttnn::distributed::TensorToMesh> mapper =
        collapsed_1d_label ? ttnn::distributed::replicate_tensor_to_mesh_mapper(*device)
                           : ttnn::distributed::create_mesh_mapper(
                                 *device,
                                 ttnn::distributed::MeshMapperConfig{
                                     .placements = {
                                         ttnn::distributed::MeshMapperConfig::Replicate{},
                                         ttnn::distributed::MeshMapperConfig::Replicate{}}});
    auto tensor = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(w, device, ttnn::Layout::TILE, mapper.get());
    return autograd::create_tensor(tensor, /* requires_grad */ true);
}

// A gradient with the parameter's per-device shape (1, 1, 64, 64) but the wrong label: a (1, 1, 64, 128) host
// array sharded on dim 3 across mesh axis 1, i.e. {1, 2} / [Replicate, Shard(3)] -- what a CCL output that kept
// a stale Shard on the axis it reduced looks like.
inline ttnn::Tensor make_mislabelled_grad(ttnn::distributed::MeshDevice* device, uint32_t seed) {
    const std::array<std::size_t, 4> wide{1U, 1U, 64U, 128U};
    xt::xarray<float> g = make_uniform_xarray<float>(wide, -0.1F, 0.1F, seed);
    auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, 3, /* cluster_axis */ 1);
    return core::from_xtensor<float, ttnn::DataType::BFLOAT16>(g, device, ttnn::Layout::TILE, mapper.get());
}

// Two steps against a mislabelled gradient (re-set before each step): the parameter and every tensor leaf of the
// optimizer state must still carry the parameter's original topology afterwards, and the gradient its own. Without
// the composites restoring the topology after set_value, ttnn's union rule
// (ttnn::device_operation::detail::compute_output_placements_and_shape) hands the parameter the gradient's
// {1, 2} / [Replicate, Shard(3)] label in both cases: in the N-D case by overlaying the gradient's Shard, in the
// collapsed 1-D case outright, because the union ignores a fully replicated input of lower distribution rank. A
// checkpoint would then gather the parameter as a 2x-wide tensor.
template <class Optimizer, class Config>
void expect_step_keeps_topology(const Config& config, bool collapsed_1d_label) {
    auto* device = &autograd::ctx().get_device();
    auto param = make_replicated_param(device, collapsed_1d_label);
    // By value: tensor_topology() is a reference into attributes the step overwrites.
    const tt::tt_metal::TensorTopology expected =
        param->get_value(autograd::PreferredPrecision::HALF).tensor_topology();
    ASSERT_EQ(expected.distribution_shape().dims(), collapsed_1d_label ? 1U : 2U) << "precondition: label rank";

    serialization::NamedParameters params{{"w", param}};
    Optimizer optimizer(params, config);

    for (uint32_t step = 0; step < 2U; ++step) {
        auto grad = make_mislabelled_grad(device, 100U + step);
        const tt::tt_metal::TensorTopology grad_topology = grad.tensor_topology();
        ASSERT_NE(grad_topology, expected) << "precondition: the gradient must carry a different label";
        param->set_grad(grad);

        optimizer.step();

        EXPECT_EQ(param->get_value(autograd::PreferredPrecision::HALF).tensor_topology(), expected)
            << "step " << step << " relabelled the parameter";
        EXPECT_EQ(param->get_grad().tensor_topology(), grad_topology)
            << "step " << step << " relabelled the caller's gradient";
        size_t checked_leaves = 0;
        for (const auto& [key, entry] : optimizer.get_state_dict()) {
            const auto* leaves = std::get_if<serialization::NamedParameters>(&entry);
            if (leaves == nullptr) {
                continue;
            }
            for (const auto& [name, leaf] : *leaves) {
                ++checked_leaves;
                EXPECT_EQ(leaf->get_value(autograd::PreferredPrecision::HALF).tensor_topology(), expected)
                    << "step " << step << " relabelled state " << key << "[" << name << "]";
            }
        }
        EXPECT_GT(checked_leaves, 0U) << "the state dict holds no per-parameter tensors";
    }
}

inline bool check_board_is_n300() {
    return tt::umd::Cluster::create_cluster_descriptor()->get_board_type(0) == tt::BoardType::N300;
}

// Opens one (1, 2) mesh per test suite. An N300 opens it natively; any other board needs TT_MESH_GRAPH_DESC_PATH
// naming a 1x2 mesh graph descriptor (on blackhole tt-train/configs/mgd/bh_galaxy_1_2_line_line.textproto, as the
// Python tp_mesh fixture sets), since tt-train has no built-in default for two chips -- without one the suite is
// skipped instead of failing in open_device.
class MeshTopologyTest : public ::testing::Test {
public:
    static void SetUpTestSuite() {
        s_skip_reason.clear();
        if (tt::tt_metal::GetNumAvailableDevices() < 2U) {
            s_skip_reason = "a (1, 2) mesh needs at least two chips";
            return;
        }
        if (!check_board_is_n300() && std::getenv("TT_MESH_GRAPH_DESC_PATH") == nullptr) {
            s_skip_reason = "off an N300, set TT_MESH_GRAPH_DESC_PATH to a 1x2 mesh graph descriptor";
            return;
        }
        ttnn_fixed::distributed::enable_fabric(2U);
        autograd::ctx().open_device(tt::tt_metal::distributed::MeshShape(1, 2));
    }

    static void TearDownTestSuite() {
        if (s_skip_reason.empty()) {
            autograd::ctx().close_device();
        }
    }

protected:
    void SetUp() override {
        if (!s_skip_reason.empty()) {
            GTEST_SKIP() << "Skipping: " << s_skip_reason;
        }
        autograd::ctx().set_seed(42);
    }

    void TearDown() override {
        if (s_skip_reason.empty()) {
            autograd::ctx().reset_graph();
        }
    }

private:
    inline static std::string s_skip_reason;
};

}  // namespace ttml::test_utils::optimizer_topology
