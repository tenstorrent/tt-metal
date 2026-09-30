// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "optimizers/muon_composite.hpp"

#include <gtest/gtest.h>

#include <array>
#include <core/ttnn_all_includes.hpp>
#include <tt-metalium/host_api.hpp>
#include <umd/device/cluster.hpp>
#include <variant>
#include <xtensor-blas/xlinalg.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn_fixed/distributed/tt_metal.hpp"

namespace {

xt::xarray<float> newtonschulz5_cpu(const xt::xarray<float>& G, int steps, float eps) {
    constexpr float a = 3.4445f;
    constexpr float b = -4.7750f;
    constexpr float c = 2.0315f;

    auto X = G;
    auto last_2d = xt::view(X, 0, 0, xt::all(), xt::all());
    auto norm = std::sqrt(xt::sum(last_2d * last_2d)());
    X = X / (norm + eps);

    auto shape = X.shape();
    bool needs_transpose = (shape[2] > shape[3]);
    if (needs_transpose) {
        X = xt::transpose(X, {0, 1, 3, 2});
    }

    for (int iter = 0; iter < steps; ++iter) {
        auto X_2d = xt::view(X, 0, 0, xt::all(), xt::all());
        auto A = xt::linalg::dot(X_2d, xt::transpose(X_2d));
        auto B = b * A + c * xt::linalg::dot(A, A);
        xt::view(X, 0, 0, xt::all(), xt::all()) = a * X_2d + xt::linalg::dot(B, X_2d);
    }

    if (needs_transpose) {
        X = xt::transpose(X, {0, 1, 3, 2});
    }
    return X;
}

xt::xarray<float> muon_step_cpu(
    const xt::xarray<float>& param,
    const xt::xarray<float>& grad,
    xt::xarray<float>& momentum_buffer,
    float lr,
    float momentum,
    int ns_steps,
    size_t step) {
    if (step > 0 && momentum != 0.0f) {
        momentum_buffer = momentum * momentum_buffer + grad;
    } else {
        momentum_buffer = grad;
    }

    auto update = newtonschulz5_cpu(momentum_buffer, ns_steps, 1e-7f);
    return param - lr * update;
}

}  // namespace

struct MuonTestCase {
    std::array<std::size_t, 4> shape;
    float lr;
    float momentum;
    int ns_steps;
    uint32_t steps;
    std::string name;
};

void PrintTo(const MuonTestCase& tc, std::ostream* os) {
    *os << tc.name;
}

class MuonCorrectnessTest : public ::testing::TestWithParam<MuonTestCase> {
public:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }
    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }

protected:
    void SetUp() override {
        ttml::autograd::ctx().set_seed(42);
    }
    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
    }
};

TEST_P(MuonCorrectnessTest, DeviceMatchesCPU) {
    using namespace ttml;
    const auto& tc = GetParam();

    xt::xarray<float> w0 = ttml::test_utils::make_uniform_xarray<float>(tc.shape, -1.0F, 1.0F, 42U);
    xt::xarray<float> g0 = ttml::test_utils::make_uniform_xarray<float>(tc.shape, -1.0F, 1.0F, 43U);

    // CPU reference
    xt::xarray<float> w_cpu = w0;
    xt::xarray<float> mom_cpu = xt::zeros<float>(tc.shape);
    for (uint32_t i = 0; i < tc.steps; ++i) {
        w_cpu = muon_step_cpu(w_cpu, g0, mom_cpu, tc.lr, tc.momentum, tc.ns_steps, i);
    }

    // Device
    auto param = autograd::create_tensor(core::from_xtensor(w0, &autograd::ctx().get_device()), true);
    param->set_grad(core::from_xtensor(g0, &autograd::ctx().get_device()));

    serialization::NamedParameters params{{"theta", param}};
    optimizers::MuonConfig config{.lr = tc.lr, .momentum = tc.momentum, .ns_steps = tc.ns_steps};
    optimizers::MuonComposite optimizer(params, config);

    for (uint32_t i = 0; i < tc.steps; ++i) {
        optimizer.step();
    }

    auto w_device = core::to_xtensor(param->get_value());
    EXPECT_TRUE(xt::allclose(w_device, w_cpu, 1e-2f, 1e-2f));
}

static const MuonTestCase kMuonCases[] = {
    {{1, 1, 32, 128}, 1e-2f, 0.95f, 5, 1, "Wide"},
    {{1, 1, 128, 32}, 1e-2f, 0.95f, 5, 2, "Tall_2_step"},
    {{1, 1, 128, 128}, 1e-2f, 0.95f, 5, 3, "Square_3_step"},
};

INSTANTIATE_TEST_SUITE_P(MuonCorrectness, MuonCorrectnessTest, ::testing::ValuesIn(kMuonCases), [](const auto& info) {
    return info.param.name;
});

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
// optimizer state must still carry the parameter's original topology afterwards. Without the pin, the union rule
// (ttnn::device_operation::detail::compute_output_placements_and_shape) hands both back with the gradient's
// {1, 2} / [Replicate, Shard(3)] label: the N-D parameter picks up the Shard (a checkpoint would then gather it as
// a 2x-wide tensor), and so does the collapsed 1-D one, because a fully replicated input of lower rank is ignored
// whenever a higher-rank sharded input is present -- a rank change plus a wrong Shard, not a fallback to a default.
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

// An N300 opens a (1, 2) mesh natively; every other board needs a mesh graph descriptor (as in comm_ops_test.cpp).
bool check_board_is_n300() {
    return tt::umd::Cluster::create_cluster_descriptor()->get_board_type(0) == tt::BoardType::N300;
}

}  // namespace

// Needs a (1, 2) mesh: guarded on the available chip count rather than the board type so it also runs on larger
// meshes. Off an N300 the mesh comes from TT_MESH_GRAPH_DESC_PATH (a 1x2 descriptor, as the Python tp_mesh fixture
// sets); get_mgd_path has no built-in default for two chips, so skip rather than let open_device fail.
class MuonCompositeMeshTopologyTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (tt::tt_metal::GetNumAvailableDevices() < 2U) {
            GTEST_SKIP() << "Skipping: a (1, 2) mesh needs at least two chips";
        }
        if (!check_board_is_n300() && !ttml::ttnn_fixed::distributed::get_mgd_path(2U).has_value()) {
            GTEST_SKIP() << "Skipping: a (1, 2) mesh on this board needs TT_MESH_GRAPH_DESC_PATH to name a 1x2 mesh "
                            "graph descriptor";
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

TEST_F(MuonCompositeMeshTopologyTest, MuonCompositeStepKeepsNDParameterTopology) {
    // Step 0 aliases the momentum buffer to the gradient; step 1 takes the momentum branch.
    ttml::optimizers::MuonConfig config{.lr = 1e-2F, .momentum = 0.95F, .ns_steps = 5};
    expect_step_keeps_topology<ttml::optimizers::MuonComposite>(config, /* collapsed_1d_label */ false);
}

TEST_F(MuonCompositeMeshTopologyTest, MuonCompositeStepKeeps1DParameterTopology) {
    ttml::optimizers::MuonConfig config{.lr = 1e-2F, .momentum = 0.95F, .ns_steps = 5};
    expect_step_keeps_topology<ttml::optimizers::MuonComposite>(config, /* collapsed_1d_label */ true);
}
