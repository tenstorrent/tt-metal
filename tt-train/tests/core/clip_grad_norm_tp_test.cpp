// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <xtensor/containers/xarray.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/clip_grad_norm.hpp"
#include "core/tt_tensor_utils.hpp"
#include "test_utils/mesh_utils.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/operations/data_movement/fill_pad/fill_pad.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"

using namespace ttml;

namespace {

// Mesh is 1x2, so tensor parallelism runs along mesh axis 1 and sharded tensors are split on their last dim.
constexpr uint32_t kClusterAxis = 1U;
constexpr int kShardDim = 3;
constexpr uint32_t kTpSize = 2U;

// Every intermediate (step1 partial sums, their sum, sqrt, the coefficient, the scaled grads) is stored in
// bf16, so each stage rounds to 2^-8 relative. Five such stages bound the error at ~2e-2 for gradients of at most
// a few hundred tiles per device; beyond that step1's bf16 accumulation dominates (see DISABLED_LargeGradientNorm).
constexpr float kRtol = 2e-2F;
constexpr float kAtol = 1e-3F;

using Shape = std::vector<size_t>;

struct ClipGradNormTpCase {
    std::string name;
    uint32_t num_sharded;
    uint32_t num_replicated;
    float max_norm_ratio;  // max_norm = ratio * true norm; > 1 means nothing is clipped
    // Adds (compute cores + 1) tensors to each group so step1/step3 need more than one launch per group.
    bool exceed_cores = false;
    float p_norm_type = 2.0F;
};

uint32_t num_compute_cores() {
    const auto grid = autograd::ctx().get_device().compute_with_storage_grid_size();
    return static_cast<uint32_t>(grid.x * grid.y);
}

// Values are multiples of 1/8 in [-4, 4], exactly representable in bf16, so inputs carry no rounding error.
// Seeded from the context generator, which the fixtures seed, so each test's data doesn't depend on test order.
xt::xarray<float> random_bf16_exact(const Shape& shape) {
    const auto seed = static_cast<uint32_t>(autograd::ctx().get_generator()());
    xt::xarray<float> values = test_utils::make_uniform_xarray<int>(shape, -32, 32, seed);
    return values / 8.0F;
}

struct TpGrads {
    std::vector<xt::xarray<float>> sharded_full;  // full (unsharded) grads
    std::vector<xt::xarray<float>> replicated;
    serialization::NamedParameters sharded_params;
    serialization::NamedParameters replicated_params;
};

// The value is placed like its grad, as in a real TP model, so clip_grad_norm_tp's placement check accepts it.
autograd::TensorPtr make_param(const ttnn::Tensor& grad) {
    auto value = core::zeros(grad.logical_shape(), grad.device());
    value.update_tensor_topology(grad.tensor_topology());
    auto param = autograd::create_tensor(value);
    param->set_grad(grad);
    param->set_requires_grad(true);
    return param;
}

// Without a cluster axis the tensor is sharded across every device and carries a single placement for the whole
// mesh, as GPT-2's TP linears are.
ttnn::Tensor shard(const xt::xarray<float>& full, std::optional<int> cluster_axis = kClusterAxis) {
    auto* device = &autograd::ctx().get_device();
    const auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, kShardDim, cluster_axis);
    return core::from_xtensor(full, device, ttnn::Layout::TILE, mapper.get());
}

ttnn::Tensor replicate(const xt::xarray<float>& values) {
    auto* device = &autograd::ctx().get_device();
    const auto mapper = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
    return core::from_xtensor(values, device, ttnn::Layout::TILE, mapper.get());
}

// Sharded shapes are the full (unsharded) shapes; their last dim is split across the TP group.
TpGrads make_grads(
    const std::vector<Shape>& sharded_shapes,
    const std::vector<Shape>& replicated_shapes,
    const std::function<xt::xarray<float>(const Shape&)>& make_values = random_bf16_exact) {
    TpGrads grads;
    for (size_t i = 0; i < sharded_shapes.size(); ++i) {
        auto full = make_values(sharded_shapes[i]);
        grads.sharded_params[fmt::format("sharded_{}", i)] = make_param(shard(full));
        grads.sharded_full.push_back(std::move(full));
    }
    for (size_t i = 0; i < replicated_shapes.size(); ++i) {
        auto values = make_values(replicated_shapes[i]);
        grads.replicated_params[fmt::format("replicated_{}", i)] = make_param(replicate(values));
        grads.replicated.push_back(std::move(values));
    }
    return grads;
}

TpGrads make_grads(uint32_t num_sharded, uint32_t num_replicated) {
    return make_grads(
        std::vector<Shape>(num_sharded, Shape{1U, 1U, 32U, 32U * kTpSize}),
        std::vector<Shape>(num_replicated, Shape{1U, 1U, 32U, 32U}));
}

// Global p-norm over the full tensors, each replicated tensor counted once.
float reference_norm(const TpGrads& grads, float p = 2.0F) {
    float pow_sum = 0.0F;
    for (const auto& g : grads.sharded_full) {
        pow_sum += xt::sum(xt::pow(xt::abs(g), p))();
    }
    for (const auto& g : grads.replicated) {
        pow_sum += xt::sum(xt::pow(xt::abs(g), p))();
    }
    return std::pow(pow_sum, 1.0F / p);
}

void expect_close(const xt::xarray<float>& actual, const xt::xarray<float>& expected, const std::string& label) {
    EXPECT_TRUE(xt::allclose(actual, expected, kRtol, kAtol)) << label;
}

std::vector<xt::xarray<float>> norm_per_device(const autograd::TensorPtr& norm) {
    return core::to_xtensor(norm->get_value(), core::IdentityComposer{});
}

// Compares the on-device result against the host reference: the norm must match on every device and the
// gradients, once gathered back, must equal the host grads times the expected coefficient.
void verify(const TpGrads& grads, const autograd::TensorPtr& norm, float max_norm, float p = 2.0F) {
    auto* device = &autograd::ctx().get_device();
    const float expected_norm = reference_norm(grads, p);
    const float coef = std::min(1.0F, max_norm / (expected_norm + 1e-6F));

    const auto norms = norm_per_device(norm);
    ASSERT_EQ(norms.size(), kTpSize);
    for (size_t d = 0; d < kTpSize; ++d) {
        EXPECT_NEAR(norms[d](0), expected_norm, kRtol * expected_norm + kAtol) << "norm on device " << d;
    }

    const auto composer = ttnn::distributed::concat_mesh_to_tensor_composer(*device, kShardDim);
    for (size_t i = 0; i < grads.sharded_full.size(); ++i) {
        const auto& param = grads.sharded_params.at(fmt::format("sharded_{}", i));
        expect_close(
            core::to_xtensor(param->get_grad(), *composer), grads.sharded_full[i] * coef, fmt::format("sharded_{}", i));
    }
    for (size_t i = 0; i < grads.replicated.size(); ++i) {
        const auto& param = grads.replicated_params.at(fmt::format("replicated_{}", i));
        const auto per_device = core::to_xtensor(param->get_grad(), core::IdentityComposer{});
        for (size_t d = 0; d < kTpSize; ++d) {
            expect_close(per_device[d], grads.replicated[i] * coef, fmt::format("replicated_{} on device {}", i, d));
        }
    }
}

// Scales by 8 the grads that step1/step3's second launch handles: those past the first `first_launch` in the map's
// iteration order, which is the order clip_grad_norm_tp collects them in. Unweighted, losing those few tensors would
// move the norm by less than kRtol.
void boost_second_launch(
    serialization::NamedParameters& params,
    std::vector<xt::xarray<float>>& host,
    const std::string& prefix,
    uint32_t first_launch) {
    uint32_t position = 0;
    for (auto& [name, param] : params) {
        if (position++ < first_launch) {
            continue;
        }
        param->set_grad(ttnn::multiply(param->get_grad(), 8.0F));
        host[std::stoul(name.substr(prefix.size()))] *= 8.0F;
    }
}

// Fills each grad's tile padding with a large value so a leak into the norm would show.
void poison_padding(serialization::NamedParameters& params) {
    for (auto& [_, param] : params) {
        param->set_grad(ttnn::fill_implicit_tile_padding(param->get_grad(), 1000.0F));
    }
}

}  // namespace

class ClipGradNormTpTest : public test_utils::Mesh1x2Fixture, public ::testing::WithParamInterface<ClipGradNormTpCase> {
protected:
    ClipGradNormTpTest() : Mesh1x2Fixture(/*seed=*/42) {
    }
};

TEST_P(ClipGradNormTpTest, MatchesHostReference) {
    const auto& param = GetParam();
    const uint32_t extra = param.exceed_cores ? num_compute_cores() + 1U : 0U;
    auto grads = make_grads(param.num_sharded + extra, param.num_replicated + extra);
    if (param.exceed_cores) {
        boost_second_launch(grads.sharded_params, grads.sharded_full, "sharded_", num_compute_cores());
        boost_second_launch(grads.replicated_params, grads.replicated, "replicated_", num_compute_cores());
    }
    const float max_norm = param.max_norm_ratio * reference_norm(grads, param.p_norm_type);

    auto norm = core::clip_grad_norm_tp(
        grads.sharded_params, grads.replicated_params, max_norm, kClusterAxis, param.p_norm_type);

    verify(grads, norm, max_norm, param.p_norm_type);
}

INSTANTIATE_TEST_SUITE_P(
    ClipGradNormTp,
    ClipGradNormTpTest,
    ::testing::Values(
        ClipGradNormTpCase{.name = "sharded_only", .num_sharded = 3, .num_replicated = 0, .max_norm_ratio = 0.5F},
        ClipGradNormTpCase{.name = "replicated_only", .num_sharded = 0, .num_replicated = 3, .max_norm_ratio = 0.5F},
        // Replicated contributes as much as sharded: counting it per device would inflate the norm by ~22%.
        ClipGradNormTpCase{.name = "mixed", .num_sharded = 2, .num_replicated = 4, .max_norm_ratio = 0.5F},
        ClipGradNormTpCase{.name = "below_max_norm", .num_sharded = 2, .num_replicated = 2, .max_norm_ratio = 2.0F},
        ClipGradNormTpCase{
            .name = "more_than_cores",
            .num_sharded = 1,
            .num_replicated = 1,
            .max_norm_ratio = 0.5F,
            .exceed_cores = true},
        // p != 2 takes the ttnn::pow path instead of ttnn::sqrt.
        ClipGradNormTpCase{
            .name = "p1", .num_sharded = 2, .num_replicated = 2, .max_norm_ratio = 0.5F, .p_norm_type = 1.0F},
        ClipGradNormTpCase{
            .name = "p3", .num_sharded = 2, .num_replicated = 2, .max_norm_ratio = 0.5F, .p_norm_type = 3.0F}),
    [](const ::testing::TestParamInfo<ClipGradNormTpCase>& info) { return info.param.name; });

// Second call with fresh gradient buffers of the same shapes must hit the program cache and still be correct.
TEST_F(ClipGradNormTpTest, ProgramCacheHit) {
    auto* device = &autograd::ctx().get_device();

    auto first = make_grads(2, 2);
    core::clip_grad_norm_tp(first.sharded_params, first.replicated_params, 1.0F, kClusterAxis);
    const auto entries_after_first = device->num_program_cache_entries();

    auto second = make_grads(2, 2);
    auto norm = core::clip_grad_norm_tp(second.sharded_params, second.replicated_params, 1.0F, kClusterAxis);

    EXPECT_EQ(device->num_program_cache_entries(), entries_after_first);
    verify(second, norm, 1.0F);
}

// Gradients of different sizes in one call: step1/step3 take accessor args from the first input only, so each
// core must still read its own tensor's tile count.
TEST_F(ClipGradNormTpTest, MixedMultiTileShapes) {
    auto grads = make_grads(
        {{1U, 1U, 32U, 64U}, {1U, 1U, 64U, 128U}, {2U, 1U, 32U, 256U}},
        {{1U, 1U, 32U, 32U}, {1U, 2U, 64U, 96U}, {1U, 1U, 96U, 32U}});
    const float max_norm = 0.5F * reference_norm(grads);

    auto norm = core::clip_grad_norm_tp(grads.sharded_params, grads.replicated_params, max_norm, kClusterAxis);

    verify(grads, norm, max_norm);
}

// Shapes that don't fill whole tiles: the tile padding must not leak into the norm. from_xtensor zero-pads, which
// would hide a leak (|0|^p = 0), so the padding is filled with a large value first.
TEST_F(ClipGradNormTpTest, UnalignedShapes) {
    auto grads = make_grads({{1U, 1U, 20U, 80U}}, {{1U, 1U, 17U, 40U}, {1U, 3U, 5U, 7U}});
    poison_padding(grads.sharded_params);
    poison_padding(grads.replicated_params);
    const float max_norm = 0.5F * reference_norm(grads);

    auto norm = core::clip_grad_norm_tp(grads.sharded_params, grads.replicated_params, max_norm, kClusterAxis);

    verify(grads, norm, max_norm);
}

// Sharded grads whose params carry a single placement for the whole mesh, as GPT-2's TP linears produce.
TEST_F(ClipGradNormTpTest, CollapsedShardPlacement) {
    auto grads = make_grads({}, std::vector<Shape>(2, Shape{1U, 1U, 32U, 32U}));
    for (size_t i = 0; i < 2; ++i) {
        auto full = random_bf16_exact({1U, 1U, 32U, 32U * kTpSize});
        grads.sharded_params[fmt::format("sharded_{}", i)] = make_param(shard(full, /*cluster_axis=*/std::nullopt));
        grads.sharded_full.push_back(std::move(full));
    }
    ASSERT_EQ(grads.sharded_params.at("sharded_0")->get_value().tensor_topology().placements().size(), 1U);
    const float max_norm = 0.5F * reference_norm(grads);

    auto norm = core::clip_grad_norm_tp(grads.sharded_params, grads.replicated_params, max_norm, kClusterAxis);

    verify(grads, norm, max_norm);
}

// Column-parallel (dim 2) and row-parallel (dim 3) grads are clipped in one step3 launch, which relabels its outputs
// with the union of its inputs' placements; each grad must keep its own label.
TEST_F(ClipGradNormTpTest, PreservesGradTopology) {
    auto* device = &autograd::ctx().get_device();
    auto shard_on = [&](const xt::xarray<float>& values, int dim) {
        const auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, dim, kClusterAxis);
        return core::from_xtensor(values, device, ttnn::Layout::TILE, mapper.get());
    };
    serialization::NamedParameters sharded;
    sharded["dim2"] = make_param(shard_on(random_bf16_exact({1U, 1U, 32U * kTpSize, 32U}), 2));
    sharded["dim3"] = make_param(shard_on(random_bf16_exact({1U, 1U, 32U, 32U * kTpSize}), 3));
    serialization::NamedParameters replicated;
    replicated["replicated"] = make_param(replicate(random_bf16_exact({1U, 1U, 32U, 32U})));

    std::unordered_map<std::string, std::string> before;
    for (const auto* params : {&sharded, &replicated}) {
        for (const auto& [name, param] : *params) {
            before[name] = fmt::format("{}", param->get_grad().tensor_topology().placements());
        }
    }

    core::clip_grad_norm_tp(sharded, replicated, 1.0F, kClusterAxis);

    for (const auto* params : {&sharded, &replicated}) {
        for (const auto& [name, param] : *params) {
            EXPECT_EQ(fmt::format("{}", param->get_grad().tensor_topology().placements()), before.at(name)) << name;
        }
    }
}

// A zero norm hits the 1e-6 guard: the coefficient clamps to 1 and the gradients stay zero instead of NaN.
TEST_F(ClipGradNormTpTest, ZeroGradients) {
    auto grads = make_grads(
        std::vector<Shape>(2, Shape{1U, 1U, 32U, 32U * kTpSize}),
        std::vector<Shape>(2, Shape{1U, 1U, 32U, 32U}),
        [](const Shape& shape) { return xt::xarray<float>(xt::zeros<float>(shape)); });

    auto norm = core::clip_grad_norm_tp(grads.sharded_params, grads.replicated_params, 1.0F, kClusterAxis);

    verify(grads, norm, 1.0F);
}

// Parameters that don't require a grad, or have none yet, are left out of the norm and left untouched.
TEST_F(ClipGradNormTpTest, SkipsParamsWithoutGrad) {
    auto grads = make_grads(1, 1);
    const auto frozen_values = random_bf16_exact({1U, 1U, 32U, 32U});
    // set_requires_grad(false) drops the grad, so set the grad on a tensor that never required one.
    auto frozen = autograd::create_tensor(core::zeros(ttnn::Shape({1U, 1U, 32U, 32U}), &autograd::ctx().get_device()));
    frozen->set_grad(replicate(frozen_values));
    grads.replicated_params["frozen"] = frozen;
    auto no_grad = autograd::create_tensor(core::zeros(ttnn::Shape({1U, 1U, 32U, 32U}), &autograd::ctx().get_device()));
    no_grad->set_requires_grad(true);
    grads.replicated_params["no_grad"] = no_grad;
    const float max_norm = 0.5F * reference_norm(grads);

    auto norm = core::clip_grad_norm_tp(grads.sharded_params, grads.replicated_params, max_norm, kClusterAxis);

    verify(grads, norm, max_norm);
    EXPECT_FALSE(no_grad->is_grad_initialized());
    for (const auto& values : core::to_xtensor(frozen->get_grad(), core::IdentityComposer{})) {
        EXPECT_TRUE(xt::all(xt::equal(values, frozen_values))) << "frozen grad was scaled";
    }
}

TEST_F(ClipGradNormTpTest, RejectsInvalidArguments) {
    auto grads = make_grads(1, 1);
    const auto& sharded = grads.sharded_params;
    const auto& replicated = grads.replicated_params;
    constexpr float kInf = std::numeric_limits<float>::infinity();
    constexpr float kNan = std::numeric_limits<float>::quiet_NaN();

    for (float p : {0.0F, -1.0F, kInf, kNan}) {
        EXPECT_THROW(core::clip_grad_norm_tp(sharded, replicated, 1.0F, kClusterAxis, p), std::logic_error)
            << "p_norm_type " << p;
    }
    EXPECT_THROW(core::clip_grad_norm_tp({}, {}, 1.0F, kClusterAxis), std::logic_error) << "no params";
    EXPECT_THROW(core::clip_grad_norm_tp(sharded, replicated, 1.0F, /*cluster_axis=*/2U), std::logic_error)
        << "cluster_axis out of range";

    auto* device = &autograd::ctx().get_device();
    const auto mapper = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
    const auto values = random_bf16_exact({1U, 1U, 32U, 32U});

    auto with_fp32 = replicated;
    with_fp32["fp32"] = make_param(
        core::from_xtensor<float, ttnn::DataType::FLOAT32>(values, device, ttnn::Layout::TILE, mapper.get()));
    EXPECT_THROW(core::clip_grad_norm_tp(sharded, with_fp32, 1.0F, kClusterAxis), std::logic_error) << "dtype mismatch";

    auto with_2d = replicated;
    xt::xarray<float> values_2d = xt::reshape_view(values, {32U, 32U});
    with_2d["2d"] = make_param(core::from_xtensor(values_2d, device, ttnn::Layout::TILE, mapper.get()));
    EXPECT_THROW(core::clip_grad_norm_tp(sharded, with_2d, 1.0F, kClusterAxis), std::logic_error) << "2D grad";

    using ::testing::HasSubstr;
    using ::testing::ThrowsMessage;
    EXPECT_THAT(
        [&] { core::clip_grad_norm_tp(replicated, {}, 1.0F, kClusterAxis); },
        ThrowsMessage<std::logic_error>(HasSubstr("is passed as sharded but is replicated")));
    EXPECT_THAT(
        [&] { core::clip_grad_norm_tp({}, sharded, 1.0F, kClusterAxis); },
        ThrowsMessage<std::logic_error>(HasSubstr("is passed as replicated but is sharded")));

    // Only cluster_axis is reduced, so a parameter sharded on another mesh axis would get a per-device norm.
    EXPECT_THAT(
        [&] { core::clip_grad_norm_tp(sharded, replicated, 1.0F, /*cluster_axis=*/0U); },
        ThrowsMessage<std::logic_error>(HasSubstr("is sharded on mesh axis 1, but only cluster_axis 0 is reduced")));
    serialization::NamedParameters collapsed;
    collapsed["collapsed"] = make_param(shard(random_bf16_exact({1U, 1U, 32U, 32U * kTpSize}), std::nullopt));
    EXPECT_THAT(
        [&] { core::clip_grad_norm_tp(collapsed, {}, 1.0F, /*cluster_axis=*/0U); },
        ThrowsMessage<std::logic_error>(HasSubstr("is sharded across all 2 devices")));

    auto with_duplicate = replicated;
    with_duplicate["duplicate"] = sharded.begin()->second;
    EXPECT_THAT(
        [&] { core::clip_grad_norm_tp(sharded, with_duplicate, 1.0F, kClusterAxis); },
        ThrowsMessage<std::logic_error>(HasSubstr("is in both sharded and replicated")));
}

// Single device: with everything replicated and a TP group of one, the result must match clip_grad_norm.
class ClipGradNormTpSingleDeviceTest : public ::testing::Test {
protected:
    void SetUp() override {
        autograd::ctx().open_device();
        autograd::ctx().set_seed(42);
    }

    void TearDown() override {
        autograd::ctx().close_device();
    }
};

TEST_F(ClipGradNormTpSingleDeviceTest, MatchesClipGradNorm) {
    auto* device = &autograd::ctx().get_device();
    constexpr float max_norm = 1.0F;

    serialization::NamedParameters tp_params;
    serialization::NamedParameters ref_params;
    for (uint32_t i = 0; i < 3; ++i) {
        auto values = random_bf16_exact({1U, 1U, 32U, 32U});
        tp_params[std::to_string(i)] = make_param(core::from_xtensor(values, device));
        ref_params[std::to_string(i)] = make_param(core::from_xtensor(values, device));
    }

    auto tp_norm = core::clip_grad_norm_tp({}, tp_params, max_norm, /*cluster_axis=*/0U);
    auto ref_norm = core::clip_grad_norm(ref_params, max_norm, 2.0F, /*error_if_nonfinite=*/false);

    EXPECT_NEAR(
        core::to_vector(tp_norm->get_value()).front(),
        core::to_vector(ref_norm->get_value()).front(),
        kRtol * core::to_vector(ref_norm->get_value()).front());
    for (uint32_t i = 0; i < 3; ++i) {
        expect_close(
            core::to_xtensor(tp_params[std::to_string(i)]->get_grad()),
            core::to_xtensor(ref_params[std::to_string(i)]->get_grad()),
            std::to_string(i));
    }
}

// step1 accumulates each gradient's Sum[|g|^p] in bf16, so the norm drifts low as gradients grow: about 0.6x the
// true norm at 4096 tiles. TODO(tstepanovic): enable once step1 accumulates in fp32 (#63505).
TEST_F(ClipGradNormTpSingleDeviceTest, DISABLED_LargeGradientNorm) {
    auto* device = &autograd::ctx().get_device();
    const auto values = test_utils::make_uniform_xarray<float>(Shape{1U, 1U, 32U * 4096U, 32U}, -1.0F, 1.0F, 42U);
    serialization::NamedParameters params;
    params["large"] = make_param(core::from_xtensor(values, device));

    // fp64 reference over the bf16-rounded values the device holds.
    double pow_sum = 0.0;
    for (const float v : core::to_xtensor(params["large"]->get_grad())) {
        pow_sum += static_cast<double>(v) * v;
    }
    const double expected = std::sqrt(pow_sum);

    auto norm = core::clip_grad_norm_tp({}, params, 1.0F, /*cluster_axis=*/0U);

    EXPECT_NEAR(core::to_vector(norm->get_value()).front(), expected, kRtol * expected);
}
