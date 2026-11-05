// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/generators/xbuilder.hpp>
#include <xtensor/views/xstrided_view.hpp>
#include <xtensor/views/xview.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/clip_grad_norm.hpp"
#include "core/tt_tensor_utils.hpp"
#include "modules/distributed/gpt_block.hpp"
#include "test_utils/mesh_utils.hpp"
#include "test_utils/program_cache.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/operations/data_movement/fill_pad/fill_pad.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"

using namespace ttml;

namespace {

// Tensor parallelism runs along mesh axis 1 and sharded tensors are split on their last dim. The mesh is 1x2, or 2x2
// with data parallelism on axis 0.
constexpr int kTpAxis = 1;
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
    // Adds (worker cores + 1) tensors to each group so step1 needs more than one launch per group (and step3, which
    // takes every group at once, more than one in all).
    bool exceed_cores = false;
    float p_norm_type = 2.0F;
};

// Values are multiples of 1/8 in [-4, 4], exactly representable in bf16, so inputs carry no rounding error.
// Seeded from the context generator, which the fixtures seed, so each test's data doesn't depend on test order.
xt::xarray<float> random_bf16_exact(const Shape& shape) {
    const auto seed = static_cast<uint32_t>(autograd::ctx().get_generator()());
    xt::xarray<float> values = test_utils::make_uniform_xarray<int>(shape, -32, 32, seed);
    return values / 8.0F;
}

// Scales the second half of `values` along `dim` by `factor`, a power of two so the values stay exact in bf16. The
// devices sharing a sharded grad then hold pieces of clearly different norms, so a reduction over the wrong mesh axis
// moves the norm well past the tolerance instead of landing near the right one.
xt::xarray<float> scale_second_half(xt::xarray<float> values, int dim, float factor) {
    xt::xstrided_slice_vector slice(values.dimension(), xt::all());
    const auto size = static_cast<std::ptrdiff_t>(values.shape(dim));
    slice[dim] = xt::range(size / 2, size);
    xt::strided_view(values, slice) *= factor;
    return values;
}

std::unique_ptr<ttnn::distributed::TensorToMesh> shard_mapper(int dim, std::optional<int> cluster_axis = kTpAxis) {
    return ttnn::distributed::shard_tensor_to_mesh_mapper(autograd::ctx().get_device(), dim, cluster_axis);
}

std::unique_ptr<ttnn::distributed::TensorToMesh> replicate_mapper() {
    return ttnn::distributed::replicate_tensor_to_mesh_mapper(autograd::ctx().get_device());
}

ttnn::Tensor distribute(const xt::xarray<float>& values, const ttnn::distributed::TensorToMesh& mapper) {
    return core::from_xtensor(values, &autograd::ctx().get_device(), ttnn::Layout::TILE, &mapper);
}

// A parameter's host copies: the full gradient the norm is taken over, and the piece each device holds before
// clipping.
struct HostGrad {
    xt::xarray<float> full;
    std::vector<xt::xarray<float>> pieces;
};

struct Grads {
    serialization::NamedParameters params;
    std::map<std::string, HostGrad> host;  // by parameter name
};

// clip_grad_norm reads each parameter's placement from its value, so the value is placed like its grad, as in a real
// TP model.
autograd::TensorPtr make_param(const ttnn::Tensor& grad) {
    auto value = core::zeros(grad.logical_shape(), grad.device());
    value.update_tensor_topology(grad.tensor_topology());
    auto param = autograd::create_tensor(value);
    param->set_grad(grad);
    param->set_requires_grad(true);
    return param;
}

// Adds a parameter whose grad is `full` distributed by `mapper`. The values are exact in bf16, so the pieces read back
// are exactly what each device holds.
void add_grad(
    Grads& grads,
    const std::string& name,
    const xt::xarray<float>& full,
    const ttnn::distributed::TensorToMesh& mapper) {
    auto grad = distribute(full, mapper);
    grads.host[name] = HostGrad{.full = full, .pieces = core::to_xtensor(grad, core::IdentityComposer{})};
    grads.params[name] = make_param(grad);
}

// Parameters are named sharded_<i> and replicated_<i>. Sharded shapes are the full (unsharded) shapes; their last dim
// is split across the TP group, with the second TP piece scaled up.
Grads make_grads(
    const std::vector<Shape>& sharded_shapes,
    const std::vector<Shape>& replicated_shapes,
    const std::function<xt::xarray<float>(const Shape&)>& make_values = random_bf16_exact) {
    Grads grads;
    const auto sharded = shard_mapper(kShardDim);
    for (size_t i = 0; i < sharded_shapes.size(); ++i) {
        add_grad(
            grads,
            fmt::format("sharded_{}", i),
            scale_second_half(make_values(sharded_shapes[i]), kShardDim, 2.0F),
            *sharded);
    }
    const auto replicated = replicate_mapper();
    for (size_t i = 0; i < replicated_shapes.size(); ++i) {
        add_grad(grads, fmt::format("replicated_{}", i), make_values(replicated_shapes[i]), *replicated);
    }
    return grads;
}

Grads make_grads(uint32_t num_sharded, uint32_t num_replicated) {
    return make_grads(
        std::vector<Shape>(num_sharded, Shape{1U, 1U, 32U, 32U * kTpSize}),
        std::vector<Shape>(num_replicated, Shape{1U, 1U, 32U, 32U}));
}

// clip_grad_norm without error_if_nonfinite's host read, as a training step would call it.
autograd::TensorPtr clip(const serialization::NamedParameters& params, float max_norm, float p = 2.0F) {
    return core::clip_grad_norm(params, max_norm, p, /*error_if_nonfinite=*/false);
}

// Global p-norm over the full tensors, each counted once.
float reference_norm(const Grads& grads, float p = 2.0F) {
    float pow_sum = 0.0F;
    for (const auto& [_, host] : grads.host) {
        pow_sum += xt::sum(xt::pow(xt::abs(host.full), p))();
    }
    return std::pow(pow_sum, 1.0F / p);
}

void expect_close(const xt::xarray<float>& actual, const xt::xarray<float>& expected, const std::string& label) {
    EXPECT_TRUE(xt::allclose(actual, expected, kRtol, kAtol)) << label;
}

std::vector<xt::xarray<float>> norm_per_device(const autograd::TensorPtr& norm) {
    return core::to_xtensor(norm->get_value(), core::IdentityComposer{});
}

// Compares the on-device result against the host reference. The norm must match it on every device, bit for bit
// across devices so that every device scales by the same coefficient. Each device's gradient must equal its piece
// times the expected coefficient, and devices that held the same piece (the copies of a replicated grad, or DP rows)
// must end bit-identical, since replicas have to stay in lockstep.
void verify(const Grads& grads, const autograd::TensorPtr& norm, float max_norm, float p = 2.0F) {
    const size_t num_devices = autograd::ctx().get_device().num_devices();
    const float expected_norm = reference_norm(grads, p);
    const float coef = std::min(1.0F, max_norm / (expected_norm + 1e-6F));

    const auto norms = norm_per_device(norm);
    ASSERT_EQ(norms.size(), num_devices);
    for (size_t d = 0; d < num_devices; ++d) {
        EXPECT_NEAR(norms[d](0), expected_norm, kRtol * expected_norm + kAtol) << "norm on device " << d;
        EXPECT_EQ(norms[d](0), norms[0](0)) << "norm on device " << d << " differs from device 0";
    }

    for (const auto& [name, host] : grads.host) {
        const auto per_device = core::to_xtensor(grads.params.at(name)->get_grad(), core::IdentityComposer{});
        ASSERT_EQ(per_device.size(), num_devices) << name;
        for (size_t d = 0; d < num_devices; ++d) {
            expect_close(per_device[d], host.pieces[d] * coef, fmt::format("{} on device {}", name, d));
            for (size_t first = 0; first < d; ++first) {
                if (xt::all(xt::equal(host.pieces[first], host.pieces[d]))) {
                    EXPECT_TRUE(xt::all(xt::equal(per_device[d], per_device[first])))
                        << name << " on device " << d << " differs from device " << first
                        << ", which held the same piece";
                    break;
                }
            }
        }
    }
}

// Scales by 8 the grads of one group that step1's second launch handles: those past the first `first_launch` of the
// group in the map's iteration order, which is the order clip_grad_norm collects them in. Unweighted, losing those
// few tensors would move the norm by less than kRtol.
void boost_second_launch(Grads& grads, const std::string& prefix, uint32_t first_launch) {
    uint32_t position = 0;
    for (auto& [name, param] : grads.params) {
        if (!name.starts_with(prefix) || position++ < first_launch) {
            continue;
        }
        param->set_grad(ttnn::multiply(param->get_grad(), 8.0F));
        auto& host = grads.host.at(name);
        host.full *= 8.0F;
        for (auto& piece : host.pieces) {
            piece *= 8.0F;
        }
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
    const uint32_t cores = test_utils::num_worker_cores();
    const uint32_t extra = param.exceed_cores ? cores + 1U : 0U;
    auto grads = make_grads(param.num_sharded + extra, param.num_replicated + extra);
    if (param.exceed_cores) {
        boost_second_launch(grads, "sharded_", cores);
        boost_second_launch(grads, "replicated_", cores);
    }
    const float max_norm = param.max_norm_ratio * reference_norm(grads, param.p_norm_type);

    auto norm = clip(grads.params, max_norm, param.p_norm_type);

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
    clip(first.params, 1.0F);
    const auto entries_after_first = device->num_program_cache_entries();

    auto second = make_grads(2, 2);
    auto norm = clip(second.params, 1.0F);

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

    auto norm = clip(grads.params, max_norm);

    verify(grads, norm, max_norm);
}

// Shapes that don't fill whole tiles: the tile padding must not leak into the norm. from_xtensor zero-pads, which
// would hide a leak (|0|^p = 0), so the padding is filled with a large value first.
TEST_F(ClipGradNormTpTest, UnalignedShapes) {
    auto grads = make_grads({{1U, 1U, 20U, 80U}}, {{1U, 1U, 17U, 40U}, {1U, 3U, 5U, 7U}});
    poison_padding(grads.params);
    const float max_norm = 0.5F * reference_norm(grads);

    auto norm = clip(grads.params, max_norm);

    verify(grads, norm, max_norm);
}

// Sharded grads whose params carry a single placement for the whole mesh, as GPT-2's TP linears produce.
TEST_F(ClipGradNormTpTest, CollapsedShardPlacement) {
    auto grads = make_grads({}, std::vector<Shape>(2, Shape{1U, 1U, 32U, 32U}));
    const auto collapsed = shard_mapper(kShardDim, /*cluster_axis=*/std::nullopt);
    for (size_t i = 0; i < 2; ++i) {
        add_grad(
            grads,
            fmt::format("sharded_{}", i),
            scale_second_half(random_bf16_exact({1U, 1U, 32U, 32U * kTpSize}), kShardDim, 2.0F),
            *collapsed);
    }
    ASSERT_EQ(grads.params.at("sharded_0")->get_value().tensor_topology().placements().size(), 1U);
    const float max_norm = 0.5F * reference_norm(grads);

    auto norm = clip(grads.params, max_norm);

    verify(grads, norm, max_norm);
}

// A real TP block rather than synthetic grads: DistributedGPTMLP's column- and row-parallel linears after a forward
// and backward pass, against the host norm of their gathered grads. Which parameters are sharded comes from the
// block's structure here, not from the placements clip_grad_norm reads.
TEST_F(ClipGradNormTpTest, DistributedGptMlpMatchesGatheredGrads) {
    constexpr uint32_t kEmbedding = 64U;
    modules::distributed::DistributedGPTMLP mlp(kEmbedding, /*dropout_prob=*/0.0F);
    const auto replicated = replicate_mapper();
    auto input = autograd::create_tensor(distribute(random_bf16_exact({2U, 1U, 32U, kEmbedding}), *replicated));
    auto output = mlp(input);
    output->set_grad(distribute(random_bf16_exact({2U, 1U, 32U, kEmbedding}), *replicated));
    output->backward();

    // The dim each sharded parameter is split on; fc2's bias is the block's one replicated parameter.
    const std::map<std::string, int> sharded_dims = {
        {"gpt_mlp/fc1/weight", 2}, {"gpt_mlp/fc1/bias", 3}, {"gpt_mlp/fc2/weight", 3}};
    const std::string replicated_name = "gpt_mlp/fc2/bias";
    Grads grads;
    grads.params = mlp.parameters();
    ASSERT_EQ(grads.params.size(), sharded_dims.size() + 1U);
    for (const auto& [name, param] : grads.params) {
        auto pieces = core::to_xtensor(param->get_grad(), core::IdentityComposer{});
        ASSERT_EQ(pieces.size(), kTpSize) << name;
        xt::xarray<float> full;
        if (name == replicated_name) {
            ASSERT_TRUE(xt::all(xt::equal(pieces[0], pieces[1]))) << name << " differs across devices";
            full = pieces[0];
        } else {
            ASSERT_TRUE(sharded_dims.contains(name)) << "unexpected parameter " << name;
            full = xt::concatenate(xt::xtuple(pieces[0], pieces[1]), sharded_dims.at(name));
        }
        grads.host[name] = HostGrad{.full = std::move(full), .pieces = std::move(pieces)};
    }
    const float max_norm = 0.5F * reference_norm(grads);

    auto norm = clip(grads.params, max_norm);

    verify(grads, norm, max_norm);
}

// Column-parallel (dim 2), row-parallel (dim 3) and replicated grads are clipped in one step3 launch; each grad must
// keep its own label rather than one merged from all the launch's inputs.
TEST_F(ClipGradNormTpTest, PreservesGradTopology) {
    serialization::NamedParameters params;
    params["dim2"] = make_param(distribute(random_bf16_exact({1U, 1U, 32U * kTpSize, 32U}), *shard_mapper(2)));
    params["dim3"] = make_param(distribute(random_bf16_exact({1U, 1U, 32U, 32U * kTpSize}), *shard_mapper(3)));
    params["replicated"] = make_param(distribute(random_bf16_exact({1U, 1U, 32U, 32U}), *replicate_mapper()));

    std::unordered_map<std::string, std::string> before;
    for (const auto& [name, param] : params) {
        before[name] = fmt::format("{}", param->get_grad().tensor_topology().placements());
    }

    clip(params, 1.0F);

    for (const auto& [name, param] : params) {
        EXPECT_EQ(fmt::format("{}", param->get_grad().tensor_topology().placements()), before.at(name)) << name;
    }
}

// A zero norm hits the 1e-6 guard: the coefficient clamps to 1 and the gradients stay zero instead of NaN.
TEST_F(ClipGradNormTpTest, ZeroGradients) {
    auto grads = make_grads(
        std::vector<Shape>(2, Shape{1U, 1U, 32U, 32U * kTpSize}),
        std::vector<Shape>(2, Shape{1U, 1U, 32U, 32U}),
        [](const Shape& shape) { return xt::xarray<float>(xt::zeros<float>(shape)); });

    auto norm = clip(grads.params, 1.0F);

    verify(grads, norm, 1.0F);
}

// error_if_nonfinite reads the norm back from every device and throws before anything is scaled; without it, the
// non-finite norm is returned. A finite norm must pass the check on a mesh.
TEST_F(ClipGradNormTpTest, NonfiniteNorm) {
    using ::testing::HasSubstr;
    using ::testing::ThrowsMessage;
    constexpr float kInf = std::numeric_limits<float>::infinity();

    auto finite = make_grads(1, 1);
    const float max_norm = 0.5F * reference_norm(finite);
    autograd::TensorPtr norm;
    ASSERT_NO_THROW(norm = core::clip_grad_norm(finite.params, max_norm, 2.0F, /*error_if_nonfinite=*/true));
    verify(finite, norm, max_norm);

    // Only device 0's piece holds the inf; the all-reduce carries it to every device.
    auto full = random_bf16_exact({1U, 1U, 32U, 32U * kTpSize});
    full(0, 0, 0, 0) = kInf;
    serialization::NamedParameters sharded;
    sharded["inf"] = make_param(distribute(full, *shard_mapper(kShardDim)));
    EXPECT_THAT(
        [&] { core::clip_grad_norm(sharded, 1.0F, 2.0F, /*error_if_nonfinite=*/true); },
        ThrowsMessage<std::runtime_error>(HasSubstr("is non-finite on device 0")));
    for (const auto& device_norm : norm_per_device(clip(sharded, 1.0F))) {
        EXPECT_FALSE(std::isfinite(device_norm(0)));
    }

    // A replicated parameter whose copies of the grad have diverged, with the inf only in device 1's: device 0's norm
    // stays finite, so a check that read device 0 alone would let this through.
    auto diverged = random_bf16_exact({1U, 1U, 32U, 32U * kTpSize});
    diverged(0, 0, 0, 32U * kTpSize - 1U) = kInf;  // in device 1's half
    auto param = autograd::create_tensor(
        distribute(xt::xarray<float>(xt::zeros<float>(Shape{1U, 1U, 32U, 32U})), *replicate_mapper()));
    param->set_grad(distribute(diverged, *shard_mapper(kShardDim)));
    param->set_requires_grad(true);
    serialization::NamedParameters replicated;
    replicated["diverged"] = param;
    EXPECT_THAT(
        [&] { core::clip_grad_norm(replicated, 1.0F, 2.0F, /*error_if_nonfinite=*/true); },
        ThrowsMessage<std::runtime_error>(HasSubstr("is non-finite on device 1")));
}

// Parameters that don't require a grad, or have none yet, are left out of the norm and left untouched.
TEST_F(ClipGradNormTpTest, SkipsParamsWithoutGrad) {
    auto grads = make_grads(1, 1);
    const auto frozen_values = random_bf16_exact({1U, 1U, 32U, 32U});
    // set_requires_grad(false) drops the grad, so set the grad on a tensor that never required one.
    auto frozen = autograd::create_tensor(core::zeros(ttnn::Shape({1U, 1U, 32U, 32U}), &autograd::ctx().get_device()));
    frozen->set_grad(distribute(frozen_values, *replicate_mapper()));
    grads.params["frozen"] = frozen;
    auto no_grad = autograd::create_tensor(core::zeros(ttnn::Shape({1U, 1U, 32U, 32U}), &autograd::ctx().get_device()));
    no_grad->set_requires_grad(true);
    grads.params["no_grad"] = no_grad;
    const float max_norm = 0.5F * reference_norm(grads);

    auto norm = clip(grads.params, max_norm);

    verify(grads, norm, max_norm);
    EXPECT_FALSE(no_grad->is_grad_initialized());
    for (const auto& values : core::to_xtensor(frozen->get_grad(), core::IdentityComposer{})) {
        EXPECT_TRUE(xt::all(xt::equal(values, frozen_values))) << "frozen grad was scaled";
    }
}

TEST_F(ClipGradNormTpTest, RejectsInvalidArguments) {
    auto grads = make_grads(1, 1);
    const auto& params = grads.params;
    constexpr float kInf = std::numeric_limits<float>::infinity();
    constexpr float kNan = std::numeric_limits<float>::quiet_NaN();

    for (float p : {0.0F, -1.0F, kInf, kNan}) {
        EXPECT_THROW(clip(params, 1.0F, p), std::logic_error) << "p_norm_type " << p;
    }
    EXPECT_THROW(clip({}, 1.0F), std::logic_error) << "no params";

    auto* device = &autograd::ctx().get_device();
    const auto mapper = replicate_mapper();
    const auto values = random_bf16_exact({1U, 1U, 32U, 32U});

    auto with_fp32 = params;
    with_fp32["fp32"] = make_param(
        core::from_xtensor<float, ttnn::DataType::FLOAT32>(values, device, ttnn::Layout::TILE, mapper.get()));
    EXPECT_THROW(clip(with_fp32, 1.0F), std::logic_error) << "dtype mismatch";

    auto with_2d = params;
    xt::xarray<float> values_2d = xt::reshape_view(values, {32U, 32U});
    with_2d["2d"] = make_param(distribute(values_2d, *mapper));
    EXPECT_THROW(clip(with_2d, 1.0F), std::logic_error) << "2D grad";
}

// DP x TP: DP on mesh axis 0 and TP on axis 1, as after DDP gradient sync.
class ClipGradNormTp2x2Test : public test_utils::MeshFixture {
protected:
    ClipGradNormTp2x2Test() : MeshFixture(tt::tt_metal::distributed::MeshShape(2, 2), /*seed=*/42) {
    }
};

// DDP sync leaves the grads replicated across DP, so each DP row reduces its own copy over the TP axis and must reach
// the same result. Reducing over the DP axis instead would add each TP piece to its own copy.
TEST_F(ClipGradNormTp2x2Test, EveryDpRowMatchesHostReference) {
    auto grads = make_grads(2, 4);
    const float max_norm = 0.5F * reference_norm(grads);

    auto norm = clip(grads.params, max_norm);

    verify(grads, norm, max_norm);
}

// Grads sharded on the DP axis (FSDP), on both axes (FSDP x TP), or across the whole mesh under a single placement are
// each reduced over the axes they are sharded on, alongside TP-sharded and replicated ones.
TEST_F(ClipGradNormTp2x2Test, ShardedOnAnyAxesMatchesHostReference) {
    using Shard = ttnn::distributed::MeshMapperConfig::Shard;
    auto grads = make_grads(1, 1);
    add_grad(
        grads,
        "fsdp",
        scale_second_half(random_bf16_exact({1U, 1U, 32U, 64U}), kShardDim, 4.0F),
        *shard_mapper(kShardDim, /*cluster_axis=*/0));
    const auto fsdp_tp = ttnn::distributed::create_mesh_mapper(
        autograd::ctx().get_device(), ttnn::distributed::MeshMapperConfig{.placements = {Shard{2}, Shard{kShardDim}}});
    add_grad(
        grads,
        "fsdp_tp",
        scale_second_half(scale_second_half(random_bf16_exact({1U, 1U, 64U, 64U}), 2, 4.0F), kShardDim, 2.0F),
        *fsdp_tp);
    add_grad(
        grads,
        "whole_mesh",
        scale_second_half(random_bf16_exact({1U, 1U, 32U, 128U}), kShardDim, 2.0F),
        *shard_mapper(kShardDim, /*cluster_axis=*/std::nullopt));
    const float max_norm = 0.5F * reference_norm(grads);

    auto norm = clip(grads.params, max_norm);

    verify(grads, norm, max_norm);
}

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

// step1 accumulates each gradient's Sum[|g|^p] in bf16, so the norm drifts low as gradients grow: about 0.6x the
// true norm at 4096 tiles (#63505). TODO(tstepanovic): enable with the ttml::metal op, which accumulates in fp32
// (#64517).
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

    auto norm = clip(params, 1.0F);

    EXPECT_NEAR(core::to_vector(norm->get_value()).front(), expected, kRtol * expected);
}
