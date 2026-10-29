// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cmath>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/generators/xrandom.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/clip_grad_norm.hpp"
#include "core/tt_tensor_utils.hpp"
#include "test_utils/mesh_utils.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"

using namespace ttml;

namespace {

// Mesh is 1x2, so tensor parallelism runs along mesh axis 1 and sharded tensors are split on their last dim.
constexpr uint32_t kClusterAxis = 1U;
constexpr int kShardDim = 3;
constexpr uint32_t kTpSize = 2U;

// Every intermediate (step1 partial sums, their sum, sqrt, the coefficient, the scaled grads) is stored in
// bf16, so each stage rounds to 2^-8 relative. Five such stages bound the error at ~2e-2.
constexpr float kRtol = 2e-2F;
constexpr float kAtol = 1e-3F;

struct ClipGradNormTpCase {
    std::string name;
    uint32_t num_sharded;
    uint32_t num_replicated;
    float max_norm_ratio;  // max_norm = ratio * true norm; > 1 means nothing is clipped
    // Adds (compute cores + 1) tensors to each group so step1/step3 need more than one launch per group.
    bool exceed_cores = false;
};

uint32_t num_compute_cores() {
    const auto grid = autograd::ctx().get_device().compute_with_storage_grid_size();
    return static_cast<uint32_t>(grid.x * grid.y);
}

// Values are multiples of 1/8 in [-4, 4], exactly representable in bf16, so inputs carry no rounding error.
xt::xarray<float> random_bf16_exact(const std::vector<size_t>& shape) {
    xt::xarray<float> values = xt::random::randint<int>(shape, -32, 33);
    return values / 8.0F;
}

struct TpGrads {
    std::vector<xt::xarray<float>> sharded_full;  // full (unsharded) grads, shape [1,1,32,64]
    std::vector<xt::xarray<float>> replicated;    // shape [1,1,32,32]
    serialization::NamedParameters sharded_params;
    serialization::NamedParameters replicated_params;
};

autograd::TensorPtr make_param(const ttnn::Tensor& grad) {
    auto param = autograd::create_tensor(core::zeros(grad.logical_shape(), grad.device()));
    param->set_grad(grad);
    param->set_requires_grad(true);
    return param;
}

TpGrads make_grads(uint32_t num_sharded, uint32_t num_replicated) {
    auto* device = &autograd::ctx().get_device();
    const auto shard_mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, kShardDim, kClusterAxis);
    const auto replicate_mapper = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);

    TpGrads grads;
    for (uint32_t i = 0; i < num_sharded; ++i) {
        auto full = random_bf16_exact({1U, 1U, 32U, 32U * kTpSize});
        grads.sharded_params[fmt::format("sharded_{}", i)] =
            make_param(core::from_xtensor(full, device, ttnn::Layout::TILE, shard_mapper.get()));
        grads.sharded_full.push_back(std::move(full));
    }
    for (uint32_t i = 0; i < num_replicated; ++i) {
        auto values = random_bf16_exact({1U, 1U, 32U, 32U});
        grads.replicated_params[fmt::format("replicated_{}", i)] =
            make_param(core::from_xtensor(values, device, ttnn::Layout::TILE, replicate_mapper.get()));
        grads.replicated.push_back(std::move(values));
    }
    return grads;
}

// Global L2 norm over the full tensors, each replicated tensor counted once.
float reference_norm(const TpGrads& grads) {
    float pow_sum = 0.0F;
    for (const auto& g : grads.sharded_full) {
        pow_sum += xt::sum(g * g)();
    }
    for (const auto& g : grads.replicated) {
        pow_sum += xt::sum(g * g)();
    }
    return std::sqrt(pow_sum);
}

void expect_close(const xt::xarray<float>& actual, const xt::xarray<float>& expected, const std::string& label) {
    EXPECT_TRUE(xt::allclose(actual, expected, kRtol, kAtol)) << label;
}

// Compares the on-device result against the host reference: the norm must match on every device and the
// gradients, once gathered back, must equal the host grads times the expected coefficient.
void verify(const TpGrads& grads, const autograd::TensorPtr& norm, float max_norm) {
    auto* device = &autograd::ctx().get_device();
    const float expected_norm = reference_norm(grads);
    const float coef = std::min(1.0F, max_norm / (expected_norm + 1e-6F));

    const auto norm_per_device = core::to_xtensor(norm->get_value(), core::IdentityComposer{});
    ASSERT_EQ(norm_per_device.size(), kTpSize);
    for (size_t d = 0; d < kTpSize; ++d) {
        EXPECT_NEAR(norm_per_device[d](0), expected_norm, kRtol * expected_norm) << "norm on device " << d;
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
    const float max_norm = param.max_norm_ratio * reference_norm(grads);

    auto norm = core::clip_grad_norm_tp(grads.sharded_params, grads.replicated_params, max_norm, kClusterAxis);

    verify(grads, norm, max_norm);
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
            .exceed_cores = true}),
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
