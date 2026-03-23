// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/polynorm_op.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/random.hpp"
#include "core/system_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/losses.hpp"

class PolyNormOpTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
        ttml::autograd::ctx().set_seed(42);
    }

    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }
};

namespace {

struct PolyNormCaseData {
    xt::xarray<float> input;
    xt::xarray<float> weight;
    xt::xarray<float> bias;
};

xt::xarray<float> rms_norm_last_dim(const xt::xarray<float>& x, float epsilon) {
    auto shape = x.shape();
    auto out = xt::xarray<float>::from_shape(shape);
    std::fill(out.begin(), out.end(), 0.0F);
    const auto b_dim = shape[0];
    const auto n_dim = shape[1];
    const auto s_dim = shape[2];
    const auto c_dim = shape[3];

    for (std::size_t b = 0; b < b_dim; ++b) {
        for (std::size_t n = 0; n < n_dim; ++n) {
            for (std::size_t s = 0; s < s_dim; ++s) {
                float mean_sq = 0.0F;
                for (std::size_t c = 0; c < c_dim; ++c) {
                    const float v = x(b, n, s, c);
                    mean_sq += v * v;
                }
                mean_sq /= static_cast<float>(c_dim);
                const float inv_rms = 1.0F / std::sqrt(mean_sq + epsilon);
                for (std::size_t c = 0; c < c_dim; ++c) {
                    out(b, n, s, c) = x(b, n, s, c) * inv_rms;
                }
            }
        }
    }
    return out;
}

xt::xarray<float> polynorm_reference(
    const xt::xarray<float>& x, const xt::xarray<float>& weight, const xt::xarray<float>& bias, float epsilon) {
    const float w0 = weight(0, 0, 0, 0);
    const float w1 = weight(0, 0, 0, 1);
    const float w2 = weight(0, 0, 0, 2);
    const float b = bias(0, 0, 0, 0);

    auto x2 = xt::square(x);
    auto x3 = x * x2;
    return w0 * rms_norm_last_dim(x3, epsilon) + w1 * rms_norm_last_dim(x2, epsilon) +
           w2 * rms_norm_last_dim(x, epsilon) + b;
}

xt::xarray<float> make_random_xtensor(const std::vector<uint32_t>& shape, float low, float high) {
    auto& rng = ttml::autograd::ctx().get_generator();
    xt::xarray<float> x = xt::empty<float>(shape);
    ttml::core::parallel_generate<float>(
        x, [low, high]() { return std::uniform_real_distribution<float>(low, high); }, rng());
    return x;
}

PolyNormCaseData make_case_data(const std::vector<uint32_t>& input_shape) {
    PolyNormCaseData data{
        .input = make_random_xtensor(input_shape, -1.0F, 1.0F),
        .weight = xt::xarray<float>::from_shape({1, 1, 1, 3}),
        .bias = xt::xarray<float>::from_shape({1, 1, 1, 1}),
    };

    data.weight(0, 0, 0, 0) = 0.2F;
    data.weight(0, 0, 0, 1) = 0.3F;
    data.weight(0, 0, 0, 2) = 0.5F;
    data.bias(0, 0, 0, 0) = 0.1F;
    return data;
}

void expect_allclose_with_metrics(
    const xt::xarray<float>& out_tt,
    const xt::xarray<float>& out_ref,
    float rtol,
    float atol,
    const std::string& label) {
    const bool allclose = xt::allclose(out_tt, out_ref, rtol, atol);
    const auto abs_diff = xt::abs(out_tt - out_ref);
    const float max_abs_diff = xt::amax(abs_diff)();
    const float rmse = std::sqrt(xt::mean(xt::square(abs_diff))());
    EXPECT_TRUE(allclose) << label << " allclose failed"
                          << " rtol=" << rtol << " atol=" << atol << " rmse=" << rmse
                          << " max_abs_diff=" << max_abs_diff;
}

void CompareKernelVsReferenceWithShape(const std::vector<uint32_t>& shape, float epsilon = 1e-5F) {
    using namespace ttml;
    const auto data = make_case_data(shape);
    auto* device = &autograd::ctx().get_device();

    auto x = autograd::create_tensor(core::from_xtensor(data.input, device), /*requires_grad=*/true);
    auto w = autograd::create_tensor(core::from_xtensor(data.weight, device), /*requires_grad=*/true);
    auto b = autograd::create_tensor(core::from_xtensor(data.bias, device), /*requires_grad=*/true);

    auto out = ops::polynorm3(x, w, b, epsilon);
    const auto out_xt = core::to_xtensor(out->get_value());
    auto out_reference_xt = polynorm_reference(data.input, data.weight, data.bias, epsilon);

    EXPECT_EQ(out_xt.shape(), out_reference_xt.shape());
    EXPECT_TRUE(xt::all(xt::isfinite(out_xt)));
    EXPECT_TRUE(xt::all(xt::isfinite(out_reference_xt)));
    expect_allclose_with_metrics(out_xt, out_reference_xt, 8.0e-2F, 8.0e-2F, "fused_forward_vs_xt_reference");

    auto target = autograd::create_tensor(core::zeros_like(out->get_value()));
    auto mse = ops::mse_loss(out, target);
    mse->backward();

    const auto grad_x = core::to_xtensor(x->get_grad());
    const auto grad_w = core::to_xtensor(w->get_grad());
    const auto grad_b = core::to_xtensor(b->get_grad());

    EXPECT_EQ(grad_x.shape(), data.input.shape());
    EXPECT_EQ(grad_w.shape(), data.weight.shape());
    EXPECT_EQ(grad_b.shape(), data.bias.shape());

    EXPECT_TRUE(xt::all(xt::isfinite(grad_x)));
    EXPECT_TRUE(xt::all(xt::isfinite(grad_w)));
    EXPECT_TRUE(xt::all(xt::isfinite(grad_b)));

    autograd::ctx().reset_graph();
}

}  // namespace

// ============================================================================
// Section 1: PolyNorm Fused Kernel vs Reference
// ============================================================================
TEST_F(PolyNormOpTest, PolyNorm_Compare_BasicSmall) {
    CompareKernelVsReferenceWithShape({1, 1, 2, 32});
}

TEST_F(PolyNormOpTest, PolyNorm_Compare_BlockSizeRemainders) {
    CompareKernelVsReferenceWithShape({1, 1, 1, 32});   // Wt=1, Wt%4=1
    CompareKernelVsReferenceWithShape({1, 1, 1, 64});   // Wt=2, Wt%4=2
    CompareKernelVsReferenceWithShape({1, 1, 1, 96});   // Wt=3, Wt%4=3
    CompareKernelVsReferenceWithShape({1, 1, 1, 128});  // Wt=4, Wt%4=0
}

TEST_F(PolyNormOpTest, PolyNorm_Compare_EpsilonVariants) {
    CompareKernelVsReferenceWithShape({1, 1, 2, 64}, 1e-6F);
    CompareKernelVsReferenceWithShape({1, 1, 2, 64}, 1e-5F);
    CompareKernelVsReferenceWithShape({1, 1, 2, 64}, 1e-4F);
}

TEST_F(PolyNormOpTest, PolyNorm_RepeatedRuns_NoHang) {
    for (int i = 0; i < 2; ++i) {
        CompareKernelVsReferenceWithShape({8, 1, 8, 64});
    }
}

TEST_F(PolyNormOpTest, PolyNorm_FusedForwardRejectsNonTileAlignedChannels) {
    using namespace ttml;
    const auto data = make_case_data({1, 1, 2, 100});
    auto* device = &autograd::ctx().get_device();

    auto x = autograd::create_tensor(core::from_xtensor(data.input, device), /*requires_grad=*/true);
    auto w = autograd::create_tensor(core::from_xtensor(data.weight, device), /*requires_grad=*/true);
    auto b = autograd::create_tensor(core::from_xtensor(data.bias, device), /*requires_grad=*/true);

    EXPECT_THROW(
        {
            auto out = ops::polynorm3(x, w, b, 1e-5F);
            (void)out;
        },
        std::runtime_error);
    autograd::ctx().reset_graph();
}

// ============================================================================
// Section 2: Nightly Larger Shape Coverage
// ============================================================================
TEST_F(PolyNormOpTest, NIGHTLY_PolyNorm_Compare_ProgressiveSmall) {
    CompareKernelVsReferenceWithShape({1, 1, 16, 128}, 1e-5F);
}

TEST_F(PolyNormOpTest, NIGHTLY_PolyNorm_Compare_ProgressiveMedium) {
    CompareKernelVsReferenceWithShape({2, 1, 64, 512}, 1e-5F);
}

TEST_F(PolyNormOpTest, NIGHTLY_PolyNorm_Compare_ProgressiveLarge) {
    CompareKernelVsReferenceWithShape({4, 1, 128, 768}, 1e-5F);
}
