// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/polynorm_op.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <optional>
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

class ScopedPolyNormFusedForwardEnv {
public:
    explicit ScopedPolyNormFusedForwardEnv(bool enabled) {
        constexpr const char* env_name = "TTML_POLYNORM_USE_FUSED_FW";
        const char* previous = std::getenv(env_name);
        if (previous != nullptr) {
            m_previous = std::string(previous);
        }
        setenv(env_name, enabled ? "1" : "0", /*overwrite=*/1);
    }

    ~ScopedPolyNormFusedForwardEnv() {
        constexpr const char* env_name = "TTML_POLYNORM_USE_FUSED_FW";
        if (m_previous.has_value()) {
            setenv(env_name, m_previous->c_str(), /*overwrite=*/1);
            return;
        }
        unsetenv(env_name);
    }

private:
    std::optional<std::string> m_previous;
};

void CompareKernelVsReferenceWithShape(
    const std::vector<uint32_t>& shape, float epsilon = 1e-5F, bool run_backward = true) {
    using namespace ttml;
    const auto data = make_case_data(shape);
    auto* device = &autograd::ctx().get_device();

    auto x_fused = autograd::create_tensor(core::from_xtensor(data.input, device), /*requires_grad=*/true);
    auto w_fused = autograd::create_tensor(core::from_xtensor(data.weight, device), /*requires_grad=*/true);
    auto b_fused = autograd::create_tensor(core::from_xtensor(data.bias, device), /*requires_grad=*/true);

    auto x_ref = autograd::create_tensor(core::from_xtensor(data.input, device), /*requires_grad=*/true);
    auto w_ref = autograd::create_tensor(core::from_xtensor(data.weight, device), /*requires_grad=*/true);
    auto b_ref = autograd::create_tensor(core::from_xtensor(data.bias, device), /*requires_grad=*/true);

    std::shared_ptr<autograd::Tensor> out_fused;
    {
        ScopedPolyNormFusedForwardEnv env_guard(/*enabled=*/true);
        out_fused = ops::polynorm(x_fused, w_fused, b_fused, epsilon);
    }

    std::shared_ptr<autograd::Tensor> out_composite;
    {
        ScopedPolyNormFusedForwardEnv env_guard(/*enabled=*/false);
        out_composite = ops::polynorm(x_ref, w_ref, b_ref, epsilon);
    }

    auto out_fused_xt = core::to_xtensor(out_fused->get_value());
    auto out_composite_xt = core::to_xtensor(out_composite->get_value());
    auto out_reference_xt = polynorm_reference(data.input, data.weight, data.bias, epsilon);

    EXPECT_EQ(out_fused_xt.shape(), out_reference_xt.shape());
    EXPECT_EQ(out_composite_xt.shape(), out_reference_xt.shape());
    EXPECT_TRUE(xt::all(xt::isfinite(out_fused_xt)));
    EXPECT_TRUE(xt::all(xt::isfinite(out_composite_xt)));
    EXPECT_TRUE(xt::all(xt::isfinite(out_reference_xt)));
    expect_allclose_with_metrics(out_fused_xt, out_reference_xt, 8.0e-2F, 8.0e-2F, "fused_forward_vs_xt_reference");
    expect_allclose_with_metrics(
        out_composite_xt, out_reference_xt, 8.0e-2F, 8.0e-2F, "composite_forward_vs_xt_reference");
    expect_allclose_with_metrics(out_fused_xt, out_composite_xt, 8.0e-2F, 8.0e-2F, "fused_forward_vs_composite");

    if (run_backward) {
        auto target_fused = autograd::create_tensor(core::zeros_like(out_fused->get_value()));
        auto target_ref = autograd::create_tensor(core::zeros_like(out_composite->get_value()));
        auto mse_fused = ops::mse_loss(out_fused, target_fused);
        auto mse_ref = ops::mse_loss(out_composite, target_ref);
        mse_fused->backward();
        mse_ref->backward();

        const auto grad_x_fused = core::to_xtensor(x_fused->get_grad());
        const auto grad_w_fused = core::to_xtensor(w_fused->get_grad());
        const auto grad_b_fused = core::to_xtensor(b_fused->get_grad());
        const auto grad_x_ref = core::to_xtensor(x_ref->get_grad());
        const auto grad_w_ref = core::to_xtensor(w_ref->get_grad());
        const auto grad_b_ref = core::to_xtensor(b_ref->get_grad());

        EXPECT_EQ(grad_x_fused.shape(), data.input.shape());
        EXPECT_EQ(grad_w_fused.shape(), data.weight.shape());
        EXPECT_EQ(grad_b_fused.shape(), data.bias.shape());
        EXPECT_EQ(grad_x_ref.shape(), data.input.shape());
        EXPECT_EQ(grad_w_ref.shape(), data.weight.shape());
        EXPECT_EQ(grad_b_ref.shape(), data.bias.shape());

        EXPECT_TRUE(xt::all(xt::isfinite(grad_x_fused)));
        EXPECT_TRUE(xt::all(xt::isfinite(grad_w_fused)));
        EXPECT_TRUE(xt::all(xt::isfinite(grad_b_fused)));
        EXPECT_TRUE(xt::all(xt::isfinite(grad_x_ref)));
        EXPECT_TRUE(xt::all(xt::isfinite(grad_w_ref)));
        EXPECT_TRUE(xt::all(xt::isfinite(grad_b_ref)));

        expect_allclose_with_metrics(grad_x_fused, grad_x_ref, 2.0e-2F, 2.0e-2F, "grad_x_fused_vs_composite");
        expect_allclose_with_metrics(grad_w_fused, grad_w_ref, 2.0e-2F, 2.0e-2F, "grad_w_fused_vs_composite");
        expect_allclose_with_metrics(grad_b_fused, grad_b_ref, 2.0e-2F, 2.0e-2F, "grad_b_fused_vs_composite");
    }

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

// ============================================================================
// Section 2: Nightly Larger Shape Coverage
// ============================================================================
TEST_F(PolyNormOpTest, NIGHTLY_PolyNorm_Compare_ProgressiveSmall) {
    CompareKernelVsReferenceWithShape({1, 1, 16, 128}, 1e-5F, /*run_backward=*/false);
}

TEST_F(PolyNormOpTest, NIGHTLY_PolyNorm_Compare_ProgressiveMedium) {
    CompareKernelVsReferenceWithShape({2, 1, 64, 512}, 1e-5F, /*run_backward=*/false);
}

TEST_F(PolyNormOpTest, NIGHTLY_PolyNorm_Compare_ProgressiveLarge) {
    CompareKernelVsReferenceWithShape({4, 1, 128, 768}, 1e-5F, /*run_backward=*/false);
}
