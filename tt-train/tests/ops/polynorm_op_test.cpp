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
#include <iostream>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/random.hpp"
#include "core/system_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "modules/linear_module.hpp"
#include "ops/binary_ops.hpp"
#include "ops/losses.hpp"
#include "optimizers/sgd.hpp"

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

enum class PolyNormForwardImpl {
    CompositeOp,
    FusedMetalKernel,
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

float relative_l2(const xt::xarray<float>& test, const xt::xarray<float>& reference) {
    auto diff = test - reference;
    const float diff_l2 = std::sqrt(xt::sum(xt::square(diff))());
    const float ref_l2 = std::sqrt(xt::sum(xt::square(reference))());
    return diff_l2 / (ref_l2 + 1e-12F);
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
    const float rel_l2 = relative_l2(out_tt, out_ref);
    EXPECT_TRUE(allclose) << label << " allclose failed"
                          << " rtol=" << rtol << " atol=" << atol << " rel_l2=" << rel_l2 << " rmse=" << rmse
                          << " max_abs_diff=" << max_abs_diff;
}

xt::xarray<float> run_polynorm_forward(const PolyNormCaseData& data, PolyNormForwardImpl impl, float epsilon = 1e-5F) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();

    if (impl == PolyNormForwardImpl::CompositeOp) {
        auto x_t = autograd::create_tensor(core::from_xtensor(data.input, device));
        auto w_t = autograd::create_tensor(core::from_xtensor(data.weight, device));
        auto b_t = autograd::create_tensor(core::from_xtensor(data.bias, device));
        auto out = ops::polynorm(x_t, w_t, b_t, epsilon);
        return core::to_xtensor(out->get_value());
    }

    const auto x = core::from_xtensor(data.input, device);
    const auto w0 = data.weight(0, 0, 0, 0);
    const auto w1 = data.weight(0, 0, 0, 1);
    const auto w2 = data.weight(0, 0, 0, 2);
    const auto bias = data.bias(0, 0, 0, 0);
    auto out = metal::polynorm_fw(x, w0, w1, w2, bias, epsilon);
    return core::to_xtensor(out);
}

void compare_kernel_vs_reference(
    const PolyNormCaseData& data, PolyNormForwardImpl impl = PolyNormForwardImpl::CompositeOp, float epsilon = 1e-5F) {
    auto out_tt = run_polynorm_forward(data, impl, epsilon);
    auto out_ref = polynorm_reference(data.input, data.weight, data.bias, epsilon);

    EXPECT_EQ(out_tt.shape(), out_ref.shape());
    EXPECT_TRUE(xt::all(xt::isfinite(out_tt)));
    EXPECT_TRUE(xt::all(xt::isfinite(out_ref)));
    expect_allclose_with_metrics(out_tt, out_ref, 8.0e-2F, 8.0e-2F, "forward_kernel_vs_reference");
}

float device_polynorm_mse_loss(
    const xt::xarray<float>& x, const xt::xarray<float>& weight, const xt::xarray<float>& bias, float epsilon) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();

    auto x_t = autograd::create_tensor(core::from_xtensor(x, device), /*requires_grad=*/false);
    auto w_t = autograd::create_tensor(core::from_xtensor(weight, device), /*requires_grad=*/false);
    auto b_t = autograd::create_tensor(core::from_xtensor(bias, device), /*requires_grad=*/false);
    auto out = ops::polynorm(x_t, w_t, b_t, epsilon);
    auto loss = ops::mse_loss(out, autograd::create_tensor(core::zeros_like(out->get_value())));
    return core::to_vector(loss->get_value()).front();
}

void append_params(
    ttml::serialization::NamedParameters& dst,
    const ttml::serialization::NamedParameters& src,
    const std::string& prefix) {
    for (const auto& [name, tensor] : src) {
        dst.emplace(prefix + name, tensor);
    }
}

}  // namespace

// ============================================================================
// Section 1: PolyNorm Forward - Kernel vs Reference
// ============================================================================
TEST_F(PolyNormOpTest, Forward_BasicSmall) {
    compare_kernel_vs_reference(make_case_data({2, 1, 4, 16}));
}

TEST_F(PolyNormOpTest, Forward_BlockSize_Remainder0) {
    compare_kernel_vs_reference(make_case_data({1, 1, 1, 128}));
}

TEST_F(PolyNormOpTest, Forward_BlockSize_Remainder1) {
    compare_kernel_vs_reference(make_case_data({1, 1, 1, 160}));
}

TEST_F(PolyNormOpTest, Forward_BlockSize_Remainder2) {
    compare_kernel_vs_reference(make_case_data({1, 1, 1, 192}));
}

TEST_F(PolyNormOpTest, Forward_BlockSize_Remainder3) {
    compare_kernel_vs_reference(make_case_data({1, 1, 1, 224}));
}

TEST_F(PolyNormOpTest, Forward_NanoLikeShape) {
    compare_kernel_vs_reference(make_case_data({2, 1, 64, 512}));
}

TEST_F(PolyNormOpTest, Forward_MultiBatchMultiSeq) {
    compare_kernel_vs_reference(make_case_data({4, 1, 128, 768}));
}

TEST_F(PolyNormOpTest, Forward_RepeatedRuns_NoHang) {
    const auto data = make_case_data({57, 1, 32, 96});
    for (int i = 0; i < 3; ++i) {
        compare_kernel_vs_reference(data);
    }
}

// Debug-only fused-kernel tests.
// Keep disabled until the dedicated fused forward path is made numerically stable.
TEST_F(PolyNormOpTest, DISABLED_FusedForward_BasicSmall) {
    compare_kernel_vs_reference(make_case_data({2, 1, 4, 16}), PolyNormForwardImpl::FusedMetalKernel);
}

TEST_F(PolyNormOpTest, DISABLED_FusedForward_LargeChannels) {
    compare_kernel_vs_reference(make_case_data({1, 1, 1, 128}), PolyNormForwardImpl::FusedMetalKernel);
}

TEST_F(PolyNormOpTest, DISABLED_FusedForward_DiagnosticTermOrdering) {
    constexpr float eps = 1e-5F;
    const auto data = make_case_data({1, 1, 1, 128});
    const auto fused_out = run_polynorm_forward(data, PolyNormForwardImpl::FusedMetalKernel, eps);

    auto make_weight = [](float w0, float w1, float w2) {
        xt::xarray<float> w = xt::xarray<float>::from_shape({1, 1, 1, 3});
        w(0, 0, 0, 0) = w0;
        w(0, 0, 0, 1) = w1;
        w(0, 0, 0, 2) = w2;
        return w;
    };

    const auto a = data.weight(0, 0, 0, 0);
    const auto b = data.weight(0, 0, 0, 1);
    const auto c = data.weight(0, 0, 0, 2);

    const std::array<std::pair<const char*, xt::xarray<float>>, 6> candidates{{
        {"w0,w1,w2", make_weight(a, b, c)},
        {"w0,w2,w1", make_weight(a, c, b)},
        {"w1,w0,w2", make_weight(b, a, c)},
        {"w1,w2,w0", make_weight(b, c, a)},
        {"w2,w0,w1", make_weight(c, a, b)},
        {"w2,w1,w0", make_weight(c, b, a)},
    }};

    for (const auto& [label, w] : candidates) {
        const auto ref = polynorm_reference(data.input, w, data.bias, eps);
        std::cout << "candidate=" << label << " rel_l2=" << relative_l2(fused_out, ref) << std::endl;
    }
}

TEST_F(PolyNormOpTest, DISABLED_FusedForward_DiagnosticSingleTerm) {
    constexpr float eps = 1e-5F;
    auto data = make_case_data({1, 1, 1, 128});
    data.bias(0, 0, 0, 0) = 0.0F;

    auto make_weight = [](float w0, float w1, float w2) {
        xt::xarray<float> w = xt::xarray<float>::from_shape({1, 1, 1, 3});
        w(0, 0, 0, 0) = w0;
        w(0, 0, 0, 1) = w1;
        w(0, 0, 0, 2) = w2;
        return w;
    };

    const std::array<std::pair<const char*, xt::xarray<float>>, 3> terms{{
        {"x3_only", make_weight(1.0F, 0.0F, 0.0F)},
        {"x2_only", make_weight(0.0F, 1.0F, 0.0F)},
        {"x_only", make_weight(0.0F, 0.0F, 1.0F)},
    }};

    for (const auto& [label, w] : terms) {
        data.weight = w;
        const auto fused_out = run_polynorm_forward(data, PolyNormForwardImpl::FusedMetalKernel, eps);
        const auto ref = polynorm_reference(data.input, w, data.bias, eps);
        std::cout << "single_term=" << label << " rel_l2=" << relative_l2(fused_out, ref) << std::endl;
    }
}

TEST_F(PolyNormOpTest, DISABLED_FusedForward_DiagnosticInputScaleSweep) {
    constexpr float eps = 1e-5F;
    const std::vector<std::pair<float, float>> ranges = {
        {-1.0e-3F, 1.0e-3F},
        {-0.9F, 0.9F},
        {-3.0F, 3.0F},
        {-8.0F, 8.0F},
    };

    auto make_weight = [](float w0, float w1, float w2) {
        xt::xarray<float> w = xt::xarray<float>::from_shape({1, 1, 1, 3});
        w(0, 0, 0, 0) = w0;
        w(0, 0, 0, 1) = w1;
        w(0, 0, 0, 2) = w2;
        return w;
    };

    const std::array<std::pair<const char*, xt::xarray<float>>, 3> terms{{
        {"x3_only", make_weight(1.0F, 0.0F, 0.0F)},
        {"x2_only", make_weight(0.0F, 1.0F, 0.0F)},
        {"x_only", make_weight(0.0F, 0.0F, 1.0F)},
    }};

    for (const auto& [low, high] : ranges) {
        PolyNormCaseData data{
            .input = make_random_xtensor({1, 1, 1, 128}, low, high),
            .weight = xt::xarray<float>::from_shape({1, 1, 1, 3}),
            .bias = xt::xarray<float>::from_shape({1, 1, 1, 1}),
        };
        data.weight(0, 0, 0, 0) = 0.2F;
        data.weight(0, 0, 0, 1) = 0.3F;
        data.weight(0, 0, 0, 2) = 0.5F;
        data.bias(0, 0, 0, 0) = 0.1F;

        const auto fused_out = run_polynorm_forward(data, PolyNormForwardImpl::FusedMetalKernel, eps);
        const auto ref_out = polynorm_reference(data.input, data.weight, data.bias, eps);
        const float rel_full = relative_l2(fused_out, ref_out);
        const bool fused_finite = xt::all(xt::isfinite(fused_out));
        const bool ref_finite = xt::all(xt::isfinite(ref_out));
        std::cout << "range=[" << low << "," << high << "]"
                  << " full_rel_l2=" << rel_full << " finite_fused=" << fused_finite << " finite_ref=" << ref_finite
                  << std::endl;

        for (const auto& [label, w] : terms) {
            data.weight = w;
            data.bias(0, 0, 0, 0) = 0.0F;
            const auto fused_term = run_polynorm_forward(data, PolyNormForwardImpl::FusedMetalKernel, eps);
            const auto ref_term = polynorm_reference(data.input, w, data.bias, eps);
            const float rel_term = relative_l2(fused_term, ref_term);
            const bool term_finite = xt::all(xt::isfinite(fused_term));
            std::cout << "  term=" << label << " rel_l2=" << rel_term << " finite=" << term_finite << std::endl;
        }
    }
}

// ============================================================================
// Section 2: PolyNorm Backward - Deferred While Forward Bring-Up
// ============================================================================
TEST_F(PolyNormOpTest, DISABLED_BackwardMatchesFiniteDifferences) {
    using namespace ttml;

    const std::vector<uint32_t> shape = {1, 1, 2, 8};
    auto& rng = autograd::ctx().get_generator();

    xt::xarray<float> x = xt::empty<float>(shape);
    core::parallel_generate<float>(x, []() { return std::uniform_real_distribution<float>(-0.9F, 0.9F); }, rng());
    xt::xarray<float> weight = xt::xarray<float>::from_shape({1, 1, 1, 3});
    weight(0, 0, 0, 0) = 0.33F;
    weight(0, 0, 0, 1) = 0.31F;
    weight(0, 0, 0, 2) = 0.36F;
    xt::xarray<float> bias = xt::xarray<float>::from_shape({1, 1, 1, 1});
    bias(0, 0, 0, 0) = 0.05F;

    auto x_t = autograd::create_tensor(core::from_xtensor(x, &autograd::ctx().get_device()), /*requires_grad=*/true);
    auto w_t =
        autograd::create_tensor(core::from_xtensor(weight, &autograd::ctx().get_device()), /*requires_grad=*/true);
    auto b_t = autograd::create_tensor(core::from_xtensor(bias, &autograd::ctx().get_device()), /*requires_grad=*/true);

    constexpr float eps = 1e-5F;
    auto out = ops::polynorm(x_t, w_t, b_t, eps);
    auto loss = ops::mse_loss(out, autograd::create_tensor(core::zeros_like(out->get_value())));
    loss->backward();

    auto grad_x = core::to_xtensor(x_t->get_grad());
    auto grad_w = core::to_xtensor(w_t->get_grad());
    auto grad_b = core::to_xtensor(b_t->get_grad());

    constexpr float h = 5e-2F;

    xt::xarray<float> grad_x_num = xt::zeros_like(x);
    for (std::size_t i = 0; i < x.size(); ++i) {
        xt::xarray<float> x_plus = x;
        xt::xarray<float> x_minus = x;
        x_plus.flat(i) += h;
        x_minus.flat(i) -= h;
        const float lp = device_polynorm_mse_loss(x_plus, weight, bias, eps);
        const float lm = device_polynorm_mse_loss(x_minus, weight, bias, eps);
        grad_x_num.flat(i) = (lp - lm) / (2.0F * h);
    }

    xt::xarray<float> grad_w_num = xt::zeros_like(weight);
    for (std::size_t i = 0; i < weight.size(); ++i) {
        xt::xarray<float> w_plus = weight;
        xt::xarray<float> w_minus = weight;
        w_plus.flat(i) += h;
        w_minus.flat(i) -= h;
        const float lp = device_polynorm_mse_loss(x, w_plus, bias, eps);
        const float lm = device_polynorm_mse_loss(x, w_minus, bias, eps);
        grad_w_num.flat(i) = (lp - lm) / (2.0F * h);
    }

    xt::xarray<float> grad_b_num = xt::zeros_like(bias);
    {
        xt::xarray<float> b_plus = bias;
        xt::xarray<float> b_minus = bias;
        b_plus(0, 0, 0, 0) += h;
        b_minus(0, 0, 0, 0) -= h;
        const float lp = device_polynorm_mse_loss(x, weight, b_plus, eps);
        const float lm = device_polynorm_mse_loss(x, weight, b_minus, eps);
        grad_b_num(0, 0, 0, 0) = (lp - lm) / (2.0F * h);
    }

    auto rel_l2 = [](const xt::xarray<float>& a, const xt::xarray<float>& b) {
        auto diff = a - b;
        const float diff_l2 = std::sqrt(xt::sum(xt::square(diff))());
        const float ref_l2 = std::sqrt(xt::sum(xt::square(b))());
        return diff_l2 / (ref_l2 + 1e-12F);
    };

    const float x_rel_l2 = rel_l2(grad_x, grad_x_num);
    const float w_rel_l2 = rel_l2(grad_w, grad_w_num);
    const float b_rel_l2 = rel_l2(grad_b, grad_b_num);

    // Device BF16 math plus finite differences introduces higher noise.
    EXPECT_LT(x_rel_l2, 0.60F);
    EXPECT_LT(w_rel_l2, 0.35F);
    EXPECT_LT(b_rel_l2, 0.35F);
}

// ============================================================================
// Section 3: PolyNorm Shape/Broadcast Behavior (Deferred)
// ============================================================================
TEST_F(PolyNormOpTest, DISABLED_BroadcastShapesAndGradShapes) {
    using namespace ttml;

    xt::xarray<float> x = xt::ones<float>({2, 1, 3, 32});
    xt::xarray<float> weight = xt::xarray<float>::from_shape({1, 1, 1, 3});
    weight(0, 0, 0, 0) = 1.0F / 3.0F;
    weight(0, 0, 0, 1) = 1.0F / 3.0F;
    weight(0, 0, 0, 2) = 1.0F / 3.0F;
    xt::xarray<float> bias = xt::xarray<float>::from_shape({1, 1, 1, 1});
    bias(0, 0, 0, 0) = 0.0F;

    auto x_t = autograd::create_tensor(core::from_xtensor(x, &autograd::ctx().get_device()), /*requires_grad=*/true);
    auto w_t =
        autograd::create_tensor(core::from_xtensor(weight, &autograd::ctx().get_device()), /*requires_grad=*/true);
    auto b_t = autograd::create_tensor(core::from_xtensor(bias, &autograd::ctx().get_device()), /*requires_grad=*/true);

    auto out = ops::polynorm(x_t, w_t, b_t);
    auto loss = ops::mse_loss(out, autograd::create_tensor(core::zeros_like(out->get_value())));
    loss->backward();

    const auto expected_out_shape = std::array<uint32_t, 4>{2, 1, 3, 32};
    const auto expected_weight_grad_shape = std::array<uint32_t, 4>{1, 1, 1, 3};
    const auto expected_bias_grad_shape = std::array<uint32_t, 4>{1, 1, 1, 1};
    EXPECT_EQ(out->get_value().logical_shape().to_array_4D(), expected_out_shape);
    EXPECT_EQ(w_t->get_grad().logical_shape().to_array_4D(), expected_weight_grad_shape);
    EXPECT_EQ(b_t->get_grad().logical_shape().to_array_4D(), expected_bias_grad_shape);
}

// ============================================================================
// Section 4: Tiny Integration - PolyNorm in Gated FFN (Deferred)
// ============================================================================
TEST_F(PolyNormOpTest, DISABLED_TinyGatedFFNTrainsFewSteps) {
    using namespace ttml;

    constexpr uint32_t batch = 4;
    constexpr uint32_t seq = 8;
    constexpr uint32_t in_features = 32;
    constexpr uint32_t hidden = 64;
    constexpr uint32_t steps = 5;
    constexpr float eps = 1e-5F;

    auto& rng = autograd::ctx().get_generator();
    xt::xarray<float> input_xt = xt::xarray<float>::from_shape({batch, 1U, seq, in_features});
    xt::xarray<float> target_xt = xt::xarray<float>::from_shape({batch, 1U, seq, in_features});
    core::parallel_generate<float>(
        input_xt, []() { return std::uniform_real_distribution<float>(-1.0F, 1.0F); }, rng());
    core::parallel_generate<float>(
        target_xt, []() { return std::uniform_real_distribution<float>(-0.5F, 0.5F); }, rng());

    auto input = autograd::create_tensor(core::from_xtensor(input_xt, &autograd::ctx().get_device()));
    auto target = autograd::create_tensor(core::from_xtensor(target_xt, &autograd::ctx().get_device()));

    modules::LinearLayer w1(in_features, hidden, /*has_bias=*/false);
    modules::LinearLayer w2(hidden, in_features, /*has_bias=*/false);
    modules::LinearLayer w3(in_features, hidden, /*has_bias=*/false);

    auto poly_weight = autograd::create_tensor(
        core::from_vector(
            {1.0F / 3.0F, 1.0F / 3.0F, 1.0F / 3.0F}, ttnn::Shape({1, 1, 1, 3}), &autograd::ctx().get_device()),
        /*requires_grad=*/true);
    auto poly_bias = autograd::create_tensor(
        core::from_vector({0.0F}, ttnn::Shape({1, 1, 1, 1}), &autograd::ctx().get_device()),
        /*requires_grad=*/true);

    serialization::NamedParameters params;
    append_params(params, w1.parameters(), "w1/");
    append_params(params, w2.parameters(), "w2/");
    append_params(params, w3.parameters(), "w3/");
    params.emplace("polynorm/weight", poly_weight);
    params.emplace("polynorm/bias", poly_bias);

    optimizers::SGD optimizer(params, {.lr = 5e-2F, .momentum = 0.0F});

    float first_loss = 0.0F;
    float last_loss = 0.0F;
    for (uint32_t step = 0; step < steps; ++step) {
        optimizer.zero_grad();
        auto activated = ops::polynorm(w1(input), poly_weight, poly_bias, eps);
        auto gate = w3(input);
        auto gated = ops::mul(activated, gate);
        auto output = w2(gated);
        auto loss = ops::mse_loss(output, target);
        loss->backward();
        const float loss_value = core::to_vector(loss->get_value()).front();
        if (step == 0U) {
            first_loss = loss_value;
        }
        last_loss = loss_value;
        optimizer.step();
        autograd::ctx().reset_graph();
    }

    EXPECT_TRUE(std::isfinite(first_loss));
    EXPECT_TRUE(std::isfinite(last_loss));
    EXPECT_LE(last_loss, first_loss + 1.0F);
}
