// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Parity tests for PolyNorm3 forward and backward (fused metal kernel vs. composite TTNN ops).
// Each test compares both paths against an xtensor ground-truth reference implementation,
// checking forward output, grad_x, grad_w, and grad_b within tolerance. Section 2 relaunches the
// fused kernels on fresh tensors to check their program-cache hits.

#include "ops/polynorm_op.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <tt-metalium/bfloat16.hpp>
#include <tuple>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/ops/polynorm_bw/device/polynorm_bw_device_operation.hpp"
#include "metal/ops/polynorm_bw/polynorm_bw.hpp"
#include "metal/ops/polynorm_fw/device/polynorm_fw_device_operation.hpp"
#include "metal/ops/polynorm_fw/polynorm_fw.hpp"
#include "ops/losses.hpp"
#include "test_utils/program_cache.hpp"
#include "test_utils/random_data.hpp"

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
constexpr float kForwardRtol = 2.0e-2F;
constexpr float kForwardAtol = 2.0e-2F;
constexpr float kBackwardRtol = 1.0e-2F;
constexpr float kBackwardAtol = 1.0e-2F;

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

std::pair<xt::xarray<float>, xt::xarray<float>> rms_norm_with_inv_rms_last_dim(
    const xt::xarray<float>& x, float epsilon) {
    auto shape = x.shape();
    auto out = xt::xarray<float>::from_shape(shape);
    auto inv_rms = xt::xarray<float>::from_shape({shape[0], shape[1], shape[2], 1});
    std::fill(out.begin(), out.end(), 0.0F);
    std::fill(inv_rms.begin(), inv_rms.end(), 0.0F);

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
                const float inv = 1.0F / std::sqrt(mean_sq + epsilon);
                inv_rms(b, n, s, 0) = inv;
                for (std::size_t c = 0; c < c_dim; ++c) {
                    out(b, n, s, c) = x(b, n, s, c) * inv;
                }
            }
        }
    }
    return {out, inv_rms};
}

xt::xarray<float> grad_wrt_rmsnorm_input_reference(
    const xt::xarray<float>& term,
    const xt::xarray<float>& grad_normed_term,
    const xt::xarray<float>& inv_rms,
    float inv_channel_count) {
    auto shape = term.shape();
    auto out = xt::xarray<float>::from_shape(shape);
    std::fill(out.begin(), out.end(), 0.0F);

    const auto b_dim = shape[0];
    const auto n_dim = shape[1];
    const auto s_dim = shape[2];
    const auto c_dim = shape[3];

    for (std::size_t b = 0; b < b_dim; ++b) {
        for (std::size_t n = 0; n < n_dim; ++n) {
            for (std::size_t s = 0; s < s_dim; ++s) {
                float scale = 0.0F;
                for (std::size_t c = 0; c < c_dim; ++c) {
                    scale += term(b, n, s, c) * grad_normed_term(b, n, s, c);
                }
                const float inv = inv_rms(b, n, s, 0);
                const float inv3 = inv * inv * inv;
                for (std::size_t c = 0; c < c_dim; ++c) {
                    const float lhs = grad_normed_term(b, n, s, c) * inv;
                    const float rhs = term(b, n, s, c) * scale * inv3 * inv_channel_count;
                    out(b, n, s, c) = lhs - rhs;
                }
            }
        }
    }
    return out;
}

std::tuple<xt::xarray<float>, xt::xarray<float>, xt::xarray<float>> polynorm_reference_backward(
    const xt::xarray<float>& x, const xt::xarray<float>& weight, const xt::xarray<float>& dL_dout, float epsilon) {
    const float w0 = weight(0, 0, 0, 0);
    const float w1 = weight(0, 0, 0, 1);
    const float w2 = weight(0, 0, 0, 2);

    const auto x2 = xt::square(x);
    const auto x3 = x * x2;

    const auto [x_norm, x_inv_rms] = rms_norm_with_inv_rms_last_dim(x, epsilon);
    const auto [x2_norm, x2_inv_rms] = rms_norm_with_inv_rms_last_dim(x2, epsilon);
    const auto [x3_norm, x3_inv_rms] = rms_norm_with_inv_rms_last_dim(x3, epsilon);

    const float inv_channels = 1.0F / static_cast<float>(x.shape()[3]);

    const auto dL_dx_term1 = grad_wrt_rmsnorm_input_reference(x, dL_dout * w2, x_inv_rms, inv_channels);
    const auto dL_dx2 = grad_wrt_rmsnorm_input_reference(x2, dL_dout * w1, x2_inv_rms, inv_channels);
    const auto dL_dx_term2 = dL_dx2 * (2.0F * x);
    const auto dL_dx3 = grad_wrt_rmsnorm_input_reference(x3, dL_dout * w0, x3_inv_rms, inv_channels);
    const auto dL_dx_term3 = dL_dx3 * (3.0F * x2);
    const auto dL_dx = dL_dx_term1 + dL_dx_term2 + dL_dx_term3;

    xt::xarray<float> dL_dw = xt::xarray<float>::from_shape({1, 1, 1, 3});
    dL_dw(0, 0, 0, 0) = xt::sum(dL_dout * x3_norm)();
    dL_dw(0, 0, 0, 1) = xt::sum(dL_dout * x2_norm)();
    dL_dw(0, 0, 0, 2) = xt::sum(dL_dout * x_norm)();

    xt::xarray<float> dL_db = xt::xarray<float>::from_shape({1, 1, 1, 1});
    dL_db(0, 0, 0, 0) = xt::sum(dL_dout)();

    return {dL_dx, dL_dw, dL_db};
}

PolyNormCaseData make_case_data(const std::vector<uint32_t>& input_shape) {
    auto& rng = ttml::autograd::ctx().get_generator();
    PolyNormCaseData data{
        .input = ttml::test_utils::make_uniform_xarray<float>(input_shape, -1.0F, 1.0F, rng()),
        .weight = xt::xarray<float>::from_shape({1, 1, 1, 3}),
        .bias = xt::xarray<float>::from_shape({1, 1, 1, 1}),
    };

    data.weight(0, 0, 0, 0) = 0.2F;
    data.weight(0, 0, 0, 1) = 0.3F;
    data.weight(0, 0, 0, 2) = 0.5F;
    data.bias(0, 0, 0, 0) = 0.1F;
    return data;
}

enum class BackwardKernelVariant {
    FusedPartials,
};

std::string backward_variant_name(BackwardKernelVariant variant) {
    if (variant == BackwardKernelVariant::FusedPartials) {
        return "fused_partials";
    }
    return "fused_partials";
}

std::tuple<xt::xarray<float>, xt::xarray<float>, xt::xarray<float>> run_backward_kernel_variant(
    BackwardKernelVariant variant,
    const xt::xarray<float>& input,
    const xt::xarray<float>& dL_dout,
    const xt::xarray<float>& weight,
    float epsilon,
    ttnn::distributed::MeshDevice* device) {
    const auto input_tensor = ttml::core::from_xtensor(input, device);
    const auto dL_dout_tensor = ttml::core::from_xtensor(dL_dout, device);
    const auto weight_tensor = ttml::core::from_xtensor(weight, device);

    std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> outputs;
    outputs = ttml::metal::polynorm3_bw(input_tensor, dL_dout_tensor, weight_tensor, epsilon);
    auto grad_x = ttml::core::to_xtensor(std::get<0>(outputs));
    auto grad_w = ttml::core::to_xtensor(std::get<1>(outputs));
    auto grad_b = ttml::core::to_xtensor(std::get<2>(outputs));
    return {std::move(grad_x), std::move(grad_w), std::move(grad_b)};
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

void CompareKernelVsReferenceWithShape(
    const std::vector<uint32_t>& shape,
    float epsilon = 1e-5F,
    float backward_rtol = kBackwardRtol,
    float backward_atol = kBackwardAtol) {
    using namespace ttml;
    const auto data = make_case_data(shape);
    auto* device = &autograd::ctx().get_device();

    auto x = autograd::create_tensor(core::from_xtensor(data.input, device), /*requires_grad=*/true);
    auto w = autograd::create_tensor(core::from_xtensor(data.weight, device), /*requires_grad=*/true);
    auto b = autograd::create_tensor(core::from_xtensor(data.bias, device), /*requires_grad=*/true);

    auto out =
        ops::polynorm3(x, w, b, epsilon, ops::PolyNorm3ForwardVariant::Fused, ops::PolyNorm3BackwardVariant::Fused);
    const auto out_xt = core::to_xtensor(out->get_value());
    auto out_reference_xt = polynorm_reference(data.input, data.weight, data.bias, epsilon);

    EXPECT_EQ(out_xt.shape(), out_reference_xt.shape());
    EXPECT_TRUE(xt::all(xt::isfinite(out_xt)));
    EXPECT_TRUE(xt::all(xt::isfinite(out_reference_xt)));
    expect_allclose_with_metrics(out_xt, out_reference_xt, kForwardRtol, kForwardAtol, "fused_forward_vs_xt_reference");

    auto target = autograd::create_tensor(core::zeros_like(out->get_value()));
    auto mse = ops::mse_loss(out, target);
    mse->backward();

    const auto grad_x_autograd = core::to_xtensor(x->get_grad());
    const auto grad_w_autograd = core::to_xtensor(w->get_grad());
    const auto grad_b_autograd = core::to_xtensor(b->get_grad());
    const auto dL_dout = core::to_xtensor(out->get_grad());
    const auto [grad_x_ref, grad_w_ref, grad_b_ref] =
        polynorm_reference_backward(data.input, data.weight, dL_dout, epsilon);

    EXPECT_EQ(grad_x_autograd.shape(), data.input.shape());
    EXPECT_EQ(grad_w_autograd.shape(), data.weight.shape());
    EXPECT_EQ(grad_b_autograd.shape(), data.bias.shape());
    EXPECT_TRUE(xt::all(xt::isfinite(grad_x_autograd))) << "autograd fused grad_x has non-finite values";
    EXPECT_TRUE(xt::all(xt::isfinite(grad_w_autograd))) << "autograd fused grad_w has non-finite values";
    EXPECT_TRUE(xt::all(xt::isfinite(grad_b_autograd))) << "autograd fused grad_b has non-finite values";
    expect_allclose_with_metrics(
        grad_x_autograd, grad_x_ref, backward_rtol, backward_atol, "autograd_fused_backward_grad_x_vs_xt_reference");
    expect_allclose_with_metrics(
        grad_w_autograd, grad_w_ref, backward_rtol, backward_atol, "autograd_fused_backward_grad_w_vs_xt_reference");
    expect_allclose_with_metrics(
        grad_b_autograd, grad_b_ref, backward_rtol, backward_atol, "autograd_fused_backward_grad_b_vs_xt_reference");

    autograd::ctx().reset_graph();

    for (const auto variant : {BackwardKernelVariant::FusedPartials}) {
        const auto [grad_x, grad_w, grad_b] =
            run_backward_kernel_variant(variant, data.input, dL_dout, data.weight, epsilon, device);
        const auto variant_name = backward_variant_name(variant);

        EXPECT_EQ(grad_x.shape(), data.input.shape());
        EXPECT_EQ(grad_w.shape(), data.weight.shape());
        EXPECT_EQ(grad_b.shape(), data.bias.shape());

        EXPECT_TRUE(xt::all(xt::isfinite(grad_x))) << variant_name << " grad_x has non-finite values";
        EXPECT_TRUE(xt::all(xt::isfinite(grad_w))) << variant_name << " grad_w has non-finite values";
        EXPECT_TRUE(xt::all(xt::isfinite(grad_b))) << variant_name << " grad_b has non-finite values";
        expect_allclose_with_metrics(
            grad_x, grad_x_ref, backward_rtol, backward_atol, variant_name + "_backward_grad_x_vs_xt_reference");
        expect_allclose_with_metrics(
            grad_w, grad_w_ref, backward_rtol, backward_atol, variant_name + "_backward_grad_w_vs_xt_reference");
        expect_allclose_with_metrics(
            grad_b, grad_b_ref, backward_rtol, backward_atol, variant_name + "_backward_grad_b_vs_xt_reference");
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

TEST_F(PolyNormOpTest, PolyNorm_Compare_FusedVsCompositeForward_Small) {
    using namespace ttml;
    const float epsilon = 1e-5F;
    const auto data = make_case_data({1, 1, 2, 128});
    auto* device = &autograd::ctx().get_device();

    auto x = autograd::create_tensor(core::from_xtensor(data.input, device), /*requires_grad=*/true);
    auto w = autograd::create_tensor(core::from_xtensor(data.weight, device), /*requires_grad=*/true);
    auto b = autograd::create_tensor(core::from_xtensor(data.bias, device), /*requires_grad=*/true);

    const auto fused =
        ops::polynorm3(x, w, b, epsilon, ops::PolyNorm3ForwardVariant::Fused, ops::PolyNorm3BackwardVariant::Fused);
    const auto composite = ops::polynorm3(
        x,
        w,
        b,
        epsilon,
        ops::PolyNorm3ForwardVariant::CompositeComparisonOnly,
        ops::PolyNorm3BackwardVariant::CompositeComparisonOnly);

    const auto fused_xt = core::to_xtensor(fused->get_value());
    const auto composite_xt = core::to_xtensor(composite->get_value());
    EXPECT_EQ(fused_xt.shape(), composite_xt.shape());
    EXPECT_TRUE(xt::all(xt::isfinite(fused_xt)));
    EXPECT_TRUE(xt::all(xt::isfinite(composite_xt)));
    expect_allclose_with_metrics(
        fused_xt, composite_xt, kForwardRtol, kForwardAtol, "fused_forward_vs_composite_forward");

    autograd::ctx().reset_graph();
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
            auto out = ops::polynorm3(
                x, w, b, 1e-5F, ops::PolyNorm3ForwardVariant::Fused, ops::PolyNorm3BackwardVariant::Fused);
            (void)out;
        },
        std::runtime_error);
    autograd::ctx().reset_graph();
}

TEST_F(PolyNormOpTest, NIGHTLY_PolyNorm_Compare_NanoLlama3LikeChannelShape) {
    // NanoLlama3 embedding_dim is 384 with max_sequence_length 256.
    CompareKernelVsReferenceWithShape({1, 1, 256, 384}, 1e-5F);
}

TEST_F(PolyNormOpTest, NIGHTLY_PolyNorm_Compare_TinyLlamaLikeChannelShape) {
    // TinyLlama/Llama1B-like shape: embedding_dim 2048 and max sequence length 2048.
    CompareKernelVsReferenceWithShape({1, 1, 2048, 2048}, 1e-5F);
}

TEST_F(PolyNormOpTest, NIGHTLY_PolyNorm_Compare_TinyLlamaLikeSequenceHiddenShape) {
    // TinyLlama max sequence length with MLP hidden dim.
    CompareKernelVsReferenceWithShape({1, 1, 2048, 5632}, 1e-5F);
}

TEST_F(PolyNormOpTest, NIGHTLY_PolyNorm_Compare_UndisclosedModelSequenceHiddenShape) {
    // Undisclosed-model max sequence length with expert hidden dim.
    CompareKernelVsReferenceWithShape({1, 1, 4096, 512}, 1e-5F);
}

TEST_F(PolyNormOpTest, NIGHTLY_PolyNorm_Compare_UndisclosedModelLongSequenceHiddenShape) {
    // Undisclosed-model long-context sequence length with expert hidden dim.
    CompareKernelVsReferenceWithShape({1, 1, 32768, 512}, 1e-5F);
}

TEST_F(PolyNormOpTest, NIGHTLY_PolyNorm_Compare_Llama1bLikeHiddenShape) {
    // Llama1B MLP hidden dim is 8192. Keep sequence short for test runtime.
    CompareKernelVsReferenceWithShape({1, 1, 32, 8192}, 1e-5F);
}

// ============================================================================
// Section 2: Program-cache hits
// ============================================================================
// A launch whose specs match a cached program reuses it, and only the buffer addresses of the new
// tensors are patched in. The cache-hit tests clear the cache so that their first launch is the miss,
// and keep every tensor of the earlier launches alive so that a later launch cannot land on a freed
// address: a hit must read and write the new buffers and leave the earlier ones untouched.
// ============================================================================

namespace {

using ttml::test_utils::CacheExpectation;
using ttml::test_utils::expect_cache;
using ttml::test_utils::expect_new_buffer;
using ttml::test_utils::expect_unchanged;
using ttml::test_utils::num_worker_cores;
using ttml::test_utils::two_core_group_rows_shape;

// Both factories split tile rows over the cores, so two_core_group_rows_shape(kCacheTestC) gives them two core
// groups on any grid. C = 64 keeps the backward on its one-block (matmul row-reduce) path.
constexpr uint32_t kCacheTestC = 64U;

// Still two core groups (one core takes 2 rows, the others 1) at half the data of two_core_group_rows_shape, for
// the test that launches eight times.
std::vector<uint32_t> small_two_core_group_shape() {
    return {1U, 1U, (num_worker_cores() + 1U) * 32U, kCacheTestC};
}

// Big enough that losing it (a compute runtime arg dropped on a hit, or never set on the second core group)
// moves the results past the tolerance. The 1e-6..1e-4 of EpsilonVariants moves them by ~1e-4.
constexpr float kCacheTestEpsilon = 0.0625F;
// The epsilon of the launch that follows the hits. Far enough from kCacheTestEpsilon that a launch run
// with the cached epsilon misses its reference.
constexpr float kOtherEpsilon = 0.25F;

// bf16-exact {w0, w1, w2, b}. Every coefficient differs between any two sets, so a launch that reads
// another launch's weight or bias misses its reference.
struct PolyNormCoeffs {
    float w0;
    float w1;
    float w2;
    float b;
};
constexpr PolyNormCoeffs kCoeffsA{0.25F, 0.25F, 0.5F, 0.125F};
constexpr PolyNormCoeffs kCoeffsB{-0.25F, 0.375F, 0.375F, -0.25F};
constexpr PolyNormCoeffs kCoeffsC{0.375F, -0.125F, 0.25F, 0.25F};

// Fill value of preallocated outputs, far from any polynorm output: an output the kernels never wrote fails.
constexpr float kUnwrittenSentinel = 1000.0F;

// polynorm_bw's packed partials hold four FP32 tiles per tile row: [dw0, dw1, dw2, db].
constexpr uint32_t kPackedPartialsWidth = 128U;

// Whether the launch passes a preallocated output (only the prims take one) or lets the op allocate it.
enum class Output { Allocated, Preallocated };

// Uniform values truncated to bf16: the device stores them exactly, so they are what the kernels read.
xt::xarray<float> make_bf16_uniform(const std::vector<uint32_t>& shape, float min, float max, uint32_t seed) {
    auto values = ttml::test_utils::make_uniform_xarray<float>(shape, min, max, seed);
    for (auto& v : values) {
        v = static_cast<float>(bfloat16::truncate(v));
    }
    return values;
}

xt::xarray<float> filled(const std::vector<uint32_t>& shape, float value) {
    xt::xarray<float> values = xt::xarray<float>::from_shape(std::vector<std::size_t>(shape.begin(), shape.end()));
    values.fill(value);
    return values;
}

xt::xarray<float> weight_of(const PolyNormCoeffs& coeffs) {
    xt::xarray<float> weight = xt::xarray<float>::from_shape({1, 1, 1, 3});
    weight(0, 0, 0, 0) = coeffs.w0;
    weight(0, 0, 0, 1) = coeffs.w1;
    weight(0, 0, 0, 2) = coeffs.w2;
    return weight;
}

// Per-row [dw0, dw1, dw2, db], each [B, N, S]: what the fused backward writes to column 0 of the four
// packed-partials tiles of a tile row, for each of its 32 rows. polynorm3_bw sums them over the rows.
std::array<xt::xarray<float>, 4> polynorm_reference_row_partials(
    const xt::xarray<float>& x, const xt::xarray<float>& dL_dout, float epsilon) {
    const xt::xarray<float> x2 = xt::square(x);
    const xt::xarray<float> x3 = x * x2;
    return {
        xt::xarray<float>(xt::sum(dL_dout * rms_norm_last_dim(x3, epsilon), {3})),
        xt::xarray<float>(xt::sum(dL_dout * rms_norm_last_dim(x2, epsilon), {3})),
        xt::xarray<float>(xt::sum(dL_dout * rms_norm_last_dim(x, epsilon), {3})),
        xt::xarray<float>(xt::sum(dL_dout, {3})),
    };
}

// One forward launch. Holding it keeps all of the launch's buffers allocated.
struct ForwardLaunch {
    PolyNormCaseData data;  // bf16-exact, so also what the device stores
    float epsilon{};
    ttnn::Tensor input;
    ttnn::Tensor weight;
    ttnn::Tensor bias;
    std::optional<ttnn::Tensor> preallocated_output;
    ttnn::Tensor output;
    xt::xarray<float> output_result;  // read back right after the launch
};

// Launches on fresh tensors through metal::polynorm3_fw, or through the prim when the output is preallocated
// (the wrapper never passes one), and checks the output against the reference.
ForwardLaunch run_forward(
    const std::vector<uint32_t>& shape,
    const PolyNormCoeffs& coeffs,
    uint32_t seed,
    Output output,
    float epsilon,
    CacheExpectation expectation,
    const std::string& label) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    const bool preallocate_output = output == Output::Preallocated;
    ForwardLaunch launch;
    launch.data.input = make_bf16_uniform(shape, -1.0F, 1.0F, seed);
    launch.data.weight = weight_of(coeffs);
    launch.data.bias = filled({1U, 1U, 1U, 1U}, coeffs.b);
    launch.epsilon = epsilon;
    launch.input = core::from_xtensor(launch.data.input, device);
    launch.weight = core::from_xtensor(launch.data.weight, device);
    launch.bias = core::from_xtensor(launch.data.bias, device);
    if (preallocate_output) {
        launch.preallocated_output = test_utils::filled_on_device(shape, kUnwrittenSentinel);
    }

    const auto entries_before = device->num_program_cache_entries();
    launch.output = preallocate_output
                        ? ttnn::prim::ttml_polynorm3_fw(
                              launch.input, launch.weight, launch.bias, epsilon, launch.preallocated_output)
                        : metal::polynorm3_fw(launch.input, launch.weight, launch.bias, epsilon);
    expect_cache(entries_before, expectation, label);

    launch.output_result = core::to_xtensor(launch.output);
    const auto expected = polynorm_reference(launch.data.input, launch.data.weight, launch.data.bias, epsilon);
    EXPECT_EQ(launch.output_result.shape(), expected.shape()) << label;
    expect_allclose_with_metrics(launch.output_result, expected, kForwardRtol, kForwardAtol, label + " output");
    if (preallocate_output) {
        EXPECT_EQ(launch.output.buffer()->address(), launch.preallocated_output->buffer()->address())
            << label << " must return the preallocated output";
    }
    return launch;
}

void expect_new_buffers(const ForwardLaunch& earlier, const ForwardLaunch& later, const std::string& label) {
    expect_new_buffer(earlier.input, later.input, label + " input");
    expect_new_buffer(earlier.weight, later.weight, label + " weight");
    expect_new_buffer(earlier.bias, later.bias, label + " bias");
    expect_new_buffer(earlier.output, later.output, label + " output");
}

void expect_untouched(const ForwardLaunch& launch, const std::string& label) {
    expect_unchanged(launch.input, launch.data.input, label + " input");
    expect_unchanged(launch.weight, launch.data.weight, label + " weight");
    expect_unchanged(launch.bias, launch.data.bias, label + " bias");
    expect_unchanged(launch.output, launch.output_result, label + " output");
}

// One backward launch, through the prim: it hands back packed_partials (polynorm3_bw frees it, so the next
// launch could land on its address) and is the only entry point that takes the optional outputs.
struct BackwardLaunch {
    xt::xarray<float> input;  // bf16-exact, so also what the device stores
    xt::xarray<float> dL_dout;
    xt::xarray<float> weight;
    float epsilon{};
    ttnn::Tensor input_tensor;
    ttnn::Tensor dL_dout_tensor;
    ttnn::Tensor weight_tensor;
    std::optional<ttnn::Tensor> preallocated_dL_dx;
    std::optional<ttnn::Tensor> preallocated_packed_partials;
    ttnn::Tensor dL_dx;
    ttnn::Tensor packed_partials;
    xt::xarray<float> dL_dx_result;  // read back right after the launch
    xt::xarray<float> packed_partials_result;
};

void expect_backward_matches_reference(const BackwardLaunch& launch, const std::string& label) {
    const auto dL_dx_expected =
        std::get<0>(polynorm_reference_backward(launch.input, launch.weight, launch.dL_dout, launch.epsilon));
    ASSERT_EQ(launch.dL_dx_result.shape(), dL_dx_expected.shape()) << label;
    expect_allclose_with_metrics(launch.dL_dx_result, dL_dx_expected, kBackwardRtol, kBackwardAtol, label + " dL_dx");

    ASSERT_EQ(launch.packed_partials_result.dimension(), 4U) << label;
    ASSERT_EQ(launch.packed_partials_result.shape()[3], kPackedPartialsWidth) << label;
    const auto partials_expected = polynorm_reference_row_partials(launch.input, launch.dL_dout, launch.epsilon);
    for (std::size_t k = 0; k < partials_expected.size(); ++k) {
        const xt::xarray<float> partial =
            xt::view(launch.packed_partials_result, xt::all(), xt::all(), xt::all(), k * 32U);
        expect_allclose_with_metrics(
            partial,
            partials_expected[k],
            kBackwardRtol,
            kBackwardAtol,
            label + " packed partial " + std::to_string(k));
    }
}

// Launches on fresh tensors, optionally into preallocated outputs, and checks both outputs against the
// reference.
BackwardLaunch run_backward(
    const std::vector<uint32_t>& shape,
    const PolyNormCoeffs& coeffs,
    uint32_t seed,
    Output dL_dx_output,
    Output packed_partials_output,
    float epsilon,
    CacheExpectation expectation,
    const std::string& label) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    const bool preallocate_dL_dx = dL_dx_output == Output::Preallocated;
    const bool preallocate_packed_partials = packed_partials_output == Output::Preallocated;
    BackwardLaunch launch;
    launch.input = make_bf16_uniform(shape, -1.0F, 1.0F, seed);
    // |dL_dout| <= 0.1 keeps dL_dx and the per-row partials around 0.1..1: far above the tolerance, so a stale
    // or unwritten output fails, while the kernels' bf16 rounding stays well inside it.
    launch.dL_dout = make_bf16_uniform(shape, -0.1F, 0.1F, seed + 1U);
    launch.weight = weight_of(coeffs);
    launch.epsilon = epsilon;
    launch.input_tensor = core::from_xtensor(launch.input, device);
    launch.dL_dout_tensor = core::from_xtensor(launch.dL_dout, device);
    launch.weight_tensor = core::from_xtensor(launch.weight, device);
    if (preallocate_dL_dx) {
        launch.preallocated_dL_dx = test_utils::filled_on_device(shape, kUnwrittenSentinel);
    }
    if (preallocate_packed_partials) {
        launch.preallocated_packed_partials = core::full(
            ttnn::Shape({shape[0], shape[1], shape[2], kPackedPartialsWidth}),
            kUnwrittenSentinel,
            device,
            ttnn::DataType::FLOAT32);
    }

    const auto entries_before = device->num_program_cache_entries();
    const auto outputs = ttnn::prim::ttml_polynorm3_bw(
        launch.input_tensor,
        launch.dL_dout_tensor,
        launch.weight_tensor,
        epsilon,
        launch.preallocated_dL_dx,
        launch.preallocated_packed_partials);
    expect_cache(entries_before, expectation, label);
    launch.dL_dx = outputs.at(0);
    launch.packed_partials = outputs.at(1);

    launch.dL_dx_result = core::to_xtensor(launch.dL_dx);
    launch.packed_partials_result = core::to_xtensor(launch.packed_partials);
    expect_backward_matches_reference(launch, label);
    if (preallocate_dL_dx) {
        EXPECT_EQ(launch.dL_dx.buffer()->address(), launch.preallocated_dL_dx->buffer()->address())
            << label << " must return the preallocated dL_dx";
    }
    if (preallocate_packed_partials) {
        EXPECT_EQ(launch.packed_partials.buffer()->address(), launch.preallocated_packed_partials->buffer()->address())
            << label << " must return the preallocated packed partials";
    }
    return launch;
}

void expect_new_buffers(const BackwardLaunch& earlier, const BackwardLaunch& later, const std::string& label) {
    expect_new_buffer(earlier.input_tensor, later.input_tensor, label + " input");
    expect_new_buffer(earlier.dL_dout_tensor, later.dL_dout_tensor, label + " dL_dout");
    expect_new_buffer(earlier.weight_tensor, later.weight_tensor, label + " weight");
    expect_new_buffer(earlier.dL_dx, later.dL_dx, label + " dL_dx");
    expect_new_buffer(earlier.packed_partials, later.packed_partials, label + " packed partials");
}

void expect_untouched(const BackwardLaunch& launch, const std::string& label) {
    expect_unchanged(launch.input_tensor, launch.input, label + " input");
    expect_unchanged(launch.dL_dout_tensor, launch.dL_dout, label + " dL_dout");
    expect_unchanged(launch.weight_tensor, launch.weight, label + " weight");
    expect_unchanged(launch.dL_dx, launch.dL_dx_result, label + " dL_dx");
    expect_unchanged(launch.packed_partials, launch.packed_partials_result, label + " packed partials");
}

}  // namespace

// A miss, then two hits on fresh tensors while the earlier launches stay alive: each hit must replace the
// addresses the launch before it left in the program. Last, a launch with another epsilon, which must run
// with its own epsilon whether or not the program key includes it.
TEST_F(PolyNormOpTest, PolyNormFw_CacheHitReadsFreshBuffers_TwoCoreGroups) {
    ttml::autograd::ctx().get_device().clear_program_cache();
    const auto shape = two_core_group_rows_shape(kCacheTestC);

    const auto a = run_forward(
        shape, kCoeffsA, 11U, Output::Allocated, kCacheTestEpsilon, CacheExpectation::Miss, "launch A (cache miss)");
    const auto b = run_forward(
        shape, kCoeffsB, 21U, Output::Allocated, kCacheTestEpsilon, CacheExpectation::Hit, "launch B (cache hit)");
    const auto c = run_forward(
        shape, kCoeffsC, 31U, Output::Allocated, kCacheTestEpsilon, CacheExpectation::Hit, "launch C (second hit)");
    const auto d = run_forward(
        shape, kCoeffsA, 41U, Output::Allocated, kOtherEpsilon, CacheExpectation::Unchecked, "launch D (new epsilon)");
    EXPECT_FALSE(xt::allclose(
        polynorm_reference(d.data.input, d.data.weight, d.data.bias, kOtherEpsilon),
        polynorm_reference(d.data.input, d.data.weight, d.data.bias, kCacheTestEpsilon),
        kForwardRtol,
        kForwardAtol))
        << "the epsilon change is within tolerance, so a launch D run with the cached epsilon would pass";
    expect_new_buffers(a, b, "launch B");
    expect_new_buffers(a, c, "launch C vs A");
    expect_new_buffers(b, c, "launch C vs B");
    expect_untouched(a, "launch A");
    expect_untouched(b, "launch B");
    expect_untouched(c, "launch C");
}

// Without, with, without and with a preallocated output. Whether or not the key tells the two apart, the
// last two must hit the programs of the first two, and every launch must write its own output.
TEST_F(PolyNormOpTest, PolyNormFw_PreallocatedOutputPresenceFlips_CacheHits_TwoCoreGroups) {
    ttml::autograd::ctx().get_device().clear_program_cache();
    const auto shape = two_core_group_rows_shape(kCacheTestC);
    constexpr float epsilon = kCacheTestEpsilon;

    const auto a =
        run_forward(shape, kCoeffsA, 51U, Output::Allocated, epsilon, CacheExpectation::Miss, "launch A (allocated)");
    const auto b = run_forward(
        shape, kCoeffsA, 61U, Output::Preallocated, epsilon, CacheExpectation::Unchecked, "launch B (preallocated)");
    const auto c =
        run_forward(shape, kCoeffsB, 71U, Output::Allocated, epsilon, CacheExpectation::Hit, "launch C (allocated)");
    const auto d = run_forward(
        shape, kCoeffsB, 81U, Output::Preallocated, epsilon, CacheExpectation::Hit, "launch D (preallocated)");
    expect_new_buffers(a, c, "launch C");
    expect_new_buffers(b, d, "launch D");
    expect_untouched(a, "launch A");
    expect_untouched(b, "launch B");
    expect_untouched(c, "launch C");
}

// Same sequence as PolyNormFw_CacheHitReadsFreshBuffers_TwoCoreGroups.
TEST_F(PolyNormOpTest, PolyNormBw_CacheHitReadsFreshBuffers_TwoCoreGroups) {
    ttml::autograd::ctx().get_device().clear_program_cache();
    const auto shape = two_core_group_rows_shape(kCacheTestC);
    const auto launch = [&shape](
                            const PolyNormCoeffs& coeffs,
                            uint32_t seed,
                            float epsilon,
                            CacheExpectation expectation,
                            const std::string& label) {
        return run_backward(shape, coeffs, seed, Output::Allocated, Output::Allocated, epsilon, expectation, label);
    };

    const auto a = launch(kCoeffsA, 111U, kCacheTestEpsilon, CacheExpectation::Miss, "launch A (cache miss)");
    const auto b = launch(kCoeffsB, 121U, kCacheTestEpsilon, CacheExpectation::Hit, "launch B (cache hit)");
    const auto c = launch(kCoeffsC, 131U, kCacheTestEpsilon, CacheExpectation::Hit, "launch C (second hit)");
    const auto d = launch(kCoeffsA, 141U, kOtherEpsilon, CacheExpectation::Unchecked, "launch D (new epsilon)");
    EXPECT_FALSE(xt::allclose(
        std::get<0>(polynorm_reference_backward(d.input, d.weight, d.dL_dout, kOtherEpsilon)),
        std::get<0>(polynorm_reference_backward(d.input, d.weight, d.dL_dout, kCacheTestEpsilon)),
        kBackwardRtol,
        kBackwardAtol))
        << "the epsilon change is within tolerance, so a launch D run with the cached epsilon would pass";
    expect_new_buffers(a, b, "launch B");
    expect_new_buffers(a, c, "launch C vs A");
    expect_new_buffers(b, c, "launch C vs B");
    expect_untouched(a, "launch A");
    expect_untouched(b, "launch B");
    expect_untouched(c, "launch C");
}

// Every presence pattern of the two optional outputs, launched once and then again on fresh tensors (a hit
// on the first launch's program) while all earlier launches stay alive. Presence changes which tensors the
// op is given; each hit must still read and write only its own launch's buffers.
// PolyNormBw_CacheHitReadsFreshBuffers_TwoCoreGroups covers the larger split, so the eight launches here use
// the smaller one.
TEST_F(PolyNormOpTest, PolyNormBw_PreallocatedOutputsPresenceFlips_CacheHits_TwoCoreGroups) {
    ttml::autograd::ctx().get_device().clear_program_cache();
    const auto shape = small_two_core_group_shape();
    struct Presence {
        Output dL_dx;
        Output packed_partials;
        const char* name;
    };
    constexpr std::array<Presence, 4> kPresences{{
        {Output::Allocated, Output::Allocated, "allocated outputs"},
        {Output::Preallocated, Output::Allocated, "preallocated dL_dx"},
        {Output::Allocated, Output::Preallocated, "preallocated packed partials"},
        {Output::Preallocated, Output::Preallocated, "both preallocated"},
    }};

    // First round: one program per pattern (only the first launch is asserted to compile one). Second round:
    // the same patterns on fresh tensors and other weights, each a hit on its first-round program.
    std::vector<BackwardLaunch> launches;
    std::vector<std::string> labels;
    for (const bool second_round : {false, true}) {
        for (const auto& presence : kPresences) {
            CacheExpectation expectation = CacheExpectation::Hit;
            if (!second_round) {
                expectation = launches.empty() ? CacheExpectation::Miss : CacheExpectation::Unchecked;
            }
            labels.push_back(std::string(second_round ? "second" : "first") + " launch, " + presence.name);
            launches.push_back(run_backward(
                shape,
                second_round ? kCoeffsB : kCoeffsA,
                200U + 10U * static_cast<uint32_t>(launches.size()),
                presence.dL_dx,
                presence.packed_partials,
                kCacheTestEpsilon,
                expectation,
                labels.back()));
        }
    }
    for (std::size_t i = 0; i < kPresences.size(); ++i) {
        expect_new_buffers(launches[i], launches[kPresences.size() + i], labels[kPresences.size() + i]);
    }
    for (std::size_t i = 0; i + 1U < launches.size(); ++i) {
        expect_untouched(launches[i], labels[i]);
    }
}
