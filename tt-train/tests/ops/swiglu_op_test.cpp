// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/swiglu_op.hpp"

#include <gtest/gtest.h>

#include <xtensor-blas/xlinalg.hpp>

#include "autograd/auto_context.hpp"
#include "core/random.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/ops/swiglu_elemwise_bw/swiglu_elemwise_bw.hpp"

class SwiGLUForwardTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }

    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }
};

class SwiGLUBackwardTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }

    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }
};

// ============================================================================
// Section 1: SwiGLU Forward Pass Test
// ============================================================================
// This test validates the SwiGLU forward pass implementation
//
// Test methodology:
// 1. Create test tensors x, w1, w2, w3
// 2. Compute SwiGLU forward pass
// 3. Verify the result is finite and has expected shape
// ============================================================================

namespace {

/**
 * Reference implementation of SwiGLU forward pass using xtensor
 * SwiGLU(x, w1, w2, w3) = (silu(x @ w1) * (x @ w3)) @ w2
 *
 * @param x  Input tensor [B, N, S, C]
 * @param w1 Weight tensor [1, 1, C, H]
 * @param w2 Weight tensor [1, 1, H, C]
 * @param w3 Weight tensor [1, 1, C, H]
 * @return   SwiGLU activation result [B, N, S, C]
 */
xt::xarray<float> swiglu_forward_reference(
    const xt::xarray<float>& x, const xt::xarray<float>& w1, const xt::xarray<float>& w2, const xt::xarray<float>& w3) {
    // Strip the first two singleton dimensions from weight tensors:
    auto w1_mat = xt::squeeze(w1, {0, 1});  // [C, H]
    auto w2_mat = xt::squeeze(w2, {0, 1});  // [H, C]
    auto w3_mat = xt::squeeze(w3, {0, 1});  // [C, H]

    // x @ w1: -> [B, N, S, H]
    auto xw1 = xt::linalg::tensordot(x, w1_mat, {3}, {0});
    // x @ w3: -> [B, N, S, H]
    auto xw3 = xt::linalg::tensordot(x, w3_mat, {3}, {0});

    // SiLU activation: z * sigmoid(z)
    // Note: Must return xt::xarray to force evaluation, not a lazy expression
    auto silu = [](const xt::xarray<float>& t) -> xt::xarray<float> {
        auto sigmoid = 1.0f / (1.0f + xt::exp(-t));
        return t * sigmoid;
    };

    // Gate: [B, N, S, H]
    auto gated = silu(xw1) * xw3;

    // gated @ w2: -> [B, N, S, C]
    return xt::linalg::tensordot(gated, w2_mat, {3}, {0});
}

void CompareKernelVsReference(
    const xt::xarray<float>& input_data,
    const xt::xarray<float>& w1_data,
    const xt::xarray<float>& w2_data,
    const xt::xarray<float>& w3_data) {
    using namespace ttml;

    // Create input tensors for kernel implementation
    auto input_kernel = autograd::create_tensor(core::from_xtensor(input_data, &autograd::ctx().get_device()));
    // Canonical swiglu uses LinearLayer convention:
    // w1,w3 [H,D], w2 [D,H]. Convert from [1,1,D,H] / [1,1,H,D] test fixtures.
    auto w1_linear = xt::xarray<float>(xt::transpose(xt::squeeze(w1_data, {0, 1})));  // [H, D]
    auto w2_linear = xt::xarray<float>(xt::transpose(xt::squeeze(w2_data, {0, 1})));  // [D, H]
    auto w3_linear = xt::xarray<float>(xt::transpose(xt::squeeze(w3_data, {0, 1})));  // [H, D]
    auto w1_kernel = autograd::create_tensor(core::from_xtensor(w1_linear, &autograd::ctx().get_device()));
    auto w2_kernel = autograd::create_tensor(core::from_xtensor(w2_linear, &autograd::ctx().get_device()));
    auto w3_kernel = autograd::create_tensor(core::from_xtensor(w3_linear, &autograd::ctx().get_device()));

    // Forward pass - kernel implementation
    auto result_kernel = ops::swiglu(input_kernel, w1_kernel, w2_kernel, w3_kernel);
    result_kernel->get_value();
    auto result_kernel_xtensor = core::to_xtensor(result_kernel->get_value());

    // Forward pass - reference implementation
    auto result_reference = swiglu_forward_reference(input_data, w1_data, w2_data, w3_data);

    // Verify shapes match
    EXPECT_EQ(result_kernel_xtensor.shape(), result_reference.shape())
        << "Shape mismatch between kernel and reference results";
    // Verify both results are finite
    EXPECT_TRUE(xt::all(xt::isfinite(result_kernel_xtensor))) << "SwiGLU kernel result contains NaN or Inf values";
    EXPECT_TRUE(xt::all(xt::isfinite(result_reference))) << "SwiGLU reference result contains NaN or Inf values";

    // We validate using relative L2 error instead of elementwise allclose().
    //
    // SwiGLU involves two matmuls + SiLU, so the output magnitude scales with
    // hidden_dim and tensor size. Combined with BF16 internal math (≈1e-2 relative
    // noise per element), elementwise tolerances become unstable: they may fail on
    // large tensors (outliers) or fail on small ones (tiny reference values).
    //
    // Relative L2 is scale-invariant and reflects the total numerical deviation of
    // the tensor, giving consistent results across all shapes. A threshold of 0.01
    // is standard for BF16 vs FP32 reference comparisons.
    auto diff = result_kernel_xtensor - result_reference;

    // Compute L2 norms.
    float diff_l2 = std::sqrt(xt::sum(xt::square(diff))());
    float ref_l2 = std::sqrt(xt::sum(xt::square(result_reference))());
    const float eps = 1e-12f;
    float rel_l2 = diff_l2 / (ref_l2 + eps);

    // Check the error bound.
    const float tolerance = 1e-2f;
    EXPECT_LT(rel_l2, tolerance) << "Relative L2 error too large: " << rel_l2 << " (expected < " << tolerance << ")";
}

/**
 * Helper function to test SwiGLU forward pass with given input shape
 *
 * @param input_shape Input tensor shape [B, N, S, C] where:
 *   - B: batch size
 *   - N: number of channels (usually 1 for transformers)
 *   - S: sequence length (height for transformers)
 *   - C: feature dimension (width/embedding dimension)
 * @param hidden_dim Hidden dimension for the weight matrices
 */
static void CompareKernelVsReferenceWithShape(const std::vector<uint32_t>& input_shape, const uint32_t hidden_dim) {
    using namespace ttml;

    // Generate random input data using parallel_generate.
    // Use non-zero-mean distribution [0, 1] so that outputs have meaningful magnitude;
    // zero-mean inputs (e.g. [-1, 1]) often yield near-zero outputs, making precision
    // checks less discriminating (see code review discussion on test faithfulness).
    auto& rng = autograd::ctx().get_generator();

    const float low = 0.0f;
    const float high = 1.0f;

    xt::xarray<float> input_data = xt::empty<float>(input_shape);
    core::parallel_generate<float>(
        input_data, [low, high]() { return std::uniform_real_distribution<float>(low, high); }, /* seed */ rng());

    // Create weight tensors: w1, w3 [1, 1, C, H], w2 [1, 1, H, C]
    const uint32_t input_dim = input_shape.back();
    std::vector<uint32_t> w1_w3_shape = {1, 1, input_dim, hidden_dim};
    std::vector<uint32_t> w2_shape = {1, 1, hidden_dim, input_dim};

    xt::xarray<float> w1_data = xt::empty<float>(w1_w3_shape);
    core::parallel_generate<float>(
        w1_data, [low, high]() { return std::uniform_real_distribution<float>(low, high); }, /* seed */ rng());

    xt::xarray<float> w2_data = xt::empty<float>(w2_shape);
    core::parallel_generate<float>(
        w2_data, [low, high]() { return std::uniform_real_distribution<float>(low, high); }, /* seed */ rng());
    xt::xarray<float> w3_data = xt::empty<float>(w1_w3_shape);
    core::parallel_generate<float>(
        w3_data, [low, high]() { return std::uniform_real_distribution<float>(low, high); }, /* seed */ rng());

    CompareKernelVsReference(input_data, w1_data, w2_data, w3_data);
}

float relative_l2(const xt::xarray<float>& a, const xt::xarray<float>& b) {
    auto diff = a - b;
    float diff_l2 = std::sqrt(xt::sum(xt::square(diff))());
    float ref_l2 = std::sqrt(xt::sum(xt::square(b))());
    return diff_l2 / (ref_l2 + 1e-12f);
}

void CompareOptimizedVsBaseline(const std::vector<uint32_t>& input_shape, const uint32_t hidden_dim) {
    using namespace ttml;

    auto& rng = autograd::ctx().get_generator();
    auto* device = &autograd::ctx().get_device();

    const uint32_t input_dim = input_shape.back();
    const float act_std = 1.0f;
    const float w13_std = 1.0f / std::sqrt(static_cast<float>(input_dim));
    const float w2_std = 1.0f / std::sqrt(static_cast<float>(hidden_dim));

    // Generate weights in LinearLayer convention: w1,w3 [H,D], w2 [D,H]
    xt::xarray<float> input_data = xt::empty<float>(input_shape);
    core::parallel_generate<float>(
        input_data, [act_std]() { return std::normal_distribution<float>(0.0f, act_std); }, rng());

    std::vector<uint32_t> w13_lin_shape = {hidden_dim, input_dim};
    std::vector<uint32_t> w2_lin_shape = {input_dim, hidden_dim};

    xt::xarray<float> w1_lin = xt::empty<float>(w13_lin_shape);
    core::parallel_generate<float>(
        w1_lin, [w13_std]() { return std::normal_distribution<float>(0.0f, w13_std); }, rng());
    xt::xarray<float> w2_lin = xt::empty<float>(w2_lin_shape);
    core::parallel_generate<float>(w2_lin, [w2_std]() { return std::normal_distribution<float>(0.0f, w2_std); }, rng());
    xt::xarray<float> w3_lin = xt::empty<float>(w13_lin_shape);
    core::parallel_generate<float>(
        w3_lin, [w13_std]() { return std::normal_distribution<float>(0.0f, w13_std); }, rng());

    struct BackwardSnapshot {
        xt::xarray<float> fwd;
        xt::xarray<float> dx;
        xt::xarray<float> dw1;
        xt::xarray<float> dw2;
        xt::xarray<float> dw3;
    };

    // Canonical path: swiglu uses LinearLayer [H,D]/[D,H] weights directly.
    auto run_baseline = [&]() {
        auto x = autograd::create_tensor(core::from_xtensor(input_data, device));
        auto w1 = autograd::create_tensor(core::from_xtensor(w1_lin, device));
        auto w2 = autograd::create_tensor(core::from_xtensor(w2_lin, device));
        auto w3 = autograd::create_tensor(core::from_xtensor(w3_lin, device));
        x->set_requires_grad(true);
        w1->set_requires_grad(true);
        w2->set_requires_grad(true);
        w3->set_requires_grad(true);

        auto out = ops::swiglu(x, w1, w2, w3, /*dropout_prob=*/0.0F);
        auto fwd = core::to_xtensor(out->get_value());
        out->set_grad(core::ones_like(out->get_value()));
        out->backward();

        auto dx = core::to_xtensor(x->get_grad());
        auto dw1 = core::to_xtensor(w1->get_grad());
        auto dw2 = core::to_xtensor(w2->get_grad());
        auto dw3 = core::to_xtensor(w3->get_grad());
        autograd::ctx().reset_graph();
        return BackwardSnapshot{
            .fwd = std::move(fwd),
            .dx = std::move(dx),
            .dw1 = std::move(dw1),
            .dw2 = std::move(dw2),
            .dw3 = std::move(dw3)};
    };

    // Alias path: swiglu_optimized currently forwards to canonical swiglu.
    auto run_optimized = [&]() {
        auto x = autograd::create_tensor(core::from_xtensor(input_data, device));
        auto w1 = autograd::create_tensor(core::from_xtensor(w1_lin, device));
        auto w2 = autograd::create_tensor(core::from_xtensor(w2_lin, device));
        auto w3 = autograd::create_tensor(core::from_xtensor(w3_lin, device));
        x->set_requires_grad(true);
        w1->set_requires_grad(true);
        w2->set_requires_grad(true);
        w3->set_requires_grad(true);

        auto out = ops::swiglu_optimized(x, w1, w2, w3, /*dropout_prob=*/0.0F);
        auto fwd = core::to_xtensor(out->get_value());
        out->set_grad(core::ones_like(out->get_value()));
        out->backward();

        auto dx = core::to_xtensor(x->get_grad());
        auto dw1 = core::to_xtensor(w1->get_grad());
        auto dw2 = core::to_xtensor(w2->get_grad());
        auto dw3 = core::to_xtensor(w3->get_grad());
        autograd::ctx().reset_graph();
        return BackwardSnapshot{
            .fwd = std::move(fwd),
            .dx = std::move(dx),
            .dw1 = std::move(dw1),
            .dw2 = std::move(dw2),
            .dw3 = std::move(dw3)};
    };

    auto base = run_baseline();
    auto opt = run_optimized();

    const float tol = 1e-2f;
    EXPECT_EQ(base.fwd.shape(), opt.fwd.shape());

    float fwd_err = relative_l2(opt.fwd, base.fwd);
    EXPECT_LT(fwd_err, tol) << "Forward mismatch: rel L2 = " << fwd_err;

    float dx_err = relative_l2(opt.dx, base.dx);
    EXPECT_LT(dx_err, tol) << "dL/dx mismatch: rel L2 = " << dx_err;

    float dw1_err = relative_l2(opt.dw1, base.dw1);
    EXPECT_LT(dw1_err, tol) << "dL/dW1 mismatch: rel L2 = " << dw1_err;

    float dw2_err = relative_l2(opt.dw2, base.dw2);
    EXPECT_LT(dw2_err, tol) << "dL/dW2 mismatch: rel L2 = " << dw2_err;

    float dw3_err = relative_l2(opt.dw3, base.dw3);
    EXPECT_LT(dw3_err, tol) << "dL/dW3 mismatch: rel L2 = " << dw3_err;
}

std::pair<xt::xarray<float>, xt::xarray<float>> swiglu_elemwise_bw_reference(
    const xt::xarray<float>& linear1, const xt::xarray<float>& gate, const xt::xarray<float>& dL_dprod) {
    auto sigmoid = 1.0f / (1.0f + xt::exp(-linear1));
    auto swished = linear1 * sigmoid;

    xt::xarray<float> dL_dgate = swished * dL_dprod;
    auto dL_dswished = gate * dL_dprod;
    auto silu_grad = sigmoid * (1.0f + linear1 * (1.0f - sigmoid));
    xt::xarray<float> dL_dlinear1 = dL_dswished * silu_grad;

    return {dL_dlinear1, dL_dgate};
}

void CompareSwiGLUElemwiseBwKernel(const std::vector<uint32_t>& shape) {
    using namespace ttml;

    auto& rng = autograd::ctx().get_generator();
    auto* device = &autograd::ctx().get_device();

    xt::xarray<float> linear1_data = xt::empty<float>(shape);
    core::parallel_generate<float>(linear1_data, []() { return std::normal_distribution<float>(0.0f, 1.0f); }, rng());
    xt::xarray<float> gate_data = xt::empty<float>(shape);
    core::parallel_generate<float>(gate_data, []() { return std::normal_distribution<float>(0.0f, 1.0f); }, rng());
    xt::xarray<float> dL_dprod_data = xt::empty<float>(shape);
    core::parallel_generate<float>(dL_dprod_data, []() { return std::normal_distribution<float>(0.0f, 0.1f); }, rng());

    auto [ref_dL_dlinear1, ref_dL_dgate] = swiglu_elemwise_bw_reference(linear1_data, gate_data, dL_dprod_data);

    auto linear1_tt = core::from_xtensor(linear1_data, device);
    auto gate_tt = core::from_xtensor(gate_data, device);
    auto dL_dprod_tt = core::from_xtensor(dL_dprod_data, device);

    auto result = metal::swiglu_elemwise_bw(linear1_tt, gate_tt, dL_dprod_tt);
    auto kernel_dL_dlinear1 = core::to_xtensor(result.dL_dlinear1);
    auto kernel_dL_dgate = core::to_xtensor(result.dL_dgate);

    const float tol = 1e-2f;
    float dl1_err = relative_l2(kernel_dL_dlinear1, ref_dL_dlinear1);
    EXPECT_LT(dl1_err, tol) << "swiglu_elemwise_bw dL_dlinear1 mismatch: rel L2 = " << dl1_err;

    float dg_err = relative_l2(kernel_dL_dgate, ref_dL_dgate);
    EXPECT_LT(dg_err, tol) << "swiglu_elemwise_bw dL_dgate mismatch: rel L2 = " << dg_err;
}

}  // namespace

// ============================================================================
// Section 2: SwiGLUForward tests
// ============================================================================

TEST_F(SwiGLUForwardTest, Basic_1x1x32x32) {
    CompareKernelVsReferenceWithShape({1, 1, 32, 32}, 32);
}
TEST_F(SwiGLUForwardTest, MultiTile_1x1x32x64) {
    CompareKernelVsReferenceWithShape({1, 1, 32, 64}, 64);
}
TEST_F(SwiGLUForwardTest, MultiTile_1x1x64x32) {
    CompareKernelVsReferenceWithShape({1, 1, 64, 32}, 32);
}
TEST_F(SwiGLUForwardTest, MultiBatch_8x1x32x32) {
    CompareKernelVsReferenceWithShape({8, 1, 32, 32}, 32);
}
TEST_F(SwiGLUForwardTest, Medium_2x1x32x128) {
    CompareKernelVsReferenceWithShape({2, 1, 32, 128}, 128);
}
TEST_F(SwiGLUForwardTest, Large_4x1x64x256) {
    CompareKernelVsReferenceWithShape({4, 1, 64, 256}, 256);
}

TEST_F(SwiGLUForwardTest, RepeatedRuns_NoHang) {
    for (int iteration = 0; iteration < 3; ++iteration) {
        CompareKernelVsReferenceWithShape({100, 1, 32, 64}, 128);
    }
}

TEST_F(SwiGLUForwardTest, NIGHTLY_Large_2x1x128x512) {
    CompareKernelVsReferenceWithShape({2, 1, 128, 512}, 512);
}
TEST_F(SwiGLUForwardTest, NIGHTLY_VeryLarge_1x1x1024x1024) {
    CompareKernelVsReferenceWithShape({1, 1, 1024, 1024}, 1024);
}
TEST_F(SwiGLUForwardTest, NIGHTLY_NanoLlama_64x1x256x384) {
    CompareKernelVsReferenceWithShape({64, 1, 256, 384}, 1024);
}
TEST_F(SwiGLUForwardTest, UnbalancedWorkload_57x1x32x32) {
    CompareKernelVsReferenceWithShape({57, 1, 32, 32}, 32);
}

// Shape validation tests.
TEST_F(SwiGLUForwardTest, ShapeMismatch_W1WrongLayout) {
    using namespace ttml;

    const size_t embed_dim = 64;
    const size_t hidden_dim = 128;

    std::vector<size_t> input_shape = {1, 1, 32, embed_dim};
    std::vector<size_t> w1_wrong_shape = {hidden_dim, embed_dim / 2};
    std::vector<size_t> w2_shape = {embed_dim, hidden_dim};
    std::vector<size_t> w3_shape = {hidden_dim, embed_dim};

    xt::xarray<float> input_data = xt::ones<float>(input_shape);
    xt::xarray<float> w1_wrong = xt::ones<float>(w1_wrong_shape);
    xt::xarray<float> w2_data = xt::ones<float>(w2_shape);
    xt::xarray<float> w3_data = xt::ones<float>(w3_shape);

    auto input = autograd::create_tensor(core::from_xtensor(input_data, &autograd::ctx().get_device()));
    auto w1 = autograd::create_tensor(core::from_xtensor(w1_wrong, &autograd::ctx().get_device()));
    auto w2 = autograd::create_tensor(core::from_xtensor(w2_data, &autograd::ctx().get_device()));
    auto w3 = autograd::create_tensor(core::from_xtensor(w3_data, &autograd::ctx().get_device()));

    testing::internal::CaptureStdout();
    EXPECT_THROW(ops::swiglu(input, w1, w2, w3), std::exception);
    testing::internal::GetCapturedStdout();
}

TEST_F(SwiGLUForwardTest, ShapeMismatch_W3DoesntMatchW1) {
    using namespace ttml;

    const size_t embed_dim = 64;
    const size_t hidden_dim = 128;

    std::vector<size_t> input_shape = {1, 1, 32, embed_dim};
    std::vector<size_t> w1_shape = {hidden_dim, embed_dim};
    std::vector<size_t> w2_shape = {embed_dim, hidden_dim};
    std::vector<size_t> w3_wrong_shape = {hidden_dim * 2, embed_dim};

    xt::xarray<float> input_data = xt::ones<float>(input_shape);
    xt::xarray<float> w1_data = xt::ones<float>(w1_shape);
    xt::xarray<float> w2_data = xt::ones<float>(w2_shape);
    xt::xarray<float> w3_wrong = xt::ones<float>(w3_wrong_shape);

    auto input = autograd::create_tensor(core::from_xtensor(input_data, &autograd::ctx().get_device()));
    auto w1 = autograd::create_tensor(core::from_xtensor(w1_data, &autograd::ctx().get_device()));
    auto w2 = autograd::create_tensor(core::from_xtensor(w2_data, &autograd::ctx().get_device()));
    auto w3 = autograd::create_tensor(core::from_xtensor(w3_wrong, &autograd::ctx().get_device()));

    testing::internal::CaptureStdout();
    EXPECT_THROW(ops::swiglu(input, w1, w2, w3), std::exception);
    testing::internal::GetCapturedStdout();
}

TEST_F(SwiGLUForwardTest, ShapeMismatch_W2WrongDimensions) {
    using namespace ttml;

    const size_t embed_dim = 64;
    const size_t hidden_dim = 128;

    std::vector<size_t> input_shape = {1, 1, 32, embed_dim};
    std::vector<size_t> w1_shape = {hidden_dim, embed_dim};
    std::vector<size_t> w2_wrong_shape = {hidden_dim};
    std::vector<size_t> w3_shape = {hidden_dim, embed_dim};

    xt::xarray<float> input_data = xt::ones<float>(input_shape);
    xt::xarray<float> w1_data = xt::ones<float>(w1_shape);
    xt::xarray<float> w2_wrong = xt::ones<float>(w2_wrong_shape);
    xt::xarray<float> w3_data = xt::ones<float>(w3_shape);

    auto input = autograd::create_tensor(core::from_xtensor(input_data, &autograd::ctx().get_device()));
    auto w1 = autograd::create_tensor(core::from_xtensor(w1_data, &autograd::ctx().get_device()));
    auto w2 = autograd::create_tensor(core::from_xtensor(w2_wrong, &autograd::ctx().get_device()));
    auto w3 = autograd::create_tensor(core::from_xtensor(w3_data, &autograd::ctx().get_device()));

    testing::internal::CaptureStdout();
    EXPECT_THROW(ops::swiglu(input, w1, w2, w3), std::exception);
    testing::internal::GetCapturedStdout();
}

// Full backward equivalence (dx, dW1, dW2, dW3).
TEST_F(SwiGLUForwardTest, BackwardAccuracy_1x1x32x32) {
    CompareOptimizedVsBaseline({1, 1, 32, 32}, 32);
}
TEST_F(SwiGLUForwardTest, BackwardAccuracy_1x1x32x64) {
    CompareOptimizedVsBaseline({1, 1, 32, 64}, 64);
}
TEST_F(SwiGLUForwardTest, BackwardAccuracy_8x1x32x32) {
    CompareOptimizedVsBaseline({8, 1, 32, 32}, 32);
}
TEST_F(SwiGLUForwardTest, BackwardAccuracy_2x1x32x128) {
    CompareOptimizedVsBaseline({2, 1, 32, 128}, 128);
}
TEST_F(SwiGLUForwardTest, BackwardAccuracy_4x1x64x256) {
    CompareOptimizedVsBaseline({4, 1, 64, 256}, 256);
}
TEST_F(SwiGLUForwardTest, NIGHTLY_BackwardAccuracy_2x1x128x512) {
    CompareOptimizedVsBaseline({2, 1, 128, 512}, 512);
}
TEST_F(SwiGLUForwardTest, NIGHTLY_BackwardAccuracy_1x1x1024x1024) {
    CompareOptimizedVsBaseline({1, 1, 1024, 1024}, 1024);
}
TEST_F(SwiGLUForwardTest, NIGHTLY_BackwardAccuracy_32x1x256x384) {
    CompareOptimizedVsBaseline({32, 1, 256, 384}, 1024);
}

// ============================================================================
// Section 3: SwiGLUBackward tests (fused elemwise BW kernel)
// ============================================================================

TEST_F(SwiGLUBackwardTest, ElemwiseBw_Basic_1x1x32x32) {
    CompareSwiGLUElemwiseBwKernel({1, 1, 32, 32});
}
TEST_F(SwiGLUBackwardTest, ElemwiseBw_MultiTile_1x1x32x64) {
    CompareSwiGLUElemwiseBwKernel({1, 1, 32, 64});
}
TEST_F(SwiGLUBackwardTest, ElemwiseBw_MultiRow_1x1x64x32) {
    CompareSwiGLUElemwiseBwKernel({1, 1, 64, 32});
}
TEST_F(SwiGLUBackwardTest, ElemwiseBw_Batch_8x1x32x32) {
    CompareSwiGLUElemwiseBwKernel({8, 1, 32, 32});
}
TEST_F(SwiGLUBackwardTest, ElemwiseBw_Medium_2x1x32x128) {
    CompareSwiGLUElemwiseBwKernel({2, 1, 32, 128});
}
TEST_F(SwiGLUBackwardTest, ElemwiseBw_Large_4x1x64x256) {
    CompareSwiGLUElemwiseBwKernel({4, 1, 64, 256});
}
TEST_F(SwiGLUBackwardTest, ElemwiseBw_NonAligned_1x1x32x48) {
    CompareSwiGLUElemwiseBwKernel({1, 1, 32, 48});
}
TEST_F(SwiGLUBackwardTest, ElemwiseBw_NonAligned_2x1x32x96) {
    CompareSwiGLUElemwiseBwKernel({2, 1, 32, 96});
}
TEST_F(SwiGLUBackwardTest, ElemwiseBw_NonAligned_1x1x64x160) {
    CompareSwiGLUElemwiseBwKernel({1, 1, 64, 160});
}
TEST_F(SwiGLUBackwardTest, ElemwiseBw_NonAlignedH_1x1x48x64) {
    CompareSwiGLUElemwiseBwKernel({1, 1, 48, 64});
}
TEST_F(SwiGLUBackwardTest, ElemwiseBw_NonAlignedH_4x1x96x128) {
    CompareSwiGLUElemwiseBwKernel({4, 1, 96, 128});
}
TEST_F(SwiGLUBackwardTest, NIGHTLY_ElemwiseBw_VeryLarge_1x1x1024x1024) {
    CompareSwiGLUElemwiseBwKernel({1, 1, 1024, 1024});
}
