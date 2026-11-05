// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/swiglu_op.hpp"

#include <gtest/gtest.h>

#include <optional>
#include <set>
#include <string>
#include <vector>
#include <xtensor-blas/xlinalg.hpp>

#include "autograd/auto_context.hpp"
#include "core/random.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/ops/swiglu_elemwise_bw/swiglu_elemwise_bw.hpp"
#include "test_utils/program_cache.hpp"
#include "test_utils/random_data.hpp"
#include "test_utils/relative_l2.hpp"

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
// Section 1: Shared reference helpers
// ============================================================================
// Contains xtensor reference implementations and reusable comparison helpers
// for SwiGLU forward and backward validation.
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
    const float rel_l2 = test_utils::relative_l2(result_kernel_xtensor, result_reference);

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

struct SwigluBackwardReference {
    xt::xarray<float> out;
    xt::xarray<float> linear1;
    xt::xarray<float> gate;
    xt::xarray<float> dL_dprod;
    xt::xarray<float> dL_dlinear1;
    xt::xarray<float> dL_dgate;
    xt::xarray<float> dL_dx;
    xt::xarray<float> dL_dw1;
    xt::xarray<float> dL_dw2;
    xt::xarray<float> dL_dw3;
};

xt::xarray<float> flatten_to_2d(const xt::xarray<float>& tensor) {
    const size_t trailing = tensor.shape().back();
    const size_t leading = tensor.size() / trailing;
    return xt::xarray<float>(xt::reshape_view(tensor, {leading, trailing}));
}

SwigluBackwardReference build_swiglu_backward_reference(
    const xt::xarray<float>& input,
    const xt::xarray<float>& w1,
    const xt::xarray<float>& w2,
    const xt::xarray<float>& w3) {
    // ops::swiglu uses matmul(x, w*, false, true), so linear projections use w*.T
    auto linear1 = xt::xarray<float>(xt::linalg::tensordot(input, xt::transpose(w1), {3}, {0}));
    auto gate = xt::xarray<float>(xt::linalg::tensordot(input, xt::transpose(w3), {3}, {0}));
    auto sigmoid = 1.0f / (1.0f + xt::exp(-linear1));
    auto swished = linear1 * sigmoid;
    auto gated = swished * gate;
    auto out = xt::xarray<float>(xt::linalg::tensordot(gated, xt::transpose(w2), {3}, {0}));

    auto dL_dout = xt::ones_like(out);
    auto dL_dprod = xt::xarray<float>(xt::linalg::tensordot(dL_dout, w2, {3}, {0}));
    auto [dL_dlinear1, dL_dgate] = swiglu_elemwise_bw_reference(linear1, gate, dL_dprod);

    auto dL_dx_from_w1 = xt::xarray<float>(xt::linalg::tensordot(dL_dlinear1, w1, {3}, {0}));
    auto dL_dx_from_w3 = xt::xarray<float>(xt::linalg::tensordot(dL_dgate, w3, {3}, {0}));
    auto dL_dx = dL_dx_from_w1 + dL_dx_from_w3;

    auto dL_dw1 =
        xt::xarray<float>(xt::linalg::dot(xt::transpose(flatten_to_2d(dL_dlinear1)), flatten_to_2d(input)));  // [H, D]
    auto dL_dw2 =
        xt::xarray<float>(xt::linalg::dot(xt::transpose(flatten_to_2d(dL_dout)), flatten_to_2d(gated)));  // [D, H]
    auto dL_dw3 =
        xt::xarray<float>(xt::linalg::dot(xt::transpose(flatten_to_2d(dL_dgate)), flatten_to_2d(input)));  // [H, D]

    return SwigluBackwardReference{
        .out = std::move(out),
        .linear1 = std::move(linear1),
        .gate = std::move(gate),
        .dL_dprod = std::move(dL_dprod),
        .dL_dlinear1 = std::move(dL_dlinear1),
        .dL_dgate = std::move(dL_dgate),
        .dL_dx = std::move(dL_dx),
        .dL_dw1 = std::move(dL_dw1),
        .dL_dw2 = std::move(dL_dw2),
        .dL_dw3 = std::move(dL_dw3)};
}

void CompareSwiGLUBackwardAgainstReference(const std::vector<uint32_t>& input_shape, uint32_t hidden_dim) {
    using namespace ttml;

    auto& rng = autograd::ctx().get_generator();
    auto* device = &autograd::ctx().get_device();

    const uint32_t input_dim = input_shape.back();
    const float act_std = 1.0f;
    const float w13_std = 1.0f / std::sqrt(static_cast<float>(input_dim));
    const float w2_std = 1.0f / std::sqrt(static_cast<float>(hidden_dim));

    xt::xarray<float> input_data = xt::empty<float>(input_shape);
    core::parallel_generate<float>(
        input_data, [act_std]() { return std::normal_distribution<float>(0.0f, act_std); }, rng());

    std::vector<uint32_t> w13_shape = {hidden_dim, input_dim};  // [H, D]
    std::vector<uint32_t> w2_shape = {input_dim, hidden_dim};   // [D, H]
    xt::xarray<float> w1_data = xt::empty<float>(w13_shape);
    core::parallel_generate<float>(
        w1_data, [w13_std]() { return std::normal_distribution<float>(0.0f, w13_std); }, rng());
    xt::xarray<float> w2_data = xt::empty<float>(w2_shape);
    core::parallel_generate<float>(
        w2_data, [w2_std]() { return std::normal_distribution<float>(0.0f, w2_std); }, rng());
    xt::xarray<float> w3_data = xt::empty<float>(w13_shape);
    core::parallel_generate<float>(
        w3_data, [w13_std]() { return std::normal_distribution<float>(0.0f, w13_std); }, rng());

    auto ref = build_swiglu_backward_reference(input_data, w1_data, w2_data, w3_data);

    // Full op run (forward + full backward grads)
    auto x = autograd::create_tensor(core::from_xtensor(input_data, device), /*requires_grad=*/true);
    auto w1 = autograd::create_tensor(core::from_xtensor(w1_data, device), /*requires_grad=*/true);
    auto w2 = autograd::create_tensor(core::from_xtensor(w2_data, device), /*requires_grad=*/true);
    auto w3 = autograd::create_tensor(core::from_xtensor(w3_data, device), /*requires_grad=*/true);

    auto out = ops::swiglu(x, w1, w2, w3, /*dropout_prob=*/0.0F);
    auto out_kernel = core::to_xtensor(out->get_value());
    out->set_grad(core::ones_like(out->get_value()));
    out->backward();

    auto dx_kernel = core::to_xtensor(x->get_grad());
    auto dw1_kernel = core::to_xtensor(w1->get_grad());
    auto dw2_kernel = core::to_xtensor(w2->get_grad());
    auto dw3_kernel = core::to_xtensor(w3->get_grad());
    autograd::ctx().reset_graph();

    // Elemwise BW kernel run on the same reference intermediates.
    auto linear1_tt = core::from_xtensor(ref.linear1, device);
    auto gate_tt = core::from_xtensor(ref.gate, device);
    auto dL_dprod_tt = core::from_xtensor(ref.dL_dprod, device);
    auto elemwise_result = metal::swiglu_elemwise_bw(linear1_tt, gate_tt, dL_dprod_tt);
    auto dL_dlinear1_kernel = core::to_xtensor(elemwise_result.dL_dlinear1);
    auto dL_dgate_kernel = core::to_xtensor(elemwise_result.dL_dgate);

    const float tol = 1e-2f;
    EXPECT_LT(test_utils::relative_l2(out_kernel, ref.out), tol) << "forward mismatch";
    EXPECT_LT(test_utils::relative_l2(dL_dlinear1_kernel, ref.dL_dlinear1), tol)
        << "swiglu_elemwise_bw dL_dlinear1 mismatch";
    EXPECT_LT(test_utils::relative_l2(dL_dgate_kernel, ref.dL_dgate), tol) << "swiglu_elemwise_bw dL_dgate mismatch";
    EXPECT_LT(test_utils::relative_l2(dx_kernel, ref.dL_dx), tol) << "dL/dx mismatch";
    EXPECT_LT(test_utils::relative_l2(dw1_kernel, ref.dL_dw1), tol) << "dL/dW1 mismatch";
    EXPECT_LT(test_utils::relative_l2(dw2_kernel, ref.dL_dw2), tol) << "dL/dW2 mismatch";
    EXPECT_LT(test_utils::relative_l2(dw3_kernel, ref.dL_dw3), tol) << "dL/dW3 mismatch";
}

}  // namespace

// ============================================================================
// Section 2: Forward suite (SwiGLUForwardTest)
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
TEST_F(SwiGLUForwardTest, ShapeMismatch_W1WrongTrailingDims) {
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

// ============================================================================
// Section 3: Backward suite (SwiGLUBackwardTest)
// ============================================================================
// Full backward equivalence against xtensor reference.
TEST_F(SwiGLUBackwardTest, BackwardAccuracy_1x1x32x32) {
    CompareSwiGLUBackwardAgainstReference({1, 1, 32, 32}, 32);
}
TEST_F(SwiGLUBackwardTest, BackwardAccuracy_1x1x32x64) {
    CompareSwiGLUBackwardAgainstReference({1, 1, 32, 64}, 64);
}
TEST_F(SwiGLUBackwardTest, BackwardAccuracy_8x1x32x32) {
    CompareSwiGLUBackwardAgainstReference({8, 1, 32, 32}, 32);
}
TEST_F(SwiGLUBackwardTest, BackwardAccuracy_2x1x32x128) {
    CompareSwiGLUBackwardAgainstReference({2, 1, 32, 128}, 128);
}
TEST_F(SwiGLUBackwardTest, BackwardAccuracy_4x1x64x256) {
    CompareSwiGLUBackwardAgainstReference({4, 1, 64, 256}, 256);
}
TEST_F(SwiGLUBackwardTest, NIGHTLY_BackwardAccuracy_2x1x128x512) {
    CompareSwiGLUBackwardAgainstReference({2, 1, 128, 512}, 512);
}
TEST_F(SwiGLUBackwardTest, NIGHTLY_BackwardAccuracy_1x1x1024x1024) {
    CompareSwiGLUBackwardAgainstReference({1, 1, 1024, 1024}, 1024);
}
TEST_F(SwiGLUBackwardTest, NIGHTLY_BackwardAccuracy_32x1x256x384) {
    CompareSwiGLUBackwardAgainstReference({32, 1, 256, 384}, 1024);
}

// ============================================================================
// Section 4: Backward nightly-only stress coverage
// ============================================================================
// Extra non-aligned and very large backward cases kept under NIGHTLY to avoid
// inflating default test runtime.
// ============================================================================

TEST_F(SwiGLUBackwardTest, ElemwiseBw_NonAligned_1x1x32x48) {
    CompareSwiGLUBackwardAgainstReference({1, 1, 32, 48}, 48);
}
TEST_F(SwiGLUBackwardTest, ElemwiseBw_NonAligned_2x1x32x96) {
    CompareSwiGLUBackwardAgainstReference({2, 1, 32, 96}, 96);
}
TEST_F(SwiGLUBackwardTest, NIGHTLY_ElemwiseBw_NonAligned_1x1x64x160) {
    CompareSwiGLUBackwardAgainstReference({1, 1, 64, 160}, 160);
}
TEST_F(SwiGLUBackwardTest, NIGHTLY_ElemwiseBw_NonAlignedH_1x1x48x64) {
    CompareSwiGLUBackwardAgainstReference({1, 1, 48, 64}, 64);
}
TEST_F(SwiGLUBackwardTest, NIGHTLY_ElemwiseBw_NonAlignedH_4x1x96x128) {
    CompareSwiGLUBackwardAgainstReference({4, 1, 96, 128}, 128);
}
TEST_F(SwiGLUBackwardTest, NIGHTLY_ElemwiseBw_VeryLarge_1x1x1024x1024) {
    CompareSwiGLUBackwardAgainstReference({1, 1, 1024, 1024}, 1024);
}

// ============================================================================
// Section 5: metal::swiglu_elemwise_bw program-cache hits
// ============================================================================
// A launch whose specs match a cached program reuses it, and only the buffer addresses of the new
// tensors are patched in. Each test clears the cache so that its first launch is the miss, and keeps
// every tensor of the earlier launches alive so that a later launch cannot land on a freed address:
// the hit must read and write the new buffers and leave the earlier ones untouched. Production runs the
// op in place (dL_dlinear1 overwrites linear1, swiglu_op.cpp and moe_ffn_swiglu_op.cpp), so that pattern
// is covered on its own, after launches that preallocate differently, and on a batched shape that leaves
// most of the grid idle.
// ============================================================================

namespace {

using ttml::test_utils::device_with_empty_program_cache;
using ttml::test_utils::expect_unchanged;
using ttml::test_utils::two_core_group_rows_shape;
using ttml::test_utils::zeros_on_device;

// The outputs a launch hands to the op instead of letting it allocate them.
enum class ElemwiseBwOutputs { Allocated, InPlace, PreallocatedDLinear1, PreallocatedDGate, PreallocatedBoth };

std::string to_string(ElemwiseBwOutputs outputs) {
    switch (outputs) {
        case ElemwiseBwOutputs::Allocated: return "allocated outputs";
        case ElemwiseBwOutputs::InPlace: return "dL_dlinear1 in place over linear1";
        case ElemwiseBwOutputs::PreallocatedDLinear1: return "preallocated dL_dlinear1";
        case ElemwiseBwOutputs::PreallocatedDGate: return "preallocated dL_dgate";
        case ElemwiseBwOutputs::PreallocatedBoth: return "both outputs preallocated";
    }
    return "unknown";
}

struct ElemwiseBwLaunch {
    ElemwiseBwOutputs outputs = ElemwiseBwOutputs::Allocated;
    ttnn::Tensor linear1;
    ttnn::Tensor gate;
    ttnn::Tensor dL_dprod;
    xt::xarray<float> linear1_stored;  // what the kernels read: the values rounded to bf16
    xt::xarray<float> gate_stored;
    xt::xarray<float> dL_dprod_stored;
    ttml::metal::SwigluElemwiseBwResult result;
    xt::xarray<float> dL_dlinear1_result;  // read back right after the launch
    xt::xarray<float> dL_dgate_result;
    size_t entries_added = 0;  // program-cache entries the launch added
};

// Launches swiglu_elemwise_bw on fresh inputs of `shape`, handing it the `outputs` pattern, and checks
// both outputs against the reference.
ElemwiseBwLaunch launch_elemwise_bw(
    const std::vector<uint32_t>& shape, uint32_t seed, ElemwiseBwOutputs outputs, const std::string& label) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();

    ElemwiseBwLaunch launch;
    launch.outputs = outputs;
    launch.linear1 = core::from_xtensor(test_utils::make_uniform_xarray<float>(shape, -3.0F, 3.0F, seed), device);
    launch.gate = core::from_xtensor(test_utils::make_uniform_xarray<float>(shape, -2.0F, 2.0F, seed + 1U), device);
    launch.dL_dprod = core::from_xtensor(test_utils::make_uniform_xarray<float>(shape, -1.0F, 1.0F, seed + 2U), device);
    launch.linear1_stored = core::to_xtensor(launch.linear1);
    launch.gate_stored = core::to_xtensor(launch.gate);
    launch.dL_dprod_stored = core::to_xtensor(launch.dL_dprod);

    std::optional<ttnn::Tensor> preallocated_dL_dlinear1;
    std::optional<ttnn::Tensor> preallocated_dL_dgate;
    if (outputs == ElemwiseBwOutputs::InPlace) {
        preallocated_dL_dlinear1 = launch.linear1;
    }
    if (outputs == ElemwiseBwOutputs::PreallocatedDLinear1 || outputs == ElemwiseBwOutputs::PreallocatedBoth) {
        preallocated_dL_dlinear1 = zeros_on_device(shape);
    }
    if (outputs == ElemwiseBwOutputs::PreallocatedDGate || outputs == ElemwiseBwOutputs::PreallocatedBoth) {
        preallocated_dL_dgate = zeros_on_device(shape);
    }

    const auto entries_before = device->num_program_cache_entries();
    launch.result = metal::swiglu_elemwise_bw(
        launch.linear1, launch.gate, launch.dL_dprod, preallocated_dL_dlinear1, preallocated_dL_dgate);
    launch.entries_added = device->num_program_cache_entries() - entries_before;

    if (preallocated_dL_dlinear1.has_value()) {
        EXPECT_EQ(launch.result.dL_dlinear1.buffer()->address(), preallocated_dL_dlinear1->buffer()->address())
            << label << ": dL_dlinear1 must be the preallocated tensor";
    }
    if (preallocated_dL_dgate.has_value()) {
        EXPECT_EQ(launch.result.dL_dgate.buffer()->address(), preallocated_dL_dgate->buffer()->address())
            << label << ": dL_dgate must be the preallocated tensor";
    }
    launch.dL_dlinear1_result = core::to_xtensor(preallocated_dL_dlinear1.value_or(launch.result.dL_dlinear1));
    launch.dL_dgate_result = core::to_xtensor(preallocated_dL_dgate.value_or(launch.result.dL_dgate));

    const auto [expected_dL_dlinear1, expected_dL_dgate] =
        swiglu_elemwise_bw_reference(launch.linear1_stored, launch.gate_stored, launch.dL_dprod_stored);
    const float tol = 1e-2f;
    EXPECT_LT(test_utils::relative_l2(launch.dL_dlinear1_result, expected_dL_dlinear1), tol)
        << label << ": dL_dlinear1 mismatch";
    EXPECT_LT(test_utils::relative_l2(launch.dL_dgate_result, expected_dL_dgate), tol)
        << label << ": dL_dgate mismatch";
    return launch;
}

// Every tensor of `launch` still holds what it held right after the launch.
void expect_launch_unchanged(const ElemwiseBwLaunch& launch, const std::string& label) {
    if (launch.outputs != ElemwiseBwOutputs::InPlace) {  // in place, linear1 is dL_dlinear1, checked below
        expect_unchanged(launch.linear1, launch.linear1_stored, label + " linear1");
    }
    expect_unchanged(launch.gate, launch.gate_stored, label + " gate");
    expect_unchanged(launch.dL_dprod, launch.dL_dprod_stored, label + " dL_dprod");
    expect_unchanged(launch.result.dL_dlinear1, launch.dL_dlinear1_result, label + " dL_dlinear1");
    expect_unchanged(launch.result.dL_dgate, launch.dL_dgate_result, label + " dL_dgate");
}

void expect_distinct_buffers(const ElemwiseBwLaunch& a, const ElemwiseBwLaunch& b) {
    EXPECT_NE(b.linear1.buffer()->address(), a.linear1.buffer()->address());
    EXPECT_NE(b.gate.buffer()->address(), a.gate.buffer()->address());
    EXPECT_NE(b.dL_dprod.buffer()->address(), a.dL_dprod.buffer()->address());
    EXPECT_NE(b.result.dL_dlinear1.buffer()->address(), a.result.dL_dlinear1.buffer()->address());
    EXPECT_NE(b.result.dL_dgate.buffer()->address(), a.result.dL_dgate.buffer()->address());
}

// Launches the patterns in order on fresh inputs of the same specs, every launch alive until the end. A
// pattern that already ran in this sequence must hit its cached program; a new pattern may or may not,
// depending on whether the cache keys on which outputs are preallocated, and must write its own outputs
// either way.
void run_elemwise_bw_pattern_sequence(const std::vector<ElemwiseBwOutputs>& patterns, uint32_t seed) {
    device_with_empty_program_cache();
    const auto shape = two_core_group_rows_shape(32U);

    std::vector<ElemwiseBwLaunch> launches;
    std::set<ElemwiseBwOutputs> launched;
    for (size_t i = 0; i < patterns.size(); ++i) {
        const auto label = "launch " + std::to_string(i) + " (" + to_string(patterns[i]) + ")";
        launches.push_back(launch_elemwise_bw(shape, seed + 10U * static_cast<uint32_t>(i), patterns[i], label));
        if (i == 0U) {
            ASSERT_GT(launches.back().entries_added, 0U) << "the first launch must compile a program";
        } else {
            expect_distinct_buffers(launches[i - 1U], launches[i]);
            if (launched.contains(patterns[i])) {
                EXPECT_EQ(launches.back().entries_added, 0U) << label << " must reuse the program of the same pattern";
            }
        }
        launched.insert(patterns[i]);
    }
    for (size_t i = 0; i < launches.size(); ++i) {
        expect_launch_unchanged(launches[i], "launch " + std::to_string(i));
    }
}

// Launch A compiles the program of `outputs`; launch B, on new inputs of the same specs while launch A's tensors stay
// alive, must reuse it.
void run_hit_after_miss(
    const std::vector<uint32_t>& shape, ElemwiseBwOutputs outputs, uint32_t seed_a, uint32_t seed_b) {
    device_with_empty_program_cache();
    const bool in_place = outputs == ElemwiseBwOutputs::InPlace;
    const auto a = launch_elemwise_bw(shape, seed_a, outputs, in_place ? "launch A (in place)" : "launch A");
    ASSERT_GT(a.entries_added, 0U) << "the first launch must compile a program";
    const auto b = launch_elemwise_bw(
        shape, seed_b, outputs, in_place ? "launch B (in place, cache hit)" : "launch B (cache hit)");
    EXPECT_EQ(b.entries_added, 0U) << "launch B must reuse launch A's program";
    expect_distinct_buffers(a, b);
    expect_launch_unchanged(a, "launch A");
}

}  // namespace

TEST_F(SwiGLUBackwardTest, ElemwiseBw_CacheHitReadsFreshBuffers_TwoCoreGroups) {
    run_hit_after_miss(two_core_group_rows_shape(64U), ElemwiseBwOutputs::Allocated, 101U, 201U);
}

// The production call: dL_dlinear1 is written over linear1, so linear1's buffer appears twice among the
// op's tensors.
TEST_F(SwiGLUBackwardTest, ElemwiseBw_InPlaceCacheHit_TwoCoreGroups) {
    run_hit_after_miss(two_core_group_rows_shape(64U), ElemwiseBwOutputs::InPlace, 301U, 401U);
}

// The production call on fewer tile-rows than any grid has cores: two batch entries of 71 rows pad to
// 2 * 3 = 6 tile-rows, one per core. The hit must patch exactly those six cores; on the full-grid shapes
// above, a walk over logical tile-rows (4 here), one batch entry (3) or tiles (12) patches the same cores.
TEST_F(SwiGLUBackwardTest, ElemwiseBw_InPlaceCacheHit_BatchedPartialGrid) {
    run_hit_after_miss({2U, 1U, 71U, 64U}, ElemwiseBwOutputs::InPlace, 701U, 801U);
}

// Allocated outputs first, then the production in-place call and every preallocation pattern. The last
// two launches repeat the dL_dgate patterns, so that they hit the programs those patterns compiled when
// the cache keys on which outputs are preallocated.
TEST_F(SwiGLUBackwardTest, ElemwiseBw_PreallocatedOutputsAfterAllocatedOutputs_TwoCoreGroups) {
    using enum ElemwiseBwOutputs;
    run_elemwise_bw_pattern_sequence(
        {Allocated,
         InPlace,
         InPlace,
         PreallocatedDLinear1,
         PreallocatedDGate,
         PreallocatedBoth,
         Allocated,
         PreallocatedDGate,
         PreallocatedBoth},
        501U);
}

// A preallocated output first, then patterns that hand the op fewer or different tensors.
TEST_F(SwiGLUBackwardTest, ElemwiseBw_AllocatedOutputsAfterPreallocatedOutputs_TwoCoreGroups) {
    using enum ElemwiseBwOutputs;
    run_elemwise_bw_pattern_sequence(
        {PreallocatedDLinear1, Allocated, PreallocatedDGate, InPlace, PreallocatedDLinear1}, 601U);
}
