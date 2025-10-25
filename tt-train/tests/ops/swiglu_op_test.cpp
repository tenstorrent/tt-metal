// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/swiglu_op.hpp"

#include <gtest/gtest.h>

#include <cassert>
#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/misc/xmanipulation.hpp"

class SwiGLUOpTest : public ::testing::Test {
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
    auto silu = [](const xt::xarray<float>& t) {
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
    auto w1_kernel = autograd::create_tensor(core::from_xtensor(w1_data, &autograd::ctx().get_device()));
    auto w2_kernel = autograd::create_tensor(core::from_xtensor(w2_data, &autograd::ctx().get_device()));
    auto w3_kernel = autograd::create_tensor(core::from_xtensor(w3_data, &autograd::ctx().get_device()));

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

    // We use dynamic tolerances because the output magnitude of SwiGLU grows with input/hidden size
    // (due to large dot-product summations in the matmuls). A fixed absolute tolerance would either:
    //   - fail for large tensors (legitimate numerical growth triggers abs-error check), or
    //   - be too loose for small tensors (hiding real bugs).
    // The kernel runs in BF16 internally, so we allow ~2% relative error vs the FP32 reference.
    // The absolute tolerance is scaled to the magnitude of the reference output for this test case.
    float rtol = 2e-2f;
    float atol = rtol * static_cast<float>(xt::amax(xt::abs(result_reference))());
    EXPECT_TRUE(xt::allclose(result_kernel_xtensor, result_reference, rtol, atol))
        << "SwiGLU kernel and reference implementations differ";
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
static void CompareKernelVsReferenceWithShape(const std::vector<uint32_t>& input_shape, uint32_t hidden_dim) {
    using namespace ttml;

    // Generate random input data
    const float bound = 1.0F;
    xt::random::seed(42);
    xt::xarray<float> input_data = xt::random::rand<float>(input_shape, -bound, bound);

    // Create weight tensors: w1, w3 [1, 1, C, H], w2 [1, 1, H, C]
    uint32_t input_dim = input_shape.back();
    std::vector<uint32_t> w1_w3_shape = {1, 1, input_dim, hidden_dim};
    std::vector<uint32_t> w2_shape = {1, 1, hidden_dim, input_dim};

    xt::xarray<float> w1_data = xt::random::rand<float>(w1_w3_shape, -bound, bound);
    xt::xarray<float> w2_data = xt::random::rand<float>(w2_shape, -bound, bound);
    xt::xarray<float> w3_data = xt::random::rand<float>(w1_w3_shape, -bound, bound);

    CompareKernelVsReference(input_data, w1_data, w2_data, w3_data);
}

}  // namespace

// ============================================================================
// Section 2: SwiGLU Kernel vs Reference Implementation Tests
// ============================================================================
// These tests compare the SwiGLU kernel implementation against
// the reference implementation to ensure correctness.
// Only tests where C and hidden_dim are divisible by 32 are included.
//
// TODO: Add tests for masking in C and H (mask_w, mask_hw) when implementation supports it
// ============================================================================

// 1. Basic single tile test: 1x1x32x32, hidden_dim=32
TEST_F(SwiGLUOpTest, SwiGLU_Basic_1x1x32x32) {
    CompareKernelVsReferenceWithShape({1, 1, 32, 32}, 32);
}

// 2. Multi-tile width test: 1x1x32x64, hidden_dim=64 (C > 32)
TEST_F(SwiGLUOpTest, SwiGLU_MultiTile_1x1x32x64) {
    CompareKernelVsReferenceWithShape({1, 1, 32, 64}, 64);
}

// 3. Multi-tile height test: 1x1x64x32, hidden_dim=32 (S > 32)
TEST_F(SwiGLUOpTest, SwiGLU_MultiTile_1x1x64x32) {
    CompareKernelVsReferenceWithShape({1, 1, 64, 32}, 32);
}

// 4. Multi-batch test: 8x1x32x32, hidden_dim=32 (B != 32)
TEST_F(SwiGLUOpTest, SwiGLU_MultiBatch_8x1x32x32) {
    CompareKernelVsReferenceWithShape({8, 1, 32, 32}, 32);
}

// 5. Medium test: 2x1x32x128, hidden_dim=128
TEST_F(SwiGLUOpTest, SwiGLU_Medium_2x1x32x128) {
    CompareKernelVsReferenceWithShape({2, 1, 32, 128}, 128);
}

// 6. Large test: 4x1x64x256, hidden_dim=256
TEST_F(SwiGLUOpTest, SwiGLU_Large_4x1x64x256) {
    CompareKernelVsReferenceWithShape({4, 1, 64, 256}, 256);
}

// 7. NanoGPT-like shape: 32x1x256x384, hidden_dim=1024
TEST_F(SwiGLUOpTest, SwiGLU_NanoGPT_32x1x256x384) {
    CompareKernelVsReferenceWithShape({32, 1, 256, 384}, 1024);
}
