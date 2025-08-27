// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/swiglu_op.hpp"

#include <gtest/gtest.h>

#include <cassert>
#include <core/ttnn_all_includes.hpp>
#include <iostream>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "xtensor-blas/xlinalg.hpp"

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

// TODO(maciek): Check and create consistent naming of variables denoting dimensions (especially hidden ones)

/**
 * Reference implementation of SwiGLU forward pass using xt library
 * SwiGLU(x, w1, w2, w3) = (silu(x @ w1) * (x @ w3)) @ w2
 *
 * @param x Input tensor [B, N, S, C]
 * @param w1 Weight matrix 1 [C, H]
 * @param w2 Weight matrix 2 [H, C]
 * @param w3 Weight matrix 3 [C, H]
 * @return SwiGLU activation result
 */
xt::xarray<float> swiglu_forward_reference(
    const xt::xarray<float>& x, const xt::xarray<float>& w1, const xt::xarray<float>& w2, const xt::xarray<float>& w3) {
    // x @ w1: [B, N, S, C] @ [C, H] -> [B, N, S, H]
    auto xw1 = xt::linalg::tensordot(x, w1, {3}, {0});
    // x @ w3: [B, N, S, C] @ [C, H] -> [B, N, S, H]
    auto xw3 = xt::linalg::tensordot(x, w3, {3}, {0});
    // Apply SiLU activation: SiLU(x) = x * sigmoid(x)
    auto silu = [](const xt::xarray<float>& t) {
        auto sigmoid = 1.0f / (1.0f + xt::exp(-t));
        return t * sigmoid;
    };
    auto gated = silu(xw1) * xw3;

    return xt::linalg::tensordot(gated, w2, {3}, {0});
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
    std::cout << std::endl << "SwiGLU kernel result from kernel in test: " << std::endl;
    auto result_kernel_xtensor = core::to_xtensor(result_kernel->get_value());

    // Forward pass - reference implementation
    auto result_reference = swiglu_forward_reference(input_data, w1_data, w2_data, w3_data);
    std::cout << std::endl << "SwiGLU reference result in test:" << std::endl;

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

    // Create weight matrices - w1, w3 map from input_dim to hidden_dim, w2 maps hidden_dim to input_dim
    uint32_t input_dim = input_shape.back();
    std::vector<uint32_t> w1_w3_shape = {input_dim, hidden_dim};
    std::vector<uint32_t> w2_shape = {hidden_dim, input_dim};

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
// the reference implementation to ensure correctness
// ============================================================================

// 1. Basic 1x1x32x32 test (no masking)
TEST_F(SwiGLUOpTest, SwiGLU_Basic_1x1x32x32) {
    CompareKernelVsReferenceWithShape({1, 1, 32, 32}, 32);
}

// 2. Medium test: 2x1x32x128, hidden_dim=128 (all divisible by 32)
TEST_F(SwiGLUOpTest, SwiGLU_Medium_2x1x32x128) {
    CompareKernelVsReferenceWithShape({2, 1, 32, 128}, 128);
}

// 3. Medium test: 2x1x32x256, hidden_dim=256 (all divisible by 32)
TEST_F(SwiGLUOpTest, SwiGLU_Medium_2x1x32x256) {
    CompareKernelVsReferenceWithShape({2, 1, 32, 256}, 256);
}

// 4. Medium test: 2x1x35x128, hidden_dim=128 (rows not divisible by 32)
TEST_F(SwiGLUOpTest, SwiGLU_MaskRows_Medium) {
    CompareKernelVsReferenceWithShape({2, 1, 35, 128}, 128);
}

// 5. Medium test: 2x1x32x100, hidden_dim=128 (C not divisible by 32, mask_w)
TEST_F(SwiGLUOpTest, SwiGLU_MaskW_Medium) {
    CompareKernelVsReferenceWithShape({2, 1, 32, 100}, 128);
}

// 6. Medium test: 2x1x32x128, hidden_dim=100 (hidden_dim not divisible by 32, mask_hw)
TEST_F(SwiGLUOpTest, SwiGLU_MaskHW_Medium) {
    CompareKernelVsReferenceWithShape({2, 1, 32, 128}, 100);
}

// 7. Larger test: 4x1x64x256, hidden_dim=256 (all divisible by 32)
TEST_F(SwiGLUOpTest, SwiGLU_Large_4x1x64x256) {
    CompareKernelVsReferenceWithShape({4, 1, 64, 256}, 256);
}

// 8. Larger test: 4x1x64x512, hidden_dim=256 (all divisible by 32)
TEST_F(SwiGLUOpTest, SwiGLU_Large_4x1x64x512) {
    CompareKernelVsReferenceWithShape({4, 1, 64, 512}, 256);
}

// 9. Larger test: 4x1x65x256, hidden_dim=256 (rows not divisible by 32)
TEST_F(SwiGLUOpTest, SwiGLU_MaskRows_Large) {
    CompareKernelVsReferenceWithShape({4, 1, 65, 256}, 256);
}

// 10. Larger test: 4x1x64x260, hidden_dim=260 (C and hidden_dim not divisible by 32)
TEST_F(SwiGLUOpTest, SwiGLU_MaskW_MaskHW_Large) {
    CompareKernelVsReferenceWithShape({4, 1, 64, 260}, 260);
}
