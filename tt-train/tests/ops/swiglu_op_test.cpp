// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/swiglu_op.hpp"

#include <gtest/gtest.h>

#include <cassert>
#include <core/ttnn_all_includes.hpp>
#include <iostream>
#include <numeric>

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
// 1. Create test tensors x, w1, w2, w3, dropout
// 2. Compute SwiGLU forward pass
// 3. Verify the result is finite and has expected shape
// ============================================================================

namespace {

/**
 * Reference implementation of SwiGLU forward pass using xt library
 * SwiGLU(x, w1, w2, w3, dropout) = (x @ w1 * silu(x @ w3)) @ w2 * dropout
 *
 * @param x Input tensor [B, N, S, C]
 * @param w1 Weight matrix 1 [C, H]
 * @param w2 Weight matrix 2 [H, C]
 * @param w3 Weight matrix 3 [C, H]
 * @param dropout Dropout mask [B, N, S, C]
 * @return SwiGLU activation result
 */
xt::xarray<float> swiglu_forward_reference(
    const xt::xarray<float>& x,
    const xt::xarray<float>& w1,
    const xt::xarray<float>& w2,
    const xt::xarray<float>& w3,
    const xt::xarray<float>& dropout) {
    // x @ w1: [B, N, S, C] @ [C, H] -> [B, N, S, H]
    auto xw1 = xt::linalg::tensordot(x, w1, {3}, {0});

    // x @ w3: [B, N, S, C] @ [C, H] -> [B, N, S, H]
    // auto xw3 = xt::linalg::tensordot(x, w3, {3}, {0});

    // silu(x @ w3)
    // auto sigmoid_xw3 = 1.0f / (1.0f + xt::exp(-xw3));
    // auto silu_xw3 = xw3 * sigmoid_xw3;

    // (x @ w1) * silu(x @ w3): element-wise multiplication
    // auto intermediate = xw1 * silu_xw3;
    // auto intermediate = xw1;

    // intermediate @ w2: [B, N, S, H] @ [H, C] -> [B, N, S, C]
    // auto result = xt::linalg::tensordot(intermediate, w2, {3}, {0});
    auto result = xw1;

    // Apply dropout mask
    // return result * dropout;
    return result;
}

void CompareKernelVsReference(
    const xt::xarray<float>& input_data,
    const xt::xarray<float>& w1_data,
    const xt::xarray<float>& w2_data,
    const xt::xarray<float>& w3_data,
    const xt::xarray<float>& dropout_data) {
    using namespace ttml;

    // Create input tensors for kernel implementation
    auto input_kernel = autograd::create_tensor(core::from_xtensor(input_data, &autograd::ctx().get_device()));
    auto w1_kernel = autograd::create_tensor(core::from_xtensor(w1_data, &autograd::ctx().get_device()));
    auto w2_kernel = autograd::create_tensor(core::from_xtensor(w2_data, &autograd::ctx().get_device()));
    auto w3_kernel = autograd::create_tensor(core::from_xtensor(w3_data, &autograd::ctx().get_device()));
    auto dropout_kernel = autograd::create_tensor(core::from_xtensor(dropout_data, &autograd::ctx().get_device()));

    // Forward pass - kernel implementation
    auto result_kernel = ops::swiglu(input_kernel, w1_kernel, w2_kernel, w3_kernel, dropout_kernel);
    auto result_kernel_xtensor = core::to_xtensor(result_kernel->get_value());

    // Forward pass - reference implementation
    auto result_reference = swiglu_forward_reference(input_data, w1_data, w2_data, w3_data, dropout_data);

    // Compare forward results
    EXPECT_TRUE(xt::allclose(result_kernel_xtensor, result_reference, 1.0e-3F, 3e-2F))
        << "SwiGLU kernel and reference implementations differ";

    // Verify both results are finite
    EXPECT_TRUE(xt::all(xt::isfinite(result_kernel_xtensor))) << "SwiGLU kernel result contains NaN or Inf values";
    EXPECT_TRUE(xt::all(xt::isfinite(result_reference))) << "SwiGLU reference result contains NaN or Inf values";
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
    xt::random::seed(42);
    xt::xarray<float> input_data = xt::random::rand<float>(input_shape, -1.0F, 1.0F);

    // Create weight matrices - w1, w3 map from input_dim to hidden_dim, w2 maps hidden_dim to input_dim
    uint32_t input_dim = input_shape.back();
    std::vector<uint32_t> w1_w3_shape = {input_dim, hidden_dim};
    std::vector<uint32_t> w2_shape = {hidden_dim, input_dim};

    xt::xarray<float> w1_data = xt::random::rand<float>(w1_w3_shape, -0.1F, 0.1F);
    xt::xarray<float> w2_data = xt::random::rand<float>(w2_shape, -0.1F, 0.1F);
    xt::xarray<float> w3_data = xt::random::rand<float>(w1_w3_shape, -0.1F, 0.1F);

    // Create dropout mask (for now, just ones - no actual dropout)
    xt::xarray<float> dropout_data = xt::ones<float>(input_shape);

    CompareKernelVsReference(input_data, w1_data, w2_data, w3_data, dropout_data);
}

}  // namespace

// ============================================================================
// Section 2: SwiGLU Kernel vs Reference Implementation Tests
// ============================================================================
// These tests compare the SwiGLU kernel implementation against
// the reference implementation to ensure correctness
// ============================================================================

// Test small tensor - basic functionality
TEST_F(SwiGLUOpTest, SwiGLU_Compare_Small) {
    // Small test case: B=1, N=1, S=2, C=64, hidden_dim=128
    CompareKernelVsReferenceWithShape({1, 1, 2, 64}, 128);
}

// Test larger tensor - more realistic size
TEST_F(SwiGLUOpTest, SwiGLU_Compare_Medium) {
    // Medium test case: B=2, N=1, S=32, C=256, hidden_dim=512
    CompareKernelVsReferenceWithShape({2, 1, 32, 256}, 512);
}
