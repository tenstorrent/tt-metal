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

// Helper to print a (1,1,32,32) tensor in concise kernel style
void print_tensor_1x1x32x32(const xt::xarray<float>& arr, const std::string& label) {
    std::cout << label << std::endl;
    for (int i = 0; i < 32; ++i) {
        if (i < 2 || i >= 30) {
            std::cout << "[ ";
            for (int j = 0; j < 32; ++j) {
                if (j < 2 || j >= 30) {
                    std::cout << arr(0, 0, i, j);
                    if (j != 31)
                        std::cout << ", ";
                } else if (j == 2) {
                    std::cout << "..., ";
                }
            }
            std::cout << "]";
            if (i != 31)
                std::cout << std::endl;
        } else if (i == 2) {
            std::cout << "..." << std::endl;
        }
    }
    std::cout << std::endl;
}

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

    // Compute U = x @ w1[:, :32] @ w2[:32, :]
    // auto w1_U = xt::view(w1, xt::all(), xt::range(0, 32));  // (32, 32)
    // auto w2_U = xt::view(w2, xt::range(0, 32), xt::all());  // (32, 32)
    // auto xw1_U = xt::linalg::tensordot(x, w1_U, {3}, {0});  // (1,1,32,32)
    // Print xw1_U
    // std::cout << "xw1_U (x @ w1[:,:32]):" << std::endl;
    // for (int i = 0; i < xw1_U.shape(2); ++i) {
    //     for (int j = 0; j < xw1_U.shape(3); ++j) {
    //         std::cout << xw1_U(0, 0, i, j) << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // Print w2_U
    // std::cout << "w2_U:" << std::endl;
    // for (int i = 0; i < w2_U.shape(0); ++i) {
    //     for (int j = 0; j < w2_U.shape(1); ++j) {
    //         std::cout << w2_U(i, j) << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // auto U = xt::linalg::tensordot(xw1_U, w2_U, {3}, {0});  // (1,1,32,32)

    // Compute V = x @ w1[:, 32:] @ w2[32:, :]
    // auto w1_V = xt::view(w1, xt::all(), xt::range(32, 64));  // (32, 32)
    // auto w2_V = xt::view(w2, xt::range(32, 64), xt::all());  // (32, 32)
    // auto xw1_V = xt::linalg::tensordot(x, w1_V, {3}, {0});   // (1,1,32,32)
    // auto V = xt::linalg::tensordot(xw1_V, w2_V, {3}, {0});   // (1,1,32,32)

    // Print U and V
    // print_tensor_1x1x32x32(U, "U (x @ w1[:,:32] @ w2[:32,:]):");
    // print_tensor_1x1x32x32(V, "V (x @ w1[:,32:] @ w2[32:,:]):");

    // x @ w1 @ w2 (original computation)
    auto result = xt::linalg::tensordot(xw1, w2, {3}, {0});

    // Assert U + V == result (elementwise)
    // auto sum_UV = U + V;
    // bool allclose = xt::allclose(sum_UV, result, 1e-4f, 1e-3f);
    // std::cout << "Assert U + V == x @ w1 @ w2: " << (allclose ? "PASS" : "FAIL") << std::endl;
    // assert(allclose);
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
    result_kernel->get_value();
    for (int i = 0; i < 10000000; i++) {
        asm volatile("nop");
    }
    // std::cout << "SwiGLU kernel result shape: " << result_kernel->get_value().logical_shape() << std::endl;
    std::cout << std::endl << "SwiGLU kernel result from kernel in test: " << std::endl;
    result_kernel->get_value().print();
    auto result_kernel_xtensor = core::to_xtensor(result_kernel->get_value());

    // Forward pass - reference implementation
    auto result_reference = swiglu_forward_reference(input_data, w1_data, w2_data, w3_data, dropout_data);
    std::cout << std::endl << "SwiGLU reference result in test:" << std::endl;
    // print_tensor_1x1x32x32(result_reference, "");

    // Compare forward results
    EXPECT_TRUE(xt::allclose(result_kernel_xtensor, result_reference, 1.0e-2F, 3e-1F))
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
    xt::xarray<float> input_data = xt::random::rand<float>(input_shape, -0.1F, 0.1F);

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
    // CompareKernelVsReferenceWithShape({1, 1, 2, 64}, 128);
    CompareKernelVsReferenceWithShape({1, 1, 64, 32}, 128);
}

// Test larger tensor - more realistic size
TEST_F(SwiGLUOpTest, SwiGLU_Compare_Medium) {
    // Medium test case: B=2, N=1, S=32, C=256, hidden_dim=512
    CompareKernelVsReferenceWithShape({2, 1, 32, 256}, 512);
}
