// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cassert>
#include <core/ttnn_all_includes.hpp>
#include <iostream>
#include <numeric>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "metal/ops/relu_sq_bw/device/relu_sq_bw_device_operation.hpp"  // Direct include for ttnn::prim registration
#include "ops/losses.hpp"

class ReLUSquaredBwOpTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }

    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }
};

// ============================================================================
// Section 1: ReLU Squared Backward Kernel vs Reference Implementation
// ============================================================================
// These tests validate the relu_sq_bw kernel implementation against
// a reference implementation.
//
// Test methodology:
// 1. Create test tensors: input and grad
// 2. Compute backward using kernel implementation (ttnn::prim::ttml_relu_sq_bw)
// 3. Compute backward using reference implementation
// 4. Compare results for numerical correctness
// ============================================================================

namespace {

/**
 * Reference implementation of ReLU Squared backward pass
 * d/dx(relu(x)^2) = 2 * relu(x) * grad
 * where relu(x) = max(0, x)
 *
 * @param x Input tensor (original input to relu_squared)
 * @param grad Gradient from upstream
 * @return Gradient with respect to input
 */
xt::xarray<float> relu_squared_backward_reference(const xt::xarray<float>& x, const xt::xarray<float>& grad) {
    auto relu_x = xt::maximum(x, 0.0f);
    return 2.0f * relu_x * grad;
}

/**
 * Helper function to compare kernel vs reference implementations
 */
void CompareKernelVsReference(const xt::xarray<float>& input_data, const xt::xarray<float>& grad_data) {
    using namespace ttml;

    // Create input tensors
    auto input_tensor = autograd::create_tensor(core::from_xtensor(input_data, &autograd::ctx().get_device()));
    auto grad_tensor = autograd::create_tensor(core::from_xtensor(grad_data, &autograd::ctx().get_device()));

    // Kernel implementation - call the ttml kernel directly
    auto kernel_result = ttnn::prim::ttml_relu_sq_bw(input_tensor->get_value(), grad_tensor->get_value());
    auto kernel_result_xtensor = core::to_xtensor(kernel_result);

    // Reference implementation
    auto reference_result = relu_squared_backward_reference(input_data, grad_data);

    // Compare results
    EXPECT_TRUE(xt::allclose(kernel_result_xtensor, reference_result, 1.0e-3F, 3e-2F));
}

/**
 * Helper function to test with a specific shape
 */
static void CompareKernelVsReferenceWithShape(const std::vector<uint32_t>& shape) {
    using namespace ttml;

    // Generate random input data
    xt::random::seed(42);
    xt::xarray<float> input_data = xt::random::rand<float>(shape, -10.0F, 10.0F);
    xt::xarray<float> grad_data = xt::random::rand<float>(shape, -100.0F, 100.0F);

    CompareKernelVsReference(input_data, grad_data);
}

}  // namespace

// ============================================================================
// Section 2: Comprehensive Tests
// ============================================================================

// Test small tensor - basic functionality
TEST_F(ReLUSquaredBwOpTest, ReLUSquaredBw_Compare_Small) {
    CompareKernelVsReferenceWithShape({1, 1, 1, 128});
}

// Test block_size alignment patterns
TEST_F(ReLUSquaredBwOpTest, ReLUSquaredBw_Compare_BlockSize_Remainder0) {
    // C=128, Wt=4, Wt%4=0
    CompareKernelVsReferenceWithShape({1, 1, 1, 128});
}

TEST_F(ReLUSquaredBwOpTest, ReLUSquaredBw_Compare_BlockSize_Remainder1) {
    // C=160, Wt=5, Wt%4=1
    CompareKernelVsReferenceWithShape({1, 1, 1, 160});
}

TEST_F(ReLUSquaredBwOpTest, ReLUSquaredBw_Compare_BlockSize_Remainder2) {
    // C=192, Wt=6, Wt%4=2
    CompareKernelVsReferenceWithShape({1, 1, 1, 192});
}

TEST_F(ReLUSquaredBwOpTest, ReLUSquaredBw_Compare_BlockSize_Remainder3) {
    // C=224, Wt=7, Wt%4=3
    CompareKernelVsReferenceWithShape({1, 1, 1, 224});
}

// Test realistic transformer shapes
TEST_F(ReLUSquaredBwOpTest, ReLUSquaredBw_Compare_NanoLlama) {
    // NanoLlama-like shape: [B=2, N=1, S=64, C=512]
    CompareKernelVsReferenceWithShape({2, 1, 64, 512});
}

TEST_F(ReLUSquaredBwOpTest, ReLUSquaredBw_Compare_LargeSequence) {
    // Large sequence length
    CompareKernelVsReferenceWithShape({1, 1, 2048, 512});
}

// Test edge cases
TEST_F(ReLUSquaredBwOpTest, ReLUSquaredBw_Compare_AllNegative) {
    using namespace ttml;

    std::vector<uint32_t> shape = {1, 1, 32, 32};
    xt::xarray<float> input_data = xt::ones<float>(shape) * -5.0f;  // All negative
    xt::xarray<float> grad_data = xt::random::rand<float>(shape, -100.0F, 100.0F);

    CompareKernelVsReference(input_data, grad_data);
}

TEST_F(ReLUSquaredBwOpTest, ReLUSquaredBw_Compare_AllPositive) {
    using namespace ttml;

    std::vector<uint32_t> shape = {1, 1, 32, 32};
    xt::xarray<float> input_data = xt::random::rand<float>(shape, 0.1F, 10.0F);  // All positive
    xt::xarray<float> grad_data = xt::random::rand<float>(shape, -100.0F, 100.0F);

    CompareKernelVsReference(input_data, grad_data);
}

TEST_F(ReLUSquaredBwOpTest, ReLUSquaredBw_Compare_Mixed) {
    using namespace ttml;

    std::vector<uint32_t> shape = {1, 1, 32, 32};
    xt::xarray<float> input_data = xt::random::rand<float>(shape, -10.0F, 10.0F);  // Mixed positive/negative
    xt::xarray<float> grad_data = xt::random::rand<float>(shape, -100.0F, 100.0F);

    CompareKernelVsReference(input_data, grad_data);
}
