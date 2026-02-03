// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/concat_op.hpp"

#include <gtest/gtest.h>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/losses.hpp"

class ConcatOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
        ttml::autograd::ctx().set_seed(42);
    }

    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
        ttml::autograd::ctx().close_device();
    }
};

// ============================================================================
// Section 1: Concat Forward Pass Tests
// ============================================================================
// These tests validate the concat forward pass implementation by comparing
// against expected results computed with xt::concatenate.
// ============================================================================

TEST_F(ConcatOpTest, Concat_TwoTensors_LastDim) {
    using namespace ttml;

    uint32_t N = 1, C = 1, H = 32, W = 64;
    auto prod = N * C * H * W;

    // Create input tensors (normalize to avoid bfloat16 precision issues)
    xt::xarray<float> xtensor_a = xt::arange<float>(0.F, static_cast<float>(prod)).reshape({N, C, H, W});
    xt::xarray<float> xtensor_b =
        xt::arange<float>(static_cast<float>(prod), static_cast<float>(2 * prod)).reshape({N, C, H, W});
    xtensor_a = xtensor_a / static_cast<float>(prod);
    xtensor_b = xtensor_b / static_cast<float>(prod);

    // Expected result from xtensor concatenation
    xt::xarray<float> expected = xt::concatenate(xt::xtuple(xtensor_a, xtensor_b), 3);

    // Create autograd tensors
    auto* device = &autograd::ctx().get_device();
    auto tensor_a = autograd::create_tensor(core::from_xtensor(xtensor_a, device));
    auto tensor_b = autograd::create_tensor(core::from_xtensor(xtensor_b, device));

    // Perform concat operation
    auto result = ops::concat({tensor_a, tensor_b}, 3);
    auto result_xtensor = core::to_xtensor(result->get_value());

    // Verify shape
    EXPECT_EQ(result_xtensor.shape()[0], N);
    EXPECT_EQ(result_xtensor.shape()[1], C);
    EXPECT_EQ(result_xtensor.shape()[2], H);
    EXPECT_EQ(result_xtensor.shape()[3], W * 2);

    // Verify values (use tolerance suitable for bfloat16)
    EXPECT_TRUE(xt::allclose(result_xtensor, expected, 7e-3F, 1e-6F));
}

TEST_F(ConcatOpTest, Concat_TwoTensors_HeightDim) {
    using namespace ttml;

    uint32_t N = 1, C = 1, H = 32, W = 64;
    auto prod = N * C * H * W;

    // Create input tensors (normalize to avoid bfloat16 precision issues)
    xt::xarray<float> xtensor_a = xt::arange<float>(0.F, static_cast<float>(prod)).reshape({N, C, H, W});
    xt::xarray<float> xtensor_b =
        xt::arange<float>(static_cast<float>(prod), static_cast<float>(2 * prod)).reshape({N, C, H, W});
    xtensor_a = xtensor_a / static_cast<float>(prod);
    xtensor_b = xtensor_b / static_cast<float>(prod);

    // Expected result from xtensor concatenation
    xt::xarray<float> expected = xt::concatenate(xt::xtuple(xtensor_a, xtensor_b), 2);

    // Create autograd tensors
    auto* device = &autograd::ctx().get_device();
    auto tensor_a = autograd::create_tensor(core::from_xtensor(xtensor_a, device));
    auto tensor_b = autograd::create_tensor(core::from_xtensor(xtensor_b, device));

    // Perform concat operation
    auto result = ops::concat({tensor_a, tensor_b}, 2);
    auto result_xtensor = core::to_xtensor(result->get_value());

    // Verify shape
    EXPECT_EQ(result_xtensor.shape()[0], N);
    EXPECT_EQ(result_xtensor.shape()[1], C);
    EXPECT_EQ(result_xtensor.shape()[2], H * 2);
    EXPECT_EQ(result_xtensor.shape()[3], W);

    // Verify values (use tolerance suitable for bfloat16)
    EXPECT_TRUE(xt::allclose(result_xtensor, expected, 7e-3F, 1e-6F));
}

TEST_F(ConcatOpTest, Concat_ThreeTensors_LastDim) {
    using namespace ttml;

    uint32_t N = 1, C = 1, H = 32, W = 32;
    auto prod = N * C * H * W;

    // Create three input tensors (normalize to avoid bfloat16 precision issues)
    xt::xarray<float> xtensor_a = xt::arange<float>(0.F, static_cast<float>(prod)).reshape({N, C, H, W});
    xt::xarray<float> xtensor_b =
        xt::arange<float>(static_cast<float>(prod), static_cast<float>(2 * prod)).reshape({N, C, H, W});
    xt::xarray<float> xtensor_c =
        xt::arange<float>(static_cast<float>(2 * prod), static_cast<float>(3 * prod)).reshape({N, C, H, W});
    xtensor_a = xtensor_a / static_cast<float>(3 * prod);
    xtensor_b = xtensor_b / static_cast<float>(3 * prod);
    xtensor_c = xtensor_c / static_cast<float>(3 * prod);

    // Expected result from xtensor concatenation
    xt::xarray<float> expected = xt::concatenate(xt::xtuple(xtensor_a, xtensor_b, xtensor_c), 3);

    // Create autograd tensors
    auto* device = &autograd::ctx().get_device();
    auto tensor_a = autograd::create_tensor(core::from_xtensor(xtensor_a, device));
    auto tensor_b = autograd::create_tensor(core::from_xtensor(xtensor_b, device));
    auto tensor_c = autograd::create_tensor(core::from_xtensor(xtensor_c, device));

    // Perform concat operation
    auto result = ops::concat({tensor_a, tensor_b, tensor_c}, 3);
    auto result_xtensor = core::to_xtensor(result->get_value());

    // Verify shape
    EXPECT_EQ(result_xtensor.shape()[0], N);
    EXPECT_EQ(result_xtensor.shape()[1], C);
    EXPECT_EQ(result_xtensor.shape()[2], H);
    EXPECT_EQ(result_xtensor.shape()[3], W * 3);

    // Verify values (use tolerance suitable for bfloat16)
    EXPECT_TRUE(xt::allclose(result_xtensor, expected, 7e-3F, 1e-6F));
}

TEST_F(ConcatOpTest, Concat_DifferentSizes_LastDim) {
    using namespace ttml;

    uint32_t N = 1, C = 1, H = 32;
    uint32_t W_a = 64, W_b = 128;
    auto prod_a = N * C * H * W_a;
    auto prod_b = N * C * H * W_b;

    // Create input tensors with different widths (normalize to avoid bfloat16 precision issues)
    xt::xarray<float> xtensor_a = xt::arange<float>(0.F, static_cast<float>(prod_a)).reshape({N, C, H, W_a});
    xt::xarray<float> xtensor_b = xt::arange<float>(0.F, static_cast<float>(prod_b)).reshape({N, C, H, W_b});
    xtensor_a = xtensor_a / static_cast<float>(prod_b);
    xtensor_b = xtensor_b / static_cast<float>(prod_b);

    // Expected result from xtensor concatenation
    xt::xarray<float> expected = xt::concatenate(xt::xtuple(xtensor_a, xtensor_b), 3);

    // Create autograd tensors
    auto* device = &autograd::ctx().get_device();
    auto tensor_a = autograd::create_tensor(core::from_xtensor(xtensor_a, device));
    auto tensor_b = autograd::create_tensor(core::from_xtensor(xtensor_b, device));

    // Perform concat operation
    auto result = ops::concat({tensor_a, tensor_b}, 3);
    auto result_xtensor = core::to_xtensor(result->get_value());

    // Verify shape
    EXPECT_EQ(result_xtensor.shape()[0], N);
    EXPECT_EQ(result_xtensor.shape()[1], C);
    EXPECT_EQ(result_xtensor.shape()[2], H);
    EXPECT_EQ(result_xtensor.shape()[3], W_a + W_b);

    // Verify values (use tolerance suitable for bfloat16)
    EXPECT_TRUE(xt::allclose(result_xtensor, expected, 7e-3F, 1e-6F));
}

TEST_F(ConcatOpTest, Concat_BatchDim) {
    using namespace ttml;

    uint32_t N = 2, C = 1, H = 32, W = 64;
    auto prod = N * C * H * W;

    // Create input tensors (normalize to avoid bfloat16 precision issues)
    xt::xarray<float> xtensor_a = xt::arange<float>(0.F, static_cast<float>(prod)).reshape({N, C, H, W});
    xt::xarray<float> xtensor_b =
        xt::arange<float>(static_cast<float>(prod), static_cast<float>(2 * prod)).reshape({N, C, H, W});
    xtensor_a = xtensor_a / static_cast<float>(2 * prod);
    xtensor_b = xtensor_b / static_cast<float>(2 * prod);

    // Expected result from xtensor concatenation along batch dimension
    xt::xarray<float> expected = xt::concatenate(xt::xtuple(xtensor_a, xtensor_b), 0);

    // Create autograd tensors
    auto* device = &autograd::ctx().get_device();
    auto tensor_a = autograd::create_tensor(core::from_xtensor(xtensor_a, device));
    auto tensor_b = autograd::create_tensor(core::from_xtensor(xtensor_b, device));

    // Perform concat operation
    auto result = ops::concat({tensor_a, tensor_b}, 0);
    auto result_xtensor = core::to_xtensor(result->get_value());

    // Verify shape
    EXPECT_EQ(result_xtensor.shape()[0], N * 2);
    EXPECT_EQ(result_xtensor.shape()[1], C);
    EXPECT_EQ(result_xtensor.shape()[2], H);
    EXPECT_EQ(result_xtensor.shape()[3], W);

    // Verify values (use tolerance suitable for bfloat16)
    EXPECT_TRUE(xt::allclose(result_xtensor, expected, 7e-3F, 1e-6F));
}

// ============================================================================
// Section 2: Concat Backward Pass Tests
// ============================================================================
// These tests validate the concat backward pass by verifying that gradients
// are correctly sliced and propagated back to the input tensors.
// ============================================================================

TEST_F(ConcatOpTest, Concat_Backward_TwoTensors_LastDim) {
    using namespace ttml;

    uint32_t N = 1, C = 1, H = 32, W = 64;
    auto prod = N * C * H * W;

    // Create input tensors with known values
    xt::xarray<float> xtensor_a = xt::arange<float>(0.F, static_cast<float>(prod)).reshape({N, C, H, W});
    xt::xarray<float> xtensor_b =
        xt::arange<float>(static_cast<float>(prod), static_cast<float>(2 * prod)).reshape({N, C, H, W});

    // Create autograd tensors
    auto* device = &autograd::ctx().get_device();
    auto tensor_a = autograd::create_tensor(core::from_xtensor(xtensor_a, device));
    auto tensor_b = autograd::create_tensor(core::from_xtensor(xtensor_b, device));

    // Perform concat operation
    auto result = ops::concat({tensor_a, tensor_b}, 3);

    // Compute loss (MSE with zeros target to trigger backward)
    auto target = autograd::create_tensor(core::zeros_like(result->get_value()));
    auto loss = ops::mse_loss(result, target);
    loss->backward();

    // Get gradients
    auto grad_a = core::to_xtensor(tensor_a->get_grad());
    auto grad_b = core::to_xtensor(tensor_b->get_grad());

    // Verify gradient shapes match input shapes
    EXPECT_EQ(grad_a.shape(), xtensor_a.shape());
    EXPECT_EQ(grad_b.shape(), xtensor_b.shape());

    // Verify gradients are finite
    EXPECT_TRUE(xt::all(xt::isfinite(grad_a)));
    EXPECT_TRUE(xt::all(xt::isfinite(grad_b)));

    // For MSE loss = mean((concat - 0)^2), d_loss/d_concat = 2 * concat / N
    // The gradient for tensor_a should correspond to the first half of the concat gradient
    // and gradient for tensor_b should correspond to the second half
    // Verify gradients are proportional to input values (since MSE with zeros target)
    auto concat_result = core::to_xtensor(result->get_value());
    auto total_elements = static_cast<float>(concat_result.size());

    // Expected gradients: 2 * input / total_elements
    xt::xarray<float> expected_grad_a = 2.0F * xtensor_a / total_elements;
    xt::xarray<float> expected_grad_b = 2.0F * xtensor_b / total_elements;

    EXPECT_TRUE(xt::allclose(grad_a, expected_grad_a, 1e-2F, 1e-3F));
    EXPECT_TRUE(xt::allclose(grad_b, expected_grad_b, 1e-2F, 1e-3F));
}

TEST_F(ConcatOpTest, Concat_Backward_ThreeTensors_HeightDim) {
    using namespace ttml;

    uint32_t N = 1, C = 1, H = 32, W = 64;
    auto prod = N * C * H * W;

    // Create three input tensors
    xt::xarray<float> xtensor_a = xt::ones<float>({N, C, H, W}) * 1.0F;
    xt::xarray<float> xtensor_b = xt::ones<float>({N, C, H, W}) * 2.0F;
    xt::xarray<float> xtensor_c = xt::ones<float>({N, C, H, W}) * 3.0F;

    // Create autograd tensors
    auto* device = &autograd::ctx().get_device();
    auto tensor_a = autograd::create_tensor(core::from_xtensor(xtensor_a, device));
    auto tensor_b = autograd::create_tensor(core::from_xtensor(xtensor_b, device));
    auto tensor_c = autograd::create_tensor(core::from_xtensor(xtensor_c, device));

    // Perform concat operation along height dimension
    auto result = ops::concat({tensor_a, tensor_b, tensor_c}, 2);

    // Compute loss
    auto target = autograd::create_tensor(core::zeros_like(result->get_value()));
    auto loss = ops::mse_loss(result, target);
    loss->backward();

    // Get gradients
    auto grad_a = core::to_xtensor(tensor_a->get_grad());
    auto grad_b = core::to_xtensor(tensor_b->get_grad());
    auto grad_c = core::to_xtensor(tensor_c->get_grad());

    // Verify gradient shapes
    EXPECT_EQ(grad_a.shape(), xtensor_a.shape());
    EXPECT_EQ(grad_b.shape(), xtensor_b.shape());
    EXPECT_EQ(grad_c.shape(), xtensor_c.shape());

    // Verify gradients are finite
    EXPECT_TRUE(xt::all(xt::isfinite(grad_a)));
    EXPECT_TRUE(xt::all(xt::isfinite(grad_b)));
    EXPECT_TRUE(xt::all(xt::isfinite(grad_c)));

    // Since inputs are uniform values, gradients should also be uniform
    // grad = 2 * input / total_elements
    auto total_elements = static_cast<float>(prod * 3);
    float expected_grad_a_val = 2.0F * 1.0F / total_elements;
    float expected_grad_b_val = 2.0F * 2.0F / total_elements;
    float expected_grad_c_val = 2.0F * 3.0F / total_elements;

    EXPECT_TRUE(xt::allclose(grad_a, xt::ones_like(grad_a) * expected_grad_a_val, 1e-2F, 1e-3F));
    EXPECT_TRUE(xt::allclose(grad_b, xt::ones_like(grad_b) * expected_grad_b_val, 1e-2F, 1e-3F));
    EXPECT_TRUE(xt::allclose(grad_c, xt::ones_like(grad_c) * expected_grad_c_val, 1e-2F, 1e-3F));
}

TEST_F(ConcatOpTest, Concat_Backward_DifferentSizes_LastDim) {
    using namespace ttml;

    uint32_t N = 1, C = 1, H = 32;
    uint32_t W_a = 64, W_b = 128;

    // Create input tensors with different widths but uniform values for easy gradient verification
    xt::xarray<float> xtensor_a = xt::ones<float>({N, C, H, W_a}) * 1.0F;
    xt::xarray<float> xtensor_b = xt::ones<float>({N, C, H, W_b}) * 2.0F;

    // Create autograd tensors
    auto* device = &autograd::ctx().get_device();
    auto tensor_a = autograd::create_tensor(core::from_xtensor(xtensor_a, device));
    auto tensor_b = autograd::create_tensor(core::from_xtensor(xtensor_b, device));

    // Perform concat operation
    auto result = ops::concat({tensor_a, tensor_b}, 3);

    // Compute loss
    auto target = autograd::create_tensor(core::zeros_like(result->get_value()));
    auto loss = ops::mse_loss(result, target);
    loss->backward();

    // Get gradients
    auto grad_a = core::to_xtensor(tensor_a->get_grad());
    auto grad_b = core::to_xtensor(tensor_b->get_grad());

    // Verify gradient shapes match input shapes (critical for different-sized inputs)
    EXPECT_EQ(grad_a.shape(), xtensor_a.shape());
    EXPECT_EQ(grad_b.shape(), xtensor_b.shape());

    // Verify gradients are finite
    EXPECT_TRUE(xt::all(xt::isfinite(grad_a)));
    EXPECT_TRUE(xt::all(xt::isfinite(grad_b)));

    // Verify gradient values
    auto total_elements = static_cast<float>(N * C * H * (W_a + W_b));
    float expected_grad_a_val = 2.0F * 1.0F / total_elements;
    float expected_grad_b_val = 2.0F * 2.0F / total_elements;

    EXPECT_TRUE(xt::allclose(grad_a, xt::ones_like(grad_a) * expected_grad_a_val, 1e-2F, 1e-3F));
    EXPECT_TRUE(xt::allclose(grad_b, xt::ones_like(grad_b) * expected_grad_b_val, 1e-2F, 1e-3F));
}

// ============================================================================
// Section 3: Edge Cases and Larger Tensor Tests
// ============================================================================

TEST_F(ConcatOpTest, NIGHTLY_Concat_LargeTensors_LastDim) {
    using namespace ttml;

    // Test with larger tensors similar to real model usage
    uint32_t N = 8, C = 1, H = 256, W = 384;
    auto prod = N * C * H * W;

    xt::xarray<float> xtensor_a = xt::arange<float>(0.F, static_cast<float>(prod)).reshape({N, C, H, W});
    xt::xarray<float> xtensor_b =
        xt::arange<float>(static_cast<float>(prod), static_cast<float>(2 * prod)).reshape({N, C, H, W});

    // Normalize to avoid numerical issues with large values
    xtensor_a = xtensor_a / static_cast<float>(prod);
    xtensor_b = xtensor_b / static_cast<float>(prod);

    xt::xarray<float> expected = xt::concatenate(xt::xtuple(xtensor_a, xtensor_b), 3);

    auto* device = &autograd::ctx().get_device();
    auto tensor_a = autograd::create_tensor(core::from_xtensor(xtensor_a, device));
    auto tensor_b = autograd::create_tensor(core::from_xtensor(xtensor_b, device));

    auto result = ops::concat({tensor_a, tensor_b}, 3);
    auto result_xtensor = core::to_xtensor(result->get_value());

    EXPECT_EQ(result_xtensor.shape()[0], N);
    EXPECT_EQ(result_xtensor.shape()[1], C);
    EXPECT_EQ(result_xtensor.shape()[2], H);
    EXPECT_EQ(result_xtensor.shape()[3], W * 2);

    EXPECT_TRUE(xt::allclose(result_xtensor, expected, 1e-2F, 1e-3F));
}

TEST_F(ConcatOpTest, NIGHTLY_Concat_LargeTensors_WithBackward) {
    using namespace ttml;

    // Test backward pass with larger tensors
    uint32_t N = 4, C = 1, H = 128, W = 256;

    xt::xarray<float> xtensor_a = xt::ones<float>({N, C, H, W}) * 0.5F;
    xt::xarray<float> xtensor_b = xt::ones<float>({N, C, H, W}) * 1.5F;

    auto* device = &autograd::ctx().get_device();
    auto tensor_a = autograd::create_tensor(core::from_xtensor(xtensor_a, device));
    auto tensor_b = autograd::create_tensor(core::from_xtensor(xtensor_b, device));

    auto result = ops::concat({tensor_a, tensor_b}, 3);

    auto target = autograd::create_tensor(core::zeros_like(result->get_value()));
    auto loss = ops::mse_loss(result, target);
    loss->backward();

    auto grad_a = core::to_xtensor(tensor_a->get_grad());
    auto grad_b = core::to_xtensor(tensor_b->get_grad());

    // Verify shapes
    EXPECT_EQ(grad_a.shape(), xtensor_a.shape());
    EXPECT_EQ(grad_b.shape(), xtensor_b.shape());

    // Verify gradients are finite
    EXPECT_TRUE(xt::all(xt::isfinite(grad_a)));
    EXPECT_TRUE(xt::all(xt::isfinite(grad_b)));

    // Verify gradient values
    auto total_elements = static_cast<float>(N * C * H * W * 2);
    float expected_grad_a_val = 2.0F * 0.5F / total_elements;
    float expected_grad_b_val = 2.0F * 1.5F / total_elements;

    EXPECT_TRUE(xt::allclose(grad_a, xt::ones_like(grad_a) * expected_grad_a_val, 1e-2F, 1e-3F));
    EXPECT_TRUE(xt::allclose(grad_b, xt::ones_like(grad_b) * expected_grad_b_val, 1e-2F, 1e-3F));
}
