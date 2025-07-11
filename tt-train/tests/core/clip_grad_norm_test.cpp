// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "core/clip_grad_norm.hpp"

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/views/xview.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"

class ClipGradNormTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(ClipGradNormTest, ClipGradNorm_GENEROUS_TOLERANCE) {
    using namespace ttml;

    auto* device = &autograd::ctx().get_device();

    // Test tensors with manually set gradients
    const uint32_t num_tensors = 3U;
    const uint32_t tensor_size = 4U;
    std::vector<autograd::TensorPtr> tensors;
    std::vector<xt::xarray<float>> expected_grads = {
        {1.0F, 2.0F, 3.0F, 4.0F}, {-2.0F, -3.0F, -4.0F, 5.0F}, {0.5F, -1.5F, 2.5F, -3.5F}};

    // Compute reference total p2 norm manually; the "total norm" in pytorch
    // parlance is the L2 norm of tensor obtained by concatenating all the grads
    // together.
    auto p2_norm = [](const xt::xarray<float>& grad) { return std::sqrt(xt::sum(grad * grad)()); };

    auto concatenated_grads = core::concat(expected_grads);

    float expected_total_p2_norm = p2_norm(concatenated_grads);

    // reshape grads to 4D, required for compatibility with ttnn ops.
    for (auto& grad : expected_grads) {
        grad.reshape({1U, 1U, 1U, tensor_size});
    }

    // Create tensors and set their gradients
    for (uint32_t i = 0; i < num_tensors; i++) {
        auto tensor = autograd::create_tensor(core::zeros(ttnn::Shape({1U, 1U, 1U, tensor_size}), device));
        auto grad_tensor = core::from_xtensor(expected_grads[i], device);
        tensor->set_grad(grad_tensor);
        tensors.push_back(tensor);
    }

    // Set max norm to be half of the actual norm to force clipping
    float max_norm = expected_total_p2_norm / 2.0F;
    float scale = max_norm / expected_total_p2_norm;

    // Compute expected clipped gradients
    std::vector<xt::xarray<float>> expected_clipped_grads;
    expected_clipped_grads.reserve(expected_grads.size());
    for (const auto& grad : expected_grads) {
        expected_clipped_grads.push_back(grad * scale);
    }

    // Run clip_grad_norm
    serialization::NamedParameters named_params;
    for (uint32_t i = 0; i < num_tensors; i++) {
        named_params[std::to_string(i)] = tensors[i];
    }
    auto result = core::clip_grad_norm(named_params, max_norm, 2.0F);
    float computed_total_norm = core::to_vector(result->get_value()).front();

    // Verify total norm is approx. correct
    // Total norm is a lot further off due to compounded bf16 errors
    EXPECT_NEAR(computed_total_norm, expected_total_p2_norm, 4e-1F);

    // Verify gradients were scaled approx. correctly
    for (uint32_t i = 0; i < num_tensors; i++) {
        auto grad_data = core::to_xtensor(tensors[i]->get_grad());
        EXPECT_TRUE(xt::allclose(grad_data, expected_clipped_grads[i], 7e-2F));
    }
}
