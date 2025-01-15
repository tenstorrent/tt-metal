// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "core/clip_grad_norm.hpp"

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>

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
    const uint32_t num_tensors = 3;
    const uint32_t tensor_size = 4;
    std::vector<autograd::TensorPtr> tensors;
    std::vector<float> expected_grads = {
        1.0f,
        2.0f,
        3.0f,
        4.0f,  // tensor 1
        -2.0f,
        3.0f,
        -4.0f,
        5.0f,  // tensor 2
        0.5f,
        -1.5f,
        2.5f,
        -3.5f};  // tensor 3

    // Create tensors and set their gradients
    for (uint32_t i = 0; i < num_tensors; i++) {
        auto tensor = autograd::create_tensor(core::zeros(core::create_shape({tensor_size}), device));
        auto grad_tensor = core::from_vector(
            std::vector<float>(
                expected_grads.begin() + i * tensor_size, expected_grads.begin() + (i + 1) * tensor_size),
            core::create_shape({1, 1, 1, tensor_size}),
            device);
        tensor->set_grad(grad_tensor);
        tensors.push_back(tensor);
    }

    // Compute reference total p2 norm manually
    float total_sqr_p2_norm = 0.0f;
    for (const auto& grad : expected_grads) {
        total_sqr_p2_norm = total_sqr_p2_norm + grad * grad;
    }
    float expected_total_p2_norm = std::sqrt(total_sqr_p2_norm);

    // Set max norm to be half of the actual norm to force clipping
    float max_norm = expected_total_p2_norm / 2.0f;
    float scale = max_norm / expected_total_p2_norm;

    // Compute expected clipped gradients
    std::vector<float> expected_clipped_grads;
    expected_clipped_grads.reserve(expected_grads.size());
    for (const auto& grad : expected_grads) {
        expected_clipped_grads.push_back(grad * scale);
    }

    // Run clip_grad_norm
    serialization::NamedParameters named_params;
    for (uint32_t i = 0; i < num_tensors; i++) {
        named_params[std::to_string(i)] = tensors[i];
    }
    auto result = core::clip_grad_norm(named_params, max_norm, 2.0f);
    float computed_total_norm = core::to_vector(result->get_value()).front();

    // Verify total norm is approx. correct
    // Total norm is a lot further off due to compounded bf16 errors
    EXPECT_NEAR(computed_total_norm, expected_total_p2_norm, 1e-0f);

    // Verify gradients were scaled approx. correctly
    for (uint32_t i = 0; i < num_tensors; i++) {
        auto grad_data = core::to_vector(tensors[i]->get_grad());
        for (uint32_t j = 0; j < tensor_size; j++) {
            EXPECT_NEAR(grad_data[j], expected_clipped_grads[i * tensor_size + j], 1e-1f);
        }
    }
}
