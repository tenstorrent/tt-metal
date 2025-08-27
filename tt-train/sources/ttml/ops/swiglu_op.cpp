// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <core/ttnn_all_includes.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/matmul/matmul.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"
#include "autograd/tensor.hpp"
#include "metal/operations.hpp"
#include "ops/unary_ops.hpp"

namespace ttml::ops {

autograd::TensorPtr swiglu(
    const autograd::TensorPtr& tensor,
    const autograd::TensorPtr& w1,
    const autograd::TensorPtr& w2,
    const autograd::TensorPtr& w3,
    const autograd::TensorPtr& dropout) {
    auto a_shape = tensor->get_value().logical_shape();
    if (a_shape.rank() != 4) {
        throw std::runtime_error("swiglu only supports rank-4 input tensors.");
    }

    auto device = &autograd::ctx().get_device();

    auto swiglu_fw_result = ttml::metal::swiglu_fw(
        tensor->get_value(), w1->get_value(), w2->get_value(), w3->get_value(), dropout->get_value());

    auto out = autograd::create_tensor(swiglu_fw_result);

    autograd::GradFunction grad = [tensor, w1, w2, w3, dropout, out]() {
        auto dL_dout = out->get_grad();

        // Manual backward computation - we need to recompute forward intermediates
        constexpr auto none = ttsl::Span<const ttnn::operations::unary::UnaryWithParam>{};

        // Recompute forward intermediates for backward pass
        auto linear1 = ttnn::matmul(tensor->get_value(), w1->get_value());  // x @ w1
        auto gate = ttnn::matmul(tensor->get_value(), w3->get_value());     // x @ w3
        auto sigmoid_linear1 = ttnn::sigmoid(linear1);                      // sigmoid(x @ w1)
        auto swished = ttnn::multiply(
            linear1, sigmoid_linear1, std::nullopt, std::nullopt, std::nullopt, none, none, none, false);  // silu(x @
                                                                                                           // w1)
        auto gated = ttnn::multiply(
            swished, gate, std::nullopt, std::nullopt, std::nullopt, none, none, none, false);  // silu(x @ w1) * (x @
                                                                                                // w3)
        auto projected = ttnn::matmul(gated, w2->get_value());                                  // gated @ w2

        // Backward through dropout: dL_dprojected = dL_dout * dropout
        auto dL_dprojected = ttnn::multiply(
            dL_dout, dropout->get_value(), std::nullopt, std::nullopt, std::nullopt, none, none, none, false);

        // Backward through final matmul: dL_dgated = dL_dprojected @ w2^T
        auto dL_dgated = ttnn::matmul(dL_dprojected, ttnn::transpose(w2->get_value(), -2, -1));

        // Backward through element-wise multiply:
        // dL_dswished = dL_dgated * gate
        // dL_dgate = dL_dgated * swished
        auto dL_dswished =
            ttnn::multiply(dL_dgated, gate, std::nullopt, std::nullopt, std::nullopt, none, none, none, false);
        auto dL_dgate =
            ttnn::multiply(dL_dgated, swished, std::nullopt, std::nullopt, std::nullopt, none, none, none, false);

        // For SiLU backward: SiLU'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        auto silu_grad = ttnn::multiply(
            sigmoid_linear1,
            ttnn::add(
                ttnn::ones_like(sigmoid_linear1),
                ttnn::multiply(
                    linear1,
                    ttnn::subtract(
                        ttnn::ones_like(sigmoid_linear1),
                        sigmoid_linear1,
                        std::nullopt,
                        std::nullopt,
                        std::nullopt,
                        none,
                        none,
                        none,
                        false),
                    std::nullopt,
                    std::nullopt,
                    std::nullopt,
                    none,
                    none,
                    none,
                    false),
                std::nullopt,
                std::nullopt,
                std::nullopt,
                none,
                none,
                none,
                false),
            std::nullopt,
            std::nullopt,
            std::nullopt,
            none,
            none,
            none,
            false);

        auto dL_dlinear1 =
            ttnn::multiply(dL_dswished, silu_grad, std::nullopt, std::nullopt, std::nullopt, none, none, none, false);
        auto dL_dtensor_from_w1 = ttnn::matmul(dL_dlinear1, ttnn::transpose(w1->get_value(), -2, -1));
        auto dL_dtensor_from_w3 = ttnn::matmul(dL_dgate, ttnn::transpose(w3->get_value(), -2, -1));

        // Combine gradients from both paths
        auto dL_dtensor = ttnn::add(
            dL_dtensor_from_w1, dL_dtensor_from_w3, std::nullopt, std::nullopt, std::nullopt, none, none, none, false);

        tensor->add_grad(dL_dtensor);
    };

    auto links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

autograd::TensorPtr swiglu_composite(
    const autograd::TensorPtr& tensor,
    const autograd::TensorPtr& w1,
    const autograd::TensorPtr& w2,
    const autograd::TensorPtr& w3,
    const autograd::TensorPtr& dropout_mask) {
    // x * W1, then apply SiLU activation
    auto swished = ops::silu(autograd::create_tensor(ttnn::matmul(tensor->get_value(), w1->get_value())));

    // x * W3 (gate)
    auto gate = autograd::create_tensor(ttnn::matmul(tensor->get_value(), w3->get_value()));

    // Element-wise multiply: SiLU(x * W1) ⊙ (x * W3)
    auto gated = autograd::create_tensor(ttnn::multiply(swished->get_value(), gate->get_value()));

    // Final projection: result * W2
    auto x = autograd::create_tensor(ttnn::matmul(gated->get_value(), w2->get_value()));

    // Apply dropout last (after final projection)
    auto output = autograd::create_tensor(ttnn::multiply(x->get_value(), dropout_mask->get_value()));

    // Set up gradient computation for input tensor
    autograd::GradFunction grad = [tensor, w1, w2, w3, dropout_mask, output, swished, gate, gated, x]() {
        auto dL_dout = output->get_grad();

        // Backward through dropout: dL_dx = dL_dout * dropout_mask
        constexpr auto none = ttsl::Span<const ttnn::operations::unary::UnaryWithParam>{};
        auto dL_dx = ttnn::multiply(
            dL_dout, dropout_mask->get_value(), std::nullopt, std::nullopt, std::nullopt, none, none, none, false);

        // Backward through final matmul: dL_dgated = dL_dx @ w2^T
        auto dL_dgated = ttnn::matmul(dL_dx, ttnn::transpose(w2->get_value(), -2, -1));

        // Backward through element-wise multiply:
        // dL_dswished = dL_dgated * gate
        // dL_dgate = dL_dgated * swished
        auto dL_dswished = ttnn::multiply(
            dL_dgated, gate->get_value(), std::nullopt, std::nullopt, std::nullopt, none, none, none, false);
        auto dL_dgate = ttnn::multiply(
            dL_dgated, swished->get_value(), std::nullopt, std::nullopt, std::nullopt, none, none, none, false);

        // Backward through matmuls to get gradients w.r.t. input tensor:
        // dL_dtensor_from_swished = dL_dswished @ w1^T (but need to account for SiLU gradient)
        // dL_dtensor_from_gate = dL_dgate @ w3^T

        // For SiLU backward, we need the original input to W1
        auto x_w1 = ttnn::matmul(tensor->get_value(), w1->get_value());
        auto sigmoid_x_w1 = ttnn::sigmoid(x_w1);
        auto silu_grad = ttnn::multiply(
            sigmoid_x_w1,
            ttnn::add(
                ttnn::ones_like(sigmoid_x_w1),
                ttnn::multiply(
                    x_w1,
                    ttnn::subtract(
                        ttnn::ones_like(sigmoid_x_w1),
                        sigmoid_x_w1,
                        std::nullopt,
                        std::nullopt,
                        std::nullopt,
                        none,
                        none,
                        none,
                        false),
                    std::nullopt,
                    std::nullopt,
                    std::nullopt,
                    none,
                    none,
                    none,
                    false),
                std::nullopt,
                std::nullopt,
                std::nullopt,
                none,
                none,
                none,
                false),
            std::nullopt,
            std::nullopt,
            std::nullopt,
            none,
            none,
            none,
            false);

        auto dL_dx_w1 =
            ttnn::multiply(dL_dswished, silu_grad, std::nullopt, std::nullopt, std::nullopt, none, none, none, false);
        auto dL_dtensor_from_w1 = ttnn::matmul(dL_dx_w1, ttnn::transpose(w1->get_value(), -2, -1));

        auto dL_dtensor_from_w3 = ttnn::matmul(dL_dgate, ttnn::transpose(w3->get_value(), -2, -1));

        // Combine gradients from both paths
        auto dL_dtensor = ttnn::add(
            dL_dtensor_from_w1, dL_dtensor_from_w3, std::nullopt, std::nullopt, std::nullopt, none, none, none, false);

        tensor->add_grad(dL_dtensor);
    };

    auto links = autograd::get_links(tensor);
    output->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return output;
}

}  // namespace ttml::ops
