// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <core/ttnn_all_includes.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"
#include "autograd/tensor.hpp"
#include "metal/operations.hpp"
#include "ops/binary_ops.hpp"
#include "ops/unary_ops.hpp"
#include "ttnn_fixed/matmuls.hpp"

namespace ttml::ops {

autograd::TensorPtr swiglu(
    const autograd::TensorPtr& tensor,
    const autograd::TensorPtr& w1,
    const autograd::TensorPtr& w2,
    const autograd::TensorPtr& w3) {
    auto a_shape = tensor->get_value().logical_shape();
    if (a_shape.rank() != 4) {
        throw std::runtime_error("swiglu only supports rank-4 input tensors.");
    }

    ttnn::Tensor swiglu_fw_result =
        ttml::metal::swiglu_fw(tensor->get_value(), w1->get_value(), w2->get_value(), w3->get_value());
    auto out = autograd::create_tensor(swiglu_fw_result);

    autograd::GradFunction grad = [tensor, w1, w2, w3, out]() {
        auto dL_dout = out->get_grad();
        // Recompute forward intermediates for backward pass
        auto linear1 = ttnn_fixed::matmul(tensor->get_value(), w1->get_value());  // x @ w1
        auto gate = ttnn_fixed::matmul(tensor->get_value(), w3->get_value());     // x @ w3
        auto sigmoid_linear1 = ttnn::sigmoid(linear1);                            // sigmoid(x @ w1)
        auto swished = ttnn::multiply(linear1, sigmoid_linear1);                  // silu(x @ w1)
        auto gated = ttnn::multiply(swished, gate);                               // silu(x @ w1) * (x @ w3)
        auto projected = ttnn_fixed::matmul(gated, w2->get_value());              // gated @ w2

        // Backward through final matmul: dL_dgated = dL_dout @ w2^T
        auto dL_dgated = ttnn_fixed::matmul(dL_dout, w2->get_value(), false, true);

        // Backward through element-wise multiply:
        // dL_dswished = dL_dgated * gate
        // dL_dgate = dL_dgated * swished
        auto dL_dswished = ttnn::multiply(dL_dgated, gate);
        auto dL_dgate = ttnn::multiply(dL_dgated, swished);

        // For SiLU backward: SiLU'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        auto silu_grad = ttnn::multiply(
            sigmoid_linear1,
            ttnn::add(
                ttnn::ones_like(sigmoid_linear1),
                ttnn::multiply(linear1, ttnn::subtract(ttnn::ones_like(sigmoid_linear1), sigmoid_linear1))));

        auto dL_dlinear1 = ttnn::multiply(dL_dswished, silu_grad);
        auto dL_dtensor_from_w1 = ttnn_fixed::matmul(dL_dlinear1, w1->get_value(), false, true);

        auto dL_dtensor_from_w3 = ttnn_fixed::matmul(dL_dgate, w3->get_value(), false, true);

        // Combine gradients from both paths
        auto dL_dtensor = ttnn::add(dL_dtensor_from_w1, dL_dtensor_from_w3);
        tensor->add_grad(dL_dtensor);

        // W2 grad: g^T @ dL_dout
        auto dL_dW2 = ttnn_fixed::matmul(gated, dL_dout, true, false);
        w2->add_grad(ttnn::sum(dL_dW2, 0, true));

        // W1 grad: x^T @ dL_dlinear1
        auto dL_dW1 = ttnn_fixed::matmul(tensor->get_value(), dL_dlinear1, true, false);
        w1->add_grad(ttnn::sum(dL_dW1, 0, true));

        // W3 grad: x^T @ dL_dgate
        auto dL_dW3 = ttnn_fixed::matmul(tensor->get_value(), dL_dgate, true, false);
        w3->add_grad(ttnn::sum(dL_dW3, 0, true));
    };

    auto links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

}  // namespace ttml::ops
