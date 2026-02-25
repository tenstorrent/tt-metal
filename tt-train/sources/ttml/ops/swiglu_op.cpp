// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "swiglu_op.hpp"

#include <ttnn/operations/eltwise/binary/binary.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"
#include "autograd/tensor.hpp"
#include "core/compute_kernel_config.hpp"
#include "metal/operations.hpp"
#include "ops/binary_ops.hpp"
#include "ops/unary_ops.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn_fixed/matmuls.hpp"

namespace ttml::ops {

namespace {
// Zero-copy flatten of all dims except the last into a single leading dim: [B,N,S,D] -> [B*N*S, D]
ttnn::Tensor flatten_leading(const ttnn::Tensor& t) {
    auto vol = t.logical_volume() / static_cast<uint64_t>(t.logical_shape()[-1]);
    return t.reshape(ttnn::Shape({static_cast<uint32_t>(vol), t.logical_shape()[-1]}));
}

}  // namespace

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

    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor, w1, w2, w3));

    return out;
}

autograd::TensorPtr swiglu_optimized(
    const autograd::TensorPtr& tensor,
    const autograd::TensorPtr& w1,
    const autograd::TensorPtr& w2,
    const autograd::TensorPtr& w3) {
    auto a_shape = tensor->get_value().logical_shape();
    if (a_shape.rank() != 4) {
        throw std::runtime_error("swiglu only supports rank-4 input tensors.");
    }

    // Composite forward: 3 optimized ttnn matmuls + 2 in-place eltwise
    auto linear1 = ttnn_fixed::matmul(tensor->get_value(), w1->get_value());
    auto gate = ttnn_fixed::matmul(tensor->get_value(), w3->get_value());
    ttnn::silu(linear1, std::nullopt, linear1);
    ttnn::multiply_(linear1, gate);
    auto out = autograd::create_tensor(ttnn_fixed::matmul(linear1, w2->get_value()));

    autograd::GradFunction grad = [tensor, w1, w2, w3, out]() {
        auto dL_dout = out->get_grad();

        // Recompute forward intermediates
        auto linear1 = ttnn_fixed::matmul(tensor->get_value(), w1->get_value());
        auto gate = ttnn_fixed::matmul(tensor->get_value(), w3->get_value());
        auto swished = ttnn::silu(linear1);

        // W2 grad first — free gated early
        {
            auto gated = ttnn::multiply(swished, gate);
            auto dL_dW2 = ttnn_fixed::matmul(flatten_leading(gated), flatten_leading(dL_dout), true, false);
            w2->add_grad(dL_dW2.reshape(w2->get_value().logical_shape()));
        }

        auto dL_dprod = ttnn_fixed::matmul(dL_dout, w2->get_value(), false, true);
        dL_dout.deallocate();

        ttnn::multiply_(gate, dL_dprod);
        auto& dL_dswished = gate;

        ttnn::multiply_(swished, dL_dprod);
        auto& dL_dgate = swished;
        dL_dprod.deallocate();

        // Fused SiLU backward — reuse linear1's buffer for the output
        auto dL_dlinear1 = ttml::metal::silu_bw(linear1, dL_dswished, linear1);
        dL_dswished.deallocate();

        // Input grads: two matmuls + in-place add
        auto dL_dtensor = ttnn_fixed::matmul(dL_dlinear1, w1->get_value(), false, true);
        auto dL_dtensor_from_w3 = ttnn_fixed::matmul(dL_dgate, w3->get_value(), false, true);
        ttnn::add_(dL_dtensor, dL_dtensor_from_w3);
        dL_dtensor_from_w3.deallocate();
        tensor->add_grad(dL_dtensor);
        dL_dtensor.deallocate();

        // W1 & W3 grads via flatten — sum absorbed into matmul
        auto flat_x = flatten_leading(tensor->get_value());

        {
            auto dL_dW1 = ttnn_fixed::matmul(flat_x, flatten_leading(dL_dlinear1), true, false);
            w1->add_grad(dL_dW1.reshape(w1->get_value().logical_shape()));
        }
        dL_dlinear1.deallocate();

        {
            auto dL_dW3 = ttnn_fixed::matmul(flat_x, flatten_leading(dL_dgate), true, false);
            w3->add_grad(dL_dW3.reshape(w3->get_value().logical_shape()));
        }
        dL_dgate.deallocate();
    };

    auto links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

}  // namespace ttml::ops
