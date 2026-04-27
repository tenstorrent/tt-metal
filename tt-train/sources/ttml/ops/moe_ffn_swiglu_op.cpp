// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_ffn_swiglu_op.hpp"

#include <utility>

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"
#include "metal/operations.hpp"
#include "ops/sparse_matmul_op.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"

namespace ttml::ops {

autograd::TensorPtr moe_ffn_swiglu_fw(
    const autograd::TensorPtr& grouped,
    const ttnn::Tensor& offsets,
    const autograd::TensorPtr& w_gate,
    const autograd::TensorPtr& w_up,
    const autograd::TensorPtr& w_down) {
    // Forward: linear1 = X @ W_gate, gate = X @ W_up, gated = silu(linear1) * gate,
    // Y = gated @ W_down
    auto linear1 = sparse_matmul_forward(grouped->get_value(), w_gate->get_value(), offsets);
    auto gate = sparse_matmul_forward(grouped->get_value(), w_up->get_value(), offsets);
    auto gated = ttnn::multiply(ttnn::silu(linear1), gate);
    auto y = sparse_matmul_forward(gated, w_down->get_value(), offsets);
    // `gated` is recomputed in backward from saved `linear1` and `gate`
    gated.deallocate();

    auto out = autograd::create_tensor(y);

    autograd::GradFunction grad =
        [grouped, w_gate, w_up, w_down, offsets, out, linear1 = std::move(linear1), gate = std::move(gate)]() mutable {
            auto dY = out->get_grad();

            // Recompute gated = silu(linear1) * gate
            auto gated = ttnn::multiply(ttnn::silu(linear1), gate);
            auto [dgated, dW_down] = sparse_matmul_backward(gated, w_down->get_value(), dY, offsets);
            w_down->add_grad(dW_down);
            gated.deallocate();

            // SwiGLU eltwise backward: gated = silu(linear1) * gate
            // use fused kernel; reuse `linear1` storage as the dL/dlinear1 output.
            auto [d_linear1, d_gate] = ttml::metal::swiglu_elemwise_bw(linear1, gate, dgated, linear1);
            gate.deallocate();
            dgated.deallocate();

            // dW_gate, dX from the SiLU branch.
            auto [dX_via_gate, dW_gate] =
                sparse_matmul_backward(grouped->get_value(), w_gate->get_value(), d_linear1, offsets);
            w_gate->add_grad(dW_gate);
            d_linear1.deallocate();

            // dW_up, dX from the multiplier branch.
            auto [dX_via_up, dW_up] = sparse_matmul_backward(grouped->get_value(), w_up->get_value(), d_gate, offsets);
            w_up->add_grad(dW_up);
            d_gate.deallocate();

            // dX is the sum of the two branches' contributions.
            auto dX = ttnn::add(dX_via_gate, dX_via_up);
            dX_via_gate.deallocate();
            dX_via_up.deallocate();
            grouped->add_grad(dX);
        };

    out->set_node(autograd::add_backward_node(std::move(grad), out, grouped, w_gate, w_up, w_down));
    return out;
}

}  // namespace ttml::ops
