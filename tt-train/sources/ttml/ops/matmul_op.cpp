// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_op.hpp"

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"
#include "ttnn_fixed/matmuls.hpp"

namespace ttml::ops {
autograd::TensorPtr matmul_op(
    const autograd::TensorPtr& a, const autograd::TensorPtr& b, bool transpose_a, bool transpose_b) {
    auto out = autograd::create_tensor();
    auto res = ttnn_fixed::matmul(a->get_value(), b->get_value(), transpose_a, transpose_b);
    out->set_value(res);

    autograd::GradFunction grad = [a, b, out, transpose_a, transpose_b]() {
        auto [a_grad, b_grad] =
            ttnn_fixed::matmul_backward(a->get_value(), b->get_value(), out->get_grad(), transpose_a, transpose_b);
        a->add_grad(a_grad);
        b->add_grad(b_grad);
    };

    auto links = autograd::get_links(a, b);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

}  // namespace ttml::ops
