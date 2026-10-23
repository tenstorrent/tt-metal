// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_op.hpp"

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"
#include "ttnn_fixed/matmuls.hpp"

namespace ttml::ops {

namespace {

uint64_t batch_volume(const ttnn::Tensor& t) {
    const auto& shape = t.logical_shape();
    const uint64_t matrix_volume = shape.rank() > 2U ? static_cast<uint64_t>(shape[-2]) * shape[-1] : 0U;
    return matrix_volume != 0U ? t.logical_volume() / matrix_volume : 1U;
}

}  // namespace

autograd::TensorPtr matmul_op(
    const autograd::TensorPtr& a, const autograd::TensorPtr& b, bool transpose_a, bool transpose_b) {
    if (autograd::any_requires_grad(a, b)) {
        // With transpose_a, a's stored rows are op(a)'s columns, so flattening a batched a misaligns its rows with
        // dL/dout's: dA would come back scrambled and dB would fail on a shape mismatch.
        if (transpose_a) {
            const uint64_t a_batch = batch_volume(a->get_value());
            TT_FATAL(
                a_batch == 1U,
                "matmul_op backward does not support transpose_a with a batched a: got a of shape {} (batch {}).",
                a->get_value().logical_shape(),
                a_batch);
        }
        // Backward reduces dB over all of a's flattened rows, so b must be shared across the batch.
        const uint64_t b_batch = batch_volume(b->get_value());
        TT_FATAL(
            b_batch == 1U,
            "matmul_op backward does not support a batched b: got b of shape {} (batch {}).",
            b->get_value().logical_shape(),
            b_batch);
    }

    auto out = autograd::create_tensor();
    auto res = ttnn_fixed::matmul(a->get_value(), b->get_value(), transpose_a, transpose_b);
    out->set_value(res);

    // Each grad's matmul runs only for an input that requires grad; a frozen input's grad is never computed.
    autograd::GradFunction grad = [a, b, out, transpose_a, transpose_b]() {
        const auto& a_value = a->get_value();
        const auto& b_value = b->get_value();
        const auto& out_grad = out->get_grad();

        // Flatten dL/dout's leading dims into rows, so dB reduces over the whole batch in one matmul.
        const uint32_t grad_cols = out_grad.logical_shape()[-1];
        const auto flat_grad = ttnn::reshape(
            out_grad, ttnn::Shape({static_cast<uint32_t>(out_grad.logical_volume() / grad_cols), grad_cols}));

        if (a->get_requires_grad()) {
            // dA_eff = dL/dout @ op(b)^T. With transpose_a, a holds A^T, so dA = dA_eff^T = op(b) @ (dL/dout)^T.
            const auto flat_a_grad = transpose_a ? ttnn_fixed::matmul(b_value, flat_grad, transpose_b, true)
                                                 : ttnn_fixed::matmul(flat_grad, b_value, false, !transpose_b);
            a->add_grad(ttnn::reshape(flat_a_grad, a_value.logical_shape()));
        }

        if (b->get_requires_grad()) {
            const uint32_t a_cols = a_value.logical_shape()[-1];
            const auto flat_a =
                ttnn::reshape(a_value, ttnn::Shape({static_cast<uint32_t>(a_value.logical_volume() / a_cols), a_cols}));
            // dB_eff = op(a)^T @ dL/dout. With transpose_b, b holds B^T, so dB = dB_eff^T = (dL/dout)^T @ op(a).
            const auto flat_b_grad = transpose_b ? ttnn_fixed::matmul(flat_grad, flat_a, true, transpose_a)
                                                 : ttnn_fixed::matmul(flat_a, flat_grad, !transpose_a, false);
            b->add_grad(ttnn::reshape(flat_b_grad, b_value.logical_shape()));
        }
    };

    out->set_node(autograd::add_backward_node(std::move(grad), out, a, b));
    return out;
}

}  // namespace ttml::ops
