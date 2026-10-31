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
    auto out = autograd::create_tensor();
    auto res = ttnn_fixed::matmul(a->get_value(), b->get_value(), transpose_a, transpose_b);
    out->set_value(res);

    // Each grad's matmul runs only for an input that requires grad; a frozen input's grad is never computed.
    autograd::GradFunction grad = [a, b, out, transpose_a, transpose_b]() {
        // The shape limits below are backward's, so they are checked here: forward alone (e.g. inference) takes any
        // shape ttnn's matmul does.
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

        auto [a_grad, b_grad] = ttnn_fixed::matmul_backward(
            a->get_value(),
            b->get_value(),
            out->get_grad(),
            transpose_a,
            transpose_b,
            {.a = a->get_requires_grad(), .b = b->get_requires_grad()});
        if (a_grad) {
            a->add_grad(*a_grad);
        }
        if (b_grad) {
            b->add_grad(*b_grad);
        }
    };

    out->set_node(autograd::add_backward_node(std::move(grad), out, a, b));
    return out;
}

}  // namespace ttml::ops
