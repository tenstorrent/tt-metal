// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sp_linear_ops.hpp"

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"
#include "ttnn_fixed/matmuls.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

namespace ttml::ops::distributed {

namespace {

// [B,1,S,X] -> [B*S, X]: a metadata-only view of a tile-aligned activation, the form in which
// ttnn_linear_backward issues its matmuls (and therefore the form that keeps Composed bit-identical to it).
ttnn::Tensor flatten_rows(const ttnn::Tensor& t) {
    const auto& shape = t.logical_shape();
    const auto rows = static_cast<uint32_t>(t.logical_volume() / shape[-1]);
    return ttnn::reshape(t, ttnn::Shape({rows, shape[-1]}));
}

// grad [rows, N] and the activation it was produced from ([..., K]) -> grad^T @ activation as `weight_shape`.
ttnn::Tensor weight_grad(const ttnn::Tensor& grad2d, const ttnn::Tensor& activation, const ttnn::Shape& weight_shape) {
    auto wgrad = ttnn_fixed::matmul(grad2d, flatten_rows(activation), /* transpose_a */ true, /* transpose_b */ false);
    return ttnn::reshape(wgrad, weight_shape);
}

void validate_linear_operands(const char* op, const ttnn::Tensor& x, const ttnn::Tensor& weight) {
    const auto& x_shape = x.logical_shape();
    const auto& w_shape = weight.logical_shape();
    TT_FATAL(x_shape.rank() == 4, "{}: expected a rank-4 (B, 1, S, K) activation, got {}", op, x_shape);
    TT_FATAL(
        w_shape.rank() == 4 && w_shape[0] == 1 && w_shape[1] == 1,
        "{}: expected a [1, 1, N, K] weight, got {}",
        op,
        w_shape);
    TT_FATAL(
        x_shape[-1] == w_shape[-1],
        "{}: the activation's features ({}) must equal the weight's input features ({})",
        op,
        x_shape[-1],
        w_shape[-1]);
}

}  // namespace

autograd::TensorPtr sp_column_parallel_linear(
    const autograd::TensorPtr& x,
    const autograd::TensorPtr& weight,
    const autograd::TensorPtr& bias,
    uint32_t cluster_axis) {
    validate_linear_operands("sp_column_parallel_linear", x->get_value(), weight->get_value());

    auto gathered_and_mm = ttnn_fixed::distributed::all_gather_matmul(
        x->get_value(),
        weight->get_value(),
        cluster_axis,
        /* transpose_b */ true,
        bias != nullptr ? std::optional<ttnn::Tensor>(bias->get_value()) : std::nullopt);
    const auto& gathered_x = gathered_and_mm.first;
    auto out = autograd::create_tensor(gathered_and_mm.second);

    autograd::GradFunction grad = [x, weight, bias, out, gathered_x, cluster_axis]() {
        if (!out->is_grad_initialized()) {
            return;
        }
        const auto& grad_out = out->get_grad();  // [B,1,S,N/T]
        // The mirror of the forward: multiply, then reduce-scatter onto the sequence shards.
        x->add_grad(ttnn_fixed::distributed::matmul_reduce_scatter(
            grad_out, weight->get_value(), cluster_axis, /* transpose_b */ false));
        auto grad2d = flatten_rows(grad_out);
        weight->add_grad(weight_grad(grad2d, gathered_x, weight->get_value().logical_shape()));
        if (bias != nullptr) {
            bias->add_grad(
                ttnn::reshape(ttnn_fixed::sum_over_dim(grad2d, /* axis */ 0), bias->get_value().logical_shape()));
        }
    };
    out->set_node(autograd::add_backward_node(std::move(grad), out, x, weight, bias));
    return out;
}

autograd::TensorPtr sp_row_parallel_linear(
    const autograd::TensorPtr& x, const autograd::TensorPtr& weight, uint32_t cluster_axis) {
    validate_linear_operands("sp_row_parallel_linear", x->get_value(), weight->get_value());

    auto out = autograd::create_tensor(ttnn_fixed::distributed::matmul_reduce_scatter(
        x->get_value(), weight->get_value(), cluster_axis, /* transpose_b */ true));

    autograd::GradFunction grad = [x, weight, out, cluster_axis]() {
        if (!out->is_grad_initialized()) {
            return;
        }
        // The mirror of the forward: gather the sequence-sharded grad, then multiply. The gathered grad is
        // also the left operand of the weight gradient.
        auto grad_full_and_dgrad = ttnn_fixed::distributed::all_gather_matmul(
            out->get_grad(), weight->get_value(), cluster_axis, /* transpose_b */ false);
        x->add_grad(grad_full_and_dgrad.second);
        weight->add_grad(
            weight_grad(flatten_rows(grad_full_and_dgrad.first), x->get_value(), weight->get_value().logical_shape()));
    };
    out->set_node(autograd::add_backward_node(std::move(grad), out, x, weight));
    return out;
}

}  // namespace ttml::ops::distributed
