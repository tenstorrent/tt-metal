// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_ffn_swiglu_op.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/full/full.hpp"
#include "ttnn_fixed/matmuls.hpp"

namespace ttml::ops {

namespace {

// Slice rows [row_lo, row_hi) of a [1, 1, T_cap, inner_dim] tensor.
ttnn::Tensor slice_rows(const ttnn::Tensor& tensor, uint32_t row_lo, uint32_t row_hi, uint32_t inner_dim) {
    static const ttsl::SmallVector<uint32_t> step = {1U, 1U, 1U, 1U};
    const ttsl::SmallVector<uint32_t> start = {0U, 0U, row_lo, 0U};
    const ttsl::SmallVector<uint32_t> end = {1U, 1U, row_hi, inner_dim};
    return ttnn::slice(tensor, start, end, step);
}

}  // namespace

autograd::TensorPtr moe_ffn_swiglu_fw(
    const autograd::TensorPtr& grouped,
    const ttnn::Tensor& offsets,
    const std::vector<autograd::TensorPtr>& w_gate,
    const std::vector<autograd::TensorPtr>& w_up,
    const std::vector<autograd::TensorPtr>& w_down) {
    const auto& grouped_value = grouped->get_value();
    const auto grouped_shape = grouped_value.logical_shape();
    const uint32_t token_capacity = grouped_shape[-2];
    const uint32_t hidden_dim = grouped_shape[-1];
    const uint32_t num_experts = static_cast<uint32_t>(w_gate.size());

    if (num_experts == 0U) {
        throw std::runtime_error("moe_ffn_swiglu_fw: weight lists are empty.");
    }
    if (w_up.size() != num_experts || w_down.size() != num_experts) {
        throw std::runtime_error("moe_ffn_swiglu_fw: w_gate/w_up/w_down must have the same length.");
    }

    const auto wg0_shape = w_gate[0]->get_value().logical_shape();
    if (wg0_shape[-1] != hidden_dim) {
        throw std::runtime_error("moe_ffn_swiglu_fw: w_gate[0] inner dim must equal grouped's hidden_dim.");
    }

    auto offsets_host = offsets.to_vector<uint32_t>();
    if (offsets_host.size() != num_experts + 1U) {
        throw std::runtime_error("moe_ffn_swiglu_fw: offsets size must be num_experts + 1.");
    }
    // Per-expert forward: slice grouped once per expert, run gate+up matmuls
    // directly against the per-expert weight tensors (no slicing of weights),
    // silu·multiply on the chunk, then down matmul. One concat at the end.
    // gate_proj_e and up_proj_e are saved per-expert for backward; activated_e
    // is recomputed in backward.
    //
    // Fused activation for the silu·multiply step: ttnn::multiply with a unary
    // activation on lhs computes silu(lhs) * rhs in one ttnn op (one read of
    // each input, one write — half the DRAM traffic of a separate silu + mul
    // pair).
    using EltwiseUnary = ttnn::operations::unary::EltwiseUnaryWithParam;
    const EltwiseUnary silu_act{ttnn::operations::unary::UnaryOpType::SILU};
    const ttsl::Span<const EltwiseUnary> no_acts;
    const ttsl::Span<const EltwiseUnary> silu_lhs(&silu_act, 1);

    std::vector<ttnn::Tensor> down_proj_parts;
    std::vector<ttnn::Tensor> gate_proj_parts;
    std::vector<ttnn::Tensor> up_proj_parts;
    down_proj_parts.reserve(num_experts);
    gate_proj_parts.reserve(num_experts);
    up_proj_parts.reserve(num_experts);

    for (uint32_t e = 0; e < num_experts; ++e) {
        const uint32_t row_lo = offsets_host[e];
        const uint32_t row_hi = offsets_host[e + 1U];
        if (row_hi < row_lo) {
            throw std::runtime_error("moe_ffn_swiglu_fw: offsets are not monotonic.");
        }
        if (row_hi == row_lo) {
            // empty expert
            continue;
        }

        auto X_e = slice_rows(grouped_value, row_lo, row_hi, hidden_dim);
        const auto& w_gate_e = w_gate[e]->get_value();
        const auto& w_up_e = w_up[e]->get_value();
        const auto& w_down_e = w_down[e]->get_value();

        auto gate_proj_e = ttnn_fixed::matmul(X_e, w_gate_e, false, true);  // [1,1,len,intermediate_dim]
        auto up_proj_e = ttnn_fixed::matmul(X_e, w_up_e, false, true);      // [1,1,len,intermediate_dim]
        auto activated_e = ttnn::multiply(
            gate_proj_e,
            up_proj_e,
            /*output_dtype=*/std::nullopt,
            /*memory_config=*/std::nullopt,
            /*output_tensor=*/std::nullopt,
            /*post_op_activations=*/no_acts,
            /*input_a_activations=*/silu_lhs);  // silu(gate_proj_e) * up_proj_e in one op
        auto down_proj_e = ttnn_fixed::matmul(activated_e, w_down_e, false, true);  // [1,1,len,hidden_dim]
        activated_e.deallocate();

        gate_proj_parts.push_back(std::move(gate_proj_e));
        up_proj_parts.push_back(std::move(up_proj_e));
        down_proj_parts.push_back(std::move(down_proj_e));
    }

    // moe_group allocates `grouped` for a worst-case T_cap; the actual used range
    // is [0, offsets[-1]). Append a zero tensor for the trailing slack so the
    // output shape matches what moe_ungroup expects ([1,1,T_cap,H]).
    const uint32_t produced_rows = offsets_host.back();
    if (token_capacity > produced_rows) {
        const uint32_t pad_rows = token_capacity - produced_rows;
        down_proj_parts.push_back(ttnn::moreh_full(
            ttsl::SmallVector<uint32_t>{1U, 1U, pad_rows, hidden_dim},
            0.0F,
            &autograd::ctx().get_device(),
            grouped_value.dtype(),
            grouped_value.layout(),
            grouped_value.memory_config()));
    }

    const auto y = (down_proj_parts.size() == 1U) ? down_proj_parts.front() : ttnn::concat(down_proj_parts, /*dim=*/2);
    down_proj_parts.clear();

    auto out = autograd::create_tensor(y);

    autograd::GradFunction grad = [grouped,
                                   w_gate,
                                   w_up,
                                   w_down,
                                   out,
                                   offsets_host = std::move(offsets_host),
                                   gate_proj_parts = std::move(gate_proj_parts),
                                   up_proj_parts = std::move(up_proj_parts),
                                   num_experts,
                                   hidden_dim,
                                   token_capacity]() mutable {
        const auto dY = out->get_grad();
        const auto& grouped_value = grouped->get_value();

        std::vector<ttnn::Tensor> dX_parts;
        dX_parts.reserve(num_experts);

        std::size_t nonempty_idx = 0U;
        for (uint32_t e = 0; e < num_experts; ++e) {
            const uint32_t row_lo = offsets_host[e];
            const uint32_t row_hi = offsets_host[e + 1U];

            if (row_hi == row_lo) {
                // empty expert
                continue;
            }

            auto X_e = slice_rows(grouped_value, row_lo, row_hi, hidden_dim);
            auto dY_e = slice_rows(dY, row_lo, row_hi, hidden_dim);
            const auto& w_gate_e = w_gate[e]->get_value();
            const auto& w_up_e = w_up[e]->get_value();
            const auto& w_down_e = w_down[e]->get_value();

            // Recompute activated_e from saved gate_proj_e, up_proj_e using
            // the same fused silu·multiply pattern as the forward.
            using EltwiseUnary = ttnn::operations::unary::EltwiseUnaryWithParam;
            const EltwiseUnary silu_act{ttnn::operations::unary::UnaryOpType::SILU};
            const ttsl::Span<const EltwiseUnary> no_acts;
            const ttsl::Span<const EltwiseUnary> silu_lhs(&silu_act, 1);

            auto& gate_proj_e = gate_proj_parts[nonempty_idx];
            auto& up_proj_e = up_proj_parts[nonempty_idx];
            ++nonempty_idx;
            auto activated_e = ttnn::multiply(
                gate_proj_e,
                up_proj_e,
                /*output_dtype=*/std::nullopt,
                /*memory_config=*/std::nullopt,
                /*output_tensor=*/std::nullopt,
                /*post_op_activations=*/no_acts,
                /*input_a_activations=*/silu_lhs);

            // Down branch (w_down_e is [hidden, intermediate]):
            //   d_activated_e = dY_e @ w_down_e,  dW_down_e = dY_e^T @ activated_e
            auto d_activated_e = ttnn_fixed::matmul(dY_e, w_down_e, /*transpose_a=*/false, /*transpose_b=*/false);
            const auto dW_down_e = ttnn_fixed::matmul(dY_e, activated_e, /*transpose_a=*/true, /*transpose_b=*/false);
            w_down[e]->add_grad(dW_down_e);
            activated_e.deallocate();
            dY_e.deallocate();

            auto [d_gate_proj_e, d_up_proj_e] =
                ttml::metal::swiglu_elemwise_bw(gate_proj_e, up_proj_e, d_activated_e, gate_proj_e);
            up_proj_e.deallocate();
            d_activated_e.deallocate();

            // w_gate_e, w_up_e are [intermediate, hidden]:
            //   dW_gate_e = d_gate_proj_e^T @ X_e,  dW_up_e = d_up_proj_e^T @ X_e
            const auto dW_gate_e = ttnn_fixed::matmul(d_gate_proj_e, X_e, /*transpose_a=*/true, /*transpose_b=*/false);
            w_gate[e]->add_grad(dW_gate_e);
            const auto dW_up_e = ttnn_fixed::matmul(d_up_proj_e, X_e, /*transpose_a=*/true, /*transpose_b=*/false);
            w_up[e]->add_grad(dW_up_e);
            X_e.deallocate();

            // dX_e = d_gate_proj_e @ w_gate_e  +  d_up_proj_e @ w_up_e
            auto dX_via_gate_e =
                ttnn_fixed::matmul(d_gate_proj_e, w_gate_e, /*transpose_a=*/false, /*transpose_b=*/false);
            auto dX_via_up_e = ttnn_fixed::matmul(d_up_proj_e, w_up_e, /*transpose_a=*/false, /*transpose_b=*/false);
            d_gate_proj_e.deallocate();
            d_up_proj_e.deallocate();

            auto dX_e = ttnn::add(dX_via_gate_e, dX_via_up_e);
            dX_via_gate_e.deallocate();
            dX_via_up_e.deallocate();
            dX_parts.push_back(std::move(dX_e));
        }

        gate_proj_parts.clear();
        up_proj_parts.clear();

        // Pad the trailing slack (rows the op never read) with zeros so dX
        // matches grouped's shape [1, 1, T_cap, H] for add_grad.
        const uint32_t produced_rows = offsets_host.back();
        if (token_capacity > produced_rows) {
            const uint32_t pad_rows = token_capacity - produced_rows;
            dX_parts.push_back(ttnn::moreh_full(
                ttsl::SmallVector<uint32_t>{1U, 1U, pad_rows, hidden_dim},
                0.0F,
                &autograd::ctx().get_device(),
                grouped_value.dtype(),
                grouped_value.layout(),
                grouped_value.memory_config()));
        }

        const auto dX = (dX_parts.size() == 1U) ? dX_parts.front() : ttnn::concat(dX_parts, /*dim=*/2);
        grouped->add_grad(dX);
    };

    // Pack the runtime-sized input set (grouped + 3 weight lists) into one
    // vector and register the backward node via the runtime-arity overload.
    std::vector<autograd::TensorPtr> inputs;
    inputs.reserve(1U + 3U * num_experts);
    inputs.push_back(grouped);
    inputs.insert(inputs.end(), w_gate.begin(), w_gate.end());
    inputs.insert(inputs.end(), w_up.begin(), w_up.end());
    inputs.insert(inputs.end(), w_down.begin(), w_down.end());
    out->set_node(autograd::add_backward_node(std::move(grad), out, inputs));

    return out;
}

}  // namespace ttml::ops
