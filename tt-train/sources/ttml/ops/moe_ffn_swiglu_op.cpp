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
#include "metal/operations.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
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

const ttml::metal::VariableMatmulConfig kVarMmConfig{
    .M_block_size = 4,
    .K_block_size = 8,
    .N_block_size = 8,
    .subblock_h = 2,
    .subblock_w = 2,
    .compute_with_storage_grid_size = {10, 10},
};

const ttml::metal::VariableMatmulConfig kVarMmConfigTransposeA = [] {
    auto c = kVarMmConfig;
    c.transpose_a = true;
    return c;
}();

const ttml::metal::VariableMatmulConfig kVarMmConfigTransposeB = [] {
    auto c = kVarMmConfig;
    c.transpose_b = true;
    return c;
}();

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
    if (wg0_shape[-2] != hidden_dim) {
        throw std::runtime_error("moe_ffn_swiglu_fw: w_gate[0] inner dim must equal grouped's hidden_dim.");
    }

    auto offsets_host = offsets.to_vector<uint32_t>();
    if (offsets_host.size() != num_experts + 1U) {
        throw std::runtime_error("moe_ffn_swiglu_fw: offsets size must be num_experts + 1.");
    }
    if (offsets_host.back() != token_capacity) {
        throw std::runtime_error("moe_ffn_swiglu_fw: offsets[-1] must equal token_capacity.");
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

        const auto& w_gate_e = w_gate[e]->get_value();
        const auto& w_up_e = w_up[e]->get_value();
        const auto& w_down_e = w_down[e]->get_value();

        // Offset-read into the grouped tensor — avoids materializing X_e = slice_rows(grouped).
        // Tile-aligned offsets are guaranteed by the dispatch convention (counts rounded to 32).
        const uint32_t offset_tiles = row_lo / 32U;
        const uint32_t len_tiles = (row_hi - row_lo) / 32U;

        auto gate_proj_e =
            ttml::metal::variable_matmul(grouped_value, w_gate_e, kVarMmConfig, std::nullopt, offset_tiles, len_tiles);
        auto up_proj_e =
            ttml::metal::variable_matmul(grouped_value, w_up_e, kVarMmConfig, std::nullopt, offset_tiles, len_tiles);
        auto activated_e = ttnn::multiply(
            gate_proj_e,
            up_proj_e,
            /*output_dtype=*/std::nullopt,
            /*memory_config=*/std::nullopt,
            /*output_tensor=*/std::nullopt,
            /*post_op_activations=*/no_acts,
            /*input_a_activations=*/silu_lhs);  // silu(gate_proj_e) * up_proj_e in one op
        auto down_proj_e = ttml::metal::variable_matmul(activated_e, w_down_e, kVarMmConfig);
        activated_e.deallocate();

        gate_proj_parts.push_back(std::move(gate_proj_e));
        up_proj_parts.push_back(std::move(up_proj_e));
        down_proj_parts.push_back(std::move(down_proj_e));
    }

    if (down_proj_parts.empty()) {
        throw std::runtime_error("moe_ffn_swiglu_fw: all experts empty (token_capacity == 0).");
    }
    auto y = (down_proj_parts.size() == 1U) ? down_proj_parts.front() : ttnn::concat(down_proj_parts, /*dim=*/2);
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
                                   hidden_dim]() mutable {
        auto dY = out->get_grad();
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

            // Down branch:  d_activated_e = dY_e @ w_down_e^T,  dW_down_e = activated_e^T @ dY_e
            auto d_activated_e = ttml::metal::variable_matmul(dY_e, w_down_e, kVarMmConfigTransposeB);
            auto dW_down_e = ttml::metal::variable_matmul(activated_e, dY_e, kVarMmConfigTransposeA);
            w_down[e]->add_grad(dW_down_e);
            activated_e.deallocate();
            dY_e.deallocate();

            auto [d_gate_proj_e, d_up_proj_e] =
                ttml::metal::swiglu_elemwise_bw(gate_proj_e, up_proj_e, d_activated_e, gate_proj_e);
            up_proj_e.deallocate();
            d_activated_e.deallocate();

            // dW_gate_e = X_e^T @ d_gate_proj_e,  dW_up_e = X_e^T @ d_up_proj_e — added directly to per-expert grads.
            auto dW_gate_e = ttml::metal::variable_matmul(X_e, d_gate_proj_e, kVarMmConfigTransposeA);
            w_gate[e]->add_grad(dW_gate_e);
            auto dW_up_e = ttml::metal::variable_matmul(X_e, d_up_proj_e, kVarMmConfigTransposeA);
            w_up[e]->add_grad(dW_up_e);
            X_e.deallocate();

            // dX_e = d_gate_proj_e @ w_gate_e^T  +  d_up_proj_e @ w_up_e^T
            auto dX_via_gate_e = ttml::metal::variable_matmul(d_gate_proj_e, w_gate_e, kVarMmConfigTransposeB);
            auto dX_via_up_e = ttml::metal::variable_matmul(d_up_proj_e, w_up_e, kVarMmConfigTransposeB);
            d_gate_proj_e.deallocate();
            d_up_proj_e.deallocate();

            auto dX_e = ttnn::add(dX_via_gate_e, dX_via_up_e);
            dX_via_gate_e.deallocate();
            dX_via_up_e.deallocate();
            dX_parts.push_back(std::move(dX_e));
        }

        gate_proj_parts.clear();
        up_proj_parts.clear();

        auto dX = (dX_parts.size() == 1U) ? dX_parts.front() : ttnn::concat(dX_parts, /*dim=*/2);
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
