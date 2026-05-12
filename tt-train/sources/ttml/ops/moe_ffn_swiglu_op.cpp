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
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"

namespace ttml::ops {

namespace {

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
    const auto& grouped_shape = grouped_value.logical_shape();
    const uint32_t token_capacity = grouped_shape[-2];
    const uint32_t hidden_dim = grouped_shape[-1];
    const uint32_t num_experts = static_cast<uint32_t>(w_gate.size());

    if (num_experts == 0U) {
        throw std::runtime_error("moe_ffn_swiglu_fw: weight lists are empty.");
    }
    if (w_up.size() != num_experts || w_down.size() != num_experts) {
        throw std::runtime_error("moe_ffn_swiglu_fw: w_gate/w_up/w_down must have the same length.");
    }

    const auto& wg0_shape = w_gate[0]->get_value().logical_shape();
    if (wg0_shape[-1] != hidden_dim) {
        throw std::runtime_error("moe_ffn_swiglu_fw: w_gate[0] inner dim must equal grouped's hidden_dim.");
    }

    auto offsets_host = offsets.to_vector<uint32_t>();
    if (offsets_host.size() != num_experts + 1U) {
        throw std::runtime_error("moe_ffn_swiglu_fw: offsets size must be num_experts + 1.");
    }
    if (offsets_host.back() > token_capacity) {
        throw std::runtime_error("moe_ffn_swiglu_fw: offsets[-1] exceeds token_capacity.");
    }

    // Per-expert forward: variable_matmul reads each expert's rows directly from
    // grouped via row-offset, runs gate+up matmuls against the per-expert weights,
    // applies silu·multiply, then down_proj writes directly into the corresponding
    // row range of the pre-allocated output. No per-expert slicing or output concat.
    // gate_proj_e and up_proj_e are saved per-expert for backward; activated_e is
    // recomputed in backward.
    //
    // Fused activation for the silu·multiply step: ttnn::multiply with a unary
    // activation on lhs computes silu(lhs) * rhs in one ttnn op (one read of
    // each input, one write — half the DRAM traffic of a separate silu + mul
    // pair).
    using EltwiseUnary = ttnn::operations::unary::EltwiseUnaryWithParam;
    const EltwiseUnary silu_act{ttnn::operations::unary::UnaryOpType::SILU};
    const ttsl::Span<const EltwiseUnary> no_acts;
    const ttsl::Span<const EltwiseUnary> silu_lhs(&silu_act, 1);

    std::vector<ttnn::Tensor> gate_proj_parts;
    std::vector<ttnn::Tensor> up_proj_parts;
    gate_proj_parts.reserve(num_experts);
    up_proj_parts.reserve(num_experts);

    // Pre-allocate the full output tensor [T_cap, H], zero-initialized so per-expert
    // pad rows and any trailing slack (offsets[-1] < T_cap) stay zero. Each expert's
    // down_proj writes directly into rows [row_lo, row_hi) of y — no concat.
    auto y = ttml::core::zeros(
        ttnn::Shape({1U, 1U, token_capacity, hidden_dim}), &ttml::autograd::ctx().get_device(), grouped_value.dtype());

    bool any_nonempty = false;
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
        any_nonempty = true;

        const auto& w_gate_e = w_gate[e]->get_value();
        const auto& w_up_e = w_up[e]->get_value();
        const auto& w_down_e = w_down[e]->get_value();

        // Offset-read into grouped — avoids materializing a per-expert slice.
        // Tile-aligned offsets are guaranteed by the dispatch convention (counts rounded to 32).
        const uint32_t offset_tiles = row_lo / 32U;
        const uint32_t len_tiles = (row_hi - row_lo) / 32U;

        // w_gate / w_up are stored as [I, H], so matmul uses transpose_b: X_e @ w^T.
        auto gate_proj_e = ttml::metal::variable_matmul(
            grouped_value, w_gate_e, kVarMmConfigTransposeB, std::nullopt, offset_tiles, len_tiles);
        auto up_proj_e = ttml::metal::variable_matmul(
            grouped_value, w_up_e, kVarMmConfigTransposeB, std::nullopt, offset_tiles, len_tiles);
        auto activated_e = ttnn::multiply(
            gate_proj_e,
            up_proj_e,
            /*output_dtype=*/std::nullopt,
            /*memory_config=*/std::nullopt,
            /*output_tensor=*/std::nullopt,
            /*post_op_activations=*/no_acts,
            /*input_a_activations=*/silu_lhs);  // silu(gate_proj_e) * up_proj_e in one op
        // w_down is [H, I]; down_proj = activated_e @ w_down^T. Write directly into
        // y[row_lo:row_hi] — no per-expert allocation, no concat.
        ttml::metal::variable_matmul(
            activated_e,
            w_down_e,
            kVarMmConfigTransposeB,
            std::nullopt,
            /*in0_row_offset_tiles=*/0U,
            /*effective_M_tiles=*/0U,
            /*in0_k_offset_tiles=*/0U,
            /*in1_k_offset_tiles=*/0U,
            /*output_tensor=*/y,
            /*out_row_offset_tiles=*/offset_tiles);
        activated_e.deallocate();

        gate_proj_parts.push_back(std::move(gate_proj_e));
        up_proj_parts.push_back(std::move(up_proj_e));
    }

    if (!any_nonempty) {
        throw std::runtime_error("moe_ffn_swiglu_fw: all experts empty (token_capacity == 0).");
    }

    auto out = autograd::create_tensor(y);

    autograd::GradFunction grad = [grouped,
                                   w_gate,
                                   w_up,
                                   w_down,
                                   out,
                                   offsets_host = std::move(offsets_host),
                                   gate_proj_parts = std::move(gate_proj_parts),
                                   up_proj_parts = std::move(up_proj_parts),
                                   num_experts]() mutable {
        const auto dY = out->get_grad();
        const auto& grouped_value = grouped->get_value();
        const auto& grouped_shape = grouped_value.logical_shape();

        // Pre-allocate full dX_via_gate and dX_via_up tensors [T_cap, H]; per-expert matmuls
        // write directly into row ranges. After the loop, one ttnn::add yields dX (single
        // pass over the full tensor instead of E small adds + concat).
        auto dX_via_gate =
            ttml::core::empty(grouped_shape, &ttml::autograd::ctx().get_device(), grouped_value.memory_config());
        auto dX_via_up =
            ttml::core::empty(grouped_shape, &ttml::autograd::ctx().get_device(), grouped_value.memory_config());

        // Fused silu·multiply spans for the bw recomputation of activated_e (same pattern as fwd).
        using EltwiseUnary = ttnn::operations::unary::EltwiseUnaryWithParam;
        const EltwiseUnary silu_act{ttnn::operations::unary::UnaryOpType::SILU};
        const ttsl::Span<const EltwiseUnary> no_acts;
        const ttsl::Span<const EltwiseUnary> silu_lhs(&silu_act, 1);

        std::size_t nonempty_idx = 0U;
        for (uint32_t e = 0; e < num_experts; ++e) {
            const uint32_t row_lo = offsets_host[e];
            const uint32_t row_hi = offsets_host[e + 1U];

            if (row_hi == row_lo) {
                // empty expert
                continue;
            }

            // X_e and dY_e slices elided: dW_gate/dW_up use K-axis offset on grouped_value;
            // d_activated_e uses M-offset on dY; dW_down_e uses K-axis offset on dY (in1 side).
            const uint32_t row_lo_tiles = row_lo / 32U;
            const uint32_t M_e_tiles = (row_hi - row_lo) / 32U;
            const auto& w_gate_e = w_gate[e]->get_value();
            const auto& w_up_e = w_up[e]->get_value();
            const auto& w_down_e = w_down[e]->get_value();

            // Recompute activated_e from saved gate_proj_e, up_proj_e (fused silu·multiply).
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

            // Down branch (w_down_e is [H, I]):
            //   d_activated_e = dY_e @ w_down_e   (no transpose)
            //   dW_down_e     = dY_e^T @ activated_e
            // dY rows [row_lo, row_hi) are read by offset: M-axis for d_activated_e,
            // K-axis (= stored rows under transpose_a) for dW_down_e.
            auto d_activated_e = ttml::metal::variable_matmul(
                dY,
                w_down_e,
                kVarMmConfig,
                std::nullopt,
                /*in0_row_offset_tiles=*/row_lo_tiles,
                /*effective_M_tiles=*/M_e_tiles);
            auto dW_down_e = ttml::metal::variable_matmul(
                dY,
                activated_e,
                kVarMmConfigTransposeA,
                std::nullopt,
                /*in0_row_offset_tiles=*/0U,
                /*effective_M_tiles=*/0U,
                /*in0_k_offset_tiles=*/row_lo_tiles);
            w_down[e]->add_grad(dW_down_e);
            activated_e.deallocate();

            auto [d_gate_proj_e, d_up_proj_e] =
                ttml::metal::swiglu_elemwise_bw(gate_proj_e, up_proj_e, d_activated_e, gate_proj_e);
            up_proj_e.deallocate();
            d_activated_e.deallocate();

            // w_gate_e, w_up_e are [I, H]:
            //   dW_gate_e = d_gate_proj_e^T @ X_e,  dW_up_e = d_up_proj_e^T @ X_e
            // Read grouped_value's [row_lo, row_hi) rows directly via in1 K-axis offset.
            auto dW_gate_e = ttml::metal::variable_matmul(
                d_gate_proj_e,
                grouped_value,
                kVarMmConfigTransposeA,
                std::nullopt,
                /*in0_row_offset_tiles=*/0U,
                /*effective_M_tiles=*/0U,
                /*in0_k_offset_tiles=*/0U,
                /*in1_k_offset_tiles=*/row_lo_tiles);
            w_gate[e]->add_grad(dW_gate_e);
            auto dW_up_e = ttml::metal::variable_matmul(
                d_up_proj_e,
                grouped_value,
                kVarMmConfigTransposeA,
                std::nullopt,
                /*in0_row_offset_tiles=*/0U,
                /*effective_M_tiles=*/0U,
                /*in0_k_offset_tiles=*/0U,
                /*in1_k_offset_tiles=*/row_lo_tiles);
            w_up[e]->add_grad(dW_up_e);

            // dX = d_gate_proj_e @ w_gate_e  +  d_up_proj_e @ w_up_e   (w_gate/up are [I, H])
            // Write each per-expert matmul into row range of the full-shape dX_via_* tensors.
            ttml::metal::variable_matmul(
                d_gate_proj_e,
                w_gate_e,
                kVarMmConfig,
                std::nullopt,
                /*in0_row_offset_tiles=*/0U,
                /*effective_M_tiles=*/0U,
                /*in0_k_offset_tiles=*/0U,
                /*in1_k_offset_tiles=*/0U,
                /*output_tensor=*/dX_via_gate,
                /*out_row_offset_tiles=*/row_lo_tiles);
            ttml::metal::variable_matmul(
                d_up_proj_e,
                w_up_e,
                kVarMmConfig,
                std::nullopt,
                /*in0_row_offset_tiles=*/0U,
                /*effective_M_tiles=*/0U,
                /*in0_k_offset_tiles=*/0U,
                /*in1_k_offset_tiles=*/0U,
                /*output_tensor=*/dX_via_up,
                /*out_row_offset_tiles=*/row_lo_tiles);
            d_gate_proj_e.deallocate();
            d_up_proj_e.deallocate();
        }

        gate_proj_parts.clear();
        up_proj_parts.clear();

        auto dX = ttnn::add(dX_via_gate, dX_via_up);
        dX_via_gate.deallocate();
        dX_via_up.deallocate();
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
