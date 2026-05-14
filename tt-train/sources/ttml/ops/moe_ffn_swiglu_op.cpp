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

    // EP-friendly: kernel reads per-expert offsets on-device, no offsets.to_vector(). All
    // host-side branching on per-expert sizes (empty-skip, trailing-slack-only zero-fill) is
    // gone — every per-expert matmul runs unconditionally and `y` is pre-zeroed.
    TT_FATAL(offsets.dtype() == ttnn::DataType::UINT32, "moe_ffn_swiglu_fw: offsets must be UINT32.");
    TT_FATAL(offsets.layout() == ttnn::Layout::ROW_MAJOR, "moe_ffn_swiglu_fw: offsets must be ROW_MAJOR.");
    TT_FATAL(
        offsets.logical_shape()[-1] == num_experts + 1U, "moe_ffn_swiglu_fw: offsets size must be num_experts + 1.");

    // Upper bound on per-expert M tiles. Heuristic: 2× the average, clamped to T_cap_tiles.
    // The CALLER is responsible for ensuring max(per-expert token count) ≤ upper_M_tiles * 32
    // — otherwise the InputRow-overridden matmul reads will truncate. For pathologically
    // skewed loads this ceiling should be raised (eventually plumbed as a parameter).
    const uint32_t t_cap_tiles = std::max(1U, token_capacity / 32U);
    const uint32_t avg_tiles = (token_capacity + 32U * num_experts - 1U) / (32U * num_experts);
    const uint32_t upper_M_tiles = std::min(t_cap_tiles, std::max(1U, 2U * avg_tiles));

    using EltwiseUnary = ttnn::operations::unary::EltwiseUnaryWithParam;
    const EltwiseUnary silu_act{ttnn::operations::unary::UnaryOpType::SILU};
    const ttsl::Span<const EltwiseUnary> no_acts;
    const ttsl::Span<const EltwiseUnary> silu_lhs(&silu_act, 1);

    std::vector<ttnn::Tensor> gate_proj_parts;
    std::vector<ttnn::Tensor> up_proj_parts;
    gate_proj_parts.reserve(num_experts);
    up_proj_parts.reserve(num_experts);

    // Pre-zero y so per-expert pad rows + trailing slack [offsets[E], T_cap) stay zero —
    // each down_proj writes only its expert's row range; the rest is untouched.
    auto y = ttml::core::zeros(
        ttnn::Shape({1U, 1U, token_capacity, hidden_dim}), &ttml::autograd::ctx().get_device(), grouped_value.dtype());

    const uint32_t intermediate_dim = wg0_shape[-2];
    for (uint32_t e = 0; e < num_experts; ++e) {
        const auto& w_gate_e = w_gate[e]->get_value();
        const auto& w_up_e = w_up[e]->get_value();
        const auto& w_down_e = w_down[e]->get_value();

        // gate_proj_e / up_proj_e: pre-zeroed [upper*32, I] tensors passed as output_tensor.
        // variable_matmul + InputRow reads grouped[offsets[e]:offsets[e+1]] and writes only
        // rows [0:actual_eff_M]. Pad rows [actual:upper] stay zero — so activated_e is zero in
        // pad rows (silu(0)*0=0), the down_proj writes zeros into y's pad/slack rows, and the
        // bwd dW matmuls (K-range from offsets) only read valid [0:actual] anyway.
        auto gate_proj_e = ttml::core::zeros(
            ttnn::Shape({1U, 1U, upper_M_tiles * 32U, intermediate_dim}),
            &ttml::autograd::ctx().get_device(),
            grouped_value.dtype());
        auto up_proj_e = ttml::core::zeros(
            ttnn::Shape({1U, 1U, upper_M_tiles * 32U, intermediate_dim}),
            &ttml::autograd::ctx().get_device(),
            grouped_value.dtype());
        ttml::metal::variable_matmul(
            grouped_value,
            w_gate_e,
            kVarMmConfigTransposeB,
            std::nullopt,
            /*in0_row_offset_tiles=*/0U,
            /*effective_M_tiles=*/upper_M_tiles,
            /*in0_k_offset_tiles=*/0U,
            /*in1_k_offset_tiles=*/0U,
            /*output_tensor=*/gate_proj_e,
            /*out_row_offset_tiles=*/0U,
            /*offsets_tensor=*/offsets,
            /*offsets_role=*/ttml::metal::OffsetsRole::InputRow,
            /*offsets_start_index=*/e);
        ttml::metal::variable_matmul(
            grouped_value,
            w_up_e,
            kVarMmConfigTransposeB,
            std::nullopt,
            0U,
            upper_M_tiles,
            0U,
            0U,
            up_proj_e,
            0U,
            offsets,
            ttml::metal::OffsetsRole::InputRow,
            e);
        auto activated_e = ttnn::multiply(
            gate_proj_e,
            up_proj_e,
            /*output_dtype=*/std::nullopt,
            /*memory_config=*/std::nullopt,
            /*output_tensor=*/std::nullopt,
            /*post_op_activations=*/no_acts,
            /*input_a_activations=*/silu_lhs);
        // down_proj writes activated_e @ w_down^T into y[offsets[e]:offsets[e+1]] — OutputRow
        // overrides out_row_offset_tiles from on-device offsets.
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
            /*out_row_offset_tiles=*/0U,
            offsets,
            ttml::metal::OffsetsRole::OutputRow,
            e);
        activated_e.deallocate();

        gate_proj_parts.push_back(std::move(gate_proj_e));
        up_proj_parts.push_back(std::move(up_proj_e));
    }

    auto out = autograd::create_tensor(y);

    autograd::GradFunction grad = [grouped,
                                   offsets,
                                   w_gate,
                                   w_up,
                                   w_down,
                                   out,
                                   gate_proj_parts = std::move(gate_proj_parts),
                                   up_proj_parts = std::move(up_proj_parts),
                                   num_experts,
                                   upper_M_tiles]() mutable {
        const auto dY = out->get_grad();
        const auto& grouped_value = grouped->get_value();
        const auto& grouped_shape = grouped_value.logical_shape();

        // Pre-zero dX_via_* so per-expert pad rows + trailing slack stay zero in dX.
        auto dX_via_gate = ttml::core::zeros(grouped_shape, &ttml::autograd::ctx().get_device(), grouped_value.dtype());
        auto dX_via_up = ttml::core::zeros(grouped_shape, &ttml::autograd::ctx().get_device(), grouped_value.dtype());

        using EltwiseUnary = ttnn::operations::unary::EltwiseUnaryWithParam;
        const EltwiseUnary silu_act{ttnn::operations::unary::UnaryOpType::SILU};
        const ttsl::Span<const EltwiseUnary> no_acts;
        const ttsl::Span<const EltwiseUnary> silu_lhs(&silu_act, 1);

        for (uint32_t e = 0; e < num_experts; ++e) {
            auto& gate_proj_e = gate_proj_parts[e];
            auto& up_proj_e = up_proj_parts[e];
            const auto& w_gate_e = w_gate[e]->get_value();
            const auto& w_up_e = w_up[e]->get_value();
            const auto& w_down_e = w_down[e]->get_value();

            auto activated_e =
                ttnn::multiply(gate_proj_e, up_proj_e, std::nullopt, std::nullopt, std::nullopt, no_acts, silu_lhs);

            // d_activated_e = dY[offsets[e]:offsets[e+1]] @ w_down_e — InputRow on dY's M-axis.
            // Pre-zero so pad rows propagate as zero through swiglu_bw and into d_gate/d_up.
            const uint32_t intermediate_dim = gate_proj_e.logical_shape()[-1];
            auto d_activated_e = ttml::core::zeros(
                ttnn::Shape({1U, 1U, upper_M_tiles * 32U, intermediate_dim}),
                &ttml::autograd::ctx().get_device(),
                grouped_value.dtype());
            ttml::metal::variable_matmul(
                dY,
                w_down_e,
                kVarMmConfig,
                std::nullopt,
                /*in0_row_offset_tiles=*/0U,
                /*effective_M_tiles=*/upper_M_tiles,
                0U,
                0U,
                d_activated_e,
                0U,
                offsets,
                ttml::metal::OffsetsRole::InputRow,
                e);

            // dW_down_e = dY^T @ activated_e — InputK overrides in0_k_offset AND K_tiles from
            // offsets[e..e+1], so only dY[offsets[e]:offsets[e+1]] participates in K-reduce.
            auto dW_down_e = ttml::metal::variable_matmul(
                dY,
                activated_e,
                kVarMmConfigTransposeA,
                std::nullopt,
                0U,
                0U,
                /*in0_k_offset_tiles=*/0U,
                0U,
                std::nullopt,
                0U,
                offsets,
                ttml::metal::OffsetsRole::InputK,
                e);
            w_down[e]->add_grad(dW_down_e);
            activated_e.deallocate();

            auto [d_gate_proj_e, d_up_proj_e] =
                ttml::metal::swiglu_elemwise_bw(gate_proj_e, up_proj_e, d_activated_e, gate_proj_e);
            up_proj_e.deallocate();
            d_activated_e.deallocate();

            // dW_gate_e / dW_up_e = d_*_proj_e^T @ grouped — WeightK overrides in1_k_offset
            // AND K_tiles, so only grouped[offsets[e]:offsets[e+1]] participates in K-reduce.
            auto dW_gate_e = ttml::metal::variable_matmul(
                d_gate_proj_e,
                grouped_value,
                kVarMmConfigTransposeA,
                std::nullopt,
                0U,
                0U,
                0U,
                /*in1_k_offset_tiles=*/0U,
                std::nullopt,
                0U,
                offsets,
                ttml::metal::OffsetsRole::WeightK,
                e);
            w_gate[e]->add_grad(dW_gate_e);
            auto dW_up_e = ttml::metal::variable_matmul(
                d_up_proj_e,
                grouped_value,
                kVarMmConfigTransposeA,
                std::nullopt,
                0U,
                0U,
                0U,
                0U,
                std::nullopt,
                0U,
                offsets,
                ttml::metal::OffsetsRole::WeightK,
                e);
            w_up[e]->add_grad(dW_up_e);

            // dX_via_* write into the expert's row range via OutputRow.
            ttml::metal::variable_matmul(
                d_gate_proj_e,
                w_gate_e,
                kVarMmConfig,
                std::nullopt,
                0U,
                0U,
                0U,
                0U,
                /*output_tensor=*/dX_via_gate,
                0U,
                offsets,
                ttml::metal::OffsetsRole::OutputRow,
                e);
            ttml::metal::variable_matmul(
                d_up_proj_e,
                w_up_e,
                kVarMmConfig,
                std::nullopt,
                0U,
                0U,
                0U,
                0U,
                /*output_tensor=*/dX_via_up,
                0U,
                offsets,
                ttml::metal::OffsetsRole::OutputRow,
                e);
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
