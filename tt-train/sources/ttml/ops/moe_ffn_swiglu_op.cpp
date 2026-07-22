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
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/full_like/full_like.hpp"

namespace ttml::ops {

namespace {

// Build the variable_matmul config for moe_ffn. Block sizes are tuned; grid is
// device->compute_with_storage_grid_size() so we always use the full grid (110 cores on
// Blackhole, 64 on Wormhole, etc.) without per-arch hardcoding.
ttml::metal::VariableMatmulConfig make_var_mm_config(ttnn::distributed::MeshDevice* device) {
    const auto grid = device->compute_with_storage_grid_size();
    return ttml::metal::VariableMatmulConfig{
        // M/K/N block sizes tuned for the MoE matmul shapes; 2x2 subblock uses the full
        // FP32 DEST register budget (subblock_h*subblock_w == max_dest_volume).
        .M_block_size = 4,
        .K_block_size = 8,
        .N_block_size = 8,
        .subblock_h = 2,
        .subblock_w = 2,
        .compute_with_storage_grid_size = grid,
    };
}

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

    const uint32_t t_cap_tiles = std::max(1U, (token_capacity + 31U) / 32U);
    // Representative per-expert matmul-M in tiles (~ t_cap / num_experts, rounded up) — NOT a cap:
    // each expert's real row count is re-derived at runtime from offsets, and a skewed expert may
    // exceed it. Its only effect is the factory's transpose_core_grid layout decision; passing
    // t_cap_tiles itself would skew that toward the (much larger) shared-tensor M and pick the
    // wrong NOC orientation for shapes where the real per-expert M < N.
    const uint32_t per_expert_M_tiles = std::max(1U, (t_cap_tiles + num_experts - 1U) / num_experts);

    // EP-friendly: kernel reads per-expert offsets on-device, no offsets.to_vector().
    TT_FATAL(offsets.dtype() == ttnn::DataType::UINT32, "moe_ffn_swiglu_fw: offsets must be UINT32.");
    TT_FATAL(offsets.layout() == ttnn::Layout::ROW_MAJOR, "moe_ffn_swiglu_fw: offsets must be ROW_MAJOR.");
    TT_FATAL(
        offsets.logical_shape()[-1] == num_experts + 1U, "moe_ffn_swiglu_fw: offsets size must be num_experts + 1.");

    // Shared-output design: instead of E per-expert [upper*32, I] gate_proj / up_proj tensors,
    // use one [T_cap, I] tensor for each and route every expert's matmul into its own
    // [offsets[e]:offsets[e+1]) slice via OffsetsRole::InputAndOutputRow. Eliminates the
    // upper_M_tiles ceiling (so pathological skews don't truncate) and halves persistent fwd→bwd
    // memory (E×upper×I → 1×T_cap×I per intermediate; upper×E ≈ 2·T_cap, so 2×T_cap → T_cap).
    using EltwiseUnary = ttnn::operations::unary::EltwiseUnaryWithParam;
    const EltwiseUnary silu_act{ttnn::operations::unary::UnaryOpType::SILU};
    const ttsl::Span<const EltwiseUnary> no_acts;
    const ttsl::Span<const EltwiseUnary> silu_lhs(&silu_act, 1);

    // y must be pre-zeroed because the caller can read trailing slack rows
    // [offsets.back(), T_cap) which no expert's down_proj writes — those would be allocator
    // garbage in ttnn::empty. gate_proj/up_proj are internal: slack garbage propagates into
    // activated and then into d_gate_proj/d_up_proj's slack rows, but those slack rows are
    // never read by the bwd matmuls (dW_* slice K to expert offsets, dX_via_* write only
    // expert slots). So skip the zero for those two and leave them uninitialized.
    const uint32_t intermediate_dim = wg0_shape[-2];
    auto* device = &ttml::autograd::ctx().get_device();
    const auto cfg = make_var_mm_config(device);
    const auto dtype = grouped_value.dtype();
    auto y = ttnn::moreh_full_like(
        ttnn::empty(
            ttnn::Shape({1U, 1U, token_capacity, hidden_dim}),
            dtype,
            ttnn::Layout::TILE,
            device,
            ttnn::DRAM_MEMORY_CONFIG),
        0.F);
    auto empty_device = [&](const ttnn::Shape& shape) {
        return ttnn::empty(shape, dtype, ttnn::Layout::TILE, device, ttnn::DRAM_MEMORY_CONFIG);
    };
    auto gate_proj = empty_device(ttnn::Shape({1U, 1U, token_capacity, intermediate_dim}));
    auto up_proj = empty_device(ttnn::Shape({1U, 1U, token_capacity, intermediate_dim}));

    // gate_proj / up_proj: each expert reads grouped[offsets[e]:offsets[e+1]] and writes into
    // the matching slice of the shared tensor (InputAndOutputRow). w_gate / w_up are [I, H],
    // so transpose_b: x_e @ w^T.
    for (uint32_t e = 0; e < num_experts; ++e) {
        const auto& w_gate_e = w_gate[e]->get_value();
        const auto& w_up_e = w_up[e]->get_value();
        ttml::metal::variable_matmul_into_rows(
            /*input_tensor=*/grouped_value,
            /*weight_tensor=*/w_gate_e,
            /*config=*/cfg,
            /*offsets_tensor=*/offsets,
            /*output_tensor=*/gate_proj,
            /*offsets_start_index=*/e,
            /*expected_M_tiles=*/per_expert_M_tiles,
            /*transpose_a=*/false,
            /*transpose_b=*/true);
        ttml::metal::variable_matmul_into_rows(
            /*input_tensor=*/grouped_value,
            /*weight_tensor=*/w_up_e,
            /*config=*/cfg,
            /*offsets_tensor=*/offsets,
            /*output_tensor=*/up_proj,
            /*offsets_start_index=*/e,
            /*expected_M_tiles=*/per_expert_M_tiles,
            /*transpose_a=*/false,
            /*transpose_b=*/true);
    }
    // Bulk silu·multiply over the full shared tensors — pad rows compute silu(0)·0=0 so the
    // wasted work is cheap and leaves the result's pad rows zero.
    auto activated = ttnn::multiply(
        gate_proj,
        up_proj,
        /*output_dtype=*/std::nullopt,
        /*memory_config=*/std::nullopt,
        /*output_tensor=*/std::nullopt,
        /*post_op_activations=*/no_acts,
        /*input_a_activations=*/silu_lhs);

    // down_proj: read activated[offsets[e]:offsets[e+1]], write y[offsets[e]:offsets[e+1]].
    for (uint32_t e = 0; e < num_experts; ++e) {
        const auto& w_down_e = w_down[e]->get_value();
        ttml::metal::variable_matmul_into_rows(
            /*input_tensor=*/activated,
            /*weight_tensor=*/w_down_e,
            /*config=*/cfg,
            /*offsets_tensor=*/offsets,
            /*output_tensor=*/y,
            /*offsets_start_index=*/e,
            /*expected_M_tiles=*/per_expert_M_tiles,
            /*transpose_a=*/false,
            /*transpose_b=*/true);
    }
    activated.deallocate();

    auto out = autograd::create_tensor(y);

    autograd::GradFunction grad = [grouped,
                                   offsets,
                                   w_gate,
                                   w_up,
                                   w_down,
                                   out,
                                   gate_proj = std::move(gate_proj),
                                   up_proj = std::move(up_proj),
                                   num_experts,
                                   per_expert_M_tiles]() mutable {
        const auto dY = out->get_grad();
        const auto& grouped_value = grouped->get_value();
        const auto& grouped_shape = grouped_value.logical_shape();
        const uint32_t intermediate_dim = gate_proj.logical_shape()[-1];

        // dX_via_gate + dX_via_up are summed into dX which becomes grouped's gradient — slack
        // rows escape to the caller, so they must be zero. d_activated is internal: its slack
        // garbage flows into d_gate_proj/d_up_proj slack rows, but those are never read by
        // dW_* (K-sliced to expert offsets) or dX_via_* (writes only expert slots). Skip its
        // zero. The matmul writes the full expert slot for each variable_matmul, so within-slot
        // pad rows (in the last tile of each expert) compute correctly from zero K-inputs.
        auto* device = &ttml::autograd::ctx().get_device();
        const auto cfg = make_var_mm_config(device);
        const auto dtype = grouped_value.dtype();
        auto zeros_device = [&](const ttnn::Shape& shape) {
            return ttnn::moreh_full_like(
                ttnn::empty(shape, dtype, ttnn::Layout::TILE, device, ttnn::DRAM_MEMORY_CONFIG), 0.F);
        };
        auto d_activated = ttnn::empty(
            ttnn::Shape({1U, 1U, gate_proj.logical_shape()[-2], intermediate_dim}),
            dtype,
            ttnn::Layout::TILE,
            device,
            ttnn::DRAM_MEMORY_CONFIG);
        auto dX_via_gate = zeros_device(grouped_shape);
        auto dX_via_up = zeros_device(grouped_shape);

        using EltwiseUnary = ttnn::operations::unary::EltwiseUnaryWithParam;
        const EltwiseUnary silu_act{ttnn::operations::unary::UnaryOpType::SILU};
        const ttsl::Span<const EltwiseUnary> no_acts;
        const ttsl::Span<const EltwiseUnary> silu_lhs(&silu_act, 1);

        // d_activated[offsets[e]:offsets[e+1]] = dY[offsets[e]:offsets[e+1]] @ w_down_e.
        for (uint32_t e = 0; e < num_experts; ++e) {
            const auto& w_down_e = w_down[e]->get_value();
            ttml::metal::variable_matmul_into_rows(
                /*input_tensor=*/dY,
                /*weight_tensor=*/w_down_e,
                /*config=*/cfg,
                /*offsets_tensor=*/offsets,
                /*output_tensor=*/d_activated,
                /*offsets_start_index=*/e,
                /*expected_M_tiles=*/per_expert_M_tiles,
                /*transpose_a=*/false,
                /*transpose_b=*/false);
        }

        // Bulk activated = silu(gate_proj) * up_proj — needed for dW_down's K-reduce.
        auto activated =
            ttnn::multiply(gate_proj, up_proj, std::nullopt, std::nullopt, std::nullopt, no_acts, silu_lhs);
        for (uint32_t e = 0; e < num_experts; ++e) {
            // dW_down_e = dY^T @ activated — K-slice BOTH (matmul-K is the T_cap row axis on both
            // operands), reducing only over the expert's row range offsets[e..e+1].
            auto dW_down_e = ttml::metal::variable_matmul_k_sliced(
                /*input_tensor=*/dY,
                /*weight_tensor=*/activated,
                /*config=*/cfg,
                /*offsets_tensor=*/offsets,
                /*offsets_start_index=*/e,
                /*transpose_a=*/true,
                /*transpose_b=*/false);
            w_down[e]->add_grad(dW_down_e);
        }
        activated.deallocate();

        // Bulk swiglu·multiply backward — operates over the full shared tensors. Pad rows are
        // zero in/zero out. d_gate_proj aliases gate_proj (swiglu_bw writes in place).
        auto [d_gate_proj, d_up_proj] = ttml::metal::swiglu_elemwise_bw(gate_proj, up_proj, d_activated, gate_proj);
        up_proj.deallocate();
        d_activated.deallocate();

        for (uint32_t e = 0; e < num_experts; ++e) {
            const auto& w_gate_e = w_gate[e]->get_value();
            const auto& w_up_e = w_up[e]->get_value();

            // dW_gate_e / dW_up_e = d_*_proj^T @ grouped — K-slice BOTH.
            auto dW_gate_e = ttml::metal::variable_matmul_k_sliced(
                /*input_tensor=*/d_gate_proj,
                /*weight_tensor=*/grouped_value,
                /*config=*/cfg,
                /*offsets_tensor=*/offsets,
                /*offsets_start_index=*/e,
                /*transpose_a=*/true,
                /*transpose_b=*/false);
            w_gate[e]->add_grad(dW_gate_e);
            auto dW_up_e = ttml::metal::variable_matmul_k_sliced(
                /*input_tensor=*/d_up_proj,
                /*weight_tensor=*/grouped_value,
                /*config=*/cfg,
                /*offsets_tensor=*/offsets,
                /*offsets_start_index=*/e,
                /*transpose_a=*/true,
                /*transpose_b=*/false);
            w_up[e]->add_grad(dW_up_e);

            // dX_via_gate / dX_via_up: read d_*_proj[offsets[e]:offsets[e+1]], write
            // dX[offsets[e]:offsets[e+1]].
            ttml::metal::variable_matmul_into_rows(
                /*input_tensor=*/d_gate_proj,
                /*weight_tensor=*/w_gate_e,
                /*config=*/cfg,
                /*offsets_tensor=*/offsets,
                /*output_tensor=*/dX_via_gate,
                /*offsets_start_index=*/e,
                /*expected_M_tiles=*/per_expert_M_tiles,
                /*transpose_a=*/false,
                /*transpose_b=*/false);
            ttml::metal::variable_matmul_into_rows(
                /*input_tensor=*/d_up_proj,
                /*weight_tensor=*/w_up_e,
                /*config=*/cfg,
                /*offsets_tensor=*/offsets,
                /*output_tensor=*/dX_via_up,
                /*offsets_start_index=*/e,
                /*expected_M_tiles=*/per_expert_M_tiles,
                /*transpose_a=*/false,
                /*transpose_b=*/false);
        }
        d_gate_proj.deallocate();
        d_up_proj.deallocate();

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
