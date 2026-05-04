// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_group_op.hpp"

#include <ttnn/operations/copy/typecast/typecast.hpp>
#include <ttnn/operations/core/core.hpp>
#include <ttnn/operations/creation/creation.hpp>
#include <ttnn/operations/data_movement/reshape_view/reshape.hpp>
#include <ttnn/operations/data_movement/transpose/transpose.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/eltwise/unary/unary.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"
#include "autograd/tensor.hpp"
#include "metal/ops/moe_group/moe_group.hpp"
#include "metal/ops/moe_ungroup/moe_ungroup.hpp"

namespace ttml::ops {

MoEGroupOutputs moe_group_op(
    const autograd::TensorPtr& dispatched,
    const ttnn::Tensor& metadata,
    const autograd::TensorPtr& scores,
    const ttnn::Tensor& local_expert_ids,
    uint32_t e_local,
    uint32_t k) {
    const auto& dispatched_v = dispatched->get_value();
    const auto x_shape = dispatched_v.logical_shape();
    if (x_shape.rank() != 4) {
        throw std::runtime_error("moe_group_op expects dispatched to be rank-4 [D, B, S, H].");
    }
    const uint32_t d = x_shape[0];
    const uint32_t b = x_shape[1];
    const uint32_t s = x_shape[2];

    auto [grouped_v, grouped_scores_v, k_slot, counts, offsets, plan] =
        ttml::metal::moe_group(dispatched_v, metadata, scores->get_value(), local_expert_ids, e_local, k);

    const uint32_t t_cap = grouped_v.logical_shape()[2];

    auto grouped_out = autograd::create_tensor(grouped_v);
    auto grouped_scores_out = autograd::create_tensor(grouped_scores_v);

    autograd::GradFunction grad =
        [dispatched, scores, grouped_out, grouped_scores_out, plan, offsets, k_slot, e_local, k, d, b, s, t_cap]() {
            auto* device = &autograd::ctx().get_device();
            // ones_gs: per-row weight = 1 so moe_ungroup applies no scaling.
            auto ones_gs = ttnn::full(
                ttnn::Shape({1, 1, 1, t_cap}),
                1.0F,
                ttnn::DataType::BFLOAT16,
                ttnn::Layout::ROW_MAJOR,
                std::ref(*device));

            // Step A — d(dispatched) via row-scatter, H = hidden_dim. Skipped
            // when grouped's grad was never accumulated (some test paths only
            // exercise the d(scores) branch).
            if (grouped_out->is_grad_initialized()) {
                auto d_grouped = grouped_out->get_grad();
                auto d_dispatched = ttml::metal::moe_ungroup(d_grouped, plan, offsets, ones_gs, e_local, d, b, s);
                dispatched->add_grad(d_dispatched);
            }

            // Step B — d(scores) via K-wide sparse-scatter, H = K.
            // sparse_dot[1,1,T_cap,K] has d(grouped_scores)[i] at column
            // k_slot[i] and 0 elsewhere; metal::moe_ungroup with that as
            // expert_out and ones_gs as the per-row weight produces
            // d(scores)[D,B,S,K].
            if (!grouped_scores_out->is_grad_initialized()) {
                return;
            }
            auto d_grouped_scores = grouped_scores_out->get_grad();

            // ttnn::eq doesn't accept uint16 — widen k_slot to uint32. Then
            // route to TILE [1,1,T_cap,1] via transpose rather than RM reshape
            // (RM→RM reshape repaginates across DRAM banks; TILE transpose is
            // a per-tile swap).
            auto k_slot_u32_rm = ttnn::typecast(k_slot, ttnn::DataType::UINT32);
            auto k_slot_u32_tile = ttnn::to_layout(k_slot_u32_rm, ttnn::Layout::TILE);
            auto k_slot_col_tile = ttnn::transpose(k_slot_u32_tile, -2, -1);

            auto arange_K = ttnn::arange(
                /*start=*/0,
                /*stop=*/static_cast<int64_t>(k),
                /*step=*/1,
                ttnn::DataType::UINT32,
                std::ref(*device));
            arange_K = ttnn::reshape(arange_K, ttnn::Shape({1, 1, 1, k}));
            auto arange_K_tile = ttnn::to_layout(arange_K, ttnn::Layout::TILE);

            // [1,1,T_cap,1] eq [1,1,1,K] -> [1,1,T_cap,K] uint32, broadcast in TILE.
            // Direct uint32→bf16 typecast is broken (returns 2^31 for value 1),
            // so go through float32.
            auto one_hot_u32 = ttnn::eq(k_slot_col_tile, arange_K_tile);
            auto one_hot_f32 = ttnn::typecast(one_hot_u32, ttnn::DataType::FLOAT32);
            auto one_hot_tile = ttnn::typecast(one_hot_f32, ttnn::DataType::BFLOAT16);

            // d_grouped_scores is RM [1,1,1,T_cap]; we need TILE [1,1,T_cap,1]
            // for the broadcast multiply. Same TILE-transpose route.
            auto d_tile = ttnn::to_layout(d_grouped_scores, ttnn::Layout::TILE);
            auto dot_col_tile = ttnn::transpose(d_tile, -2, -1);

            auto sparse_dot_tile = ttnn::multiply(one_hot_tile, dot_col_tile);

            auto d_scores = ttml::metal::moe_ungroup(sparse_dot_tile, plan, offsets, ones_gs, e_local, d, b, s);
            scores->add_grad(d_scores);
        };

    // grad function consumes gradients from BOTH grouped_out and
    // grouped_scores_out. Attach it to grouped_out as the primary node,
    // and add a sync-only node on grouped_scores_out depending on the
    // primary so that d(grouped_scores) is accumulated before grad runs.
    auto primary_node = autograd::add_backward_node(std::move(grad), grouped_out, dispatched, scores);
    grouped_out->set_node(primary_node);
    if (primary_node.has_value()) {
        grouped_scores_out->set_node(
            autograd::add_backward_node_always([]() {}, grouped_scores_out, dispatched, scores, grouped_out));
    }

    return MoEGroupOutputs{
        .grouped = grouped_out,
        .grouped_scores = grouped_scores_out,
        .k_slot = std::move(k_slot),
        .counts = std::move(counts),
        .offsets = std::move(offsets),
        .plan = std::move(plan),
    };
}

}  // namespace ttml::ops
