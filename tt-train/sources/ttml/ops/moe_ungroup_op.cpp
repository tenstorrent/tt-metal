// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_ungroup_op.hpp"

#include <ttnn/operations/core/core.hpp>
#include <ttnn/operations/creation/creation.hpp>
#include <ttnn/operations/data_movement/repeat/repeat.hpp>
#include <ttnn/operations/data_movement/reshape_view/reshape.hpp>
#include <ttnn/operations/data_movement/transpose/transpose.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"
#include "autograd/tensor.hpp"
#include "core/compute_kernel_config.hpp"
#include "metal/ops/moe_group/moe_group.hpp"
#include "metal/ops/moe_ungroup/moe_ungroup.hpp"

namespace ttml::ops {

autograd::TensorPtr moe_ungroup_op(
    const autograd::TensorPtr& expert_out,
    const autograd::TensorPtr& grouped_scores,
    const ttnn::Tensor& metadata,
    const ttnn::Tensor& local_expert_ids,
    const ttnn::Tensor& plan,
    const ttnn::Tensor& offsets,
    uint32_t e_local,
    uint32_t k,
    uint32_t d,
    uint32_t b,
    uint32_t s) {
    auto ungrouped_v =
        ttml::metal::moe_ungroup(expert_out->get_value(), plan, offsets, grouped_scores->get_value(), e_local, d, b, s);

    auto out = autograd::create_tensor(ungrouped_v);

    autograd::GradFunction grad = [expert_out, grouped_scores, out, metadata, local_expert_ids, e_local, k]() {
        auto* dev = &autograd::ctx().get_device();
        auto d_ungrouped = out->get_grad();  // [D, B, S, H]  bf16  ROW_MAJOR

        // Step A — gather d(ungrouped) into the grouped layout by reusing
        // metal::moe_group as a pure gather. dummy_scores is irrelevant
        // (we discard the resulting grouped_scores side-output); zeros are
        // a safe filler.
        auto dummy_scores =
            ttnn::zeros(metadata.logical_shape(), ttnn::DataType::BFLOAT16, ttnn::Layout::ROW_MAJOR, std::ref(*dev));
        auto [grad_grouped, _gs, _ks, _cnt, _off, _plan] =
            ttml::metal::moe_group(d_ungrouped, metadata, dummy_scores, local_expert_ids, e_local, k);
        // grad_grouped is [1, 1, T_cap, H]  TILE  bf16

        // Step B — d(expert_out) = grouped_scores ⊙ grad_grouped (broadcast over H).
        // grouped_scores is ROW_MAJOR [1,1,1,T_cap]; we need [1,1,T_cap,1] in TILE
        // for the broadcasting multiply. Going RM→RM via reshape forces a
        // physical repagination across DRAM banks (1 page → T_cap tiny pages);
        // route through TILE and transpose the last two dims instead — TILE
        // transpose is a per-tile swap, not a bank-stripe rewrite.
        // Pad-row values in d(expert_out) are not meaningful (the moe_group
        // kernel doesn't zero pad rows of `grouped`, and downstream consumers
        // never read pad rows) — by-design.
        auto gs_tile = ttnn::to_layout(grouped_scores->get_value(), ttnn::Layout::TILE);
        auto gs_col_tile = ttnn::transpose(gs_tile, -2, -1);
        auto d_expert_out = ttnn::multiply(grad_grouped, gs_col_tile);
        expert_out->add_grad(d_expert_out);

        // Step C — dot[i] = Σ_h expert_out[i, h] · grad_grouped[i, h].
        const auto& y_grouped_val = expert_out->get_value();  // [1,1,T_cap,H] TILE bf16
        auto prod = ttnn::multiply(y_grouped_val, grad_grouped);
        auto dot = ttnn::sum(
            prod,
            /*dim_arg=*/ttsl::SmallVector<int>{-1},
            /*keep_dim=*/true,
            /*output_mem_config=*/std::nullopt,
            /*compute_kernel_config=*/core::ComputeKernelConfig::precise());
        // dot is [1, 1, T_cap, 1] TILE bf16. Need [1, 1, 1, T_cap] ROW_MAJOR
        // to match grouped_scores' layout. Transpose-then-untilize avoids the
        // expensive RM→RM reshape across bank stripes.
        dot = ttnn::transpose(dot, -2, -1);
        dot = ttnn::to_layout(dot, ttnn::Layout::ROW_MAJOR);

        // Step D — feed dot back as d(grouped_scores).
        grouped_scores->add_grad(dot);
    };

    out->set_node(autograd::add_backward_node(std::move(grad), out, expert_out, grouped_scores));

    return out;
}

}  // namespace ttml::ops
