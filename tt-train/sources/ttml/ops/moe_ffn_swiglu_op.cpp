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

// Slice rows [row_lo, row_hi) of a [1,1,T,inner] tensor.
ttnn::Tensor slice_rows(const ttnn::Tensor& T, uint32_t row_lo, uint32_t row_hi, uint32_t inner) {
    static const ttsl::SmallVector<uint32_t> step = {1U, 1U, 1U, 1U};
    const ttsl::SmallVector<uint32_t> start = {0U, 0U, row_lo, 0U};
    const ttsl::SmallVector<uint32_t> end = {1U, 1U, row_hi, inner};
    return ttnn::slice(T, start, end, step);
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
    // linear1_e and gate_e are saved per-expert for backward; gated_e is
    // recomputed in backward.
    std::vector<ttnn::Tensor> y_parts;
    std::vector<ttnn::Tensor> linear1_parts;
    std::vector<ttnn::Tensor> gate_parts;
    y_parts.reserve(num_experts);
    linear1_parts.reserve(num_experts);
    gate_parts.reserve(num_experts);

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
        const auto& Wg_e = w_gate[e]->get_value();
        const auto& Wu_e = w_up[e]->get_value();
        const auto& Wd_e = w_down[e]->get_value();

        auto linear1_e = ttnn_fixed::matmul(X_e, Wg_e, false, false);  // [1,1,len,intermediate_dim]
        auto gate_e = ttnn_fixed::matmul(X_e, Wu_e, false, false);     // [1,1,len,intermediate_dim]
        auto gated_e = ttnn::multiply(ttnn::silu(linear1_e), gate_e);  // [1,1,len,intermediate_dim]
        auto y_e = ttnn_fixed::matmul(gated_e, Wd_e, false, false);    // [1,1,len,hidden_dim]
        gated_e.deallocate();

        linear1_parts.push_back(std::move(linear1_e));
        gate_parts.push_back(std::move(gate_e));
        y_parts.push_back(std::move(y_e));
    }

    if (y_parts.empty()) {
        throw std::runtime_error("moe_ffn_swiglu_fw: all experts empty (token_capacity == 0).");
    }
    auto y = (y_parts.size() == 1U) ? y_parts.front() : ttnn::concat(y_parts, /*dim=*/2);
    y_parts.clear();

    auto out = autograd::create_tensor(y);

    autograd::GradFunction grad = [grouped,
                                   w_gate,
                                   w_up,
                                   w_down,
                                   out,
                                   offsets_host = std::move(offsets_host),
                                   linear1_parts = std::move(linear1_parts),
                                   gate_parts = std::move(gate_parts),
                                   num_experts,
                                   hidden_dim]() mutable {
        auto dY = out->get_grad();
        const auto& grouped_value = grouped->get_value();

        std::vector<ttnn::Tensor> dX_parts;
        dX_parts.reserve(num_experts);

        std::size_t saved_idx = 0U;
        for (uint32_t e = 0; e < num_experts; ++e) {
            const uint32_t row_lo = offsets_host[e];
            const uint32_t row_hi = offsets_host[e + 1U];

            if (row_hi == row_lo) {
                // Empty expert: zero gradient — nothing to add to w_*[e].
                continue;
            }

            auto X_e = slice_rows(grouped_value, row_lo, row_hi, hidden_dim);
            auto dY_e = slice_rows(dY, row_lo, row_hi, hidden_dim);
            const auto& Wg_e = w_gate[e]->get_value();
            const auto& Wu_e = w_up[e]->get_value();
            const auto& Wd_e = w_down[e]->get_value();

            // Recompute gated_e from saved linear1_e, gate_e (one eltwise pass).
            auto& linear1_e = linear1_parts[saved_idx];
            auto& gate_e = gate_parts[saved_idx];
            ++saved_idx;
            auto gated_e = ttnn::multiply(ttnn::silu(linear1_e), gate_e);

            // Down branch:  dgated_e = dY_e @ Wd_e^T,  dW_down_e = gated_e^T @ dY_e
            auto dgated_e = ttnn_fixed::matmul(dY_e, Wd_e, /*transpose_a=*/false, /*transpose_b=*/true);
            auto dW_down_e = ttnn_fixed::matmul(gated_e, dY_e, /*transpose_a=*/true, /*transpose_b=*/false);
            w_down[e]->add_grad(dW_down_e);
            gated_e.deallocate();
            dY_e.deallocate();

            // SwiGLU eltwise BW (in-place into linear1_e's storage).
            auto [d_linear1_e, d_gate_e] = ttml::metal::swiglu_elemwise_bw(linear1_e, gate_e, dgated_e, linear1_e);
            gate_e.deallocate();
            dgated_e.deallocate();

            // dW_gate_e = X_e^T @ d_linear1_e,  dW_up_e = X_e^T @ d_gate_e — added directly to per-expert grads.
            auto dW_gate_e = ttnn_fixed::matmul(X_e, d_linear1_e, /*transpose_a=*/true, /*transpose_b=*/false);
            w_gate[e]->add_grad(dW_gate_e);
            auto dW_up_e = ttnn_fixed::matmul(X_e, d_gate_e, /*transpose_a=*/true, /*transpose_b=*/false);
            w_up[e]->add_grad(dW_up_e);
            X_e.deallocate();

            // dX_e = d_linear1_e @ Wg_e^T  +  d_gate_e @ Wu_e^T
            auto dX_via_gate_e = ttnn_fixed::matmul(d_linear1_e, Wg_e, /*transpose_a=*/false, /*transpose_b=*/true);
            auto dX_via_up_e = ttnn_fixed::matmul(d_gate_e, Wu_e, /*transpose_a=*/false, /*transpose_b=*/true);
            d_linear1_e.deallocate();
            d_gate_e.deallocate();

            auto dX_e = ttnn::add(dX_via_gate_e, dX_via_up_e);
            dX_via_gate_e.deallocate();
            dX_via_up_e.deallocate();
            dX_parts.push_back(std::move(dX_e));
        }

        linear1_parts.clear();
        gate_parts.clear();

        auto dX = (dX_parts.size() == 1U) ? dX_parts.front() : ttnn::concat(dX_parts, /*dim=*/2);
        grouped->add_grad(dX);
    };

    // Manual autograd registration: variadic add_backward_node template can't
    // unpack a runtime-sized weight vector, so we build the link list directly
    // and call ctx().add_backward_node ourselves.
    bool needs_grad = (grouped != nullptr) && grouped->get_requires_grad();
    auto check_list = [&](const std::vector<autograd::TensorPtr>& v) {
        for (const auto& t : v) {
            if (t && t->get_requires_grad()) {
                needs_grad = true;
                return;
            }
        }
    };
    check_list(w_gate);
    check_list(w_up);
    check_list(w_down);
    out->set_requires_grad(needs_grad);

    if (needs_grad) {
        std::vector<autograd::NodeId> links;
        links.reserve(1U + 3U * num_experts);
        auto add_link = [&](const autograd::TensorPtr& t) {
            if (!t) {
                return;
            }
            const auto& node = t->get_node();
            if (node) {
                links.push_back(node.value());
            }
        };
        add_link(grouped);
        for (const auto& w : w_gate) add_link(w);
        for (const auto& w : w_up) add_link(w);
        for (const auto& w : w_down) add_link(w);
        out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    }

    return out;
}

}  // namespace ttml::ops
