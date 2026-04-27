// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_ffn_swiglu_op.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"
#include "metal/operations.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/narrow/narrow.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/view/view.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn_fixed/matmuls.hpp"

namespace ttml::ops {

namespace {

// Get expert e's per-expert weight slot from W [num_experts, K, N] as rank-4 [1,1,K,N].
//
// Uses ttnn::narrow (zero-copy view) when its DRAM bank-alignment check passes,
// otherwise falls back to ttnn::slice (DRAM-to-DRAM copy). Length is always 1
// so the length-divides-dim-size constraint is trivially satisfied; the only
// runtime question is bank alignment of the per-expert block:
//
//   start_page_id = e * K * N / TILE_HW   must be divisible by num_banks
//   ⇔  K * N  divisible by  num_banks * TILE_HW
//
// On P150 (8 banks) this holds for any K, N where K*N is a multiple of 8192 (assuming 32x32 tiles)
// on P100 (7 banks) it requires a factor of 7 in K*N
ttnn::Tensor slice_expert_weight(const ttnn::Tensor& W, uint32_t e, uint32_t K, uint32_t N) {
    auto* device = W.device();
    const uint32_t num_banks = device->allocator()->get_num_banks(tt::tt_metal::BufferType::DRAM);
    const uint64_t tile_hw = W.tensor_spec().tile().get_tile_hw();
    const uint64_t bank_block = static_cast<uint64_t>(num_banks) * tile_hw;
    const bool can_narrow = (static_cast<uint64_t>(K) * N) % bank_block == 0U;

    if (can_narrow) {
        auto W_e_3d = ttnn::narrow(W, /*dim=*/0, /*start=*/static_cast<int32_t>(e), /*length=*/1U);
        return ttnn::view(W_e_3d, ttnn::Shape({1U, 1U, K, N}));
    }
    static const ttsl::SmallVector<uint32_t> step = {1U, 1U, 1U};
    const ttsl::SmallVector<uint32_t> start = {e, 0U, 0U};
    const ttsl::SmallVector<uint32_t> end = {e + 1U, K, N};
    auto W_e_3d = ttnn::slice(W, start, end, step);
    return W_e_3d.reshape(ttnn::Shape({1U, 1U, K, N}));
}

// Slice rows [row_lo, row_hi) of [1,1,T,inner] tensor
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
    const autograd::TensorPtr& w_gate,
    const autograd::TensorPtr& w_up,
    const autograd::TensorPtr& w_down) {
    const auto& grouped_value = grouped->get_value();
    const auto& w_gate_value = w_gate->get_value();
    const auto& w_up_value = w_up->get_value();
    const auto& w_down_value = w_down->get_value();

    const auto grouped_shape = grouped_value.logical_shape();
    const auto w_gate_shape = w_gate_value.logical_shape();
    const auto w_up_shape = w_up_value.logical_shape();
    const auto w_down_shape = w_down_value.logical_shape();

    const uint32_t token_capacity = grouped_shape[-2];   // total tokens
    const uint32_t hidden_dim = grouped_shape[-1];       // hidden dim
    const uint32_t num_experts = w_gate_shape[0];        // expert local
    const uint32_t intermediate_dim = w_gate_shape[-1];  // intermediate dim

    if (w_gate_shape[-2] != hidden_dim || w_up_shape[-2] != hidden_dim || w_down_shape[-1] != hidden_dim) {
        throw std::runtime_error("moe_ffn_swiglu_fw: weight inner dims do not match grouped's hidden_dim.");
    }
    if (w_up_shape[-1] != intermediate_dim || w_down_shape[-2] != intermediate_dim) {
        throw std::runtime_error("moe_ffn_swiglu_fw: gate/up/down intermediate dim mismatch.");
    }

    auto offsets_host = offsets.to_vector<uint32_t>();
    if (offsets_host.size() != num_experts + 1U) {
        throw std::runtime_error("moe_ffn_swiglu_fw: offsets size must be E_local + 1.");
    }
    if (offsets_host.back() != token_capacity) {
        throw std::runtime_error("moe_ffn_swiglu_fw: offsets[-1] must equal token_capacity.");
    }

    // Per-expert forward: slice grouped_value once per expert, run gate+up matmuls,
    // silu·multiply on the per-expert chunk, run down matmul. Output is
    // assembled with a single concat at the end. linear1_e and gate_e are
    // saved (per-expert chunks) for backward; gated_e is recomputed in
    // backward (one elementwise pass)
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
        auto Wg_e = slice_expert_weight(w_gate_value, e, hidden_dim, intermediate_dim);
        auto Wu_e = slice_expert_weight(w_up_value, e, hidden_dim, intermediate_dim);
        auto Wd_e = slice_expert_weight(w_down_value, e, intermediate_dim, hidden_dim);

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
                                   hidden_dim,
                                   intermediate_dim]() mutable {
        auto dY = out->get_grad();
        const auto& grouped_value = grouped->get_value();
        const auto& w_gate_value = w_gate->get_value();
        const auto& w_up_value = w_up->get_value();
        const auto& w_down_value = w_down->get_value();

        auto* device = grouped_value.device();
        const auto dram = ttnn::MemoryConfig{ttnn::TensorMemoryLayout::INTERLEAVED, ttnn::BufferType::DRAM};

        std::vector<ttnn::Tensor> dX_parts;
        std::vector<ttnn::Tensor> dW_gate_parts;
        std::vector<ttnn::Tensor> dW_up_parts;
        std::vector<ttnn::Tensor> dW_down_parts;
        dX_parts.reserve(num_experts);
        dW_gate_parts.reserve(num_experts);
        dW_up_parts.reserve(num_experts);
        dW_down_parts.reserve(num_experts);

        std::size_t saved_idx = 0U;
        for (uint32_t e = 0; e < num_experts; ++e) {
            const uint32_t row_lo = offsets_host[e];
            const uint32_t row_hi = offsets_host[e + 1U];

            if (row_hi == row_lo) {
                // Empty expert: contribute zero per-expert dW slots so concat
                // keeps the [num_experts, *, *] shape; no dX rows because the range is empty.
                dW_gate_parts.push_back(ttnn::zeros(
                    ttnn::Shape({1U, hidden_dim, intermediate_dim}),
                    w_gate_value.dtype(),
                    ttnn::Layout::TILE,
                    std::ref(*device),
                    dram));
                dW_up_parts.push_back(ttnn::zeros(
                    ttnn::Shape({1U, hidden_dim, intermediate_dim}),
                    w_up_value.dtype(),
                    ttnn::Layout::TILE,
                    std::ref(*device),
                    dram));
                dW_down_parts.push_back(ttnn::zeros(
                    ttnn::Shape({1U, intermediate_dim, hidden_dim}),
                    w_down_value.dtype(),
                    ttnn::Layout::TILE,
                    std::ref(*device),
                    dram));
                continue;
            }

            auto X_e = slice_rows(grouped_value, row_lo, row_hi, hidden_dim);
            auto dY_e = slice_rows(dY, row_lo, row_hi, hidden_dim);
            auto Wg_e = slice_expert_weight(w_gate_value, e, hidden_dim, intermediate_dim);
            auto Wu_e = slice_expert_weight(w_up_value, e, hidden_dim, intermediate_dim);
            auto Wd_e = slice_expert_weight(w_down_value, e, intermediate_dim, hidden_dim);

            // Recompute gated_e from saved linear1_e, gate_e (one eltwise pass).
            auto& linear1_e = linear1_parts[saved_idx];
            auto& gate_e = gate_parts[saved_idx];
            ++saved_idx;
            auto gated_e = ttnn::multiply(ttnn::silu(linear1_e), gate_e);

            // Down branch:  dgated_e = dY_e @ Wd_e^T,  dW_down_e = gated_e^T @ dY_e
            auto dgated_e = ttnn_fixed::matmul(dY_e, Wd_e, /*transpose_a=*/false, /*transpose_b=*/true);
            auto dW_down_e_4d = ttnn_fixed::matmul(gated_e, dY_e, /*transpose_a=*/true, /*transpose_b=*/false);
            dW_down_parts.push_back(dW_down_e_4d.reshape(ttnn::Shape({1U, intermediate_dim, hidden_dim})));
            gated_e.deallocate();
            dY_e.deallocate();
            Wd_e.deallocate();

            // SwiGLU eltwise BW (in-place into linear1_e's storage).
            auto [d_linear1_e, d_gate_e] = ttml::metal::swiglu_elemwise_bw(linear1_e, gate_e, dgated_e, linear1_e);
            gate_e.deallocate();
            dgated_e.deallocate();

            // dW_gate_e = X_e^T @ d_linear1_e,  dW_up_e = X_e^T @ d_gate_e
            auto dW_gate_e_4d = ttnn_fixed::matmul(X_e, d_linear1_e, /*transpose_a=*/true, /*transpose_b=*/false);
            dW_gate_parts.push_back(dW_gate_e_4d.reshape(ttnn::Shape({1U, hidden_dim, intermediate_dim})));
            auto dW_up_e_4d = ttnn_fixed::matmul(X_e, d_gate_e, /*transpose_a=*/true, /*transpose_b=*/false);
            dW_up_parts.push_back(dW_up_e_4d.reshape(ttnn::Shape({1U, hidden_dim, intermediate_dim})));
            X_e.deallocate();

            // dX_e = d_linear1_e @ Wg_e^T  +  d_gate_e @ Wu_e^T
            auto dX_via_gate_e = ttnn_fixed::matmul(d_linear1_e, Wg_e, /*transpose_a=*/false, /*transpose_b=*/true);
            auto dX_via_up_e = ttnn_fixed::matmul(d_gate_e, Wu_e, /*transpose_a=*/false, /*transpose_b=*/true);
            d_linear1_e.deallocate();
            d_gate_e.deallocate();
            Wg_e.deallocate();
            Wu_e.deallocate();

            auto dX_e = ttnn::add(dX_via_gate_e, dX_via_up_e);
            dX_via_gate_e.deallocate();
            dX_via_up_e.deallocate();
            dX_parts.push_back(std::move(dX_e));
        }

        linear1_parts.clear();
        gate_parts.clear();

        auto dX = (dX_parts.size() == 1U) ? dX_parts.front() : ttnn::concat(dX_parts, /*dim=*/2);
        auto dW_gate = ttnn::concat(dW_gate_parts, /*dim=*/0);
        auto dW_up = ttnn::concat(dW_up_parts, /*dim=*/0);
        auto dW_down = ttnn::concat(dW_down_parts, /*dim=*/0);

        grouped->add_grad(dX);
        w_gate->add_grad(dW_gate);
        w_up->add_grad(dW_up);
        w_down->add_grad(dW_down);
    };

    out->set_node(autograd::add_backward_node(std::move(grad), out, grouped, w_gate, w_up, w_down));
    return out;
}

}  // namespace ttml::ops
