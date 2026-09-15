// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_attention_sdpa.hpp"

#include <fmt/core.h>

#include <cmath>
#include <limits>
#include <tt-metalium/distributed.hpp>
#include <umd/device/cluster.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/random.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/common/const_utils.hpp"
#include "metal/ops/ring_sdpa_bw/ring_sdpa_bw.hpp"
#include "metal/ops/ring_sdpa_fw/ring_sdpa_fw.hpp"
#include "ops/binary_ops.hpp"
#include "ops/distributed/comm_ops.hpp"
#include "ops/scaled_dot_product_attention.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/data_movement/copy/copy.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn_fixed/distributed/tt_metal.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace ttml::ops::distributed {

namespace {

// Pads a (B, H, S, 1) FP32 logsumexp tensor into the (B, H, S, 32) intermediates layout
// the SDPA kernels expect: lse in column 0, the remaining 31 columns are ignored padding.
ttnn::Tensor pad_lse_to_intermediates_layout(const ttnn::Tensor& lse) {
    const ttsl::SmallVector<ttnn::operations::data_movement::PadSpecDim> padding = {
        {0, 0},  // batch
        {0, 0},  // heads
        {0, 0},  // seq_len
        {0, 31}  // width: pad 31 zeros on the right (1 -> 32)
    };
    return ttnn::pad(lse, padding, 0.0F, false, std::nullopt);
}

}  // namespace

autograd::TensorPtr ring_attention_sdpa(
    const autograd::TensorPtr& query,
    const autograd::TensorPtr& key,
    const autograd::TensorPtr& value,
    const std::optional<autograd::TensorPtr>& mask,
    const ttml::metal::AttentionMaskType mask_type) {
    if (!autograd::ctx().is_parallelism_context_initialized() ||
        !autograd::ctx().get_parallelism_context().is_cp_enabled()) {
        return ttml::ops::scaled_dot_product_attention(query, key, value, mask);
    }

    const auto& pctx = autograd::ctx().get_parallelism_context();
    const uint32_t cp_axis_value = pctx.get_cp_axis().value();
    const uint32_t ring_size = pctx.get_cp_size();
    const auto& query_tensor = query->get_value();
    auto* mesh_device = query_tensor.device();
    TT_FATAL(mesh_device != nullptr, "Query tensor must be on a mesh device for ring attention");
    TT_FATAL(
        !mask.has_value(),
        "Non-causal mask is not supported in CP mode for now, pass nullopt if you want to use causal mask");
    TT_FATAL(
        mask_type != ttml::metal::AttentionMaskType::Arbitrary,
        "Arbitrary attention mask is not supported in CP mode, use None or Causal");
    tt::tt_metal::distributed::Synchronize(*mesh_device, std::nullopt, std::vector<tt::tt_metal::SubDeviceId>());

    auto [batch_num, heads, seq_len_local, dim] = query_tensor.logical_shape().to_array_4D();
    // Initialize current K and V (will be ring-shifted each step)
    // Use raw ttnn::Tensor since we define backward manually
    ttnn::Tensor k_current = key->get_value();
    ttnn::Tensor v_current = value->get_value();

    // Initialize accumulators for online softmax.
    // output_accum: weighted sum of outputs from all steps. Kept FP32: the online-softmax
    // rescaling rewrites the whole accumulator every ring step, so a bf16 accumulator
    // compounds ~ring_size rounding errors into the saved O. Backward consumes that O in
    // u = rowsum(dO * O), where the error is amplified by the softmax-backward
    // cancellation in (dP - u) — enough to push dK outside test tolerance.
    ttnn::Tensor output_accum = ttnn::full(
        ttnn::Shape{batch_num, heads, seq_len_local, dim},
        0.0F,
        ttnn::DataType::FLOAT32,
        ttnn::Layout::TILE,
        std::ref(*mesh_device));

    // global_lse: running logsumexp across all steps
    // lse = log(sum(exp(scale * score_i))) — the log of the softmax normalizer
    // Initialized to -inf (no contribution: exp(-inf) = 0)
    ttnn::Tensor global_lse = ttnn::full(
        ttnn::Shape{batch_num, heads, seq_len_local, 1U},
        -std::numeric_limits<float>::infinity(),
        ttnn::DataType::FLOAT32,
        ttnn::Layout::TILE,
        std::ref(*mesh_device));

    // Allocate output and intermediate tensors (mesh tensors)
    // These will be reused each step
    ttnn::Tensor output_tensor = ttnn::empty_like(query_tensor);
    ttnn::Tensor intermediate_tensor = ttnn::empty(
        ttnn::Shape{batch_num, heads, seq_len_local, 32U},
        ttnn::DataType::FLOAT32,
        ttnn::Layout::TILE,
        mesh_device,
        ttnn::MemoryConfig(tt::tt_metal::TensorMemoryLayout::INTERLEAVED, ttnn::BufferType::DRAM));

    // "no contribution" intermediate: logsumexp = -inf (col 0), rest zeros
    // exp(-inf) = 0, so this chunk contributes nothing to the combined softmax
    ttnn::Tensor col0_neg_inf = ttnn::full(
        ttnn::Shape{batch_num, heads, seq_len_local, 1U},
        -std::numeric_limits<float>::infinity(),
        ttnn::DataType::FLOAT32,
        ttnn::Layout::TILE,
        std::ref(*mesh_device));
    ttnn::Tensor no_contrib_intermediate = pad_lse_to_intermediates_layout(col0_neg_inf);

    for (uint32_t step = 0; step < ring_size; ++step) {
        // For causal masking, initialize intermediate_tensor to "no contribution" values
        // Devices that are skipped will keep these values, indicating zero contribution
        if (mask_type == ttml::metal::AttentionMaskType::Causal) {
            ttnn::copy(no_contrib_intermediate, intermediate_tensor);
        }

        auto [out_tensor, inter_tensor] = ttml::metal::ring_sdpa_fw(
            query_tensor,
            k_current,
            v_current,
            ring_size,
            cp_axis_value,
            step,
            mask_type,
            ttml::metal::ops::ring_sdpa_fw::RingDirection::Backward,
            output_tensor,
            intermediate_tensor);

        // Extract logsumexp from column 0 of intermediate
        // Intermediate shape: (B, H, S, 32) FP32, logsumexp in column 0
        const ttsl::SmallVector<uint32_t> slice_step = {1, 1, 1, 1};
        const ttsl::SmallVector<uint32_t> lse_start = {0, 0, 0, 0};
        const ttsl::SmallVector<uint32_t> lse_end = {batch_num, heads, seq_len_local, 1};
        ttnn::Tensor lse_chunk = ttnn::slice(intermediate_tensor, lse_start, lse_end, slice_step);

        // Combine via logaddexp: new_lse = log(exp(global_lse) + exp(lse_chunk))
        // Numerically stable form: m = max(a,b); result = m + log(exp(a-m) + exp(b-m))
        ttnn::Tensor m = ttnn::maximum(global_lse, lse_chunk);
        ttnn::Tensor exp_global = ttnn::exp(ttnn::subtract(global_lse, m));
        ttnn::Tensor exp_chunk = ttnn::exp(ttnn::subtract(lse_chunk, m));
        ttnn::Tensor new_lse = ttnn::add(m, ttnn::log(ttnn::add(exp_global, exp_chunk)));

        // Weights for combining outputs: w = exp(lse - new_lse) = Z_i / Z_combined
        ttnn::Tensor old_weight = ttnn::exp(ttnn::subtract(global_lse, new_lse));
        ttnn::Tensor new_weight = ttnn::exp(ttnn::subtract(lse_chunk, new_lse));

        // Weighted combination of accumulated output and this step's output.
        // The step output is upcast so the whole combine stays FP32; mixed-dtype
        // binary ops would otherwise round the products back to bf16.
        ttnn::Tensor step_output_fp32 = ttnn::typecast(output_tensor, ttnn::DataType::FLOAT32);
        output_accum =
            ttnn::add(ttnn::multiply(output_accum, old_weight), ttnn::multiply(step_output_fp32, new_weight));

        global_lse = new_lse;

        if (step < ring_size - 1) {
            k_current = ttnn_fixed::distributed::ring_shift(
                k_current, cp_axis_value, ttnn_fixed::distributed::RingShiftDirection::Backward);
            v_current = ttnn_fixed::distributed::ring_shift(
                v_current, cp_axis_value, ttnn_fixed::distributed::RingShiftDirection::Backward);
        }
    }

    // Single rounding to the input dtype; the graph (and the saved O used by backward)
    // sees the same dtype as before.
    auto out = autograd::create_tensor(ttnn::typecast(output_accum, query_tensor.dtype()));
    ttnn::Tensor final_lse = global_lse;

    autograd::GradFunction grad_fn = [query,
                                      key,
                                      value,
                                      out,
                                      final_lse,
                                      k_current,  // K at end of forward (position k1)
                                      v_current,  // V at end of forward (position v1)
                                      ring_size,
                                      cp_axis_value,
                                      mask_type,
                                      mesh_device]() mutable {
        tt::tt_metal::distributed::Synchronize(*mesh_device, std::nullopt, std::vector<tt::tt_metal::SubDeviceId>());
        const auto& grad_output = out->get_grad();
        const auto& attn_output = out->get_value();
        const auto& query_tensor = query->get_value();

        // FP32 host accumulators: each ring step contributes a bf16 kernel output, but
        // summing them in bf16 rounds the full running magnitude every step. The
        // accumulators are cast back to the input dtype once, after the loop.
        ttnn::Tensor grad_Q_accum = ttnn::zeros_like(query_tensor, ttnn::DataType::FLOAT32);
        ttnn::Tensor grad_K_accum = ttnn::zeros_like(key->get_value(), ttnn::DataType::FLOAT32);
        ttnn::Tensor grad_V_accum = ttnn::zeros_like(value->get_value(), ttnn::DataType::FLOAT32);

        ttnn::Tensor grad_Q_step = ttnn::zeros_like(query_tensor);
        ttnn::Tensor grad_K_step = ttnn::zeros_like(key->get_value());
        ttnn::Tensor grad_V_step = ttnn::zeros_like(value->get_value());

        // Standard ring-flash-attention backward: every step gets the UNSCALED upstream
        // gradient, the GLOBAL forward output, and the GLOBAL logsumexp. The sdpa_bw
        // kernels recompute P = exp(scale*QK^T - lse) from the supplied lse, so the global
        // lse yields the global attention weights restricted to this chunk's columns, and
        // u = rowsum(dO * O_global) is the global softmax-backward correction. Per-chunk
        // contributions then sum to the exact dQ/dK/dV. Feeding per-chunk O/lse here would
        // bias dQ/dK (per-chunk u instead of global) — only dV would come out right.
        ttnn::Tensor global_intermediates = pad_lse_to_intermediates_layout(final_lse);

        // Zero sources for resetting the step buffers before every ring step.
        const ttnn::Tensor zero_Q = ttnn::zeros_like(grad_Q_step);
        const ttnn::Tensor zero_K = ttnn::zeros_like(grad_K_step);
        const ttnn::Tensor zero_V = ttnn::zeros_like(grad_V_step);

        // Loop over ring steps in reverse order (from last to first)
        for (int step = ring_size - 1; step >= 0; --step) {
            const uint32_t step_idx = step;

            // Devices skipped by the causal schedule at this step do not run the kernels,
            // so their step buffers must be zeroed to contribute nothing to the accumulators.
            ttnn::copy(zero_Q, grad_Q_step);
            ttnn::copy(zero_K, grad_K_step);
            ttnn::copy(zero_V, grad_V_step);

            // Use Backward direction (same as forward) since src = (device + step) % ring_size
            auto [grad_Q_result, grad_K_result, grad_V_result] = ttml::metal::ring_sdpa_bw(
                grad_output,
                attn_output,  // Global forward output saved by autograd
                query_tensor,
                k_current,             // K at current ring position
                v_current,             // V at current ring position
                global_intermediates,  // Global logsumexp in intermediates layout
                ring_size,
                cp_axis_value,
                step_idx,
                mask_type,
                ttml::metal::ops::ring_sdpa_bw::RingDirection::Backward,
                grad_Q_step,
                grad_K_step,
                grad_V_step);

            // The results alias the preallocated step buffers; skipped devices keep the
            // zeros written above. Upcast so the accumulation stays FP32.
            grad_Q_accum = ttnn::add(grad_Q_accum, ttnn::typecast(grad_Q_result, ttnn::DataType::FLOAT32));
            grad_K_accum = ttnn::add(grad_K_accum, ttnn::typecast(grad_K_result, ttnn::DataType::FLOAT32));
            grad_V_accum = ttnn::add(grad_V_accum, ttnn::typecast(grad_V_result, ttnn::DataType::FLOAT32));

            // Ring shift K/V and grad accumulators in FORWARD direction
            // K/V: replays the forward pass in reverse (gets K/V for previous step)
            // grad_K/V: routes accumulated gradients back to correct device
            if (step > 0) {
                // Shift K/V forward to get position for next backward iteration
                k_current = ttnn_fixed::distributed::ring_shift(
                    k_current, cp_axis_value, ttnn_fixed::distributed::RingShiftDirection::Forward);
                v_current = ttnn_fixed::distributed::ring_shift(
                    v_current, cp_axis_value, ttnn_fixed::distributed::RingShiftDirection::Forward);

                // Shift grad accumulators
                grad_K_accum = ttnn_fixed::distributed::ring_shift(
                    grad_K_accum, cp_axis_value, ttnn_fixed::distributed::RingShiftDirection::Forward);
                grad_V_accum = ttnn_fixed::distributed::ring_shift(
                    grad_V_accum, cp_axis_value, ttnn_fixed::distributed::RingShiftDirection::Forward);
            }
        }

        // Apply gradients, rounded once to the parameter dtype
        query->add_grad(ttnn::typecast(grad_Q_accum, query_tensor.dtype()));
        key->add_grad(ttnn::typecast(grad_K_accum, key->get_value().dtype()));
        value->add_grad(ttnn::typecast(grad_V_accum, value->get_value().dtype()));
    };

    out->set_node(autograd::add_backward_node(std::move(grad_fn), out, query, key, value));

    return out;
}

}  // namespace ttml::ops::distributed
