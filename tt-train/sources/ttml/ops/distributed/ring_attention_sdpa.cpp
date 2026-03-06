// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_attention_sdpa.hpp"

#include <fmt/core.h>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/hal.hpp>
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
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
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
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, std::vector<tt::tt_metal::SubDeviceId>());

    auto [batch_num, heads, seq_len_local, dim] = query_tensor.logical_shape().to_array_4D();
    // Initialize current K and V (will be ring-shifted each step)
    // Use raw ttnn::Tensor since we define backward manually
    ttnn::Tensor k_current = key->get_value();
    ttnn::Tensor v_current = value->get_value();

    // Initialize accumulators for online softmax
    // output_accum: weighted sum of outputs from all steps
    ttnn::Tensor output_accum = ttnn::zeros_like(query_tensor);

    // global_sum_exp: running sum of exp(qk - max) across all steps
    ttnn::Tensor global_sum_exp = ttnn::zeros(
        ttnn::Shape{batch_num, heads, seq_len_local, 1U},
        ttnn::DataType::BFLOAT16,
        ttnn::Layout::TILE,
        std::ref(*mesh_device));

    // global_max: running maximum attention score across all steps
    ttnn::Tensor global_max = ttnn::full(
        ttnn::Shape{batch_num, heads, seq_len_local, 1U},
        std::bit_cast<float>(0xF8000000U),  // -inf in bfloat16
        ttnn::DataType::BFLOAT16,
        ttnn::Layout::TILE,
        std::ref(*mesh_device));

    // Allocate output and intermediate tensors (mesh tensors)
    // These will be reused each step
    ttnn::Tensor output_tensor = ttnn::empty_like(query_tensor);
    ttnn::Tensor intermediate_tensor = ttnn::empty(
        ttnn::Shape{batch_num, heads, seq_len_local, 64U},
        ttnn::DataType::BFLOAT16,
        ttnn::Layout::TILE,
        mesh_device,
        ttnn::MemoryConfig(ttnn::TensorMemoryLayout::INTERLEAVED, ttnn::BufferType::DRAM));

    // Create "no contribution" intermediate tensor on device
    // Shape: (B, H, S, 64), mostly zeros except:
    // - Column 0: -inf (max_val, so any real max will dominate)
    // - Column 32: +inf (recip_sum_exp, so sum_exp = 1/inf = 0)
    ttnn::Tensor col0_neg_inf = ttnn::full(
        ttnn::Shape{batch_num, heads, seq_len_local, 1U},
        -tt::tt_metal::hal::get_inf(),  // -inf in bfloat16
        ttnn::DataType::BFLOAT16,
        ttnn::Layout::TILE,
        std::ref(*mesh_device));

    ttnn::Tensor col32_pos_inf = ttnn::full(
        ttnn::Shape{batch_num, heads, seq_len_local, 1U},
        tt::tt_metal::hal::get_inf(),  // +inf
        ttnn::DataType::BFLOAT16,
        ttnn::Layout::TILE,
        std::ref(*mesh_device));

    // Pad to 32 columns each (adds 31 zeros on the right of last dim)
    ttnn::SmallVector<ttnn::operations::data_movement::PadSpecDim> padding_spec = {
        {0, 0},  // batch
        {0, 0},  // heads
        {0, 0},  // seq_len
        {0, 31}  // width: pad 31 zeros on the right (1 -> 32)
    };
    ttnn::Tensor first_32_cols = ttnn::pad(col0_neg_inf, padding_spec, 0.0F, false, std::nullopt);
    ttnn::Tensor last_32_cols = ttnn::pad(col32_pos_inf, padding_spec, 0.0F, false, std::nullopt);

    // Concat to get full 64 columns
    ttnn::Tensor no_contrib_intermediate = ttnn::concat(std::vector<ttnn::Tensor>{first_32_cols, last_32_cols}, 3);

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

        // Extract intermediates for online softmax combination
        // Intermediate shape: (B, H, S, 64)
        // - Column 0: max_val (maximum attention score for this chunk)
        // - Column 32: recip_sum_exp (1 / sum(exp(qk - max)))

        // Extract max_val from column 0
        // Slice: [0:B, 0:H, 0:S, 0:1]
        const ttnn::SmallVector<uint32_t> slice_step = {1, 1, 1, 1};
        const ttnn::SmallVector<uint32_t> max_start = {0, 0, 0, 0};
        const ttnn::SmallVector<uint32_t> max_end = {batch_num, heads, seq_len_local, 1};
        ttnn::Tensor max_val_chunk = ttnn::slice(intermediate_tensor, max_start, max_end, slice_step);

        // Extract recip_sum_exp from column 32
        // Slice: [0:B, 0:H, 0:S, 32:33]
        const ttnn::SmallVector<uint32_t> recip_start = {0, 0, 0, 32};
        const ttnn::SmallVector<uint32_t> recip_end = {batch_num, heads, seq_len_local, 33};
        ttnn::Tensor recip_sum_exp_chunk = ttnn::slice(intermediate_tensor, recip_start, recip_end, slice_step);

        // Step 1: Compute sum_exp_chunk = 1 / recip_sum_exp
        // This is sum(exp(qk - max_chunk)) for this chunk
        ttnn::Tensor sum_exp_chunk = ttnn::reciprocal(recip_sum_exp_chunk);

        // Step 2: Update global_max = max(global_max, max_val_chunk)
        ttnn::Tensor new_global_max = ttnn::maximum(global_max, max_val_chunk);

        // Step 3: Compute rescaling factors
        // rescale_global = exp(global_max - new_global_max)  [to downscale old accumulator]
        // rescale_chunk = exp(max_val_chunk - new_global_max) [to align chunk with new max]
        ttnn::Tensor global_max_diff = ttnn::subtract(global_max, new_global_max);
        ttnn::Tensor chunk_max_diff = ttnn::subtract(max_val_chunk, new_global_max);

        ttnn::Tensor rescale_global = ttnn::exp(global_max_diff);
        ttnn::Tensor rescale_chunk = ttnn::exp(chunk_max_diff);

        // Step 4: Rescale and combine sum_exp
        // global_sum_exp tracks sum(exp(qk - current_global_max)) across all chunks seen so far
        // new_sum_exp = global_sum_exp * rescale_global + sum_exp_chunk * rescale_chunk
        //             = sum(exp(qk - new_global_max)) for all chunks
        ttnn::Tensor rescaled_global_sum = ttnn::multiply(global_sum_exp, rescale_global);
        ttnn::Tensor rescaled_chunk_sum = ttnn::multiply(sum_exp_chunk, rescale_chunk);
        ttnn::Tensor new_global_sum_exp = ttnn::add(rescaled_global_sum, rescaled_chunk_sum);

        // Step 5: Compute weights for weighted combination
        // old_weight = rescaled_global_sum / new_global_sum_exp
        // new_weight = rescaled_chunk_sum / new_global_sum_exp
        ttnn::Tensor reciprocal_new_sum = ttnn::reciprocal(new_global_sum_exp);
        ttnn::Tensor old_weight = ttnn::multiply(rescaled_global_sum, reciprocal_new_sum);
        ttnn::Tensor new_weight = ttnn::multiply(rescaled_chunk_sum, reciprocal_new_sum);

        // Step 6: Rescale and combine outputs
        // new_output = output_accum * old_weight + chunk_output * new_weight
        ttnn::Tensor scaled_old = ttnn::multiply(output_accum, old_weight);
        ttnn::Tensor scaled_new = ttnn::multiply(output_tensor, new_weight);
        output_accum = ttnn::add(scaled_old, scaled_new);

        // Step 7: Update global state
        global_max = new_global_max;
        global_sum_exp = new_global_sum_exp;

        if (step < ring_size - 1) {
            k_current = ttnn_fixed::distributed::ring_shift(
                k_current, cp_axis_value, ttnn_fixed::distributed::RingShiftDirection::Backward);
            v_current = ttnn_fixed::distributed::ring_shift(
                v_current, cp_axis_value, ttnn_fixed::distributed::RingShiftDirection::Backward);
        }
    }

    auto out = autograd::create_tensor(output_accum);
    ttnn::Tensor final_global_max = global_max;
    ttnn::Tensor final_global_sum_exp = global_sum_exp;

    autograd::GradFunction grad_fn = [query,
                                      key,
                                      value,
                                      out,
                                      final_global_max,
                                      final_global_sum_exp,
                                      k_current,  // K at end of forward (position k1)
                                      v_current,  // V at end of forward (position v1)
                                      ring_size,
                                      cp_axis_value,
                                      mask_type,
                                      mesh_device]() mutable {
        tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, std::vector<tt::tt_metal::SubDeviceId>());
        const auto& grad_output = out->get_grad();
        const auto& query_tensor = query->get_value();
        auto* mesh_device = query_tensor.device();
        auto [batch_num, heads, seq_len_local, dim] = query_tensor.logical_shape().to_array_4D();

        ttnn::Tensor grad_Q_accum = ttnn::zeros_like(query_tensor);
        ttnn::Tensor grad_K_accum = ttnn::zeros_like(key->get_value());
        ttnn::Tensor grad_V_accum = ttnn::zeros_like(value->get_value());

        ttnn::Tensor recomputed_output = ttnn::empty_like(query_tensor);
        ttnn::Tensor recomputed_intermediate = ttnn::empty(
            ttnn::Shape{batch_num, heads, seq_len_local, 64U},
            ttnn::DataType::BFLOAT16,
            ttnn::Layout::TILE,
            mesh_device,
            ttnn::MemoryConfig(ttnn::TensorMemoryLayout::INTERLEAVED, ttnn::BufferType::DRAM));
        ttnn::Tensor grad_Q_step = ttnn::zeros_like(query_tensor);
        ttnn::Tensor grad_K_step = ttnn::zeros_like(key->get_value());
        ttnn::Tensor grad_V_step = ttnn::zeros_like(value->get_value());

        ttnn::Tensor recip_final_sum = ttnn::reciprocal(final_global_sum_exp);

        // Slice parameters for extracting max and sum_exp from intermediate
        const ttnn::SmallVector<uint32_t> slice_step = {1, 1, 1, 1};
        const ttnn::SmallVector<uint32_t> max_start = {0, 0, 0, 0};
        const ttnn::SmallVector<uint32_t> max_end = {batch_num, heads, seq_len_local, 1};
        const ttnn::SmallVector<uint32_t> recip_start = {0, 0, 0, 32};
        const ttnn::SmallVector<uint32_t> recip_end = {batch_num, heads, seq_len_local, 33};

        // Loop over ring steps in reverse order (from last to first)
        for (int step = ring_size - 1; step >= 0; --step) {
            const uint32_t step_idx = step;

            {
                ttnn::Tensor zero_Q = ttnn::zeros_like(grad_Q_step);
                ttnn::Tensor zero_K = ttnn::zeros_like(grad_K_step);
                ttnn::Tensor zero_V = ttnn::zeros_like(grad_V_step);
                ttnn::copy(zero_Q, grad_Q_step);
                ttnn::copy(zero_K, grad_K_step);
                ttnn::copy(zero_V, grad_V_step);
            }

            // RECOMPUTE: Run forward SDPA to get output and intermediate for this step
            // K/V are already at the correct ring position (same as forward used at this step)
            // Use Backward direction (same as forward) since src = (device + step) % ring_size
            auto [step_output, step_intermediate] = ttml::metal::ring_sdpa_fw(
                query_tensor,
                k_current,
                v_current,
                ring_size,
                cp_axis_value,
                step_idx,
                mask_type,
                ttml::metal::ops::ring_sdpa_fw::RingDirection::Backward,
                recomputed_output,
                recomputed_intermediate);

            // RECOMPUTE: Calculate effective weight from recomputed intermediate and final global stats
            // eff_weight = sum_exp_j * exp(max_j - global_max_final) / global_sum_exp_final
            ttnn::Tensor max_j = ttnn::slice(step_intermediate, max_start, max_end, slice_step);
            ttnn::Tensor recip_sum_exp_j = ttnn::slice(step_intermediate, recip_start, recip_end, slice_step);
            ttnn::Tensor sum_exp_j = ttnn::reciprocal(recip_sum_exp_j);
            ttnn::Tensor max_diff = ttnn::subtract(max_j, final_global_max);
            ttnn::Tensor exp_diff = ttnn::exp(max_diff);
            ttnn::Tensor numerator = ttnn::multiply(sum_exp_j, exp_diff);
            ttnn::Tensor step_weight = ttnn::multiply(numerator, recip_final_sum);

            // Scale grad_output by the weight applied to this chunk in forward
            // d(chunk_output_j) = weight_j * d(final_output)
            ttnn::Tensor scaled_grad_output = ttnn::multiply(grad_output, step_weight);

            auto [grad_Q_result, grad_K_result, grad_V_result] = ttml::metal::ring_sdpa_bw(
                scaled_grad_output,
                step_output,  // Recomputed
                query_tensor,
                k_current,          // K at current ring position
                v_current,          // V at current ring position
                step_intermediate,  // Recomputed
                ring_size,
                cp_axis_value,
                step_idx,
                mask_type,
                ttml::metal::ops::ring_sdpa_bw::RingDirection::Backward,
                grad_Q_step,
                grad_K_step,
                grad_V_step);

            grad_Q_accum = ttnn::add(grad_Q_accum, grad_Q_step);
            grad_K_accum = ttnn::add(grad_K_accum, grad_K_step);
            grad_V_accum = ttnn::add(grad_V_accum, grad_V_step);

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

        // Apply gradients
        query->add_grad(grad_Q_accum);
        key->add_grad(grad_K_accum);
        value->add_grad(grad_V_accum);
    };

    auto links = autograd::get_links(query, key, value);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad_fn), links));

    return out;
}

}  // namespace ttml::ops::distributed
