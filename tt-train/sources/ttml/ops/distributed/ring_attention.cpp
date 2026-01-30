// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_attention.hpp"

#include "autograd/auto_context.hpp"
#include "ops/binary_ops.hpp"
#include "ops/distributed/comm_ops.hpp"
#include "ops/scaled_dot_product_attention.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

namespace ttml::ops::distributed {

namespace {

// Roll mask along K dimension using concat + tensor-indexed slice.
// This allows each device to roll by a different amount based on cp_rank_tensor.
// Input mask shape: (1, 1, S_local, S_full) where S_full = S_local * cp_size
// Output: (1, 1, S_local, S_full) with columns rolled so local K is first
ttnn::Tensor roll_mask_by_rank(
    const ttnn::Tensor& mask,
    const ttnn::Tensor& cp_rank_tensor,
    uint32_t seq_len_local,
    uint32_t cp_size,
    ttnn::distributed::MeshDevice& device) {
    auto mask_shape = mask.logical_shape();
    uint32_t s_local = mask_shape[2];
    uint32_t s_full = mask_shape[3];

    // Concat mask with itself along K dimension: [mask | mask]
    // Shape: (1, 1, S_local, 2*S_full)
    auto mask_doubled = ttnn::concat(std::vector<ttnn::Tensor>{mask, mask}, /*dim=*/3);

    // cp_rank_tensor is (1, 1, 1, 1) per device with value = device rank
    // Reshape to 1D: (1,)
    auto rank_1d = ttnn::reshape(cp_rank_tensor, ttnn::Shape{1});

    // roll_offset = rank * seq_len_local
    auto seq_len_tensor = ttnn::full(
        ttnn::Shape{1},
        static_cast<float>(seq_len_local),
        ttnn::DataType::BFLOAT16,
        ttnn::Layout::ROW_MAJOR,
        std::ref(device));
    auto roll_offset = ttnn::multiply(rank_1d, seq_len_tensor);

    // end_col = roll_offset + s_full
    auto s_full_tensor = ttnn::full(
        ttnn::Shape{1},
        static_cast<float>(s_full),
        ttnn::DataType::BFLOAT16,
        ttnn::Layout::ROW_MAJOR,
        std::ref(device));
    auto end_col = ttnn::add(roll_offset, s_full_tensor);

    // Create constant parts for index tensors
    auto zeros_3 = ttnn::zeros(ttnn::Shape{3}, ttnn::DataType::BFLOAT16, ttnn::Layout::ROW_MAJOR, std::ref(device));
    auto ones_2 = ttnn::ones(ttnn::Shape{2}, ttnn::DataType::BFLOAT16, ttnn::Layout::ROW_MAJOR, std::ref(device));
    auto s_local_tensor = ttnn::full(
        ttnn::Shape{1},
        static_cast<float>(s_local),
        ttnn::DataType::BFLOAT16,
        ttnn::Layout::ROW_MAJOR,
        std::ref(device));

    // Build start = [0, 0, 0, roll_offset]
    auto start = ttnn::concat(std::vector<ttnn::Tensor>{zeros_3, roll_offset}, /*dim=*/0);

    // Build end = [1, 1, S_local, end_col]
    auto end = ttnn::concat(std::vector<ttnn::Tensor>{ones_2, s_local_tensor, end_col}, /*dim=*/0);

    // Use tensor-indexed slice to roll each device's mask differently
    std::optional<ttnn::SmallVector<uint32_t>> step = std::nullopt;
    auto mask_rolled = ttnn::slice(
        mask_doubled,
        start,
        end,
        step,
        /*memory_config=*/std::nullopt,
        /*output_tensor=*/std::nullopt,
        /*pad_value=*/std::nullopt,
        /*slice_dim=*/std::nullopt,
        /*num_devices=*/cp_size);

    return mask_rolled;
}

}  // namespace

autograd::TensorPtr ring_attention_sdpa(
    const autograd::TensorPtr& query,
    const autograd::TensorPtr& key,
    const autograd::TensorPtr& value,
    const std::optional<autograd::TensorPtr>& mask) {
    auto& pctx = autograd::ctx().get_parallelism_context();
    std::optional<uint32_t> cp_axis = pctx.get_cp_axis();
    const uint32_t cp_size = pctx.get_cp_size();

    // If CP not enabled, fall back to regular SDPA
    if (!cp_axis.has_value() || cp_size <= 1) {
        return ttml::ops::scaled_dot_product_attention(query, key, value, mask);
    }

    const uint32_t cp_axis_value = cp_axis.value();

    auto k_current = key;
    auto v_current = value;

    auto [batch_num, heads, seq_len_local, dim] = query->get_value().logical_shape().to_array_4D();

    auto& device = autograd::ctx().get_device();

    // Roll mask so each device's local K chunk is at columns [0:S_local]
    std::optional<ttnn::Tensor> rolled_mask;
    if (mask.has_value()) {
        auto cp_rank_tensor_opt = pctx.get_cp_rank_tensor();
        TT_FATAL(
            cp_rank_tensor_opt.has_value(),
            "ParallelismContext must be initialized with CP enabled for ring attention with mask");
        rolled_mask =
            roll_mask_by_rank(mask.value()->get_value(), cp_rank_tensor_opt.value(), seq_len_local, cp_size, device);
    }

    // Initialize accumulators for online softmax as TensorPtr for autograd
    // output_accum: weighted sum of outputs
    // global_scale: running total of unnormalized attention scales (sum_exp * exp_max)
    autograd::TensorPtr output_accum = autograd::create_tensor(ttnn::zeros_like(query->get_value()));
    autograd::TensorPtr global_scale = autograd::create_tensor(ttnn::zeros(
        ttnn::Shape{batch_num, heads, seq_len_local, 1U},
        ttnn::DataType::BFLOAT16,
        ttnn::Layout::TILE,
        std::ref(device)));

    // Create a ones tensor for reciprocal computation (1/x = ones/x)
    autograd::TensorPtr ones_scale = autograd::create_tensor(ttnn::ones(
        ttnn::Shape{batch_num, heads, seq_len_local, 1U},
        ttnn::DataType::BFLOAT16,
        ttnn::Layout::TILE,
        std::ref(device)));

    for (uint32_t step = 0; step < cp_size; ++step) {
        // Get mask for this step
        std::optional<autograd::TensorPtr> step_mask;

        if (rolled_mask.has_value()) {
            // Slice from pre-rolled mask: columns [step*S_local : (step+1)*S_local]
            // After rolling, local K is at columns [0:S_local], next chunk at [S_local:2*S_local], etc.
            uint32_t col_start = step * seq_len_local;
            uint32_t col_end = col_start + seq_len_local;
            auto mask_shape = rolled_mask.value().logical_shape();

            ttnn::SmallVector<uint32_t> start = {0, 0, 0, col_start};
            ttnn::SmallVector<uint32_t> end = {
                static_cast<uint32_t>(mask_shape[0]),
                static_cast<uint32_t>(mask_shape[1]),
                static_cast<uint32_t>(mask_shape[2]),
                col_end};
            ttnn::SmallVector<uint32_t> step_vec = {1, 1, 1, 1};

            auto mask_slice = ttnn::slice(rolled_mask.value(), start, end, step_vec);
            step_mask = autograd::create_tensor(mask_slice);
        }

        // Get SDPA output with scale values for online softmax combination
        auto sdpa_result =
            ttml::ops::scaled_dot_product_attention_with_intermediates(query, k_current, v_current, step_mask);

        // Compute chunk scale = sum_exp * exp_max = sum(exp(qk_scaled))
        // This is the unnormalized attention weight sum for this KV chunk
        autograd::TensorPtr chunk_scale = sdpa_result.sum_exp * sdpa_result.exp_max;

        // New total scale
        autograd::TensorPtr new_global_scale = global_scale + chunk_scale;

        // Scale factors for weighted combination
        // old_weight = global_scale / new_global_scale
        // new_weight = chunk_scale / new_global_scale
        autograd::TensorPtr reciprocal_new_scale = ones_scale / new_global_scale;
        autograd::TensorPtr old_weight = global_scale * reciprocal_new_scale;
        autograd::TensorPtr new_weight = chunk_scale * reciprocal_new_scale;

        // Accumulate: output = old_output * old_weight + new_output * new_weight
        // Broadcasting is handled automatically by operator* (reduces gradients along broadcast dims)
        output_accum = output_accum * old_weight + sdpa_result.output * new_weight;
        global_scale = new_global_scale;

        // Ring shift KV to get next chunk (Backward: device i receives from i+1)
        // This gives us K chunks in order: [K_i, K_{i+1}, K_{i+2}, ...] matching the rolled mask
        if (step < cp_size - 1) {
            k_current = ring_shift(k_current, cp_axis_value, ttnn_fixed::distributed::RingShiftDirection::Backward);
            v_current = ring_shift(v_current, cp_axis_value, ttnn_fixed::distributed::RingShiftDirection::Backward);
        }
    }

    return output_accum;
}

}  // namespace ttml::ops::distributed
