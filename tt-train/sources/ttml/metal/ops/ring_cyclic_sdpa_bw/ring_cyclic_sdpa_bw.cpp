// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_cyclic_sdpa_bw.hpp"

namespace ttml::metal {

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> ring_cyclic_sdpa_bw(
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    const ttnn::Tensor& grad_output,
    const ttnn::Tensor& log_sum_exp,
    const ttnn::Tensor& row_scalar,
    uint32_t ring_size,
    uint32_t ring_axis,
    uint32_t step,
    AttentionMaskType mask_type,
    RingCyclicDirection ring_direction,
    uint32_t rows_per_block_tiles,
    bool use_barrier,
    bool accumulate_into_outputs,
    const std::optional<ttnn::Tensor>& preallocated_grad_query,
    const std::optional<ttnn::Tensor>& preallocated_grad_key,
    const std::optional<ttnn::Tensor>& preallocated_grad_value,
    ttml::metal::ops::RingLayout layout,
    uint32_t zigzag_pair,
    bool grad_query_in_tile_transposed,
    bool grad_query_out_tile_transposed) {
    auto result = ttnn::prim::ttml_ring_cyclic_sdpa_bw(
        query,
        key,
        value,
        grad_output,
        log_sum_exp,
        row_scalar,
        ring_size,
        ring_axis,
        step,
        mask_type,
        ring_direction,
        rows_per_block_tiles,
        use_barrier,
        accumulate_into_outputs,
        preallocated_grad_query,
        preallocated_grad_key,
        preallocated_grad_value,
        layout,
        zigzag_pair,
        grad_query_in_tile_transposed,
        grad_query_out_tile_transposed);
    return {result[0], result[1], result[2]};
}

}  // namespace ttml::metal
