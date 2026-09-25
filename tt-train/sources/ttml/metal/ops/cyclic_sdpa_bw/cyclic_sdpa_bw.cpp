// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cyclic_sdpa_bw.hpp"

#include "device/cyclic_sdpa_bw_device_operation.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

namespace ttml::metal {

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> cyclic_sdpa_bw(
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    const ttnn::Tensor& grad_output,
    const ttnn::Tensor& log_sum_exp,
    const ttnn::Tensor& row_scalar,
    uint32_t rows_per_block_tiles,
    bool use_barrier,
    AttentionMaskType mask_type,
    bool accumulate_into_outputs,
    const std::optional<ttnn::Tensor>& preallocated_grad_query,
    const std::optional<ttnn::Tensor>& preallocated_grad_key,
    const std::optional<ttnn::Tensor>& preallocated_grad_value,
    uint32_t max_groups,
    uint32_t sequence_chunks,
    const std::vector<uint32_t>& row_chunks,
    const std::vector<uint32_t>& col_chunks,
    bool grad_query_in_tile_transposed,
    bool grad_query_out_tile_transposed) {
    auto result = ttnn::prim::ttml_cyclic_sdpa_bw(
        query, key, value, grad_output, log_sum_exp, row_scalar, rows_per_block_tiles, use_barrier, mask_type,
        accumulate_into_outputs, preallocated_grad_query, preallocated_grad_key, preallocated_grad_value,
        max_groups, sequence_chunks, row_chunks, col_chunks, grad_query_in_tile_transposed,
        grad_query_out_tile_transposed);
    return {result[0], result[1], result[2]};
}

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> cyclic_sdpa_bw_from_forward(
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    const ttnn::Tensor& grad_output,
    const ttnn::Tensor& attn_output,
    const ttnn::Tensor& log_sum_exp,
    uint32_t rows_per_block_tiles,
    bool use_barrier,
    AttentionMaskType mask_type) {
    // D = rowsum(dO . O), one value per row, which the kernel reads from
    // column 0 of a tile -- which is where a width reduction leaves it. The
    // product is written in Float32 rather than in the operands' bfloat16,
    // because D is subtracted from dP and a rounding here lands directly in
    // dS. The multiply takes the bfloat16 operands and writes Float32 itself,
    // as the ring driver does; typecasting both operands first cost two more
    // passes over (B, H, N, d) Float32 tensors (1.1 of 1.8 ms at 32 heads of
    // 128 and 5632 rows) for the same product.
    // sum_over_dim goes through moreh_sum, which takes only bfloat16 and
    // int32; the generic reduction takes Float32, which is the point of doing
    // the product in it.
    const auto row_scalar = ttml::ttnn_fixed::sum_ttnn(
        ttnn::multiply(grad_output, attn_output, ttnn::DataType::FLOAT32), /* dim */ 3, /* keep_dim */ true);

    return cyclic_sdpa_bw(
        query, key, value, grad_output, log_sum_exp, row_scalar, rows_per_block_tiles, use_barrier, mask_type);
}

}  // namespace ttml::metal
