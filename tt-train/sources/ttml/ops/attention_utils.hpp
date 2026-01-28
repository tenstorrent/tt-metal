// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <core/ttnn_all_includes.hpp>

#include "ttnn_fixed/matmuls.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

namespace ttml::ops::detail {

// Wrapper around matmul to handle sharing of KV heads across groups of query heads (GQA).
// For Q @ K^T or attention_weights @ V:
// - G == H: standard matmul
// - G != H: broadcast KV across query head groups
inline ttnn::Tensor group_shared_matmul(
    const ttnn::Tensor& query_tensor,
    const ttnn::Tensor& kv_tensor,
    bool transpose_a = false,
    bool transpose_b = false) {
    auto [batch_num, heads, seq_len, embedding_dim] = query_tensor.logical_shape().to_array_4D();
    auto [batch_num_v, groups, seq_len_v, embedding_dim_v] = kv_tensor.logical_shape().to_array_4D();

    if (heads == groups) {
        return ttnn_fixed::matmul(query_tensor, kv_tensor, transpose_a, transpose_b);
    }

    auto M = transpose_a ? embedding_dim : seq_len;
    auto N = transpose_b ? seq_len_v : embedding_dim_v;

    auto query_tensor_grouped =
        ttnn::reshape(query_tensor, ttnn::Shape{batch_num * groups, heads / groups, seq_len, embedding_dim});
    auto kv_tensor_batched = ttnn::reshape(kv_tensor, ttnn::Shape{batch_num * groups, 1U, seq_len_v, embedding_dim_v});

    ttnn::Tensor kv_tensor_repeated = ttnn::repeat(kv_tensor_batched, ttnn::Shape{1U, heads / groups, 1U, 1U});
    auto bcasted_mm = ttnn_fixed::matmul(query_tensor_grouped, kv_tensor_repeated, transpose_a, transpose_b);
    return ttnn::reshape(bcasted_mm, ttnn::Shape{batch_num, heads, M, N});
}

// Sum gradients over query head groups for GQA backward pass.
// [B,H,S,E] -> [B,G,S,E] where G is the number of KV groups
inline ttnn::Tensor sum_over_groups(const ttnn::Tensor& ungrouped_grads, uint32_t groups) {
    auto [batch_num, num_heads, seq_len, embedding_dim] = ungrouped_grads.logical_shape().to_array_4D();
    if (groups == num_heads) {
        return ungrouped_grads;
    }
    auto grouped_grads =
        ttnn::reshape(ungrouped_grads, ttnn::Shape{batch_num * groups, num_heads / groups, seq_len, embedding_dim});
    auto summed_grads = ttnn_fixed::sum_moreh(grouped_grads, /*dim=*/1, /*keep_dim=*/true);
    return ttnn::reshape(summed_grads, ttnn::Shape{batch_num, groups, seq_len, embedding_dim});
}

}  // namespace ttml::ops::detail
