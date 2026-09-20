// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>
#include <vector>

#include "metal/common/const_utils.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// Scaled-dot-product-attention forward on the cyclic schedule of
// cyclic_sdpa_bw (tt-flash-attn, Algorithm 8): key/value columns resident on
// their owner cores, the query block and its online-softmax state relayed
// core to core as a packet, no chip-wide barrier and no atomic accumulation.
//
// Same constraints as the backward: a head dimension that is a multiple of
// 32, a sequence length (per chunk) the schedule divides, causal or unmasked.
// Grouped-query attention as in the backward: the key and value may carry a
// divisor of the query's heads.
//
// Returns the attention output in bfloat16 (the query's shape) and the
// per-row log-sum-exp of the scaled, masked scores as (batch, heads,
// sequence, 32) Float32 with the value in column 0 -- what sdpa_fw returns
// and what cyclic_sdpa_bw and the ring's merge consume.
std::tuple<ttnn::Tensor, ttnn::Tensor> cyclic_sdpa_fw(
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    uint32_t rows_per_block_tiles = 1U,
    AttentionMaskType mask_type = AttentionMaskType::Causal,
    const std::optional<ttnn::Tensor>& preallocated_output = std::nullopt,
    const std::optional<ttnn::Tensor>& preallocated_intermediates = std::nullopt,
    uint32_t max_groups = 0U,
    // Sub-problems as chunk pairs; see CyclicSDPAForwardParams.
    uint32_t sequence_chunks = 1U,
    const std::vector<uint32_t>& row_chunks = {},
    const std::vector<uint32_t>& col_chunks = {});

}  // namespace ttml::metal
