// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"
#include "metal/ops/ring_sdpa_workload/ring_sdpa_bw_device_operation.hpp"
#include "metal/ops/ring_sdpa_workload/ring_sdpa_device_operation.hpp"

namespace ttml::ops::distributed {

/**
 * Ring Attention SDPA for Context Parallelism.
 *
 * Computes full-sequence attention when the sequence is sharded across CP devices.
 * Each device holds a chunk of Q, K, V and collaboratively computes attention over
 * the full sequence using ring communication.
 *
 * If CP is not enabled in ParallelismContext, falls back to regular SDPA.
 *
 * Algorithm:
 *   For each device holding Q_local, K_local, V_local:
 *     Initialize: O = 0, L = -inf (log-sum-exp accumulator)
 *     For step in 0..cp_size:
 *       Compute local attention scores and accumulate with online softmax
 *       K_current, V_current = ring_shift(K_current, V_current, cp_axis)
 *     Return O
 *
 * @param query Local query tensor (B, H, S_local, D)
 * @param key Local key tensor (B, G, S_local, D)
 * @param value Local value tensor (B, G, S_local, D)
 * @param mask Optional attention mask with shape (1, 1, S_local, S_full)
 *             where S_full = S_local * cp_size. Each device's mask contains
 *             attention values for its Q chunk attending to ALL K positions.
 *             Rolled per-device using ParallelismContext::get_cp_rank_tensor().
 * @param use_causal_mask If true, use causal masking with optimized ring attention.
 *             Each device only computes attention for valid chunks:
 *             - Step 0: causal mask (local K/V)
 *             - Steps where source device < current device: full attention
 *             - Steps where source device > current device: skip (no computation)
 * @return Attention output tensor (B, H, S_local, D)
 *
 * Note: KV is passed in the ring (rather than Q) because in GQA, num_groups <= num_heads,
 * making KV typically lighter to transfer.
 */
autograd::TensorPtr ring_attention_sdpa(
    const autograd::TensorPtr& query,
    const autograd::TensorPtr& key,
    const autograd::TensorPtr& value,
    const std::optional<autograd::TensorPtr>& mask = std::nullopt,
    const ttml::metal::AttentionMaskType mask_type = ttml::metal::AttentionMaskType::Causal);

}  // namespace ttml::ops::distributed
