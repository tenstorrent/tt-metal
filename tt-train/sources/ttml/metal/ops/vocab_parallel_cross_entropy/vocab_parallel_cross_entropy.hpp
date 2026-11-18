// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <array>
#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"

// Collapse-resistant vocabulary-parallel cross-entropy (arXiv:2610.31642, Section 2) on the fused row-softmax
// program (metal/ops/log_softmax/device/row_softmax_program.hpp). The vocabulary is sharded across the TP devices:
// shard r of `local_V = logits.logical_shape()[-1]` columns owns global columns [r * local_V, (r + 1) * local_V), with
// r the device's coordinate on `cluster_axis` (its linear mesh index when nullopt). Targets are global indices,
// replicated across the TP devices; a target outside [0, tp * local_V) contributes a zero target logit and no -s term,
// as in the single-device fused cross-entropy.
//
//   stats   = vocab_parallel_ce_stats(logits, target)          per device, [B, 1, S, W] FP32
//   all     = all_gather(stats, dim 3, cluster_axis)           [B, 1, S, W * tp] FP32 (exact, no reduction)
//   loss, state = vocab_parallel_ce_combine(all)               loss [B, 1, S, 1] FP32, state [B, 1, S, 2 * W]
//   dx      = vocab_parallel_ce_bw(logits, target, state, ...) per device, local, no communication
// with W the logits' tile width; at most W shards.
namespace ttml::metal {

// Per shard and row: (m, global first-maximum column, tail without that element, target logit or 0), the
// first four columns of an FP32 tile row.
ttnn::Tensor vocab_parallel_ce_stats(
    const ttnn::Tensor& logits, const ttnn::Tensor& target, std::optional<uint32_t> cluster_axis);

// Combines the gathered shard records of each row as lanes: one global first maximum K (lowest shard on ties),
// T = sum_s t_s exp(m_s - M) + sum_{s != K's shard} exp(m_s - M), l = log1p(T) in FP32. Returns the per-row
// loss -((x_t - M) - l), exactly l when the target is K, and the row state (M, l) for the backward.
std::array<ttnn::Tensor, 2> vocab_parallel_ce_combine(const ttnn::Tensor& gathered_stats);

// dx = s * softmax(x) on every shard, with dx[t] = s * expm1((x_t - M) - l) on the target's shard (the sum of
// all other gradients, Eq. 7, when t = K). s = scaler times the upstream gradient: one value ([1, 1, 1, 1])
// or one per row ([B, 1, S, 1]) is folded into the kernel, anything else is multiplied afterwards.
ttnn::Tensor vocab_parallel_ce_bw(
    const ttnn::Tensor& logits,
    const ttnn::Tensor& target,
    const ttnn::Tensor& state,
    const ttnn::Tensor& upstream,
    float scaler,
    std::optional<uint32_t> cluster_axis);

}  // namespace ttml::metal
