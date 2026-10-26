// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "autograd/tensor.hpp"

namespace ttml::ops::distributed {

// RMSNorm over a hidden dimension sharded across `cluster_axis`.
//
// tensor : [B, 1, S, C / tp] slice of the input, sharded along the last dim (e.g. with
//          shard_tensor_to_mesh_mapper(device, 3, cluster_axis)). C / tp must be tile-aligned.
// gamma  : [1, 1, 1, C / tp] slice of the gain, sharded like the input.
// cluster_axis : mesh axis the hidden dim is sharded across (TP axis).
//                nullopt for a 1-D mesh where all devices are TP.
//
// Only per-row values travel between devices. Forward all-gathers one fp32 tile column per
// device (sum of x^2). Backward all-gathers one fp32 tile column per device (sum of
// x * gamma * dy). When a backward node is recorded, the forward also reduces the gathered stats
// to rms, one value per row, which is all the node keeps.
//
// The forward runs ttnn's interleaved rms_norm_pre_all_gather and rms_norm_post_all_gather, which
// keep the whole local row in L1. The per-device width is the largest tile-aligned width whose
// circular buffers fit in L1 on the opened device (near 4500 elements on Blackhole). Wider shards
// are rejected here, before those ops run.
//
// Gradients are sharded like their inputs and dL/dgamma needs no reduction across
// `cluster_axis`. Any other mesh axis is untouched: with the batch sharded over it, dL/dgamma
// is a per-replica partial sum that the data-parallel gradient sync all-reduces.
//
// With a single shard (one device, or an axis of size 1) this is ops::rmsnorm.
autograd::TensorPtr rmsnorm(
    const autograd::TensorPtr& tensor,
    const autograd::TensorPtr& gamma,
    float epsilon,
    std::optional<uint32_t> cluster_axis = std::nullopt);

// rms of the full row, [B, 1, S, 1] bf16, from the all-gathered rms_norm_pre_all_gather stats
// [B, 1, S, 32 * num_shards] fp32. Each shard's tile holds its sum(x^2) in column 0 and zeros in
// columns 1..31.
ttnn::Tensor rms_from_gathered_stats(const ttnn::Tensor& stats, uint32_t full_width, float epsilon);

}  // namespace ttml::ops::distributed
