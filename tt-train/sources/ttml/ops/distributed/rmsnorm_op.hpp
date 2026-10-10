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
// Only per-row statistics travel between devices: one fp32 tile column per device is
// all-gathered in forward (sum x^2) and one in backward (sum x * g). The forward stats are
// kept for backward, so rms is never recomputed from x.
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

}  // namespace ttml::ops::distributed
