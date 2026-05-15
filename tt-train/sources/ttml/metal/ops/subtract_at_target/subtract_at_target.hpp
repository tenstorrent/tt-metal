// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// Per-token scalar subtraction at target positions for a (possibly vocab-sharded) tensor.
//
// Inputs
//   input          : [N, 1, S, local_V] TILE BFLOAT16.  This device's local vocab shard,
//                    covering the global vocab range [device_first_v, device_first_v + local_V).
//   target         : [N, S] ROW_MAJOR UINT32.  GLOBAL token indices in [0, V_global),
//                    replicated identically across all TP devices.
//   subtract_value : scalar subtracted from `input` at each in-range target column.
//
// Per-device shard window (derived inside the program factory)
//   tp_rank        = cluster_axis ? mesh_coord[*cluster_axis] : mesh_coord.to_linear_index(mesh_shape)
//   device_first_v = first_v + tp_rank * local_V
//   device_last_v  = device_first_v + local_V
//
// Output
//   [N, 1, S, local_V] TILE BFLOAT16, same shape/layout as `input`.  For each (n, s):
//     output[n, 0, s, target[n, s] - device_first_v] = input[…] - subtract_value
//                                                       if device_first_v <= target[n, s] < device_last_v
//     All other positions are copied from `input` unchanged.  Devices whose shard does not own
//     the target column copy `input` through untouched, so the result is consistent under any
//     vocab sharding of `input`.
//
// Usage
//   Single-device, full vocab : local_V = V_global, cluster_axis = nullopt.
//   TP-distributed            : local_V = V_global / tp_size, cluster_axis = TP axis.
//                               Each device independently produces its local piece of the
//                               subtracted output, no cross-device reduction required.
//   Single-device, partial    : pass an explicit `first_v` to simulate a non-zero shard
//                               window without a multi-device mesh.
//
// Primary use case: distributed cross-entropy backward, where the gradient is
// softmax/N minus one_hot_target/N.  Instead of materializing the full one-hot
// tensor, compute softmax*(1/N) and then subtract 1/N at the target positions.
//
// Paired with select_target_logit for the distributed cross-entropy forward.
ttnn::Tensor subtract_at_target(
    const ttnn::Tensor& input,
    const ttnn::Tensor& target,
    uint32_t local_V,
    std::optional<uint32_t> cluster_axis = std::nullopt,
    uint32_t first_v = 0U,
    float subtract_value = 1.0F);

}  // namespace ttml::metal
