// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// Per-token target-logit selection for a (possibly vocab-sharded) logit tensor.
//
// Inputs
//   logit  : [N, 1, S, local_V] TILE BFLOAT16.  This device's local vocab shard,
//            covering the global vocab range [device_first_v, device_first_v + local_V).
//   target : [N, S] ROW_MAJOR UINT32.  GLOBAL token indices in [0, V_global),
//            replicated identically across all TP devices.
//
// Per-device shard window (derived inside the program factory)
//   tp_rank        = cluster_axis ? mesh_coord[*cluster_axis] : 0
//   device_first_v = first_v + tp_rank * local_V
//   device_last_v  = device_first_v + local_V
//
// Output
//   [N, 1, S, 1] TILE BFLOAT16.  For each (n, s):
//     output[n, 0, s, 0] = logit[n, 0, s, target[n, s] - device_first_v]
//                            if device_first_v <= target[n, s] < device_last_v
//                        = 0.0
//                            otherwise
//
// The output tile is zero-initialized and only positions whose target falls in the device's
// shard window are overwritten.
//
// Usage
//   Single-device, full vocab : local_V = V_global, cluster_axis = nullopt.
//   TP-distributed            : local_V = V_global / tp_size, cluster_axis = TP axis.
//                               A subsequent all-reduce over the TP axis yields the full
//                               global target logit [N, 1, S, 1].
//   Single-device, partial    : pass an explicit `first_v` to simulate a non-zero shard
//                               window without a multi-device mesh.
//
// Paired with subtract_at_target for the distributed cross-entropy backward.
ttnn::Tensor select_target_logit(
    const ttnn::Tensor& logit,
    const ttnn::Tensor& target,
    uint32_t local_V,
    std::optional<uint32_t> cluster_axis = std::nullopt,
    uint32_t first_v = 0U);

}  // namespace ttml::metal
