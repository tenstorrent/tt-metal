// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// Per-token scalar subtraction at target positions for a (possibly vocab-sharded) tensor.
//
// Inputs
//   input          : [N, 1, H, local_V] TILE BFLOAT16.  This device's local vocab
//                    shard, covering the global vocab range [first_v, last_v); so
//                    local_V == last_v - first_v.
//   target         : [N, H] ROW_MAJOR UINT32.  GLOBAL token indices in [0, V_global),
//                    replicated identically across all TP devices.
//   subtract_value : scalar subtracted from `input` at each in-range target column.
//
// Output
//   [N, 1, H, local_V] TILE BFLOAT16, same shape/layout as `input`.  For each (n, h):
//     output[n, 0, h, target[n, h] - first_v] = input[n, 0, h, target[n, h] - first_v]
//                                               - subtract_value
//                                               if first_v <= target[n, h] < last_v
//     All other positions are copied from `input` unchanged.
//
//   `target[n,h] - first_v` translates the global token id into a column inside
//   this device's local vocab shard.  Devices whose shard does not own the target
//   column copy `input` through untouched, so the result is consistent under any
//   vocab sharding of `input`.
//
// Usage
//   Single-device: pass first_v=0, last_v=V_global (every target is in range).
//   TP-distributed: each device passes its own shard boundaries; each device
//                   independently produces its local piece of the subtracted
//                   output, no cross-device reduction required.
//
// Primary use case: distributed cross-entropy backward, where the gradient is
// softmax/N minus one_hot_target/N.  Instead of materializing the full one-hot
// tensor, compute softmax*(1/N) and then subtract 1/N at the target positions.
//
// Paired with select_target_logit for the distributed cross-entropy forward.
ttnn::Tensor subtract_at_target(
    const ttnn::Tensor& input,   // [N, 1, H, local_V] TILE BFLOAT16
    const ttnn::Tensor& target,  // [N, H]             ROW_MAJOR UINT32
    uint32_t first_v,
    uint32_t last_v,
    float subtract_value = 1.0F);

}  // namespace ttml::metal
