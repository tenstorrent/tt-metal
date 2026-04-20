// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// Per-token target-logit selection for a (possibly vocab-sharded) logit tensor.
//
// Inputs
//   logit  : [N, 1, H, local_V] TILE BFLOAT16.  This device's local vocab shard,
//            covering the global vocab range [first_v, last_v); so
//            local_V == last_v - first_v.
//   target : [N, H] ROW_MAJOR UINT32.  GLOBAL token indices in [0, V_global),
//            replicated identically across all TP devices.
//
// Output
//   [N, 1, H, 1] TILE BFLOAT16.  For each (n, h):
//     output[n, 0, h, 0] = logit[n, 0, h, target[n, h] - first_v]
//                            if first_v <= target[n, h] < last_v
//                        = 0.0
//                            otherwise
//
//   The output tile is zero-initialized and only positions whose target falls
//   in [first_v, last_v) are overwritten.  `target[n,h] - first_v` translates
//   the global token id into a column inside this device's local vocab shard.
//
// Usage
//   Single-device: pass first_v=0, last_v=V_global (every position is in range).
//   TP-distributed: each device passes its own shard boundaries; a subsequent
//                   all-reduce over the TP axis yields the full global target
//                   logit [N, 1, H, 1].
//
// Paired with subtract_at_target for the distributed cross-entropy backward.
ttnn::Tensor select_target_logit(
    const ttnn::Tensor& logit,   // [N, 1, H, local_V] TILE BFLOAT16
    const ttnn::Tensor& target,  // [N, H]             ROW_MAJOR UINT32
    uint32_t first_v,
    uint32_t last_v);

}  // namespace ttml::metal
