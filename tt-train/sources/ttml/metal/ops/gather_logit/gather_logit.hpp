// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// Gather per-token target logit values from a tiled logit tensor.
//
// For each (n, h):
//   c = target[n, h]
//   output[n, 0, h, 0] = logit[n, 0, h, c]   if first_v <= c < last_v
//                        = 0.0                  otherwise
//
// In the single-device case use first_v=0, last_v=V (full vocab).
// In the TP-distributed case each rank passes its shard boundaries so that
// only the device owning the target token writes a non-zero value.  An
// all-reduce of the output across ranks then yields the global target logit.
ttnn::Tensor gather_logit(
    const ttnn::Tensor& logit,   // [N, 1, H, V] TILE BFLOAT16
    const ttnn::Tensor& target,  // [N, H]       ROW_MAJOR UINT32
    uint32_t first_v,
    uint32_t last_v);

}  // namespace ttml::metal
