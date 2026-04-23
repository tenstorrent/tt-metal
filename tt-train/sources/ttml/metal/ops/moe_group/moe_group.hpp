// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tuple>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// Return type: (grouped, counts, offsets, plan)
//   grouped : [1, 1, T_cap, H]  TILE bf16  DRAM
//   counts  : [E_local]          uint32     DRAM
//   offsets : [E_local + 1]      uint32     DRAM
//   plan    : [T_cap]            uint32     DRAM
using MoeGroupResult = std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor, ttnn::Tensor>;

MoeGroupResult moe_group(
    const ttnn::Tensor& dispatched,        // [D, B, S, H]  ROW_MAJOR bf16
    const ttnn::Tensor& metadata,          // [D, B, S, K]  ROW_MAJOR int32
    const ttnn::Tensor& local_expert_ids,  // [E_local]      ROW_MAJOR uint32
    uint32_t e_local,
    uint32_t k);

}  // namespace ttml::metal
