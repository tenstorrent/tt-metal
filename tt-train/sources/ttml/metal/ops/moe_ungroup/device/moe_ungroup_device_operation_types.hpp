// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::moe_ungroup::device {

struct MoeUngroupAttributes {
    uint32_t e_local{};
    uint32_t d{};
    uint32_t b{};
    uint32_t s{};
    uint32_t h{};
    uint32_t t_cap{};
};

struct MoeUngroupTensorArgs {
    ttnn::Tensor expert_out;      // [1, 1, T_cap, H]     TILE      bf16
    ttnn::Tensor plan;            // [1, 1, 1, T_cap]     ROW_MAJOR uint32
    ttnn::Tensor offsets;         // [1, 1, 1, E_local+1] ROW_MAJOR uint32
    ttnn::Tensor grouped_scores;  // [1, 1, 1, T_cap]     ROW_MAJOR bf16
};

using MoeUngroupSpecReturn = tt::tt_metal::TensorSpec;
using MoeUngroupTensorReturn = ttnn::Tensor;

using operation_attributes_t = MoeUngroupAttributes;
using tensor_args_t = MoeUngroupTensorArgs;
using spec_return_value_t = MoeUngroupSpecReturn;
using tensor_return_value_t = MoeUngroupTensorReturn;

}  // namespace ttml::metal::ops::moe_ungroup::device
