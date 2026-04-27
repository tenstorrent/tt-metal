// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::moe_ungroup::device {

struct operation_attributes_t {
    uint32_t e_local{};
    uint32_t k{};
    uint32_t d{};
    uint32_t b{};
    uint32_t s{};
    uint32_t h{};
    uint32_t t_cap{};
};

struct tensor_args_t {
    ttnn::Tensor expert_out;        // [1, 1, T_cap, H]   TILE  bf16
    ttnn::Tensor plan;              // [1, 1, 1, T_cap]   ROW_MAJOR uint32
    ttnn::Tensor offsets;           // [1, 1, 1, E_local+1] ROW_MAJOR uint32
    ttnn::Tensor counts;            // [1, 1, 1, E_local] ROW_MAJOR uint32
    ttnn::Tensor metadata;          // [D, B, S, K]       ROW_MAJOR uint16
    ttnn::Tensor scores;            // [D, B, S, K]       ROW_MAJOR bf16
    ttnn::Tensor local_expert_ids;  // [E_local]          ROW_MAJOR uint16
};

using spec_return_value_t = ttnn::TensorSpec;
using tensor_return_value_t = ttnn::Tensor;

}  // namespace ttml::metal::ops::moe_ungroup::device
