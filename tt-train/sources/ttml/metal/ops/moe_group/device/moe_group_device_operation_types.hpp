// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tuple>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::moe_group::device {

struct operation_attributes_t {
    uint32_t e_local{};
    uint32_t k{};
    uint32_t d{};
    uint32_t b{};
    uint32_t s{};
    uint32_t h{};
    uint32_t t_cap{};  // min(e_local,k)*d*b*s + e_local*32
};

struct tensor_args_t {
    ttnn::Tensor dispatched;        // [D, B, S, H]  ROW_MAJOR bf16
    ttnn::Tensor metadata;          // [D, B, S, K]  ROW_MAJOR uint16
    ttnn::Tensor local_expert_ids;  // [E_local]      ROW_MAJOR uint16
};

// (grouped, counts, offsets, plan)
using spec_return_value_t = std::vector<ttnn::TensorSpec>;
using tensor_return_value_t = std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor, ttnn::Tensor>;

}  // namespace ttml::metal::ops::moe_group::device
