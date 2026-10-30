// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::transpose_split::device {

struct Params {
    uint32_t rows = 0;       // rows of the input that become dx's columns (multiple of 32)
    uint32_t tail_rows = 0;  // rows after them that become tail's columns (<= 32)
    float tail_scale = 1.0F;
};

struct Inputs {
    ttnn::Tensor input;  // [1, 1, R, T] bf16 TILE
};

using operation_attributes_t = Params;
using tensor_args_t = Inputs;
using spec_return_value_t = std::tuple<tt::tt_metal::TensorSpec, tt::tt_metal::TensorSpec>;
using tensor_return_value_t = std::tuple<ttnn::Tensor, ttnn::Tensor>;

}  // namespace ttml::metal::ops::transpose_split::device
