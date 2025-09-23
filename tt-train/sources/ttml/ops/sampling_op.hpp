// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

namespace ttml::ops {

autograd::TensorPtr sample_op(
    autograd::TensorPtr& t,
    float temperature,
    uint32_t seed,
    std::optional<tt::tt_metal::Tensor> logits_padding_mask = std::nullopt);

}  // namespace ttml::ops
