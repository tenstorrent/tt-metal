// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

namespace ttml::ops {

autograd::TensorPtr sample_op(
    const autograd::TensorPtr& logits,
    float temperature,
    uint32_t seed,
    const autograd::TensorPtr& logits_padding_mask = nullptr);

}  // namespace ttml::ops
