// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"

namespace ttml::ops {

// PolyNorm activation with trainable coefficients for x^3, x^2, x terms plus bias.
autograd::TensorPtr polynorm(
    const autograd::TensorPtr& tensor,
    const autograd::TensorPtr& weight,
    const autograd::TensorPtr& bias,
    float epsilon = 1e-5F);

}  // namespace ttml::ops
