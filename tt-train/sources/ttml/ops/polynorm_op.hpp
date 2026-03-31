// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"

namespace ttml::ops {

// PolyNorm3 activation with trainable coefficients for x^3, x^2, x terms plus bias.
// NOTE: This op is intentionally fixed to 3 terms ([1, 1, 1, 3] weight) for now.
// Generalizing to N terms is a possible future extension.
autograd::TensorPtr polynorm3(
    const autograd::TensorPtr& tensor,
    const autograd::TensorPtr& weight,
    const autograd::TensorPtr& bias,
    float epsilon = 1e-5F);

// Composite-forward variant with the same autograd backward as polynorm3.
// Intended for parity checks and experimentation.
autograd::TensorPtr polynorm3_composite(
    const autograd::TensorPtr& tensor,
    const autograd::TensorPtr& weight,
    const autograd::TensorPtr& bias,
    float epsilon = 1e-5F);

}  // namespace ttml::ops
