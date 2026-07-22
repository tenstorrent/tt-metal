// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "autograd/tensor.hpp"

namespace ttml::ops {

enum class PolyNorm3ForwardVariant : uint8_t {
    Fused,
    CompositeComparisonOnly,
};

enum class PolyNorm3BackwardVariant : uint8_t {
    Fused,
    CompositeComparisonOnly,
};

// PolyNorm3 activation with trainable coefficients for x^3, x^2, x terms plus bias.
// NOTE: This op is intentionally fixed to 3 terms ([1, 1, 1, 3] weight) for now.
// Generalizing to N terms is a possible future extension.
autograd::TensorPtr polynorm3(
    const autograd::TensorPtr& tensor,
    const autograd::TensorPtr& weight,
    const autograd::TensorPtr& bias,
    float epsilon = 1e-5F,
    PolyNorm3ForwardVariant forward_variant = PolyNorm3ForwardVariant::Fused,
    PolyNorm3BackwardVariant backward_variant = PolyNorm3BackwardVariant::Fused);

// Fully composite forward and backward (no fused kernels). Intended for parity checks and experimentation.
autograd::TensorPtr polynorm3_composite(
    const autograd::TensorPtr& tensor,
    const autograd::TensorPtr& weight,
    const autograd::TensorPtr& bias,
    float epsilon = 1e-5F);

}  // namespace ttml::ops
