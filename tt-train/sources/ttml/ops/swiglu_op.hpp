// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"
#include "metal/ops/swiglu_fw/swiglu_fw.hpp"

namespace ttml::ops {

// Re-export algorithm enum for convenience
using SwiGLUAlgorithm = metal::SwiGLUAlgorithm;

autograd::TensorPtr swiglu(
    const autograd::TensorPtr& tensor,
    const autograd::TensorPtr& w1,
    const autograd::TensorPtr& w2,
    const autograd::TensorPtr& w3,
    SwiGLUAlgorithm algorithm = SwiGLUAlgorithm::AUTO);

}  // namespace ttml::ops
