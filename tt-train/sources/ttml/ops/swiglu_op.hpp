// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "autograd/tensor.hpp"
#include "metal/ops/swiglu_fw/swiglu_fw.hpp"

namespace ttml::ops {

// When path is nullopt, uses get_swiglu_path() (env TTML_SWIGLU_PATH).
autograd::TensorPtr swiglu(
    const autograd::TensorPtr& tensor,
    const autograd::TensorPtr& w1,
    const autograd::TensorPtr& w2,
    const autograd::TensorPtr& w3);

}  // namespace ttml::ops
