// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <core/ttnn_all_includes.hpp>

#include "autograd/tensor.hpp"

namespace ttml::ops {

autograd::TensorPtr matmul_op(
    const autograd::TensorPtr& a, const autograd::TensorPtr& b, bool transpose_a = false, bool transpose_b = false);

}  // namespace ttml::ops
