// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <span>

#include "autograd/tensor.hpp"

namespace ttml::ops {

autograd::TensorPtr reshape(const autograd::TensorPtr& tensor, std::span<uint32_t> shape);

}  // namespace ttml::ops
