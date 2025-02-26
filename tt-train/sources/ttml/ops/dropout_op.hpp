// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "autograd/tensor.hpp"

namespace ttml::ops {

autograd::TensorPtr dropout(const autograd::TensorPtr& tensor, float probability, bool use_per_device_seed = true);

}  // namespace ttml::ops
