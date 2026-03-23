// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <ttnn/tensor/tensor.hpp>

namespace ttml::ops {

tt::tt_metal::Tensor newtonschulz5(const tt::tt_metal::Tensor& G, int steps = 5, float eps = 1e-7f);

tt::tt_metal::Tensor newtonschulz(const tt::tt_metal::Tensor& G, int steps, float eps, float a, float b, float c);

}  // namespace ttml::ops
