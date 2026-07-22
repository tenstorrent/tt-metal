// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <ttnn/tensor/tensor.hpp>

namespace ttml::ops {

ttnn::Tensor newtonschulz5(const ttnn::Tensor& G, int steps = 5, float eps = 1e-7f);

ttnn::Tensor newtonschulz(const ttnn::Tensor& G, int steps, float eps, float a, float b, float c);

}  // namespace ttml::ops
