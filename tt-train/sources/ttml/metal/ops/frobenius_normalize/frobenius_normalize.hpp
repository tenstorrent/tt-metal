// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

ttnn::Tensor frobenius_normalize(const ttnn::Tensor& input_tensor, float epsilon = 1e-7F);

}  // namespace ttml::metal
