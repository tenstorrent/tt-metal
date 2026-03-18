// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

ttnn::Tensor polynorm_fw(
    const ttnn::Tensor& input_tensor, float w0, float w1, float w2, float bias, float epsilon = 1e-5F);

}  // namespace ttml::metal
