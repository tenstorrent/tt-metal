// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

ttnn::Tensor silu_bw(const ttnn::Tensor& input_tensor, const ttnn::Tensor& dL_dout_tensor);

}  // namespace ttml::metal
