// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

ttnn::Tensor profiler_no_op(const ttnn::Tensor& input_tensor, const std::string& identifier);

}  // namespace ttml::metal
