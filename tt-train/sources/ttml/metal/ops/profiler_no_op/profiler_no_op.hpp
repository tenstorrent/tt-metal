// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::profiler_no_op {

struct ProfilerNoopOperation {
    static ttnn::Tensor invoke(const ttnn::Tensor& input_tensor);
};

}  // namespace ttml::metal::ops::profiler_no_op
