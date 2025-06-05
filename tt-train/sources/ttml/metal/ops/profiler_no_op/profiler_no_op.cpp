// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "profiler_no_op.hpp"

#include "device/profiler_no_op_device_operation.hpp"

namespace ttml::metal::ops::profiler_no_op {

ttnn::Tensor ProfilerNoopOperation::invoke(const ttnn::Tensor& input_tensor) {
    return ttnn::prim::ttml_profiler_no_op(input_tensor);
}
}  // namespace ttml::metal::ops::profiler_no_op
