// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_row_test_op.hpp"

#include "device/reduce_row_test_op_device_operation.hpp"

namespace ttml::metal::ops::reduce_row_test_op {

ttnn::Tensor ReduceRowTestOperation::invoke(const ttnn::Tensor& input, const bool use_matmul) {
    return ttnn::prim::ttml_reduce_row_test_op(input, use_matmul);
}
}  // namespace ttml::metal::ops::reduce_row_test_op
