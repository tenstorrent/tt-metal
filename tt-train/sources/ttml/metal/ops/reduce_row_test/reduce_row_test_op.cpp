// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_row_test_op.hpp"

#include "device/reduce_row_test_op_device_operation.hpp"

namespace ttml::metal::ops::reduce_row_test_op {

ttnn::Tensor ReduceRowTestOperation::invoke(const ttnn::Tensor& first_input, const ttnn::Tensor& second_input) {
    return ttnn::prim::ttml_reduce_row_test_op(first_input, second_input);
}
}  // namespace ttml::metal::ops::reduce_row_test_op
