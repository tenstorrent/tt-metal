// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::reduce_row_test_op {

struct ReduceRowTestOperation {
    static ttnn::Tensor invoke(const ttnn::Tensor& input, const bool use_matmul = false);
};

}  // namespace ttml::metal::ops::reduce_row_test_op
