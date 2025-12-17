// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::test_utils {

// Pearson Correlation Coefficient for two float vectors
float pcc(const std::vector<float>& x, const std::vector<float>& y);

// Dispatches a series of elementwise arithmetic operations over a tensor to `cq_id`, according to the expression:
// `output_tensor = - 32 * (input_tensor) + 128`
Tensor dispatch_ops_to_device(Tensor input_tensor, QueueId cq_id);

}  // namespace ttnn::test_utils
