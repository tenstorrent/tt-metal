// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::matmul_test::device {

// Test cases for different data format combinations
enum class TestCase {
    BF16_BF16 = 0,  // Both operands in BF16
    FP32_FP32 = 1,  // Both operands in FP32
    BF16_FP32 = 2,  // First operand BF16, second FP32
    FP32_BF16 = 3   // First operand FP32, second BF16
};

// Attributes for the matmul test operation
struct operation_attributes_t {
    TestCase test_case;
};

// Tensors required for matmul test
struct tensor_args_t {
    ttnn::Tensor input_a;
    ttnn::Tensor input_b;
};

// Output tensor specs and tensors
using spec_return_value_t = std::vector<ttnn::TensorSpec>;
using tensor_return_value_t = ttnn::Tensor;

}  // namespace ttml::metal::ops::matmul_test::device
