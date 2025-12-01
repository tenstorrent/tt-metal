// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_test.hpp"

#include "device/matmul_test_device_operation.hpp"

namespace ttml::metal::ops::matmul_test {

ttnn::Tensor MatmulTestOperation::invoke(
    const ttnn::Tensor& input_a, const ttnn::Tensor& input_b, device::TestCase test_case) {
    return ttnn::prim::ttml_matmul_test(input_a, input_b, test_case);
}

}  // namespace ttml::metal::ops::matmul_test
