// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/matmul_test_device_operation_types.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::matmul_test {

struct MatmulTestOperation {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_a,
        const ttnn::Tensor& input_b,
        device::TestCase test_case = device::TestCase::BF16_BF16);
};

}  // namespace ttml::metal::ops::matmul_test
