// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor_layout.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace test_utils {
void test_tensor_on_device(const ttnn::SimpleShape& input_shape, const tt::tt_metal::TensorLayout& layout, tt::tt_metal::Device* device);
void test_tensor_on_device(const ttnn::SimpleShape& input_shape, const tt::tt_metal::TensorLayout& layout);
}
