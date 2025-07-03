// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt {
namespace tt_metal {
class IDevice;
class Shape;
class TensorLayout;
}  // namespace tt_metal
}  // namespace tt

namespace test_utils {
void test_tensor_on_device(
    const ttnn::Shape& input_shape,
    const tt::tt_metal::TensorLayout& layout,
    tt::tt_metal::distributed::MeshDevice* device);
void test_tensor_on_device(const ttnn::Shape& input_shape, const tt::tt_metal::TensorLayout& layout);
}  // namespace test_utils
