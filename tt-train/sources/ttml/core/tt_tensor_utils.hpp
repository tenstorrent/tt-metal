// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <core/ttnn_all_includes.hpp>
#include <initializer_list>
#include <vector>

namespace ttml::core {

void print_tensor_stats(const tt::tt_metal::Tensor& tensor, const std::string& name);

tt::tt_metal::Tensor zeros_like(const tt::tt_metal::Tensor& tensor);
tt::tt_metal::Tensor ones_like(const tt::tt_metal::Tensor& tensor);

tt::tt_metal::Tensor empty(
    const ttnn::Shape& shape, ttnn::distributed::MeshDevice* device, const MemoryConfig& memory_config);
tt::tt_metal::Tensor full(
    const ttnn::Shape& shape, float value, ttnn::distributed::MeshDevice* device, DataType dtype = DataType::BFLOAT16);
tt::tt_metal::Tensor zeros(
    const ttnn::Shape& shape, ttnn::distributed::MeshDevice* device, DataType dtype = DataType::BFLOAT16);
tt::tt_metal::Tensor ones(
    const ttnn::Shape& shape, ttnn::distributed::MeshDevice* device, DataType dtype = DataType::BFLOAT16);

template <class VectorType = float, DataType TensorType = DataType::BFLOAT16>
[[nodiscard]] tt::tt_metal::Tensor from_vector(
    const std::vector<VectorType>& buffer,
    const ttnn::Shape& shape,
    ttnn::distributed::MeshDevice* device,
    Layout layout = Layout::TILE);

template <class T = float>
[[nodiscard]] std::vector<T> to_vector(const tt::tt_metal::Tensor& tensor);

[[nodiscard]] bool is_tensor_initialized(const tt::tt_metal::Tensor& tensor);

[[nodiscard]] ttnn::Shape create_shape(const std::array<uint32_t, 4>& args);

}  // namespace ttml::core
