// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <core/ttnn_all_includes.hpp>
#include <core/xtensor_utils.hpp>
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
    ttnn::distributed::MeshDevice,
    const std::optional<DistributedTensorConfig> config = std::nullopt);

template <class T = float>
[[nodiscard]] std::vector<T> to_vector(const tt::tt_metal::Tensor& tensor);

[[nodiscard]] bool is_tensor_initialized(const tt::tt_metal::Tensor& tensor);

[[nodiscard]] ttnn::Shape create_shape(const std::array<uint32_t, 4>& args);

template <
    class T = float,
    xt::layout_type layout_type = xt::layout_type::dynamic,
    DataType TensorType = DataType::BFLOAT16>
[[nodiscard]] tt::tt_metal::Tensor from_xtensor(
    const xt::xarray<T, layout_type>& buffer, ttnn::distributed::MeshDevice* device, Layout layout = Layout::TILE) {
    auto shape = create_shape(get_shape_4d(buffer));
    return from_vector<T, TensorType>(xtensor_to_span(buffer), shape, device, layout);
}

template <class T = float, xt::layout_type layout_type = xt::layout_type::dynamic>
[[nodiscard]] xt::xarray<T, layout_type> to_xtensor(const tt::tt_metal::Tensor& tensor) {
    auto vec = to_vector<T>(tensor);
    auto shape = tensor.get_shape().logical_shape();  // TODO: check if this is correct shape
    return span_to_xtensor(vec, shape);
}

}  // namespace ttml::core
