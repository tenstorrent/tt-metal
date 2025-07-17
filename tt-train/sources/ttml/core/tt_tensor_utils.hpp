// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <core/ttnn_all_includes.hpp>
#include <core/xtensor_utils.hpp>
#include <ttnn/distributed/distributed_tensor.hpp>
#include <vector>

#include "core/distributed_mapping.hpp"
#include "fmt/color.h"

namespace ttml::core {

void print_tensor_stats(const tt::tt_metal::Tensor& tensor, const std::string& name);

tt::tt_metal::Tensor zeros_like(const tt::tt_metal::Tensor& tensor);
tt::tt_metal::Tensor ones_like(const tt::tt_metal::Tensor& tensor);

tt::tt_metal::Tensor empty(
    const ttnn::Shape& shape, ttnn::distributed::MeshDevice* device, const ttnn::MemoryConfig& memory_config);
tt::tt_metal::Tensor full(
    const ttnn::Shape& shape,
    float value,
    ttnn::distributed::MeshDevice* device,
    ttnn::DataType dtype = ttnn::DataType::BFLOAT16);
tt::tt_metal::Tensor zeros(
    const ttnn::Shape& shape, ttnn::distributed::MeshDevice* device, ttnn::DataType dtype = ttnn::DataType::BFLOAT16);
tt::tt_metal::Tensor ones(
    const ttnn::Shape& shape, ttnn::distributed::MeshDevice* device, ttnn::DataType dtype = ttnn::DataType::BFLOAT16);

template <class VectorType = float, ttnn::DataType TensorType = ttnn::DataType::BFLOAT16>
[[nodiscard]] tt::tt_metal::Tensor from_vector(
    const std::vector<VectorType>& buffer,
    const ttnn::Shape& shape,
    ttnn::distributed::MeshDevice* device,
    ttnn::Layout layout = ttnn::Layout::TILE,
    const ttnn::distributed::TensorToMesh* mesh_mapper = nullptr);

template <class T = float>
[[nodiscard]] std::vector<T> to_vector(const tt::tt_metal::Tensor& tensor) {
    return tensor.to_vector<T>();
}

[[nodiscard]] bool is_tensor_initialized(const tt::tt_metal::Tensor& tensor);

template <class T = float, ttnn::DataType TensorType = ttnn::DataType::BFLOAT16>
[[nodiscard]] tt::tt_metal::Tensor from_xtensor(
    const xt::xarray<T>& buffer,
    ttnn::distributed::MeshDevice* device,
    ttnn::Layout layout = ttnn::Layout::TILE,
    const ttnn::distributed::TensorToMesh* mesh_mapper = nullptr) {
    auto shape = ttnn::experimental::xtensor::get_shape_from_xarray(buffer);
    auto buffer_view = xtensor_to_span(buffer);
    return from_vector<T, TensorType>(
        std::vector<T>(buffer_view.begin(), buffer_view.end()), shape, device, layout, mesh_mapper);
}

template <class T = float>
[[nodiscard]] xt::xarray<T> to_xtensor(const tt::tt_metal::Tensor& tensor) {
    auto vec = tensor.to_vector<T>();
    const auto& shape = tensor.logical_shape();
    std::vector<size_t> shape_vec(shape.cbegin(), shape.cend());
    // adapt creates view of the vector, but return will copy this data anyway (by creation of xt::array)
    return xt::adapt(vec, shape_vec);
}

template <class T = float>
auto to_xtensor(const tt::tt_metal::Tensor& tensor, const MeshToXTensorVariant<T>& composer) {
    auto cpu_tensor = tensor.cpu();
    cpu_tensor = cpu_tensor.to_layout(ttnn::Layout::ROW_MAJOR);
    auto cpu_tensors = ttnn::distributed::get_device_tensors(cpu_tensor);
    std::vector<xt::xarray<T>> res;
    res.reserve(cpu_tensors.size());
    for (const auto& shard : cpu_tensors) {
        res.push_back(to_xtensor<T>(shard));
    }
    return std::visit([&res](auto&& arg) { return arg.compose(res); }, composer);
}

std::vector<std::span<std::byte>> get_bytes_from_cpu_tensor(ttnn::Tensor& cpu_tensor);

}  // namespace ttml::core
