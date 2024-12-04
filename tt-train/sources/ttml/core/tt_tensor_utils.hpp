// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <core/ttnn_all_includes.hpp>
#include <core/xtensor_utils.hpp>
#include <ttnn/tensor/enum_types.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <vector>

#include "core/distributed_mapping.hpp"

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

template <class VectorType = float, DataType TensorType = DataType::BFLOAT16>
[[nodiscard]] tt::tt_metal::Tensor from_xtensors_to_host(
    const std::vector<xt::xarray<VectorType>>& buffers, const std::unordered_map<std::string, std::string>& config);

template <class T = float>
[[nodiscard]] std::vector<T> to_vector(const tt::tt_metal::Tensor& tensor);

[[nodiscard]] bool is_tensor_initialized(const tt::tt_metal::Tensor& tensor);

[[nodiscard]] ttnn::Shape create_shape(const std::array<uint32_t, 4>& args);

template <class T = float, DataType TensorType = DataType::BFLOAT16>
[[nodiscard]] tt::tt_metal::Tensor from_xtensor(
    const xt::xarray<T>& buffer, ttnn::distributed::MeshDevice* device, Layout layout = Layout::TILE) {
    auto shape = create_shape(get_shape_4d(buffer));
    return from_vector<T, TensorType>(xtensor_to_span(buffer), shape, device, layout);
}

template <class T = float>
[[nodiscard]] xt::xarray<T> to_xtensor(const tt::tt_metal::Tensor& tensor) {
    auto vec = to_vector<T>(tensor);
    auto shape = tensor.get_shape().logical_shape();  // TODO: check if this is correct shape
    return span_to_xtensor(vec, shape);
}

template <class T = float, template <class> class MeshToTensor = ConcatMeshToTensor>
auto to_xtensor(const tt::tt_metal::Tensor& tensor, const MeshToTensor<T>& composer) {
    auto cpu_tensor = tensor.cpu();
    cpu_tensor = cpu_tensor.to(Layout::ROW_MAJOR);

    return composer.compose(cpu_tensor);
}

template <class T = float>
tt::tt_metal::Tensor from_xtensor(
    const xt::xarray<T>& tensor,
    ttnn::distributed::MeshDevice* device,
    const TensorToMeshVariant<T>& composer,
    Layout layout = Layout::TILE) {
    auto sharded_tensors = std::visit([&tensor](auto&& arg) { return arg.map(tensor); }, composer);
    auto config = std::visit([](auto&& arg) { return arg.config(); }, composer);
    auto output = from_xtensors_to_host(sharded_tensors, composer.get_config());
    MemoryConfig output_mem_config{};
    output = ttnn::to_device(output, device, output_mem_config);
    if (layout == Layout::TILE) {
        output = ttnn::tilize_with_zero_padding(output, output_mem_config, std::nullopt, /* multicore */ true);
    }
    return output;
}

}  // namespace ttml::core
