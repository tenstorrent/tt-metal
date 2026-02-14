// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_tensor_utils.hpp"

#include <fmt/base.h>
#include <fmt/color.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <ttnn/tensor/types.hpp>

#include "core/xtensor_utils.hpp"

namespace {

template <typename T>
T get_median(std::vector<T>& vec) {
    assert(!vec.empty());
    std::nth_element(vec.begin(), vec.begin() + vec.size() / 2, vec.end());
    if (vec.size() & 1U) {
        return vec[vec.size() / 2];
    }
    auto neighbor = *std::max_element(vec.begin(), vec.begin() + vec.size() / 2);
    return std::midpoint(neighbor, vec[vec.size() / 2]);
};

template <typename T>
void print_tensor_stats_(const tt::tt_metal::Tensor& tensor, const std::string& name) {
    auto tensor_shape = tensor.logical_shape();
    auto tensor_vec = tensor.to_vector<T>();

    auto median = get_median(tensor_vec);
    auto mean = std::accumulate(tensor_vec.begin(), tensor_vec.end(), 0.F) / static_cast<float>(tensor_vec.size());
    auto mean_sq =
        std::accumulate(
            tensor_vec.begin(), tensor_vec.end(), 0.F, [](float acc, float val) { return acc + val * val; }) /
        static_cast<float>(tensor_vec.size());
    auto variance = mean_sq - mean * mean;

    fmt::print(
        "{}: shape: {} min: {} max: {} median: {} mean: {} variance: {}\n",
        name,
        tensor_shape,
        *std::min_element(tensor_vec.begin(), tensor_vec.end()),
        *std::max_element(tensor_vec.begin(), tensor_vec.end()),
        median,
        mean,
        variance);
}

template <typename T>
tt::tt_metal::Tensor ttml_create_owned_tensor(
    std::vector<T>&& data, const ttnn::Shape& shape, tt::tt_metal::DataType data_type, tt::tt_metal::Layout layout) {
    auto buffer = tt::tt_metal::HostBuffer(std::move(data));
    return {std::move(buffer), shape, data_type, layout};
}

std::vector<tt::tt_metal::HostBuffer> get_as(const ttnn::Tensor& tensor) {
    return std::visit(
        [](auto&& storage) -> std::vector<tt::tt_metal::HostBuffer> {
            using StorageType = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<StorageType, tt::tt_metal::HostStorage>) {
                std::vector<tt::tt_metal::HostBuffer> buffers;
                buffers.reserve(storage.buffer().shard_coords().size());
                storage.buffer().apply([&buffers](const tt::tt_metal::HostBuffer& shard) { buffers.push_back(shard); });
                return buffers;
            } else {
                throw std::runtime_error("Tensor must be on host");
            }
        },
        tensor.storage());
}

}  // namespace
namespace ttml::core {

tt::tt_metal::Tensor zeros_like(const tt::tt_metal::Tensor& tensor) {
    return ttnn::moreh_full_like(tensor, 0.F, tensor.dtype(), tensor.layout(), tensor.memory_config());
}

tt::tt_metal::Tensor ones_like(const tt::tt_metal::Tensor& tensor) {
    return ttnn::moreh_full_like(tensor, 1.F, tensor.dtype(), tensor.layout(), tensor.memory_config());
}

tt::tt_metal::Tensor empty(
    const ttnn::Shape& shape, ttnn::distributed::MeshDevice* device, const ttnn::MemoryConfig& memory_config) {
    return ttnn::empty(shape, ttnn::DataType::BFLOAT16, ttnn::Layout::TILE, device, memory_config);
}

tt::tt_metal::Tensor full(
    const ttnn::Shape& shape, float value, ttnn::distributed::MeshDevice* device, ttnn::DataType dtype) {
    return ttnn::full(shape, value, dtype, ttnn::Layout::TILE, std::ref(*device));
}

tt::tt_metal::Tensor zeros(const ttnn::Shape& shape, ttnn::distributed::MeshDevice* device, ttnn::DataType dtype) {
    return core::full(shape, 0.F, device, dtype);
}

tt::tt_metal::Tensor ones(const ttnn::Shape& shape, ttnn::distributed::MeshDevice* device, ttnn::DataType dtype) {
    return core::full(shape, 1.F, device, dtype);
}
template <class VectorType, ttnn::DataType TensorType>
tt::tt_metal::Tensor from_vector(
    const std::vector<VectorType>& buffer,
    const ttnn::Shape& shape,
    ttnn::distributed::MeshDevice* device,
    ttnn::Layout layout,
    const ttnn::distributed::TensorToMesh* mesh_mapper) {
    if (device == nullptr) {
        throw std::runtime_error("from_vector: device == nullptr. Device is required");
    }

    ttnn::MemoryConfig output_mem_config{};
    size_t volume = shape.volume();
    if (buffer.size() != volume) {
        throw std::logic_error(
            fmt::format("Current buffer size is {} different from shape volume {}", buffer.size(), volume));
    }

    const auto tensor_layout =
        ttnn::TensorLayout(TensorType, ttnn::PageConfig(ttnn::Layout::ROW_MAJOR), tt::tt_metal::MemoryConfig{});

    ttnn::Tensor output;
    if (mesh_mapper != nullptr) {
        output = ttnn::distributed::create_distributed_tensor(
            ttsl::make_const_span(buffer), shape, tensor_layout, *mesh_mapper);
    } else {
        output = ttnn::Tensor::from_vector(buffer, ttnn::TensorSpec(shape, tensor_layout));
    }

    if constexpr (TensorType == ttnn::DataType::FLOAT32) {
        output = ttnn::to_layout(output, ttnn::Layout::TILE);
        output = ttnn::to_device(output, device, output_mem_config);
    } else {
        output = ttnn::to_device(output, device, output_mem_config);
        output = ttnn::tilize_with_zero_padding(output, output_mem_config, std::nullopt, /* multicore */ true);
    }

    return output;
}

template tt::tt_metal::Tensor from_vector<bfloat16, ttnn::DataType::BFLOAT16>(
    const std::vector<bfloat16>&,
    const ttnn::Shape&,
    ttnn::distributed::MeshDevice*,
    ttnn::Layout,
    const ttnn::distributed::TensorToMesh*);
template tt::tt_metal::Tensor from_vector<float, ttnn::DataType::BFLOAT16>(
    const std::vector<float>&,
    const ttnn::Shape&,
    ttnn::distributed::MeshDevice*,
    ttnn::Layout,
    const ttnn::distributed::TensorToMesh*);
template tt::tt_metal::Tensor from_vector<float, ttnn::DataType::FLOAT32>(
    const std::vector<float>&,
    const ttnn::Shape&,
    ttnn::distributed::MeshDevice*,
    ttnn::Layout,
    const ttnn::distributed::TensorToMesh*);
template tt::tt_metal::Tensor from_vector<uint32_t, ttnn::DataType::UINT32>(
    const std::vector<uint32_t>&,
    const ttnn::Shape&,
    ttnn::distributed::MeshDevice*,
    ttnn::Layout,
    const ttnn::distributed::TensorToMesh*);
template tt::tt_metal::Tensor from_vector<int32_t, ttnn::DataType::INT32>(
    const std::vector<int32_t>&,
    const ttnn::Shape&,
    ttnn::distributed::MeshDevice*,
    ttnn::Layout,
    const ttnn::distributed::TensorToMesh*);

bool is_tensor_initialized(const tt::tt_metal::Tensor& tensor) {
    return tensor.tensor_attributes != nullptr;
}

void print_tensor_stats(const tt::tt_metal::Tensor& tensor, const std::string& name) {
    if (tensor.dtype() == ttnn::DataType::BFLOAT16 || tensor.dtype() == ttnn::DataType::FLOAT32) {
        print_tensor_stats_<float>(tensor, name);
    } else {
        print_tensor_stats_<uint32_t>(tensor, name);
    }
}

std::vector<std::span<std::byte>> get_bytes_from_cpu_tensor(ttnn::Tensor& tensor) {
    std::vector<std::span<std::byte>> res;
    auto cpu_tensor = tensor;
    auto buffers = get_as(cpu_tensor);

    res.reserve(buffers.size());
    for (auto& buffer : buffers) {
        auto view = buffer.view_bytes();
        auto span = std::as_writable_bytes(std::span{view.begin(), view.end()});
        res.push_back(span);
    }
    return res;
}
}  // namespace ttml::core
