// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
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

// copypaste from deprecated tensor pybinds ttnn
tt::tt_metal::HostBuffer create_owned_buffer_from_vector_of_floats(
    const std::vector<float>& data, ttnn::DataType data_type) {
    switch (data_type) {
        case ttnn::DataType::BFLOAT8_B: {
            auto uint32_vector = pack_fp32_vec_as_bfp8_tiles(data, /*row_major_input=*/false, /*is_exp_a=*/false);
            return tt::tt_metal::HostBuffer(std::move(uint32_vector));
        }
        case ttnn::DataType::BFLOAT4_B: {
            auto uint32_vector = pack_fp32_vec_as_bfp4_tiles(data, /*row_major_input=*/false, /*is_exp_a=*/false);
            return tt::tt_metal::HostBuffer(std::move(uint32_vector));
        }
        case ttnn::DataType::FLOAT32: {
            auto data_copy = data;
            return tt::tt_metal::HostBuffer(std::move(data_copy));
        }
        case ttnn::DataType::BFLOAT16: {
            std::vector<bfloat16> bfloat16_data(data.size());
            std::transform(std::begin(data), std::end(data), std::begin(bfloat16_data), [](float value) {
                return bfloat16(value);
            });
            return tt::tt_metal::HostBuffer(std::move(bfloat16_data));
        }
        default: {
            throw std::runtime_error("Cannot create a host buffer!");
        }
    }
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
                return {storage.buffer};
            } else if constexpr (std::is_same_v<StorageType, tt::tt_metal::MultiDeviceHostStorage>) {
                std::vector<tt::tt_metal::HostBuffer> buffers;
                buffers.reserve(storage.distributed_buffer().shard_coords().size());
                storage.distributed_buffer().apply(
                    [&buffers](const tt::tt_metal::HostBuffer& shard) { buffers.push_back(shard); });
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

template <class T, ttnn::DataType TensorType>
[[nodiscard]] tt::tt_metal::Tensor from_xtensors_to_host(
    const std::vector<xt::xarray<T>>& buffers, const std::unordered_map<std::string, std::string>& config) {
    std::vector<tt::tt_metal::HostBuffer> host_owned_buffers;
    host_owned_buffers.reserve(buffers.size());
    if (buffers.empty()) {
        throw std::runtime_error("Cannot create a host buffer from an empty vector of xtensors!");
    }

    auto first_shape = buffers.front().shape();
    for (int i = 0; i < buffers.size(); ++i) {
        if (buffers[i].shape() != first_shape) {
            throw std::runtime_error(fmt::format(
                "Cannot create a host buffer from xtensors with different shapes: {} vs {}!",
                ttnn::experimental::xtensor::get_shape_from_xarray(buffers[0]),
                ttnn::experimental::xtensor::get_shape_from_xarray(buffers[i])));
        }
    }
    auto tensor_spec = ttnn::TensorSpec(
        ttnn::experimental::xtensor::get_shape_from_xarray(buffers.front()),
        ttnn::TensorLayout(TensorType, ttnn::PageConfig(ttnn::Layout::ROW_MAJOR), ttnn::MemoryConfig{}));

    for (const auto& buffer : buffers) {
        if constexpr (std::is_same_v<T, float>) {
            auto owned_buffer =
                create_owned_buffer_from_vector_of_floats(std::vector<T>(buffer.begin(), buffer.end()), TensorType);
            host_owned_buffers.push_back(std::move(owned_buffer));
        } else {
            auto owned_buffer = tt::tt_metal::HostBuffer(std::vector<T>(buffer.begin(), buffer.end()));
            host_owned_buffers.push_back(std::move(owned_buffer));
        }
    }

    return ttnn::Tensor(
        tt::tt_metal::MultiDeviceHostStorage(std::move(host_owned_buffers)),
        tensor_spec,
        tt::tt_metal::get_distributed_tensor_config(config));
}

template tt::tt_metal::Tensor from_xtensors_to_host<float, ttnn::DataType::BFLOAT16>(
    const std::vector<xt::xarray<float>>& buffers, const std::unordered_map<std::string, std::string>& config);
template tt::tt_metal::Tensor from_xtensors_to_host<uint32_t, ttnn::DataType::UINT32>(
    const std::vector<xt::xarray<uint32_t>>& buffers, const std::unordered_map<std::string, std::string>& config);
template tt::tt_metal::Tensor from_xtensors_to_host<int32_t, tt::tt_metal::DataType::INT32>(
    const std::vector<xt::xarray<int32_t>>& buffers, const std::unordered_map<std::string, std::string>& config);

template <>
tt::tt_metal::Tensor from_vector<float, ttnn::DataType::BFLOAT16>(
    const std::vector<float>& buffer,
    const ttnn::Shape& shape,
    ttnn::distributed::MeshDevice* device,
    ttnn::Layout layout) {
    assert(device != nullptr);
    const ttnn::DataType data_type = ttnn::DataType::BFLOAT16;
    ttnn::MemoryConfig output_mem_config{};
    size_t volume = shape.volume();
    if (buffer.size() != volume) {
        throw std::logic_error(
            fmt::format("Current buffer size is {} different from shape volume {}", buffer.size(), volume));
    }
    auto owned_buffer = create_owned_buffer_from_vector_of_floats(buffer, data_type);
    // remove possible paddings from the shape (it conflicts with ROW MAJOR)
    auto output = tt::tt_metal::Tensor(std::move(owned_buffer), shape, data_type, ttnn::Layout::ROW_MAJOR);

    const size_t MAX_TILE_DIMENSION = 16384;
    // Temporary workaround for the issue with tilize for large size
    // https://github.com/tenstorrent/tt-metal/issues/15950
    if (shape[-1] >= MAX_TILE_DIMENSION && layout == ttnn::Layout::TILE) {
        output = ttnn::to_layout(output, ttnn::Layout::TILE, std::nullopt, output_mem_config, device);
        output = ttnn::to_device(output, device, output_mem_config);
    } else {
        output = ttnn::to_device(output, device, output_mem_config);
        if (layout == ttnn::Layout::TILE) {
            output = ttnn::tilize_with_zero_padding(output, output_mem_config, std::nullopt, /* multicore */ true);
        }
    }

    return output;
}

// Workaround implementation due to issue with tilize for float32
// it is expected that tilize will be fixed in the after next tt-metal main update
template <>
tt::tt_metal::Tensor from_vector<float, ttnn::DataType::FLOAT32>(
    const std::vector<float>& buffer,
    const ttnn::Shape& shape,
    ttnn::distributed::MeshDevice* device,
    ttnn::Layout layout) {
    auto tensor = from_vector<float, ttnn::DataType::BFLOAT16>(buffer, shape, device, layout);
    return ttnn::typecast(tensor, ttnn::DataType::FLOAT32);
}

/*
From vector uint32 doesn't support tilize_with_zero_padding on device
*/
template <>
tt::tt_metal::Tensor from_vector<uint32_t, ttnn::DataType::UINT32>(
    const std::vector<uint32_t>& buffer,
    const ttnn::Shape& shape,
    ttnn::distributed::MeshDevice* device,
    ttnn::Layout layout) {
    ttnn::MemoryConfig output_mem_config{};
    auto volume = shape.volume();
    if (buffer.size() != volume) {
        throw std::logic_error(
            fmt::format("Current buffer size is {} different from shape volume {}", buffer.size(), volume));
    }

    // remove possible paddings from the shape (it conflicts with ROW MAJOR)
    std::vector<uint32_t> buffer_copy = buffer;
    auto output =
        ttml_create_owned_tensor(std::move(buffer_copy), shape, ttnn::DataType::UINT32, ttnn::Layout::ROW_MAJOR);
    if (device != nullptr) {
        if (layout != ttnn::Layout::ROW_MAJOR) {
            output = ttnn::to_layout(output, layout, std::nullopt, output_mem_config, device);
        }
        output = ttnn::to_device(output, device, output_mem_config);
    }

    return output;
}

/*
From vector int32 doesn't support tilize_with_zero_padding on device
*/
template <>
tt::tt_metal::Tensor from_vector<int32_t, ttnn::DataType::INT32>(
    const std::vector<int32_t>& buffer,
    const ttnn::Shape& shape,
    ttnn::distributed::MeshDevice* device,
    ttnn::Layout layout) {
    ttnn::MemoryConfig output_mem_config{};
    auto volume = shape.volume();
    if (buffer.size() != volume) {
        throw std::logic_error(
            fmt::format("Current buffer size is {} different from shape volume {}", buffer.size(), volume));
    }

    // remove possible paddings from the shape (it conflicts with ROW MAJOR)
    std::vector<int32_t> buffer_copy = buffer;
    auto output =
        ttml_create_owned_tensor(std::move(buffer_copy), shape, ttnn::DataType::INT32, ttnn::Layout::ROW_MAJOR);
    if (device != nullptr) {
        if (layout != ttnn::Layout::ROW_MAJOR) {
            output = ttnn::to_layout(output, layout, std::nullopt, output_mem_config, device);
        }
        output = ttnn::to_device(output, device, output_mem_config);
    }

    return output;
}

bool is_tensor_initialized(const tt::tt_metal::Tensor& tensor) {
    return tensor.tensor_attributes != nullptr;
}

ttnn::Shape create_shape(const std::array<uint32_t, 4>& args) {
    return ttnn::Shape{args};
}

void print_tensor_stats(const tt::tt_metal::Tensor& tensor, const std::string& name) {
    if (tensor.dtype() == ttnn::DataType::BFLOAT16 || tensor.dtype() == ttnn::DataType::FLOAT32) {
        print_tensor_stats_<float>(tensor, name);
    } else {
        print_tensor_stats_<uint32_t>(tensor, name);
    }
}

template <class T, ttnn::DataType TensorType>
tt::tt_metal::Tensor from_xtensor(
    const xt::xarray<T>& tensor,
    ttnn::distributed::MeshDevice* device,
    const XTensorToMeshVariant<T>& composer,
    ttnn::Layout layout) {
    auto sharded_tensors = std::visit([&tensor](auto&& arg) { return arg.map(tensor); }, composer);
    auto config = std::visit([](auto&& arg) { return arg.config(); }, composer);
    auto output = from_xtensors_to_host<T, TensorType>(sharded_tensors, config);
    ttnn::MemoryConfig output_mem_config{};

    if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>) {
        if (layout != ttnn::Layout::ROW_MAJOR) {
            output = ttnn::to_layout(output, layout, std::nullopt, output_mem_config, device);
        }
        output = ttnn::to_device(output, device, output_mem_config);
    } else {
        output = ttnn::to_device(output, device, output_mem_config);
        if (layout == ttnn::Layout::TILE) {
            output = ttnn::tilize_with_zero_padding(output, output_mem_config, std::nullopt, /* multicore */ true);
        }
    }
    return output;
}

template tt::tt_metal::Tensor from_xtensor<float, ttnn::DataType::BFLOAT16>(
    const xt::xarray<float>& tensor,
    ttnn::distributed::MeshDevice* device,
    const XTensorToMeshVariant<float>& composer,
    ttnn::Layout layout);

template tt::tt_metal::Tensor from_xtensor<int32_t, ttnn::DataType::INT32>(
    const xt::xarray<int32_t>& tensor,
    ttnn::distributed::MeshDevice* device,
    const XTensorToMeshVariant<int32_t>& composer,
    ttnn::Layout layout);

template tt::tt_metal::Tensor from_xtensor<uint32_t, ttnn::DataType::UINT32>(
    const xt::xarray<uint32_t>& tensor,
    ttnn::distributed::MeshDevice* device,
    const XTensorToMeshVariant<uint32_t>& composer,
    ttnn::Layout layout);

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
