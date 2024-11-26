// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_tensor_utils.hpp"

#include <fmt/base.h>
#include <fmt/color.h>

#include <algorithm>
#include <core/ttnn_all_includes.hpp>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>

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
    auto tensor_shape = tensor.get_shape();
    auto tensor_vec = ttml::core::to_vector<T>(tensor);

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
tt::tt_metal::OwnedBuffer create_owned_buffer_from_vector_of_floats(
    const std::vector<float>& data, DataType data_type) {
    switch (data_type) {
        case DataType::BFLOAT8_B: {
            auto uint32_vector = pack_fp32_vec_as_bfp8_tiles(data, /*row_major_input=*/false, /*is_exp_a=*/false);
            return tt::tt_metal::owned_buffer::create<uint32_t>(std::move(uint32_vector));
        }
        case DataType::BFLOAT4_B: {
            auto uint32_vector = pack_fp32_vec_as_bfp4_tiles(data, /*row_major_input=*/false, /*is_exp_a=*/false);
            return tt::tt_metal::owned_buffer::create<uint32_t>(std::move(uint32_vector));
        }
        case DataType::FLOAT32: {
            auto data_copy = data;
            return tt::tt_metal::owned_buffer::create<float>(std::move(data_copy));
        }
        case DataType::BFLOAT16: {
            std::vector<bfloat16> bfloat16_data(data.size());
            std::transform(std::begin(data), std::end(data), std::begin(bfloat16_data), [](float value) {
                return bfloat16(value);
            });
            return tt::tt_metal::owned_buffer::create<bfloat16>(std::move(bfloat16_data));
        }
        default: {
            throw std::runtime_error("Cannot create a host buffer!");
        }
    }
}

template <typename T>
tt::tt_metal::Tensor ttml_create_owned_tensor(
    std::vector<T>&& data, const ttnn::Shape& shape, tt::tt_metal::DataType data_type, tt::tt_metal::Layout layout) {
    auto buffer = tt::tt_metal::owned_buffer::create(std::move(data));
    auto storage = OwnedStorage{std::move(buffer)};
    return {std::move(storage), shape, data_type, layout};
}

// TODO: optimize precomputing multipliers
template <class T = float, class InternalT = bfloat16>
std::vector<T> untile_tensor_to_vec(const tt::tt_metal::Tensor& cpu_tensor) {
    auto tiled_buffer = tt::tt_metal::host_buffer::get_as<InternalT>(cpu_tensor);
    auto untiled_shape = cpu_tensor.get_logical_shape();
    auto tiled_shape = cpu_tensor.get_padded_shape();

    // Calculate total size of the untiled tensor
    size_t total_size = untiled_shape.volume();

    std::vector<T> untiled_data(total_size);

    auto compute_flat_index = [](const std::vector<uint32_t>& indices, ttnn::SimpleShape& shape) -> uint32_t {
        uint32_t flat_index = 0;
        uint32_t multiplier = 1;
        for (int i = (int)indices.size() - 1; i >= 0; --i) {
            flat_index += indices[i] * multiplier;
            multiplier *= shape[i];
        }
        return flat_index;
    };

    std::vector<uint32_t> indices(tiled_shape.rank(), 0);

    for (size_t idx = 0; idx < total_size; ++idx) {
        uint32_t untiled_index = compute_flat_index(indices, untiled_shape);
        uint32_t tiled_index = compute_flat_index(indices, tiled_shape);
        if constexpr (std::is_same_v<InternalT, bfloat16>) {
            untiled_data[untiled_index] = tiled_buffer[tiled_index].to_float();
        } else {
            untiled_data[untiled_index] = tiled_buffer[tiled_index];
        }

        for (int dim = (int)tiled_shape.rank() - 1; dim >= 0; --dim) {
            if (++indices[dim] < untiled_shape[dim]) {
                break;
            }
            indices[dim] = 0;
        }
    }

    return untiled_data;
}

}  // namespace
namespace ttml::core {

tt::tt_metal::Tensor zeros_like(const tt::tt_metal::Tensor& tensor) {
    return ttnn::moreh_full_like(tensor, 0.F, tensor.get_dtype(), tensor.get_layout(), tensor.memory_config());
}

tt::tt_metal::Tensor ones_like(const tt::tt_metal::Tensor& tensor) {
    return ttnn::moreh_full_like(tensor, 1.F, tensor.get_dtype(), tensor.get_layout(), tensor.memory_config());
}

tt::tt_metal::Tensor empty(
    const ttnn::Shape& shape, ttnn::distributed::MeshDevice* device, const MemoryConfig& memory_config) {
    return ttnn::empty(shape, DataType::BFLOAT16, Layout::TILE, device, memory_config);
}

tt::tt_metal::Tensor full(
    const ttnn::Shape& shape, float value, ttnn::distributed::MeshDevice* device, DataType dtype) {
    auto padded = shape.with_tile_padding();
    // if the shape is not divisible by TILE_SIZE, we need to add padding
    if (padded[2] % ttnn::types::TILE_SIZE != 0 || padded[3] % ttnn::types::TILE_SIZE != 0) {
        int additional_padding_h =
            (ttnn::types::TILE_SIZE - (int)padded[2] % ttnn::types::TILE_SIZE) % ttnn::types::TILE_SIZE;
        int additional_padding_w =
            (ttnn::types::TILE_SIZE - (int)padded[3] % ttnn::types::TILE_SIZE) % ttnn::types::TILE_SIZE;
        auto padded_shape = ttnn::Shape(
            {shape[0], shape[1], shape[2], shape[3]},
            {
                padded[0],
                padded[1],
                (padded[2] + additional_padding_h),
                (padded[3] + additional_padding_w),
            });
        // temporary solution to avoid using the device, and use only MeshDevice in highlevel api
        return ttnn::full(padded_shape, value, dtype, Layout::TILE, std::ref(*device->get_device(0)));
    }
    // if not padding available, we can just create a tensor with the given shape
    return ttnn::full(shape, value, dtype, Layout::TILE, std::ref(*device->get_device(0)));
}

tt::tt_metal::Tensor zeros(const ttnn::Shape& shape, ttnn::distributed::MeshDevice* device, DataType dtype) {
    return core::full(shape, 0.F, device, dtype);
}

tt::tt_metal::Tensor ones(const ttnn::Shape& shape, ttnn::distributed::MeshDevice* device, DataType dtype) {
    return core::full(shape, 1.F, device, dtype);
}

template <>
tt::tt_metal::Tensor from_vector<float, DataType::BFLOAT16>(
    const std::vector<float>& buffer, const ttnn::Shape& shape, ttnn::distributed::MeshDevice* device, Layout layout) {
    assert(device != nullptr);
    const DataType data_type = DataType::BFLOAT16;
    MemoryConfig output_mem_config{};
    auto logical_shape = shape.logical_shape();
    size_t volume = logical_shape.volume();
    if (buffer.size() != volume) {
        throw std::logic_error(
            fmt::format("Current buffer size is {} different from shape volume {}", buffer.size(), volume));
    }
    auto owned_buffer = create_owned_buffer_from_vector_of_floats(buffer, data_type);
    // remove possible paddings from the shape (it conflicts with ROW MAJOR)
    auto output = tt::tt_metal::Tensor(OwnedStorage{owned_buffer}, logical_shape, data_type, Layout::ROW_MAJOR);

    auto to_device_even_fast = [&]() {
        output = ttnn::to_device(output, device, output_mem_config);
        if (layout == Layout::TILE) {
            output = ttnn::tilize_with_zero_padding(output, output_mem_config, std::nullopt, /* multicore */ true);
        }

        return output;
    };

    output = to_device_even_fast();

    return output;
}

// Workaround implementation due to issue with tilize for float32
// it is expected that tilize will be fixed in the after next tt-metal main update
template <>
tt::tt_metal::Tensor from_vector<float, DataType::FLOAT32>(
    const std::vector<float>& buffer, const ttnn::Shape& shape, ttnn::distributed::MeshDevice* device, Layout layout) {
    auto tensor = from_vector<float, DataType::BFLOAT16>(buffer, shape, device, layout);
    return ttnn::typecast(tensor, DataType::FLOAT32);
}

template <>
std::vector<float> to_vector<float>(const tt::tt_metal::Tensor& tensor) {
    auto cpu_tensor = tensor.cpu();
    cpu_tensor = cpu_tensor.to(Layout::ROW_MAJOR);
    if (cpu_tensor.get_dtype() == DataType::BFLOAT16) {
        return untile_tensor_to_vec<float, bfloat16>(cpu_tensor);
    }
    assert(cpu_tensor.get_dtype() == DataType::FLOAT32);
    return untile_tensor_to_vec<float, float>(cpu_tensor);
}

/*
From vector uint32 doesn't support tilize_with_zero_padding on device
*/
template <>
tt::tt_metal::Tensor from_vector<uint32_t, DataType::UINT32>(
    const std::vector<uint32_t>& buffer,
    const ttnn::Shape& shape,
    ttnn::distributed::MeshDevice* device,
    Layout layout) {
    MemoryConfig output_mem_config{};
    auto logical_shape = shape.logical_shape();
    auto volume = logical_shape.volume();
    if (buffer.size() != volume) {
        throw std::logic_error(
            fmt::format("Current buffer size is {} different from shape volume {}", buffer.size(), volume));
    }

    // remove possible paddings from the shape (it conflicts with ROW MAJOR)
    std::vector<uint32_t> buffer_copy = buffer;
    auto output = ttml_create_owned_tensor(std::move(buffer_copy), logical_shape, DataType::UINT32, Layout::ROW_MAJOR);
    if (device != nullptr) {
        if (layout != Layout::ROW_MAJOR) {
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
tt::tt_metal::Tensor from_vector<int32_t, DataType::INT32>(
    const std::vector<int32_t>& buffer,
    const ttnn::Shape& shape,
    ttnn::distributed::MeshDevice* device,
    Layout layout) {
    MemoryConfig output_mem_config{};
    auto logical_shape = shape.logical_shape();
    auto volume = logical_shape.volume();
    if (buffer.size() != volume) {
        throw std::logic_error(
            fmt::format("Current buffer size is {} different from shape volume {}", buffer.size(), volume));
    }

    // remove possible paddings from the shape (it conflicts with ROW MAJOR)
    std::vector<int32_t> buffer_copy = buffer;
    auto output = ttml_create_owned_tensor(std::move(buffer_copy), logical_shape, DataType::INT32, Layout::ROW_MAJOR);
    if (device != nullptr) {
        if (layout != Layout::ROW_MAJOR) {
            output = ttnn::to_layout(output, layout, std::nullopt, output_mem_config, device);
        }
        output = ttnn::to_device(output, device, output_mem_config);
    }

    return output;
}

template <>
std::vector<uint32_t> to_vector<uint32_t>(const tt::tt_metal::Tensor& tensor) {
    auto cpu_tensor = tensor.cpu();
    cpu_tensor = cpu_tensor.to(Layout::ROW_MAJOR);

    return untile_tensor_to_vec<uint32_t, uint32_t>(cpu_tensor);
}

template <>
std::vector<int32_t> to_vector<int32_t>(const tt::tt_metal::Tensor& tensor) {
    auto cpu_tensor = tensor.cpu();
    cpu_tensor = cpu_tensor.to(Layout::ROW_MAJOR);

    return untile_tensor_to_vec<int32_t, int32_t>(cpu_tensor);
}

bool is_tensor_initialized(const tt::tt_metal::Tensor& tensor) {
    return tensor.tensor_attributes != nullptr;
}

ttnn::Shape create_shape(const std::array<uint32_t, 4>& args) {
    return ttnn::Shape{args};
}

void print_tensor_stats(const tt::tt_metal::Tensor& tensor, const std::string& name) {
    if (tensor.get_dtype() == DataType::BFLOAT16 || tensor.get_dtype() == DataType::FLOAT32) {
        print_tensor_stats_<float>(tensor, name);
    } else {
        print_tensor_stats_<uint32_t>(tensor, name);
    }
}

}  // namespace ttml::core
