// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "helper_funcs.h"

#include <fstream>
#include <memory>
#include <random>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "ttnn/distributed/api.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn_mobilenetv2.h"
#include <tt-metalium/bfloat4.hpp>
#include <tt-metalium/bfloat8.hpp>

namespace {

ttnn::Tensor create_tensor_from_float_data(
    const std::vector<float>& data,
    const std::vector<uint32_t>& shape,
    std::optional<ttnn::DataType> dtype,
    std::optional<ttnn::Layout> layout) {
    auto data_type = dtype.value_or(ttnn::DataType::FLOAT32);
    auto tensor_spec = tt::tt_metal::TensorSpec(
        ttnn::Shape(shape),
        tt::tt_metal::TensorLayout(
            data_type, tt::tt_metal::PageConfig(layout.value_or(ttnn::Layout::ROW_MAJOR)), ttnn::MemoryConfig{}));

    switch (data_type) {
        case ttnn::DataType::FLOAT32:
        case ttnn::DataType::BFLOAT8_B:
        case ttnn::DataType::BFLOAT4_B:
            return ttnn::Tensor::template from_span<float>(
                tt::stl::Span<const float>(data.data(), data.size()), tensor_spec);
        case ttnn::DataType::BFLOAT16: {
            std::vector<bfloat16> converted(data.begin(), data.end());
            return ttnn::Tensor::template from_span<bfloat16>(
                tt::stl::Span<const bfloat16>(converted.data(), converted.size()), tensor_spec);
        }
        default: TT_THROW("Unsupported DataType for float-backed tensor: {}", static_cast<int>(data_type));
    }
}

std::vector<float> load_binary_tensor(const std::string& file_path, std::size_t expected_numel) {
    std::ifstream input(file_path, std::ios::binary | std::ios::ate);
    TT_FATAL(input.is_open(), "Failed to open tensor file {}", file_path);
    const auto file_size = static_cast<std::size_t>(input.tellg());
    input.seekg(0, std::ios::beg);

    const auto expected_size = expected_numel * sizeof(float);
    TT_FATAL(
        file_size == expected_size, "Tensor file {} has {} bytes, expected {}", file_path, file_size, expected_size);

    std::vector<float> data(expected_numel);
    input.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(expected_size));
    TT_FATAL(input.good(), "Failed reading tensor file {}", file_path);
    return data;
}

std::size_t tensor_numel(const std::vector<uint32_t>& shape) {
    std::size_t numel = 1;
    for (auto dim : shape) {
        numel *= dim;
    }
    return numel;
}

MobileNetV2HostInput make_host_input(uint32_t batch, uint32_t channels, uint32_t height, uint32_t width) {
    MobileNetV2HostInput input;
    input.batch = batch;
    input.channels = channels;
    input.height = height;
    input.width = width;
    input.nhwc_flattened_data.resize(static_cast<std::size_t>(batch) * height * width * channels);
    return input;
}

}  // namespace

MobileNetV2HostInput create_mobilenetv2_host_input(int batch, int input_channels, int input_height, int input_width) {
    MobileNetV2HostInput input = make_host_input(batch, input_channels, input_height, input_width);
    std::mt19937 generator(0);
    std::normal_distribution<float> distribution(0.0f, 1.0f);

    for (int n = 0; n < batch; ++n) {
        for (int h = 0; h < input_height; ++h) {
            for (int w = 0; w < input_width; ++w) {
                for (int c = 0; c < input_channels; ++c) {
                    const auto nhwc_index = static_cast<std::size_t>(n) * input_height * input_width * input_channels +
                                            static_cast<std::size_t>(h) * input_width * input_channels +
                                            static_cast<std::size_t>(w) * input_channels + static_cast<std::size_t>(c);
                    input.nhwc_flattened_data[nhwc_index] = distribution(generator);
                }
            }
        }
    }

    return input;
}

ttnn::Tensor host_input_to_ttnn(const MobileNetV2HostInput& input) {
    return create_tensor_from_float_data(
        input.nhwc_flattened_data,
        {1, 1, input.batch * input.height * input.width, input.channels},
        ttnn::DataType::BFLOAT16,
        ttnn::Layout::ROW_MAJOR);
}

std::unordered_map<std::string, ttnn::Tensor> create_mobilenetv2_model_parameters(
    const std::string& weights_dir, const std::shared_ptr<ttnn::MeshDevice>& device) {
    TT_FATAL(!weights_dir.empty(), "weights_dir cannot be empty!");

    std::ifstream manifest_stream(fmt::format("{}/manifest.json", weights_dir));
    TT_FATAL(manifest_stream.is_open(), "Failed to open manifest.json in {}", weights_dir);

    nlohmann::json manifest = nlohmann::json::parse(manifest_stream);
    std::unordered_map<std::string, ttnn::Tensor> model_parameters;

    for (const auto& [name, tensor_info] : manifest.at("tensors").items()) {
        auto shape = tensor_info.at("shape").get<std::vector<uint32_t>>();
        auto file_name = tensor_info.at("file").get<std::string>();
        auto layout_name = tensor_info.value("layout", std::string("ROW_MAJOR"));
        auto layout = layout_name == "TILE" ? ttnn::Layout::TILE : ttnn::Layout::ROW_MAJOR;
        auto tensor_data = load_binary_tensor(fmt::format("{}/{}", weights_dir, file_name), tensor_numel(shape));

        auto tensor = create_tensor_from_float_data(tensor_data, shape, ttnn::DataType::FLOAT32, layout);
        if (name == "classifier_1_weight" || name == "classifier_1_bias") {
            tensor = tensor.to_device(device.get());
        }
        model_parameters[name] = tensor;
    }

    return model_parameters;
}

uint32_t get_ttbuffer_address(const ttnn::Tensor& tensor) {
    if (tensor.storage_type() != ttnn::StorageType::DEVICE) {
        TT_THROW("Tensor is not on device, cannot get buffer address");
    }

    const auto& storage = std::get<tt::tt_metal::DeviceStorage>(tensor.tensor_attributes->get_storage());
    if (storage.mesh_buffer) {
        return storage.mesh_buffer->address();
    }
    TT_THROW("Tensor is not allocated.");
}

uint32_t divup(uint32_t x, uint32_t y) { return static_cast<uint32_t>((x + y - 1) / y); }

bool isWormholeB0() { return true; }

std::shared_ptr<TtMobileNetV2> loadTtnnModel(
    const std::shared_ptr<ttnn::MeshDevice>& device, const std::string& weights_dir, int batch_size) {
    auto model_parameters = create_mobilenetv2_model_parameters(weights_dir, device);
    return std::make_shared<TtMobileNetV2>(model_parameters, device, batch_size);
}
