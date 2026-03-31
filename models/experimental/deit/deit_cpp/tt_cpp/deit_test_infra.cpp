// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "deit_test_infra.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/creation.hpp"
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace deit_inference {
namespace {

using json = nlohmann::json;
namespace fs = std::filesystem;

std::vector<uint32_t> json_shape_to_vector(const json& shape_json) {
    std::vector<uint32_t> shape;
    shape.reserve(shape_json.size());
    for (const auto& dim : shape_json) {
        shape.push_back(dim.get<uint32_t>());
    }
    return shape;
}

size_t volume(const std::vector<uint32_t>& shape) {
    size_t elements = 1;
    for (uint32_t dim : shape) {
        elements *= dim;
    }
    return elements;
}

std::string require_string(const json& object, const char* key) {
    if (!object.contains(key) || !object.at(key).is_string()) {
        throw std::runtime_error(std::string("Missing string field: ") + key);
    }
    return object.at(key).get<std::string>();
}

const json& require_object(const json& object, const char* key) {
    if (!object.contains(key) || !object.at(key).is_object()) {
        throw std::runtime_error(std::string("Missing object field: ") + key);
    }
    return object.at(key);
}

const json& require_array(const json& object, const char* key) {
    if (!object.contains(key) || !object.at(key).is_array()) {
        throw std::runtime_error(std::string("Missing array field: ") + key);
    }
    return object.at(key);
}

}  // namespace

ttnn::Tensor DeitTestInfra::create_tensor_from_vector(
    const std::vector<float>& data,
    const std::vector<uint32_t>& shape,
    ttnn::DataType dtype,
    ttnn::Layout layout,
    const std::optional<ttnn::MemoryConfig>& memory_config) const {
    const bool block_float = dtype == ttnn::DataType::BFLOAT4_B || dtype == ttnn::DataType::BFLOAT8_B;
    const auto host_layout = block_float ? ttnn::TILE_LAYOUT : ttnn::ROW_MAJOR_LAYOUT;
    const ttnn::TensorSpec spec(
        ttnn::Shape(shape), ttnn::TensorLayout(dtype, ttnn::PageConfig(host_layout), ttnn::MemoryConfig{}));

    auto tensor = ttnn::Tensor::from_vector(data, spec);

    if (!block_float && layout == ttnn::TILE_LAYOUT) {
        tensor = ttnn::to_layout(tensor, ttnn::TILE_LAYOUT, dtype, memory_config);
    }

    return ttnn::to_device(tensor, device_, memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG));
}

std::vector<float> DeitTestInfra::read_tensor_data(
    const std::string& weights_root, const std::string& relative_path) const {
    const fs::path full_path = fs::path(weights_root) / relative_path;
    std::ifstream stream(full_path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Failed to open tensor data file: " + full_path.string());
    }

    stream.seekg(0, std::ios::end);
    const std::streamsize bytes = stream.tellg();
    stream.seekg(0, std::ios::beg);

    if (bytes < 0 || bytes % static_cast<std::streamsize>(sizeof(float)) != 0) {
        throw std::runtime_error("Invalid tensor data size: " + full_path.string());
    }

    std::vector<float> values(static_cast<size_t>(bytes) / sizeof(float));
    if (!stream.read(reinterpret_cast<char*>(values.data()), bytes)) {
        throw std::runtime_error("Failed to read tensor data file: " + full_path.string());
    }

    return values;
}

ttnn::Tensor DeitTestInfra::load_tensor_from_manifest(
    const std::string& tensor_name,
    const std::unordered_map<std::string, std::string>& tensor_files,
    const std::unordered_map<std::string, std::vector<uint32_t>>& tensor_shapes,
    const std::string& weights_root,
    ttnn::DataType dtype,
    ttnn::Layout layout,
    const std::optional<ttnn::MemoryConfig>& memory_config) const {
    const auto file_it = tensor_files.find(tensor_name);
    const auto shape_it = tensor_shapes.find(tensor_name);
    if (file_it == tensor_files.end() || shape_it == tensor_shapes.end()) {
        throw std::runtime_error("Tensor missing from manifest: " + tensor_name);
    }

    auto data = read_tensor_data(weights_root, file_it->second);
    if (data.size() != volume(shape_it->second)) {
        throw std::runtime_error("Tensor volume mismatch for: " + tensor_name);
    }

    return create_tensor_from_vector(data, shape_it->second, dtype, layout, memory_config);
}

std::vector<float> DeitTestInfra::pad_channels_to_four(
    const std::vector<float>& nchw, int batch, int height, int width) const {
    std::vector<float> padded(static_cast<size_t>(batch) * height * width * 4, 0.0f);
    for (int n = 0; n < batch; ++n) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                for (int c = 0; c < 3; ++c) {
                    const size_t src = ((static_cast<size_t>(n) * 3 + c) * height + h) * width + w;
                    const size_t dst = (((static_cast<size_t>(n) * height + h) * width + w) * 4) + c;
                    padded[dst] = nchw[src];
                }
            }
        }
    }
    return padded;
}

std::vector<float> DeitTestInfra::reshape_for_patch_input(
    const std::vector<float>& nhwc_padded, int batch, int height, int width) const {
    constexpr int patch_size = 16;
    const int patch_columns = width / patch_size;
    std::vector<float> reshaped(static_cast<size_t>(batch) * height * patch_columns * (4 * patch_size));

    for (int n = 0; n < batch; ++n) {
        for (int h = 0; h < height; ++h) {
            for (int patch_col = 0; patch_col < patch_columns; ++patch_col) {
                for (int i = 0; i < patch_size; ++i) {
                    for (int c = 0; c < 4; ++c) {
                        const int w = patch_col * patch_size + i;
                        const size_t src = (((static_cast<size_t>(n) * height + h) * width + w) * 4) + c;
                        const size_t dst =
                            ((((static_cast<size_t>(n) * height + h) * patch_columns + patch_col) * (4 * patch_size)) +
                             i * 4 + c);
                        reshaped[dst] = nhwc_padded[src];
                    }
                }
            }
        }
    }

    return reshaped;
}

std::vector<float> DeitTestInfra::make_attention_mask(int sequence_size) const {
    return std::vector<float>(static_cast<size_t>(batch_size_) * sequence_size, 1.0f);
}

std::vector<float> DeitTestInfra::make_deterministic_input(int batch, int channels, int height, int width) const {
    std::vector<float> input(static_cast<size_t>(batch) * channels * height * width);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>((i % 251) - 125) / 125.0f;
    }
    return input;
}

DeitTestInfra::DeitTestInfra(ttnn::MeshDevice* device, int batch_size, const std::string& model_name) :
    device_(device), batch_size_(batch_size) {
    update_model_config(config_, batch_size);

    const fs::path manifest_path = model_name;
    std::ifstream manifest_stream(manifest_path);
    if (!manifest_stream) {
        throw std::runtime_error("Failed to open DeiT manifest: " + manifest_path.string());
    }

    const json manifest = json::parse(manifest_stream);
    const std::string weights_root = (manifest_path.parent_path() / require_string(manifest, "weights_dir")).string();

    std::unordered_map<std::string, std::string> tensor_files;
    std::unordered_map<std::string, std::vector<uint32_t>> tensor_shapes;

    const json& tensors = require_object(manifest, "tensors");
    for (const auto& [name, tensor_info] : tensors.items()) {
        tensor_files[name] = require_string(tensor_info, "file");
        tensor_shapes[name] = json_shape_to_vector(require_array(tensor_info, "shape"));
    }

    cls_token_ = load_tensor_from_manifest(
        "deit.embeddings.cls_token",
        tensor_files,
        tensor_shapes,
        weights_root,
        ttnn::DataType::BFLOAT16,
        ttnn::Layout::ROW_MAJOR);
    distillation_token_ = load_tensor_from_manifest(
        "deit.embeddings.distillation_token",
        tensor_files,
        tensor_shapes,
        weights_root,
        ttnn::DataType::BFLOAT16,
        ttnn::Layout::ROW_MAJOR);
    position_embeddings_ = load_tensor_from_manifest(
        "deit.embeddings.position_embeddings",
        tensor_files,
        tensor_shapes,
        weights_root,
        ttnn::DataType::BFLOAT8_B,
        ttnn::Layout::TILE);

    for (const auto& [name, shape] : tensor_shapes) {
        if (name == "deit.embeddings.cls_token" || name == "deit.embeddings.distillation_token" ||
            name == "deit.embeddings.position_embeddings") {
            continue;
        }

        const bool layer_norm_bias = name.find("layernorm") != std::string::npos || name == "deit.layernorm.weight" ||
                                     name == "deit.layernorm.bias";
        const bool use_row_major = name.find("patch_embeddings") == std::string::npos &&
                                   (name.ends_with(".bias") || name == "cls_classifier.weight" ||
                                    name == "distillation_classifier.weight" || name == "classifier.weight");

        parameters_[name] = load_tensor_from_manifest(
            name,
            tensor_files,
            tensor_shapes,
            weights_root,
            layer_norm_bias ? ttnn::DataType::BFLOAT16 : ttnn::DataType::BFLOAT8_B,
            (layer_norm_bias || use_row_major) ? ttnn::Layout::ROW_MAJOR : ttnn::Layout::TILE);
    }

    const auto attention_mask_data = make_attention_mask(224);
    for (int i = 0; i < config_.num_layers; ++i) {
        head_masks_.push_back(create_tensor_from_vector(
            attention_mask_data,
            {static_cast<uint32_t>(batch_size_), 1, 1, 224},
            ttnn::DataType::BFLOAT8_B,
            ttnn::Layout::TILE,
            ttnn::L1_MEMORY_CONFIG));
    }

    input_pixels_ = make_deterministic_input(batch_size_, 3, 224, 224);
}

std::pair<ttnn::Tensor, ttnn::MemoryConfig> DeitTestInfra::setup_l1_sharded_input() {
    const auto padded = pad_channels_to_four(input_pixels_, batch_size_, 224, 224);
    const auto reshaped = reshape_for_patch_input(padded, batch_size_, 224, 224);

    ttnn::Tensor tt_inputs_host = create_tensor_from_vector(
        reshaped,
        {static_cast<uint32_t>(batch_size_), 224, 14, 64},
        ttnn::DataType::BFLOAT16,
        ttnn::Layout::ROW_MAJOR,
        std::nullopt);
    tt_inputs_host = ttnn::from_device(tt_inputs_host);

    const int n_cores = batch_size_ * 3;
    ttnn::CoreRangeSet shard_grid({ttnn::CoreRange(ttnn::CoreCoord(0, 0), ttnn::CoreCoord(batch_size_ - 1, 3))});
    std::array<uint32_t, 2> shard_shape = {
        static_cast<uint32_t>((batch_size_ * 224 * 14) / n_cores),
        64,
    };

    tt::tt_metal::ShardSpec shard_spec(shard_grid, shard_shape, ttnn::ShardOrientation::ROW_MAJOR);
    ttnn::MemoryConfig input_mem_config(ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ttnn::BufferType::L1, shard_spec);

    return {tt_inputs_host, input_mem_config};
}

std::tuple<ttnn::Tensor, ttnn::MemoryConfig, ttnn::MemoryConfig> DeitTestInfra::setup_dram_sharded_input(
    const std::optional<ttnn::Tensor>& tt_input_tensor) {
    if (tt_input_tensor.has_value()) {
        auto dram_grid_size = device_->dram_grid_size();
        ttnn::CoreRangeSet dram_grid(
            {ttnn::CoreRange(ttnn::CoreCoord(0, 0), ttnn::CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))});

        const auto [default_host, input_mem_config] = setup_l1_sharded_input();
        const uint32_t width = tt_input_tensor->logical_shape()[-1];
        const uint32_t volume_value = tt_input_tensor->logical_volume();
        const uint32_t height = volume_value / width;
        const uint32_t shard_height = (height + dram_grid_size.x - 1) / dram_grid_size.x;
        tt::tt_metal::ShardSpec dram_shard_spec(
            dram_grid, {shard_height, width}, tt::tt_metal::ShardOrientation::ROW_MAJOR);
        ttnn::MemoryConfig sharded_mem_config_dram(
            ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ttnn::BufferType::DRAM, dram_shard_spec);
        return {*tt_input_tensor, sharded_mem_config_dram, input_mem_config};
    }

    auto [tt_inputs_host, input_mem_config] = setup_l1_sharded_input();

    auto dram_grid_size = device_->dram_grid_size();
    ttnn::CoreRangeSet dram_grid(
        {ttnn::CoreRange(ttnn::CoreCoord(0, 0), ttnn::CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))});
    uint32_t width = tt_inputs_host.logical_shape()[-1];
    uint32_t volume_value = tt_inputs_host.logical_volume();
    uint32_t height = volume_value / width;
    uint32_t shard_height = (height + dram_grid_size.x - 1) / dram_grid_size.x;

    tt::tt_metal::ShardSpec dram_shard_spec(
        dram_grid, {shard_height, width}, tt::tt_metal::ShardOrientation::ROW_MAJOR);
    ttnn::MemoryConfig sharded_mem_config_dram(
        ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ttnn::BufferType::DRAM, dram_shard_spec);

    return {tt_inputs_host, sharded_mem_config_dram, input_mem_config};
}

ttnn::Tensor DeitTestInfra::run(const std::optional<ttnn::Tensor>& tt_input_tensor) {
    if (tt_input_tensor.has_value()) {
        input_tensor = tt_input_tensor.value();
    }

    output_tuple =
        deit(config_, input_tensor, head_masks_, cls_token_, distillation_token_, position_embeddings_, parameters_);
    logits = std::get<0>(output_tuple);
    cls_logits = std::get<1>(output_tuple);
    distillation_logits = std::get<2>(output_tuple);
    output_tensor = logits;
    return output_tensor;
}

std::shared_ptr<DeitTestInfra> create_test_infra(
    ttnn::MeshDevice* device, int batch_size, const std::string& model_name) {
    return std::make_shared<DeitTestInfra>(device, batch_size, model_name);
}

}  // namespace deit_inference
