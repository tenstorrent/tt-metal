// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "deit_inference.h"
#include "ttnn/device.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace deit_inference {

class DeitTestInfra {
public:
    DeitTestInfra(
        ttnn::MeshDevice* device,
        int batch_size,
        const std::string& model_name = "models/experimental/deit/deit_cpp/deit_model/manifest.json");
    ~DeitTestInfra() = default;

    std::pair<ttnn::Tensor, ttnn::MemoryConfig> setup_l1_sharded_input();
    std::tuple<ttnn::Tensor, ttnn::MemoryConfig, ttnn::MemoryConfig> setup_dram_sharded_input(
        const std::optional<ttnn::Tensor>& tt_input_tensor = std::nullopt);

    ttnn::Tensor run(const std::optional<ttnn::Tensor>& tt_input_tensor = std::nullopt);

    ttnn::Tensor input_tensor;
    ttnn::Tensor output_tensor;
    std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> output_tuple;

    ttnn::Tensor logits;
    ttnn::Tensor cls_logits;
    ttnn::Tensor distillation_logits;

private:
    ttnn::MeshDevice* device_;
    int batch_size_;
    DeiTConfig config_;

    std::vector<std::optional<ttnn::Tensor>> head_masks_;
    ttnn::Tensor cls_token_;
    ttnn::Tensor distillation_token_;
    ttnn::Tensor position_embeddings_;
    std::vector<float> input_pixels_;

    // Parameters (weights/biases) for all layers
    std::unordered_map<std::string, ttnn::Tensor> parameters_;

    ttnn::Tensor create_tensor_from_vector(
        const std::vector<float>& data,
        const std::vector<uint32_t>& shape,
        ttnn::DataType dtype,
        ttnn::Layout layout,
        std::optional<ttnn::MemoryConfig> memory_config = std::nullopt) const;

    std::vector<float> read_tensor_data(const std::string& weights_root, const std::string& relative_path) const;
    ttnn::Tensor load_tensor_from_manifest(
        const std::string& tensor_name,
        const std::unordered_map<std::string, std::string>& tensor_files,
        const std::unordered_map<std::string, std::vector<uint32_t>>& tensor_shapes,
        const std::string& weights_root,
        ttnn::DataType dtype,
        ttnn::Layout layout,
        std::optional<ttnn::MemoryConfig> memory_config = std::nullopt) const;

    std::vector<float> pad_channels_to_four(const std::vector<float>& nchw, int batch, int height, int width) const;
    std::vector<float> reshape_for_patch_input(
        const std::vector<float>& nhwc_padded, int batch, int height, int width) const;
    std::vector<float> make_attention_mask(int sequence_size) const;
    std::vector<float> make_deterministic_input(int batch, int channels, int height, int width) const;
};

std::shared_ptr<DeitTestInfra> create_test_infra(
    ttnn::MeshDevice* device,
    int batch_size,
    const std::string& model_name = "models/experimental/deit/deit_cpp/deit_model/manifest.json");

}  // namespace deit_inference
