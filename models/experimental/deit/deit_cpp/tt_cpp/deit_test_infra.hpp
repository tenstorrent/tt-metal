#pragma once

#include "deit_inference.h"
#include "ttnn/device.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <memory>
#include <vector>
#include <optional>
#include <torch/torch.h>

namespace deit_inference {

class DeitTestInfra {
public:
    DeitTestInfra(
        ttnn::MeshDevice* device, int batch_size, const std::string& model_name = "facebook/deit-tiny-patch16-224");
    ~DeitTestInfra() = default;

    std::pair<ttnn::Tensor, ttnn::MemoryConfig> setup_l1_sharded_input(
        const std::optional<torch::Tensor>& torch_pixel_values = std::nullopt);
    std::tuple<ttnn::Tensor, ttnn::MemoryConfig, ttnn::MemoryConfig> setup_dram_sharded_input(
        const std::optional<torch::Tensor>& torch_input_tensor = std::nullopt);

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

    // Parameters (weights/biases) for all layers
    // We will use the python-like structure map in C++ or a dedicated struct
    std::unordered_map<std::string, ttnn::Tensor> parameters_;

    // Not strictly needed if we pass torch_pixel_values to setup_l1_sharded_input
    // but kept for compatibility with potential future needs
    torch::Tensor torch_pixel_values_;
};

std::shared_ptr<DeitTestInfra> create_test_infra(
    ttnn::MeshDevice* device, int batch_size, const std::string& model_name = "facebook/deit-tiny-patch16-224");

}  // namespace deit_inference
