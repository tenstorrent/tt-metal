// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "autograd/tensor.hpp"
#include "models/base_transformer.hpp"
#include "modules/module_base.hpp"

namespace ttml::modules {
class LinearLayer;
}  // namespace ttml::modules

namespace YAML {
class Node;
}  // namespace YAML

namespace ttml::models {

struct LoRAConfig {
    uint32_t r = 128U;
    std::optional<std::vector<std::string>> target_modules;
    float lora_alpha = 1.0F;
    float lora_dropout = 0.0F;
    bool is_bias_trainable = false;
    bool use_rslora = false;
    std::optional<std::vector<uint32_t>> ranks;
    std::optional<std::vector<float>> alphas;

    static LoRAConfig from_yaml(const YAML::Node& yaml_config);
};

class LoraModel : public BaseTransformer {
private:
    ttml::modules::ModuleBasePtr m_base_model;
    LoRAConfig m_config;

    // Determine if a module at the given path should be replaced
    [[nodiscard]] bool should_replace_module(const std::string& module_name) const;

    // Freeze all parameters in the base model
    void freeze_base_model_weights();

    // Create a LoRA layer from an existing linear layer
    [[nodiscard]] ttml::modules::ModuleBasePtr create_lora_from_linear(
        ttml::modules::LinearLayer* linear_layer, const std::string& full_name);

    // Recursive traversal to find and process attention modules using registered replacers
    void replace_attention_modules_recursive(ttml::modules::ModuleBase* module, const std::string& prefix);

public:
    LoraModel(
        std::shared_ptr<BaseTransformer> base_model, const LoRAConfig& config, const std::string& name = "lora_model");

    ~LoraModel() override = default;

    [[nodiscard]] ttml::autograd::TensorPtr operator()(
        const ttml::autograd::TensorPtr& x, const ttml::autograd::TensorPtr& mask) override;

    void load_from_safetensors(const std::filesystem::path& model_path) override;

    [[nodiscard]] std::shared_ptr<BaseTransformer> get_base_model() const {
        return std::dynamic_pointer_cast<BaseTransformer>(m_base_model);
    }
};

}  // namespace ttml::models
