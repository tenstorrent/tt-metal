// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "autograd/tensor.hpp"
#include "base_transformer.hpp"
#include "modules/module_base.hpp"

namespace ttml::models {

struct LoRAConfig {
    uint32_t r = 128U;
    std::optional<std::vector<std::string>> target_modules;
    float lora_alpha = 1.0F;
    float lora_dropout = 0.0F;
    bool is_bias_trainable = false;
};

class LoraModel : public BaseTransformer {
private:
    ttml::modules::ModuleBasePtr m_base_model;
    LoRAConfig m_config;

    void replace_linear_modules_recursive(ttml::modules::ModuleBase* module, const std::string& prefix);

    bool should_replace_module(const std::string& module_name) const;

    void freeze_base_model_weights();

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
