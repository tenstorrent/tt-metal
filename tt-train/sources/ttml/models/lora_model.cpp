// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "lora_model.hpp"

#include <memory>

#include "modules/linear_module.hpp"
#include "modules/lora_linear_module.hpp"

namespace ttml::models {

LoraModel::LoraModel(std::shared_ptr<BaseTransformer> base_model, const LoRAConfig& config, const std::string& name) :
    m_base_model(std::move(base_model)), m_config(config) {
    create_name(name);

    // Register the base model as a submodule
    register_module(m_base_model, "base_model");

    // Freeze base model weights first
    freeze_base_model_weights();

    // Replace Linear modules with LoRA variants
    replace_linear_modules_recursive(m_base_model.get(), m_base_model->get_name());

    fmt::print("LoraModel created: Replaced linear modules with LoRA variants\n");
}

bool LoraModel::should_replace_module(const std::string& module_name) const {
    // If no target modules specified, replace all linear modules
    if (!m_config.target_modules.has_value()) {
        return true;
    }

    // Check if module name matches any of the target modules
    for (const auto& target : *m_config.target_modules) {
        if (module_name.find(target) != std::string::npos) {
            return true;
        }
    }

    return false;
}

void LoraModel::freeze_base_model_weights() {
    auto params = m_base_model->parameters();

    for (auto& [name, tensor_ptr] : params) {
        tensor_ptr->set_requires_grad(false);
    }

    fmt::print("Froze {} parameters in base model\n", params.size());
}

void LoraModel::replace_linear_modules_recursive(ttml::modules::ModuleBase* module, const std::string& prefix) {
    if (module == nullptr) {
        return;
    }

    // Get all named modules from current module
    const auto& named_modules = module->named_modules();

    // We need to collect replacements first to avoid modifying while iterating
    std::vector<std::pair<std::string, std::shared_ptr<ttml::modules::ModuleBase>>> replacements;

    for (const auto& [module_name, submodule_ptr_ptr] : named_modules) {
        std::string full_name = prefix + "/" + module_name;
        const auto& submodule_ptr = *submodule_ptr_ptr;

        // Check if this is a LinearLayer
        auto* linear_layer = dynamic_cast<ttml::modules::LinearLayer*>(submodule_ptr.get());

        if (linear_layer != nullptr && should_replace_module(full_name)) {
            // Get the weight from the linear layer
            auto weight = linear_layer->get_weight();

            // Get bias from the linear layer's named tensors if it exists
            auto& tensors = linear_layer->named_tensors();
            auto bias_it = tensors.find("bias");

            // Create LoRALayerConfig from LoRAConfig
            ttml::modules::LoRALayerConfig lora_layer_config;
            lora_layer_config.rank = m_config.r;
            lora_layer_config.alpha = m_config.lora_alpha;
            lora_layer_config.dropout = m_config.lora_dropout;
            lora_layer_config.is_bias_trainable = m_config.is_bias_trainable;

            // Create LoRA layer as replacement
            std::shared_ptr<ttml::modules::LoRALinearLayer> lora_layer;
            if (bias_it != tensors.end()) {
                // Use existing bias
                lora_layer =
                    std::make_shared<ttml::modules::LoRALinearLayer>(lora_layer_config, weight, bias_it->second);
            } else {
                // No bias
                lora_layer = std::make_shared<ttml::modules::LoRALinearLayer>(lora_layer_config, weight, false);
            }

            // Store replacement
            replacements.emplace_back(module_name, lora_layer);

            fmt::print(
                "Replacing {} with LoRA layer (r={}, alpha={}, has_bias={})\n",
                full_name,
                m_config.r,
                m_config.lora_alpha,
                (bias_it != tensors.end()));
        } else {
            // Recursively process submodules
            replace_linear_modules_recursive(submodule_ptr.get(), full_name);
        }
    }

    // Apply replacements
    for (const auto& [module_name, lora_layer] : replacements) {
        module->override_module(module_name, lora_layer);
    }
}

ttml::autograd::TensorPtr LoraModel::operator()(
    const ttml::autograd::TensorPtr& x, const ttml::autograd::TensorPtr& mask) {
    return (*m_base_model)(x, mask);
}

void LoraModel::load_from_safetensors(const std::filesystem::path& model_path) {
    throw std::runtime_error(
        "load_from_safetensors is not supported for LoraModel. "
        "Please load the base model from safetensors before wrapping it with LoRA. "
        "The model structure has been modified with LoRA layers, and loading from safetensors "
        "would not correctly map to the modified structure.");
}

}  // namespace ttml::models
