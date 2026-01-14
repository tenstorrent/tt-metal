// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "lora_model.hpp"

#include <algorithm>
#include <memory>

#include "attention_lora_replacer.hpp"
#include "modules/linear_module.hpp"
#include "modules/lora/lora_linear_module.hpp"
#include "yaml-cpp/yaml.h"

namespace ttml::models {

LoRAConfig LoRAConfig::from_yaml(const YAML::Node& yaml_config) {
    LoRAConfig config;
    auto lora_config = yaml_config["lora_config"];

    config.r = lora_config["r"].as<uint32_t>(config.r);
    config.lora_alpha = lora_config["lora_alpha"].as<float>(config.lora_alpha);
    config.lora_dropout = lora_config["lora_dropout"].as<float>(config.lora_dropout);
    config.is_bias_trainable = lora_config["is_bias_trainable"].as<bool>(config.is_bias_trainable);
    config.use_rslora = lora_config["use_rslora"].as<bool>(config.use_rslora);

    auto target_modules_node = lora_config["target_modules"];
    if (target_modules_node && target_modules_node.IsSequence()) {
        config.target_modules = target_modules_node.as<std::vector<std::string>>();
    }

    auto ranks_node = lora_config["ranks"];
    if (ranks_node && ranks_node.IsSequence()) {
        config.ranks = ranks_node.as<std::vector<uint32_t>>();
    }

    auto alphas_node = lora_config["alphas"];
    if (alphas_node && alphas_node.IsSequence()) {
        config.alphas = alphas_node.as<std::vector<float>>();
    }

    // Validate that ranks and alphas length match target_modules length if provided
    if (config.target_modules.has_value()) {
        size_t target_modules_size = config.target_modules->size();
        if (config.ranks.has_value() && config.ranks->size() != target_modules_size) {
            throw std::runtime_error(
                "ranks length (" + std::to_string(config.ranks->size()) + ") must match target_modules length (" +
                std::to_string(target_modules_size) + ")");
        }
        if (config.alphas.has_value() && config.alphas->size() != target_modules_size) {
            throw std::runtime_error(
                "alphas length (" + std::to_string(config.alphas->size()) + ") must match target_modules length (" +
                std::to_string(target_modules_size) + ")");
        }
    } else {
        fmt::print("[WARNING] No target modules specified, LoRA does nothing\n");
    }

    return config;
}

LoraModel::LoraModel(std::shared_ptr<BaseTransformer> base_model, const LoRAConfig& config, const std::string& name) :
    m_base_model(std::move(base_model)), m_config(config) {
    create_name(name);

    // Register the base model as a submodule
    register_module(m_base_model, "base_model");

    // Freeze base model weights first
    freeze_base_model_weights();

    // Replace linear modules in attention layers with LoRA variants
    replace_attention_modules_recursive(m_base_model.get(), m_base_model->get_name());

    fmt::print("LoraModel created: Replaced linear modules with LoRA variants\n");
}

bool LoraModel::should_replace_module(const std::string& module_name) const {
    // If no target modules specified, replace all linear modules
    if (!m_config.target_modules.has_value()) {
        return true;
    }

    // Check if module name matches any of the target modules
    return std::find(m_config.target_modules->begin(), m_config.target_modules->end(), module_name) !=
           m_config.target_modules->end();
}

void LoraModel::freeze_base_model_weights() {
    auto params = m_base_model->parameters();

    for (auto& [name, tensor_ptr] : params) {
        tensor_ptr->set_requires_grad(false);
    }

    fmt::print("Froze {} parameters in base model\n", params.size());
}

ttml::modules::ModuleBasePtr LoraModel::create_lora_from_linear(
    ttml::modules::LinearLayer* linear_layer, const std::string& full_name) {
    // Get the weight from the linear layer
    auto weight = linear_layer->get_weight();

    // Get bias from the linear layer's named tensors if it exists
    auto& tensors = linear_layer->named_tensors();
    auto bias_it = tensors.find("bias");

    // Find the index of this module in target_modules to use per-layer configs
    uint32_t layer_rank = m_config.r;
    float layer_alpha = m_config.lora_alpha;

    if (m_config.target_modules.has_value()) {
        auto it = std::find(m_config.target_modules->begin(), m_config.target_modules->end(), full_name);
        if (it != m_config.target_modules->end()) {
            size_t index = std::distance(m_config.target_modules->begin(), it);

            // Use per-layer rank if available
            if (m_config.ranks.has_value() && index < m_config.ranks->size()) {
                layer_rank = (*m_config.ranks)[index];
            }

            // Use per-layer alpha if available
            if (m_config.alphas.has_value() && index < m_config.alphas->size()) {
                layer_alpha = (*m_config.alphas)[index];
            }
        }
    }

    // Create LoRALayerConfig from LoRAConfig
    ttml::modules::LoRALayerConfig lora_layer_config;
    lora_layer_config.rank = layer_rank;
    lora_layer_config.alpha = layer_alpha;
    lora_layer_config.dropout = m_config.lora_dropout;
    lora_layer_config.is_bias_trainable = m_config.is_bias_trainable;
    lora_layer_config.use_rslora = m_config.use_rslora;

    // Create LoRA layer as replacement
    std::shared_ptr<ttml::modules::LoRALinearLayer> lora_layer;
    if (bias_it != tensors.end()) {
        // Use existing bias
        lora_layer = std::make_shared<ttml::modules::LoRALinearLayer>(lora_layer_config, weight, bias_it->second);
    } else {
        // No bias
        lora_layer = std::make_shared<ttml::modules::LoRALinearLayer>(lora_layer_config, weight, false);
    }

    fmt::print(
        "Replacing {} with LoRA layer (r={}, alpha={}, has_bias={}, use_rslora={})\n",
        full_name,
        layer_rank,
        layer_alpha,
        (bias_it != tensors.end()),
        m_config.use_rslora);

    return lora_layer;
}

void LoraModel::replace_attention_modules_recursive(ttml::modules::ModuleBase* module, const std::string& prefix) {
    if (module == nullptr) {
        return;
    }

    // Check if any registered replacer can handle this module
    auto& registry = AttentionLoraReplacerRegistry::instance();
    if (auto* replacer = registry.find_replacer(module); replacer != nullptr) {
        // Create callbacks for the replacer
        auto should_replace = [this](const std::string& path) { return should_replace_module(path); };
        auto create_lora = [this](modules::LinearLayer* linear, const std::string& name) {
            return create_lora_from_linear(linear, name);
        };

        replacer->replace(module, prefix, should_replace, create_lora);
        return;  // Don't recurse further into attention internals
    }

    // Recurse into submodules
    const auto& named_modules = module->named_modules();
    for (const auto& [module_name, submodule_ptr] : named_modules) {
        std::string full_name = prefix + "/" + module_name;
        replace_attention_modules_recursive(submodule_ptr.get(), full_name);
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
