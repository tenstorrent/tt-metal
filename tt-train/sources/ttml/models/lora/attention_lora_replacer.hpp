// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "modules/module_base.hpp"

namespace ttml::modules {
class LinearLayer;
}

namespace ttml::models {

// Callback types for LoRA replacement operations
using ShouldReplaceFunc = std::function<bool(const std::string& module_path)>;
using CreateLoraFunc =
    std::function<modules::ModuleBasePtr(modules::LinearLayer* linear, const std::string& full_name)>;

/**
 * @brief Interface for attention-specific LoRA replacement strategies.
 *
 * Implement this interface to add LoRA support for new attention mechanisms
 * without modifying the LoraModel source code.
 *
 * Example implementation:
 * @code
 * class MyCustomAttentionReplacer : public IAttentionLoraReplacer {
 * public:
 *     bool handles(const modules::ModuleBase* module) const override {
 *         return dynamic_cast<const MyCustomAttention*>(module) != nullptr;
 *     }
 *
 *     void replace(
 *         modules::ModuleBase* module,
 *         const std::string& prefix,
 *         ShouldReplaceFunc should_replace,
 *         CreateLoraFunc create_lora) override
 *     {
 *         auto* attn = dynamic_cast<MyCustomAttention*>(module);
 *         if (should_replace(prefix + "/my_linear")) {
 *             auto* linear = dynamic_cast<LinearLayer*>(attn->my_linear.get());
 *             if (linear) {
 *                 auto lora = create_lora(linear, prefix + "/my_linear");
 *                 attn->my_linear = attn->override_module("my_linear", lora);
 *             }
 *         }
 *     }
 * };
 * @endcode
 */
class IAttentionLoraReplacer {
public:
    virtual ~IAttentionLoraReplacer() = default;

    /**
     * @brief Check if this replacer can handle the given module type.
     * @param module The module to check
     * @return true if this replacer handles this module type
     */
    [[nodiscard]] virtual bool handles(const modules::ModuleBase* module) const = 0;

    /**
     * @brief Perform LoRA replacement on the attention module.
     * @param module The attention module to modify
     * @param prefix The full path prefix for this module (e.g., "llama/block_0/attention")
     * @param should_replace Callback to check if a specific linear layer should be replaced
     * @param create_lora Callback to create a LoRA layer from a LinearLayer
     */
    virtual void replace(
        modules::ModuleBase* module,
        const std::string& prefix,
        ShouldReplaceFunc should_replace,
        CreateLoraFunc create_lora) = 0;
};

/**
 * @brief Registry for attention LoRA replacer strategies.
 *
 * This singleton registry allows registration of custom attention replacers.
 * Replacers are checked in order of registration, so more specific handlers
 * should be registered before generic ones.
 *
 * Built-in replacers for GroupedQueryAttention and MultiHeadAttention are
 * registered automatically.
 *
 * Usage:
 * @code
 * // Register a custom replacer (typically at static initialization time)
 * AttentionLoraReplacerRegistry::instance().register_replacer(
 *     std::make_unique<MyCustomAttentionReplacer>());
 * @endcode
 */
class AttentionLoraReplacerRegistry {
public:
    /**
     * @brief Get the singleton registry instance.
     * @return Reference to the global registry
     */
    static AttentionLoraReplacerRegistry& instance();

    /**
     * @brief Register a new attention replacer.
     *
     * Replacers are checked in registration order. Register more specific
     * handlers before generic ones.
     *
     * @param replacer The replacer to register (ownership is transferred)
     */
    void register_replacer(std::unique_ptr<IAttentionLoraReplacer> replacer);

    /**
     * @brief Find a replacer that can handle the given module.
     * @param module The module to find a handler for
     * @return Pointer to the replacer, or nullptr if none found
     */
    [[nodiscard]] IAttentionLoraReplacer* find_replacer(const modules::ModuleBase* module) const;

private:
    AttentionLoraReplacerRegistry();
    std::vector<std::unique_ptr<IAttentionLoraReplacer>> m_replacers;
};

}  // namespace ttml::models
