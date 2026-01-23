// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "models/lora/attention_lora_replacer.hpp"

namespace ttml::models {

/**
 * @brief LoRA replacer for MultiHeadAttention modules.
 *
 * Handles replacement of qkv_linear and out_linear layers.
 */
class MHALoraReplacer : public IAttentionLoraReplacer {
public:
    [[nodiscard]] bool handles(const modules::ModuleBase* module) const override;

    void replace(
        modules::ModuleBase* module,
        const std::string& prefix,
        ShouldReplaceFunc should_replace,
        CreateLoraFunc create_lora) override;
};

}  // namespace ttml::models
