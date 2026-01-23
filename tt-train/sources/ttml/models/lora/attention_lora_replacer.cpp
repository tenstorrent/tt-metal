// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "attention_lora_replacer.hpp"

#include "replacers/grouped_query_attention.hpp"
#include "replacers/multi_head_attention.hpp"

namespace ttml::models {

AttentionLoraReplacerRegistry::AttentionLoraReplacerRegistry() {
    // Register built-in replacers
    m_replacers.push_back(std::make_unique<GQALoraReplacer>());
    m_replacers.push_back(std::make_unique<MHALoraReplacer>());
}

AttentionLoraReplacerRegistry& AttentionLoraReplacerRegistry::instance() {
    static AttentionLoraReplacerRegistry registry;
    return registry;
}

void AttentionLoraReplacerRegistry::register_replacer(std::unique_ptr<IAttentionLoraReplacer> replacer) {
    m_replacers.push_back(std::move(replacer));
}

IAttentionLoraReplacer* AttentionLoraReplacerRegistry::find_replacer(const modules::ModuleBase* module) const {
    for (const auto& replacer : m_replacers) {
        if (replacer->handles(module)) {
            return replacer.get();
        }
    }
    return nullptr;
}

}  // namespace ttml::models
