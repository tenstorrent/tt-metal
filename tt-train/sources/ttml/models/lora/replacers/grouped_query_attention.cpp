// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "grouped_query_attention.hpp"

#include "modules/grouped_query_attention.hpp"
#include "modules/linear_module.hpp"

namespace ttml::models {

bool GQALoraReplacer::handles(const modules::ModuleBase* module) const {
    return dynamic_cast<const modules::GroupedQueryAttention*>(module) != nullptr;
}

void GQALoraReplacer::replace(
    modules::ModuleBase* module,
    const std::string& prefix,
    ShouldReplaceFunc should_replace,
    CreateLoraFunc create_lora) {
    auto* gqa = dynamic_cast<modules::GroupedQueryAttention*>(module);
    if (gqa == nullptr) {
        return;
    }

    // Replace q_linear if targeted
    if (should_replace(prefix + "/q_linear")) {
        auto* linear = dynamic_cast<modules::LinearLayer*>(gqa->m_q_linear.get());
        if (linear != nullptr) {
            auto lora = create_lora(linear, prefix + "/q_linear");
            gqa->m_q_linear = gqa->override_module("q_linear", lora);
        }
    }

    // Replace kv_linear if targeted
    if (should_replace(prefix + "/kv_linear")) {
        auto* linear = dynamic_cast<modules::LinearLayer*>(gqa->m_kv_linear.get());
        if (linear != nullptr) {
            auto lora = create_lora(linear, prefix + "/kv_linear");
            gqa->m_kv_linear = gqa->override_module("kv_linear", lora);
        }
    }

    // Replace out_linear if targeted
    if (should_replace(prefix + "/out_linear")) {
        auto* linear = dynamic_cast<modules::LinearLayer*>(gqa->m_out_linear.get());
        if (linear != nullptr) {
            auto lora = create_lora(linear, prefix + "/out_linear");
            gqa->m_out_linear = gqa->override_module("out_linear", lora);
        }
    }
}

}  // namespace ttml::models
