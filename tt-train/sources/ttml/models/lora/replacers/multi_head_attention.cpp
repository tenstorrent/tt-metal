// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "multi_head_attention.hpp"

#include "modules/linear_module.hpp"
#include "modules/multi_head_attention.hpp"

namespace ttml::models {

bool MHALoraReplacer::handles(const modules::ModuleBase* module) const {
    return dynamic_cast<const modules::MultiHeadAttention*>(module) != nullptr;
}

void MHALoraReplacer::replace(
    modules::ModuleBase* module,
    const std::string& prefix,
    ShouldReplaceFunc should_replace,
    CreateLoraFunc create_lora) {
    auto* mha = dynamic_cast<modules::MultiHeadAttention*>(module);
    if (mha == nullptr) {
        return;
    }

    // Replace qkv_linear if targeted
    if (should_replace(prefix + "/qkv_linear")) {
        auto* linear = dynamic_cast<modules::LinearLayer*>(mha->m_qkv_linear.get());
        if (linear != nullptr) {
            auto lora = create_lora(linear, prefix + "/qkv_linear");
            mha->m_qkv_linear = mha->override_module("qkv_linear", lora);
        }
    }

    // Replace out_linear if targeted
    if (should_replace(prefix + "/out_linear")) {
        auto* linear = dynamic_cast<modules::LinearLayer*>(mha->m_out_linear.get());
        if (linear != nullptr) {
            auto lora = create_lora(linear, prefix + "/out_linear");
            mha->m_out_linear = mha->override_module("out_linear", lora);
        }
    }
}

}  // namespace ttml::models
