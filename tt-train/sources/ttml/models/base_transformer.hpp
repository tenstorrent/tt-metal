// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <fmt/format.h>

#include <filesystem>
#include <stdexcept>

#include "modules/module_base.hpp"

namespace ttml::models {
class BaseTransformer : public ttml::modules::ModuleBase {
public:
    virtual ~BaseTransformer() = default;

    // Only the single-device GPT-2 and Llama models implement this. The others throw rather than keep their random
    // initialization while the caller believes the weights were loaded.
    virtual void load_from_safetensors(const std::filesystem::path& model_path) {
        throw std::logic_error(fmt::format(
            "This model can't load safetensors weights ({}): only the single-device GPT-2 and Llama models implement "
            "load_from_safetensors",
            model_path.string()));
    }
};
}  // namespace ttml::models
