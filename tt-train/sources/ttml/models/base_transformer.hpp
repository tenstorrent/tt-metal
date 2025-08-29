// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "autograd/module_base.hpp"

namespace ttml::models {
class BaseTransformer : public ttml::autograd::ModuleBase {
public:
    virtual ~BaseTransformer() = default;

    virtual void load_from_safetensors(const std::filesystem::path& model_path) {
    }
};
}  // namespace ttml::models
