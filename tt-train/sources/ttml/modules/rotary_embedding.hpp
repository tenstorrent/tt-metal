// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "modules/module_base.hpp"
#include "ops/rope_op.hpp"

namespace ttml::modules {
class RotaryEmbedding : public ModuleBase {
private:
    ops::RotaryEmbeddingParams m_rope_params;

public:
    explicit RotaryEmbedding(const ops::RotaryEmbeddingParams &rope_params);
    [[nodiscard]] autograd::TensorPtr operator()(const autograd::TensorPtr &input) override;
    [[nodiscard]] autograd::TensorPtr operator()(const autograd::TensorPtr &input, const uint32_t token_position);
};

}  // namespace ttml::modules
