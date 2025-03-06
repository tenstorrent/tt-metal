// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/module_base.hpp"
#include "ops/rope_op.hpp"

namespace ttml::modules {
class RotaryEmbedding : public autograd::ModuleBase {
private:
    ops::RotaryEmbeddingParams m_rope_params;

public:
    explicit RotaryEmbedding(const ops::RotaryEmbeddingParams &rope_params);
    [[nodiscard]] autograd::TensorPtr operator()(const autograd::TensorPtr &input) override;
};

}  // namespace ttml::modules
