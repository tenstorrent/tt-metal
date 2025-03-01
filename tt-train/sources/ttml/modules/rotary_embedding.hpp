// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/module_base.hpp"
#include "ops/rope_op.hpp"

namespace ttml::modules {
// FIXME: subclass from PositionalEmbeddingBase?
class RotaryEmbedding : public autograd::ModuleBase {
private:
    ops::RotaryEmbeddingParams &m_rope_params;

public:
    explicit RotaryEmbedding(ops::RotaryEmbeddingParams &rope_params);
    [[nodiscard]] autograd::TensorPtr operator()(const autograd::TensorPtr &input);

    static ops::RotaryEmbeddingParams build_params(uint32_t sequence_length, uint32_t head_dim, float theta = 10000.0F);
};

}  // namespace ttml::modules
