// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "modules/rotary_embedding.hpp"

#include "autograd/auto_context.hpp"
#include "ops/rope_op.hpp"

namespace ttml::modules {

RotaryEmbedding::RotaryEmbedding(const ops::RotaryEmbeddingParams& rope_params) : m_rope_params(rope_params) {
}

autograd::TensorPtr RotaryEmbedding::operator()(const autograd::TensorPtr& input) {
    return ttml::ops::rope(input, m_rope_params, 0U);
}

autograd::TensorPtr RotaryEmbedding::operator()(const autograd::TensorPtr& input, const uint32_t token_position) {
    auto params = m_rope_params;
    return ttml::ops::rope(input, params, token_position);
}

}  // namespace ttml::modules
