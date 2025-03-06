// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "modules/rotary_embedding.hpp"

#include "autograd/auto_context.hpp"
#include "ops/rope_op.hpp"

namespace ttml::modules {

RotaryEmbedding::RotaryEmbedding(const ops::RotaryEmbeddingParams& rope_params) : m_rope_params(rope_params) {
}

autograd::TensorPtr RotaryEmbedding::operator()(const autograd::TensorPtr& input) {
    return ttml::ops::rope(input, m_rope_params);
}

}  // namespace ttml::modules
