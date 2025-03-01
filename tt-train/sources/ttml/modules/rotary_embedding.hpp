// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/module_base.hpp"

namespace ttml::modules {

struct RotaryEmbeddingParams {
    ttnn::Tensor cos_cache;
    ttnn::Tensor sin_cache;
    ttnn::Tensor neg_cos_cache;
    ttnn::Tensor neg_sin_cache;
    ttnn::Tensor trans_mat;
};

// FIXME: subclass from PositionalEmbeddingBase?
class RotaryEmbedding : public autograd::ModuleBase {
private:
    RotaryEmbeddingParams &m_rope_params;

public:
    explicit RotaryEmbedding(RotaryEmbeddingParams &rope_params);
    [[nodiscard]] autograd::TensorPtr operator()(const autograd::TensorPtr &input);

    static RotaryEmbeddingParams build_params(uint32_t sequence_length, uint32_t head_dim, float theta = 10000.0F);
};

}  // namespace ttml::modules
