// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"

namespace ttml::ops {

struct RotaryEmbeddingParams {
    ttnn::Tensor cos_cache{};
    ttnn::Tensor sin_cache{};
    ttnn::Tensor neg_cos_cache{};
    ttnn::Tensor neg_sin_cache{};
    ttnn::Tensor trans_mat{};

    uint32_t sequence_length = 0;
    uint32_t head_dim = 0;
};

autograd::TensorPtr rope(const autograd::TensorPtr& input, const RotaryEmbeddingParams& rope_params);

RotaryEmbeddingParams build_rope_params(uint32_t sequence_length, uint32_t head_dim, float theta = 10000.0F);
// Throws an exception if the input is bad, parameters are bad, or the two are
// incompatible with one another.
void validate_rope_input_and_params(const autograd::TensorPtr& input, const RotaryEmbeddingParams& rope_params);

}  // namespace ttml::ops
