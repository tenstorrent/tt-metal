// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"

namespace ttml::ops {

struct RotaryEmbeddingParams {
    ttnn::Tensor cos_cache;
    ttnn::Tensor sin_cache;
    ttnn::Tensor neg_cos_cache;
    ttnn::Tensor neg_sin_cache;
    ttnn::Tensor trans_mat;
};

autograd::TensorPtr rope(const autograd::TensorPtr& input, const RotaryEmbeddingParams& rope_params);

}  // namespace ttml::ops
