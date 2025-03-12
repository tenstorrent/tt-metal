// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "autograd/tensor.hpp"

namespace ttml::ops {

std::tuple<autograd::TensorPtr, autograd::TensorPtr, autograd::TensorPtr> heads_creation(
    const autograd::TensorPtr& qkv, uint32_t num_heads);

autograd::TensorPtr heads_fusion(const autograd::TensorPtr& x);

std::tuple<autograd::TensorPtr, autograd::TensorPtr, autograd::TensorPtr> grouped_heads_creation(
    const autograd::TensorPtr& qs, const autograd::TensorPtr& kvs, uint32_t num_heads, uint32_t num_groups);

}  // namespace ttml::ops
