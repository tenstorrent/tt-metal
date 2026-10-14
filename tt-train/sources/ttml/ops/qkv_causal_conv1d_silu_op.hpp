// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "autograd/tensor.hpp"

namespace ttml::ops {

// input [1,T,Q+K+V] and taps (logical volume Q+K+V) are TILE bf16; history [1,3,Q+K+V] receives no gradient.
// channel_chunk_size = 0 picks metal::qkv_causal_conv1d_silu_chunk_size (largest tile-aligned divisor <= 256).
std::tuple<autograd::TensorPtr, autograd::TensorPtr, autograd::TensorPtr> qkv_causal_conv1d_silu(
    const autograd::TensorPtr& input,
    const ttnn::Tensor& history,
    const autograd::TensorPtr& tap0,
    const autograd::TensorPtr& tap1,
    const autograd::TensorPtr& tap2,
    const autograd::TensorPtr& tap3,
    uint32_t q_width,
    uint32_t k_width,
    uint32_t v_width,
    uint32_t channel_chunk_size = 0);

}  // namespace ttml::ops
