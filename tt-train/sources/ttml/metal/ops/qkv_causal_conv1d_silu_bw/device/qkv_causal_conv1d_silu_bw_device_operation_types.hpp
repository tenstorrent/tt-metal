// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::qkv_causal_conv1d_silu_bw::device {

using ttnn::Tensor;

struct QkvCausalConv1dSiluBackwardParams {
    uint32_t row_count;
    uint32_t q_width;
    uint32_t k_width;
    uint32_t v_width;
    uint32_t channel_chunk_size;
    uint32_t row_splits;
    bool compute_tap_grads = true;  // false: no partial, no grad_tap output
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct QkvCausalConv1dSiluBackwardInputs {
    Tensor grad_q;
    Tensor grad_k;
    Tensor grad_v;
    Tensor input;
    Tensor history;
    Tensor tap0;
    Tensor tap1;
    Tensor tap2;
    Tensor tap3;
    std::optional<Tensor> preallocated_grad_input = std::nullopt;
};

}  // namespace ttml::metal::ops::qkv_causal_conv1d_silu_bw::device
