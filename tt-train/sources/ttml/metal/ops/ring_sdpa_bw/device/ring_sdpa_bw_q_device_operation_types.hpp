// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/common/const_utils.hpp"
#include "metal/ttnn_all_includes.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace ttml::metal::ops::ring_sdpa_bw::q {

using RingDirection = ttnn_fixed::distributed::RingShiftDirection;

// ============== Backward Q Types ==============

struct operation_attributes_t {
    uint32_t ring_size = 0;
    uint32_t ring_axis = 0;
    uint32_t step = 0;
    ttml::metal::AttentionMaskType mask_type = ttml::metal::AttentionMaskType::None;
    RingDirection ring_direction =
        ttnn_fixed::distributed::RingShiftDirection::Backward;  // Direction K/V is shifting in the ring
};

struct tensor_args_t {
    const ttnn::Tensor& grad_output;
    const ttnn::Tensor& attn_output;
    const ttnn::Tensor& query;
    const ttnn::Tensor& key;
    const ttnn::Tensor& value;
    const ttnn::Tensor& intermediates;
    std::optional<ttnn::Tensor> preallocated_grad_query;  // Preallocated output buffer
};

using tensor_return_value_t = ttnn::Tensor;  // [grad_Q]

using spec_return_value_t = ttnn::TensorSpec;

}  // namespace ttml::metal::ops::ring_sdpa_bw::q
