// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/ring_sdpa_bw_kv_device_operation.hpp"
#include "device/ring_sdpa_bw_q_device_operation.hpp"
#include "metal/common/const_utils.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

using RingDirection = ttml::metal::ops::ring_sdpa_bw::RingDirection;

struct RingSDPABwChunk {
    ttnn::Tensor grad_output;
    ttnn::Tensor attn_output;
    ttnn::Tensor query;
    ttnn::Tensor key;
    ttnn::Tensor value;
    ttnn::Tensor intermediates;
    ttnn::Tensor grad_query;
    ttnn::Tensor grad_key;
    ttnn::Tensor grad_value;
};

// Each device writes only the grads of the chunk get_ring_step_work assigns it; other grads keep their contents.
void ring_sdpa_bw(
    std::vector<RingSDPABwChunk> chunks,
    uint32_t ring_size,
    uint32_t ring_axis,
    uint32_t step,
    AttentionMaskType mask_type,
    RingDirection ring_direction = RingDirection::Backward,
    ops::RingLoadBalancing load_balancing = ops::RingLoadBalancing::None);

}  // namespace ttml::metal
