// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/common/const_utils.hpp"
#include "metal/ops/common/ring_sdpa_utils.hpp"
#include "metal/ttnn_all_includes.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace ttml::metal::ops::ring_sdpa_bw::q {

using RingDirection = ttnn_fixed::distributed::RingShiftDirection;

// ============== Backward Q Types ==============

struct RingSDPABwQParams {
    uint32_t ring_size = 0;
    uint32_t ring_axis = 0;
    uint32_t step = 0;
    ttml::metal::AttentionMaskType mask_type = ttml::metal::AttentionMaskType::None;
    RingDirection ring_direction =
        ttnn_fixed::distributed::RingShiftDirection::Backward;  // Direction K/V is shifting in the ring
    RingLoadBalancing load_balancing = RingLoadBalancing::None;
};

struct RingSDPABwQChunk {
    ttnn::Tensor grad_output;
    ttnn::Tensor attn_output;
    ttnn::Tensor query;
    ttnn::Tensor key;
    ttnn::Tensor value;
    ttnn::Tensor intermediates;
    ttnn::Tensor grad_query;
};

struct RingSDPABwQInputs {
    std::vector<RingSDPABwQChunk> chunks;
};

struct RingSDPABwQChunkOutputs {
    ttnn::Tensor grad_query;
};

struct RingSDPABwQOutputs {
    std::vector<RingSDPABwQChunkOutputs> chunks;
    ttnn::Tensor u_scaler;
};

struct RingSDPABwQChunkOutputSpecs {
    tt::tt_metal::TensorSpec grad_query;
};

struct RingSDPABwQOutputSpecs {
    std::vector<RingSDPABwQChunkOutputSpecs> chunks;
    tt::tt_metal::TensorSpec u_scaler;
};

using operation_attributes_t = RingSDPABwQParams;
using tensor_args_t = RingSDPABwQInputs;

using tensor_return_value_t = RingSDPABwQOutputs;

using spec_return_value_t = RingSDPABwQOutputSpecs;

}  // namespace ttml::metal::ops::ring_sdpa_bw::q
