// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/common/const_utils.hpp"
#include "metal/ops/common/ring_sdpa_utils.hpp"
#include "metal/ttnn_all_includes.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace ttml::metal::ops::ring_sdpa_fw {

using RingDirection = ttnn_fixed::distributed::RingShiftDirection;

// ============== Forward Pass Types ==============

struct RingSDPAFwParams {
    uint32_t ring_size = 0;
    uint32_t ring_axis = 0;
    uint32_t step = 0;
    ttml::metal::AttentionMaskType mask_type = ttml::metal::AttentionMaskType::None;
    RingDirection ring_direction =
        ttnn_fixed::distributed::RingShiftDirection::Backward;  // Direction K/V is shifting in the ring
    RingLoadBalancing load_balancing = RingLoadBalancing::None;
};

struct RingSDPAFwChunk {
    ttnn::Tensor query;
    ttnn::Tensor key;
    ttnn::Tensor value;
    ttnn::Tensor output;
    ttnn::Tensor intermediates;
};

struct RingSDPAFwInputs {
    std::vector<RingSDPAFwChunk> chunks;
};

struct RingSDPAFwChunkOutputs {
    ttnn::Tensor output;
    ttnn::Tensor intermediates;
};

struct RingSDPAFwChunkOutputSpecs {
    tt::tt_metal::TensorSpec output;
    tt::tt_metal::TensorSpec intermediates;
};

using operation_attributes_t = RingSDPAFwParams;
using tensor_args_t = RingSDPAFwInputs;

using tensor_return_value_t = std::vector<RingSDPAFwChunkOutputs>;

using spec_return_value_t = std::vector<RingSDPAFwChunkOutputSpecs>;

}  // namespace ttml::metal::ops::ring_sdpa_fw
