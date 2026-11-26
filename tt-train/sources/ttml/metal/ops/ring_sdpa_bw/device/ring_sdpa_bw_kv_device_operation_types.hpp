// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/common/const_utils.hpp"
#include "metal/ops/common/ring_sdpa_utils.hpp"
#include "metal/ttnn_all_includes.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace ttml::metal::ops::ring_sdpa_bw::kv {

using RingDirection = ttnn_fixed::distributed::RingShiftDirection;

// ============== Backward KV Types ==============

struct RingSDPABwKVParams {
    uint32_t ring_size = 0;
    uint32_t ring_axis = 0;
    uint32_t step = 0;
    ttml::metal::AttentionMaskType mask_type = ttml::metal::AttentionMaskType::None;
    RingDirection ring_direction =
        ttnn_fixed::distributed::RingShiftDirection::Backward;  // Direction K/V is shifting in the ring
    RingLoadBalancing load_balancing = RingLoadBalancing::None;
};

struct RingSDPABwKVChunk {
    ttnn::Tensor grad_output;
    ttnn::Tensor query;
    ttnn::Tensor key;
    ttnn::Tensor value;
    ttnn::Tensor intermediates;
    ttnn::Tensor grad_key;
    ttnn::Tensor grad_value;
};

struct RingSDPABwKVInputs {
    std::vector<RingSDPABwKVChunk> chunks;
    ttnn::Tensor u_scaler;  // Precomputed rowsum(dO * O) from Q kernel
};

struct RingSDPABwKVChunkOutputs {
    ttnn::Tensor grad_key;
    ttnn::Tensor grad_value;
};

struct RingSDPABwKVChunkOutputSpecs {
    tt::tt_metal::TensorSpec grad_key;
    tt::tt_metal::TensorSpec grad_value;
};

using operation_attributes_t = RingSDPABwKVParams;
using tensor_args_t = RingSDPABwKVInputs;

using tensor_return_value_t = std::vector<RingSDPABwKVChunkOutputs>;

using spec_return_value_t = std::vector<RingSDPABwKVChunkOutputSpecs>;

}  // namespace ttml::metal::ops::ring_sdpa_bw::kv
