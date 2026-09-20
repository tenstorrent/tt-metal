// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/common/const_utils.hpp"
#include "metal/ops/common/ring_sdpa_utils.hpp"
#include "metal/ttnn_all_includes.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace ttml::metal::ops::ring_cyclic_sdpa_fw {

using RingDirection = ttnn_fixed::distributed::RingShiftDirection;

// One step of a ring-attention forward through the cyclic-schedule forward
// (cyclic_sdpa_fw), on the chips the step selects: the contiguous layout's
// causal rule (get_device_execution_info) or the zigzag layout's visitor rule
// (zigzag_visitor_runs). A drop-in for ring_ttnn_sdpa_fw: the output in
// bfloat16, the intermediates a (B, H, S, 32) Float32 tensor with
// lse = a max + ln sum in column 0, and chips that run nothing leave both
// preallocated tensors as they are.
struct RingCyclicSdpaFwParams {
    uint32_t ring_size = 0;
    uint32_t ring_axis = 0;
    uint32_t step = 0;
    ttml::metal::AttentionMaskType mask_type = ttml::metal::AttentionMaskType::Causal;
    RingDirection ring_direction = ttnn_fixed::distributed::RingShiftDirection::Backward;
    bool zigzag = false;
    ops::ZigzagVisitor visitor = ops::ZigzagVisitor::Any;
    // The cyclic schedule's block height, in tiles; C = N / (2 Bt 32) cores per slice.
    uint32_t rows_per_block_tiles = 1;
};

struct RingCyclicSdpaFwInputs {
    ttnn::Tensor query;
    ttnn::Tensor key;
    ttnn::Tensor value;
    std::optional<ttnn::Tensor> preallocated_output;
    std::optional<ttnn::Tensor> preallocated_intermediates;
};

using operation_attributes_t = RingCyclicSdpaFwParams;
using tensor_args_t = RingCyclicSdpaFwInputs;
// output, intermediates, and the single-chip op's two Float32 spill scratch tensors.
using tensor_return_value_t = std::vector<ttnn::Tensor>;
using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;

}  // namespace ttml::metal::ops::ring_cyclic_sdpa_fw
