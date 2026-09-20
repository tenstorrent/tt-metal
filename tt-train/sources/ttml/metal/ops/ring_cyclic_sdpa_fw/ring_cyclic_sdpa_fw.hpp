// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/ring_cyclic_sdpa_fw_device_operation.hpp"
#include "metal/common/const_utils.hpp"
#include "metal/ops/common/ring_sdpa_utils.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// One ring-attention forward step through the cyclic-schedule forward
// (cyclic_sdpa_fw), on the chips the step selects. A drop-in for
// ring_ttnn_sdpa_fw and ring_sdpa_fw: the output in bfloat16 and the
// intermediates as (B, H, S, 32) Float32 with lse in column 0; chips that run
// nothing leave both preallocated tensors as they are. rows_per_block_tiles
// is the schedule's block height (C = N / (2 Bt 32) cores per slice).
std::tuple<ttnn::Tensor, ttnn::Tensor> ring_cyclic_sdpa_fw(
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    uint32_t ring_size,
    uint32_t ring_axis,
    uint32_t step,
    AttentionMaskType mask_type,
    ops::ring_cyclic_sdpa_fw::RingDirection ring_direction = ops::ring_cyclic_sdpa_fw::RingDirection::Backward,
    bool zigzag = false,
    ops::ZigzagVisitor visitor = ops::ZigzagVisitor::Any,
    uint32_t rows_per_block_tiles = 1U,
    const std::optional<ttnn::Tensor>& preallocated_output = std::nullopt,
    const std::optional<ttnn::Tensor>& preallocated_intermediates = std::nullopt);

}  // namespace ttml::metal
