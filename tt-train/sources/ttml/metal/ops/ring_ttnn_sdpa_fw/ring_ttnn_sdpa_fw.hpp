// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/ring_ttnn_sdpa_fw_device_operation.hpp"
#include "metal/common/const_utils.hpp"
#include "metal/ops/common/ring_sdpa_utils.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// One ring-attention forward step through ttnn's chunk-blocked flash-attention
// kernel with its log-sum-exp output, on the chips the step selects. A
// drop-in for ring_sdpa_fw (zigzag = false: the contiguous layout's causal
// rule) and ring_zigzag_sdpa_fw (zigzag = true: the visitor rule, mask as
// given). Returns the output in the query's dtype and the intermediates as a
// (B, H, S, 32) Float32 tensor with lse in column 0; chips that run nothing
// leave both preallocated tensors as they are. chunk_size is ttnn's query and
// key chunk (256 fits L1 at d = 128; 512 does not).
std::tuple<ttnn::Tensor, ttnn::Tensor> ring_ttnn_sdpa_fw(
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    uint32_t ring_size,
    uint32_t ring_axis,
    uint32_t step,
    AttentionMaskType mask_type,
    ops::ring_ttnn_sdpa_fw::RingDirection ring_direction = ops::ring_ttnn_sdpa_fw::RingDirection::Backward,
    bool zigzag = false,
    ops::ZigzagVisitor visitor = ops::ZigzagVisitor::Any,
    uint32_t chunk_size = 256U,
    const std::optional<ttnn::Tensor>& preallocated_output = std::nullopt,
    const std::optional<ttnn::Tensor>& preallocated_intermediates = std::nullopt);

}  // namespace ttml::metal
