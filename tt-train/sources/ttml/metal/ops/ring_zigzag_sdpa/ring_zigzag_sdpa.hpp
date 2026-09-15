// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/ring_zigzag_bw_kv_device_operation.hpp"
#include "device/ring_zigzag_bw_q_device_operation.hpp"
#include "device/ring_zigzag_fw_device_operation.hpp"
#include "metal/common/const_utils.hpp"
#include "metal/ops/common/ring_sdpa_utils.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// The single-chip SDPA forward on the chips a zigzag step selects.
//
// The zigzag ring stores two chunks per chip and every step has two live
// chunk pairs per chip; the single-chip kernels take whole tensors of one
// chunk length, so the driver passes chunk-sized tensors and runs one pair
// per launch. The (1, 0) block is the same on every chip and goes through
// ring_sdpa_fw with mask None; the other block is (0, 0) on chips whose
// visitor is an earlier chunk and (1, 1) on chips whose visitor is later,
// and that is what `visitor` selects. Chips not selected run nothing and
// leave the preallocated outputs as they are, which the driver pre-fills
// with the no-contribution values.
std::tuple<ttnn::Tensor, ttnn::Tensor> ring_zigzag_sdpa_fw(
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    uint32_t ring_size,
    uint32_t ring_axis,
    uint32_t step,
    ops::ZigzagVisitor visitor,
    AttentionMaskType mask_type = AttentionMaskType::None,
    ops::ring_zigzag_fw::RingDirection ring_direction = ops::ring_zigzag_fw::RingDirection::Backward,
    const std::optional<ttnn::Tensor>& preallocated_output = std::nullopt,
    const std::optional<ttnn::Tensor>& preallocated_intermediates = std::nullopt);

// The two-pass single-chip backward (a dQ pass, then a dK/dV pass) on the
// chips a zigzag step selects; the reference ring_sdpa_bw's kernels, with
// only the chip selection different. Chips not selected leave the
// preallocated gradients as they are.
std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> ring_zigzag_sdpa_bw(
    const ttnn::Tensor& grad_output,
    const ttnn::Tensor& attn_output,
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    const ttnn::Tensor& intermediates,
    uint32_t ring_size,
    uint32_t ring_axis,
    uint32_t step,
    ops::ZigzagVisitor visitor,
    AttentionMaskType mask_type = AttentionMaskType::None,
    ops::ring_zigzag_bw::RingDirection ring_direction = ops::ring_zigzag_bw::RingDirection::Backward,
    const std::optional<ttnn::Tensor>& preallocated_grad_query = std::nullopt,
    const std::optional<ttnn::Tensor>& preallocated_grad_key = std::nullopt,
    const std::optional<ttnn::Tensor>& preallocated_grad_value = std::nullopt);

}  // namespace ttml::metal
