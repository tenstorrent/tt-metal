// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_ttnn_sdpa_fw.hpp"

namespace ttml::metal {

std::tuple<ttnn::Tensor, ttnn::Tensor> ring_ttnn_sdpa_fw(
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    uint32_t ring_size,
    uint32_t ring_axis,
    uint32_t step,
    AttentionMaskType mask_type,
    ops::ring_ttnn_sdpa_fw::RingDirection ring_direction,
    bool zigzag,
    ops::ZigzagVisitor visitor,
    uint32_t chunk_size,
    const std::optional<ttnn::Tensor>& preallocated_output,
    const std::optional<ttnn::Tensor>& preallocated_intermediates) {
    return ttnn::prim::ttml_ring_ttnn_sdpa_fw(
        query,
        key,
        value,
        ring_size,
        ring_axis,
        step,
        mask_type,
        ring_direction,
        zigzag,
        visitor,
        chunk_size,
        preallocated_output,
        preallocated_intermediates);
}

}  // namespace ttml::metal
