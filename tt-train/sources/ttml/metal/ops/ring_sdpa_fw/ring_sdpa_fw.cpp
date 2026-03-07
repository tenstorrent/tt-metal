// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_sdpa_fw.hpp"

#include "device/ring_sdpa_fw_device_operation.hpp"

namespace ttml::metal {

std::tuple<ttnn::Tensor, ttnn::Tensor> ring_sdpa_fw(
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    uint32_t ring_size,
    uint32_t ring_axis,
    uint32_t step,
    AttentionMaskType mask_type,
    ops::ring_sdpa_fw::RingDirection ring_direction,
    const std::optional<ttnn::Tensor>& preallocated_output,
    const std::optional<ttnn::Tensor>& preallocated_intermediates) {
    return ttnn::prim::ttml_ring_sdpa_fw(
        query,
        key,
        value,
        ring_size,
        ring_axis,
        step,
        mask_type,
        ring_direction,
        preallocated_output,
        preallocated_intermediates);
}

}  // namespace ttml::metal
