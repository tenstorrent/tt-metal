// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/common/const_utils.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::ring_sdpa {

// Create MeshWorkload for forward SDPA
// mask_type determines which mask is used per device:
// - None: No masking (full attention)
// - Causal: Uses causal mask for step 0, full for earlier chunks, skips later chunks
tt::tt_metal::distributed::MeshWorkload create_ring_sdpa_workload(
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    ttnn::Tensor& output,
    ttnn::Tensor& intermediates,
    uint32_t ring_size,
    uint32_t ring_axis,
    uint32_t step,
    ttml::metal::AttentionMaskType mask_type);

}  // namespace ttml::metal::ops::ring_sdpa
