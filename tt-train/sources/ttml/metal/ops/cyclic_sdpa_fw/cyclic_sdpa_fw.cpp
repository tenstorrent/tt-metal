// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cyclic_sdpa_fw.hpp"

#include "device/cyclic_sdpa_fw_device_operation.hpp"

namespace ttml::metal {

std::tuple<ttnn::Tensor, ttnn::Tensor> cyclic_sdpa_fw(
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    uint32_t rows_per_block_tiles,
    AttentionMaskType mask_type,
    const std::optional<ttnn::Tensor>& preallocated_output,
    const std::optional<ttnn::Tensor>& preallocated_intermediates,
    uint32_t max_groups,
    uint32_t sequence_chunks,
    const std::vector<uint32_t>& row_chunks,
    const std::vector<uint32_t>& col_chunks) {
    auto result = ttnn::prim::ttml_cyclic_sdpa_fw(
        query, key, value, rows_per_block_tiles, mask_type, preallocated_output, preallocated_intermediates,
        max_groups, sequence_chunks, row_chunks, col_chunks);
    // result[2] and result[3] are the kernels' Float32 spill scratch; nothing outside needs them.
    return {result[0], result[1]};
}

}  // namespace ttml::metal
