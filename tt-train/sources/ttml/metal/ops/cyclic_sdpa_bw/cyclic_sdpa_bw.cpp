// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cyclic_sdpa_bw.hpp"

#include "device/cyclic_sdpa_bw_device_operation.hpp"

namespace ttml::metal {

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> cyclic_sdpa_bw(
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    const ttnn::Tensor& grad_output,
    const ttnn::Tensor& log_sum_exp,
    const ttnn::Tensor& row_scalar,
    uint32_t rows_per_block_tiles,
    bool use_barrier) {
    auto result = ttnn::prim::ttml_cyclic_sdpa_bw(
        query, key, value, grad_output, log_sum_exp, row_scalar, rows_per_block_tiles, use_barrier);
    return {result[0], result[1], result[2]};
}

}  // namespace ttml::metal
