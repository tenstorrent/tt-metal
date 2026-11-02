// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "add_rmsnorm_fw_rows.hpp"

#include "device/add_rmsnorm_fw_rows_device_operation.hpp"

namespace ttml::metal {

std::vector<ttnn::Tensor> add_rmsnorm_fw_rows(
    const ttnn::Tensor& residual,
    const ttnn::Tensor& delta,
    const ttnn::Tensor& gamma,
    uint32_t num_slices,
    uint32_t slice_tiles,
    float epsilon,
    bool delta_transposed) {
    return ttnn::prim::ttml_add_rmsnorm_fw_rows(
        residual, delta, gamma, num_slices, slice_tiles, epsilon, delta_transposed);
}

}  // namespace ttml::metal
