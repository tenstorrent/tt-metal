// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "add_rmsnorm_fw_allcore.hpp"

#include "device/add_rmsnorm_fw_allcore_device_operation.hpp"
#include "metal/ops/rmsnorm_bw_allcore/device/rmsnorm_bw_allcore_device_operation_types.hpp"

namespace ttml::metal {

std::vector<ttnn::Tensor> add_rmsnorm_fw_allcore(
    const ttnn::Tensor& residual_tensor,
    const ttnn::Tensor& delta_tensor,
    const ttnn::Tensor& gamma_tensor,
    float epsilon) {
    // Same (tile-row, slice) work split as the all-core backward: phase A adds the two inputs, writes the sum and a
    // partial sum of squares per slice; phase B folds the partials into the row's rms and scales the sum by
    // gamma / rms.
    const auto& padded = residual_tensor.padded_shape();
    const uint32_t rows = padded[0] * padded[1] * (padded[2] / tt::constants::TILE_HEIGHT);
    const uint32_t Wt = padded[3] / tt::constants::TILE_WIDTH;
    const auto grid = residual_tensor.device()->compute_with_storage_grid_size();
    const auto split = ops::rmsnorm_bw_allcore::device::choose_work_split(rows, Wt, grid.x * grid.y);

    auto partial = ttnn::prim::ttml_add_rmsnorm_fw_allcore_partial(residual_tensor, delta_tensor, split);
    auto result = ttnn::prim::ttml_add_rmsnorm_fw_allcore(partial[0], gamma_tensor, partial[1], split, epsilon);
    return {partial[0], result[0], result[1]};
}

}  // namespace ttml::metal
