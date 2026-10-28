// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_fw.hpp"

#include "device/rmsnorm_fw_device_operation.hpp"

namespace ttml::metal {

std::vector<std::optional<ttnn::Tensor>> rmsnorm_fw(
    const ttnn::Tensor& input_tensor, const ttnn::Tensor& gamma_tensor, bool return_intermediates, float epsilon) {
    // Spread each row over several cores: phase A produces per-slice partial sums of x^2, phase B folds them
    // into the row's rms and emits x * gamma / rms. Both phases are parallel over (tile-row, slice) items,
    // so a [1, 1, T, C] input is not limited to T/32 cores.
    const auto& padded = input_tensor.padded_shape();
    const uint32_t rows = padded[0] * padded[1] * (padded[2] / tt::constants::TILE_HEIGHT);
    const uint32_t Wt = padded[3] / tt::constants::TILE_WIDTH;
    const auto grid = input_tensor.device()->compute_with_storage_grid_size();
    const auto split = ops::rmsnorm_fw::device::choose_work_split(rows, Wt, grid.x * grid.y);

    auto partials = ttnn::prim::ttml_rmsnorm_fw_partial(input_tensor, split);
    auto result =
        ttnn::prim::ttml_rmsnorm_fw(input_tensor, gamma_tensor, partials, split, return_intermediates, epsilon);

    if (result.size() == 1U) {
        return {result[0], std::nullopt};
    }

    return {result[0], result[1]};
}

}  // namespace ttml::metal
