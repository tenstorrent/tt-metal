// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "add_rmsnorm_fw_allcore.hpp"

#include <cstdlib>

#include "device/add_rmsnorm_fw_allcore_device_operation.hpp"
#include "metal/ops/add_rmsnorm_fw_rows/add_rmsnorm_fw_rows.hpp"
#include "metal/ops/rmsnorm_bw_allcore/device/rmsnorm_bw_allcore_device_operation_types.hpp"

namespace ttml::metal {

std::vector<ttnn::Tensor> add_rmsnorm_fw_allcore(
    const ttnn::Tensor& residual_tensor,
    const ttnn::Tensor& delta_tensor,
    const ttnn::Tensor& gamma_tensor,
    float epsilon,
    bool delta_transposed,
    uint32_t out_tail_cols) {
    // Same (tile-row, slice) work split as the all-core backward: phase A adds the two inputs, writes the sum and a
    // partial sum of squares per slice; phase B folds the partials into the row's rms and scales the sum by
    // gamma / rms.
    const auto& padded = residual_tensor.padded_shape();
    const uint32_t rows = padded[0] * padded[1] * (padded[2] / tt::constants::TILE_HEIGHT);
    const uint32_t Wt = padded[3] / tt::constants::TILE_WIDTH;
    const auto grid = residual_tensor.device()->compute_with_storage_grid_size();
    const uint32_t num_cores = grid.x * grid.y;
    // Single-phase path (iteration 19): one (row, slice) item per core with the residual / delta / gamma slices
    // and the sum resident in L1, the row's sums of squares exchanged over the NoC; needs rows * slices <= cores
    // (3 x 54-tile slices for a [1024, 5120] input on 110 cores). TTML_ADD_RMSNORM_TWO_PHASE=1 forces the two-phase
    // op (A/B checks).
    static const bool force_two_phase = std::getenv("TTML_ADD_RMSNORM_TWO_PHASE") != nullptr;
    if (!force_two_phase && rows <= num_cores && residual_tensor.logical_shape()[-1] % tt::constants::TILE_WIDTH == 0) {
        const uint32_t slices =
            std::min<uint32_t>(num_cores / rows, 16U);  // 13 on the 8-row SP shard (review round 13)
        const uint32_t slice_tiles = (Wt + slices - 1U) / slices;
        const uint32_t slices_eff = (Wt + slice_tiles - 1U) / slice_tiles;
        // gamma and the sum resident in L1 (2 slices); at least 2 slices per row (one 160-tile slice per core is slower
        // than the two-phase op and its 80 fp32 accumulator round trips cost accuracy, iter19_norm_check.log).
        if (slices_eff >= 2U && slice_tiles * 2U * tt::tile_size(tt::DataFormat::Float16_b) <= 512U * 1024U) {
            return add_rmsnorm_fw_rows(
                residual_tensor,
                delta_tensor,
                gamma_tensor,
                slices_eff,
                slice_tiles,
                epsilon,
                delta_transposed,
                out_tail_cols / tt::constants::TILE_WIDTH);
        }
    }
    TT_FATAL(!delta_transposed, "add_rmsnorm_fw_allcore: a transposed delta needs the single-phase (rows) path");
    TT_FATAL(out_tail_cols == 0U, "add_rmsnorm_fw_allcore: a wide output needs the single-phase (rows) path");
    const auto split = ops::rmsnorm_bw_allcore::device::choose_work_split(rows, Wt, num_cores);

    auto partial = ttnn::prim::ttml_add_rmsnorm_fw_allcore_partial(residual_tensor, delta_tensor, split);
    auto result = ttnn::prim::ttml_add_rmsnorm_fw_allcore(partial[0], gamma_tensor, partial[1], split, epsilon);
    return {partial[0], result[0], result[1]};
}

}  // namespace ttml::metal
