// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_bw_allcore.hpp"

#include "core/compute_kernel_config.hpp"
#include "device/rmsnorm_bw_allcore_device_operation.hpp"
#include "metal/ops/rmsnorm_bw_rows/rmsnorm_bw_rows.hpp"

namespace ttml::metal {

std::vector<std::optional<ttnn::Tensor>> rmsnorm_bw_allcore(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& gamma_tensor,
    const ttnn::Tensor& rms_tensor,
    const ttnn::Tensor& dL_dout_tensor,
    bool compute_dgamma,
    const std::optional<ttnn::Tensor>& residual_grad,
    bool transposed_grad) {
    // Spread each row over several cores: phase A produces per-slice partial sums of a * gamma * dL_dout,
    // phase B folds them into the row's scale and emits the gradients. Both phases are parallel over
    // (tile-row, slice) items, so a [1, 1, T, C] input is not limited to T/32 cores.
    const auto& padded = input_tensor.padded_shape();
    const uint32_t rows = padded[0] * padded[1] * (padded[2] / tt::constants::TILE_HEIGHT);
    const uint32_t Wt = padded[3] / tt::constants::TILE_WIDTH;
    const auto grid = input_tensor.device()->compute_with_storage_grid_size();
    const uint32_t num_cores = grid.x * grid.y;
    // Single-phase variant (iteration 17): every operand read once, the row's partial dot products exchanged over
    // the NoC; one (row, slice) item per core, so it needs rows * slices <= cores (3 x 54-tile slices for a
    // [1024, 5120] input on 110 cores) and the frozen-gamma path (no dgamma components).
    if (!compute_dgamma && rows <= num_cores && input_tensor.logical_shape()[-1] % tt::constants::TILE_WIDTH == 0) {
        const uint32_t slices = std::min<uint32_t>(num_cores / rows, 8U);
        const uint32_t slice_tiles = (Wt + slices - 1U) / slices;
        const uint32_t slices_eff = (Wt + slice_tiles - 1U) / slice_tiles;
        if (slice_tiles * 3U * tt::tile_size(tt::DataFormat::Float16_b) <= 512U * 1024U) {  // resident slices in L1
            auto rows_out = ttml::metal::rmsnorm_bw_rows(
                input_tensor,
                gamma_tensor,
                rms_tensor,
                dL_dout_tensor,
                slices_eff,
                slice_tiles,
                residual_grad,
                transposed_grad);
            std::vector<std::optional<ttnn::Tensor>> out{rows_out[0], std::nullopt};
            if (transposed_grad) {
                out.emplace_back(rows_out[1]);  // [dL_dinput, None, dL_dinput^T] (iteration 37)
            }
            return out;
        }
    }
    TT_FATAL(!transposed_grad, "rmsnorm_bw_allcore: the transposed gradient needs the single-phase (rows) path");
    const auto split = ops::rmsnorm_bw_allcore::device::choose_work_split(rows, Wt, num_cores);

    auto partials = ttnn::prim::ttml_rmsnorm_bw_allcore_partial(input_tensor, gamma_tensor, dL_dout_tensor, split);
    auto result = ttnn::prim::ttml_rmsnorm_bw_allcore(
        input_tensor,    // [B,1,S,C]
        gamma_tensor,    // [1,1,1,C]
        rms_tensor,      // [B,1,S,1]
        dL_dout_tensor,  // [B,1,S,C]
        partials,
        split,
        /* epsilon */ 1e-6F,  // host-side attribute only, the kernels never read it (rms comes from the forward)
        compute_dgamma,
        residual_grad);

    std::vector<std::optional<ttnn::Tensor>> out{result[0], std::nullopt};
    if (compute_dgamma) {
        // dL_dgamma requires sum over batches so we cannot perform this sum in the kernel. Instead we return the
        // dL_dgamma_components and reduce it here.
        out[1] = ttnn::sum(
            result[1],
            /* dim_arg */ ttsl::SmallVector<int>{0, 1, 2},
            /* keep_dim */ true,
            /* output_mem_config */ std::nullopt,
            /*compute_kernel_config */ core::ComputeKernelConfig::precise());  // [B,1,S,C] -> [1,1,1,C]
    }
    return out;
}

}  // namespace ttml::metal
