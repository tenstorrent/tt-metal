// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_bw.hpp"

#include "core/compute_kernel_config.hpp"
#include "device/rmsnorm_bw_device_operation.hpp"

namespace ttml::metal {

std::vector<std::optional<ttnn::Tensor>> rmsnorm_bw(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& gamma_tensor,
    const ttnn::Tensor& rms_tensor,
    const ttnn::Tensor& dL_dout_tensor,
    bool compute_dgamma) {
    // Spread each row over several cores: phase A produces per-slice partial sums of a * gamma * dL_dout,
    // phase B folds them into the row's scale and emits the gradients. Both phases are parallel over
    // (tile-row, slice) items, so a [1, 1, T, C] input is not limited to T/32 cores.
    const auto& padded = input_tensor.padded_shape();
    const uint32_t rows = padded[0] * padded[1] * (padded[2] / tt::constants::TILE_HEIGHT);
    const uint32_t Wt = padded[3] / tt::constants::TILE_WIDTH;
    const auto grid = input_tensor.device()->compute_with_storage_grid_size();
    const auto split = ops::rmsnorm_bw::device::choose_work_split(rows, Wt, grid.x * grid.y);

    auto partials = ttnn::prim::ttml_rmsnorm_bw_partial(input_tensor, gamma_tensor, dL_dout_tensor, split);
    auto result = ttnn::prim::ttml_rmsnorm_bw(
        input_tensor,    // [B,1,S,C]
        gamma_tensor,    // [1,1,1,C]
        rms_tensor,      // [B,1,S,1]
        dL_dout_tensor,  // [B,1,S,C]
        partials,
        split,
        /* epsilon */ 1e-6F,
        compute_dgamma);

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
