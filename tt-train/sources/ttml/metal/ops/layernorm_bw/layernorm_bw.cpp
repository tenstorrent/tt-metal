// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_bw.hpp"

#include <core/ttnn_all_includes.hpp>
#include <iostream>

#include "core/compute_kernel_config.hpp"
#include "device/layernorm_bw_device_operation.hpp"

namespace ttml::metal::ops::layernorm_bw {

std::vector<std::optional<ttnn::Tensor>> LayerNormBackwardOperation::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& gamma_tensor,
    const ttnn::Tensor& mean_tensor,
    const ttnn::Tensor& rstd_tensor,
    const ttnn::Tensor& dL_dout_tensor) {
    auto device_op = ttnn::prim::ttml_layernorm_bw;

    // Call the device operation
    // Returns: [dx, dgamma_components, dbeta_components]
    auto result = device_op(input_tensor, gamma_tensor, mean_tensor, rstd_tensor, dL_dout_tensor);

    // dL_dgamma and dL_dbeta require sum over batches so we cannot perform this sum in the kernel.
    // Instead we return the components and reduce them here.

    return {
        result[0],  // dx - already complete
        ttnn::sum(
            result[1],  // dgamma_components
            /* dim_arg */ ttnn::SmallVector<int>{0, 1, 2},
            /* keep_dim */ true,
            /* output_mem_config */ std::nullopt,
            /*compute_kernel_config */ core::ComputeKernelConfig::precise()),  // [B,H,S,C] -> [1,1,1,C]
        ttnn::sum(
            result[2],  // dbeta_components
            /* dim_arg */ ttnn::SmallVector<int>{0, 1, 2},
            /* keep_dim */ true,
            /* output_mem_config */ std::nullopt,
            /*compute_kernel_config */ core::ComputeKernelConfig::precise())  // [B,H,S,C] -> [1,1,1,C]
    };
}

}  // namespace ttml::metal::ops::layernorm_bw
