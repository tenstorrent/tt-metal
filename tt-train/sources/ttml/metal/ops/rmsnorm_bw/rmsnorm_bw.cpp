// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_bw.hpp"

#include "core/compute_kernel_config.hpp"
#include "device/rmsnorm_bw_device_operation.hpp"

namespace ttml::metal::ops::rmsnorm_bw {

std::vector<std::optional<ttnn::Tensor>> RMSNormBackwardOperation::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& gamma_tensor,
    const ttnn::Tensor& rms_tensor,
    const ttnn::Tensor& dL_dout_tensor) {
    auto result = ttnn::prim::ttml_rmsnorm_bw(
        input_tensor,   // [B,1,S,C]
        gamma_tensor,   // [1,1,1,C]
        rms_tensor,     //[B,1,S,1]
        dL_dout_tensor  //[B,1,S,C]
    );

    // dL_dgamma requires sum over batches so we cannot perform this sum in the kernel. Instead we return the
    // dL_dgamma_components and reduce it here.
    return {
        result[0],
        ttnn::sum(
            result[1],
            /* dim_arg */ ttnn::SmallVector<int>{0, 1, 2},
            /* keep_dim */ true,
            /* output_mem_config */ std::nullopt,
            /*compute_kernel_config */ core::ComputeKernelConfig::precise())  // [B,1,S,C] -> [1,1,1,C]
    };
}

}  // namespace ttml::metal::ops::rmsnorm_bw
