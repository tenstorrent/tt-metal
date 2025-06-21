// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_bw.hpp"

#include "core/compute_kernel_config.hpp"
#include "device/rmsnorm_bw_device_operation.hpp"  // if needed

namespace ttml::metal::ops::rmsnorm_bw {

std::vector<std::optional<ttnn::Tensor>> RMSNormBackwardOperation::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& gamma_tensor,
    const ttnn::Tensor& rms_tensor,
    const ttnn::Tensor& dL_dout_tensor) {
    // std::cerr << "Within RMSNormBackwardOperation::invoke" << std::endl;
    // dL_dout_tensor.print();
    auto result = ttnn::prim::ttml_rmsnorm_bw(
        input_tensor,   // [B,1,S,C]
        gamma_tensor,   // [1,1,1,C]
        rms_tensor,     //[B,1,S,1]
        dL_dout_tensor  //[B,1,S,C]
    );
    // std::cerr << "RMSNormBackwardOperation done" << std::endl;

    // Always return both gradients over input and gamma. Consider parameterizing this in the future and nullopt?
    // dL_dgamma requires sum over batches so we canno
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
