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
    bool compute_dgamma,
    const std::optional<ttnn::Tensor>& gate_tensor) {
    auto result = ttnn::prim::ttml_rmsnorm_bw(
        input_tensor, gamma_tensor, rms_tensor, dL_dout_tensor, gate_tensor, compute_dgamma);

    // dL_dgamma requires sum over batches so we cannot perform this sum in the kernel. Instead we return the
    // dL_dgamma_components and reduce it here.
    std::optional<ttnn::Tensor> dL_dgamma;
    if (result[1].has_value()) {
        dL_dgamma = ttnn::sum(
            result[1].value(),
            /* dim_arg */ ttsl::SmallVector<int>{0, 1, 2},
            /* keep_dim */ true,
            /* output_mem_config */ std::nullopt,
            /*compute_kernel_config */ core::ComputeKernelConfig::precise());  // [B,1,S,C] -> [1,1,1,C]
    }
    return {result[0], dL_dgamma, result[2]};
}

}  // namespace ttml::metal
